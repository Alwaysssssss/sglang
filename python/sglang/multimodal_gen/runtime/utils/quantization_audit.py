# SPDX-License-Identifier: Apache-2.0
"""Diagnostic summaries for diffusion Linear quantization."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import torch

from sglang.multimodal_gen.runtime.layers.linear import LinearBase
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _layer_category(name: str) -> str:
    if name.startswith("blocks."):
        if ".attn1." in name:
            return "blocks.attn1"
        if ".attn2." in name:
            return "blocks.attn2"
        if ".ffn." in name:
            return "blocks.ffn"
        return "blocks.other"
    if name.startswith("condition_embedder."):
        return "condition_embedder"
    if name == "proj_out" or name.startswith("proj_out."):
        return "proj_out"
    return name.split(".", 1)[0] if name else "<root>"


def _scale_descriptor(layer: LinearBase) -> str:
    for attr in ("weight_scale", "weight_scale_inv"):
        scale = getattr(layer, attr, None)
        if isinstance(scale, torch.Tensor):
            shape = "x".join(str(dim) for dim in scale.shape) or "scalar"
            return f"{attr}:{shape}:{scale.dtype}"
    return "none"


def _kernel_route(layer: LinearBase, method_name: str) -> str:
    method = layer.quant_method
    route_fn = getattr(method, "runtime_kernel_name", None)
    if callable(route_fn):
        try:
            return str(route_fn(layer))
        except Exception as exc:
            return f"audit_error:{type(exc).__name__}"
    if method_name == "UnquantizedLinearMethod":
        return "torch.nn.functional.linear"
    return "unknown"


def build_quantization_audit(
    module: torch.nn.Module,
    *,
    component: str,
    rank: int,
) -> dict[str, Any]:
    method_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    scale_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    weight_shape_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    weight_bytes_by_dtype: Counter[str] = Counter()
    unquantized_names: list[str] = []
    fp8_dtype_mismatch_names: list[str] = []
    dequant_fallback_names: list[str] = []

    for name, layer in module.named_modules():
        if not isinstance(layer, LinearBase):
            continue

        method_name = type(layer.quant_method).__name__
        weight = getattr(layer, "weight", None)
        weight_dtype = str(getattr(weight, "dtype", None))
        weight_shape = tuple(getattr(weight, "shape", ()))
        shape_text = "x".join(str(dim) for dim in weight_shape) or "scalar"
        route = _kernel_route(layer, method_name)

        method_counts[method_name] += 1
        dtype_counts[weight_dtype] += 1
        scale_counts[_scale_descriptor(layer)] += 1
        route_counts[route] += 1
        weight_shape_counts[shape_text] += 1
        category_counts[_layer_category(name)] += 1

        if isinstance(weight, torch.Tensor):
            weight_bytes_by_dtype[weight_dtype] += (
                weight.numel() * weight.element_size()
            )

        if method_name == "UnquantizedLinearMethod":
            unquantized_names.append(name)
        if method_name == "Fp8LinearMethod":
            if not weight_dtype.startswith("torch.float8"):
                fp8_dtype_mismatch_names.append(name)
            if "dequant_fallback" in route:
                dequant_fallback_names.append(name)

    linear_total = sum(method_counts.values())
    fp8_method_count = method_counts.get("Fp8LinearMethod", 0)
    fp8_weight_count = sum(
        count
        for dtype, count in dtype_counts.items()
        if dtype.startswith("torch.float8")
    )
    true_w8a8_count = sum(
        count
        for route, count in route_counts.items()
        if (
            "fp8_scaled_mm" in route
            or "triton_scaled_mm" in route
            or "block_fp8" in route
        )
        and "dequant_fallback" not in route
    )

    return {
        "rank": rank,
        "component": component,
        "module_class": type(module).__name__,
        "linear_total": linear_total,
        "fp8_method_count": fp8_method_count,
        "fp8_weight_count": fp8_weight_count,
        "predicted_true_w8a8_count": true_w8a8_count,
        "quant_method_counts": _counter_dict(method_counts),
        "weight_dtype_counts": _counter_dict(dtype_counts),
        "weight_bytes_by_dtype": _counter_dict(weight_bytes_by_dtype),
        "weight_scale_counts": _counter_dict(scale_counts),
        "predicted_kernel_route_counts": _counter_dict(route_counts),
        "weight_shape_counts": _counter_dict(weight_shape_counts),
        "layer_category_counts": _counter_dict(category_counts),
        "unquantized_linear_names": unquantized_names,
        "fp8_weight_dtype_mismatch_names": fp8_dtype_mismatch_names,
        "fp8_dequant_fallback_names": dequant_fallback_names,
    }


def log_quantization_audit(
    module: torch.nn.Module,
    *,
    component: str,
    rank: int,
) -> dict[str, Any]:
    audit = build_quantization_audit(module, component=component, rank=rank)
    logger.info(
        "QUANTIZATION_AUDIT %s",
        json.dumps(audit, ensure_ascii=True, sort_keys=True),
    )
    return audit
