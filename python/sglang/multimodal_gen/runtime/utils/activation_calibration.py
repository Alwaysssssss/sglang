# SPDX-License-Identifier: Apache-2.0
"""Streaming activation calibration for VideoEdit FP8 Linear layers."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from sglang.multimodal_gen.runtime.layers.linear import LinearBase
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.quantization_audit import (
    build_quantization_audit,
)

logger = init_logger(__name__)

CALIBRATION_DIR_ENV = "SGLANG_VIDEOEDIT_ACTIVATION_CALIBRATION_DIR"
REQUEST_PREFIX_ENV = "SGLANG_VIDEOEDIT_ACTIVATION_CALIBRATION_REQUEST_PREFIX"
EXPECTED_LINEAR_COUNT_ENV = (
    "SGLANG_VIDEOEDIT_ACTIVATION_CALIBRATION_EXPECTED_LINEAR_COUNT"
)
HISTOGRAM_BINS_ENV = "SGLANG_VIDEOEDIT_ACTIVATION_CALIBRATION_HISTOGRAM_BINS"
HISTOGRAM_LOG2_MIN_ENV = "SGLANG_VIDEOEDIT_ACTIVATION_CALIBRATION_LOG2_MIN"
HISTOGRAM_LOG2_MAX_ENV = "SGLANG_VIDEOEDIT_ACTIVATION_CALIBRATION_LOG2_MAX"

DEFAULT_REQUEST_PREFIX = "calib_"
DEFAULT_EXPECTED_LINEAR_COUNT = 328
DEFAULT_HISTOGRAM_BINS = 2048
DEFAULT_LOG2_MIN = -24.0
DEFAULT_LOG2_MAX = 16.0
FP8_E4M3_MAX = 448.0
SCHEMA_VERSION = 1
PERCENTILES = (0.99, 0.999, 0.9999, 0.99999, 0.999999)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_save_tensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    save_file(
        {name: tensor.contiguous() for name, tensor in tensors.items()}, temporary
    )
    os.replace(temporary, path)


def checkpoint_aliases_for_runtime_linear(name: str) -> list[str]:
    """Map a fused runtime Linear name to its source checkpoint projection names."""
    exact_mappings = {
        "condition_embedder.time_embedder.mlp.fc_in": (
            "condition_embedder.time_embedder.linear_1",
        ),
        "condition_embedder.time_embedder.mlp.fc_out": (
            "condition_embedder.time_embedder.linear_2",
        ),
        "condition_embedder.time_modulation.linear": ("condition_embedder.time_proj",),
        "condition_embedder.text_embedder.fc_in": (
            "condition_embedder.text_embedder.linear_1",
        ),
        "condition_embedder.text_embedder.fc_out": (
            "condition_embedder.text_embedder.linear_2",
        ),
        "condition_embedder.image_embedder.ff.fc_in": (
            "condition_embedder.image_embedder.ff.net.0.proj",
        ),
        "condition_embedder.image_embedder.ff.fc_out": (
            "condition_embedder.image_embedder.ff.net.2",
        ),
        "proj_out": ("proj_out",),
    }
    if name in exact_mappings:
        return list(exact_mappings[name])

    parts = name.split(".", maxsplit=2)
    if len(parts) != 3 or parts[0] != "blocks" or not parts[1].isdigit():
        raise ValueError(f"Unsupported VideoEdit runtime Linear name: {name!r}")
    block_prefix = f"blocks.{parts[1]}"
    suffix_mappings = {
        "to_qkv": (
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
        ),
        "to_out": ("attn1.to_out.0",),
        "attn2.to_q": ("attn2.to_q",),
        "attn2.to_kv": (
            "attn2.to_k",
            "attn2.to_v",
        ),
        "attn2.to_out": ("attn2.to_out.0",),
        "attn2.to_added_kv": (
            "attn2.add_k_proj",
            "attn2.add_v_proj",
        ),
        "ffn.fc_in": ("ffn.net.0.proj",),
        "ffn.fc_out": ("ffn.net.2",),
    }
    source_suffixes = suffix_mappings.get(parts[2])
    if source_suffixes is None:
        raise ValueError(f"Unsupported VideoEdit runtime Linear name: {name!r}")
    return [f"{block_prefix}.{suffix}" for suffix in source_suffixes]


def _percentile_label(value: float) -> str:
    digits = f"{value * 100:.6f}".rstrip("0").rstrip(".").replace(".", "_")
    return f"p{digits}"


def histogram_percentile(
    histogram: torch.Tensor,
    *,
    zero_count: int,
    underflow_count: int,
    overflow_count: int,
    log2_min: float,
    log2_max: float,
    percentile: float,
    absmax: float,
) -> float:
    """Approximate a token-amax percentile using the upper histogram bin edge."""
    if not 0.0 < percentile <= 1.0:
        raise ValueError(f"percentile must be in (0, 1], got {percentile}")
    counts = histogram.to(dtype=torch.int64, device="cpu")
    total = (
        int(zero_count)
        + int(underflow_count)
        + int(counts.sum().item())
        + int(overflow_count)
    )
    if total <= 0:
        return 0.0
    target = max(1, math.ceil(percentile * total))
    low_count = int(zero_count) + int(underflow_count)
    if target <= int(zero_count):
        return 0.0
    if target <= low_count:
        return 2.0**log2_min
    target -= low_count
    cumulative = torch.cumsum(counts, dim=0)
    in_range = int(cumulative[-1].item()) if cumulative.numel() else 0
    if target > in_range:
        return float(absmax)
    bin_index = int(
        torch.searchsorted(cumulative, torch.tensor(target, dtype=torch.int64)).item()
    )
    width = (log2_max - log2_min) / counts.numel()
    upper_log2 = log2_min + (bin_index + 1) * width
    return min(float(absmax), 2.0**upper_log2)


@dataclass(frozen=True)
class HistogramConfig:
    bins: int = DEFAULT_HISTOGRAM_BINS
    log2_min: float = DEFAULT_LOG2_MIN
    log2_max: float = DEFAULT_LOG2_MAX

    def validate(self) -> None:
        if self.bins <= 0:
            raise ValueError("histogram bins must be positive")
        if not self.log2_min < self.log2_max:
            raise ValueError("histogram log2_min must be smaller than log2_max")


def summarize_activation_tensor(
    tensor: torch.Tensor,
    config: HistogramConfig,
) -> dict[str, torch.Tensor | int]:
    """Summarize one Linear input without retaining the full activation."""
    config.validate()
    if tensor.ndim == 0:
        raise ValueError("Linear activation must have at least one dimension")
    input_features = int(tensor.shape[-1])
    if input_features <= 0:
        raise ValueError("Linear activation input dimension must be positive")

    flat = tensor.detach().reshape(-1, input_features)
    token_amax = torch.amax(torch.abs(flat), dim=-1).float()
    finite = torch.isfinite(token_amax)
    finite_amax = token_amax[finite]
    nonfinite_tokens = (~finite).sum(dtype=torch.int64)
    zero_tokens = (finite_amax == 0).sum(dtype=torch.int64)
    positive = finite_amax[finite_amax > 0]

    lower = 2.0**config.log2_min
    upper = 2.0**config.log2_max
    underflow = (positive < lower).sum(dtype=torch.int64)
    overflow = (positive > upper).sum(dtype=torch.int64)
    in_range = positive[(positive >= lower) & (positive <= upper)]
    if in_range.numel():
        histogram = torch.histc(
            torch.log2(in_range),
            bins=config.bins,
            min=config.log2_min,
            max=config.log2_max,
        ).to(torch.int64)
    else:
        histogram = torch.zeros(
            config.bins,
            dtype=torch.int64,
            device=tensor.device,
        )
    absmax = (
        finite_amax.max()
        if finite_amax.numel()
        else torch.zeros((), dtype=torch.float32, device=tensor.device)
    )
    return {
        "histogram": histogram,
        "absmax": absmax,
        "nonfinite_token_count": nonfinite_tokens,
        "zero_token_count": zero_tokens,
        "underflow_token_count": underflow,
        "overflow_token_count": overflow,
        "token_count": int(flat.shape[0]),
        "element_count": int(flat.numel()),
        "input_features": input_features,
    }


class ActivationCalibrationCollector:
    """Collect per-Linear token-amax distributions for selected requests."""

    def __init__(
        self,
        module: torch.nn.Module,
        *,
        output_dir: Path,
        rank: int,
        request_prefix: str,
        expected_linear_count: int,
        histogram_config: HistogramConfig,
    ) -> None:
        self.output_dir = output_dir
        self.rank = int(rank)
        self.request_prefix = request_prefix
        self.expected_linear_count = int(expected_linear_count)
        self.histogram_config = histogram_config
        self.histogram_config.validate()
        self.rank_dir = output_dir / f"rank{self.rank}"
        self.rank_dir.mkdir(parents=True, exist_ok=True)

        layers = [
            (name, layer)
            for name, layer in module.named_modules()
            if isinstance(layer, LinearBase)
        ]
        self.module_names = [name for name, _ in layers]
        self.input_features = [
            int(getattr(layer, "input_size_per_partition", layer.input_size))
            for _, layer in layers
        ]
        self._validate_layers(module, layers)
        self.module_index = {
            name: index for index, name in enumerate(self.module_names)
        }
        self.handles = []

        module_count = len(self.module_names)
        bins = histogram_config.bins
        self.histogram = torch.zeros((module_count, bins), dtype=torch.int64)
        self.absmax = torch.zeros(module_count, dtype=torch.float32)
        self.nonfinite_token_count = torch.zeros(module_count, dtype=torch.int64)
        self.zero_token_count = torch.zeros(module_count, dtype=torch.int64)
        self.underflow_token_count = torch.zeros(module_count, dtype=torch.int64)
        self.overflow_token_count = torch.zeros(module_count, dtype=torch.int64)
        self.token_count = torch.zeros(module_count, dtype=torch.int64)
        self.element_count = torch.zeros(module_count, dtype=torch.int64)
        self.observation_count = torch.zeros(module_count, dtype=torch.int64)
        self.min_tokens_per_observation = torch.full(
            (module_count,), torch.iinfo(torch.int64).max, dtype=torch.int64
        )
        self.max_tokens_per_observation = torch.zeros(module_count, dtype=torch.int64)
        self.completed_requests: list[str] = []
        self.failed_requests: list[str] = []
        self._completed_set: set[str] = set()

        self._current_request_id: str | None = None
        self._current_device: torch.device | None = None
        self._request_tensors: dict[str, torch.Tensor] = {}
        self._request_observations = [0] * module_count
        self._request_token_count = [0] * module_count
        self._request_element_count = [0] * module_count
        self._request_min_tokens = [2**63 - 1] * module_count
        self._request_max_tokens = [0] * module_count
        self._context_keys: list[tuple[int, int, int]] = []
        self._context_index: dict[tuple[int, int, int], int] = {}
        self._context_absmax: list[torch.Tensor] = []

        self._load_existing_state()
        for index, (_, layer) in enumerate(layers):
            self.handles.append(layer.register_forward_pre_hook(self._make_hook(index)))
        self._write_state()
        logger.info(
            "ACTIVATION_CALIBRATION_AUDIT %s",
            json.dumps(
                {
                    "rank": self.rank,
                    "output_dir": str(self.output_dir),
                    "request_prefix": self.request_prefix,
                    "linear_total": len(self.module_names),
                    "histogram_bins": self.histogram_config.bins,
                    "status": "registered",
                },
                sort_keys=True,
            ),
        )

    def _validate_layers(
        self,
        module: torch.nn.Module,
        layers: list[tuple[str, LinearBase]],
    ) -> None:
        audit = build_quantization_audit(
            module,
            component="transformer",
            rank=self.rank,
        )
        if len(layers) != self.expected_linear_count:
            raise ValueError(
                "Activation calibration expected "
                f"{self.expected_linear_count} Transformer Linear layers, "
                f"found {len(layers)}"
            )
        if audit["fp8_method_count"] != len(layers):
            raise ValueError(
                "Activation calibration requires every Transformer Linear to use "
                f"Fp8LinearMethod: {audit['fp8_method_count']} != {len(layers)}"
            )
        if audit["predicted_true_w8a8_count"] != len(layers):
            raise ValueError(
                "Activation calibration requires true FP8 W8A8 for every "
                f"Transformer Linear: {audit['predicted_true_w8a8_count']} "
                f"!= {len(layers)}"
            )
        static_names = [
            name
            for name, layer in layers
            if getattr(layer, "input_scale", None) is not None
        ]
        if static_names:
            raise ValueError(
                "Activation calibration must run on the dynamic activation path; "
                f"static input_scale found on {static_names[:8]}"
            )

    def _state_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "histogram": self.histogram,
            "absmax": self.absmax,
            "nonfinite_token_count": self.nonfinite_token_count,
            "zero_token_count": self.zero_token_count,
            "underflow_token_count": self.underflow_token_count,
            "overflow_token_count": self.overflow_token_count,
            "token_count": self.token_count,
            "element_count": self.element_count,
            "observation_count": self.observation_count,
            "min_tokens_per_observation": self.min_tokens_per_observation,
            "max_tokens_per_observation": self.max_tokens_per_observation,
            "input_features": torch.tensor(self.input_features, dtype=torch.int64),
        }

    def _manifest(self, stats_file: str) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "updated_at": utc_now(),
            "rank": self.rank,
            "request_prefix": self.request_prefix,
            "expected_linear_count": self.expected_linear_count,
            "module_names": self.module_names,
            "checkpoint_aliases": {
                name: checkpoint_aliases_for_runtime_linear(name)
                for name in self.module_names
            },
            "histogram": {
                "kind": "token_amax_log2",
                "bins": self.histogram_config.bins,
                "log2_min": self.histogram_config.log2_min,
                "log2_max": self.histogram_config.log2_max,
            },
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "stats_file": stats_file,
        }

    def _load_existing_state(self) -> None:
        manifest_path = self.rank_dir / "manifest.json"
        if not manifest_path.exists():
            return
        if not manifest_path.is_file():
            raise ValueError(
                f"Incomplete activation calibration state in {self.rank_dir}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stats_file = manifest.get("stats_file", "activation_stats.safetensors")
        stats_path = self.rank_dir / stats_file
        if not stats_path.is_file():
            raise ValueError(f"Calibration state file is missing: {stats_path}")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Calibration resume schema version does not match")
        if manifest.get("rank") != self.rank:
            raise ValueError("Calibration resume rank does not match")
        if manifest.get("request_prefix") != self.request_prefix:
            raise ValueError("Calibration resume request prefix does not match")
        if manifest.get("expected_linear_count") != self.expected_linear_count:
            raise ValueError("Calibration resume expected Linear count does not match")
        if manifest.get("module_names") != self.module_names:
            raise ValueError("Calibration resume module names do not match")
        histogram = manifest.get("histogram", {})
        expected_histogram = {
            "kind": "token_amax_log2",
            "bins": self.histogram_config.bins,
            "log2_min": self.histogram_config.log2_min,
            "log2_max": self.histogram_config.log2_max,
        }
        if histogram != expected_histogram:
            raise ValueError(
                "Calibration resume histogram configuration does not match"
            )
        existing = load_file(stats_path)
        for name, target in self._state_tensors().items():
            loaded = existing.get(name)
            if loaded is None or loaded.shape != target.shape:
                raise ValueError(f"Invalid resumed calibration tensor: {name}")
            if name == "input_features":
                if not torch.equal(loaded, target):
                    raise ValueError(
                        "Calibration resume input feature sizes do not match"
                    )
            else:
                target.copy_(loaded)
        self.completed_requests = list(manifest.get("completed_requests", []))
        self.failed_requests = list(manifest.get("failed_requests", []))
        self._completed_set = set(self.completed_requests)

    def _write_state(self) -> None:
        previous_stats_file = None
        manifest_path = self.rank_dir / "manifest.json"
        if manifest_path.is_file():
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            previous_stats_file = previous_manifest.get("stats_file")
        stats_file = f"activation_stats.{time.time_ns()}.safetensors"
        _atomic_save_tensors(
            self.rank_dir / stats_file,
            self._state_tensors(),
        )
        _atomic_write_json(manifest_path, self._manifest(stats_file))
        if (
            previous_stats_file
            and previous_stats_file != stats_file
            and previous_stats_file.startswith("activation_stats.")
        ):
            (self.rank_dir / previous_stats_file).unlink(missing_ok=True)

    def _begin_request(self, request_id: str, device: torch.device) -> None:
        if self._current_request_id is not None:
            if self._current_request_id != request_id:
                raise RuntimeError(
                    "Activation collector saw concurrent requests: "
                    f"{self._current_request_id!r} and {request_id!r}"
                )
            return
        self._current_request_id = request_id
        self._current_device = device
        module_count = len(self.module_names)
        self._request_tensors = {
            "histogram": torch.zeros(
                (module_count, self.histogram_config.bins),
                dtype=torch.int64,
                device=device,
            ),
            "absmax": torch.zeros(module_count, dtype=torch.float32, device=device),
            "nonfinite_token_count": torch.zeros(
                module_count, dtype=torch.int64, device=device
            ),
            "zero_token_count": torch.zeros(
                module_count, dtype=torch.int64, device=device
            ),
            "underflow_token_count": torch.zeros(
                module_count, dtype=torch.int64, device=device
            ),
            "overflow_token_count": torch.zeros(
                module_count, dtype=torch.int64, device=device
            ),
        }
        self._request_observations = [0] * module_count
        self._request_token_count = [0] * module_count
        self._request_element_count = [0] * module_count
        self._request_min_tokens = [2**63 - 1] * module_count
        self._request_max_tokens = [0] * module_count
        self._context_keys = []
        self._context_index = {}
        self._context_absmax = []

    def _make_hook(self, module_index: int):
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError(
                    f"Calibration Linear {self.module_names[module_index]} "
                    "did not receive a Tensor input"
                )
            try:
                context = get_forward_context()
            except AssertionError:
                return
            request = context.forward_batch
            if request is None or request.is_warmup:
                return
            request_id = str(request.request_id or "")
            if not request_id.startswith(self.request_prefix):
                return
            if request_id in self._completed_set:
                return

            tensor = inputs[0]
            self._begin_request(request_id, tensor.device)
            if tensor.device != self._current_device:
                raise RuntimeError(
                    "Calibration request used more than one device per rank"
                )
            summary = summarize_activation_tensor(tensor, self.histogram_config)
            expected_k = self.input_features[module_index]
            if summary["input_features"] != expected_k:
                raise ValueError(
                    f"Activation K mismatch for {self.module_names[module_index]}: "
                    f"{summary['input_features']} != {expected_k}"
                )

            request_tensors = self._request_tensors
            request_tensors["histogram"][module_index].add_(summary["histogram"])
            request_tensors["absmax"][module_index] = torch.maximum(
                request_tensors["absmax"][module_index],
                summary["absmax"],
            )
            for key in (
                "nonfinite_token_count",
                "zero_token_count",
                "underflow_token_count",
                "overflow_token_count",
            ):
                request_tensors[key][module_index].add_(summary[key])

            token_count = int(summary["token_count"])
            self._request_observations[module_index] += 1
            self._request_token_count[module_index] += token_count
            self._request_element_count[module_index] += int(summary["element_count"])
            self._request_min_tokens[module_index] = min(
                self._request_min_tokens[module_index], token_count
            )
            self._request_max_tokens[module_index] = max(
                self._request_max_tokens[module_index], token_count
            )

            params = request.sampling_params
            raw_window_index = getattr(params, "runtime_window_index", None)
            window_index = -1 if raw_window_index is None else int(raw_window_index)
            branch = 1 if bool(request.is_cfg_negative) else 0
            context_key = (window_index, int(context.current_timestep), branch)
            context_index = self._context_index.get(context_key)
            if context_index is None:
                context_index = len(self._context_keys)
                self._context_index[context_key] = context_index
                self._context_keys.append(context_key)
                self._context_absmax.append(
                    torch.zeros(
                        len(self.module_names),
                        dtype=torch.float32,
                        device=tensor.device,
                    )
                )
            context_values = self._context_absmax[context_index]
            context_values[module_index] = torch.maximum(
                context_values[module_index],
                summary["absmax"],
            )

        return hook

    def _reset_request(self) -> None:
        self._current_request_id = None
        self._current_device = None
        self._request_tensors = {}
        self._context_keys = []
        self._context_index = {}
        self._context_absmax = []

    def finish_request(self, request: Any, *, success: bool) -> None:
        if bool(getattr(request, "is_warmup", False)):
            return
        request_id = str(getattr(request, "request_id", "") or "")
        if not request_id.startswith(self.request_prefix):
            return
        if request_id in self._completed_set:
            return
        if self._current_request_id is None:
            if success:
                raise RuntimeError(
                    f"Calibration request {request_id!r} produced no Linear observations"
                )
            return
        if self._current_request_id != request_id:
            raise RuntimeError(
                f"Finishing calibration request {request_id!r}, but collector "
                f"contains {self._current_request_id!r}"
            )
        if not success:
            if request_id not in self.failed_requests:
                self.failed_requests.append(request_id)
            self._reset_request()
            self._write_state()
            return

        missing = [
            self.module_names[index]
            for index, count in enumerate(self._request_observations)
            if count == 0
        ]
        if missing:
            self._reset_request()
            raise RuntimeError(
                f"Calibration request missed {len(missing)} Linear layers: {missing[:8]}"
            )

        request_cpu = {
            name: tensor.detach().cpu()
            for name, tensor in self._request_tensors.items()
        }
        self.histogram.add_(request_cpu["histogram"])
        self.absmax.copy_(torch.maximum(self.absmax, request_cpu["absmax"]))
        for key in (
            "nonfinite_token_count",
            "zero_token_count",
            "underflow_token_count",
            "overflow_token_count",
        ):
            getattr(self, key).add_(request_cpu[key])
        self.token_count.add_(
            torch.tensor(self._request_token_count, dtype=torch.int64)
        )
        self.element_count.add_(
            torch.tensor(self._request_element_count, dtype=torch.int64)
        )
        self.observation_count.add_(
            torch.tensor(self._request_observations, dtype=torch.int64)
        )
        request_min = torch.tensor(self._request_min_tokens, dtype=torch.int64)
        self.min_tokens_per_observation.copy_(
            torch.minimum(self.min_tokens_per_observation, request_min)
        )
        self.max_tokens_per_observation.copy_(
            torch.maximum(
                self.max_tokens_per_observation,
                torch.tensor(self._request_max_tokens, dtype=torch.int64),
            )
        )

        if self._context_absmax:
            context_values = torch.stack(self._context_absmax).detach().cpu()
            context_keys = torch.tensor(self._context_keys, dtype=torch.int64)
            request_dir = self.rank_dir / "requests"
            safe_request_id = "".join(
                char if char.isalnum() or char in "._-" else "_" for char in request_id
            )
            _atomic_save_tensors(
                request_dir / f"{safe_request_id}.safetensors",
                {
                    "context_absmax": context_values,
                    "context_keys": context_keys,
                },
            )

        self.failed_requests = [
            failed_id for failed_id in self.failed_requests if failed_id != request_id
        ]
        self.completed_requests.append(request_id)
        self._completed_set.add(request_id)
        self._reset_request()
        self._write_state()


def maybe_register_activation_calibration(
    module: torch.nn.Module,
    *,
    rank: int,
) -> ActivationCalibrationCollector | None:
    output_dir = os.getenv(CALIBRATION_DIR_ENV, "").strip()
    if not output_dir:
        return None
    return ActivationCalibrationCollector(
        module,
        output_dir=Path(output_dir).expanduser().resolve(),
        rank=rank,
        request_prefix=os.getenv(REQUEST_PREFIX_ENV, DEFAULT_REQUEST_PREFIX),
        expected_linear_count=int(
            os.getenv(
                EXPECTED_LINEAR_COUNT_ENV,
                str(DEFAULT_EXPECTED_LINEAR_COUNT),
            )
        ),
        histogram_config=HistogramConfig(
            bins=int(os.getenv(HISTOGRAM_BINS_ENV, str(DEFAULT_HISTOGRAM_BINS))),
            log2_min=float(os.getenv(HISTOGRAM_LOG2_MIN_ENV, str(DEFAULT_LOG2_MIN))),
            log2_max=float(os.getenv(HISTOGRAM_LOG2_MAX_ENV, str(DEFAULT_LOG2_MAX))),
        ),
    )


def merge_rank_calibration(
    calibration_dir: Path,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Merge rank-local calibration states and write candidate static scales."""
    calibration_dir = calibration_dir.expanduser().resolve()
    output_dir = (output_dir or calibration_dir).expanduser().resolve()
    rank_dirs = [
        path
        for path in calibration_dir.glob("rank*")
        if path.is_dir()
        and path.name[4:].isdigit()
        and (path / "manifest.json").is_file()
    ]
    rank_dirs.sort(key=lambda path: int(path.name[4:]))
    if not rank_dirs:
        raise FileNotFoundError(
            f"No rank calibration states found in {calibration_dir}"
        )
    rank_indices = [int(path.name[4:]) for path in rank_dirs]
    if rank_indices != list(range(len(rank_dirs))):
        raise ValueError(
            f"Calibration rank directories are not contiguous: {rank_indices}"
        )

    manifests = [
        json.loads((rank_dir / "manifest.json").read_text(encoding="utf-8"))
        for rank_dir in rank_dirs
    ]
    reference = manifests[0]
    module_names = reference["module_names"]
    histogram_config = reference["histogram"]
    completed_requests = set(reference.get("completed_requests", []))
    if not completed_requests:
        raise ValueError("Calibration contains no completed requests")
    if reference.get("failed_requests"):
        raise ValueError(
            f"Rank calibration contains failed requests: "
            f"{reference['failed_requests']}"
        )
    for manifest in manifests[1:]:
        if manifest["module_names"] != module_names:
            raise ValueError("Rank calibration module names do not match")
        if manifest["histogram"] != histogram_config:
            raise ValueError("Rank calibration histogram configurations do not match")
        if set(manifest.get("completed_requests", [])) != completed_requests:
            raise ValueError("Rank calibration completed request sets do not match")
        if manifest.get("failed_requests"):
            raise ValueError(
                f"Rank calibration contains failed requests: "
                f"{manifest['failed_requests']}"
            )

    states = [
        load_file(rank_dir / manifest["stats_file"])
        for rank_dir, manifest in zip(rank_dirs, manifests)
    ]
    if any(
        not torch.equal(state["input_features"], states[0]["input_features"])
        for state in states[1:]
    ):
        raise ValueError("Rank calibration input feature sizes do not match")
    histogram = torch.stack([state["histogram"] for state in states]).sum(dim=0)
    absmax = torch.stack([state["absmax"] for state in states]).amax(dim=0)
    sum_keys = (
        "nonfinite_token_count",
        "zero_token_count",
        "underflow_token_count",
        "overflow_token_count",
        "token_count",
        "element_count",
        "observation_count",
    )
    merged = {
        key: torch.stack([state[key] for state in states]).sum(dim=0)
        for key in sum_keys
    }
    merged["histogram"] = histogram
    merged["absmax"] = absmax
    merged["input_features"] = states[0]["input_features"]
    merged["min_tokens_per_observation"] = torch.stack(
        [state["min_tokens_per_observation"] for state in states]
    ).amin(dim=0)
    merged["max_tokens_per_observation"] = torch.stack(
        [state["max_tokens_per_observation"] for state in states]
    ).amax(dim=0)

    scales: dict[str, torch.Tensor] = {
        "input_scale_max": torch.clamp(absmax / FP8_E4M3_MAX, min=1e-12)
    }
    module_records: list[dict[str, Any]] = []
    for index, name in enumerate(module_names):
        record = {
            "name": name,
            "checkpoint_aliases": checkpoint_aliases_for_runtime_linear(name),
            "input_features": int(merged["input_features"][index].item()),
            "observation_count": int(merged["observation_count"][index].item()),
            "token_count": int(merged["token_count"][index].item()),
            "element_count": int(merged["element_count"][index].item()),
            "nonfinite_token_count": int(merged["nonfinite_token_count"][index].item()),
            "zero_token_count": int(merged["zero_token_count"][index].item()),
            "underflow_token_count": int(merged["underflow_token_count"][index].item()),
            "overflow_token_count": int(merged["overflow_token_count"][index].item()),
            "absmax": float(absmax[index].item()),
            "min_tokens_per_observation": int(
                merged["min_tokens_per_observation"][index].item()
            ),
            "max_tokens_per_observation": int(
                merged["max_tokens_per_observation"][index].item()
            ),
            "candidate_thresholds": {"max": float(absmax[index].item())},
            "candidate_scales": {"max": float(scales["input_scale_max"][index].item())},
        }
        for percentile in PERCENTILES:
            label = _percentile_label(percentile)
            threshold = histogram_percentile(
                histogram[index],
                zero_count=record["zero_token_count"],
                underflow_count=record["underflow_token_count"],
                overflow_count=record["overflow_token_count"],
                log2_min=float(histogram_config["log2_min"]),
                log2_max=float(histogram_config["log2_max"]),
                percentile=percentile,
                absmax=record["absmax"],
            )
            scale = max(threshold / FP8_E4M3_MAX, 1e-12)
            record["candidate_thresholds"][label] = threshold
            record["candidate_scales"][label] = scale
            scale_key = f"input_scale_{label}"
            if scale_key not in scales:
                scales[scale_key] = torch.empty(len(module_names), dtype=torch.float32)
            scales[scale_key][index] = scale
        module_records.append(record)

    if any(record["nonfinite_token_count"] for record in module_records):
        raise ValueError("Calibration contains non-finite activation tokens")

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_save_tensors(
        output_dir / "activation_stats.safetensors",
        {**merged, **scales},
    )
    calibration = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "source_dir": str(calibration_dir),
        "rank_count": len(rank_dirs),
        "completed_requests": sorted(completed_requests),
        "histogram": histogram_config,
        "quantization": {
            "dtype": "fp8_e4m3fn",
            "symmetric": True,
            "zero_point": None,
            "fp8_max": FP8_E4M3_MAX,
            "granularity": "per_runtime_linear",
        },
        "modules": module_records,
    }
    _atomic_write_json(output_dir / "activation_calibration.json", calibration)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "source_rank_dirs": [str(path) for path in rank_dirs],
        "rank_count": len(rank_dirs),
        "module_count": len(module_names),
        "completed_request_count": len(completed_requests),
        "completed_requests": sorted(completed_requests),
        "stats_file": "activation_stats.safetensors",
        "calibration_file": "activation_calibration.json",
    }
    _atomic_write_json(output_dir / "calibration_manifest.json", manifest)
    return manifest
