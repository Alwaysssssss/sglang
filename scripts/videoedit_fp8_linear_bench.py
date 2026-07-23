#!/usr/bin/env python3
"""Benchmark BF16 and the VideoEdit dynamic W8A8 Linear path on real shapes."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_shapes(path: Path, max_m: int, max_shapes: int) -> list[dict[str, int]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    by_kn: dict[tuple[int, int], dict[str, int]] = {}
    for record in records:
        if record.get("method") != "Fp8LinearMethod":
            continue
        if "fp8_scaled_mm" not in str(record.get("kernel", "")):
            continue
        try:
            original_m = int(record["m"])
            k = int(record["k"])
            n = int(record["n"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (k, n)
        current = by_kn.get(key)
        if current is None or original_m > current["original_m"]:
            by_kn[key] = {
                "original_m": original_m,
                "m": min(original_m, max_m),
                "k": k,
                "n": n,
            }

    shapes = sorted(
        by_kn.values(),
        key=lambda shape: 2 * shape["original_m"] * shape["k"] * shape["n"],
        reverse=True,
    )
    return shapes[:max_shapes]


def elapsed_ms(torch, fn, warmups: int, iterations: int) -> tuple[float, list[float]]:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()

    values: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return statistics.median(values), values


def benchmark_shape(
    shape: dict[str, int],
    *,
    warmups: int,
    iterations: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    from sgl_kernel import fp8_scaled_mm

    from sglang.srt.layers.quantization.fp8_kernel import (
        per_token_group_quant_fp8,
        sglang_per_token_quant_fp8,
    )

    m, k, n = shape["m"], shape["k"], shape["n"]
    device = torch.device("cuda", 0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    weight = torch.randn((n, k), device=device, dtype=torch.bfloat16)
    qweight, weight_scale = per_token_group_quant_fp8(weight, k)
    qweight_t = qweight.t()
    weight_scale_t = weight_scale.t().contiguous()
    qinput, input_scale = sglang_per_token_quant_fp8(x)

    def bf16_op():
        return functional.linear(x, weight)

    def fp8_gemm_op():
        return fp8_scaled_mm(
            qinput,
            qweight_t,
            input_scale,
            weight_scale_t,
            torch.bfloat16,
            bias=None,
        )

    def dynamic_w8a8_op():
        dynamic_qinput, dynamic_scale = sglang_per_token_quant_fp8(x)
        return fp8_scaled_mm(
            dynamic_qinput,
            qweight_t,
            dynamic_scale,
            weight_scale_t,
            torch.bfloat16,
            bias=None,
        )

    bf16_median, bf16_values = elapsed_ms(torch, bf16_op, warmups, iterations)
    fp8_median, fp8_values = elapsed_ms(torch, fp8_gemm_op, warmups, iterations)
    dynamic_median, dynamic_values = elapsed_ms(
        torch, dynamic_w8a8_op, warmups, iterations
    )

    sample_x = x[: min(m, 64)]
    sample_qinput, sample_scale = sglang_per_token_quant_fp8(sample_x)
    reference = functional.linear(sample_x, weight).float()
    quantized = fp8_scaled_mm(
        sample_qinput,
        qweight_t,
        sample_scale,
        weight_scale_t,
        torch.bfloat16,
        bias=None,
    ).float()
    cosine = functional.cosine_similarity(
        reference.flatten(),
        quantized.flatten(),
        dim=0,
    ).item()
    rmse = torch.mean((reference - quantized) ** 2).sqrt()
    normalized_rmse = (rmse / reference.square().mean().sqrt()).item()

    result = {
        **shape,
        "bf16_median_ms": bf16_median,
        "fp8_gemm_only_median_ms": fp8_median,
        "dynamic_w8a8_median_ms": dynamic_median,
        "fp8_gemm_only_speedup": bf16_median / fp8_median,
        "dynamic_w8a8_speedup": bf16_median / dynamic_median,
        "bf16_values_ms": bf16_values,
        "fp8_gemm_only_values_ms": fp8_values,
        "dynamic_w8a8_values_ms": dynamic_values,
        "cosine_similarity": cosine,
        "normalized_rmse": normalized_rmse,
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
    }

    del x, weight, qweight, qweight_t, qinput
    del input_scale, weight_scale, weight_scale_t
    del sample_x, sample_qinput, sample_scale, reference, quantized
    torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-m", type=int, default=32768)
    parser.add_argument("--max-shapes", type=int, default=6)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shapes = load_shapes(args.audit_json, args.max_m, args.max_shapes)
    if not shapes:
        print(f"[error] no FP8 CUTLASS shapes found in {args.audit_json}")
        return 2

    import torch

    if not torch.cuda.is_available():
        print("[error] CUDA is not available")
        return 2

    serialized_args = {
        **vars(args),
        "audit_json": str(args.audit_json),
        "output": str(args.output),
    }
    output: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "audit_json": str(args.audit_json),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "args": serialized_args,
        "results": [],
    }

    status = 0
    for index, shape in enumerate(shapes, start=1):
        print(
            f"[shape {index}/{len(shapes)}] "
            f"M={shape['m']} (runtime {shape['original_m']}) "
            f"K={shape['k']} N={shape['n']}",
            flush=True,
        )
        try:
            result = benchmark_shape(
                shape,
                warmups=args.warmups,
                iterations=args.iterations,
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            result = {**shape, "error": f"CUDA OOM: {exc}"}
            status = 1
        except Exception as exc:
            result = {**shape, "error": repr(exc)}
            status = 1
        output["results"].append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if "error" not in result:
            print(
                f"  GEMM-only={result['fp8_gemm_only_speedup']:.3f}x "
                f"dynamic-W8A8={result['dynamic_w8a8_speedup']:.3f}x",
                flush=True,
            )

    print(f"[output] {args.output}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
