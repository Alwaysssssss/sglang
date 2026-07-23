#!/usr/bin/env python3
"""Benchmark low-precision attention with VideoEdit runtime shapes.

The benchmark consumes ATTENTION_RUNTIME_AUDIT records emitted by a profiled
VideoEdit request. SageAttention receives BF16/FP16 QKV on every iteration, so
its reported latency includes activation quantization and smoothing overhead.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIAGNOSTICS_ROOT = REPO_ROOT / "videoedit_phase15_diagnostics"
DEFAULT_OUTPUT_ROOT = DEFAULT_DIAGNOSTICS_ROOT / "attention_microbench"
ROLE_ORDER = ("self", "text_cross", "image_cross")
DEFAULT_SHAPES = {
    "self": {
        "q_shape": [1, 41958, 20, 128],
        "k_shape": [1, 41958, 20, 128],
        "v_shape": [1, 41958, 20, 128],
    },
    "text_cross": {
        "q_shape": [1, 20979, 40, 128],
        "k_shape": [1, 512, 40, 128],
        "v_shape": [1, 512, 40, 128],
    },
    "image_cross": {
        "q_shape": [1, 20979, 40, 128],
        "k_shape": [1, 257, 40, 128],
        "v_shape": [1, 257, 40, 128],
    },
}


@dataclass(frozen=True)
class AttentionShape:
    role: str
    q_shape: tuple[int, int, int, int]
    k_shape: tuple[int, int, int, int]
    v_shape: tuple[int, int, int, int]
    source: str
    backend: str | None = None
    dtype: str | None = None


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def valid_shape(value: Any) -> tuple[int, int, int, int] | None:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or not all(isinstance(item, int) and item > 0 for item in value)
    ):
        return None
    return tuple(value)


def find_latest_audit(diagnostics_root: Path) -> Path | None:
    candidates = list(
        diagnostics_root.glob("phase15_*/fp8_layerwise/attention_runtime_audits.json")
    )
    candidates.extend(
        diagnostics_root.glob("phase15_*/fp8_nooffload/attention_runtime_audits.json")
    )
    candidates = [path for path in candidates if path.is_file()]
    return (
        max(candidates, key=lambda path: path.stat().st_mtime)
        if candidates
        else None
    )


def load_shapes(
    audit_path: Path | None,
    roles: list[str],
    *,
    allow_default_shapes: bool,
) -> list[AttentionShape]:
    records: list[dict[str, Any]] = []
    if audit_path is not None:
        data = json.loads(audit_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"attention audit must contain a JSON list: {audit_path}")
        records = [record for record in data if isinstance(record, dict)]

    shapes: list[AttentionShape] = []
    for role in roles:
        selected = None
        for record in records:
            if record.get("profile_kind") != role:
                continue
            if int(record.get("rank", 0)) != 0:
                continue
            q_shape = valid_shape(record.get("q_shape"))
            k_shape = valid_shape(record.get("k_shape"))
            v_shape = valid_shape(record.get("v_shape"))
            if q_shape and k_shape and v_shape:
                selected = AttentionShape(
                    role=role,
                    q_shape=q_shape,
                    k_shape=k_shape,
                    v_shape=v_shape,
                    source=str(audit_path),
                    backend=str(record.get("backend") or ""),
                    dtype=str(record.get("dtype") or ""),
                )
                break
        if selected is not None:
            shapes.append(selected)
            continue

        if not allow_default_shapes:
            source = str(audit_path) if audit_path is not None else "no audit file"
            raise ValueError(f"missing rank-0 {role!r} shape in {source}")
        fallback = DEFAULT_SHAPES[role]
        shapes.append(
            AttentionShape(
                role=role,
                q_shape=tuple(fallback["q_shape"]),
                k_shape=tuple(fallback["k_shape"]),
                v_shape=tuple(fallback["v_shape"]),
                source="built-in fallback",
            )
        )
    return shapes


def make_backend(name: str, softmax_scale: float) -> Callable:
    if name == "flash":
        from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
            flash_attn_func,
        )

        def run_flash(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
            return flash_attn_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=q.shape[1],
                max_seqlen_k=k.shape[1],
                softmax_scale=softmax_scale,
                causal=False,
                return_softmax_lse=False,
                ver=3,
            )

        return run_flash

    try:
        import sageattention
    except ImportError as exc:
        raise RuntimeError(
            "SageAttention is not installed. Install sageattention==2.2.0 "
            "with --no-build-isolation before running this benchmark."
        ) from exc

    common_kwargs = {
        "tensor_layout": "NHD",
        "is_causal": False,
        "qk_quant_gran": "per_thread",
        "sm_scale": softmax_scale,
        "smooth_k": True,
        "return_lse": False,
    }
    if name == "sage_fp16":
        fn = getattr(sageattention, "sageattn_qk_int8_pv_fp16_cuda")

        def run_sage_fp16(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
            return fn(
                q,
                k,
                v,
                pv_accum_dtype="fp16+fp32",
                **common_kwargs,
            )

        return run_sage_fp16

    if name == "sage_fp8":
        fn = getattr(sageattention, "sageattn_qk_int8_pv_fp8_cuda")

        def run_sage_fp8(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
            return fn(
                q,
                k,
                v,
                pv_accum_dtype="fp32+fp16",
                **common_kwargs,
            )

        return run_sage_fp8

    raise ValueError(f"unsupported backend: {name}")


def timed_calls(
    fn: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    warmups: int,
    iterations: int,
) -> tuple[list[float], torch.Tensor, int]:
    output = None
    for _ in range(warmups):
        output = fn(q, k, v)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(q.device)

    elapsed_ms: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = fn(q, k, v)
        end.record()
        end.synchronize()
        elapsed_ms.append(float(start.elapsed_time(end)))

    assert output is not None
    return elapsed_ms, output, int(torch.cuda.max_memory_allocated(q.device))


def tensor_error_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    chunk_tokens: int,
) -> dict[str, float | bool]:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"output shape mismatch: reference={reference.shape}, candidate={candidate.shape}"
        )

    dot = ref_norm = candidate_norm = squared_error = absolute_error = 0.0
    element_count = 0
    max_abs = 0.0
    finite = True
    for start in range(0, reference.shape[1], chunk_tokens):
        end = min(start + chunk_tokens, reference.shape[1])
        ref_chunk = reference[:, start:end].float()
        candidate_chunk = candidate[:, start:end].float()
        diff = candidate_chunk - ref_chunk
        finite = finite and bool(torch.isfinite(candidate_chunk).all().item())
        dot += float((ref_chunk * candidate_chunk).sum().item())
        ref_norm += float((ref_chunk * ref_chunk).sum().item())
        candidate_norm += float((candidate_chunk * candidate_chunk).sum().item())
        squared_error += float((diff * diff).sum().item())
        absolute_error += float(diff.abs().sum().item())
        max_abs = max(max_abs, float(diff.abs().max().item()))
        element_count += diff.numel()

    denominator = math.sqrt(ref_norm * candidate_norm)
    return {
        "finite": finite,
        "cosine_similarity": dot / denominator if denominator > 0 else 0.0,
        "relative_l2": (
            math.sqrt(squared_error / ref_norm) if ref_norm > 0 else math.inf
        ),
        "mean_abs_error": absolute_error / element_count,
        "max_abs_error": max_abs,
    }


def summarize_times(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = min(math.ceil(0.95 * len(ordered)) - 1, len(ordered) - 1)
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "p95_ms": ordered[p95_index],
    }


def projected_speedups(
    attention_speedup: float,
    *,
    attention_fraction: float,
    current_fp8_over_bf16: float,
) -> dict[str, float]:
    current_fp8_speedup = 1.0 / (
        (1.0 - attention_fraction) + attention_fraction / attention_speedup
    )
    return {
        "dit_over_current_fp8": current_fp8_speedup,
        "dit_over_bf16_estimate": current_fp8_speedup * current_fp8_over_bf16,
    }


def benchmark_shape(
    shape: AttentionShape,
    args: argparse.Namespace,
) -> dict[str, Any]:
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    q = torch.randn(shape.q_shape, device=device, dtype=dtype)
    k = torch.randn(shape.k_shape, device=device, dtype=dtype)
    v = torch.randn(shape.v_shape, device=device, dtype=dtype)
    softmax_scale = shape.q_shape[-1] ** -0.5

    result: dict[str, Any] = {
        "shape": asdict(shape),
        "dtype": args.dtype,
        "backends": {},
    }
    reference = None
    flash_median = None
    for backend_name in args.backends:
        record: dict[str, Any] = {"status": "running"}
        try:
            fn = make_backend(backend_name, softmax_scale)
            times, output, peak_memory = timed_calls(
                fn,
                q,
                k,
                v,
                warmups=args.warmups,
                iterations=args.iterations,
            )
            record.update(summarize_times(times))
            record["times_ms"] = times
            record["peak_memory_bytes"] = peak_memory
            record["status"] = "completed"
            if backend_name == "flash":
                reference = output.detach()
                flash_median = record["median_ms"]
            elif reference is not None:
                record["error_vs_flash"] = tensor_error_metrics(
                    reference,
                    output,
                    chunk_tokens=args.metric_chunk_tokens,
                )
            if flash_median is not None:
                record["speedup_over_flash"] = flash_median / record["median_ms"]
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = repr(exc)
        result["backends"][backend_name] = record

    del q, k, v, reference
    torch.cuda.empty_cache()
    return result


def evaluate_gate(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    self_result = next(
        (item for item in payload["results"] if item["shape"]["role"] == "self"),
        None,
    )
    if self_result is None:
        return {"status": "not_evaluated", "reason": "self shape was not benchmarked"}

    candidate = self_result["backends"].get("sage_fp8", {})
    if candidate.get("status") != "completed":
        return {
            "status": "failed",
            "reason": "sage_fp8 self-attention benchmark did not complete",
        }

    metrics = candidate.get("error_vs_flash", {})
    speedup = float(candidate.get("speedup_over_flash") or 0.0)
    checks = {
        "speedup": speedup >= args.min_attention_speedup,
        "finite": metrics.get("finite") is True,
        "cosine": float(metrics.get("cosine_similarity") or 0.0)
        >= args.min_cosine_similarity,
        "relative_l2": float(metrics.get("relative_l2") or math.inf)
        <= args.max_relative_l2,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "thresholds": {
            "min_attention_speedup": args.min_attention_speedup,
            "min_cosine_similarity": args.min_cosine_similarity,
            "max_relative_l2": args.max_relative_l2,
        },
        "projected_speedups": projected_speedups(
            speedup,
            attention_fraction=args.attention_fraction,
            current_fp8_over_bf16=args.current_fp8_over_bf16,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument(
        "--diagnostics-root",
        type=Path,
        default=DEFAULT_DIAGNOSTICS_ROOT,
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=list(ROLE_ORDER),
        default=list(ROLE_ORDER),
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["flash", "sage_fp16", "sage_fp8"],
        default=["flash", "sage_fp16", "sage_fp8"],
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric-chunk-tokens", type=int, default=1024)
    parser.add_argument("--min-attention-speedup", type=float, default=1.5)
    parser.add_argument("--min-cosine-similarity", type=float, default=0.995)
    parser.add_argument("--max-relative-l2", type=float, default=0.1)
    parser.add_argument("--attention-fraction", type=float, default=0.7486)
    parser.add_argument("--current-fp8-over-bf16", type=float, default=1.116)
    parser.add_argument("--allow-default-shapes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.warmups < 0 or args.iterations <= 0:
        parser.error(
            "--warmups must be non-negative and --iterations must be positive"
        )
    if not 0.0 < args.attention_fraction < 1.0:
        parser.error("--attention-fraction must be between 0 and 1")
    if args.backends[0] != "flash":
        parser.error("--backends must start with flash so accuracy has a reference")
    return args


def main() -> int:
    args = parse_args()
    audit_path = args.audit_json
    if audit_path is None:
        audit_path = find_latest_audit(args.diagnostics_root)
    if audit_path is not None and not audit_path.exists():
        raise FileNotFoundError(f"attention audit does not exist: {audit_path}")

    shapes = load_shapes(
        audit_path,
        args.roles,
        allow_default_shapes=args.allow_default_shapes,
    )
    output = args.output or (
        DEFAULT_OUTPUT_ROOT / f"attention_bench_{utc_tag()}.json"
    )
    dependency_state = {
        "sageattention_installed": (
            importlib.util.find_spec("sageattention") is not None
        ),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        dependency_state.update(
            {
                "device_name": torch.cuda.get_device_name(args.device),
                "device_capability": list(
                    torch.cuda.get_device_capability(args.device)
                ),
            }
        )

    payload: dict[str, Any] = {
        "status": "dry_run" if args.dry_run else "running",
        "audit_json": str(audit_path) if audit_path else None,
        "output": str(output),
        "dependencies": dependency_state,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "shapes": [asdict(shape) for shape in shapes],
        "results": [],
    }

    if args.dry_run:
        write_json(output, payload)
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the attention benchmark")

    for shape in shapes:
        print(
            f"[shape] {shape.role}: q={shape.q_shape} k={shape.k_shape} "
            f"v={shape.v_shape}",
            flush=True,
        )
        result = benchmark_shape(shape, args)
        payload["results"].append(result)
        for name, record in result["backends"].items():
            if record["status"] == "completed":
                print(
                    f"  {name:12s} median={record['median_ms']:.3f} ms "
                    f"speedup={record.get('speedup_over_flash', 1.0):.3f}x",
                    flush=True,
                )
            else:
                print(f"  {name:12s} failed: {record['error']}", flush=True)

    payload["gate"] = evaluate_gate(payload, args)
    payload["status"] = "completed"
    write_json(output, payload)
    print(f"[summary] {output}", flush=True)
    print(f"[gate] {payload['gate']['status']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
