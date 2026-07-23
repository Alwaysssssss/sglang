#!/usr/bin/env python3
"""Run the VideoEdit Phase 1.5 controlled BF16/FP8 offload matrix.

The script owns the service lifecycle. It starts one configuration at a time,
waits for readiness, runs the short profile81 benchmark, saves all logs, and
stops only the process group that it created.

Default order:
  fp8_nooffload -> bf16_nooffload -> bf16_layerwise -> fp8_layerwise

Use fp8_serialized_layerwise explicitly when --transformer-path points to an
offline FP8 checkpoint containing quantization metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_SCRIPT = REPO_ROOT / "scripts" / "videoedit_phase0_baseline.py"
MICROBENCH_SCRIPT = REPO_ROOT / "scripts" / "videoedit_fp8_linear_bench.py"
DEFAULT_MODEL_PATH = Path(
    "/mnt/nas/models/DifusserEdit/pretrain_models/VideoEdit-diffusers-model"
)
DEFAULT_TRANSFORMER_PATH = DEFAULT_MODEL_PATH / "transformer"
DEFAULT_VIDEO = REPO_ROOT / "demo" / "1080.mp4"
DEFAULT_MASK = REPO_ROOT / "demo" / "mask_1080_acc.mp4"
DEFAULT_REFERENCE = REPO_ROOT / "demo" / "local.png"
DEFAULT_OUT_ROOT = REPO_ROOT / "videoedit_phase15_diagnostics"
DEFAULT_PROMPT = (
    "A squirrel moves across a textured pavement, its bushy tail swaying as it walks."
)
NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
PROFILE_CATEGORY_EVENTS = {
    "fp8_activation_quant": "sglang.fp8.activation_quant",
    "fp8_gemm": "sglang.fp8.gemm",
    "attention_self_compute": "sglang.dit.attention.self.compute",
    "attention_text_cross_compute": "sglang.dit.attention.text_cross.compute",
    "attention_image_cross_compute": "sglang.dit.attention.image_cross.compute",
    "attention_unclassified_compute": "sglang.dit.attention.compute",
    "sp_communication": "sglang.dit.attention.sp_communication",
}
ATTENTION_CATEGORY_KEYS = (
    "attention_self_compute",
    "attention_text_cross_compute",
    "attention_image_cross_compute",
    "attention_unclassified_compute",
)
SYNTHETIC_CUDA_EVENT_NAMES = {
    "ProfilerStep*",
    *PROFILE_CATEGORY_EVENTS.values(),
}


@dataclass(frozen=True)
class Variant:
    name: str
    quantization: str | None
    layerwise_offload: bool
    serialized_checkpoint: bool = False


VARIANTS = {
    "fp8_nooffload": Variant("fp8_nooffload", "fp8_dynamic", False),
    "bf16_nooffload": Variant("bf16_nooffload", None, False),
    "bf16_layerwise": Variant("bf16_layerwise", None, True),
    "fp8_layerwise": Variant("fp8_layerwise", "fp8_dynamic", True),
    "fp8_serialized_layerwise": Variant(
        "fp8_serialized_layerwise", None, True, serialized_checkpoint=True
    ),
}
DEFAULT_VARIANTS = [
    "fp8_nooffload",
    "bf16_nooffload",
    "bf16_layerwise",
    "fp8_layerwise",
]


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, data: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def summarize_operator_profile(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    events = data.get("events", [])
    cuda_events = [event for event in events if event.get("device_type") == "cuda"]
    # key_averages includes CUDA-side copies of ProfilerStep, our profile
    # ranges, and NCCL operation aliases. They duplicate the underlying kernel
    # durations, so only raw kernels belong in the denominator and top-k list.
    raw_cuda_events = [
        event
        for event in cuda_events
        if str(event.get("name")) not in SYNTHETIC_CUDA_EVENT_NAMES
        and not str(event.get("name", "")).startswith("nccl:")
    ]
    total_gpu_us = sum(
        float(event.get("self_device_time_total_us") or 0.0)
        for event in raw_cuda_events
    )
    categories_us = {
        category: max(
            (
                float(event.get("self_device_time_total_us") or 0.0)
                for event in cuda_events
                if str(event.get("name")) == event_name
            ),
            default=0.0,
        )
        for category, event_name in PROFILE_CATEGORY_EVENTS.items()
    }
    categorized_gpu_us = sum(categories_us.values())
    categories_us["other_gpu_kernels"] = max(
        total_gpu_us - categorized_gpu_us,
        0.0,
    )
    categories = {
        name: {
            "device_time_ms": value / 1000.0,
            "percent_of_summed_gpu_kernel_time": (
                value * 100.0 / total_gpu_us if total_gpu_us > 0 else None
            ),
        }
        for name, value in categories_us.items()
    }
    attention_compute_us = sum(
        categories_us[category] for category in ATTENTION_CATEGORY_KEYS
    )
    categories["attention_compute"] = {
        "device_time_ms": attention_compute_us / 1000.0,
        "percent_of_summed_gpu_kernel_time": (
            attention_compute_us * 100.0 / total_gpu_us if total_gpu_us > 0 else None
        ),
    }
    top_cuda_kernels = sorted(
        (
            {
                "name": str(event.get("name")),
                "count": int(event.get("count") or 0),
                "self_device_time_ms": float(
                    event.get("self_device_time_total_us") or 0.0
                )
                / 1000.0,
            }
            for event in raw_cuda_events
        ),
        key=lambda event: event["self_device_time_ms"],
        reverse=True,
    )[:30]
    present_range_names = {
        str(event.get("name"))
        for event in cuda_events
        if float(event.get("self_device_time_total_us") or 0.0) > 0
    }
    required_range_names = {
        PROFILE_CATEGORY_EVENTS["sp_communication"],
    }
    fp8_range_names = {
        PROFILE_CATEGORY_EVENTS["fp8_activation_quant"],
        PROFILE_CATEGORY_EVENTS["fp8_gemm"],
    }
    if present_range_names & fp8_range_names:
        required_range_names.update(fp8_range_names)
    specialized_attention_names = {
        PROFILE_CATEGORY_EVENTS["attention_self_compute"],
        PROFILE_CATEGORY_EVENTS["attention_text_cross_compute"],
        PROFILE_CATEGORY_EVENTS["attention_image_cross_compute"],
    }
    if present_range_names & specialized_attention_names:
        required_range_names.update(specialized_attention_names)
    else:
        required_range_names.add(
            PROFILE_CATEGORY_EVENTS["attention_unclassified_compute"]
        )
    missing_ranges = sorted(required_range_names - present_range_names)
    return {
        "profile_path": str(path),
        "request_id": data.get("request_id"),
        "rank": data.get("rank"),
        "profile_mode": data.get("profile_mode"),
        "summed_gpu_kernel_time_ms": total_gpu_us / 1000.0,
        "categories": categories,
        "missing_profile_ranges": missing_ranges,
        "top_cuda_kernels": top_cuda_kernels,
    }


def summarize_profile_dir(profile_dir: Path) -> dict[str, Any]:
    profile_paths = sorted(profile_dir.glob("*.profile.json"))
    profiles: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in profile_paths:
        try:
            profiles.append(summarize_operator_profile(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"path": str(path), "error": repr(exc)})

    aggregate_categories_ms = {
        category: sum(
            float(
                profile.get("categories", {}).get(category, {}).get("device_time_ms")
                or 0.0
            )
            for profile in profiles
        )
        for category in (*PROFILE_CATEGORY_EVENTS, "other_gpu_kernels")
    }
    aggregate_total_ms = sum(
        float(profile.get("summed_gpu_kernel_time_ms") or 0.0) for profile in profiles
    )
    aggregate = {
        category: {
            "device_time_ms": value,
            "percent_of_summed_gpu_kernel_time": (
                value * 100.0 / aggregate_total_ms if aggregate_total_ms > 0 else None
            ),
        }
        for category, value in aggregate_categories_ms.items()
    }
    attention_compute_ms = sum(
        aggregate_categories_ms[category] for category in ATTENTION_CATEGORY_KEYS
    )
    aggregate["attention_compute"] = {
        "device_time_ms": attention_compute_ms,
        "percent_of_summed_gpu_kernel_time": (
            attention_compute_ms * 100.0 / aggregate_total_ms
            if aggregate_total_ms > 0
            else None
        ),
    }
    missing_ranges = sorted(
        {
            name
            for profile in profiles
            for name in profile.get("missing_profile_ranges", [])
        }
    )
    validation_errors = []
    if not profiles:
        validation_errors.append("no operator profile sidecar was generated")
    if aggregate_total_ms <= 0:
        validation_errors.append("operator profile contains no CUDA kernel time")
    if missing_ranges:
        validation_errors.append(
            "missing required profile ranges: " + ", ".join(missing_ranges)
        )
    return {
        "status": "completed" if not validation_errors else "invalid",
        "profile_dir": str(profile_dir),
        "profile_count": len(profiles),
        "summed_gpu_kernel_time_ms": aggregate_total_ms,
        "categories": aggregate,
        "profiles": profiles,
        "errors": errors,
        "validation_errors": validation_errors,
        "note": (
            "Percentages use summed rank-0 raw CUDA kernel duration after excluding "
            "ProfilerStep, profile-range, and NCCL alias duplicates. Concurrent "
            "kernels can overlap, so this is an operator attribution metric, not "
            "wall time."
        ),
    }


def print_profile_summary(summary: dict[str, Any]) -> None:
    print("[torch-profiler] rank-0 summed CUDA kernel breakdown", flush=True)
    for name, record in summary.get("categories", {}).items():
        duration = float(record.get("device_time_ms") or 0.0)
        percent = record.get("percent_of_summed_gpu_kernel_time")
        percent_text = f"{float(percent):6.2f}%" if percent is not None else "   n/a"
        print(f"  {name:24s} {duration:12.3f} ms  {percent_text}", flush=True)


def tail_text(path: Path, line_count: int = 80) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as file:
        return "".join(deque(file, maxlen=line_count))


def endpoint_ready(url: str, timeout: float = 3.0) -> bool:
    request = urllib.request.Request(url, method="GET")
    try:
        with NO_PROXY_OPENER.open(request, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def wait_for_port_free(host: str, port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not port_is_open(host, port):
            return True
        time.sleep(1)
    return not port_is_open(host, port)


def wait_for_server(
    proc: subprocess.Popen[str],
    *,
    base_url: str,
    startup_timeout: float,
    service_log: Path,
) -> float:
    started = time.monotonic()
    health_ready = False
    while time.monotonic() - started < startup_timeout:
        returncode = proc.poll()
        if returncode is not None:
            tail = tail_text(service_log)
            raise RuntimeError(
                f"service exited before readiness with code {returncode}\n{tail}"
            )

        if not health_ready:
            health_ready = endpoint_ready(f"{base_url}/health")
        if health_ready and endpoint_ready(f"{base_url}/v1/models", timeout=5):
            return time.monotonic() - started
        time.sleep(2)

    tail = tail_text(service_log)
    raise TimeoutError(
        f"service did not become ready within {startup_timeout:.0f}s\n{tail}"
    )


def process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def stop_process_group(
    proc: subprocess.Popen[str] | None,
    *,
    graceful_timeout: float,
) -> None:
    if proc is None:
        return

    # start_new_session=True makes the process PID its process-group ID.
    # The launcher can exit before a failed worker, so check the group itself.
    process_group = proc.pid
    for sig, timeout in (
        (signal.SIGINT, graceful_timeout),
        (signal.SIGTERM, 30.0),
        (signal.SIGKILL, 10.0),
    ):
        if not process_group_exists(process_group):
            return
        try:
            os.killpg(process_group, sig)
        except ProcessLookupError:
            return

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not process_group_exists(process_group):
                return
            time.sleep(0.5)


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {shlex.join(cmd)}\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
            return proc.wait()
        except KeyboardInterrupt:
            stop_process_group(proc, graceful_timeout=30)
            raise


def extract_json_marker(log_path: Path, marker: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not log_path.exists():
        return records
    with log_path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            if marker not in line:
                continue
            payload = line.split(marker, 1)[1].strip()
            try:
                records.append(json.loads(payload))
            except json.JSONDecodeError:
                records.append({"parse_error": payload})
    return records


def latest_summary(benchmark_dir: Path) -> Path | None:
    summaries = sorted(
        benchmark_dir.glob("phase0_*.summary.json"),
        key=lambda path: path.stat().st_mtime,
    )
    return summaries[-1] if summaries else None


def summarize_benchmark(summary_path: Path | None) -> dict[str, Any]:
    if summary_path is None or not summary_path.exists():
        return {"summary_path": None}

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "summary_path": str(summary_path),
        "status": data.get("status"),
        "error": data.get("error"),
        "scenarios": {},
    }
    for scenario in data.get("scenarios", []):
        scenario_name = scenario.get("scenario")
        formal_records = [
            record
            for record in scenario.get("records", [])
            if record.get("kind") == "run"
        ]
        denoise_values: list[float] = []
        for record in formal_records:
            perf_path = Path(str(record.get("perf_path", "")))
            if not perf_path.exists():
                continue
            try:
                perf = json.loads(perf_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for step in perf.get("steps", []):
                if str(step.get("name", "")).endswith("DenoisingStage"):
                    duration = step.get("duration_ms")
                    if isinstance(duration, (int, float)):
                        denoise_values.append(float(duration))
                    break

        result["scenarios"][scenario_name] = {
            "total_stats": scenario.get("stats", {}),
            "denoising_formal_count": len(denoise_values),
            "denoising_median_ms": (
                statistics.median(denoise_values) if denoise_values else None
            ),
            "denoising_values_ms": denoise_values,
        }
    return result


def build_service_command(
    args: argparse.Namespace,
    variant: Variant,
    variant_dir: Path,
) -> list[str]:
    command = [
        args.serve_executable,
        "serve",
        "--model-type",
        "diffusion",
        "--model-path",
        str(args.model_path),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--num-gpus",
        "2",
        "--sp-degree",
        "2",
        "--ulysses-degree",
        "2",
        "--ring-degree",
        "1",
        "--dit-cpu-offload",
        "false",
        "--dit-layerwise-offload",
        str(variant.layerwise_offload).lower(),
        "--text-encoder-cpu-offload",
        "true",
        "--image-encoder-cpu-offload",
        "true",
        "--vae-cpu-offload",
        "true",
        "--warmup",
        "true",
        "--warmup-steps",
        "1",
        "--output-path",
        str(variant_dir / "server_outputs"),
        "--input-save-path",
        str(variant_dir / "server_inputs"),
        "--transformer-path",
        str(args.transformer_path),
    ]
    if variant.quantization:
        command.extend(["--transformer-quantization", variant.quantization])
    command.extend(args.server_extra_arg)
    return command


def build_benchmark_command(
    args: argparse.Namespace,
    variant: Variant,
    variant_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(BASELINE_SCRIPT),
        *args.scenarios,
        "--base-url",
        f"http://127.0.0.1:{args.port}",
        "--video",
        str(args.video),
        "--mask",
        str(args.mask),
        "--reference",
        str(args.reference),
        "--out-dir",
        str(variant_dir / "benchmark"),
        "--task-prefix",
        f"phase15_{variant.name}",
        "--model-path",
        str(args.model_path),
        "--transformer-path",
        str(args.transformer_path),
        "--prompt",
        args.prompt,
        "--seed",
        str(args.seed),
        "--guidance-scale",
        str(args.guidance_scale),
        "--bbox-expand-scale",
        str(args.bbox_expand_scale),
        "--poll-interval",
        str(args.poll_interval),
        "--task-timeout",
        "-1",
        "--no-docker-logs",
    ]
    if args.negative_prompt is not None:
        command.extend(["--negative-prompt", args.negative_prompt])
    if args.benchmark_warmups is not None:
        command.extend(["--warmups", str(args.benchmark_warmups)])
    if args.benchmark_runs is not None:
        command.extend(["--runs", str(args.benchmark_runs)])
    if args.profile:
        command.extend(
            [
                "--profile",
                "--num-profiled-timesteps",
                str(args.num_profiled_timesteps),
            ]
        )
        if args.profile_all_stages:
            command.append("--profile-all-stages")
    return command


def diagnostic_env(profile_dir: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["SGLANG_DIFFUSION_QUANT_AUDIT"] = "1"
    env["SGLANG_DIFFUSION_LINEAR_RUNTIME_AUDIT"] = "1"
    env["SGLANG_DIT_ATTENTION_AUDIT"] = "1"
    if profile_dir is not None:
        env["SGLANG_TORCH_PROFILER_DIR"] = str(profile_dir)
        env["SGLANG_TORCH_PROFILER_WITH_STACK"] = "0"
        env["SGLANG_DIT_PROFILE_RANGES"] = "1"
        env["SGLANG_FP8_NVTX"] = "1"
    no_proxy = env.get("NO_PROXY") or env.get("no_proxy") or ""
    entries = [entry for entry in no_proxy.split(",") if entry]
    for entry in ("127.0.0.1", "localhost"):
        if entry not in entries:
            entries.append(entry)
    env["NO_PROXY"] = ",".join(entries)
    env["no_proxy"] = env["NO_PROXY"]
    return env


def validate_inputs(args: argparse.Namespace) -> None:
    if shutil.which(args.serve_executable) is None:
        raise FileNotFoundError(
            f"serve executable is not available: {args.serve_executable}"
        )
    for label, path in (
        ("model", args.model_path),
        ("transformer", args.transformer_path),
        ("video", args.video),
        ("mask", args.mask),
        ("reference", args.reference),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} path does not exist: {path}")
    if not BASELINE_SCRIPT.exists():
        raise FileNotFoundError(f"baseline script does not exist: {BASELINE_SCRIPT}")
    if not MICROBENCH_SCRIPT.exists():
        raise FileNotFoundError(
            f"microbenchmark script does not exist: {MICROBENCH_SCRIPT}"
        )


def validate_variant_checkpoint(args: argparse.Namespace, variant: Variant) -> None:
    if not variant.serialized_checkpoint:
        return
    config_path = args.transformer_path / "config.json"
    try:
        transformer_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot read serialized FP8 transformer config: {config_path}"
        ) from error
    quantization_config = transformer_config.get("quantization_config")
    if not isinstance(quantization_config, dict):
        raise ValueError(
            "fp8_serialized_layerwise requires quantization_config in the "
            f"transformer config: {config_path}"
        )
    if quantization_config.get("quant_method") != "fp8":
        raise ValueError(
            "fp8_serialized_layerwise requires quant_method=fp8, got "
            f"{quantization_config.get('quant_method')!r}"
        )
    if quantization_config.get("weight_scale_granularity") != "channel":
        raise ValueError("fp8_serialized_layerwise requires per-channel weight scales")


def calculate_comparisons(records: list[dict[str, Any]]) -> dict[str, Any]:
    medians: dict[str, dict[str, float | None]] = {}
    for record in records:
        benchmark = record.get("benchmark_summary", {})
        profile = benchmark.get("scenarios", {}).get("profile81", {})
        medians[record["name"]] = {
            "total_median_ms": profile.get("total_stats", {}).get("median_ms"),
            "denoising_median_ms": profile.get("denoising_median_ms"),
        }

    comparisons: dict[str, Any] = {"variant_medians": medians}
    for offload_name, bf16_name, fp8_name in (
        ("layerwise", "bf16_layerwise", "fp8_layerwise"),
        ("nooffload", "bf16_nooffload", "fp8_nooffload"),
    ):
        pair: dict[str, float | None] = {}
        for metric in ("total_median_ms", "denoising_median_ms"):
            bf16 = medians.get(bf16_name, {}).get(metric)
            fp8 = medians.get(fp8_name, {}).get(metric)
            pair[f"{metric}_speedup_bf16_over_fp8"] = (
                float(bf16) / float(fp8)
                if isinstance(bf16, (int, float))
                and isinstance(fp8, (int, float))
                and fp8 > 0
                else None
            )
        comparisons[offload_name] = pair
    return comparisons


def run_variant(
    args: argparse.Namespace,
    variant: Variant,
    run_dir: Path,
) -> dict[str, Any]:
    validate_variant_checkpoint(args, variant)
    variant_dir = run_dir / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "server_outputs").mkdir(exist_ok=True)
    (variant_dir / "server_inputs").mkdir(exist_ok=True)
    (variant_dir / "benchmark").mkdir(exist_ok=True)

    service_log = variant_dir / "service.log"
    benchmark_log = variant_dir / "benchmark_driver.log"
    profile_dir = variant_dir / "torch_profiler"
    if args.profile:
        profile_dir.mkdir(exist_ok=True)
    service_cmd = build_service_command(args, variant, variant_dir)
    benchmark_cmd = build_benchmark_command(args, variant, variant_dir)
    record: dict[str, Any] = {
        "name": variant.name,
        "config": asdict(variant),
        "started_at": utc_now(),
        "service_command": service_cmd,
        "benchmark_command": benchmark_cmd,
        "service_log": str(service_log),
        "benchmark_driver_log": str(benchmark_log),
        "status": "running",
    }

    print(f"\n[variant] {variant.name}", flush=True)
    print(f"[serve] {shlex.join(service_cmd)}", flush=True)

    proc: subprocess.Popen[str] | None = None
    service_file = None
    env = diagnostic_env(profile_dir if args.profile else None)
    try:
        if port_is_open("127.0.0.1", args.port):
            raise RuntimeError(
                f"port {args.port} is already in use; stop the existing service first"
            )

        service_file = service_log.open("w", encoding="utf-8")
        service_file.write(f"$ {shlex.join(service_cmd)}\n")
        service_file.flush()
        proc = subprocess.Popen(
            service_cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=service_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        ready_s = wait_for_server(
            proc,
            base_url=f"http://127.0.0.1:{args.port}",
            startup_timeout=args.startup_timeout,
            service_log=service_log,
        )
        record["service_ready_s"] = ready_s
        print(f"[ready] {variant.name} in {ready_s:.1f}s", flush=True)

        returncode = run_logged(benchmark_cmd, benchmark_log, env)
        record["benchmark_returncode"] = returncode
        record["status"] = "completed" if returncode == 0 else "benchmark_failed"
    except KeyboardInterrupt:
        record["status"] = "interrupted"
        record["error"] = "KeyboardInterrupt"
        raise
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = repr(exc)
        print(f"[error] {variant.name}: {exc}", file=sys.stderr, flush=True)
    finally:
        stop_process_group(proc, graceful_timeout=args.stop_timeout)
        if service_file is not None:
            service_file.close()
        record["finished_at"] = utc_now()
        record["service_tail"] = tail_text(service_log)
        record["quantization_audits"] = extract_json_marker(
            service_log, "QUANTIZATION_AUDIT "
        )
        record["linear_runtime_audits"] = extract_json_marker(
            service_log, "LINEAR_RUNTIME_AUDIT "
        )
        record["attention_runtime_audits"] = extract_json_marker(
            service_log, "ATTENTION_RUNTIME_AUDIT "
        )
        atomic_write_json(
            variant_dir / "quantization_audits.json",
            record["quantization_audits"],
        )
        atomic_write_json(
            variant_dir / "linear_runtime_audits.json",
            record["linear_runtime_audits"],
        )
        atomic_write_json(
            variant_dir / "attention_runtime_audits.json",
            record["attention_runtime_audits"],
        )
        summary_path = latest_summary(variant_dir / "benchmark")
        record["benchmark_summary"] = summarize_benchmark(summary_path)
        if args.profile:
            record["torch_profiler"] = summarize_profile_dir(profile_dir)
            atomic_write_json(
                profile_dir / "operator_breakdown.json",
                record["torch_profiler"],
            )
            print_profile_summary(record["torch_profiler"])
            if record["torch_profiler"]["status"] != "completed":
                record["status"] = "profiler_failed"
                record["profiler_error"] = "; ".join(
                    record["torch_profiler"]["validation_errors"]
                )
        if not wait_for_port_free(
            "127.0.0.1",
            args.port,
            timeout=args.port_cleanup_timeout,
        ):
            record["cleanup_error"] = (
                f"port {args.port} remained in use after stopping {variant.name}"
            )
    return record


def find_fp8_runtime_audit(run_dir: Path) -> Path | None:
    for variant_name in (
        "fp8_nooffload",
        "fp8_layerwise",
        "fp8_serialized_layerwise",
    ):
        audit_path = run_dir / variant_name / "linear_runtime_audits.json"
        if not audit_path.exists():
            continue
        try:
            records = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if any(
            record.get("method") == "Fp8LinearMethod"
            and "scaled_mm" in str(record.get("kernel", ""))
            for record in records
        ):
            return audit_path
    return None


def build_microbenchmark_command(
    args: argparse.Namespace,
    *,
    audit_path: Path,
    output_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(MICROBENCH_SCRIPT),
        "--audit-json",
        str(audit_path),
        "--output",
        str(output_path),
        "--max-m",
        str(args.microbench_max_m),
        "--max-shapes",
        str(args.microbench_max_shapes),
        "--warmups",
        str(args.microbench_warmups),
        "--iterations",
        str(args.microbench_iterations),
    ]


def run_microbenchmark(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    audit_path: Path,
) -> dict[str, Any]:
    output_path = run_dir / "fp8_linear_microbench.json"
    log_path = run_dir / "fp8_linear_microbench.log"
    command = build_microbenchmark_command(
        args,
        audit_path=audit_path,
        output_path=output_path,
    )
    record: dict[str, Any] = {
        "started_at": utc_now(),
        "audit_json": str(audit_path),
        "output": str(output_path),
        "log": str(log_path),
        "command": command,
        "status": "running",
    }
    print(f"\n[microbenchmark] {shlex.join(command)}", flush=True)
    returncode = run_logged(command, log_path, diagnostic_env())
    record["returncode"] = returncode
    record["status"] = "completed" if returncode == 0 else "failed"
    record["finished_at"] = utc_now()
    if output_path.exists():
        try:
            record["results"] = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            record["result_parse_error"] = repr(exc)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS),
        default=DEFAULT_VARIANTS,
        help="Configurations to run, in the requested order.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["smoke", "profile81", "single81", "full"],
        default=["profile81"],
        help="Benchmark scenarios for each service. Default: profile81.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--transformer-path",
        type=Path,
        default=DEFAULT_TRANSFORMER_PATH,
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Explicit run directory, useful when resuming selected variants.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--serve-executable", default="sglang")
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--stop-timeout", type=float, default=90)
    parser.add_argument("--port-cleanup-timeout", type=float, default=60)
    parser.add_argument("--poll-interval", type=int, default=15)
    parser.add_argument(
        "--benchmark-warmups",
        type=int,
        default=None,
        help="Override each selected scenario's benchmark warmup count.",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=None,
        help="Override each selected scenario's formal benchmark run count.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--bbox-expand-scale", type=float, default=0.3)
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Enable the diffusion torch profiler and operator breakdown for "
            "each benchmark request. Use the smoke scenario for a short, "
            "representative 81-frame internal shape."
        ),
    )
    parser.add_argument(
        "--num-profiled-timesteps",
        type=int,
        default=1,
        help="Denoising timesteps to capture after one profiler warmup step.",
    )
    parser.add_argument(
        "--profile-all-stages",
        action="store_true",
        help="Capture all pipeline stages instead of denoising only.",
    )
    parser.add_argument(
        "--server-extra-arg",
        action="append",
        default=[],
        help=(
            "Append one raw serve argument token. Repeat as needed; use "
            "--server-extra-arg=--flag for tokens beginning with '-'."
        ),
    )
    parser.add_argument(
        "--skip-microbench",
        action="store_true",
        help="Skip the post-matrix BF16/FP8 Linear microbenchmark.",
    )
    parser.add_argument("--microbench-max-m", type=int, default=32768)
    parser.add_argument("--microbench-max-shapes", type=int, default=6)
    parser.add_argument("--microbench-warmups", type=int, default=3)
    parser.add_argument("--microbench-iterations", type=int, default=10)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed variant.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write the plan without starting services.",
    )
    args = parser.parse_args()
    if args.num_profiled_timesteps == 0 or args.num_profiled_timesteps < -1:
        parser.error("--num-profiled-timesteps must be positive or -1")
    if args.benchmark_warmups is not None and args.benchmark_warmups < 0:
        parser.error("--benchmark-warmups must be non-negative")
    if args.benchmark_runs is not None and args.benchmark_runs < 1:
        parser.error("--benchmark-runs must be positive")
    return args


def main() -> int:
    args = parse_args()
    validate_inputs(args)

    run_dir = (
        args.run_dir.resolve()
        if args.run_dir is not None
        else (args.out_root / f"phase15_{utc_tag()}").resolve()
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "matrix_summary.json"
    previous_summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_summary = {}

    summary: dict[str, Any] = {
        "started_at": previous_summary.get("started_at", utc_now()),
        "resumed_at": utc_now() if previous_summary else None,
        "run_dir": str(run_dir),
        "args": {
            **vars(args),
            "model_path": str(args.model_path),
            "transformer_path": str(args.transformer_path),
            "video": str(args.video),
            "mask": str(args.mask),
            "reference": str(args.reference),
            "out_root": str(args.out_root),
            "run_dir": str(args.run_dir) if args.run_dir else None,
        },
        "variants": previous_summary.get("variants", []),
        "microbenchmark": previous_summary.get("microbenchmark"),
        "status": "running",
    }
    atomic_write_json(summary_path, summary)

    if args.dry_run:
        for name in args.variants:
            variant = VARIANTS[name]
            validate_variant_checkpoint(args, variant)
            variant_dir = run_dir / name
            print(f"\n[{name}]")
            print(shlex.join(build_service_command(args, variant, variant_dir)))
            print(shlex.join(build_benchmark_command(args, variant, variant_dir)))
        if not args.skip_microbench:
            audit_path = run_dir / "fp8_nooffload" / "linear_runtime_audits.json"
            output_path = run_dir / "fp8_linear_microbench.json"
            print("\n[microbenchmark after matrix]")
            print(
                shlex.join(
                    build_microbenchmark_command(
                        args,
                        audit_path=audit_path,
                        output_path=output_path,
                    )
                )
            )
        summary["status"] = "dry_run"
        summary["finished_at"] = utc_now()
        atomic_write_json(summary_path, summary)
        print(f"\n[plan] {summary_path}")
        return 0

    if port_is_open("127.0.0.1", args.port):
        print(
            f"[error] port {args.port} is already in use; stop the existing service first",
            file=sys.stderr,
        )
        summary["status"] = "blocked"
        summary["error"] = f"port {args.port} is already in use"
        summary["finished_at"] = utc_now()
        atomic_write_json(summary_path, summary)
        return 2

    try:
        for name in args.variants:
            record = run_variant(args, VARIANTS[name], run_dir)
            summary["variants"] = [
                existing
                for existing in summary["variants"]
                if existing.get("name") != name
            ]
            summary["variants"].append(record)
            variant_order = {variant_name: i for i, variant_name in enumerate(VARIANTS)}
            summary["variants"].sort(
                key=lambda existing: variant_order.get(existing.get("name"), 999)
            )
            summary["comparisons"] = calculate_comparisons(summary["variants"])
            atomic_write_json(summary_path, summary)
            if record.get("cleanup_error"):
                summary["status"] = "blocked"
                break
            if args.fail_fast and record["status"] != "completed":
                break
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        summary["finished_at"] = utc_now()
        summary["comparisons"] = calculate_comparisons(summary["variants"])
        atomic_write_json(summary_path, summary)
        print(f"\n[summary] {summary_path}", flush=True)
        return 130

    if summary.get("status") != "blocked" and not args.skip_microbench:
        audit_path = find_fp8_runtime_audit(run_dir)
        if audit_path is None:
            summary["microbenchmark"] = {
                "status": "skipped",
                "reason": "no runtime fp8_scaled_mm audit records were produced",
            }
        else:
            summary["microbenchmark"] = run_microbenchmark(
                args,
                run_dir=run_dir,
                audit_path=audit_path,
            )
        atomic_write_json(summary_path, summary)

    statuses = [record["status"] for record in summary["variants"]]
    micro_status = (summary.get("microbenchmark") or {}).get("status")
    all_completed = (
        bool(statuses)
        and all(status == "completed" for status in statuses)
        and micro_status in (None, "completed", "skipped")
    )
    if summary.get("status") != "blocked":
        summary["status"] = "completed" if all_completed else "completed_with_failures"
    summary["finished_at"] = utc_now()
    summary["comparisons"] = calculate_comparisons(summary["variants"])
    atomic_write_json(summary_path, summary)
    print(f"\n[summary] {summary_path}", flush=True)
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
