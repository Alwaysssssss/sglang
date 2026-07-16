#!/usr/bin/env python3
"""Run the fixed VividVR acceleration benchmark matrix end to end."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence
from urllib.parse import unquote, urlsplit


REPO_ROOT = Path("/home/zhiheng/sglang")


class BenchmarkError(RuntimeError):
    """Base error for benchmark configuration and execution failures."""


class BenchmarkConfigError(BenchmarkError):
    """Raised before execution when a benchmark configuration is invalid."""


class BenchmarkDataError(BenchmarkError):
    """Raised when runtime evidence does not match the experiment contract."""


class BenchmarkCleanupError(BenchmarkError):
    """Raised when an owned process cannot be cleaned up safely."""


class SchemeStatus(str, Enum):
    EXECUTABLE = "executable"
    UNSUPPORTED = "unsupported"


class RunRole(str, Enum):
    WARMUP = "warmup"
    FORMAL = "formal"


@dataclass(frozen=True)
class Scheme:
    scheme_id: str
    name: str
    gpu_count: int
    backend: str
    parallel_mode: str
    sp_degree: int
    compile_enabled: bool
    modulation_fusion: bool
    controls: tuple[str, ...]
    status: SchemeStatus = SchemeStatus.EXECUTABLE
    unsupported_reason: str | None = None

    @property
    def executable(self) -> bool:
        return self.status is SchemeStatus.EXECUTABLE

    @property
    def cfg_parallel(self) -> bool:
        return self.parallel_mode in {"cfg", "cfg_sp"}

    @property
    def expected_effective_backend(self) -> str:
        return f"{self.backend}_sp" if self.sp_degree > 1 else self.backend


@dataclass(frozen=True)
class BenchmarkConfig:
    repo_root: Path = REPO_ROOT
    python_executable: Path = REPO_ROOT / ".venv/bin/python"
    model_path: Path = Path("/home/zhiheng/ckpts/CogVideoX1.5-5B")
    vividvr_path: Path = Path("/home/zhiheng/ckpts/Vivid-VR")
    input_video: Path = Path(
        "/home/zhiheng/input/test_video_long_960x720_130f.mp4"
    )
    caption_file: Path = Path(
        "/home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars/"
        "quad-test-video-long-960x720-130f-run2-20260708T060202Z.txt"
    )
    reference_video: Path = Path(
        "/home/zhiheng/sglang/Vivid_Acceptance/result_videos/"
        "service_benchmark/downloads/"
        "quad-test-video-long-960x720-130f-run2-20260708T060202Z."
        "bridge-downloaded.mp4"
    )
    output_root: Path = REPO_ROOT / "Vivid_Acceptance/acceleration_benchmark"
    gpu_ids: tuple[int, ...] = (0, 1, 2, 3)
    host: str = "127.0.0.1"
    service_port: int = 31221
    master_port: int = 30221
    scheduler_port: int = 56221
    caption_port: int = 31200
    callback_port: int = 39090
    s3_port: int = 4566
    dist_timeout_seconds: int = 3600
    caption_timeout_seconds: int = 1800
    service_start_timeout_seconds: int = 1800
    request_timeout_seconds: int = 21600
    poll_interval_seconds: float = 10.0
    s3_bucket: str = "flowcut"
    allow_idle_gpu_processes: bool = False


def _scheme(
    scheme_id: str,
    name: str,
    *,
    gpu_count: int = 1,
    backend: str = "fa",
    parallel_mode: str = "single",
    sp_degree: int = 1,
    compile_enabled: bool = False,
    modulation_fusion: bool = False,
    controls: tuple[str, ...] = (),
) -> Scheme:
    return Scheme(
        scheme_id=scheme_id,
        name=name,
        gpu_count=gpu_count,
        backend=backend,
        parallel_mode=parallel_mode,
        sp_degree=sp_degree,
        compile_enabled=compile_enabled,
        modulation_fusion=modulation_fusion,
        controls=controls,
    )


def _unsupported(
    scheme_id: str,
    name: str,
    reason: str,
    *,
    controls: tuple[str, ...],
) -> Scheme:
    return Scheme(
        scheme_id=scheme_id,
        name=name,
        gpu_count=1,
        backend="fa",
        parallel_mode="single",
        sp_degree=1,
        compile_enabled=True,
        modulation_fusion=False,
        controls=controls,
        status=SchemeStatus.UNSUPPORTED,
        unsupported_reason=reason,
    )


SCHEMES: dict[str, Scheme] = {
    "R0": _scheme("R0", "单卡 SDPA eager", backend="sdpa"),
    "R1": _scheme("R1", "单卡 FA eager", controls=("R0",)),
    "R2": _scheme(
        "R2", "单卡 FA + torch.compile", compile_enabled=True, controls=("R1",)
    ),
    "R3": _scheme(
        "R3",
        "双卡 SP2 + FA-SP + torch.compile",
        gpu_count=2,
        parallel_mode="sp",
        sp_degree=2,
        compile_enabled=True,
        controls=("R2",),
    ),
    "R4": _scheme(
        "R4",
        "四卡 SP4 + FA-SP + torch.compile",
        gpu_count=4,
        parallel_mode="sp",
        sp_degree=4,
        compile_enabled=True,
        controls=("R2",),
    ),
    "R5": _scheme(
        "R5",
        "四卡 CFG2×SP2 + FA-SP + torch.compile",
        gpu_count=4,
        parallel_mode="cfg_sp",
        sp_degree=2,
        compile_enabled=True,
        controls=("R4",),
    ),
    "R6": _scheme(
        "R6",
        "单卡 FA + torch.compile + modulation fusion",
        compile_enabled=True,
        modulation_fusion=True,
        controls=("R2",),
    ),
    "R7": _unsupported(
        "R7",
        "Cache-DiT",
        "Cache-DiT is wired only for supported diffusers pipelines; the native "
        "VividVR denoise path has no verified integration.",
        controls=("R2",),
    ),
    "R8": _unsupported(
        "R8",
        "TeaCache",
        "TeaCache is not implemented in the native VividVR denoise path.",
        controls=("R2",),
    ),
    "R9": _unsupported(
        "R9",
        "通用量化",
        "Quantization loader plumbing exists, but no verified VividVR weight or "
        "CogVideoX linear quantization path is available.",
        controls=("R2",),
    ),
    "R99": _scheme(
        "R99",
        "双卡全量已实现加速",
        gpu_count=2,
        parallel_mode="sp",
        sp_degree=2,
        compile_enabled=True,
        modulation_fusion=True,
        controls=("R3",),
    ),
    "R100": _scheme(
        "R100",
        "四卡全量已实现加速",
        gpu_count=4,
        parallel_mode="cfg_sp",
        sp_degree=2,
        compile_enabled=True,
        modulation_fusion=True,
        controls=("R4", "R5"),
    ),
}


VIVIDVR_STAGE_NAMES = (
    "VividVRInputValidationStage",
    "VividVRPromptPreparationStage",
    "VividVRTemporalWindowPlanningStage",
    "VividVRLongClipPreparationStage",
    "VividVRTimestepPreparationStage",
    "VividVRMultiClipDenoisingStage",
    "VividVRMultiClipDecodeTrimStage",
    "VividVRTemporalStitchPostprocessStage",
)


@dataclass(frozen=True)
class PerfSummary:
    model_inference_runtime_seconds: float
    stage_seconds: dict[str, float]
    unclassified_seconds: float
    denoising_runtime_seconds: float
    denoise_fraction: float
    temporal_clip_count: int
    inference_step_count: int
    mean_step_seconds: float
    steady_step_median_seconds: float | None


def _required_number(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkDataError(f"{key} must be numeric, got {value!r}")
    return float(value)


def summarize_perf(perf: Mapping[str, Any]) -> PerfSummary:
    total_seconds = _required_number(perf, "total_duration_ms") / 1000.0
    raw_steps = perf.get("steps")
    if not isinstance(raw_steps, list):
        raise BenchmarkDataError("steps must be a list")

    stage_seconds: dict[str, float] = {}
    for item in raw_steps:
        if not isinstance(item, Mapping):
            raise BenchmarkDataError("each stage entry must be an object")
        name = item.get("name")
        if name not in VIVIDVR_STAGE_NAMES:
            continue
        if name in stage_seconds:
            raise BenchmarkDataError(f"duplicate stage in perf dump: {name}")
        stage_seconds[str(name)] = _required_number(item, "duration_ms") / 1000.0
    missing = [name for name in VIVIDVR_STAGE_NAMES if name not in stage_seconds]
    if missing:
        raise BenchmarkDataError(f"missing stages in perf dump: {', '.join(missing)}")
    stage_seconds = {name: stage_seconds[name] for name in VIVIDVR_STAGE_NAMES}

    raw_denoise_steps = perf.get("denoise_steps_ms")
    if not isinstance(raw_denoise_steps, list) or not raw_denoise_steps:
        raise BenchmarkDataError("denoise_steps_ms must be a non-empty list")
    per_step_seconds = [
        _required_number(item, "duration_ms") / 1000.0
        for item in raw_denoise_steps
        if isinstance(item, Mapping)
    ]
    if len(per_step_seconds) != len(raw_denoise_steps):
        raise BenchmarkDataError("each denoise step entry must be an object")

    debug = perf.get("meta", {}).get("vividvr_debug", {})
    if not isinstance(debug, Mapping):
        raise BenchmarkDataError("meta.vividvr_debug must be an object")
    clip_count = debug.get("num_clips")
    if isinstance(clip_count, bool) or not isinstance(clip_count, int):
        raise BenchmarkDataError("meta.vividvr_debug.num_clips must be an integer")

    denoising_seconds = stage_seconds["VividVRMultiClipDenoisingStage"]
    classified_seconds = sum(stage_seconds.values())
    unclassified_seconds = total_seconds - classified_seconds
    if unclassified_seconds < -1e-6:
        raise BenchmarkDataError(
            "sum of stage durations exceeds total_duration_ms: "
            f"{classified_seconds:.6f}s > {total_seconds:.6f}s"
        )
    return PerfSummary(
        model_inference_runtime_seconds=total_seconds,
        stage_seconds=stage_seconds,
        unclassified_seconds=max(0.0, unclassified_seconds),
        denoising_runtime_seconds=denoising_seconds,
        denoise_fraction=(
            denoising_seconds / total_seconds if total_seconds > 0 else 0.0
        ),
        temporal_clip_count=clip_count,
        inference_step_count=len(per_step_seconds),
        mean_step_seconds=statistics.fmean(per_step_seconds),
        steady_step_median_seconds=(
            statistics.median(per_step_seconds[1:])
            if len(per_step_seconds) > 1
            else None
        ),
    )


def validate_effective_config(
    scheme: Scheme, perf: Mapping[str, Any]
) -> dict[str, Any]:
    if not scheme.executable:
        raise BenchmarkDataError(f"cannot validate unsupported scheme {scheme.scheme_id}")
    debug = perf.get("meta", {}).get("vividvr_debug", {})
    if not isinstance(debug, Mapping):
        raise BenchmarkDataError("meta.vividvr_debug must be an object")

    mismatches: list[str] = []
    requested = debug.get("attention_backend_requested")
    if requested != scheme.backend:
        mismatches.append(
            f"requested backend expected {scheme.backend!r}, observed {requested!r}"
        )
    for component in ("transformer", "controlnet"):
        observed = debug.get(f"attention_backend_{component}")
        if observed != scheme.expected_effective_backend:
            mismatches.append(
                "effective backend for "
                f"{component} expected {scheme.expected_effective_backend!r}, "
                f"observed {observed!r}"
            )
    observed_mode = debug.get("vividvr_parallel_mode")
    if observed_mode != scheme.parallel_mode:
        mismatches.append(
            f"parallel mode expected {scheme.parallel_mode!r}, observed {observed_mode!r}"
        )
    observed_sp = debug.get("sp_world_size")
    if observed_sp != scheme.sp_degree:
        mismatches.append(
            f"SP world size expected {scheme.sp_degree}, observed {observed_sp!r}"
        )
    observed_cfg = debug.get("cfg_parallel_enabled")
    if bool(observed_cfg) is not scheme.cfg_parallel:
        mismatches.append(
            f"CFG parallel expected {scheme.cfg_parallel}, observed {observed_cfg!r}"
        )
    compile_values = [
        debug.get("torch_compile_requested"),
        debug.get("torch_compile_transformer"),
        debug.get("torch_compile_controlnet"),
    ]
    if scheme.compile_enabled and compile_values != [True, True, True]:
        mismatches.append(
            "torch.compile expected requested/applied on transformer and controlnet, "
            f"observed {compile_values!r}"
        )
    if not scheme.compile_enabled and bool(compile_values[0]):
        mismatches.append(
            f"torch.compile expected disabled, observed requested={compile_values[0]!r}"
        )
    fusion_values = [
        debug.get("modulation_fusion_requested"),
        debug.get("modulation_fusion_transformer"),
        debug.get("modulation_fusion_controlnet"),
    ]
    if scheme.modulation_fusion and fusion_values != [True, True, True]:
        mismatches.append(
            "modulation fusion expected requested/applied on transformer and "
            f"controlnet, observed {fusion_values!r}"
        )
    if not scheme.modulation_fusion and bool(fusion_values[0]):
        mismatches.append(
            "modulation fusion expected disabled, "
            f"observed requested={fusion_values[0]!r}"
        )
    if mismatches:
        raise BenchmarkDataError("; ".join(mismatches))
    return {
        "requested_backend": requested,
        "effective_backend": scheme.expected_effective_backend,
        "parallel_mode": observed_mode,
        "sp_world_size": observed_sp,
        "cfg_parallel_enabled": bool(observed_cfg),
        "torch_compile_applied": scheme.compile_enabled,
        "modulation_fusion_applied": scheme.modulation_fusion,
    }


def _successful_seconds(record: Mapping[str, Any] | None) -> float | None:
    if not record or record.get("status") != "succeeded":
        return None
    value = record.get("timings", {}).get("model_inference_runtime_seconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return float(value)


def _quality_passed(record: Mapping[str, Any] | None) -> bool:
    return bool(record and record.get("quality", {}).get("pass_compare") is True)


def compute_derived_metrics(
    scheme: Scheme,
    formal_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    current_seconds = _successful_seconds(formal_records.get(scheme.scheme_id))
    baseline_seconds = _successful_seconds(formal_records.get("R0"))
    control_candidates = [
        control
        for control in scheme.controls
        if _successful_seconds(formal_records.get(control)) is not None
        and _quality_passed(formal_records.get(control))
    ]
    control_id = (
        min(
            control_candidates,
            key=lambda item: _successful_seconds(formal_records[item]) or float("inf"),
        )
        if control_candidates
        else None
    )
    control_seconds = (
        _successful_seconds(formal_records.get(control_id)) if control_id else None
    )
    gpu_seconds = (
        scheme.gpu_count * current_seconds if current_seconds is not None else None
    )
    baseline_gpu_seconds = baseline_seconds
    return {
        "cumulative_speedup_vs_r0": (
            baseline_seconds / current_seconds
            if baseline_seconds is not None and current_seconds is not None
            else None
        ),
        "cumulative_speedup_reason": (
            None
            if baseline_seconds is not None and current_seconds is not None
            else "missing_successful_r0_or_current_formal_record"
        ),
        "control_scheme_id": control_id,
        "incremental_speedup": (
            control_seconds / current_seconds
            if control_seconds is not None and current_seconds is not None
            else None
        ),
        "incremental_speedup_reason": (
            None
            if control_seconds is not None and current_seconds is not None
            else "missing_quality_passing_control_or_current_formal_record"
        ),
        "gpu_seconds": gpu_seconds,
        "resource_efficiency_vs_r0": (
            baseline_gpu_seconds / gpu_seconds
            if baseline_gpu_seconds is not None and gpu_seconds is not None
            else None
        ),
        "resource_efficiency_reason": (
            None
            if baseline_gpu_seconds is not None and gpu_seconds is not None
            else "missing_successful_r0_or_current_formal_record"
        ),
    }


def _scheme_payload(scheme: Scheme) -> dict[str, Any]:
    payload = asdict(scheme)
    payload["status"] = scheme.status.value
    payload["expected_effective_backend"] = scheme.expected_effective_backend
    payload["cfg_parallel"] = scheme.cfg_parallel
    return payload


def build_unsupported_record(
    scheme: Scheme,
    config: BenchmarkConfig,
    *,
    batch_id: str,
) -> dict[str, Any]:
    if scheme.executable:
        raise BenchmarkConfigError(
            f"{scheme.scheme_id} is executable; unsupported record is invalid"
        )
    return {
        "schema_version": 1,
        "batch_id": batch_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": "unsupported",
        "run_role": None,
        "scheme": _scheme_payload(scheme),
        "capability": {
            "status": scheme.status.value,
            "reason": scheme.unsupported_reason,
        },
        "inputs": {
            "input_video": str(config.input_video),
            "caption_file": str(config.caption_file),
            "prompt_source": "caption_sidecar_file_only",
            "reference_video": str(config.reference_video),
            "num_frames": 130,
            "temporal_process_frames": 121,
            "inference_steps": 20,
            "seed": 42,
            "guidance_scale": 6.0,
            "restoration_guidance_scale": -1.0,
            "upscale": 1.0,
            "dtype": "bfloat16",
        },
        "runtime": {
            "requested_backend": scheme.backend,
            "effective_backend": None,
            "effective_backend_reason": "scheme_not_executable",
            "torch_compile_applied": None,
            "parallel_topology": scheme.parallel_mode,
            "fusion": None,
            "cache": None,
            "quantization": None,
        },
        "timings": {
            "total_runtime_seconds": None,
            "model_inference_runtime_seconds": None,
            "denoising_runtime_seconds": None,
            "denoise_fraction": None,
            "stage_seconds": {name: None for name in VIVIDVR_STAGE_NAMES},
            "unclassified_seconds": None,
            "temporal_clip_count": None,
            "inference_step_count": None,
            "mean_step_seconds": None,
            "steady_step_median_seconds": None,
            "sp_communication_seconds": None,
            "sp_communication_reason": "not_profiled",
            "cfg_communication_seconds": None,
            "cfg_communication_reason": "not_profiled",
            "cache_executed_steps": None,
            "cache_skipped_steps": None,
            "cache_steps_reason": "scheme_not_executable",
        },
        "gpu_memory": {
            "device_ids": list(config.gpu_ids[: scheme.gpu_count]),
            "per_gpu_peak_mib": None,
            "max_single_gpu_peak_mib": None,
            "max_single_gpu_peak_gib": None,
            "sampling_backend": None,
            "reason": "scheme_not_executable",
        },
        "quality": {
            "pass_compare": None,
            "ssim_mean": None,
            "ssim_min": None,
            "failed_frame_ratio": None,
            "reason": "scheme_not_executable",
        },
        "derived": {
            "cumulative_speedup_vs_r0": None,
            "incremental_speedup": None,
            "control_scheme_id": None,
            "gpu_seconds": None,
            "resource_efficiency_vs_r0": None,
            "reason": "scheme_not_executable",
        },
        "artifacts": {
            "perf_json": None,
            "result_video": None,
            "compare_json": None,
            "service_log": None,
        },
        "reproducibility": {
            "repo_root": str(config.repo_root),
            "python_executable": str(config.python_executable),
            "model_path": str(config.model_path),
            "vividvr_path": str(config.vividvr_path),
            "service_command": None,
            "service_environment": None,
            "config_fingerprint": None,
        },
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2, sort_keys=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _deep_fill_defaults(
    value: Mapping[str, Any], defaults: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(value)
    for key, default in defaults.items():
        if key not in result:
            result[key] = default
        elif isinstance(result[key], Mapping) and isinstance(default, Mapping):
            result[key] = _deep_fill_defaults(result[key], default)
    return result


def _complete_executable_record(
    record: Mapping[str, Any], scheme: Scheme, config: BenchmarkConfig
) -> dict[str, Any]:
    """Keep failure and fixture records schema-compatible with successful runs."""

    defaults: dict[str, Any] = {
        "capability": {"status": "executable", "reason": None},
        "inputs": {
            "input_video": str(config.input_video),
            "caption_file": str(config.caption_file),
            "prompt_source": "caption_sidecar_file_only",
            "reference_video": str(config.reference_video),
            "num_frames": 130,
            "temporal_process_frames": 121,
            "inference_steps": 20,
            "seed": 42,
            "guidance_scale": 6.0,
            "restoration_guidance_scale": -1.0,
            "upscale": 1.0,
            "dtype": "bfloat16",
        },
        "runtime": {
            "requested_backend": scheme.backend,
            "effective_backend": None,
            "effective_backend_reason": "request_did_not_produce_valid_perf",
            "torch_compile_applied": None,
            "parallel_topology": scheme.parallel_mode,
            "fusion": None,
            "cache": "disabled",
            "quantization": "disabled",
        },
        "timings": {
            "total_runtime_seconds": None,
            "model_inference_runtime_seconds": None,
            "denoising_runtime_seconds": None,
            "denoise_fraction": None,
            "stage_seconds": {name: None for name in VIVIDVR_STAGE_NAMES},
            "unclassified_seconds": None,
            "temporal_clip_count": None,
            "inference_step_count": None,
            "mean_step_seconds": None,
            "steady_step_median_seconds": None,
            "sp_communication_seconds": None,
            "sp_communication_reason": "not_profiled",
            "cfg_communication_seconds": None,
            "cfg_communication_reason": "not_profiled",
            "cache_executed_steps": None,
            "cache_skipped_steps": None,
            "cache_steps_reason": "cache_disabled_or_unsupported",
        },
        "gpu_memory": {
            "device_ids": list(config.gpu_ids[: scheme.gpu_count]),
            "per_gpu_peak_mib": None,
            "max_single_gpu_peak_mib": None,
            "max_single_gpu_peak_gib": None,
            "sampling_backend": None,
            "reason": "request_did_not_produce_complete_memory_metrics",
        },
        "quality": {
            "pass_compare": None,
            "ssim_mean": None,
            "ssim_min": None,
            "failed_frame_ratio": None,
            "reason": "formal_compare_not_completed",
        },
        "derived": {
            "cumulative_speedup_vs_r0": None,
            "incremental_speedup": None,
            "control_scheme_id": None,
            "gpu_seconds": None,
            "resource_efficiency_vs_r0": None,
            "reason": "request_not_successful",
        },
        "artifacts": {
            "perf_json": None,
            "result_video": None,
            "compare_json": None,
            "service_log": None,
        },
        "reproducibility": {
            "repo_root": str(config.repo_root),
            "python_executable": str(config.python_executable),
            "model_path": str(config.model_path),
            "vividvr_path": str(config.vividvr_path),
            "service_command": build_service_command(scheme, config),
            "service_environment": build_service_environment(scheme, config),
            "config_fingerprint": None,
        },
    }
    return _deep_fill_defaults(record, defaults)


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
GpuSample = Mapping[int, float]


def _run_command(
    command: Sequence[str], **kwargs: Any
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), check=False, text=True, **kwargs)


class _NvmlMemoryProvider:
    def __init__(self, gpu_ids: Sequence[int]):
        from sglang.multimodal_gen.utils import import_pynvml

        self._gpu_ids = tuple(gpu_ids)
        self._pynvml = import_pynvml()
        self._pynvml.nvmlInit()

    def __call__(self) -> dict[int, float]:
        result: dict[int, float] = {}
        for gpu_id in self._gpu_ids:
            handle = self._pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            memory = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
            result[gpu_id] = float(memory.used) / (1024.0**2)
        return result

    def close(self) -> None:
        self._pynvml.nvmlShutdown()


def _sample_gpu_memory_with_nvidia_smi(
    gpu_ids: Sequence[int], command_runner: CommandRunner = _run_command
) -> dict[int, float]:
    completed = command_runner(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
    )
    if completed.returncode != 0:
        raise BenchmarkDataError(
            "nvidia-smi memory sampling failed: "
            f"{(completed.stderr or '').strip()}"
        )
    selected = set(gpu_ids)
    result: dict[int, float] = {}
    for line in (completed.stdout or "").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        gpu_id = int(fields[0])
        if gpu_id in selected:
            result[gpu_id] = float(fields[1])
    if set(result) != selected:
        raise BenchmarkDataError(
            f"nvidia-smi did not report selected GPUs: {sorted(selected - set(result))}"
        )
    return result


class GpuMemorySampler:
    def __init__(
        self,
        gpu_ids: Sequence[int],
        *,
        sample_provider: Iterator[GpuSample] | Callable[[], GpuSample] | None = None,
        sampling_backend: str | None = None,
        interval_seconds: float = 0.25,
        command_runner: CommandRunner = _run_command,
    ):
        self.gpu_ids = tuple(gpu_ids)
        self.interval_seconds = interval_seconds
        self._iterator = (
            sample_provider
            if sample_provider is not None and hasattr(sample_provider, "__next__")
            else None
        )
        self._callable = (
            sample_provider
            if callable(sample_provider) and self._iterator is None
            else None
        )
        self._nvml_provider: _NvmlMemoryProvider | None = None
        if sample_provider is None:
            try:
                self._nvml_provider = _NvmlMemoryProvider(self.gpu_ids)
                self._callable = self._nvml_provider
                default_backend = "nvml"
            except Exception:
                self._callable = lambda: _sample_gpu_memory_with_nvidia_smi(
                    self.gpu_ids, command_runner
                )
                default_backend = "nvidia-smi"
        else:
            default_backend = "injected"
        self.sampling_backend = sampling_backend or default_backend
        self._peaks = {gpu_id: 0.0 for gpu_id in self.gpu_ids}
        self._sample_count = 0
        self._errors: list[str] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def sample_once(self) -> bool:
        try:
            if self._iterator is not None:
                sample = next(self._iterator)
            elif self._callable is not None:
                sample = self._callable()
            else:
                raise BenchmarkDataError("GPU sampler has no sample provider")
        except StopIteration:
            return False
        except Exception as error:
            self._errors.append(f"{type(error).__name__}: {error}")
            return False
        missing = set(self.gpu_ids) - set(sample)
        if missing:
            self._errors.append(f"sample missing GPU IDs {sorted(missing)}")
            return False
        for gpu_id in self.gpu_ids:
            self._peaks[gpu_id] = max(self._peaks[gpu_id], float(sample[gpu_id]))
        self._sample_count += 1
        return True

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            self.sample_once()
            self._stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        if self._thread is not None:
            raise BenchmarkConfigError("GPU memory sampler is already running")
        self.sample_once()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="vividvr-gpu-memory-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 4))
            self._thread = None
        self.sample_once()
        if self._nvml_provider is not None:
            try:
                self._nvml_provider.close()
            finally:
                self._nvml_provider = None
        return self.result()

    def result(self) -> dict[str, Any]:
        max_peak = max(self._peaks.values(), default=0.0)
        return {
            "device_ids": list(self.gpu_ids),
            "per_gpu_peak_mib": {
                str(gpu_id): peak for gpu_id, peak in self._peaks.items()
            },
            "max_single_gpu_peak_mib": max_peak,
            "max_single_gpu_peak_gib": max_peak / 1024.0,
            "sample_count": self._sample_count,
            "sampling_backend": self.sampling_backend,
            "sampling_errors": list(self._errors),
        }


class TmuxManager:
    _SESSION_PATTERN = re.compile(r"^vividvr_accel_[A-Za-z0-9_-]+$")

    def __init__(
        self,
        *,
        batch_id: str,
        ownership_dir: Path,
        command_runner: CommandRunner = _run_command,
    ):
        self.batch_id = batch_id
        self.ownership_dir = ownership_dir
        self.command_runner = command_runner

    def _owner_path(self, session: str) -> Path:
        if not self._SESSION_PATTERN.fullmatch(session):
            raise BenchmarkConfigError(f"unsafe tmux session name: {session!r}")
        return self.ownership_dir / f"{session}.json"

    def _read_owner(self, session: str) -> dict[str, Any] | None:
        owner_path = self._owner_path(session)
        try:
            value = json.loads(owner_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as error:
            raise BenchmarkConfigError(
                f"cannot read tmux ownership file {owner_path}: {error}"
            ) from error
        return value if isinstance(value, dict) else None

    def start(
        self,
        session: str,
        command: Sequence[str],
        log_path: Path,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        owner_path = self._owner_path(session)
        self.ownership_dir.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        ownership = {
            "batch_id": self.batch_id,
            "session": session,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with owner_path.open("x", encoding="utf-8") as output:
                json.dump(ownership, output, ensure_ascii=False)
                output.write("\n")
        except FileExistsError as error:
            raise BenchmarkConfigError(
                f"tmux ownership file already exists: {owner_path}"
            ) from error

        environment_command = ["env"]
        environment_command.extend(
            f"{key}={value}" for key, value in sorted((environment or {}).items())
        )
        environment_command.extend(command)
        shell_command = (
            f"{shlex.join(environment_command)} 2>&1 | tee {shlex.quote(str(log_path))}"
        )
        completed = self.command_runner(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                session,
                "bash",
                "-lc",
                shell_command,
            ],
            capture_output=True,
        )
        if completed.returncode != 0:
            owner_path.unlink(missing_ok=True)
            raise BenchmarkError(
                f"failed to start tmux session {session}: "
                f"{(completed.stderr or '').strip()}"
            )

    def stop(self, session: str) -> None:
        owner = self._read_owner(session)
        if not owner or owner.get("batch_id") != self.batch_id:
            return
        completed = self.command_runner(
            ["tmux", "kill-session", "-t", session], capture_output=True
        )
        stderr = (completed.stderr or "").lower()
        if completed.returncode != 0 and "can't find session" not in stderr:
            raise BenchmarkError(
                f"failed to stop owned tmux session {session}: "
                f"{(completed.stderr or '').strip()}"
            )
        self._owner_path(session).unlink(missing_ok=True)

    def is_running(self, session: str) -> bool:
        completed = self.command_runner(
            ["tmux", "has-session", "-t", session], capture_output=True
        )
        return completed.returncode == 0

    def cleanup_owned(self) -> None:
        if not self.ownership_dir.exists():
            return
        for owner_path in sorted(self.ownership_dir.glob("vividvr_accel_*.json")):
            session = owner_path.stem
            owner = self._read_owner(session)
            if owner and owner.get("batch_id") == self.batch_id:
                self.stop(session)


def _port_is_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((host, port))
        except OSError:
            return False
    return True


def _gpu_processes(
    gpu_ids: Sequence[int], command_runner: CommandRunner = _run_command
) -> dict[int, list[dict[str, Any]]]:
    gpu_query = command_runner(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
    )
    if gpu_query.returncode != 0:
        raise BenchmarkConfigError(
            f"nvidia-smi GPU query failed: {(gpu_query.stderr or '').strip()}"
        )
    uuid_to_index: dict[str, int] = {}
    for line in (gpu_query.stdout or "").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2:
            uuid_to_index[fields[1]] = int(fields[0])
    missing = set(gpu_ids) - set(uuid_to_index.values())
    if missing:
        raise BenchmarkConfigError(f"configured GPU IDs do not exist: {sorted(missing)}")

    process_query = command_runner(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
    )
    if process_query.returncode != 0:
        raise BenchmarkConfigError(
            "nvidia-smi compute process query failed: "
            f"{(process_query.stderr or '').strip()}"
        )
    selected = set(gpu_ids)
    result: dict[int, list[dict[str, Any]]] = {}
    for line in (process_query.stdout or "").splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=2)]
        if len(fields) != 3 or fields[0] not in uuid_to_index:
            continue
        gpu_id = uuid_to_index[fields[0]]
        if gpu_id in selected:
            result.setdefault(gpu_id, []).append(
                {"pid": int(fields[1]), "process_name": fields[2]}
            )
    return result


def _gpu_utilization(
    gpu_ids: Sequence[int], command_runner: CommandRunner = _run_command
) -> dict[int, float]:
    utilization_query = command_runner(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
    )
    if utilization_query.returncode != 0:
        raise BenchmarkConfigError(
            "nvidia-smi GPU utilization query failed: "
            f"{(utilization_query.stderr or '').strip()}"
        )
    selected = set(gpu_ids)
    result: dict[int, float] = {}
    for line in (utilization_query.stdout or "").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            gpu_id = int(fields[0])
            utilization = float(fields[1])
        except ValueError:
            continue
        if gpu_id in selected:
            result[gpu_id] = utilization
    missing = selected - set(result)
    if missing:
        raise BenchmarkConfigError(
            f"missing utilization data for configured GPU IDs: {sorted(missing)}"
        )
    return result


def run_preflight(
    config: BenchmarkConfig,
    *,
    check_runtime_resources: bool,
    command_runner: CommandRunner = _run_command,
    which: Callable[[str], str | None] = shutil.which,
    port_checker: Callable[[str, int], bool] = _port_is_available,
    gpu_process_checker: Callable[
        [Sequence[int]], Mapping[int, Sequence[Mapping[str, Any]]]
    ]
    | None = None,
    gpu_utilization_checker: Callable[
        [Sequence[int]], Mapping[int, float]
    ]
    | None = None,
) -> dict[str, Any]:
    required_paths = {
        "python_executable": config.python_executable,
        "sglang_executable": config.python_executable.parent / "sglang",
        "model_path": config.model_path,
        "vividvr_path": config.vividvr_path,
        "input_video": config.input_video,
        "caption_file": config.caption_file,
        "reference_video": config.reference_video,
    }
    missing_paths = [name for name, path in required_paths.items() if not path.exists()]
    if missing_paths:
        raise BenchmarkConfigError(
            "missing required paths: "
            + ", ".join(f"{name}={required_paths[name]}" for name in missing_paths)
        )
    required_binaries = ("tmux", "ffmpeg")
    missing_binaries = [name for name in required_binaries if which(name) is None]
    moto_server = config.python_executable.parent / "moto_server"
    if not moto_server.exists():
        missing_binaries.append(str(moto_server))
    if missing_binaries:
        raise BenchmarkConfigError(
            f"missing required executables: {', '.join(missing_binaries)}"
        )
    result: dict[str, Any] = {
        "ok": True,
        "required_paths": {name: str(path) for name, path in required_paths.items()},
        "gpu_ids": list(config.gpu_ids),
        "runtime_resources_checked": check_runtime_resources,
    }
    if not check_runtime_resources:
        return result

    ports = (
        config.service_port,
        config.master_port,
        config.scheduler_port,
        config.caption_port,
        config.callback_port,
        config.s3_port,
    )
    occupied_ports = [port for port in ports if not port_checker(config.host, port)]
    if occupied_ports:
        raise BenchmarkConfigError(f"occupied ports: {occupied_ports}")
    checker = gpu_process_checker or (
        lambda gpu_ids: _gpu_processes(gpu_ids, command_runner)
    )
    gpu_processes = {
        gpu_id: list(processes)
        for gpu_id, processes in checker(config.gpu_ids).items()
        if processes
    }
    result["gpu_process_policy"] = (
        "allow_existing_when_idle"
        if config.allow_idle_gpu_processes
        else "require_no_existing_processes"
    )
    if gpu_processes and not config.allow_idle_gpu_processes:
        details = "; ".join(
            f"GPU {gpu_id}: "
            + ", ".join(
                f"pid {process.get('pid')} ({process.get('process_name')})"
                for process in processes
            )
            for gpu_id, processes in sorted(gpu_processes.items())
            if processes
        )
        if details:
            raise BenchmarkConfigError(f"selected GPUs have active processes: {details}")
    if gpu_processes:
        utilization_checker = gpu_utilization_checker or (
            lambda gpu_ids: _gpu_utilization(gpu_ids, command_runner)
        )
        gpu_utilization = dict(utilization_checker(config.gpu_ids))
        missing = set(config.gpu_ids) - set(gpu_utilization)
        if missing:
            raise BenchmarkConfigError(
                f"missing utilization data for configured GPU IDs: {sorted(missing)}"
            )
        active = {
            gpu_id: utilization
            for gpu_id, utilization in gpu_utilization.items()
            if gpu_id in config.gpu_ids and utilization != 0
        }
        if active:
            details = "; ".join(
                f"GPU {gpu_id} utilization is {utilization:g}%"
                for gpu_id, utilization in sorted(active.items())
            )
            raise BenchmarkConfigError(
                "selected GPUs with existing processes must be idle: " + details
            )
        result["gpu_utilization_percent"] = gpu_utilization
    result["ports"] = list(ports)
    result["gpu_processes"] = gpu_processes
    return result


def _path_fingerprint(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists():
        stat = path.stat()
        result.update(
            {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "is_dir": path.is_dir(),
            }
        )
    return result


def compute_config_fingerprint(config: BenchmarkConfig, scheme: Scheme) -> str:
    config_payload = asdict(config)
    normalized_config = {
        key: (
            str(value)
            if isinstance(value, Path)
            else list(value)
            if isinstance(value, tuple)
            else value
        )
        for key, value in config_payload.items()
    }
    payload = {
        "schema_version": 1,
        "scheme": _scheme_payload(scheme),
        "config": normalized_config,
        "input_metadata": {
            "input_video": _path_fingerprint(config.input_video),
            "caption_file": _path_fingerprint(config.caption_file),
            "reference_video": _path_fingerprint(config.reference_video),
            "model_path": _path_fingerprint(config.model_path),
            "vividvr_path": _path_fingerprint(config.vividvr_path),
        },
    }
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _ensure_output_path(config: BenchmarkConfig, path: Path) -> None:
    try:
        path.resolve().relative_to(config.output_root.resolve())
    except ValueError as error:
        raise BenchmarkConfigError(
            f"benchmark artifact must be below output_root={config.output_root}: {path}"
        ) from error


def build_request_payload(
    config: BenchmarkConfig,
    *,
    task_id: str,
    callback_url: str,
    output_path: Path,
    perf_path: Path,
) -> dict[str, Any]:
    """Build the fixed 130-frame/20-step FlowCut request.

    The caption path is deliberately absent: the service must obtain it from the
    fixed caption-sidecar mock, preserving the production caption-bridge contract.
    """

    _ensure_output_path(config, output_path)
    _ensure_output_path(config, perf_path)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", task_id):
        raise BenchmarkConfigError(f"unsafe task ID: {task_id!r}")
    return {
        "taskId": task_id,
        "timeout": -1,
        "callbackUrl": callback_url,
        "video_input_path": str(config.input_video),
        "num_inference_steps": 20,
        "seed": 42,
        "num_temporal_process_frames": 121,
        "guidance_scale": 6.0,
        "restoration_guidance_scale": -1.0,
        "upscale": 1.0,
        "output_path": str(output_path),
        "outputObjectKey": f"acceleration-benchmark/{task_id}",
        "perf_dump_path": str(perf_path),
        "minioConfig": {
            "endpoint": f"{config.host}:{config.s3_port}",
            "bucket_name": config.s3_bucket,
            "access_key": "test",
            "secret_key": "test",
            "secure": False,
            "region": "us-east-1",
        },
    }


class BenchmarkRunner:
    """Coordinate sequential schemes while keeping I/O boundaries injectable."""

    def __init__(
        self,
        *,
        config: BenchmarkConfig,
        batch_id: str,
        lifecycle: Any,
        request_executor: Callable[..., Mapping[str, Any]],
        resume: bool = False,
    ):
        self.config = config
        self.batch_id = batch_id
        self.lifecycle = lifecycle
        self.request_executor = request_executor
        self.resume = resume
        self.batch_dir = config.output_root / batch_id
        self.records_dir = self.batch_dir / "records"
        self.summary_path = self.batch_dir / "batch_summary.json"

    def _record_path(self, scheme: Scheme, role: RunRole | None) -> Path:
        suffix = role.value if role is not None else "unsupported"
        return self.records_dir / f"{scheme.scheme_id}_{suffix}.json"

    def _read_record(self, scheme: Scheme, role: RunRole) -> dict[str, Any] | None:
        try:
            value = json.loads(
                self._record_path(scheme, role).read_text(encoding="utf-8")
            )
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _stamp_record(
        self,
        record: Mapping[str, Any],
        scheme: Scheme,
        role: RunRole,
        fingerprint: str,
    ) -> dict[str, Any]:
        stamped = _complete_executable_record(record, scheme, self.config)
        stamped.setdefault("schema_version", 1)
        stamped["batch_id"] = self.batch_id
        stamped["recorded_at"] = datetime.now(timezone.utc).isoformat()
        stamped["run_role"] = role.value
        stamped["scheme"] = _scheme_payload(scheme)
        reproducibility = dict(stamped.get("reproducibility") or {})
        reproducibility["config_fingerprint"] = fingerprint
        stamped["reproducibility"] = reproducibility
        return stamped

    def _execute_request(
        self, scheme: Scheme, role: RunRole, fingerprint: str
    ) -> dict[str, Any]:
        try:
            raw_record = self.request_executor(
                scheme,
                role,
                batch_id=self.batch_id,
                fingerprint=fingerprint,
            )
            if not isinstance(raw_record, Mapping):
                raise BenchmarkDataError("request executor did not return an object")
            record = self._stamp_record(raw_record, scheme, role, fingerprint)
        except Exception as error:
            record = self._stamp_record(
                {
                    "status": "failed",
                    "failure": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                },
                scheme,
                role,
                fingerprint,
            )
        atomic_write_json(self._record_path(scheme, role), record)
        return record

    def _can_resume(self, scheme: Scheme, fingerprint: str) -> bool:
        if not self.resume:
            return False
        formal = self._read_record(scheme, RunRole.FORMAL)
        return bool(
            formal
            and formal.get("status") == "succeeded"
            and formal.get("reproducibility", {}).get("config_fingerprint")
            == fingerprint
        )

    def _write_summary(self, summary: Mapping[str, Any]) -> None:
        atomic_write_json(self.summary_path, summary)

    def run(self, schemes: Sequence[Scheme]) -> dict[str, Any]:
        self.records_dir.mkdir(parents=True, exist_ok=True)
        summary: dict[str, Any] = {
            "schema_version": 1,
            "batch_id": self.batch_id,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "schemes": {},
        }
        self._write_summary(summary)
        formal_records: dict[str, Mapping[str, Any]] = {}
        cleanup_error: BenchmarkCleanupError | None = None
        try:
            self.lifecycle.start_shared()
            for scheme in schemes:
                fingerprint = compute_config_fingerprint(self.config, scheme)
                if not scheme.executable:
                    record = build_unsupported_record(
                        scheme, self.config, batch_id=self.batch_id
                    )
                    record["reproducibility"]["config_fingerprint"] = fingerprint
                    record_path = self._record_path(scheme, None)
                    atomic_write_json(record_path, record)
                    summary["schemes"][scheme.scheme_id] = {
                        "status": "unsupported",
                        "record": str(record_path),
                    }
                    self._write_summary(summary)
                    continue

                if self._can_resume(scheme, fingerprint):
                    formal = self._read_record(scheme, RunRole.FORMAL) or {}
                    formal_records[scheme.scheme_id] = formal
                    summary["schemes"][scheme.scheme_id] = {
                        "status": "resumed",
                        "formal_record": str(
                            self._record_path(scheme, RunRole.FORMAL)
                        ),
                    }
                    self._write_summary(summary)
                    continue

                try:
                    self.lifecycle.start_scheme(scheme)
                except Exception as error:
                    try:
                        self.lifecycle.stop_scheme(scheme)
                    except Exception as cleanup_failure:
                        raise BenchmarkCleanupError(
                            "failed to clean up partially started service for "
                            f"{scheme.scheme_id}: {cleanup_failure}"
                        ) from error
                    failed = self._stamp_record(
                        {
                            "status": "failed",
                            "failure": {
                                "type": type(error).__name__,
                                "message": str(error),
                                "phase": "service_start",
                            },
                        },
                        scheme,
                        RunRole.FORMAL,
                        fingerprint,
                    )
                    atomic_write_json(
                        self._record_path(scheme, RunRole.FORMAL), failed
                    )
                    summary["schemes"][scheme.scheme_id] = {
                        "status": "failed",
                        "failure_phase": "service_start",
                    }
                    self._write_summary(summary)
                    continue

                skip_formal = False
                try:
                    if scheme.compile_enabled:
                        warmup = self._execute_request(
                            scheme, RunRole.WARMUP, fingerprint
                        )
                        if warmup.get("status") != "succeeded":
                            summary["schemes"][scheme.scheme_id] = {
                                "status": "failed",
                                "failure_phase": "warmup",
                                "warmup_record": str(
                                    self._record_path(scheme, RunRole.WARMUP)
                                ),
                                "formal_status": "skipped_after_warmup_failure",
                            }
                            self._write_summary(summary)
                            skip_formal = True
                    if not skip_formal:
                        formal = self._execute_request(
                            scheme, RunRole.FORMAL, fingerprint
                        )
                        formal_records[scheme.scheme_id] = formal
                        scheme_status = str(formal.get("status", "failed"))
                        if scheme_status == "succeeded":
                            derived = compute_derived_metrics(scheme, formal_records)
                            formal["derived"] = derived
                            atomic_write_json(
                                self._record_path(scheme, RunRole.FORMAL), formal
                            )
                        summary["schemes"][scheme.scheme_id] = {
                            "status": scheme_status,
                            "warmup_record": (
                                str(self._record_path(scheme, RunRole.WARMUP))
                                if scheme.compile_enabled
                                else None
                            ),
                            "formal_record": str(
                                self._record_path(scheme, RunRole.FORMAL)
                            ),
                        }
                        self._write_summary(summary)
                finally:
                    try:
                        self.lifecycle.stop_scheme(scheme)
                    except Exception as error:
                        cleanup_error = BenchmarkCleanupError(
                            f"failed to clean up service for {scheme.scheme_id}: {error}"
                        )
                if cleanup_error is not None:
                    raise cleanup_error
        except Exception as error:
            if cleanup_error is None and isinstance(error, BenchmarkCleanupError):
                cleanup_error = error
            if cleanup_error is None:
                summary["status"] = "aborted"
                summary["failure"] = f"{type(error).__name__}: {error}"
                summary["finished_at"] = datetime.now(timezone.utc).isoformat()
                self._write_summary(summary)
            raise
        finally:
            try:
                self.lifecycle.stop_shared()
            except Exception as error:
                shared_error = BenchmarkCleanupError(
                    f"failed to clean up shared benchmark services: {error}"
                )
                if cleanup_error is None:
                    cleanup_error = shared_error
            if cleanup_error is not None:
                summary["status"] = "cleanup_failed"
                summary["failure"] = str(cleanup_error)
                summary["finished_at"] = datetime.now(timezone.utc).isoformat()
                self._write_summary(summary)
        if cleanup_error is not None:
            raise cleanup_error
        statuses = {value["status"] for value in summary["schemes"].values()}
        successful_statuses = {"succeeded", "resumed", "unsupported"}
        summary["status"] = (
            "completed_with_failures"
            if not statuses.issubset(successful_statuses)
            else "completed"
        )
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        self._write_summary(summary)
        return summary


def _merge_environment_paths(
    preferred: Sequence[Path], existing: str | None
) -> str:
    values = [str(path) for path in preferred]
    if existing:
        values.extend(value for value in existing.split(os.pathsep) if value)
    return os.pathsep.join(dict.fromkeys(values))


def _resolve_python_dev_include_paths() -> tuple[Path, ...]:
    version_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    compact_version = f"{sys.version_info.major}{sys.version_info.minor}"
    major_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    home = Path.home()
    candidates: list[Path] = []

    override = os.environ.get("SGLANG_PYTHON_DEV_INCLUDE")
    if override:
        candidates.append(Path(override).expanduser())
    for value in (
        sysconfig.get_config_var("INCLUDEPY"),
        sysconfig.get_path("include"),
        sysconfig.get_path("platinclude"),
    ):
        if value:
            candidates.append(Path(value))
    candidates.extend(
        (
            Path(sys.prefix) / "include" / version_tag,
            Path(sys.base_prefix) / "include" / version_tag,
            home
            / f"tmp_py{compact_version}dev"
            / "extracted"
            / "usr"
            / "include"
            / version_tag,
            home
            / f"tmp_py{compact_version}_headers"
            / "extracted"
            / f"libpython{major_minor}-dev"
            / "usr"
            / "include"
            / version_tag,
        )
    )

    for candidate in dict.fromkeys(candidates):
        if not (candidate / "Python.h").is_file():
            continue
        include_paths = [candidate]
        multiarch = sysconfig.get_config_var("MULTIARCH")
        if multiarch:
            multiarch_config = (
                candidate.parent / multiarch / candidate.name / "pyconfig.h"
            )
            if multiarch_config.is_file():
                include_paths.insert(0, candidate.parent)
        return tuple(include_paths)
    return ()


def build_service_environment(
    scheme: Scheme, config: BenchmarkConfig
) -> dict[str, str]:
    if not scheme.executable:
        raise BenchmarkConfigError(
            f"{scheme.scheme_id} is unsupported: {scheme.unsupported_reason}"
        )
    if len(config.gpu_ids) < scheme.gpu_count:
        raise BenchmarkConfigError(
            f"{scheme.scheme_id} requires {scheme.gpu_count} GPUs, but only "
            f"{len(config.gpu_ids)} GPU IDs were configured"
        )
    environment = {
        "CUDA_VISIBLE_DEVICES": ",".join(
            str(gpu) for gpu in config.gpu_ids[: scheme.gpu_count]
        ),
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(config.repo_root / "python"),
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "AWS_EC2_METADATA_DISABLED": "true",
        "SGLANG_FLOWCUT_PROGRESS_INTERVAL_SECONDS": "5",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    if scheme.sp_degree > 1:
        environment["SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE"] = "eager_global"
    if scheme.compile_enabled:
        python_include_paths = _resolve_python_dev_include_paths()
        if not python_include_paths:
            raise BenchmarkConfigError(
                "torch.compile requires Python development headers; set "
                "SGLANG_PYTHON_DEV_INCLUDE to a directory containing Python.h"
            )
        environment["CPATH"] = _merge_environment_paths(
            python_include_paths, os.environ.get("CPATH")
        )
        environment["C_INCLUDE_PATH"] = _merge_environment_paths(
            python_include_paths, os.environ.get("C_INCLUDE_PATH")
        )
    return environment


def build_service_command(scheme: Scheme, config: BenchmarkConfig) -> list[str]:
    if not scheme.executable:
        raise BenchmarkConfigError(
            f"{scheme.scheme_id} is unsupported: {scheme.unsupported_reason}"
        )
    build_service_environment(scheme, config)
    command = [
        str(config.python_executable.parent / "sglang"),
        "serve",
        "--model-path",
        str(config.model_path),
        "--model-id",
        "VividVR",
        "--pipeline-class-name",
        "CogVideoXVividVRControlNetPipeline",
        "--component-paths.vividvr",
        str(config.vividvr_path),
        "--num-gpus",
        str(scheme.gpu_count),
        "--tp-size",
        "1",
        "--sp-degree",
        str(scheme.sp_degree),
        "--ulysses-degree",
        str(scheme.sp_degree),
        "--ring-degree",
        "1",
        "--vividvr-parallel-mode",
        scheme.parallel_mode,
        "--dist-timeout",
        str(config.dist_timeout_seconds),
        "--attention-backend",
        scheme.backend,
        "--host",
        config.host,
        "--port",
        str(config.service_port),
        "--master-port",
        str(config.master_port),
        "--scheduler-port",
        str(config.scheduler_port),
        "--strict-ports",
        "--input-save-path",
        "",
        "--output-path",
        str(config.output_root / "service_outputs"),
        "--vividvr-caption-bridge",
        "--vividvr-caption-sidecar-url",
        f"http://{config.host}:{config.caption_port}",
        "--vividvr-caption-work-dir",
        str(config.output_root / "captions"),
        "--vividvr-caption-sidecar-timeout",
        str(config.caption_timeout_seconds),
    ]
    if scheme.cfg_parallel:
        command.append("--enable-cfg-parallel")
    if scheme.compile_enabled:
        command.append("--enable-torch-compile")
    if scheme.modulation_fusion:
        command.extend(
            [
                "--enable-cogvideox-modulation-fusion",
                "--cogvideox-modulation-fusion-targets",
                "transformer,controlnet",
            ]
        )
    return command


def _wait_for_http(
    url: str,
    *,
    timeout_seconds: float,
    process_alive: Callable[[], bool] | None = None,
) -> None:
    import httpx

    deadline = time.monotonic() + timeout_seconds
    last_error = "no response"
    with httpx.Client(follow_redirects=True, trust_env=False) as client:
        while time.monotonic() < deadline:
            if process_alive is not None and not process_alive():
                raise BenchmarkError(f"service process exited while waiting for {url}")
            try:
                response = client.get(url, timeout=5.0)
                if response.is_success:
                    return
                last_error = f"HTTP {response.status_code}"
            except Exception as error:
                last_error = f"{type(error).__name__}: {error}"
            time.sleep(1.0)
    raise BenchmarkError(
        f"timed out after {timeout_seconds}s waiting for {url}: {last_error}"
    )


def _wait_for_port(host: str, port: int, *, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return
        except OSError:
            time.sleep(0.5)
    raise BenchmarkError(f"timed out waiting for {host}:{port}")


def _safe_batch_token(batch_id: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]", "_", batch_id)
    if not token or len(token) > 80:
        raise BenchmarkConfigError(f"invalid batch ID: {batch_id!r}")
    return token


class TmuxBenchmarkLifecycle:
    """Own shared mock dependencies and one scheme service at a time."""

    def __init__(self, config: BenchmarkConfig, batch_id: str):
        self.config = config
        self.batch_id = batch_id
        self.batch_token = _safe_batch_token(batch_id)
        self.batch_dir = config.output_root / batch_id
        self.logs_dir = self.batch_dir / "logs"
        self.manager = TmuxManager(
            batch_id=batch_id,
            ownership_dir=self.batch_dir / "tmux_ownership",
        )
        prefix = f"vividvr_accel_{self.batch_token}"
        self.shared_sessions = {
            "moto": f"{prefix}_moto",
            "callback": f"{prefix}_callback",
            "caption": f"{prefix}_caption",
        }
        self.scheme_session: str | None = None

    @property
    def script_path(self) -> Path:
        return Path(__file__).resolve()

    def _start_shared_session(
        self, name: str, command: Sequence[str], log_name: str
    ) -> None:
        self.manager.start(
            self.shared_sessions[name], command, self.logs_dir / log_name
        )

    def start_shared(self) -> None:
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self._start_shared_session(
            "moto",
            [
                str(self.config.python_executable.parent / "moto_server"),
                "-H",
                self.config.host,
                "-p",
                str(self.config.s3_port),
            ],
            "moto.log",
        )
        _wait_for_port(self.config.host, self.config.s3_port, timeout_seconds=60)
        self._create_s3_bucket()

        self._start_shared_session(
            "callback",
            [
                str(self.config.python_executable),
                str(self.script_path),
                "_serve-callback",
                "--host",
                self.config.host,
                "--port",
                str(self.config.callback_port),
                "--callback-log",
                str(self.logs_dir / "callbacks.jsonl"),
            ],
            "callback.log",
        )
        _wait_for_http(
            f"http://{self.config.host}:{self.config.callback_port}/health",
            timeout_seconds=60,
            process_alive=lambda: self.manager.is_running(
                self.shared_sessions["callback"]
            ),
        )

        self._start_shared_session(
            "caption",
            [
                str(self.config.python_executable),
                str(self.script_path),
                "_serve-caption",
                "--host",
                self.config.host,
                "--port",
                str(self.config.caption_port),
                "--caption-file",
                str(self.config.caption_file),
            ],
            "caption.log",
        )
        _wait_for_http(
            f"http://{self.config.host}:{self.config.caption_port}/health",
            timeout_seconds=60,
            process_alive=lambda: self.manager.is_running(
                self.shared_sessions["caption"]
            ),
        )

    def _create_s3_bucket(self) -> None:
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            endpoint_url=f"http://{self.config.host}:{self.config.s3_port}",
            aws_access_key_id="test",
            aws_secret_access_key="test",
            region_name="us-east-1",
            config=Config(proxies={}),
        )
        client.create_bucket(Bucket=self.config.s3_bucket)

    def start_scheme(self, scheme: Scheme) -> None:
        if self.scheme_session is not None:
            raise BenchmarkCleanupError(
                f"previous scheme session is still owned: {self.scheme_session}"
            )
        session = f"vividvr_accel_{self.batch_token}_{scheme.scheme_id}_service"
        log_path = self.logs_dir / f"{scheme.scheme_id}_service.log"
        self.manager.start(
            session,
            build_service_command(scheme, self.config),
            log_path,
            environment=build_service_environment(scheme, self.config),
        )
        self.scheme_session = session
        try:
            _wait_for_http(
                f"http://{self.config.host}:{self.config.service_port}/health",
                timeout_seconds=self.config.service_start_timeout_seconds,
                process_alive=lambda: self.manager.is_running(session),
            )
        except Exception:
            try:
                self.manager.stop(session)
            finally:
                self.scheme_session = None
            raise

    def stop_scheme(self, scheme: Scheme) -> None:
        if self.scheme_session is None:
            return
        session = self.scheme_session
        self.manager.stop(session)
        self.scheme_session = None

    def stop_shared(self) -> None:
        errors: list[str] = []
        if self.scheme_session is not None:
            try:
                self.manager.stop(self.scheme_session)
                self.scheme_session = None
            except Exception as error:
                errors.append(str(error))
        for name in ("caption", "callback", "moto"):
            try:
                self.manager.stop(self.shared_sessions[name])
            except Exception as error:
                errors.append(f"{name}: {error}")
        if errors:
            raise BenchmarkCleanupError("; ".join(errors))

    def cleanup_owned(self) -> None:
        """Remove only sessions carrying this batch's ownership marker."""

        self.manager.cleanup_owned()


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkDataError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise BenchmarkDataError(f"{description} must contain a JSON object: {path}")
    return value


def _download_result(url: str, destination: Path) -> None:
    import boto3
    from botocore.config import Config

    parsed = urlsplit(url)
    object_path = unquote(parsed.path).lstrip("/")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise BenchmarkDataError(f"invalid S3 result URL: {url!r}")
    if "/" not in object_path:
        raise BenchmarkDataError(
            f"S3 result URL must contain bucket and object key: {url!r}"
        )
    bucket, object_key = object_path.split("/", 1)
    if not bucket or not object_key:
        raise BenchmarkDataError(
            f"S3 result URL must contain bucket and object key: {url!r}"
        )

    client = boto3.client(
        "s3",
        endpoint_url=f"{parsed.scheme}://{parsed.netloc}",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
        config=Config(proxies={}, s3={"addressing_style": "path"}),
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        client.download_file(bucket, object_key, str(temporary))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def run_compare(
    config: BenchmarkConfig, candidate: Path, report_path: Path
) -> dict[str, Any]:
    command = [
        str(config.python_executable),
        str(
            config.repo_root
            / "python/sglang/multimodal_gen/runtime/videoedit/compare.py"
        ),
        "--reference",
        str(config.reference_video),
        "--candidate",
        str(candidate),
        "--report-json",
        str(report_path),
        "--min-ssim",
        "0.98",
        "--max-failed-frame-ratio",
        "0",
    ]
    completed = _run_command(command, capture_output=True)
    if completed.returncode not in (0, 1) or not report_path.exists():
        raise BenchmarkDataError(
            "video comparison failed without a valid report: "
            f"exit={completed.returncode}, stderr={(completed.stderr or '').strip()}"
        )
    return _load_json_object(report_path, "compare report")


class FlowCutRequestExecutor:
    """Execute one benchmark request against the currently running service."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def __call__(
        self,
        scheme: Scheme,
        role: RunRole,
        *,
        batch_id: str,
        fingerprint: str,
    ) -> dict[str, Any]:
        import httpx

        from sglang.multimodal_gen.tools.run_flowcut_vividvr_service_acceptance import (
            poll_accepted_task,
            submit_flowcut_task_with_retry,
        )

        task_id = _safe_batch_token(f"{batch_id}-{scheme.scheme_id}-{role.value}")
        request_dir = self.config.output_root / batch_id / "requests" / task_id
        request_dir.mkdir(parents=True, exist_ok=True)
        output_path = request_dir / "service-output.mp4"
        downloaded_path = request_dir / "downloaded.mp4"
        perf_path = request_dir / "perf.json"
        compare_path = request_dir / "compare.json"
        callback_url = (
            f"http://{self.config.host}:{self.config.callback_port}/tasks/"
            f"{task_id}/callback"
        )
        payload = build_request_payload(
            self.config,
            task_id=task_id,
            callback_url=callback_url,
            output_path=output_path,
            perf_path=perf_path,
        )
        sampler = GpuMemorySampler(
            self.config.gpu_ids[: scheme.gpu_count], interval_seconds=0.25
        )
        started = time.monotonic()
        sampler.start()
        try:
            with httpx.Client(follow_redirects=True, trust_env=False) as client:
                submit_flowcut_task_with_retry(
                    client=client,
                    base_url=f"http://{self.config.host}:{self.config.service_port}",
                    payload=payload,
                    submit_timeout_s=1800.0,
                    retry_interval_seconds=30.0,
                    max_submit_attempts=60,
                )
                max_polls = max(
                    1,
                    int(
                        self.config.request_timeout_seconds
                        / self.config.poll_interval_seconds
                    ),
                )
                progress = poll_accepted_task(
                    client=client,
                    base_url=f"http://{self.config.host}:{self.config.service_port}",
                    task_id=task_id,
                    poll_interval_seconds=self.config.poll_interval_seconds,
                    max_polls=max_polls,
                )
                total_runtime_seconds = time.monotonic() - started
                if progress.get("status") != "completed":
                    raise BenchmarkError(
                        f"task {task_id} ended with status={progress.get('status')}: "
                        f"{progress}"
                    )
                detail_response = client.get(
                    f"http://{self.config.host}:{self.config.service_port}/v1/videos/"
                    f"repairs/flowcut/{task_id}",
                    timeout=60.0,
                )
                detail_response.raise_for_status()
                detail = detail_response.json()
        finally:
            gpu_memory = sampler.stop()

        result_url = detail.get("url")
        if not isinstance(result_url, str) or not result_url:
            raise BenchmarkDataError(f"completed task has no result URL: {detail}")
        _download_result(result_url, downloaded_path)
        perf = _load_json_object(perf_path, "perf dump")
        effective = validate_effective_config(scheme, perf)
        perf_summary = summarize_perf(perf)

        quality: dict[str, Any]
        if role is RunRole.FORMAL:
            compare = run_compare(self.config, downloaded_path, compare_path)
            compare_summary = compare.get("summary")
            if not isinstance(compare_summary, Mapping):
                raise BenchmarkDataError("compare report is missing summary")
            failed_frames = compare_summary.get("failed_frames")
            compared_frames = compare_summary.get("compared_frames")
            failed_ratio = (
                len(failed_frames) / compared_frames
                if isinstance(failed_frames, list)
                and isinstance(compared_frames, int)
                and compared_frames > 0
                else None
            )
            quality = {
                "pass_compare": compare_summary.get("pass_compare") is True,
                "ssim_mean": compare_summary.get("ssim_mean"),
                "ssim_min": compare_summary.get("ssim_min"),
                "failed_frame_ratio": failed_ratio,
                "reason": None,
            }
        else:
            quality = {
                "pass_compare": None,
                "ssim_mean": None,
                "ssim_min": None,
                "failed_frame_ratio": None,
                "reason": "warmup_quality_not_evaluated",
            }

        status = (
            "quality_failed"
            if role is RunRole.FORMAL and not quality["pass_compare"]
            else "succeeded"
        )
        return {
            "status": status,
            "capability": {"status": "executable", "reason": None},
            "inputs": {
                "input_video": str(self.config.input_video),
                "caption_file": str(self.config.caption_file),
                "prompt_source": "caption_sidecar_file_only",
                "reference_video": str(self.config.reference_video),
                "num_frames": 130,
                "temporal_process_frames": 121,
                "inference_steps": 20,
                "seed": 42,
                "guidance_scale": 6.0,
                "restoration_guidance_scale": -1.0,
                "upscale": 1.0,
                "dtype": "bfloat16",
            },
            "runtime": {
                **effective,
                "parallel_topology": scheme.parallel_mode,
                "fusion": (
                    "cogvideox_modulation_transformer_controlnet"
                    if scheme.modulation_fusion
                    else "disabled"
                ),
                "cache": "disabled",
                "quantization": "disabled",
            },
            "timings": {
                "total_runtime_seconds": total_runtime_seconds,
                "model_inference_runtime_seconds": (
                    perf_summary.model_inference_runtime_seconds
                ),
                "denoising_runtime_seconds": perf_summary.denoising_runtime_seconds,
                "denoise_fraction": perf_summary.denoise_fraction,
                "stage_seconds": perf_summary.stage_seconds,
                "unclassified_seconds": perf_summary.unclassified_seconds,
                "temporal_clip_count": perf_summary.temporal_clip_count,
                "inference_step_count": perf_summary.inference_step_count,
                "mean_step_seconds": perf_summary.mean_step_seconds,
                "steady_step_median_seconds": (
                    perf_summary.steady_step_median_seconds
                ),
                "sp_communication_seconds": None,
                "sp_communication_reason": "not_profiled",
                "cfg_communication_seconds": None,
                "cfg_communication_reason": "not_profiled",
                "cache_executed_steps": None,
                "cache_skipped_steps": None,
                "cache_steps_reason": "cache_disabled_or_unsupported",
            },
            "gpu_memory": gpu_memory,
            "quality": quality,
            "derived": {},
            "artifacts": {
                "perf_json": str(perf_path),
                "result_video": str(downloaded_path),
                "compare_json": (
                    str(compare_path) if role is RunRole.FORMAL else None
                ),
                "service_log": str(
                    self.config.output_root
                    / batch_id
                    / "logs"
                    / f"{scheme.scheme_id}_service.log"
                ),
            },
            "reproducibility": {
                "repo_root": str(self.config.repo_root),
                "python_executable": str(self.config.python_executable),
                "model_path": str(self.config.model_path),
                "vividvr_path": str(self.config.vividvr_path),
                "service_command": build_service_command(scheme, self.config),
                "service_environment": build_service_environment(
                    scheme, self.config
                ),
                "config_fingerprint": fingerprint,
                "request_payload": payload,
            },
        }


class _CallbackHandler(BaseHTTPRequestHandler):
    log_path: Path

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self.send_error(404)
            return
        self._reply({"status": "ok"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_error(400)
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._reply({"code": 0})

    def _reply(self, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


class _CaptionHandler(BaseHTTPRequestHandler):
    caption_file: Path
    captions: list[str]

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self.send_error(404)
            return
        self._reply({"status": "ok"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/vividvr/captions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            expected = int(request["expected_caption_count"])
            if expected != len(self.captions):
                raise ValueError(
                    f"expected_caption_count={expected}, captions={len(self.captions)}"
                )
            output_path = Path(request["output_caption_path"]).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.caption_file, output_path)
        except Exception as error:
            self._reply({"error": str(error)}, status=400)
            return
        self._reply(
            {
                "caption_file_path": str(output_path),
                "caption_count": len(self.captions),
                "manifest_path": request["manifest_path"],
                "mode": "mock",
                "worker_count": 0,
                "fallback_used": False,
                "request_id": None,
                "total_clip_count": len(self.captions),
                "assigned_clip_indices_by_worker": {},
                "timing": {"mock": True},
            }
        )

    def _reply(self, payload: Mapping[str, Any], *, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def serve_callback(host: str, port: int, log_path: Path) -> None:
    handler = type("VividVRBenchmarkCallbackHandler", (_CallbackHandler,), {})
    handler.log_path = log_path
    ThreadingHTTPServer((host, port), handler).serve_forever()


def serve_caption(host: str, port: int, caption_file: Path) -> None:
    captions = caption_file.read_text(encoding="utf-8").splitlines()
    if not captions:
        raise BenchmarkConfigError(f"caption file is empty: {caption_file}")
    handler = type("VividVRBenchmarkCaptionHandler", (_CaptionHandler,), {})
    handler.caption_file = caption_file
    handler.captions = captions
    ThreadingHTTPServer((host, port), handler).serve_forever()


def build_dry_run_report(
    config: BenchmarkConfig, schemes: Sequence[Scheme]
) -> dict[str, Any]:
    required_gpus = max(
        (scheme.gpu_count for scheme in schemes if scheme.executable), default=0
    )
    if len(config.gpu_ids) < required_gpus:
        raise BenchmarkConfigError(
            f"selected schemes require {required_gpus} GPUs, got {config.gpu_ids}"
        )
    return {
        "mode": "dry-run",
        "preflight": run_preflight(config, check_runtime_resources=False),
        "scheme_count": len(schemes),
        "schemes": [
            {
                "scheme": _scheme_payload(scheme),
                "service_command": (
                    build_service_command(scheme, config)
                    if scheme.executable
                    else None
                ),
                "service_environment": (
                    build_service_environment(scheme, config)
                    if scheme.executable
                    else None
                ),
                "unsupported_reason": scheme.unsupported_reason,
                "requests": (
                    [RunRole.WARMUP.value, RunRole.FORMAL.value]
                    if scheme.executable and scheme.compile_enabled
                    else [RunRole.FORMAL.value]
                    if scheme.executable
                    else []
                ),
            }
            for scheme in schemes
        ],
    }


def _new_batch_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-path", type=Path, default=BenchmarkConfig.model_path)
    parser.add_argument("--vividvr-path", type=Path, default=BenchmarkConfig.vividvr_path)
    parser.add_argument("--input-video", type=Path, default=BenchmarkConfig.input_video)
    parser.add_argument("--caption-file", type=Path, default=BenchmarkConfig.caption_file)
    parser.add_argument(
        "--reference-video", type=Path, default=BenchmarkConfig.reference_video
    )
    parser.add_argument("--output-root", type=Path, default=BenchmarkConfig.output_root)
    parser.add_argument(
        "--gpu-ids",
        default=",".join(str(item) for item in BenchmarkConfig.gpu_ids),
        help="Comma-separated physical GPU IDs, in allocation order.",
    )
    parser.add_argument(
        "--allow-idle-gpu-processes",
        action="store_true",
        help=(
            "Allow existing compute processes on selected GPUs only when every "
            "selected GPU reports exactly 0%% utilization."
        ),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed VividVR acceleration benchmark matrix."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("dry-run", "run-all", "run-one"):
        child = subparsers.add_parser(command)
        _add_config_arguments(child)
        if command == "run-one":
            child.add_argument("--scheme", required=True, choices=list(SCHEMES))

    internal = subparsers.add_parser("_run-batch", help=argparse.SUPPRESS)
    _add_config_arguments(internal)
    internal.add_argument("--scheme", choices=list(SCHEMES))

    callback = subparsers.add_parser("_serve-callback", help=argparse.SUPPRESS)
    callback.add_argument("--host", required=True)
    callback.add_argument("--port", type=int, required=True)
    callback.add_argument("--callback-log", type=Path, required=True)

    caption = subparsers.add_parser("_serve-caption", help=argparse.SUPPRESS)
    caption.add_argument("--host", required=True)
    caption.add_argument("--port", type=int, required=True)
    caption.add_argument("--caption-file", type=Path, required=True)
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    raw_gpu_ids = [item.strip() for item in args.gpu_ids.split(",")]
    try:
        gpu_ids = tuple(int(item) for item in raw_gpu_ids if item)
    except ValueError as error:
        raise BenchmarkConfigError("--gpu-ids must be comma-separated integers") from error
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise BenchmarkConfigError("--gpu-ids must contain unique GPU IDs")
    return BenchmarkConfig(
        model_path=args.model_path.expanduser().resolve(),
        vividvr_path=args.vividvr_path.expanduser().resolve(),
        input_video=args.input_video.expanduser().resolve(),
        caption_file=args.caption_file.expanduser().resolve(),
        reference_video=args.reference_video.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        gpu_ids=gpu_ids,
        allow_idle_gpu_processes=args.allow_idle_gpu_processes,
    )


def _selected_schemes(args: argparse.Namespace) -> list[Scheme]:
    scheme_id = getattr(args, "scheme", None)
    return [SCHEMES[scheme_id]] if scheme_id else list(SCHEMES.values())


def _config_cli_arguments(config: BenchmarkConfig) -> list[str]:
    arguments = [
        "--model-path",
        str(config.model_path),
        "--vividvr-path",
        str(config.vividvr_path),
        "--input-video",
        str(config.input_video),
        "--caption-file",
        str(config.caption_file),
        "--reference-video",
        str(config.reference_video),
        "--output-root",
        str(config.output_root),
        "--gpu-ids",
        ",".join(str(item) for item in config.gpu_ids),
    ]
    if config.allow_idle_gpu_processes:
        arguments.append("--allow-idle-gpu-processes")
    return arguments


def _launch_detached_batch(
    config: BenchmarkConfig,
    *,
    batch_id: str,
    scheme_id: str | None,
    resume: bool,
) -> dict[str, Any]:
    token = _safe_batch_token(batch_id)
    session = f"vividvr_accel_batch_{token}"
    batch_dir = config.output_root / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(config.python_executable),
        str(Path(__file__).resolve()),
        "_run-batch",
        "--batch-id",
        batch_id,
        *_config_cli_arguments(config),
    ]
    if scheme_id:
        command.extend(["--scheme", scheme_id])
    if resume:
        command.append("--resume")
    shell_command = (
        f"cd {shlex.quote(str(config.repo_root))} && "
        f"PYTHONPATH={shlex.quote(str(config.repo_root / 'python'))} "
        f"{shlex.join(command)} 2>&1 | "
        f"tee {shlex.quote(str(batch_dir / 'batch.log'))}"
    )
    completed = _run_command(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session,
            "bash",
            "-lc",
            shell_command,
        ],
        capture_output=True,
    )
    if completed.returncode != 0:
        raise BenchmarkError(
            f"failed to launch batch tmux session {session}: "
            f"{(completed.stderr or '').strip()}"
        )
    return {
        "batch_id": batch_id,
        "session": session,
        "attach_command": f"tmux attach -r -t {session}",
        "batch_dir": str(batch_dir),
    }


def _run_batch(
    config: BenchmarkConfig,
    *,
    batch_id: str,
    schemes: Sequence[Scheme],
    resume: bool,
) -> dict[str, Any]:
    required_gpus = max(
        (scheme.gpu_count for scheme in schemes if scheme.executable), default=0
    )
    if len(config.gpu_ids) < required_gpus:
        raise BenchmarkConfigError(
            f"selected schemes require {required_gpus} GPUs, got {config.gpu_ids}"
        )
    lifecycle = TmuxBenchmarkLifecycle(config, batch_id)
    if resume:
        lifecycle.cleanup_owned()
    run_preflight(config, check_runtime_resources=True)
    runner = BenchmarkRunner(
        config=config,
        batch_id=batch_id,
        lifecycle=lifecycle,
        request_executor=FlowCutRequestExecutor(config),
        resume=resume,
    )
    return runner.run(schemes)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "_serve-callback":
        serve_callback(args.host, args.port, args.callback_log)
        return 0
    if args.command == "_serve-caption":
        serve_caption(args.host, args.port, args.caption_file)
        return 0

    try:
        config = _config_from_args(args)
        schemes = _selected_schemes(args)
        if args.command == "dry-run":
            print(
                json.dumps(
                    build_dry_run_report(config, schemes),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        if args.resume and not args.batch_id:
            raise BenchmarkConfigError("--resume requires an explicit --batch-id")
        batch_id = args.batch_id or _new_batch_id()
        if args.command == "_run-batch":
            result = _run_batch(
                config,
                batch_id=batch_id,
                schemes=schemes,
                resume=args.resume,
            )
        else:
            required_gpus = max(
                (scheme.gpu_count for scheme in schemes if scheme.executable),
                default=0,
            )
            if len(config.gpu_ids) < required_gpus:
                raise BenchmarkConfigError(
                    f"selected schemes require {required_gpus} GPUs, got {config.gpu_ids}"
                )
            if args.resume:
                TmuxBenchmarkLifecycle(config, batch_id).cleanup_owned()
            run_preflight(config, check_runtime_resources=True)
            result = _launch_detached_batch(
                config,
                batch_id=batch_id,
                scheme_id=(schemes[0].scheme_id if len(schemes) == 1 else None),
                resume=args.resume,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except BenchmarkError as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
