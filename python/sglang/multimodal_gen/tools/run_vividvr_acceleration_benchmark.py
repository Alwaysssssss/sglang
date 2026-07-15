#!/usr/bin/env python3
"""Run the fixed VividVR acceleration benchmark matrix.

The public CLI and lifecycle orchestration are added incrementally below.  The
experiment registry in this module is the source of truth shared by dry-run,
execution, and result serialization.
"""

from __future__ import annotations

import json
import os
import statistics
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path("/home/zhiheng/sglang")


class BenchmarkError(RuntimeError):
    """Base error for benchmark configuration and execution failures."""


class BenchmarkConfigError(BenchmarkError):
    """Raised before execution when a benchmark configuration is invalid."""


class BenchmarkDataError(BenchmarkError):
    """Raised when runtime evidence does not match the experiment contract."""


class SchemeStatus(str, Enum):
    EXECUTABLE = "executable"
    UNSUPPORTED = "unsupported"


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
            "compile_applied": None,
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
