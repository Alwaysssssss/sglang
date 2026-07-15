#!/usr/bin/env python3
"""Run the fixed VividVR acceleration benchmark matrix.

The public CLI and lifecycle orchestration are added incrementally below.  The
experiment registry in this module is the source of truth shared by dry-run,
execution, and result serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


REPO_ROOT = Path("/home/zhiheng/sglang")


class BenchmarkError(RuntimeError):
    """Base error for benchmark configuration and execution failures."""


class BenchmarkConfigError(BenchmarkError):
    """Raised before execution when a benchmark configuration is invalid."""


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
