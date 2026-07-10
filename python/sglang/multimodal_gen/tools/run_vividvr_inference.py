# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
import sysconfig
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from sglang.multimodal_gen.configs.pipeline_configs.vividvr import VividVRPipelineConfig
from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.configs.sample.vividvr import VividVRSamplingParams
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    post_process_sample,
    prepare_request,
)
from sglang.multimodal_gen.runtime.distributed import cleanup_dist_env_and_memory
from sglang.multimodal_gen.runtime.pipelines_core import build_pipeline
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
from sglang.multimodal_gen.runtime.videoedit.compare import compare_videos

VIVIDVR_ROOT = Path("/home/zhiheng/Vivid-VR")
COGVIDEOX_ROOT = VIVIDVR_ROOT / "ckpts" / "CogVideoX1.5-5B"
VIVIDVR_CKPT_ROOT = VIVIDVR_ROOT / "ckpts" / "Vivid-VR"
DEFAULT_PROMPT_FILE = VIVIDVR_ROOT / "input" / "720p" / "prompt.txt"
ACCEPTANCE_ROOT = Path("/home/zhiheng/sglang/Vivid_Acceptance")
DEFAULT_REPORT_DIR = ACCEPTANCE_ROOT / "indicator"
DEFAULT_OUTPUT_DIR = ACCEPTANCE_ROOT / "result_videos"


def _cleanup_local_distributed_runtime() -> None:
    try:
        cleanup_dist_env_and_memory()
    except Exception:
        # Best-effort teardown avoids leaked single-process NCCL groups after QKV fusion setup.
        pass


def _synchronize_ranks_before_cleanup() -> None:
    if not torch.distributed.is_available():
        return
    if not torch.distributed.is_initialized():
        return

    try:
        if torch.distributed.get_world_size() > 1:
            torch.distributed.barrier()
    except Exception as exc:
        print(
            "[VividVR] warning: distributed barrier before cleanup failed: "
            f"{exc}"
        )


def _distributed_rank_snapshot() -> dict[str, int]:
    return {
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
    }


def _is_primary_rank() -> bool:
    return _distributed_rank_snapshot()["rank"] == 0


def _prepend_env_path(env_name: str, value: Path) -> None:
    value_str = str(value)
    existing = os.environ.get(env_name)
    if existing is None or existing == "":
        os.environ[env_name] = value_str
        return

    segments = [segment for segment in existing.split(os.pathsep) if segment]
    if value_str not in segments:
        os.environ[env_name] = os.pathsep.join([value_str, *segments])


def _candidate_python_dev_include_dirs() -> list[Path]:
    version_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    major_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    compact_version = f"{sys.version_info.major}{sys.version_info.minor}"
    home = Path.home()
    candidates: list[Path] = []

    env_override = os.environ.get("SGLANG_PYTHON_DEV_INCLUDE")
    if env_override:
        candidates.append(Path(env_override).expanduser())

    for value in (
        sysconfig.get_config_var("INCLUDEPY"),
        sysconfig.get_path("include"),
        sysconfig.get_path("platinclude"),
    ):
        if value:
            candidates.append(Path(value))

    candidates.extend(
        [
            Path(sys.prefix) / "include" / version_tag,
            Path(sys.base_prefix) / "include" / version_tag,
            home / f"tmp_py{compact_version}dev" / "extracted" / "usr" / "include" / version_tag,
            home
            / f"tmp_py{compact_version}_headers"
            / "extracted"
            / f"libpython{major_minor}-dev"
            / "usr"
            / "include"
            / version_tag,
        ]
    )

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str not in seen:
            seen.add(candidate_str)
            deduped.append(candidate)
    return deduped


def _python_dev_extra_include_dirs(include_dir: Path) -> list[Path]:
    extras = [include_dir]
    multiarch = sysconfig.get_config_var("MULTIARCH")
    if multiarch:
        multiarch_dir = include_dir.parent / multiarch / include_dir.name
        if multiarch_dir.is_dir() and (multiarch_dir / "pyconfig.h").is_file():
            extras.append(include_dir.parent)
    return extras


def _ensure_python_dev_headers_for_torch_compile() -> Path | None:
    configured_include = sysconfig.get_config_var("INCLUDEPY")
    if configured_include:
        configured_path = Path(configured_include)
        if configured_path.is_dir() and (configured_path / "Python.h").is_file():
            return configured_path

    for candidate in _candidate_python_dev_include_dirs():
        if candidate.is_dir() and (candidate / "Python.h").is_file():
            for include_dir in _python_dev_extra_include_dirs(candidate):
                _prepend_env_path("CPATH", include_dir)
                _prepend_env_path("C_INCLUDE_PATH", include_dir)
            print(
                "[VividVR] torch.compile_python_include="
                f"{os.environ.get('CPATH')}"
            )
            return candidate

    print(
        "[VividVR] warning: torch.compile Python headers were not found; "
        "Triton compilation may fail."
    )
    return None


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, torch.Size):
        return list(value)
    return value


def build_runtime_config_snapshot(
    *,
    args: argparse.Namespace,
    server_args: ServerArgs,
    debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    debug = debug or {}
    runai_streamer_env = os.environ.get("SGLANG_USE_RUNAI_MODEL_STREAMER")
    return {
        "attention_backend_requested": args.attention_backend,
        "attention_backend_effective": server_args.attention_backend,
        "attention_backend_config": _json_ready(dict(server_args.attention_backend_config)),
        "runai_model_streamer_requested": getattr(
            args, "use_runai_model_streamer", None
        ),
        "runai_model_streamer_enabled": (
            None
            if runai_streamer_env is None
            else runai_streamer_env.strip().lower() in {"1", "true", "yes", "on"}
        ),
        "enable_torch_compile": bool(server_args.enable_torch_compile),
        "enable_usp_packed_qkv_a2a": bool(
            server_args.enable_usp_packed_qkv_a2a
        ),
        "enable_usp_prefix_all_gather_into_tensor": bool(
            server_args.enable_usp_prefix_all_gather_into_tensor
        ),
        "torch_compile_mode": os.environ.get("SGLANG_TORCH_COMPILE_MODE")
        if server_args.enable_torch_compile
        else None,
        "torch_compile_transformer": bool(debug.get("torch_compile_transformer", False)),
        "torch_compile_controlnet": bool(debug.get("torch_compile_controlnet", False)),
        "enable_cogvideox_modulation_fusion": bool(
            getattr(server_args, "enable_cogvideox_modulation_fusion", False)
        ),
        "cogvideox_modulation_fusion_targets": _json_ready(
            debug.get(
                "modulation_fusion_targets",
                getattr(
                    server_args,
                    "cogvideox_modulation_fusion_targets",
                    "transformer",
                ),
            )
        ),
        "modulation_fusion_transformer": debug.get("modulation_fusion_transformer"),
        "modulation_fusion_controlnet": debug.get("modulation_fusion_controlnet"),
        "enable_cogvideox_qkv_fusion": bool(
            getattr(server_args, "enable_cogvideox_qkv_fusion", False)
        ),
        "cogvideox_qkv_fusion_targets": _json_ready(
            debug.get(
                "qkv_fusion_targets",
                getattr(server_args, "cogvideox_qkv_fusion_targets", "transformer"),
            )
        ),
        "qkv_fusion_transformer": debug.get("qkv_fusion_transformer"),
        "qkv_fusion_controlnet": debug.get("qkv_fusion_controlnet"),
        "enable_cogvideox_qk_norm_fusion": bool(
            getattr(server_args, "enable_cogvideox_qk_norm_fusion", False)
        ),
        "cogvideox_qk_norm_fusion_targets": _json_ready(
            debug.get(
                "qk_norm_fusion_targets",
                getattr(
                    server_args,
                    "cogvideox_qk_norm_fusion_targets",
                    "transformer",
                ),
            )
        ),
        "qk_norm_fusion_transformer": debug.get("qk_norm_fusion_transformer"),
        "qk_norm_fusion_controlnet": debug.get("qk_norm_fusion_controlnet"),
        "enable_cogvideox_qk_norm_rope_fusion": bool(
            getattr(server_args, "enable_cogvideox_qk_norm_rope_fusion", False)
        ),
        "cogvideox_qk_norm_rope_fusion_targets": _json_ready(
            debug.get(
                "qk_norm_rope_fusion_targets",
                getattr(
                    server_args,
                    "cogvideox_qk_norm_rope_fusion_targets",
                    "transformer",
                ),
            )
        ),
        "qk_norm_rope_fusion_transformer": debug.get(
            "qk_norm_rope_fusion_transformer"
        ),
        "qk_norm_rope_fusion_controlnet": debug.get(
            "qk_norm_rope_fusion_controlnet"
        ),
        "dit_cpu_offload": bool(server_args.dit_cpu_offload),
        "text_encoder_cpu_offload": bool(server_args.text_encoder_cpu_offload),
        "vae_cpu_offload": bool(server_args.vae_cpu_offload),
        "disable_autocast": server_args.disable_autocast,
        "denoising_autocast_enabled": debug.get("denoising_autocast_enabled"),
        "denoising_target_dtype": debug.get("denoising_target_dtype"),
        "denoising_device_type": debug.get("denoising_device_type"),
        "device_placement_helper": debug.get("device_placement_helper"),
        "denoising_step_profile_helper": debug.get(
            "denoising_step_profile_helper"
        ),
        "attn_metadata_enabled": bool(debug.get("attn_metadata_enabled", False)),
        "attn_metadata_backend": debug.get("attn_metadata_backend"),
        "attn_metadata_builder": debug.get("attn_metadata_builder"),
        "enable_sequence_shard": bool(debug.get("enable_sequence_shard", False)),
        "sp_world_size": debug.get("sp_world_size"),
        "sp_rank": debug.get("sp_rank"),
        "sp_sequence_shard_strategy": debug.get("sp_sequence_shard_strategy"),
        "sp_sequence_tokens_global": debug.get("sp_sequence_tokens_global"),
        "sp_sequence_tokens_local": debug.get("sp_sequence_tokens_local"),
        "sp_sequence_tokens_pad": debug.get("sp_sequence_tokens_pad"),
        "sp_video_token_layout": debug.get("sp_video_token_layout"),
        "denoise_loop_local_compute_ms": debug.get("denoise_loop_local_compute_ms"),
        "denoise_loop_sp_comm_ms": debug.get("denoise_loop_sp_comm_ms"),
        "runtime_num_timesteps": debug.get("runtime_num_timesteps"),
        "connector_context_mode": debug.get("connector_context_mode"),
        "control_context_shape_local": _json_ready(
            debug.get("control_context_shape_local")
        ),
        "control_context_shape_global": _json_ready(
            debug.get("control_context_shape_global")
        ),
        "vividvr_vae_decode_tiling_requested": getattr(
            args, "use_vividvr_vae_decode_tiling", None
        ),
        "vividvr_vae_decode_tiling_config": bool(server_args.pipeline_config.vae_tiling),
        "vae_tiling_enabled": bool(debug.get("vae_tiling_enabled", False)),
        "num_gpus": int(server_args.num_gpus),
        "tp_size": int(server_args.tp_size),
        "dp_size": int(server_args.dp_size),
        "dp_degree": int(server_args.dp_degree),
        "sp_degree": int(server_args.sp_degree),
        "ulysses_degree": int(server_args.ulysses_degree),
        "ring_degree": int(server_args.ring_degree),
        "enable_cfg_parallel": bool(server_args.enable_cfg_parallel),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None
        ),
        "stage_profiling_enabled": True,
        "stage_profiling_synchronized": (
            os.environ.get("SGLANG_DIFFUSION_SYNC_STAGE_PROFILING", "0") == "1"
        ),
    }


def build_request_metrics_payload(result: Any, model_inference_runtime_seconds: float) -> dict[str, Any] | None:
    metrics = getattr(result, "metrics", None)
    if metrics is None:
        return None

    if getattr(metrics, "total_duration_ms", 0.0) <= 0:
        metrics.total_duration_ms = model_inference_runtime_seconds * 1000.0

    return _json_ready(metrics.to_dict())


def build_recorded_command() -> str:
    parts: list[str] = []
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        parts.append(f"PYTHONPATH={shlex.quote(pythonpath)}")

    repo_root = Path("/home/zhiheng/sglang")
    script_path = Path(__file__).resolve()
    script_display = script_path
    try:
        script_display = script_path.relative_to(repo_root)
    except ValueError:
        pass

    parts.append(shlex.quote(str(Path(sys.executable).resolve())))
    parts.append(shlex.quote(str(script_display)))
    parts.extend(shlex.quote(arg) for arg in sys.argv[1:])
    return " ".join(parts)


def build_server_args(args: argparse.Namespace) -> ServerArgs:
    pipeline_config = VividVRPipelineConfig()
    if getattr(args, "use_vividvr_vae_decode_tiling", None) is not None:
        pipeline_config.vae_tiling = bool(args.use_vividvr_vae_decode_tiling)

    server_args = ServerArgs(
        model_path=str(args.cogvideox_ckpt_path),
        pipeline_class_name="CogVideoXVividVRControlNetPipeline",
        pipeline_config=pipeline_config,
        component_paths={"vividvr": str(args.vividvr_ckpt_path)},
        num_gpus=args.num_gpus,
        tp_size=args.tp_size,
        dp_size=args.dp_size,
        dp_degree=args.dp_degree,
        sp_degree=args.sp_degree,
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        enable_cfg_parallel=args.enable_cfg_parallel,
        master_port=args.master_port,
        dist_timeout=args.dist_timeout,
        attention_backend=args.attention_backend,
        attention_backend_config=args.attention_backend_config,
        dit_cpu_offload=args.dit_cpu_offload,
        text_encoder_cpu_offload=args.text_encoder_cpu_offload,
        vae_cpu_offload=args.vae_cpu_offload,
        enable_torch_compile=args.enable_torch_compile,
        enable_usp_packed_qkv_a2a=args.enable_usp_packed_qkv_a2a,
        enable_usp_prefix_all_gather_into_tensor=(
            args.enable_usp_prefix_all_gather_into_tensor
        ),
        enable_cogvideox_modulation_fusion=args.enable_cogvideox_modulation_fusion,
        cogvideox_modulation_fusion_targets=args.cogvideox_modulation_fusion_targets,
        enable_cogvideox_qkv_fusion=args.enable_cogvideox_qkv_fusion,
        cogvideox_qkv_fusion_targets=args.cogvideox_qkv_fusion_targets,
        enable_cogvideox_qk_norm_fusion=args.enable_cogvideox_qk_norm_fusion,
        cogvideox_qk_norm_fusion_targets=args.cogvideox_qk_norm_fusion_targets,
        enable_cogvideox_qk_norm_rope_fusion=(
            args.enable_cogvideox_qk_norm_rope_fusion
        ),
        cogvideox_qk_norm_rope_fusion_targets=(
            args.cogvideox_qk_norm_rope_fusion_targets
        ),
        warmup=args.warmup,
        warmup_steps=args.warmup_steps,
        disable_autocast=args.disable_autocast,
        nunchaku_config=None,
        output_path=str(args.output_dir),
    )
    server_args._adjust_parameters()
    set_global_server_args(server_args)
    return server_args


def build_request(
    *,
    server_args: ServerArgs,
    args: argparse.Namespace,
    output_file_name: str,
):
    request_kwargs: dict[str, Any] = {
        "prompt": " ",
        "video_input_path": str(args.input_video),
        "output_path": str(args.output_dir),
        "output_file_name": output_file_name,
        "save_output": False,
        "return_file_paths_only": False,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "restoration_guidance_scale": args.restoration_guidance_scale,
        "num_temporal_process_frames": args.num_temporal_process_frames,
        "upscale": args.upscale,
        "dtype": args.dtype,
        "enable_spatial_tiling": args.enable_spatial_tiling,
        "enable_temporal_tiling": args.enable_temporal_tiling,
        "tile_size": args.tile_size,
        "tile_stride": args.tile_stride,
    }
    if args.prompt_file is not None:
        request_kwargs["prompt_file_path"] = str(args.prompt_file)
    if args.caption_file is not None:
        request_kwargs["caption_source"] = "caption_file"
        request_kwargs["caption_file_path"] = str(args.caption_file)

    params = VividVRSamplingParams.from_user_kwargs(server_args, **request_kwargs)
    return prepare_request(server_args, params)


def wait_for_reference_video(
    *,
    reference_video: Path,
    wait_for_reference_seconds: float,
    reference_poll_seconds: float,
) -> None:
    if reference_video.exists():
        return

    if wait_for_reference_seconds <= 0:
        raise SystemExit(f"Reference video does not exist: {reference_video}")

    deadline = time.perf_counter() + wait_for_reference_seconds
    poll_seconds = max(reference_poll_seconds, 0.1)
    while not reference_video.exists():
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise SystemExit(
                "Reference video is missing after waiting "
                f"{wait_for_reference_seconds} seconds: {reference_video}"
            )
        print(
            "[VividVR] waiting_for_reference "
            f"path={reference_video} "
            f"remaining_seconds={remaining:.1f}"
        )
        time.sleep(min(poll_seconds, remaining))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generic local VividVR inference runner for the native SGLang integration. "
            "Use prompt_file mode for Phase C-style runs or caption_file mode for fair "
            "Phase D / Phase E long-video benchmarks."
        )
    )
    parser.add_argument(
        "--input-video",
        type=Path,
        required=True,
        help="Single input video consumed by the SGLang-native VividVR pipeline.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help="Prompt txt used in prompt_file mode. Kept by default for parity with Phase C.",
    )
    parser.add_argument(
        "--caption-file",
        type=Path,
        default=None,
        help="Optional caption sidecar. If provided, the run switches to caption_file mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the generated candidate video is saved.",
    )
    parser.add_argument(
        "--output-file-name",
        default=None,
        help="Exact output file name. Defaults to <artifact-prefix>_seed<seed>_<run_id>.mp4.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory where the JSON run report is saved when --write-report is enabled.",
    )
    parser.add_argument(
        "--write-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a JSON run report under --report-dir.",
    )
    parser.add_argument(
        "--artifact-prefix",
        default=None,
        help="Stem used for generated artifact names when --output-file-name is omitted.",
    )
    parser.add_argument(
        "--reference-video",
        type=Path,
        default=None,
        help="Optional reference video used for compare_videos-based benchmark checks.",
    )
    parser.add_argument(
        "--phase-label",
        default=None,
        help="Optional phase label written into the run report, such as C / D / E.",
    )
    parser.add_argument(
        "--mode-label",
        default="single_video_inference",
        help="Mode label written into the run report.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the VividVR generator.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps. Keep original default at 50 unless explicitly benchmarking a lighter preset.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=6.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--restoration-guidance-scale",
        type=float,
        default=-1.0,
        help="Restoration guidance scale used by VividVR.",
    )
    parser.add_argument(
        "--num-temporal-process-frames",
        type=int,
        default=121,
        help="Temporal clip length used for long-video split / merge orchestration.",
    )
    parser.add_argument(
        "--upscale",
        type=float,
        default=1.0,
        help=(
            "Original Vivid-VR input upscale contract. 0 scales the short side to 1024, "
            "1 keeps the input resolution, and other positive values apply a direct "
            "pre-inference resize factor."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Sampling dtype forwarded to the VividVR request contract.",
    )
    parser.add_argument(
        "--enable-spatial-tiling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable spatial tiling in the VividVR request contract.",
    )
    parser.add_argument(
        "--enable-temporal-tiling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable temporal tiling in the VividVR request contract.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=128,
        help="Spatial tile size used by the native VividVR runtime.",
    )
    parser.add_argument(
        "--tile-stride",
        type=int,
        default=64,
        help="Spatial tile stride used by the native VividVR runtime.",
    )
    parser.add_argument(
        "--attention-backend",
        default=None,
        help="Optional attention backend override for Phase E perf sweeps.",
    )
    parser.add_argument(
        "--attention-backend-config",
        default=None,
        help="Optional attention backend config string or JSON path.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Total GPU count for the VividVR runtime. Under torchrun this should match WORLD_SIZE.",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=None,
        help="Tensor parallel degree. E4.1 formal runs should keep this at 1.",
    )
    parser.add_argument(
        "--sp-degree",
        type=int,
        default=None,
        help="Sequence parallel degree. Defaults to remaining GPUs after TP/DP accounting.",
    )
    parser.add_argument(
        "--ulysses-degree",
        type=int,
        default=None,
        help="Ulysses sequence-parallel degree. Defaults to sp_degree when ring_degree is unset.",
    )
    parser.add_argument(
        "--ring-degree",
        type=int,
        default=None,
        help="Ring sequence-parallel degree. E4.1 formal runs should keep this at 1.",
    )
    parser.add_argument(
        "--enable-cfg-parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable classifier-free-guidance parallelism.",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=1,
        help="Data parallel degree. DP is not supported for VividVR E4.1 and should remain 1.",
    )
    parser.add_argument(
        "--dp-degree",
        type=int,
        default=1,
        help="Legacy DP degree field mirrored into ServerArgs for runtime snapshots.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=30005,
        help="torch.distributed master port. Under torchrun this should match MASTER_PORT if explicitly set.",
    )
    parser.add_argument(
        "--dist-timeout",
        type=int,
        default=3600,
        help="torch.distributed timeout in seconds.",
    )
    parser.add_argument(
        "--use-runai-model-streamer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional override for SGLANG_USE_RUNAI_MODEL_STREAMER during this run.",
    )
    parser.add_argument(
        "--use-vividvr-vae-decode-tiling",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional override for VividVR decode-side pipeline_config.vae_tiling.",
    )
    parser.add_argument(
        "--enable-cogvideox-modulation-fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable CogVideoX/VividVR block modulation fusion on the native pipeline.",
    )
    parser.add_argument(
        "--cogvideox-modulation-fusion-targets",
        type=str,
        default="transformer",
        help="Comma-separated VividVR components to fuse for Phase E3. Supported: transformer,controlnet.",
    )
    parser.add_argument(
        "--enable-cogvideox-qkv-fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable CogVideoX/VividVR fused QKV projection path.",
    )
    parser.add_argument(
        "--cogvideox-qkv-fusion-targets",
        type=str,
        default="transformer",
        help="Comma-separated VividVR components to fuse for Phase E3. Supported: transformer,controlnet.",
    )
    parser.add_argument(
        "--enable-cogvideox-qk-norm-rope-fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable CogVideoX/VividVR QK LayerNorm + image RoPE acceleration on the native pipeline.",
    )
    parser.add_argument(
        "--enable-cogvideox-qk-norm-fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable CogVideoX/VividVR QK LayerNorm acceleration while preserving exact image RoPE.",
    )
    parser.add_argument(
        "--cogvideox-qk-norm-rope-fusion-targets",
        type=str,
        default="transformer",
        help="Comma-separated VividVR components to accelerate for Phase E3. Supported: transformer,controlnet.",
    )
    parser.add_argument(
        "--cogvideox-qk-norm-fusion-targets",
        type=str,
        default="transformer",
        help="Comma-separated VividVR components to accelerate for Phase E3. Supported: transformer,controlnet.",
    )
    parser.add_argument(
        "--dit-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable DiT CPU offload. Default stays False to match the accepted local scripts.",
    )
    parser.add_argument(
        "--text-encoder-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable text encoder CPU offload. Default stays False to match the accepted local scripts.",
    )
    parser.add_argument(
        "--vae-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable VAE CPU offload. Default stays False to match the accepted local scripts.",
    )
    parser.add_argument(
        "--enable-torch-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable torch.compile for the native pipeline.",
    )
    parser.add_argument(
        "--enable-usp-packed-qkv-a2a",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pack USP Q/K/V input all-to-all into one collective.",
    )
    parser.add_argument(
        "--enable-usp-prefix-all-gather-into-tensor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a tensor-form functional gather for USP replicated-prefix output.",
    )
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable warmup before the main run.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        help="Warmup step count when --warmup is enabled.",
    )
    parser.add_argument(
        "--disable-autocast",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional autocast override forwarded to ServerArgs.",
    )
    parser.add_argument(
        "--cogvideox-ckpt-path",
        type=Path,
        default=COGVIDEOX_ROOT,
        help="CogVideoX base checkpoint path.",
    )
    parser.add_argument(
        "--vividvr-ckpt-path",
        type=Path,
        default=VIVIDVR_CKPT_ROOT,
        help="Native VividVR component checkpoint path.",
    )
    parser.add_argument(
        "--min-ssim",
        type=float,
        default=0.90,
        help="Minimum SSIM threshold when --reference-video is provided.",
    )
    parser.add_argument(
        "--max-mse",
        type=float,
        default=150.0,
        help="Maximum MSE threshold when --reference-video is provided.",
    )
    parser.add_argument(
        "--max-mae",
        type=float,
        default=8.0,
        help="Maximum MAE threshold when --reference-video is provided.",
    )
    parser.add_argument(
        "--allow-frame-count-delta",
        type=int,
        default=1,
        help="Maximum allowed frame-count delta for reference comparison.",
    )
    parser.add_argument(
        "--max-failed-frame-ratio",
        type=float,
        default=0.05,
        help="Maximum failed-frame ratio for reference comparison.",
    )
    parser.add_argument(
        "--wait-for-reference-seconds",
        type=float,
        default=0.0,
        help="Optional wait time if --reference-video is expected to appear later.",
    )
    parser.add_argument(
        "--reference-poll-seconds",
        type=float,
        default=10.0,
        help="Polling interval used while waiting for --reference-video.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate args, resolve artifact paths, print config JSON, and exit without loading models.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_video.is_file():
        raise SystemExit(f"Input video does not exist: {args.input_video}")

    if not args.cogvideox_ckpt_path.exists():
        raise SystemExit(
            f"CogVideoX checkpoint path does not exist: {args.cogvideox_ckpt_path}"
        )
    if not args.vividvr_ckpt_path.exists():
        raise SystemExit(f"VividVR checkpoint path does not exist: {args.vividvr_ckpt_path}")

    if args.caption_file is not None:
        if not args.caption_file.is_file():
            raise SystemExit(f"Caption file does not exist: {args.caption_file}")
    elif args.prompt_file is None or not args.prompt_file.is_file():
        raise SystemExit(f"Prompt file does not exist: {args.prompt_file}")

    if args.reference_video is not None and args.reference_video.exists():
        if not args.reference_video.is_file():
            raise SystemExit(f"Reference video is not a file: {args.reference_video}")

    if args.num_inference_steps <= 0:
        raise SystemExit("--num-inference-steps must be positive")
    if args.num_temporal_process_frames <= 0:
        raise SystemExit("--num-temporal-process-frames must be positive")
    if not math.isfinite(args.upscale) or args.upscale < 0:
        raise SystemExit("--upscale must be a finite float >= 0")
    if args.allow_frame_count_delta < 0:
        raise SystemExit("--allow-frame-count-delta must be >= 0")
    if args.num_gpus <= 0:
        raise SystemExit("--num-gpus must be positive")
    if args.dp_size <= 0:
        raise SystemExit("--dp-size must be positive")
    if args.dp_degree <= 0:
        raise SystemExit("--dp-degree must be positive")
    if args.tp_size is not None and args.tp_size <= 0:
        raise SystemExit("--tp-size must be positive when provided")
    if args.sp_degree is not None and args.sp_degree <= 0:
        raise SystemExit("--sp-degree must be positive when provided")
    if args.ulysses_degree is not None and args.ulysses_degree <= 0:
        raise SystemExit("--ulysses-degree must be positive when provided")
    if args.ring_degree is not None and args.ring_degree <= 0:
        raise SystemExit("--ring-degree must be positive when provided")
    if args.dist_timeout <= 0:
        raise SystemExit("--dist-timeout must be positive")

    rank_snapshot = _distributed_rank_snapshot()
    if rank_snapshot["world_size"] > 1 and args.num_gpus != rank_snapshot["world_size"]:
        raise SystemExit(
            "--num-gpus must match WORLD_SIZE under torchrun: "
            f"num_gpus={args.num_gpus}, WORLD_SIZE={rank_snapshot['world_size']}"
        )


def build_artifact_prefix(args: argparse.Namespace) -> str:
    if args.artifact_prefix:
        return args.artifact_prefix
    if args.output_file_name:
        return Path(args.output_file_name).stem
    return args.input_video.stem


def build_output_file_name(args: argparse.Namespace, run_id: str) -> str:
    if args.output_file_name:
        name = args.output_file_name
    else:
        artifact_prefix = build_artifact_prefix(args)
        name = f"{artifact_prefix}_seed{args.seed}_{run_id}.mp4"
    if Path(name).suffix == "":
        name = f"{name}.mp4"
    return name


def build_report_path(args: argparse.Namespace, run_id: str) -> Path | None:
    if not args.write_report:
        return None
    artifact_prefix = build_artifact_prefix(args)
    report_kind = "metrics" if args.reference_video is not None else "report"
    return args.report_dir / f"{artifact_prefix}_{report_kind}_seed{args.seed}_{run_id}.json"


def build_dry_run_payload(
    args: argparse.Namespace,
    *,
    candidate_path: Path,
    report_path: Path | None,
    run_id: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "input_video_path": str(args.input_video),
        "prompt_file_path": str(args.prompt_file) if args.prompt_file is not None else None,
        "caption_file_path": str(args.caption_file) if args.caption_file is not None else None,
        "caption_source": "caption_file" if args.caption_file is not None else "prompt_file",
        "reference_video_path": (
            str(args.reference_video) if args.reference_video is not None else None
        ),
        "candidate_video_path": str(candidate_path),
        "report_path": str(report_path) if report_path is not None else None,
        "phase_label": args.phase_label,
        "mode_label": args.mode_label,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "restoration_guidance_scale": args.restoration_guidance_scale,
        "num_temporal_process_frames": args.num_temporal_process_frames,
        "upscale": args.upscale,
        "dtype": args.dtype,
        "enable_spatial_tiling": args.enable_spatial_tiling,
        "enable_temporal_tiling": args.enable_temporal_tiling,
        "tile_size": args.tile_size,
        "tile_stride": args.tile_stride,
        "attention_backend": args.attention_backend,
        "attention_backend_config": args.attention_backend_config,
        "use_runai_model_streamer": args.use_runai_model_streamer,
        "use_vividvr_vae_decode_tiling": args.use_vividvr_vae_decode_tiling,
        "dit_cpu_offload": args.dit_cpu_offload,
        "text_encoder_cpu_offload": args.text_encoder_cpu_offload,
        "vae_cpu_offload": args.vae_cpu_offload,
        "enable_torch_compile": args.enable_torch_compile,
        "enable_usp_packed_qkv_a2a": args.enable_usp_packed_qkv_a2a,
        "enable_usp_prefix_all_gather_into_tensor": (
            args.enable_usp_prefix_all_gather_into_tensor
        ),
        "enable_cogvideox_modulation_fusion": args.enable_cogvideox_modulation_fusion,
        "cogvideox_modulation_fusion_targets": args.cogvideox_modulation_fusion_targets,
        "enable_cogvideox_qkv_fusion": args.enable_cogvideox_qkv_fusion,
        "cogvideox_qkv_fusion_targets": args.cogvideox_qkv_fusion_targets,
        "enable_cogvideox_qk_norm_fusion": args.enable_cogvideox_qk_norm_fusion,
        "cogvideox_qk_norm_fusion_targets": args.cogvideox_qk_norm_fusion_targets,
        "enable_cogvideox_qk_norm_rope_fusion": (
            args.enable_cogvideox_qk_norm_rope_fusion
        ),
        "cogvideox_qk_norm_rope_fusion_targets": (
            args.cogvideox_qk_norm_rope_fusion_targets
        ),
        "num_gpus": args.num_gpus,
        "tp_size": args.tp_size,
        "dp_size": args.dp_size,
        "dp_degree": args.dp_degree,
        "sp_degree": args.sp_degree,
        "ulysses_degree": args.ulysses_degree,
        "ring_degree": args.ring_degree,
        "enable_cfg_parallel": args.enable_cfg_parallel,
        "master_port": args.master_port,
        "dist_timeout": args.dist_timeout,
        "distributed_env": _distributed_rank_snapshot(),
        "warmup": args.warmup,
        "warmup_steps": args.warmup_steps,
        "disable_autocast": args.disable_autocast,
        "write_report": args.write_report,
    }


def main() -> int:
    try:
        total_start_time = time.perf_counter()
        args = parse_args()
        validate_args(args)

        os.environ.setdefault("PYTHONUNBUFFERED", "1")
        os.environ.setdefault("SGLANG_DIFFUSION_STAGE_LOGGING", "1")
        if args.use_runai_model_streamer is not None:
            os.environ["SGLANG_USE_RUNAI_MODEL_STREAMER"] = (
                "1" if args.use_runai_model_streamer else "0"
            )
        if args.enable_torch_compile:
            _ensure_python_dev_headers_for_torch_compile()

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_file_name = build_output_file_name(args, run_id)
        candidate_path = args.output_dir / output_file_name
        report_path = build_report_path(args, run_id)

        print(f"[VividVR] run_id={run_id}")
        print(f"[VividVR] input_video={args.input_video}")
        print(
            "[VividVR] distributed "
            f"world_size={_distributed_rank_snapshot()['world_size']} "
            f"rank={_distributed_rank_snapshot()['rank']} "
            f"local_rank={_distributed_rank_snapshot()['local_rank']}"
        )
        print(
            "[VividVR] caption_source="
            f"{'caption_file' if args.caption_file is not None else 'prompt_file'}"
        )
        print(f"[VividVR] candidate_video={candidate_path}")
        if report_path is not None:
            print(f"[VividVR] report_path={report_path}")
        if args.reference_video is not None:
            print(f"[VividVR] reference_video={args.reference_video}")

        if args.dry_run:
            payload = build_dry_run_payload(
                args,
                candidate_path=candidate_path,
                report_path=report_path,
                run_id=run_id,
            )
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        if (
            args.reference_video is not None
            and not args.reference_video.exists()
            and args.wait_for_reference_seconds <= 0
        ):
            raise SystemExit(
                "Reference video does not exist and no wait window was provided: "
                f"{args.reference_video}"
            )

        if not torch.cuda.is_available():
            raise SystemExit("CUDA is required for VividVR inference")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        if report_path is not None:
            args.report_dir.mkdir(parents=True, exist_ok=True)

        server_args = build_server_args(args)
        pipeline = build_pipeline(server_args)
        request = build_request(
            server_args=server_args,
            args=args,
            output_file_name=output_file_name,
        )

        if args.warmup:
            warmup_request = request.copy_as_warmup(args.warmup_steps)
            print(
                "[VividVR] warmup_enabled=true "
                f"warmup_steps={args.warmup_steps}"
            )
            pipeline.forward(warmup_request, server_args)

        model_inference_start_time = time.perf_counter()
        result = pipeline.forward(request, server_args)
        model_inference_runtime_seconds = round(
            time.perf_counter() - model_inference_start_time, 6
        )
        request_metrics = build_request_metrics_payload(
            result, model_inference_runtime_seconds
        )

        reference_video_for_save = None
        if args.reference_video is not None and args.reference_video.exists():
            reference_video_for_save = str(args.reference_video)

        if _is_primary_rank():
            post_process_sample(
                result.output,
                DataType.VIDEO,
                int(result.fps),
                save_output=True,
                save_file_path=str(candidate_path),
                video_reference_path=reference_video_for_save,
            )

        debug = result.extra.get("vividvr_debug", {})
        runtime_config = build_runtime_config_snapshot(
            args=args,
            server_args=server_args,
            debug=debug,
        )
        metrics_record: dict[str, Any] = {
            "phase": args.phase_label,
            "mode": args.mode_label,
            "run_id": run_id,
            "run_datetime_utc": datetime.now(timezone.utc).isoformat(),
            "command": build_recorded_command(),
            "total_runtime_seconds": round(time.perf_counter() - total_start_time, 6),
            "model_inference_runtime_seconds": model_inference_runtime_seconds,
            "seed": args.seed,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "restoration_guidance_scale": args.restoration_guidance_scale,
            "num_temporal_process_frames": args.num_temporal_process_frames,
            "dtype": args.dtype,
            "enable_usp_packed_qkv_a2a": args.enable_usp_packed_qkv_a2a,
            "enable_usp_prefix_all_gather_into_tensor": (
                args.enable_usp_prefix_all_gather_into_tensor
            ),
            "prompt_path": str(args.prompt_file) if args.prompt_file is not None else None,
            "caption_file_path": str(args.caption_file) if args.caption_file is not None else None,
            "caption_source": "caption_file" if args.caption_file is not None else "prompt_file",
            "input_video_path": str(args.input_video),
            "reference_video_path": (
                str(args.reference_video) if args.reference_video is not None else None
            ),
            "candidate_video_path": str(candidate_path),
            "compare_enabled": args.reference_video is not None,
            "runtime_config": runtime_config,
            "distributed_env": _distributed_rank_snapshot(),
            "request_metrics": request_metrics,
            "stage_metrics_ms": None if request_metrics is None else request_metrics["stages"],
            "denoising_step_metrics_ms": (
                None if request_metrics is None else request_metrics["steps"]
            ),
            "request_metrics_total_duration_ms": (
                None
                if request_metrics is None
                else request_metrics["total_duration_ms"]
            ),
            "summary": None,
            "frames": None,
            "reference_frame_count": None,
            "candidate_frame_count": None,
            "frame_count_delta": None,
            "failed_frame_ratio": None,
            "debug": debug,
        }

        exit_code = 0
        if _is_primary_rank() and args.reference_video is not None:
            wait_for_reference_video(
                reference_video=args.reference_video,
                wait_for_reference_seconds=args.wait_for_reference_seconds,
                reference_poll_seconds=args.reference_poll_seconds,
            )
            report = compare_videos(
                str(args.reference_video),
                str(candidate_path),
                min_ssim=args.min_ssim,
                max_mse=args.max_mse,
                max_mae=args.max_mae,
                allow_frame_count_delta=args.allow_frame_count_delta,
                max_failed_frame_ratio=args.max_failed_frame_ratio,
            )
            summary = report["summary"]
            reference_frame_count = int(summary["reference_frame_count"])
            candidate_frame_count = int(summary["candidate_frame_count"])
            failed_frame_ratio = (
                len(summary["failed_frames"]) / summary["compared_frames"]
                if summary["compared_frames"] > 0
                else 1.0
            )

            metrics_record.update(
                {
                    "summary": summary,
                    "frames": report["frames"],
                    "reference_frame_count": reference_frame_count,
                    "candidate_frame_count": candidate_frame_count,
                    "frame_count_delta": abs(
                        reference_frame_count - candidate_frame_count
                    ),
                    "failed_frame_ratio": failed_frame_ratio,
                    "pass_compare": bool(summary["pass_compare"]),
                }
            )
            print(
                "[VividVR] summary "
                f"pass_compare={summary['pass_compare']} "
                f"ssim_min={summary['ssim_min']:.6f} "
                f"mse_max={summary['mse_max']:.6f} "
                f"mae_max={summary['mae_max']:.6f} "
                f"failed_frame_ratio={failed_frame_ratio:.6f}"
            )
            exit_code = 0 if summary["pass_compare"] else 1
        elif _is_primary_rank():
            metrics_record["pass_compare"] = None
            print("[VividVR] summary compare_disabled reference_video=None")
        else:
            metrics_record["pass_compare"] = None

        if _is_primary_rank() and report_path is not None:
            report_path.write_text(json.dumps(metrics_record, indent=2), encoding="utf-8")
            print(f"[VividVR] report saved to {report_path}")
        if _is_primary_rank():
            print(f"[VividVR] candidate saved to {candidate_path}")

        return exit_code
    finally:
        _synchronize_ranks_before_cleanup()
        _cleanup_local_distributed_runtime()


if __name__ == "__main__":
    raise SystemExit(main())
