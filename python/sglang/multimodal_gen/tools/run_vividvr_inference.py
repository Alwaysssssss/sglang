# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
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
from sglang.multimodal_gen.runtime.pipelines_core import build_pipeline
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
from sglang.multimodal_gen.runtime.videoedit.compare import compare_videos
from sglang.multimodal_gen.runtime.videoedit.preprocess import load_video_frames

VIVIDVR_ROOT = Path("/home/zhiheng/Vivid-VR")
COGVIDEOX_ROOT = VIVIDVR_ROOT / "ckpts" / "CogVideoX1.5-5B"
VIVIDVR_CKPT_ROOT = VIVIDVR_ROOT / "ckpts" / "Vivid-VR"
DEFAULT_PROMPT_FILE = VIVIDVR_ROOT / "input" / "720p" / "prompt.txt"
ACCEPTANCE_ROOT = Path("/home/zhiheng/sglang/Vivid_Acceptance")
DEFAULT_REPORT_DIR = ACCEPTANCE_ROOT / "indicator"
DEFAULT_OUTPUT_DIR = ACCEPTANCE_ROOT / "result_videos"


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
    server_args = ServerArgs(
        model_path=str(args.cogvideox_ckpt_path),
        pipeline_class_name="CogVideoXVividVRControlNetPipeline",
        pipeline_config=VividVRPipelineConfig(),
        component_paths={"vividvr": str(args.vividvr_ckpt_path)},
        num_gpus=1,
        tp_size=1,
        dp_size=1,
        dp_degree=1,
        sp_degree=1,
        attention_backend=args.attention_backend,
        attention_backend_config=args.attention_backend_config,
        dit_cpu_offload=args.dit_cpu_offload,
        text_encoder_cpu_offload=args.text_encoder_cpu_offload,
        vae_cpu_offload=args.vae_cpu_offload,
        enable_torch_compile=args.enable_torch_compile,
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
    if args.allow_frame_count_delta < 0:
        raise SystemExit("--allow-frame-count-delta must be >= 0")


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
        "dtype": args.dtype,
        "enable_spatial_tiling": args.enable_spatial_tiling,
        "enable_temporal_tiling": args.enable_temporal_tiling,
        "tile_size": args.tile_size,
        "tile_stride": args.tile_stride,
        "attention_backend": args.attention_backend,
        "attention_backend_config": args.attention_backend_config,
        "dit_cpu_offload": args.dit_cpu_offload,
        "text_encoder_cpu_offload": args.text_encoder_cpu_offload,
        "vae_cpu_offload": args.vae_cpu_offload,
        "enable_torch_compile": args.enable_torch_compile,
        "warmup": args.warmup,
        "warmup_steps": args.warmup_steps,
        "disable_autocast": args.disable_autocast,
        "write_report": args.write_report,
    }


def main() -> int:
    total_start_time = time.perf_counter()
    args = parse_args()
    validate_args(args)

    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_file_name = build_output_file_name(args, run_id)
    candidate_path = args.output_dir / output_file_name
    report_path = build_report_path(args, run_id)

    print(f"[VividVR] run_id={run_id}")
    print(f"[VividVR] input_video={args.input_video}")
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

    model_inference_start_time = time.perf_counter()
    result = pipeline.forward(request, server_args)
    model_inference_runtime_seconds = round(
        time.perf_counter() - model_inference_start_time, 6
    )

    reference_video_for_save = None
    if args.reference_video is not None and args.reference_video.exists():
        reference_video_for_save = str(args.reference_video)

    post_process_sample(
        result.output,
        DataType.VIDEO,
        int(result.fps),
        save_output=True,
        save_file_path=str(candidate_path),
        video_reference_path=reference_video_for_save,
    )

    debug = result.extra.get("vividvr_debug", {})
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
        "prompt_path": str(args.prompt_file) if args.prompt_file is not None else None,
        "caption_file_path": str(args.caption_file) if args.caption_file is not None else None,
        "caption_source": "caption_file" if args.caption_file is not None else "prompt_file",
        "input_video_path": str(args.input_video),
        "reference_video_path": (
            str(args.reference_video) if args.reference_video is not None else None
        ),
        "candidate_video_path": str(candidate_path),
        "compare_enabled": args.reference_video is not None,
        "summary": None,
        "frames": None,
        "reference_frame_count": None,
        "candidate_frame_count": None,
        "frame_count_delta": None,
        "failed_frame_ratio": None,
        "debug": debug,
    }

    exit_code = 0
    if args.reference_video is not None:
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
        ref_frames, _ = load_video_frames(str(args.reference_video))
        cand_frames, _ = load_video_frames(str(candidate_path))
        failed_frame_ratio = (
            len(summary["failed_frames"]) / summary["compared_frames"]
            if summary["compared_frames"] > 0
            else 1.0
        )

        metrics_record.update(
            {
                "summary": summary,
                "frames": report["frames"],
                "reference_frame_count": len(ref_frames),
                "candidate_frame_count": len(cand_frames),
                "frame_count_delta": abs(len(ref_frames) - len(cand_frames)),
                "failed_frame_ratio": failed_frame_ratio,
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
    else:
        print("[VividVR] summary compare_disabled reference_video=None")

    if report_path is not None:
        report_path.write_text(json.dumps(metrics_record, indent=2), encoding="utf-8")
        print(f"[VividVR] report saved to {report_path}")
    print(f"[VividVR] candidate saved to {candidate_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
