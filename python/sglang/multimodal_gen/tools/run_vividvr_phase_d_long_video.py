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
PROMPT_FILE = VIVIDVR_ROOT / "input" / "720p" / "prompt.txt"
ACCEPTANCE_ROOT = Path("/home/zhiheng/sglang/Vivid_Acceptance")
INDICATOR_DIR = ACCEPTANCE_ROOT / "indicator"
RESULT_VIDEOS_DIR = ACCEPTANCE_ROOT / "result_videos"
PHASE_D_130F_INPUT_VIDEO = (
    VIVIDVR_ROOT / "input" / "720p_long" / "test_video_long_960x720_130f.mp4"
)
PHASE_D_130F_CAPTION_FILE = (
    VIVIDVR_ROOT / "input" / "720p_long" / "test_video_long_960x720_130f.txt"
)
PHASE_D_130F_REFERENCE_VIDEO_20STEP = (
    VIVIDVR_ROOT
    / "result"
    / "720p_long_up1_result_vivid_ori_20step"
    / "videos"
    / PHASE_D_130F_INPUT_VIDEO.name
)
PHASE_D_130F_CAPTION_FILE_50STEP = (
    VIVIDVR_ROOT / "input" / "720p_long" / "test_video_long_960x720_130f_50step.txt"
)
PHASE_D_130F_REFERENCE_VIDEO_50STEP = (
    VIVIDVR_ROOT
    / "result"
    / "720p_long_up1_result_vivid_ori_50step"
    / "videos"
    / PHASE_D_130F_INPUT_VIDEO.name
)

PHASE_D_PRESETS: dict[str, dict[str, Path | int]] = {
    "phase_d_130f_20step": {
        "prepared_input_video": PHASE_D_130F_INPUT_VIDEO,
        "reference_video": PHASE_D_130F_REFERENCE_VIDEO_20STEP,
        "caption_file": PHASE_D_130F_CAPTION_FILE,
        "num_inference_steps": 20,
    },
    "phase_d_130f_50step": {
        "prepared_input_video": PHASE_D_130F_INPUT_VIDEO,
        "reference_video": PHASE_D_130F_REFERENCE_VIDEO_50STEP,
        "caption_file": PHASE_D_130F_CAPTION_FILE_50STEP,
        "num_inference_steps": 50,
    },
}


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

    parts.append(shlex.quote(sys.executable))
    parts.append(shlex.quote(str(script_display)))
    parts.extend(shlex.quote(arg) for arg in sys.argv[1:])
    return " ".join(parts)


def build_server_args() -> ServerArgs:
    server_args = ServerArgs(
        model_path=str(COGVIDEOX_ROOT),
        pipeline_class_name="CogVideoXVividVRControlNetPipeline",
        pipeline_config=VividVRPipelineConfig(),
        component_paths={"vividvr": str(VIVIDVR_CKPT_ROOT)},
        num_gpus=1,
        tp_size=1,
        dp_size=1,
        dp_degree=1,
        sp_degree=1,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        nunchaku_config=None,
        output_path=str(RESULT_VIDEOS_DIR),
    )
    server_args._adjust_parameters()
    set_global_server_args(server_args)
    return server_args


def make_request(
    *,
    server_args: ServerArgs,
    input_video_path: Path,
    output_file_name: str,
    seed: int,
    num_inference_steps: int,
    caption_file_path: Path | None,
):
    request_kwargs = {
        "prompt": " ",
        "video_input_path": str(input_video_path),
        "prompt_file_path": str(PROMPT_FILE),
        "output_path": str(RESULT_VIDEOS_DIR),
        "output_file_name": output_file_name,
        "save_output": False,
        "return_file_paths_only": False,
        "seed": seed,
        "num_inference_steps": num_inference_steps,
    }
    if caption_file_path is not None:
        request_kwargs["caption_source"] = "caption_file"
        request_kwargs["caption_file_path"] = str(caption_file_path)

    params = VividVRSamplingParams.from_user_kwargs(
        server_args,
        **request_kwargs,
    )
    return prepare_request(server_args, params)


def probe_existing_input_video(input_video_path: Path) -> dict[str, int | float | str]:
    frames, fps = load_video_frames(str(input_video_path))
    if not frames:
        raise ValueError(f"No frames found in input video: {input_video_path}")

    width, height = frames[0].size
    return {
        "source_frame_count": len(frames),
        "prepared_frame_count": len(frames),
        "fps": float(fps),
        "height": height,
        "width": width,
        "prepared_video_path": str(input_video_path),
    }


def apply_phase_d_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset is None:
        return args

    preset = PHASE_D_PRESETS[args.preset]
    for field_name, value in preset.items():
        setattr(args, field_name, value)
    return args


def build_artifact_prefix(args: argparse.Namespace) -> str:
    if args.preset:
        return args.preset
    return "phase_d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase D long-video acceptance against an original Vivid-VR reference."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PHASE_D_PRESETS.keys()),
        default=None,
        help="Apply a predefined Phase D benchmark configuration.",
    )
    parser.add_argument(
        "--prepared-input-video",
        type=Path,
        default=PHASE_D_130F_INPUT_VIDEO,
        help="Path of the long-video input consumed by both benchmarks.",
    )
    parser.add_argument(
        "--reference-video",
        type=Path,
        default=PHASE_D_130F_REFERENCE_VIDEO_20STEP,
        help="Original Vivid-VR long-video output used as the Phase D reference.",
    )
    parser.add_argument(
        "--caption-file",
        type=Path,
        default=PHASE_D_130F_CAPTION_FILE,
        help="Caption sidecar extracted from the original Vivid-VR run and replayed by SGLang.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps used by both the SGLang and original benchmarks.",
    )
    parser.add_argument(
        "--wait-for-reference-seconds",
        type=float,
        default=7200.0,
        help="How long to wait for the original Vivid-VR reference video before failing.",
    )
    parser.add_argument(
        "--reference-poll-seconds",
        type=float,
        default=10.0,
        help="Polling interval used while waiting for the original reference video.",
    )
    return apply_phase_d_preset(parser.parse_args())


def wait_for_reference_video(
    *,
    reference_video: Path,
    wait_for_reference_seconds: float,
    reference_poll_seconds: float,
) -> None:
    deadline = time.perf_counter() + max(wait_for_reference_seconds, 0.0)
    poll_seconds = max(reference_poll_seconds, 0.1)
    while not reference_video.exists():
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise SystemExit(
                "Original Vivid-VR reference video is missing after waiting "
                f"{wait_for_reference_seconds} seconds: {reference_video}"
            )
        sleep_seconds = min(poll_seconds, remaining)
        print(
            "[PhaseD] waiting_for_reference "
            f"path={reference_video} "
            f"remaining_seconds={remaining:.1f}"
        )
        time.sleep(sleep_seconds)


def main() -> int:
    total_start_time = time.perf_counter()
    args = parse_args()

    prepared_input = probe_existing_input_video(args.prepared_input_video)
    print(
        "[PhaseD] prepared_input "
        f"path={prepared_input['prepared_video_path']} "
        f"frames={prepared_input['prepared_frame_count']} "
        f"fps={prepared_input['fps']}"
    )
    if not args.caption_file.exists():
        raise SystemExit(
            "Phase D acceptance requires an extracted caption file from the original "
            f"Vivid-VR run: {args.caption_file}"
        )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the Phase D long-video acceptance")

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    INDICATOR_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_prefix = build_artifact_prefix(args)
    candidate_path = RESULT_VIDEOS_DIR / f"{artifact_prefix}_candidate_seed42_{run_id}.mp4"
    report_path = INDICATOR_DIR / f"{artifact_prefix}_metrics_seed42_{run_id}.json"

    server_args = build_server_args()
    pipeline = build_pipeline(server_args)
    request = make_request(
        server_args=server_args,
        input_video_path=args.prepared_input_video,
        output_file_name=candidate_path.name,
        seed=42,
        num_inference_steps=args.num_inference_steps,
        caption_file_path=args.caption_file,
    )

    print(f"[PhaseD] run_id={run_id}")
    print(f"[PhaseD] candidate_video={candidate_path}")
    print(f"[PhaseD] reference_video={args.reference_video}")
    print(f"[PhaseD] caption_file={args.caption_file}")
    print(f"[PhaseD] metrics_report={report_path}")
    model_inference_start_time = time.perf_counter()
    result = pipeline.forward(request, server_args)
    model_inference_runtime_seconds = round(
        time.perf_counter() - model_inference_start_time, 6
    )
    debug = result.extra.get("vividvr_debug", {})
    execution_mode = debug.get("execution_mode")
    if execution_mode != "temporal_windowed":
        raise RuntimeError(
            "Phase D acceptance expected temporal_windowed execution, "
            f"got {execution_mode!r}"
        )

    post_process_sample(
        result.output,
        DataType.VIDEO,
        int(result.fps),
        save_output=True,
        save_file_path=str(candidate_path),
        video_reference_path=str(args.reference_video),
    )

    wait_for_reference_video(
        reference_video=args.reference_video,
        wait_for_reference_seconds=args.wait_for_reference_seconds,
        reference_poll_seconds=args.reference_poll_seconds,
    )
    report = compare_videos(
        str(args.reference_video),
        str(candidate_path),
        min_ssim=0.90,
        max_mse=150.0,
        max_mae=8.0,
        allow_frame_count_delta=1,
        max_failed_frame_ratio=0.05,
    )
    summary = report["summary"]
    ref_frames, _ = load_video_frames(str(args.reference_video))
    cand_frames, _ = load_video_frames(str(candidate_path))
    failed_frame_ratio = (
        len(summary["failed_frames"]) / summary["compared_frames"]
        if summary["compared_frames"] > 0
        else 1.0
    )

    metrics_record = {
        "phase": "D",
        "mode": "temporal_windowed_reference_alignment",
        "preset": args.preset,
        "run_id": run_id,
        "run_datetime_utc": datetime.now(timezone.utc).isoformat(),
        "command": build_recorded_command(),
        "total_runtime_seconds": round(time.perf_counter() - total_start_time, 6),
        "model_inference_runtime_seconds": model_inference_runtime_seconds,
        "seed": 42,
        "num_inference_steps": args.num_inference_steps,
        "prompt_path": str(PROMPT_FILE),
        "caption_file_path": str(args.caption_file),
        "input_video_path": str(args.prepared_input_video),
        "reference_video_path": str(args.reference_video),
        "candidate_video_path": str(candidate_path),
        "reference_frame_count": len(ref_frames),
        "candidate_frame_count": len(cand_frames),
        "frame_count_delta": abs(len(ref_frames) - len(cand_frames)),
        "prepared_input_frame_count": int(prepared_input["prepared_frame_count"]),
        "prepared_input_fps": float(prepared_input["fps"]),
        "prepared_input_height": int(prepared_input["height"]),
        "prepared_input_width": int(prepared_input["width"]),
        "failed_frame_ratio": failed_frame_ratio,
        "summary": summary,
        "frames": report["frames"],
        "debug": debug,
    }
    report_path.write_text(json.dumps(metrics_record, indent=2), encoding="utf-8")

    print(
        "[PhaseD] summary "
        f"pass_compare={summary['pass_compare']} "
        f"ssim_min={summary['ssim_min']:.6f} "
        f"mse_max={summary['mse_max']:.6f} "
        f"mae_max={summary['mae_max']:.6f} "
        f"failed_frame_ratio={failed_frame_ratio:.6f}"
    )
    print(f"[PhaseD] candidate saved to {candidate_path}")
    print(f"[PhaseD] metrics saved to {report_path}")

    return 0 if summary["pass_compare"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
