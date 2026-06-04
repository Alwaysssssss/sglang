# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
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
INPUT_VIDEO = VIVIDVR_ROOT / "input" / "720p" / "test_video_960x720.mp4"
PROMPT_FILE = VIVIDVR_ROOT / "input" / "720p" / "prompt.txt"
REFERENCE_VIDEO = (
    VIVIDVR_ROOT
    / "result"
    / "720p_up1_result_vivid_ori"
    / "videos"
    / "test_video_960x720.mp4"
)
ACCEPTANCE_ROOT = Path("/home/zhiheng/sglang/Vivid_Acceptance")
INDICATOR_DIR = ACCEPTANCE_ROOT / "indicator"
RESULT_VIDEOS_DIR = ACCEPTANCE_ROOT / "result_videos"


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
    output_file_name: str,
    seed: int,
):
    params = VividVRSamplingParams.from_user_kwargs(
        server_args,
        prompt=" ",
        video_input_path=str(INPUT_VIDEO),
        prompt_file_path=str(PROMPT_FILE),
        output_path=str(RESULT_VIDEOS_DIR),
        output_file_name=output_file_name,
        save_output=False,
        return_file_paths_only=False,
        seed=seed,
    )
    return prepare_request(server_args, params)


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the Phase C single-run acceptance")

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    INDICATOR_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate_path = RESULT_VIDEOS_DIR / f"phase_c_candidate_seed42_{run_id}.mp4"
    report_path = INDICATOR_DIR / f"phase_c_metrics_seed42_{run_id}.json"

    server_args = build_server_args()
    pipeline = build_pipeline(server_args)
    request = make_request(
        server_args=server_args,
        output_file_name=candidate_path.name,
        seed=42,
    )

    print(f"[PhaseC] run_id={run_id}")
    print(f"[PhaseC] candidate_video={candidate_path}")
    print(f"[PhaseC] metrics_report={report_path}")
    result = pipeline.forward(request, server_args)

    post_process_sample(
        result.output,
        DataType.VIDEO,
        int(result.fps),
        save_output=True,
        save_file_path=str(candidate_path),
        video_reference_path=str(REFERENCE_VIDEO),
    )

    report = compare_videos(
        str(REFERENCE_VIDEO),
        str(candidate_path),
        min_ssim=0.90,
        max_mse=150.0,
        max_mae=8.0,
        allow_frame_count_delta=1,
        max_failed_frame_ratio=0.05,
    )
    summary = report["summary"]
    ref_frames, _ = load_video_frames(str(REFERENCE_VIDEO))
    cand_frames, _ = load_video_frames(str(candidate_path))
    failed_frame_ratio = (
        len(summary["failed_frames"]) / summary["compared_frames"]
        if summary["compared_frames"] > 0
        else 1.0
    )

    metrics_record = {
        "phase": "C",
        "mode": "single_run_reference_alignment",
        "run_id": run_id,
        "run_datetime_utc": datetime.now(timezone.utc).isoformat(),
        "command": "PYTHONPATH=python uv run --with pytest --with diffusers==0.37.0 "
        "--with imageio==2.36.0 --with imageio-ffmpeg==0.5.1 --with addict==2.4.0 "
        "--with PyYAML==6.0.1 --with av==16.1.0 --with scikit-image==0.25.2 "
        "--with cache-dit==1.3.0 --with opencv-python-headless==4.10.0.84 "
        "--with trimesh python python/sglang/multimodal_gen/tools/run_vividvr_phase_c_single.py",
        "seed": 42,
        "prompt_path": str(PROMPT_FILE),
        "input_video_path": str(INPUT_VIDEO),
        "reference_video_path": str(REFERENCE_VIDEO),
        "candidate_video_path": str(candidate_path),
        "reference_frame_count": len(ref_frames),
        "candidate_frame_count": len(cand_frames),
        "frame_count_delta": abs(len(ref_frames) - len(cand_frames)),
        "failed_frame_ratio": failed_frame_ratio,
        "summary": summary,
        "frames": report["frames"],
        "debug": result.extra.get("vividvr_debug", {}),
    }
    report_path.write_text(json.dumps(metrics_record, indent=2), encoding="utf-8")

    print(
        "[PhaseC] summary "
        f"pass_compare={summary['pass_compare']} "
        f"ssim_min={summary['ssim_min']:.6f} "
        f"mse_max={summary['mse_max']:.6f} "
        f"mae_max={summary['mae_max']:.6f} "
        f"failed_frame_ratio={failed_frame_ratio:.6f}"
    )
    print(f"[PhaseC] candidate saved to {candidate_path}")
    print(f"[PhaseC] metrics saved to {report_path}")

    return 0 if summary["pass_compare"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
