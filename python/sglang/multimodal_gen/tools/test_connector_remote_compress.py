# SPDX-License-Identifier: Apache-2.0
"""Test Path B: spatial pooling of control states BEFORE all-gather.

用法:
    cd /home/zhiheng/sglang && export PYTHONPATH=python && \
    export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global && \
    export SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=2 && \
    /home/zhiheng/sglang/.venv/bin/torchrun --nproc_per_node=2 --master_port=30101 \
    python/sglang/multimodal_gen/tools/test_connector_ctrl_pool.py
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from sglang.multimodal_gen.configs.pipeline_configs.vividvr import VividVRPipelineConfig
from sglang.multimodal_gen.configs.sample.vividvr import VividVRSamplingParams
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    prepare_request,
    post_process_sample,
)
from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.pipelines_core import build_pipeline
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
from sglang.multimodal_gen.runtime.videoedit.compare import compare_videos
from sglang.multimodal_gen.runtime.videoedit.preprocess import load_video_frames

VIVIDVR_ROOT = Path("/home/zhiheng/Vivid-VR")
COGVIDEOX_ROOT = VIVIDVR_ROOT / "ckpts" / "CogVideoX1.5-5B"
VIVIDVR_CKPT_ROOT = VIVIDVR_ROOT / "ckpts" / "Vivid-VR"
# 720p short video (70f) — Phase C standard
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


def main() -> int:
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    INDICATOR_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    total_start = time.perf_counter()

    mp = int(os.environ.get("MASTER_PORT", "30101"))
    server_args = ServerArgs(
        model_path=str(COGVIDEOX_ROOT),
        pipeline_class_name="CogVideoXVividVRControlNetPipeline",
        pipeline_config=VividVRPipelineConfig(),
        component_paths={"vividvr": str(VIVIDVR_CKPT_ROOT)},
        num_gpus=2,
        tp_size=1,
        dp_size=1,
        dp_degree=1,
        sp_degree=2,
        ulysses_degree=2,
        ring_degree=1,
        dist_timeout=3600,
        master_port=mp,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        enable_torch_compile=True,
        warmup=False,
        output_path=str(RESULT_VIDEOS_DIR),
    )
    server_args._adjust_parameters()
    set_global_server_args(server_args)

    pool_size = os.environ.get("SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE", "1")
    connector_mode = os.environ.get(
        "SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE", "eager_global"
    )

    pipeline = build_pipeline(server_args)

    if dist.get_rank() == 0:
        print(f"[CtrlPool] pool_size={pool_size} connector_mode={connector_mode}")

    params = VividVRSamplingParams.from_user_kwargs(
        server_args,
        prompt=" ",
        video_input_path=str(INPUT_VIDEO),
        prompt_file_path=str(PROMPT_FILE),
        output_path=str(RESULT_VIDEOS_DIR),
        output_file_name=f"ctrl_pool_pool{pool_size}_{run_id}.mp4",
        save_output=False,
        return_file_paths_only=False,
        seed=42,
        num_inference_steps=20,
    )
    request = prepare_request(server_args, params)

    dest = dist.get_rank() == 0

    # Warmup: compile happens here, not timed (matching baseline convention)
    if dest:
        print("[CtrlPool] Running warmup (compile)...")
    warmup_request = request.copy_as_warmup(warmup_steps=1)
    pipeline.forward(warmup_request, server_args)
    dist.barrier()
    if dest:
        print("[CtrlPool] Warmup done, starting timed inference...")

    model_inference_start = time.perf_counter()
    result = pipeline.forward(request, server_args)
    if dest:
        torch.cuda.synchronize()
    model_inference_runtime = round(
        time.perf_counter() - model_inference_start, 6
    )
    dist.barrier()

    if dest:
        print(
            f"[CtrlPool] Model inference time: {model_inference_runtime:.2f}s"
        )

        candidate_path = (
            RESULT_VIDEOS_DIR / f"ctrl_pool_pool{pool_size}_{run_id}.mp4"
        )
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

        metrics = {
            "run_id": run_id,
            "pool_size": int(pool_size),
            "connector_mode": connector_mode,
            "total_runtime_seconds": round(time.perf_counter() - total_start, 6),
            "model_inference_runtime_seconds": model_inference_runtime,
            "ssim_min": summary["ssim_min"],
            "mse_max": summary["mse_max"],
            "mae_max": summary["mae_max"],
            "failed_frame_ratio": failed_frame_ratio,
            "pass_compare": summary["pass_compare"],
        }
        report_path = INDICATOR_DIR / f"ctrl_pool_pool{pool_size}_{run_id}.json"
        report_path.write_text(json.dumps(metrics, indent=2))

        print(
            f"[CtrlPool] SSIM={summary['ssim_min']:.4f} "
            f"inference={model_inference_runtime:.2f}s "
            f"total={time.perf_counter() - total_start:.1f}s "
            f"pool_size={pool_size} "
            f"pass={summary['pass_compare']}"
        )
        print(f"[CtrlPool] Report: {report_path}")

    dist.barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
