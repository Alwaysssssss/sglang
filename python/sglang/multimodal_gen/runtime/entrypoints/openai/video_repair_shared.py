import asyncio
import os
import time
from typing import Any, Dict

from fastapi import HTTPException

from sglang.multimodal_gen.configs.pipeline_configs.vividvr import (
    VividVRPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.sample.vividvr import VividVRSamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoRepairRequest,
)
from sglang.multimodal_gen.runtime.vividvr.caption_bridge import (
    VividVRCaptionBridgeConfig,
    request_vividvr_caption_sidecar,
)
from sglang.multimodal_gen.runtime.vividvr.caption_manifest import (
    build_vividvr_caption_manifest_for_video_path,
)
from sglang.multimodal_gen.runtime.vividvr.preprocess import read_prompt_file
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

VIDEOEDIT_QUEUE_CAPACITY = max(1, int(os.environ.get("VIDEOEDIT_QUEUE_CAPACITY", "1")))
VIDEOEDIT_SEMAPHORE = asyncio.Semaphore(VIDEOEDIT_QUEUE_CAPACITY)


def split_output_path(
    output_path: str | None,
    job_id: str,
    server_output_path: str | None,
) -> tuple[str | None, str]:
    if output_path and os.path.splitext(output_path)[1].lower() == ".mp4":
        return (
            os.path.dirname(os.path.abspath(output_path)),
            os.path.basename(output_path),
        )
    output_dir = output_path or server_output_path
    return output_dir, f"{job_id}.mp4"


def is_vividvr_video_repair_pipeline(server_args) -> bool:
    if isinstance(server_args.pipeline_config, VividVRPipelineConfig):
        return True

    pipeline_class_name = getattr(server_args, "pipeline_class_name", None)
    if pipeline_class_name == "CogVideoXVividVRControlNetPipeline":
        return True

    model_id = getattr(server_args, "model_id", None)
    if isinstance(model_id, str) and model_id.lower().replace("-", "") == "vividvr":
        return True

    return False


def resolve_video_repair_model_name(
    req: VideoRepairRequest, server_args, default_model_name: str
) -> str:
    return req.model or getattr(server_args, "model_id", None) or default_model_name


def _resolve_vividvr_prompt_file_path(server_args) -> str:
    prompt_file_path = getattr(server_args, "prompt_file_path", None) or getattr(
        server_args.pipeline_config, "default_prompt_file_path", None
    )
    if not prompt_file_path:
        raise HTTPException(
            status_code=500,
            detail="vividvr_prompt_file_path is not configured",
        )
    if not os.path.exists(prompt_file_path):
        raise HTTPException(
            status_code=500,
            detail=f"vividvr_prompt_file_path does not exist: {prompt_file_path}",
        )
    return prompt_file_path


def build_vividvr_repair_kwargs(
    *,
    request_id: str,
    req: VideoRepairRequest,
    server_args,
    video_input_path: str,
    output_dir: str,
    output_file_name: str,
) -> Dict[str, Any]:
    vividvr_kwargs = {
        "request_id": request_id,
        "video_input_path": video_input_path,
        "output_path": output_dir,
        "output_file_name": output_file_name,
        "seed": req.seed,
        "dtype": req.dtype,
        "enable_teacache": req.enable_teacache,
        "enable_frame_interpolation": req.enable_frame_interpolation,
        "frame_interpolation_exp": req.frame_interpolation_exp,
        "frame_interpolation_scale": req.frame_interpolation_scale,
        "enable_upscaling": req.enable_upscaling,
        "upscaling_scale": req.upscaling_scale,
        "perf_dump_path": req.perf_dump_path,
    }
    if req.caption_file_path is None:
        vividvr_prompt_file_path = _resolve_vividvr_prompt_file_path(server_args)
        vividvr_kwargs["prompt"] = read_prompt_file(vividvr_prompt_file_path)
        vividvr_kwargs["prompt_file_path"] = vividvr_prompt_file_path
    if req.output_quality not in (None, "default"):
        vividvr_kwargs["output_quality"] = req.output_quality
    if req.negative_prompt is not None:
        vividvr_kwargs["negative_prompt"] = req.negative_prompt
    if req.caption_file_path is not None:
        vividvr_kwargs["caption_source"] = "caption_file"
        vividvr_kwargs["caption_file_path"] = req.caption_file_path
    if req.num_frames is not None:
        vividvr_kwargs["num_frames"] = req.num_frames
    if req.num_inference_steps is not None:
        vividvr_kwargs["num_inference_steps"] = req.num_inference_steps
    if req.guidance_scale is not None:
        vividvr_kwargs["guidance_scale"] = req.guidance_scale
    if req.generator_device is not None:
        vividvr_kwargs["generator_device"] = req.generator_device
    if req.num_temporal_process_frames is not None:
        vividvr_kwargs["num_temporal_process_frames"] = req.num_temporal_process_frames
    if req.restoration_guidance_scale is not None:
        vividvr_kwargs["restoration_guidance_scale"] = req.restoration_guidance_scale
    if getattr(req, "upscale", None) is not None:
        vividvr_kwargs["upscale"] = req.upscale
    if req.frame_interpolation_model_path is not None:
        vividvr_kwargs["frame_interpolation_model_path"] = (
            req.frame_interpolation_model_path
        )
    if req.upscaling_model_path is not None:
        vividvr_kwargs["upscaling_model_path"] = req.upscaling_model_path
    if req.output_compression is not None:
        vividvr_kwargs["output_compression"] = req.output_compression
    return vividvr_kwargs


def _resolve_vividvr_runtime_int(req_value, server_args, attr_name: str) -> int:
    if req_value is not None:
        return int(req_value)
    pipeline_config = getattr(server_args, "pipeline_config", None)
    config_value = getattr(pipeline_config, attr_name, None)
    if config_value is not None:
        return int(config_value)
    return int(getattr(VividVRSamplingParams, attr_name))


async def ensure_vividvr_caption_file(
    *,
    request_id: str,
    req: VideoRepairRequest,
    server_args,
    video_input_path: str,
    output_dir: str,
) -> str | None:
    if req.caption_file_path:
        return req.caption_file_path
    if not getattr(server_args, "vividvr_caption_bridge", False):
        return None

    work_dir = getattr(server_args, "vividvr_caption_work_dir", None)
    if not work_dir:
        work_dir = os.path.join(output_dir, "caption_sidecars")
    os.makedirs(work_dir, exist_ok=True)

    manifest_path = os.path.join(work_dir, f"{request_id}.manifest.json")
    caption_path = os.path.join(work_dir, f"{request_id}.txt")
    num_temporal_process_frames = _resolve_vividvr_runtime_int(
        req.num_temporal_process_frames,
        server_args,
        "num_temporal_process_frames",
    )
    tile_size = _resolve_vividvr_runtime_int(
        getattr(req, "tile_size", None),
        server_args,
        "tile_size",
    )
    tile_stride = _resolve_vividvr_runtime_int(
        getattr(req, "tile_stride", None),
        server_args,
        "tile_stride",
    )

    manifest = build_vividvr_caption_manifest_for_video_path(
        video_path=video_input_path,
        num_temporal_process_frames=num_temporal_process_frames,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )
    manifest.write_json(manifest_path)
    bridge_start = time.perf_counter()
    result = await request_vividvr_caption_sidecar(
        config=VividVRCaptionBridgeConfig(
            enabled=True,
            base_url=getattr(server_args, "vividvr_caption_sidecar_url", None),
            timeout_s=float(
                getattr(server_args, "vividvr_caption_sidecar_timeout", 1800.0)
            ),
        ),
        manifest_path=manifest_path,
        output_caption_path=caption_path,
        expected_caption_count=manifest.expected_caption_count,
    )
    bridge_elapsed_s = time.perf_counter() - bridge_start
    logger.info(
        "VividVR caption bridge generated captions request_id=%s path=%s count=%s "
        "mode=%s worker_count=%s fallback_used=%s total_clip_count=%s bridge_elapsed_s=%.3f "
        "worker_assignments=%s sidecar_request_id=%s sidecar_timing=%s",
        request_id,
        result.caption_file_path,
        result.caption_count,
        getattr(result, "mode", None),
        getattr(result, "worker_count", None),
        getattr(result, "fallback_used", None),
        getattr(result, "total_clip_count", None),
        bridge_elapsed_s,
        getattr(result, "assigned_clip_indices_by_worker", None),
        getattr(result, "request_id", None),
        getattr(result, "timing", None),
    )
    return result.caption_file_path


def copy_video_repair_request_with_caption(
    req: VideoRepairRequest,
    caption_file_path: str | None,
) -> VideoRepairRequest:
    if not caption_file_path:
        return req
    return req.model_copy(update={"caption_file_path": caption_file_path})


def video_repair_job_from_sampling(
    request_id: str, req: VideoRepairRequest, sampling: SamplingParams
) -> Dict[str, Any]:
    return {
        "id": request_id,
        "object": "video",
        "model": req.model or "videoedit",
        "status": "queued",
        "progress": 0,
        "created_at": int(time.time()),
        "size": "",
        "seconds": "",
        "quality": "standard",
        "file_path": os.path.abspath(sampling.output_file_path()),
        "callback_url": req.callback_url,
        "callback_status": None,
        "callback_error": None,
    }
