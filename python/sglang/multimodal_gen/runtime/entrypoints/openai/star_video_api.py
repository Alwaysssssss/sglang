# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import asyncio
import json
import os
import shutil
import tempfile
import time
from typing import Any, Dict, Optional

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse

from sglang.multimodal_gen.configs.sample.sampling_params import (
    SamplingParams,
    generate_request_id,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    StarVideoSRRequest,
    VideoListResponse,
    VideoResponse,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.storage import cloud_storage
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import STAR_VIDEO_STORE
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    add_common_data_to_response,
    build_sampling_params,
    process_generation_batch,
    save_video_to_path,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.observability.trace import extract_trace_headers

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/star/videos", tags=["star-videos"])

STAR_DEFAULT_OUTPUT_QUALITY = "maximum"
STAR_DEFAULT_FPS = 8
STAR_DEFAULT_NUM_FRAMES = 7
STAR_DEFAULT_CONDITION_VIDEO_NUM_FRAMES = 25
STAR_DEFAULT_GUIDANCE_SCALE = 6.0
STAR_DEFAULT_NUM_INFERENCE_STEPS = 50


def _ensure_star_pipeline() -> None:
    server_args = get_global_server_args()
    pipeline_name = getattr(server_args, "pipeline_class_name", None) or type(
        server_args.pipeline_config
    ).__name__
    if (
        pipeline_name != "StarCogVideoXSRPipeline"
        and type(server_args.pipeline_config).__name__ != "StarCogVideoXSRPipelineConfig"
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Current served model is not STAR CogVideoX-SR. "
                "Please launch the server with --pipeline-class-name StarCogVideoXSRPipeline."
            ),
        )


def _build_star_video_sampling_params(
    request_id: str,
    request: StarVideoSRRequest,
) -> SamplingParams:
    return build_sampling_params(
        request_id,
        prompt=request.prompt,
        width=request.width,
        height=request.height,
        num_frames=(
            request.num_frames
            if request.num_frames is not None
            else STAR_DEFAULT_NUM_FRAMES
        ),
        fps=request.fps if request.fps is not None else STAR_DEFAULT_FPS,
        output_file_name=request_id,
        seed=request.seed,
        generator_device=request.generator_device,
        num_inference_steps=(
            request.num_inference_steps
            if request.num_inference_steps is not None
            else STAR_DEFAULT_NUM_INFERENCE_STEPS
        ),
        guidance_scale=(
            request.guidance_scale
            if request.guidance_scale is not None
            else STAR_DEFAULT_GUIDANCE_SCALE
        ),
        negative_prompt=request.negative_prompt,
        enable_teacache=request.enable_teacache,
        output_path=request.output_path,
        output_compression=request.output_compression,
        output_quality=request.output_quality or STAR_DEFAULT_OUTPUT_QUALITY,
        perf_dump_path=request.perf_dump_path,
        condition_video_path=request.condition_video_path,
        condition_video_start_frame=request.condition_video_start_frame,
        condition_video_num_frames=(
            request.condition_video_num_frames
            if request.condition_video_num_frames is not None
            else STAR_DEFAULT_CONDITION_VIDEO_NUM_FRAMES
        ),
        condition_video_sample_fps=request.condition_video_sample_fps,
        condition_video_frame_stride=request.condition_video_frame_stride,
    )


def _star_video_job_from_sampling(
    request_id: str,
    req: StarVideoSRRequest,
    sampling: SamplingParams,
) -> Dict[str, Any]:
    size_str = f"{sampling.width}x{sampling.height}"
    seconds = int(round((sampling.num_frames or 0) / float(sampling.fps or 24)))
    return {
        "id": request_id,
        "object": "video",
        "model": req.model or "star-cogvideox-sr",
        "status": "queued",
        "progress": 0,
        "created_at": int(time.time()),
        "size": size_str,
        "seconds": str(seconds),
        "quality": "standard",
        "file_path": os.path.abspath(sampling.output_file_path()),
    }


async def _dispatch_star_job_async(
    job_id: str,
    batch: Req,
    *,
    temp_dirs: list[str] | None = None,
    output_persistent: bool = True,
) -> None:
    from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client

    try:
        save_file_path_list, result = await process_generation_batch(
            async_scheduler_client, batch
        )
        save_file_path = save_file_path_list[0]

        cloud_url = await cloud_storage.upload_and_cleanup(save_file_path)
        persistent_path = (
            save_file_path if not cloud_url and output_persistent else None
        )
        update_fields = {
            "status": "completed",
            "progress": 100,
            "completed_at": int(time.time()),
            "url": cloud_url,
            "file_path": persistent_path,
        }
        update_fields = add_common_data_to_response(
            update_fields, request_id=job_id, result=result
        )
        await STAR_VIDEO_STORE.update_fields(job_id, update_fields)
    except Exception as e:
        logger.error("%s", e, exc_info=True)
        await STAR_VIDEO_STORE.update_fields(
            job_id, {"status": "failed", "error": {"message": str(e)}}
        )
    finally:
        for td in temp_dirs or []:
            shutil.rmtree(td, ignore_errors=True)


@router.post("", response_model=VideoResponse)
async def create_star_video(
    request: Request,
    prompt: Optional[str] = Form(None),
    condition_video: Optional[UploadFile] = File(None),
    condition_video_url: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    seed: Optional[int] = Form(1024),
    generator_device: Optional[str] = Form("cuda"),
    width: Optional[int] = Form(STAR_DEFAULT_FPS * 90),
    height: Optional[int] = Form(STAR_DEFAULT_FPS * 60),
    fps: Optional[int] = Form(STAR_DEFAULT_FPS),
    num_frames: Optional[int] = Form(STAR_DEFAULT_NUM_FRAMES),
    condition_video_start_frame: Optional[int] = Form(None),
    condition_video_num_frames: Optional[int] = Form(
        STAR_DEFAULT_CONDITION_VIDEO_NUM_FRAMES
    ),
    condition_video_sample_fps: Optional[int] = Form(None),
    condition_video_frame_stride: Optional[int] = Form(None),
    num_inference_steps: Optional[int] = Form(STAR_DEFAULT_NUM_INFERENCE_STEPS),
    guidance_scale: Optional[float] = Form(STAR_DEFAULT_GUIDANCE_SCALE),
    negative_prompt: Optional[str] = Form(""),
    enable_teacache: Optional[bool] = Form(False),
    output_quality: Optional[str] = Form(STAR_DEFAULT_OUTPUT_QUALITY),
    output_compression: Optional[int] = Form(None),
    output_path: Optional[str] = Form(None),
    extra_body: Optional[str] = Form(None),
):
    _ensure_star_pipeline()
    content_type = request.headers.get("content-type", "").lower()
    request_id = generate_request_id()
    server_args = get_global_server_args()

    temp_dirs: list[str] = []
    if server_args.input_save_path is not None:
        uploads_dir = server_args.input_save_path
        os.makedirs(uploads_dir, exist_ok=True)
    else:
        uploads_dir = tempfile.mkdtemp(prefix="sglang_star_input_")
        temp_dirs.append(uploads_dir)

    if "multipart/form-data" in content_type:
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")
        if condition_video is None and not condition_video_url:
            raise HTTPException(
                status_code=400,
                detail="condition_video or condition_video_url is required",
            )

        target_path = os.path.join(uploads_dir, f"{request_id}_condition_video")
        try:
            resolved_condition_video_path = await save_video_to_path(
                condition_video if condition_video is not None else condition_video_url,
                target_path,
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process condition video: {str(e)}",
            )

        extra_from_form: Dict[str, Any] = {}
        if extra_body:
            try:
                extra_from_form = json.loads(extra_body)
            except Exception:
                extra_from_form = {}

        req = StarVideoSRRequest(
            prompt=prompt,
            condition_video_path=resolved_condition_video_path,
            model=model,
            seed=seed,
            generator_device=generator_device,
            width=width if width is not None else extra_from_form.get("width", 720),
            height=height
            if height is not None
            else extra_from_form.get("height", 480),
            fps=fps if fps is not None else extra_from_form.get("fps", STAR_DEFAULT_FPS),
            num_frames=(
                num_frames
                if num_frames is not None
                else extra_from_form.get("num_frames", STAR_DEFAULT_NUM_FRAMES)
            ),
            condition_video_start_frame=(
                condition_video_start_frame
                if condition_video_start_frame is not None
                else extra_from_form.get("condition_video_start_frame")
            ),
            condition_video_num_frames=(
                condition_video_num_frames
                if condition_video_num_frames is not None
                else extra_from_form.get(
                    "condition_video_num_frames",
                    STAR_DEFAULT_CONDITION_VIDEO_NUM_FRAMES,
                )
            ),
            condition_video_sample_fps=(
                condition_video_sample_fps
                if condition_video_sample_fps is not None
                else extra_from_form.get("condition_video_sample_fps")
            ),
            condition_video_frame_stride=(
                condition_video_frame_stride
                if condition_video_frame_stride is not None
                else extra_from_form.get("condition_video_frame_stride")
            ),
            num_inference_steps=(
                num_inference_steps
                if num_inference_steps is not None
                else extra_from_form.get(
                    "num_inference_steps",
                    STAR_DEFAULT_NUM_INFERENCE_STEPS,
                )
            ),
            guidance_scale=(
                guidance_scale
                if guidance_scale is not None
                else extra_from_form.get(
                    "guidance_scale",
                    STAR_DEFAULT_GUIDANCE_SCALE,
                )
            ),
            negative_prompt=negative_prompt,
            enable_teacache=enable_teacache,
            output_quality=output_quality,
            output_compression=output_compression,
            output_path=output_path,
        )
    else:
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        try:
            merged_payload: Dict[str, Any] = dict(payload or {})
            extra = merged_payload.pop("extra_body", None)
            if isinstance(extra, dict):
                merged_payload.update(extra)
            extra_json = merged_payload.pop("extra_json", None)
            if isinstance(extra_json, dict):
                merged_payload.update(extra_json)

            if not merged_payload.get("condition_video_path") and not merged_payload.get(
                "condition_video_url"
            ):
                raise HTTPException(
                    status_code=400,
                    detail="condition_video_path or condition_video_url is required",
                )

            condition_video_source = merged_payload.get("condition_video_path")
            if not condition_video_source and merged_payload.get("condition_video_url"):
                target_path = os.path.join(uploads_dir, f"{request_id}_condition_video")
                condition_video_source = await save_video_to_path(
                    merged_payload["condition_video_url"],
                    target_path,
                )
                merged_payload["condition_video_path"] = condition_video_source

            req = StarVideoSRRequest(**merged_payload)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    effective_output_path = req.output_path or server_args.output_path
    output_persistent = True
    if effective_output_path is None:
        output_tmp = tempfile.mkdtemp(prefix="sglang_star_output_")
        temp_dirs.append(output_tmp)
        effective_output_path = output_tmp
        output_persistent = False
    req.output_path = effective_output_path

    try:
        sampling_params = _build_star_video_sampling_params(request_id, req)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = _star_video_job_from_sampling(request_id, req, sampling_params)
    await STAR_VIDEO_STORE.upsert(request_id, job)

    trace_headers = extract_trace_headers(request.headers)
    batch = prepare_request(
        server_args=server_args,
        sampling_params=sampling_params,
        external_trace_header=trace_headers,
    )
    asyncio.create_task(
        _dispatch_star_job_async(
            request_id,
            batch,
            temp_dirs=temp_dirs or None,
            output_persistent=output_persistent,
        )
    )
    return VideoResponse(**job)


@router.get("", response_model=VideoListResponse)
async def list_star_videos(
    after: Optional[str] = Query(None),
    limit: Optional[int] = Query(None, ge=1, le=100),
    order: Optional[str] = Query("desc"),
):
    order = (order or "desc").lower()
    if order not in ("asc", "desc"):
        order = "desc"
    jobs = await STAR_VIDEO_STORE.list_values()
    reverse = order != "asc"
    jobs.sort(key=lambda j: j.get("created_at", 0), reverse=reverse)

    if after is not None:
        try:
            idx = next(i for i, j in enumerate(jobs) if j["id"] == after)
            jobs = jobs[idx + 1 :]
        except StopIteration:
            jobs = []

    if limit is not None:
        jobs = jobs[:limit]
    return VideoListResponse(data=[VideoResponse(**j) for j in jobs])


@router.get("/{video_id}", response_model=VideoResponse)
async def retrieve_star_video(video_id: str = Path(...)):
    job = await STAR_VIDEO_STORE.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="STAR video not found")
    return VideoResponse(**job)


@router.delete("/{video_id}", response_model=VideoResponse)
async def delete_star_video(video_id: str = Path(...)):
    job = await STAR_VIDEO_STORE.pop(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="STAR video not found")
    job["status"] = "deleted"
    return VideoResponse(**job)


@router.get("/{video_id}/content")
async def download_star_video_content(video_id: str = Path(...)):
    job = await STAR_VIDEO_STORE.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="STAR video not found")
    if job.get("url"):
        raise HTTPException(
            status_code=400,
            detail=(
                "STAR video has been uploaded to cloud storage. "
                f"Please use the cloud URL: {job.get('url')}"
            ),
        )
    file_path = job.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Generation is still in-progress")
    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=os.path.basename(file_path),
    )
