# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import asyncio
import json
import os
import shutil
import tempfile
import time
from typing import Any, Callable, Dict, Optional

import httpx
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
from pydantic import ValidationError

from sglang.multimodal_gen.configs.sample.sampling_params import (
    SamplingParams,
    generate_request_id,
)
from sglang.multimodal_gen.configs.sample.videoedit_wan import (
    WanVideoEditSamplingParams,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
    VideoListResponse,
    VideoRepairRequest,
    VideoResponse,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.storage import (
    RequestCloudStorage,
    cloud_storage,
    normalize_object_key,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import VIDEO_STORE
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    DEFAULT_FPS,
    DEFAULT_VIDEO_SECONDS,
    add_common_data_to_response,
    build_sampling_params,
    merge_image_input_list,
    process_generation_batch,
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.videoedit.preprocess import (
    resolve_videoedit_num_frames,
)
from sglang.multimodal_gen.runtime.videoedit.progress import read_videoedit_progress

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/videos", tags=["videos"])

_VIDEOEDIT_QUEUE_CAPACITY = max(1, int(os.environ.get("VIDEOEDIT_QUEUE_CAPACITY", "1")))
_VIDEOEDIT_SEMAPHORE = asyncio.Semaphore(_VIDEOEDIT_QUEUE_CAPACITY)

CallbackPayloadBuilder = Callable[[str, Dict[str, Any]], Dict[str, Any]]

_VIDEO_REPAIR_FIELD_ALIASES = {
    "taskId": "task_id",
    "callbackUrl": "callback_url",
    "videoUrl": "video_url",
    "maskUrl": "mask_url",
    "referenceImageUrl": "reference_image_url",
    "decodeMode": "decode_mode",
    "minioConfig": "minio_config",
    "outputObjectKey": "output_object_key",
}

_VIDEO_REPAIR_MINIO_FIELD_ALIASES = {
    "bucketName": "bucket_name",
    "accessKey": "access_key",
    "secretKey": "secret_key",
}


def _video_repair_submit_response(code: int, message: str) -> Dict[str, Any]:
    return {"code": int(code), "message": str(message)}


def _normalize_aliases(
    payload: Dict[str, Any], aliases: Dict[str, str]
) -> Dict[str, Any]:
    normalized = dict(payload)
    for alias, canonical in aliases.items():
        if alias not in payload:
            continue
        normalized[canonical] = payload[alias]
        if alias != canonical:
            normalized.pop(alias, None)
    return normalized


def _normalize_video_repair_payload(body: Any) -> Dict[str, Any]:
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    payload = _normalize_aliases(body, _VIDEO_REPAIR_FIELD_ALIASES)
    minio_config = payload.get("minio_config")
    if minio_config is not None:
        if not isinstance(minio_config, dict):
            raise ValueError("minioConfig must be a JSON object")
        payload["minio_config"] = _normalize_aliases(
            minio_config, _VIDEO_REPAIR_MINIO_FIELD_ALIASES
        )
    return payload


def _exception_message(e: Exception) -> str:
    if isinstance(e, HTTPException):
        return str(e.detail)
    if isinstance(e, ValidationError):
        messages = []
        for error in e.errors():
            loc = ".".join(str(part) for part in error.get("loc", ()))
            msg = error.get("msg", "invalid value")
            messages.append(f"{loc}: {msg}" if loc else msg)
        return "; ".join(messages) or "invalid request body"
    return str(e)


def _job_error_message(job: Dict[str, Any]) -> str:
    error = job.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if message:
            return str(message)
    if isinstance(error, str):
        return error
    return "failed"


def _build_video_repair_callback_payload(
    task_id: str, job: Dict[str, Any]
) -> Dict[str, Any]:
    status = job.get("status")
    if status == "completed":
        return {
            "taskId": task_id,
            "status": "completed",
            "outputUrl": job.get("url"),
            "message": "ok",
        }
    return {
        "taskId": task_id,
        "status": "failed",
        "outputUrl": None,
        "message": _job_error_message(job),
    }


def _validate_video_repair_request(req: VideoRepairRequest) -> None:
    if not req.task_id:
        raise ValueError("taskId is required")
    if not req.callback_url:
        raise ValueError("callbackUrl is required")
    if req.timeout == 0 or req.timeout < -1:
        raise ValueError("timeout must be positive or -1")
    if not (req.video_input_path or req.video_url):
        raise ValueError("videoUrl or video_input_path is required")
    if not (req.mask_input_path or req.mask_url):
        raise ValueError("maskUrl or mask_input_path is required")
    if req.minio_config is None and (
        req.output_storage == "s3" or req.output_object_key is not None
    ):
        if not cloud_storage.is_enabled():
            raise ValueError("minioConfig is required for S3 output")


def _build_video_sampling_params(request_id: str, request: VideoGenerationsRequest):
    """Resolve video-specific defaults (fps, seconds → num_frames) then
    delegate to the shared build_sampling_params."""
    seconds = request.seconds if request.seconds is not None else DEFAULT_VIDEO_SECONDS
    fps = request.fps if request.fps is not None else DEFAULT_FPS
    num_frames = request.num_frames if request.num_frames is not None else fps * seconds

    return build_sampling_params(
        request_id,
        prompt=request.prompt,
        size=request.size,
        width=request.width,
        height=request.height,
        num_frames=num_frames,
        fps=fps,
        image_path=request.input_reference,
        output_file_name=request_id,
        seed=request.seed,
        generator_device=request.generator_device,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        guidance_scale_2=request.guidance_scale_2,
        negative_prompt=request.negative_prompt,
        enable_teacache=request.enable_teacache,
        enable_frame_interpolation=request.enable_frame_interpolation,
        frame_interpolation_exp=request.frame_interpolation_exp,
        frame_interpolation_scale=request.frame_interpolation_scale,
        frame_interpolation_model_path=request.frame_interpolation_model_path,
        enable_upscaling=request.enable_upscaling,
        upscaling_model_path=request.upscaling_model_path,
        upscaling_scale=request.upscaling_scale,
        output_path=request.output_path,
        output_compression=request.output_compression,
        output_quality=request.output_quality,
        perf_dump_path=request.perf_dump_path,
    )


# extract metadata which http_server needs to know
def _video_job_from_sampling(
    request_id: str, req: VideoGenerationsRequest, sampling: SamplingParams
) -> Dict[str, Any]:
    size_str = f"{sampling.width}x{sampling.height}"
    seconds = int(round((sampling.num_frames or 0) / float(sampling.fps or 24)))
    return {
        "id": request_id,
        "object": "video",
        "model": req.model or "sora-2",
        "status": "queued",
        "progress": 0,
        "created_at": int(time.time()),
        "size": size_str,
        "seconds": str(seconds),
        "quality": "standard",
        "file_path": os.path.abspath(sampling.output_file_path()),
    }


async def _save_first_input_image(
    image_sources, request_id: str, uploads_dir: str
) -> str | None:
    """Save the first input image from a list of sources and return its path."""
    image_list = merge_image_input_list(image_sources)
    if not image_list:
        return None
    image = image_list[0]

    os.makedirs(uploads_dir, exist_ok=True)

    filename = image.filename if hasattr(image, "filename") else "url_image"
    target_path = os.path.join(uploads_dir, f"{request_id}_{filename}")
    return await save_image_to_path(image, target_path)


def _build_video_callback_payload(
    video_id: str, job: Dict[str, Any]
) -> Dict[str, Any]:
    payload = {
        "id": video_id,
        "object": job.get("object", "video"),
        "model": job.get("model"),
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "created_at": job.get("created_at"),
        "completed_at": job.get("completed_at"),
        "file_path": job.get("file_path"),
        "url": job.get("url"),
        "error": job.get("error"),
    }
    for key in ("peak_memory_mb", "inference_time_s"):
        if key in job:
            payload[key] = job[key]
    return payload


async def _post_video_callback(
    job_id: str,
    callback_url: str | None,
    payload: Dict[str, Any],
    *,
    timeout: float = 10.0,
    max_retries: int = 3,
) -> None:
    if not callback_url:
        return

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient(
                follow_redirects=True, timeout=timeout
            ) as client:
                response = await client.post(callback_url, json=payload)
                response.raise_for_status()
            await VIDEO_STORE.update_fields(
                job_id,
                {
                    "callback_status": "succeeded",
                    "callback_error": None,
                    "callback_attempts": attempt,
                    "callback_completed_at": int(time.time()),
                },
            )
            return
        except Exception as e:
            last_error = str(e)
            logger.warning(
                "Video callback failed for job=%s attempt=%s/%s url=%s: %s",
                job_id,
                attempt,
                max_retries,
                callback_url,
                last_error,
            )
            if attempt < max_retries:
                await asyncio.sleep(min(2 ** (attempt - 1), 5))

    await VIDEO_STORE.update_fields(
        job_id,
        {
            "callback_status": "failed",
            "callback_error": last_error,
            "callback_attempts": max_retries,
            "callback_completed_at": int(time.time()),
        },
    )


async def _dispatch_job_async(
    job_id: str,
    batch: Req,
    *,
    temp_dirs: list[str] | None = None,
    output_persistent: bool = True,
    callback_url: str | None = None,
    callback_payload_builder: CallbackPayloadBuilder = _build_video_callback_payload,
    request_storage: RequestCloudStorage | None = None,
    output_object_key: str | None = None,
    output_bucket: str | None = None,
) -> None:
    from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client

    try:
        save_file_path_list, result = await process_generation_batch(
            async_scheduler_client, batch
        )
        save_file_path = save_file_path_list[0]

        if request_storage is not None:
            destination_key = output_object_key or os.path.basename(save_file_path)
            cloud_url = await request_storage.upload_and_cleanup(
                save_file_path,
                destination_key,
                bucket_name=output_bucket,
            )
        else:
            cloud_url = await cloud_storage.upload_and_cleanup(
                save_file_path,
                destination_key=output_object_key,
                bucket_name=output_bucket,
            )

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
        await VIDEO_STORE.update_fields(job_id, update_fields)
        job = await VIDEO_STORE.get(job_id)
        if job and callback_url:
            asyncio.create_task(
                _post_video_callback(
                    job_id,
                    callback_url,
                    callback_payload_builder(job_id, job),
                )
            )
    except Exception as e:
        logger.error(f"{e}")
        await VIDEO_STORE.update_fields(
            job_id, {"status": "failed", "error": {"message": str(e)}}
        )
        job = await VIDEO_STORE.get(job_id)
        if job and callback_url:
            asyncio.create_task(
                _post_video_callback(
                    job_id,
                    callback_url,
                    callback_payload_builder(job_id, job),
                )
            )
    finally:
        for td in temp_dirs or []:
            shutil.rmtree(td, ignore_errors=True)


async def _save_video_source_to_path(
    source: str, target_path: str, *, default_ext: str = ".mp4"
) -> str:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if source.lower().startswith(("http://", "https://")):
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(source, timeout=60.0)
            response.raise_for_status()
        if not os.path.splitext(target_path)[1]:
            _, ext = os.path.splitext(source.split("?", 1)[0])
            target_path = f"{target_path}{ext or default_ext}"
        with open(target_path, "wb") as f:
            f.write(response.content)
        return target_path

    if not os.path.exists(source):
        raise FileNotFoundError(f"Input video path does not exist: {source}")
    if os.path.abspath(source) == os.path.abspath(target_path):
        return source
    if not os.path.splitext(target_path)[1]:
        _, ext = os.path.splitext(source)
        target_path = f"{target_path}{ext or default_ext}"
    shutil.copyfile(source, target_path)
    return target_path


def _split_output_path(
    output_path: str | None, job_id: str, server_output_path: str | None
):
    if output_path and os.path.splitext(output_path)[1].lower() == ".mp4":
        return os.path.dirname(os.path.abspath(output_path)), os.path.basename(
            output_path
        )
    output_dir = output_path or server_output_path
    return output_dir, f"{job_id}.mp4"


def _current_video_progress(job: Dict[str, Any]) -> int:
    progress_payload = read_videoedit_progress(job.get("progress_path"))
    progress = progress_payload.get("progress", job.get("progress", 0))
    if job.get("status") == "completed":
        progress = 100
    return int(progress)


def _video_repair_job_from_sampling(
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
        "timeout": req.timeout,
        "output_object_key": req.output_object_key,
    }


async def _dispatch_video_repair_job_async(
    job_id: str,
    batch: Req,
    *,
    temp_dirs: list[str] | None = None,
    output_persistent: bool = True,
    callback_url: str | None = None,
    request_storage: RequestCloudStorage | None = None,
    output_object_key: str | None = None,
    output_bucket: str | None = None,
    timeout: int = 300,
) -> None:
    try:
        await VIDEO_STORE.update_fields(job_id, {"status": "running", "progress": 1})
        dispatch_coro = _dispatch_job_async(
            job_id,
            batch,
            temp_dirs=None,
            output_persistent=output_persistent,
            callback_url=callback_url,
            callback_payload_builder=_build_video_repair_callback_payload,
            request_storage=request_storage,
            output_object_key=output_object_key,
            output_bucket=output_bucket,
        )
        if timeout == -1:
            await dispatch_coro
        else:
            await asyncio.wait_for(dispatch_coro, timeout=timeout)
    except asyncio.TimeoutError:
        await VIDEO_STORE.update_fields(
            job_id,
            {
                "status": "failed",
                "error": {"message": "task timeout"},
            },
        )
        job = await VIDEO_STORE.get(job_id)
        if job and callback_url:
            asyncio.create_task(
                _post_video_callback(
                    job_id,
                    callback_url,
                    _build_video_repair_callback_payload(job_id, job),
                )
            )
    finally:
        _VIDEOEDIT_SEMAPHORE.release()
        for td in temp_dirs or []:
            shutil.rmtree(td, ignore_errors=True)


@router.post("/repairs")
async def create_video_repair(request: Request):
    if _VIDEOEDIT_SEMAPHORE.locked():
        return _video_repair_submit_response(2, "A task is running.")

    try:
        body = await request.json()
        payload = _normalize_video_repair_payload(body)
        req = VideoRepairRequest(**payload)
        _validate_video_repair_request(req)
    except Exception as e:
        return _video_repair_submit_response(
            1, f"Invalid request body: {_exception_message(e)}"
        )

    if _VIDEOEDIT_SEMAPHORE.locked():
        return _video_repair_submit_response(2, "A task is running.")

    await _VIDEOEDIT_SEMAPHORE.acquire()

    server_args = get_global_server_args()
    request_id = req.task_id or generate_request_id()
    temp_dirs: list[str] = []

    try:
        request_storage = (
            RequestCloudStorage(req.minio_config)
            if req.minio_config is not None
            else None
        )
        uploads_dir = server_args.input_save_path
        if uploads_dir is None:
            uploads_dir = tempfile.mkdtemp(prefix="sglang_videoedit_input_")
            temp_dirs.append(uploads_dir)
        os.makedirs(uploads_dir, exist_ok=True)

        video_input_path = req.video_input_path
        mask_input_path = req.mask_input_path
        reference_image_path = None
        if req.video_url:
            target_path = os.path.join(uploads_dir, f"{request_id}_video")
            if request_storage is not None:
                video_input_path = await request_storage.download_source(
                    req.video_url, target_path, default_ext=".mp4"
                )
            else:
                video_input_path = await _save_video_source_to_path(
                    req.video_url, target_path, default_ext=".mp4"
                )
        if req.mask_url:
            target_path = os.path.join(uploads_dir, f"{request_id}_mask")
            if request_storage is not None:
                mask_input_path = await request_storage.download_source(
                    req.mask_url, target_path, default_ext=".mp4"
                )
            else:
                mask_input_path = await _save_video_source_to_path(
                    req.mask_url, target_path, default_ext=".mp4"
                )
        if req.reference_image_url:
            target_path = os.path.join(uploads_dir, f"{request_id}_reference")
            if request_storage is not None:
                reference_image_path = await request_storage.download_source(
                    req.reference_image_url, target_path, default_ext=".png"
                )
            else:
                reference_image_path = await _save_video_source_to_path(
                    req.reference_image_url, target_path, default_ext=".png"
                )
        if not video_input_path:
            raise ValueError("videoUrl or video_input_path is required")
        if not mask_input_path:
            raise ValueError("maskUrl or mask_input_path is required")

        resolved_num_frames = resolve_videoedit_num_frames(
            req.num_frames,
            video_input_path,
            mask_input_path,
        )
        has_reference_image = bool(reference_image_path)
        effective_drop_reference_frame = (
            req.drop_reference_frame
            if req.drop_reference_frame is not None
            else has_reference_image
        )

        output_dir, output_file_name = _split_output_path(
            req.output_path, request_id, server_args.output_path
        )
        output_persistent = output_dir is not None
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="sglang_videoedit_output_")
            temp_dirs.append(output_dir)
            output_persistent = False

        progress_path = os.path.join(output_dir, f"{request_id}.progress.json")

        sampling_params = WanVideoEditSamplingParams.from_user_kwargs(
            server_args,
            request_id=request_id,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            video_input_path=video_input_path,
            mask_input_path=mask_input_path,
            reference_image_path=reference_image_path,
            output_path=output_dir,
            output_file_name=output_file_name,
            num_frames=resolved_num_frames,
            infer_len=req.infer_len,
            overlap=req.overlap,
            strength=req.strength,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            generator_device=req.generator_device,
            dtype=req.dtype,
            dynamic_cfg=req.dynamic_cfg,
            dynamic_cfg_max_step=req.dynamic_cfg_max_step,
            dynamic_cfg_min=req.dynamic_cfg_min,
            bbox_padding=req.bbox_padding,
            dilate_px=req.dilate_px,
            mask_scale=req.mask_scale,
            feather_px=req.feather_px,
            adain_boundary_dilate=req.adain_boundary_dilate,
            enable_paste_back=req.enable_paste_back,
            save_crop_only=req.save_crop_only,
            drop_reference_frame=effective_drop_reference_frame,
            keep_intermediate_windows=req.keep_intermediate_windows,
            use_repaired_context=req.use_repaired_context,
            vary_seed_by_window=req.vary_seed_by_window,
            decode_mode=req.decode_mode,
            enable_teacache=req.enable_teacache,
            enable_frame_interpolation=req.enable_frame_interpolation,
            frame_interpolation_exp=req.frame_interpolation_exp,
            frame_interpolation_scale=req.frame_interpolation_scale,
            frame_interpolation_model_path=req.frame_interpolation_model_path,
            enable_upscaling=req.enable_upscaling,
            upscaling_model_path=req.upscaling_model_path,
            upscaling_scale=req.upscaling_scale,
            output_quality=req.output_quality,
            output_compression=req.output_compression,
            perf_dump_path=req.perf_dump_path,
            progress_path=progress_path,
        )
        output_object_key = None
        if (
            request_storage is not None
            or req.output_storage == "s3"
            or req.output_object_key is not None
        ):
            output_object_key = normalize_object_key(
                req.output_object_key or f"{request_id}.mp4"
            )
        req.output_object_key = output_object_key
        job = _video_repair_job_from_sampling(request_id, req, sampling_params)
        job["progress_path"] = progress_path
        await VIDEO_STORE.upsert(request_id, job)
        batch = prepare_request(
            server_args=server_args, sampling_params=sampling_params
        )
        asyncio.create_task(
            _dispatch_video_repair_job_async(
                request_id,
                batch,
                temp_dirs=temp_dirs or None,
                output_persistent=output_persistent,
                callback_url=req.callback_url,
                request_storage=request_storage,
                output_object_key=output_object_key,
                output_bucket=req.output_bucket,
                timeout=req.timeout,
            )
        )
        return _video_repair_submit_response(0, "ok")
    except Exception as e:
        _VIDEOEDIT_SEMAPHORE.release()
        for td in temp_dirs:
            shutil.rmtree(td, ignore_errors=True)
        logger.warning("Video repair request failed: %s", _exception_message(e))
        return _video_repair_submit_response(1, _exception_message(e))


# TODO: support image to video generation
@router.post("", response_model=VideoResponse)
async def create_video(
    request: Request,
    # multipart/form-data fields (optional; used only when content-type is multipart)
    prompt: Optional[str] = Form(None),
    input_reference: Optional[UploadFile] = File(None),
    reference_url: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    seconds: Optional[int] = Form(None),
    size: Optional[str] = Form(None),
    fps: Optional[int] = Form(None),
    num_frames: Optional[int] = Form(None),
    seed: Optional[int] = Form(1024),
    generator_device: Optional[str] = Form("cuda"),
    negative_prompt: Optional[str] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    enable_teacache: Optional[bool] = Form(False),
    enable_frame_interpolation: Optional[bool] = Form(False),
    frame_interpolation_exp: Optional[int] = Form(1),
    frame_interpolation_scale: Optional[float] = Form(1.0),
    frame_interpolation_model_path: Optional[str] = Form(None),
    enable_upscaling: Optional[bool] = Form(False),
    upscaling_model_path: Optional[str] = Form(None),
    upscaling_scale: Optional[int] = Form(4),
    output_quality: Optional[str] = Form("default"),
    output_compression: Optional[int] = Form(None),
    extra_body: Optional[str] = Form(None),
):
    content_type = request.headers.get("content-type", "").lower()
    request_id = generate_request_id()

    server_args = get_global_server_args()
    task_type = server_args.pipeline_config.task_type

    # Resolve input upload directory (may be a temp dir when saving is disabled)
    temp_dirs: list[str] = []
    if server_args.input_save_path is not None:
        uploads_dir = server_args.input_save_path
        os.makedirs(uploads_dir, exist_ok=True)
    else:
        uploads_dir = tempfile.mkdtemp(prefix="sglang_input_")
        temp_dirs.append(uploads_dir)

    # Resolve output directory
    effective_output_path = server_args.output_path
    output_persistent = True
    if "multipart/form-data" not in content_type:
        # JSON body may carry a per-request output_path; checked after parsing below
        pass

    if "multipart/form-data" in content_type:
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")
        # Validate image input based on model task type
        image_sources = merge_image_input_list(input_reference, reference_url)
        if task_type.requires_image_input() and not image_sources:
            raise HTTPException(
                status_code=400,
                detail="input_reference or reference_url is required for image-to-video generation",
            )
        try:
            input_path = await _save_first_input_image(
                image_sources, request_id, uploads_dir
            )
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to process image source: {str(e)}"
            )

        # Parse extra_body JSON (if provided in multipart form) to get fps/num_frames overrides
        extra_from_form: Dict[str, Any] = {}
        if extra_body:
            try:
                extra_from_form = json.loads(extra_body)
            except Exception:
                extra_from_form = {}

        fps_val = fps if fps is not None else extra_from_form.get("fps")
        num_frames_val = (
            num_frames if num_frames is not None else extra_from_form.get("num_frames")
        )

        req = VideoGenerationsRequest(
            prompt=prompt,
            input_reference=input_path,
            model=model,
            seconds=seconds if seconds is not None else 4,
            size=size,
            fps=fps_val,
            num_frames=num_frames_val,
            seed=seed,
            generator_device=generator_device,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            enable_teacache=enable_teacache,
            enable_frame_interpolation=enable_frame_interpolation,
            frame_interpolation_exp=frame_interpolation_exp,
            frame_interpolation_scale=frame_interpolation_scale,
            frame_interpolation_model_path=frame_interpolation_model_path,
            enable_upscaling=enable_upscaling,
            upscaling_model_path=upscaling_model_path,
            upscaling_scale=upscaling_scale,
            output_compression=output_compression,
            output_quality=output_quality,
            **(
                {"guidance_scale": guidance_scale} if guidance_scale is not None else {}
            ),
        )
    else:
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            # If client uses extra_body, merge it into the top-level payload
            payload: Dict[str, Any] = dict(body or {})
            extra = payload.pop("extra_body", None)
            if isinstance(extra, dict):
                # Shallow-merge: only keys like fps/num_frames are expected
                payload.update(extra)
            # openai may turn extra_body to extra_json
            extra_json = payload.pop("extra_json", None)
            if isinstance(extra_json, dict):
                payload.update(extra_json)
            # Validate image input based on model task type
            has_image_input = payload.get("reference_url") or payload.get(
                "input_reference"
            )
            if task_type.requires_image_input() and not has_image_input:
                raise HTTPException(
                    status_code=400,
                    detail="input_reference or reference_url is required for image-to-video generation",
                )
            # for non-multipart/form-data type
            if payload.get("reference_url"):
                try:
                    input_path = await _save_first_input_image(
                        payload.get("reference_url"), request_id, uploads_dir
                    )
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to process image source: {str(e)}",
                    )
                payload["input_reference"] = input_path
            req = VideoGenerationsRequest(**payload)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    # Resolve per-request output_path override
    effective_output_path = req.output_path or server_args.output_path
    if effective_output_path is None:
        output_tmp = tempfile.mkdtemp(prefix="sglang_output_")
        temp_dirs.append(output_tmp)
        effective_output_path = output_tmp
        output_persistent = False

    # Inject resolved output_path so _build_video_sampling_params picks it up
    req.output_path = effective_output_path

    logger.debug(f"Server received from create_video endpoint: req={req}")

    try:
        sampling_params = _build_video_sampling_params(request_id, req)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = _video_job_from_sampling(request_id, req, sampling_params)
    await VIDEO_STORE.upsert(request_id, job)

    # Build Req for scheduler
    batch = prepare_request(
        server_args=server_args,
        sampling_params=sampling_params,
    )
    # Add diffusers_kwargs if provided
    if req.diffusers_kwargs:
        batch.extra["diffusers_kwargs"] = req.diffusers_kwargs
    # Enqueue the job asynchronously and return immediately
    asyncio.create_task(
        _dispatch_job_async(
            request_id,
            batch,
            temp_dirs=temp_dirs or None,
            output_persistent=output_persistent,
        )
    )
    return VideoResponse(**job)


@router.get("", response_model=VideoListResponse)
async def list_videos(
    after: Optional[str] = Query(None),
    limit: Optional[int] = Query(None, ge=1, le=100),
    order: Optional[str] = Query("desc"),
):
    # Normalize order
    order = (order or "desc").lower()
    if order not in ("asc", "desc"):
        order = "desc"
    jobs = await VIDEO_STORE.list_values()

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
    items = [VideoResponse(**j) for j in jobs]
    return VideoListResponse(data=items)


@router.get("/{video_id}", response_model=VideoResponse)
async def retrieve_video(video_id: str = Path(...)):
    job = await VIDEO_STORE.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")
    response_job = dict(job)
    response_job["progress"] = _current_video_progress(job)
    return VideoResponse(**response_job)


@router.get("/{video_id}/progress")
async def retrieve_video_progress(video_id: str = Path(...)):
    job = await VIDEO_STORE.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")

    return {
        "id": video_id,
        "status": job.get("status"),
        "progress": _current_video_progress(job),
        "file_path": job.get("file_path"),
        "url": job.get("url"),
        "error": job.get("error"),
        "callback_status": job.get("callback_status"),
        "callback_error": job.get("callback_error"),
        "callback_attempts": job.get("callback_attempts"),
    }


# TODO: support aborting a job.
@router.delete("/{video_id}", response_model=VideoResponse)
async def delete_video(video_id: str = Path(...)):
    job = await VIDEO_STORE.pop(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")
    # Mark as deleted in response semantics
    job["status"] = "deleted"
    return VideoResponse(**job)


@router.get("/{video_id}/content")
async def download_video_content(
    video_id: str = Path(...), variant: Optional[str] = Query(None)
):
    job = await VIDEO_STORE.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")

    if job.get("url"):
        raise HTTPException(
            status_code=400,
            detail=f"Video has been uploaded to cloud storage. Please use the cloud URL: {job.get('url')}",
        )

    file_path = job.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Generation is still in-progress")

    media_type = "video/mp4"  # default variant
    return FileResponse(
        path=file_path, media_type=media_type, filename=os.path.basename(file_path)
    )
