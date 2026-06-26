import asyncio
import os
import shutil
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from sglang.multimodal_gen.configs.sample.vividvr import VividVRSamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai.flowcut import (
    post_flowcut_callback,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_protocol import (
    FlowCutMinIOConfig,
    FlowCutProgressResponse,
    FlowCutResponse,
    FlowCutVideoRepairRequest,
    FlowCutVideoResponse,
)
from sglang.multimodal_gen.runtime.request_timeout import (
    TASK_TIMEOUT_MESSAGE,
    is_task_timeout_error,
    request_timeout_deadline,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import VIDEO_STORE
from sglang.multimodal_gen.runtime.entrypoints.openai import video_repair_shared
from sglang.multimodal_gen.runtime.entrypoints.openai.video_job_runner import (
    run_video_generation_job,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_progress import (
    FLOWCUT_STAGE_PROGRESS,
    VividVRFlowCutProgressReporter,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_storage import (
    VividVRFlowCutStorage,
    default_flowcut_output_object_key,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.vividvr.progress_file import (
    read_vividvr_runtime_progress,
)

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/videos", tags=["videos"])
FLOWCUT_DENOISE_MONITOR_INTERVAL_SECONDS = 1.0
FLOWCUT_VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}
_FLOWCUT_CANCEL_DIR = Path(tempfile.gettempdir()) / "sglang_vividvr_flowcut_cancel"
_FLOWCUT_TASKS: dict[str, asyncio.Task[Any]] = {}
_FLOWCUT_TASKS_LOCK = asyncio.Lock()

_FLOWCUT_FIELD_ALIASES = {
    "taskId": "task_id",
    "callbackUrl": "callback_url",
    "videoUrl": "video_url",
    "captionFilePath": "caption_file_path",
    "minioConfig": "minio_config",
    "outputStorage": "output_storage",
    "outputPath": "output_path",
    "outputBucket": "output_bucket",
    "outputObjectKey": "output_object_key",
    "numFrames": "num_frames",
    "numInferenceSteps": "num_inference_steps",
    "guidanceScale": "guidance_scale",
    "generatorDevice": "generator_device",
    "numTemporalProcessFrames": "num_temporal_process_frames",
    "restorationGuidanceScale": "restoration_guidance_scale",
    "outputQuality": "output_quality",
    "outputCompression": "output_compression",
    "perfDumpPath": "perf_dump_path",
}

_FLOWCUT_MINIO_FIELD_ALIASES = {
    "bucketName": "bucket_name",
    "accessKey": "access_key",
    "secretKey": "secret_key",
}


def _normalize_aliases(
    payload: dict[str, Any],
    aliases: dict[str, str],
) -> dict[str, Any]:
    normalized = dict(payload)
    for alias, canonical in aliases.items():
        if alias not in payload:
            continue
        normalized[canonical] = payload[alias]
        if alias != canonical:
            normalized.pop(alias, None)
    return normalized


def _normalize_vividvr_flowcut_payload(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    payload = _normalize_aliases(body, _FLOWCUT_FIELD_ALIASES)
    minio_config = payload.get("minio_config")
    if minio_config is not None:
        if not isinstance(minio_config, dict):
            raise ValueError("minioConfig must be a JSON object")
        payload["minio_config"] = _normalize_aliases(
            minio_config, _FLOWCUT_MINIO_FIELD_ALIASES
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


def _task_id_from_flowcut_body(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    task_id = body.get("task_id") or body.get("taskId")
    if task_id is None:
        return None
    task_id = str(task_id).strip()
    return task_id or None


def _failed_flowcut_submission_job(
    request_id: str,
    reason: str,
    *,
    req: FlowCutVideoRepairRequest | None = None,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    callback_url = req.callback_url if req is not None else None
    output_object_key = req.output_object_key if req is not None else None
    output_bucket = req.output_bucket if req is not None else None
    timeout = req.timeout if req is not None else None
    model = req.model if req is not None else None

    if body is not None:
        callback_url = callback_url or body.get("callback_url") or body.get(
            "callbackUrl"
        )
        output_object_key = output_object_key or body.get(
            "output_object_key"
        ) or body.get("outputObjectKey")
        output_bucket = output_bucket or body.get("output_bucket") or body.get(
            "outputBucket"
        )
        timeout = timeout if timeout is not None else body.get("timeout")
        model = model or body.get("model")

    return {
        "id": request_id,
        "object": "video",
        "model": model or "VividVR",
        "status": "failed",
        "progress": 0,
        "created_at": int(time.time()),
        "size": "",
        "seconds": "",
        "quality": "standard",
        "file_path": None,
        "url": None,
        "error": {"message": reason},
        "reason": reason,
        "callback_url": callback_url,
        "callback_status": None,
        "callback_error": None,
        "timeout": timeout,
        "output_object_key": output_object_key,
        "output_bucket": output_bucket,
    }


async def _store_failed_flowcut_submission(
    request_id: str,
    reason: str,
    *,
    req: FlowCutVideoRepairRequest | None = None,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    job = _failed_flowcut_submission_job(request_id, reason, req=req, body=body)
    await VIDEO_STORE.upsert(request_id, job)
    return job


def _validate_flowcut_request(req: FlowCutVideoRepairRequest) -> None:
    if not req.task_id:
        raise ValueError("taskId is required")
    if not req.callback_url:
        raise ValueError("callbackUrl is required")
    if not req.video_input_path and not req.video_url:
        raise ValueError("video_input_path or video_url is required")
    if req.minio_config is None and (
        req.output_storage == "s3"
        or req.output_object_key is not None
        or req.output_bucket is not None
    ):
        raise ValueError("minioConfig is required for S3 output")


async def _post_stage_callback(callback_url: str, payload: dict[str, Any]) -> int:
    return await post_flowcut_callback(callback_url, payload, timeout=2.0, max_retries=1)


async def _post_flowcut_callback_with_bookkeeping(
    task_id: str,
    callback_url: str,
    payload: dict[str, Any],
) -> int:
    try:
        attempts = await _post_stage_callback(callback_url, payload)
    except Exception as e:
        await VIDEO_STORE.update_fields(
            task_id,
            {
                "callback_status": "failed",
                "callback_error": str(e),
                "callback_attempts": 1,
                "callback_completed_at": int(time.time()),
            },
        )
        raise

    await VIDEO_STORE.update_fields(
        task_id,
        {
            "callback_status": "succeeded",
            "callback_error": None,
            "callback_attempts": attempts,
            "callback_completed_at": int(time.time()),
        },
    )
    return attempts


def _make_bookkept_flowcut_callback(task_id: str):
    async def _post_callback(callback_url: str, payload: dict[str, Any]) -> int:
        return await _post_flowcut_callback_with_bookkeeping(
            task_id,
            callback_url,
            payload,
        )

    return _post_callback


async def _send_stage_safely(
    reporter: VividVRFlowCutProgressReporter,
    stage: str,
    *,
    task_id: str,
) -> None:
    try:
        await reporter.send_stage(stage)  # type: ignore[arg-type]
    except Exception as e:
        logger.warning(
            "FlowCut stage callback failed task_id=%s stage=%s: %s",
            task_id,
            stage,
            e,
        )


async def _send_failed_safely(
    reporter: VividVRFlowCutProgressReporter | None,
    reason: str,
    *,
    task_id: str,
) -> None:
    if reporter is None:
        return
    try:
        await reporter.send_failed(reason)
    except Exception as e:
        logger.warning(
            "FlowCut failed callback failed task_id=%s reason=%s: %s",
            task_id,
            reason,
            e,
        )


def _runtime_progress_from_batch(batch, progress_path: str | None = None) -> float | None:
    file_progress = read_vividvr_runtime_progress(progress_path)
    if file_progress is not None:
        return file_progress

    sampling_params = getattr(batch, "sampling_params", None)
    for candidate in (sampling_params, batch):
        if candidate is None:
            continue
        runtime_progress = getattr(candidate, "runtime_progress", None)
        if runtime_progress is None:
            continue
        try:
            return float(runtime_progress)
        except (TypeError, ValueError):
            return None
    return None


def _attach_flowcut_progress_path(batch, progress_path: str) -> None:
    extra = getattr(batch, "extra", None)
    if isinstance(extra, dict):
        extra["vividvr_flowcut_progress_path"] = progress_path
    elif hasattr(batch, "__dict__"):
        setattr(batch, "extra", {"vividvr_flowcut_progress_path": progress_path})


async def _send_denoise_progress_safely(
    reporter: VividVRFlowCutProgressReporter,
    runtime_progress: float,
    *,
    task_id: str,
) -> None:
    try:
        sent = await reporter.send_denoise_progress(runtime_progress)
    except Exception as e:
        logger.warning(
            "FlowCut denoise callback failed task_id=%s runtime_progress=%s: %s",
            task_id,
            runtime_progress,
            e,
        )
        return
    if sent:
        await VIDEO_STORE.update_fields(
            task_id,
            {"status": "running", "progress": reporter.last_progress},
        )


async def _monitor_vividvr_denoise_progress(
    task_id: str,
    batch,
    reporter: VividVRFlowCutProgressReporter,
    generation_task: asyncio.Task,
    *,
    progress_path: str | None = None,
    poll_interval_s: float = FLOWCUT_DENOISE_MONITOR_INTERVAL_SECONDS,
) -> None:
    while not generation_task.done():
        runtime_progress = _runtime_progress_from_batch(batch, progress_path)
        if runtime_progress is not None:
            await _send_denoise_progress_safely(
                reporter,
                runtime_progress,
                task_id=task_id,
            )
        await asyncio.wait({generation_task}, timeout=poll_interval_s)

    runtime_progress = _runtime_progress_from_batch(batch, progress_path)
    if runtime_progress is not None:
        await _send_denoise_progress_safely(
            reporter,
            runtime_progress,
            task_id=task_id,
        )


def _flowcut_work_base_dir(server_args) -> tuple[str, bool]:
    base_dir = getattr(server_args, "input_save_path", None)
    if base_dir:
        return str(base_dir), False
    return tempfile.mkdtemp(prefix="sglang_vividvr_flowcut_"), True


def _flowcut_cancel_path(task_id: str) -> str:
    _FLOWCUT_CANCEL_DIR.mkdir(parents=True, exist_ok=True)
    safe_task_id = Path(task_id).name
    return str((_FLOWCUT_CANCEL_DIR / f"{safe_task_id}.cancel").resolve())


def _clear_flowcut_cancel_marker(task_id: str) -> str:
    cancel_path = Path(_flowcut_cancel_path(task_id))
    cancel_path.unlink(missing_ok=True)
    return str(cancel_path)


def _write_flowcut_cancel_marker(task_id: str) -> str:
    cancel_path = Path(_flowcut_cancel_path(task_id))
    cancel_path.parent.mkdir(parents=True, exist_ok=True)
    cancel_path.write_text("cancelled\n", encoding="utf-8")
    return str(cancel_path)


async def _register_flowcut_task(task_id: str, task: Any) -> None:
    async with _FLOWCUT_TASKS_LOCK:
        _FLOWCUT_TASKS[task_id] = task


async def _unregister_flowcut_task(task_id: str, task: Any | None) -> None:
    async with _FLOWCUT_TASKS_LOCK:
        current = _FLOWCUT_TASKS.get(task_id)
        if current is task or task is None:
            _FLOWCUT_TASKS.pop(task_id, None)


def _schedule_flowcut_task_cleanup(task_id: str, task: Any) -> None:
    if not hasattr(task, "add_done_callback"):
        return

    def _cleanup(done_task: Any) -> None:
        asyncio.create_task(_unregister_flowcut_task(task_id, done_task))

    task.add_done_callback(_cleanup)


async def _cancel_registered_flowcut_task(task_id: str) -> bool:
    async with _FLOWCUT_TASKS_LOCK:
        task = _FLOWCUT_TASKS.get(task_id)
    if task is None:
        return False
    task.cancel()
    return True


async def _get_flowcut_job_or_404(video_id: str) -> dict[str, Any]:
    job = await VIDEO_STORE.get(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return job


def _flowcut_progress_payload(video_id: str, job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": video_id,
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "file_path": job.get("file_path"),
        "url": job.get("url"),
        "error": job.get("error"),
        "reason": job.get("reason"),
        "callback_status": job.get("callback_status"),
        "callback_error": job.get("callback_error"),
        "callback_attempts": job.get("callback_attempts"),
    }


def _normalize_flowcut_video_extension(extension: str | None) -> str:
    normalized = (extension or ".mp4").lower()
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    if normalized not in FLOWCUT_VIDEO_EXTENSIONS:
        return ".mp4"
    return normalized


def _flowcut_input_extension(video_input_path: str) -> str:
    return _normalize_flowcut_video_extension(Path(video_input_path).suffix)


def _flowcut_output_target(
    output_path: str | None,
    request_id: str,
    server_output_path: str | None,
    *,
    input_extension: str,
) -> tuple[str | None, str]:
    normalized_extension = _normalize_flowcut_video_extension(input_extension)
    if output_path:
        output_path_obj = Path(output_path).expanduser()
        if output_path_obj.suffix:
            return (
                str(output_path_obj.resolve().parent),
                f"{output_path_obj.stem}{normalized_extension}",
            )
        return str(output_path_obj.resolve()), f"{request_id}{normalized_extension}"

    output_dir = server_output_path
    return output_dir, f"{request_id}{normalized_extension}"


def _flowcut_output_file_name(
    req: FlowCutVideoRepairRequest,
    request_id: str,
    *,
    input_extension: str,
) -> str:
    _, output_file_name = _flowcut_output_target(
        req.output_path,
        request_id,
        None,
        input_extension=input_extension,
    )
    return output_file_name


def _resolve_flowcut_persistent_output_path(
    req: FlowCutVideoRepairRequest,
    request_id: str,
    server_args,
    *,
    input_extension: str,
) -> str | None:
    output_dir, output_file_name = _flowcut_output_target(
        req.output_path,
        request_id,
        getattr(server_args, "output_path", None),
        input_extension=input_extension,
    )
    if output_dir is None:
        return None
    return str((Path(output_dir).expanduser().resolve() / output_file_name).resolve())


def _resolve_flowcut_output_bucket(
    req: FlowCutVideoRepairRequest,
) -> str | None:
    if req.output_bucket:
        return req.output_bucket
    if req.minio_config is not None:
        return req.minio_config.bucket_name
    return None


def _resolve_flowcut_output_object_key(
    req: FlowCutVideoRepairRequest,
    request_id: str,
    output_file_name: str,
    *,
    input_extension: str,
) -> str | None:
    if req.output_object_key:
        if Path(req.output_object_key).suffix:
            return req.output_object_key
        return f"{req.output_object_key}{_normalize_flowcut_video_extension(input_extension)}"
    if req.minio_config is None:
        return None
    extension = Path(output_file_name).suffix or _normalize_flowcut_video_extension(
        input_extension
    )
    return default_flowcut_output_object_key(request_id, extension=extension)


def _duration_from_job_result(job_result) -> float | None:
    result = getattr(job_result, "result", None)
    duration = getattr(result, "inference_time_s", None)
    if duration is None:
        duration = getattr(result, "model_inference_runtime_seconds", None)
    return duration


class _FlowCutGenerationTimeoutError(TimeoutError):
    def __init__(self, timeout_s: float, generation_task: asyncio.Task):
        super().__init__(f"generation timed out after {timeout_s:g} seconds")
        self.timeout_s = timeout_s
        self.generation_task = generation_task


async def _run_generation_with_timeout(batch, timeout: int | float | None):
    generation_task = asyncio.create_task(run_video_generation_job(batch))
    return await _await_generation_task_with_timeout(generation_task, timeout)


async def _await_generation_task_with_timeout(
    generation_task: asyncio.Task,
    timeout: int | float | None,
):
    normalized_timeout = 300 if timeout in (None, 0) else timeout
    if normalized_timeout == -1:
        return await generation_task

    timeout_s = float(normalized_timeout)
    try:
        return await asyncio.wait_for(asyncio.shield(generation_task), timeout=timeout_s)
    except asyncio.TimeoutError as e:
        raise _FlowCutGenerationTimeoutError(timeout_s, generation_task) from e


async def _dispatch_vividvr_flowcut_video_repair_job_async(
    job_id: str,
    batch,
    *,
    callback_url: str,
    storage: VividVRFlowCutStorage,
    minio_config: FlowCutMinIOConfig | None = None,
    output_object_key: str | None = None,
    output_bucket: str | None = None,
    persistent_output_path: str | None = None,
    cleanup_workdir_on_finish: bool = False,
    cleanup_base_dir_on_finish: bool = False,
    cleanup_workdir_on_cancel: bool = False,
    cleanup_base_dir_on_cancel: bool = False,
    timeout: int = 300,
) -> None:
    reporter = VividVRFlowCutProgressReporter(
        task_id=job_id,
        callback_url=callback_url,
        post_callback=_make_bookkept_flowcut_callback(job_id),
    )
    generation_task_to_drain = None
    monitor_task = None
    generation_task = None
    should_cleanup_artifacts = False
    should_cleanup_base_dir = False
    try:
        await VIDEO_STORE.update_fields(
            job_id,
            {
                "status": "running",
                "progress": FLOWCUT_STAGE_PROGRESS["caption_ready"],
            },
        )
        progress_path = str(storage.manifests_dir / "runtime_progress.json")
        _attach_flowcut_progress_path(batch, progress_path)
        generation_task = asyncio.create_task(run_video_generation_job(batch))
        monitor_task = asyncio.create_task(
            _monitor_vividvr_denoise_progress(
                job_id,
                batch,
                reporter,
                generation_task,
                progress_path=progress_path,
            )
        )
        try:
            job_result = await _await_generation_task_with_timeout(
                generation_task,
                timeout,
            )
        except Exception:
            if monitor_task is not None:
                monitor_task.cancel()
                with suppress(asyncio.CancelledError):
                    await monitor_task
            raise
        else:
            if monitor_task is not None:
                await monitor_task
        save_file_path = job_result.save_file_path

        await VIDEO_STORE.update_fields(
            job_id,
            {
                "status": "uploading",
                "progress": FLOWCUT_STAGE_PROGRESS["uploading_result"],
                "file_path": save_file_path,
            },
        )
        await _send_stage_safely(reporter, "uploading_result", task_id=job_id)
        if minio_config is None:
            result_url = storage.finalize_local_result(
                save_file_path,
                persistent_output_path=persistent_output_path,
            )
        else:
            result_url = await storage.upload_result(
                save_file_path,
                minio_config,
                object_key=output_object_key,
                bucket_name=output_bucket,
            )
        should_cleanup_artifacts = cleanup_workdir_on_finish and (
            minio_config is not None or persistent_output_path is not None
        )
        should_cleanup_base_dir = should_cleanup_artifacts and cleanup_base_dir_on_finish
        duration = _duration_from_job_result(job_result)
        await VIDEO_STORE.update_fields(
            job_id,
            {
                "status": "completed",
                "progress": 100,
                "completed_at": int(time.time()),
                "url": result_url,
                "file_path": None if minio_config is not None else result_url,
                "inference_time_s": duration,
                "output_object_key": output_object_key,
                "output_bucket": output_bucket,
            },
        )
        try:
            await reporter.send_succeeded(result_url, duration=duration)
        except Exception as callback_error:
            logger.warning(
                "FlowCut succeeded callback failed task_id=%s: %s",
                job_id,
                callback_error,
            )
    except asyncio.CancelledError:
        if monitor_task is not None:
            monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await monitor_task
        if generation_task is not None and not generation_task.done():
            generation_task.cancel()
            generation_task_to_drain = generation_task
        reason = TASK_TIMEOUT_MESSAGE
        logger.info("FlowCut Vivid-VR repair cancelled job=%s", job_id)
        should_cleanup_artifacts = cleanup_workdir_on_cancel
        should_cleanup_base_dir = cleanup_base_dir_on_cancel
        await VIDEO_STORE.update_fields(
            job_id,
            {
                "status": "failed",
                "error": {"message": reason},
                "reason": reason,
                "cancelled_at": int(time.time()),
            },
        )
        job = await VIDEO_STORE.get(job_id)
        should_send_failed = True
        if job is not None and job.get("callback_status") in {
            "succeeded",
            "cancel_requested",
        }:
            should_send_failed = False
        if should_send_failed:
            await _send_failed_safely(reporter, reason, task_id=job_id)
        raise
    except Exception as e:
        if isinstance(e, _FlowCutGenerationTimeoutError):
            generation_task_to_drain = e.generation_task
            reason = TASK_TIMEOUT_MESSAGE
            should_cleanup_artifacts = cleanup_workdir_on_cancel
            should_cleanup_base_dir = cleanup_base_dir_on_cancel
        elif is_task_timeout_error(e):
            reason = TASK_TIMEOUT_MESSAGE
            should_cleanup_artifacts = cleanup_workdir_on_cancel
            should_cleanup_base_dir = cleanup_base_dir_on_cancel
        else:
            reason = str(e)
        logger.error("FlowCut Vivid-VR repair failed job=%s: %s", job_id, reason)
        await VIDEO_STORE.update_fields(
            job_id,
            {
                "status": "failed",
                "error": {"message": reason},
                "reason": reason,
            },
        )
        job = await VIDEO_STORE.get(job_id)
        should_send_failed = True
        if is_task_timeout_error(e) or isinstance(e, _FlowCutGenerationTimeoutError):
            if job is not None and job.get("callback_status") in {
                "succeeded",
                "cancel_requested",
            }:
                should_send_failed = False
        if should_send_failed:
            await _send_failed_safely(reporter, reason, task_id=job_id)
    finally:
        if generation_task_to_drain is not None:
            try:
                await generation_task_to_drain
            except BaseException as e:
                logger.warning(
                    "FlowCut timed-out generation finished with error job=%s: %s",
                    job_id,
                    e,
                )
        if should_cleanup_artifacts:
            storage.cleanup()
            if should_cleanup_base_dir:
                shutil.rmtree(storage.base_dir, ignore_errors=True)
        current_task = asyncio.current_task()
        await _unregister_flowcut_task(job_id, current_task)
        video_repair_shared.VIDEOEDIT_SEMAPHORE.release()


@router.post("/repairs/flowcut", response_model=FlowCutResponse)
async def create_vividvr_flowcut_video_repair(request: Request):
    raw_payload = None
    try:
        raw_payload = await request.json()
    except Exception as e:
        return FlowCutResponse(code=1, message=f"invalid request: {e}")

    try:
        payload = _normalize_vividvr_flowcut_payload(raw_payload)
        req = FlowCutVideoRepairRequest.model_validate(payload)
    except Exception as e:
        detail = _exception_message(e)
        request_id = _task_id_from_flowcut_body(raw_payload)
        if request_id is not None:
            await _store_failed_flowcut_submission(
                request_id,
                detail,
                body=raw_payload if isinstance(raw_payload, dict) else None,
            )
        return FlowCutResponse(code=1, message=f"invalid request: {detail}")

    try:
        _validate_flowcut_request(req)
    except Exception as e:
        detail = _exception_message(e)
        await _store_failed_flowcut_submission(
            req.task_id or "unknown",
            detail,
            req=req,
            body=payload,
        )
        return FlowCutResponse(code=1, message=detail)

    server_args = get_global_server_args()
    if not video_repair_shared.is_vividvr_video_repair_pipeline(server_args):
        return FlowCutResponse(
            code=1,
            message="FlowCut repair endpoint requires Vivid-VR pipeline",
        )
    if video_repair_shared.VIDEOEDIT_SEMAPHORE.locked():
        return FlowCutResponse(code=2, message="A task is running.")

    await video_repair_shared.VIDEOEDIT_SEMAPHORE.acquire()
    request_id = req.task_id
    reporter = VividVRFlowCutProgressReporter(
        task_id=request_id,
        callback_url=req.callback_url,
        post_callback=_make_bookkept_flowcut_callback(request_id),
    )
    storage = None
    temp_base_dir = None
    scheduled = False

    try:
        await _send_stage_safely(reporter, "accepted", task_id=request_id)

        base_dir, base_dir_is_temp = _flowcut_work_base_dir(server_args)
        temp_base_dir = base_dir if base_dir_is_temp else None
        storage = VividVRFlowCutStorage(base_dir=base_dir, request_id=request_id)
        request_cancel_path = _clear_flowcut_cancel_marker(request_id)
        source = req.video_url or req.video_input_path
        video_input_path = await storage.materialize_video(source)
        await _send_stage_safely(reporter, "input_ready", task_id=request_id)

        input_extension = _flowcut_input_extension(video_input_path)
        output_file_name = _flowcut_output_file_name(
            req,
            request_id,
            input_extension=input_extension,
        )
        output_object_key = _resolve_flowcut_output_object_key(
            req,
            request_id,
            output_file_name,
            input_extension=input_extension,
        )
        output_bucket = _resolve_flowcut_output_bucket(req)
        persistent_output_path = _resolve_flowcut_persistent_output_path(
            req,
            request_id,
            server_args,
            input_extension=input_extension,
        )
        cleanup_workdir_on_finish = temp_base_dir is not None and (
            req.minio_config is not None or persistent_output_path is not None
        )
        output_file_path = storage.output_file_path(output_file_name)
        output_dir = str(Path(output_file_path).parent)
        try:
            caption_file_path = await video_repair_shared.ensure_vividvr_caption_file(
                request_id=request_id,
                req=req,
                server_args=server_args,
                video_input_path=video_input_path,
                output_dir=output_dir,
            )
        except Exception as e:
            await _send_failed_safely(
                reporter,
                f"caption bridge failed: {e}",
                task_id=request_id,
            )
            return FlowCutResponse(code=1, message=f"caption bridge failed: {e}")
        await _send_stage_safely(reporter, "caption_ready", task_id=request_id)

        req_for_sampling = video_repair_shared.copy_video_repair_request_with_caption(
            req,
            caption_file_path,
        )
        vividvr_kwargs = video_repair_shared.build_vividvr_repair_kwargs(
            request_id=request_id,
            req=req_for_sampling,
            server_args=server_args,
            video_input_path=video_input_path,
            output_dir=output_dir,
            output_file_name=Path(output_file_path).name,
        )
        sampling_params = VividVRSamplingParams.from_user_kwargs(
            server_args,
            **vividvr_kwargs,
        )
        sampling_params.request_cancel_path = request_cancel_path
        sampling_params.request_timeout_deadline = request_timeout_deadline(req.timeout)
        job = video_repair_shared.video_repair_job_from_sampling(
            request_id, req, sampling_params
        )
        job["model"] = video_repair_shared.resolve_video_repair_model_name(
            req, server_args, "VividVR"
        )
        job["output_object_key"] = output_object_key
        job["output_bucket"] = output_bucket
        job["timeout"] = req.timeout
        job["request_cancel_path"] = request_cancel_path
        await VIDEO_STORE.upsert(request_id, job)
        logger.info(
            "FlowCut video repair accepted task_id=%s output_path=%s callback_url=%s",
            request_id,
            job.get("file_path"),
            req.callback_url,
        )
        batch = prepare_request(server_args=server_args, sampling_params=sampling_params)
        if hasattr(batch, "__dict__"):
            batch.request_cancel_path = request_cancel_path
            batch.request_timeout_deadline = sampling_params.request_timeout_deadline
        dispatch_task = asyncio.create_task(
            _dispatch_vividvr_flowcut_video_repair_job_async(
                request_id,
                batch,
                callback_url=req.callback_url,
                storage=storage,
                minio_config=req.minio_config,
                output_object_key=output_object_key,
                output_bucket=output_bucket,
                persistent_output_path=persistent_output_path,
                cleanup_workdir_on_finish=cleanup_workdir_on_finish,
                cleanup_base_dir_on_finish=cleanup_workdir_on_finish,
                cleanup_workdir_on_cancel=temp_base_dir is not None,
                cleanup_base_dir_on_cancel=temp_base_dir is not None,
                timeout=req.timeout,
            )
        )
        await _register_flowcut_task(request_id, dispatch_task)
        _schedule_flowcut_task_cleanup(request_id, dispatch_task)
        scheduled = True
        return FlowCutResponse(code=0, message="ok")
    except Exception as e:
        detail = e.detail if isinstance(e, HTTPException) else str(e)
        await _send_failed_safely(reporter, detail, task_id=request_id)
        return FlowCutResponse(code=1, message=detail)
    finally:
        if not scheduled:
            video_repair_shared.VIDEOEDIT_SEMAPHORE.release()
            if storage is not None:
                storage.cleanup()
            if temp_base_dir is not None:
                shutil.rmtree(temp_base_dir, ignore_errors=True)


@router.get("/repairs/flowcut/{video_id}", response_model=FlowCutVideoResponse)
async def retrieve_vividvr_flowcut_video_repair(video_id: str):
    job = await _get_flowcut_job_or_404(video_id)
    return FlowCutVideoResponse(**job)


@router.get(
    "/repairs/flowcut/{video_id}/progress",
    response_model=FlowCutProgressResponse,
)
async def retrieve_vividvr_flowcut_video_repair_progress(video_id: str):
    job = await _get_flowcut_job_or_404(video_id)
    return FlowCutProgressResponse(**_flowcut_progress_payload(video_id, job))


@router.delete("/repairs/flowcut/{video_id}", response_model=FlowCutVideoResponse)
async def delete_vividvr_flowcut_video_repair(video_id: str):
    job = await _get_flowcut_job_or_404(video_id)

    if job.get("status") in {"completed", "failed", "deleted"}:
        return FlowCutVideoResponse(**job)

    reason = TASK_TIMEOUT_MESSAGE
    request_cancel_path = _write_flowcut_cancel_marker(video_id)
    updated_fields = {
        "status": "failed",
        "error": {"message": reason},
        "reason": reason,
        "cancelled_at": int(time.time()),
        "request_cancel_path": request_cancel_path,
        "callback_status": "cancel_requested",
        "callback_error": None,
    }
    await VIDEO_STORE.update_fields(video_id, updated_fields)
    await _cancel_registered_flowcut_task(video_id)

    callback_url = job.get("callback_url")
    if callback_url:
        payload = {
            "status": "failed",
            "progress": 98.0,
            "reason": reason,
            "output": "",
        }
        await _post_flowcut_callback_with_bookkeeping(video_id, callback_url, payload)

    updated_job = await VIDEO_STORE.get(video_id)
    if updated_job is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return FlowCutVideoResponse(**updated_job)
