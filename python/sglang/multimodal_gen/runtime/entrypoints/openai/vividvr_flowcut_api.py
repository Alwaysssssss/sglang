import asyncio
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from sglang.multimodal_gen.configs.sample.vividvr import VividVRSamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai.flowcut import (
    post_flowcut_callback,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    FlowCutMinIOConfig,
    FlowCutResponse,
    FlowCutVideoRepairRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import VIDEO_STORE
from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import (
    _VIDEOEDIT_SEMAPHORE,
    _build_vividvr_repair_kwargs,
    _copy_video_repair_request_with_caption,
    _ensure_vividvr_caption_file,
    _is_vividvr_video_repair_pipeline,
    _resolve_video_repair_model_name,
    _split_output_path,
    _video_repair_job_from_sampling,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.video_job_runner import (
    run_video_generation_job,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_progress import (
    VividVRFlowCutProgressReporter,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_storage import (
    VividVRFlowCutStorage,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/videos", tags=["videos"])


async def _post_stage_callback(callback_url: str, payload: dict[str, Any]) -> None:
    await post_flowcut_callback(callback_url, payload, timeout=2.0, max_retries=1)


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


def _flowcut_work_base_dir(server_args) -> tuple[str, bool]:
    base_dir = getattr(server_args, "input_save_path", None) or getattr(
        server_args, "output_path", None
    )
    if base_dir:
        return str(base_dir), False
    return tempfile.mkdtemp(prefix="sglang_vividvr_flowcut_"), True


def _flowcut_output_file_name(req: FlowCutVideoRepairRequest, request_id: str) -> str:
    _, output_file_name = _split_output_path(req.output_path, request_id, None)
    return output_file_name


def _duration_from_job_result(job_result) -> float | None:
    result = getattr(job_result, "result", None)
    duration = getattr(result, "inference_time_s", None)
    if duration is None:
        duration = getattr(result, "model_inference_runtime_seconds", None)
    return duration


async def _run_generation_with_timeout(batch, timeout: int | float | None):
    normalized_timeout = 300 if timeout in (None, 0) else timeout
    if normalized_timeout == -1:
        return await run_video_generation_job(batch)
    return await asyncio.wait_for(
        run_video_generation_job(batch),
        timeout=float(normalized_timeout),
    )


async def _dispatch_vividvr_flowcut_video_repair_job_async(
    job_id: str,
    batch,
    *,
    callback_url: str,
    storage: VividVRFlowCutStorage,
    minio_config: FlowCutMinIOConfig | None = None,
    timeout: int = 300,
) -> None:
    reporter = VividVRFlowCutProgressReporter(
        task_id=job_id,
        callback_url=callback_url,
        post_callback=_post_stage_callback,
    )
    try:
        await VIDEO_STORE.update_fields(job_id, {"status": "running", "progress": 60})
        await _send_stage_safely(reporter, "editing", task_id=job_id)
        job_result = await _run_generation_with_timeout(batch, timeout)
        save_file_path = job_result.save_file_path

        await VIDEO_STORE.update_fields(
            job_id,
            {
                "status": "uploading",
                "progress": 90,
                "file_path": save_file_path,
            },
        )
        await _send_stage_safely(reporter, "uploading_result", task_id=job_id)
        result_url = await storage.upload_result(save_file_path, minio_config)
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
    except Exception as e:
        reason = str(e)
        logger.error("FlowCut Vivid-VR repair failed job=%s: %s", job_id, reason)
        await VIDEO_STORE.update_fields(
            job_id,
            {
                "status": "failed",
                "error": {"message": reason},
            },
        )
        await _send_failed_safely(reporter, reason, task_id=job_id)
    finally:
        _VIDEOEDIT_SEMAPHORE.release()


@router.post("/repairs/flowcut", response_model=FlowCutResponse)
async def create_vividvr_flowcut_video_repair(request: Request):
    try:
        payload = await request.json()
    except Exception as e:
        return FlowCutResponse(code=1, message=f"invalid request: {e}")
    if not isinstance(payload, dict):
        return FlowCutResponse(code=1, message="invalid request: JSON object required")
    try:
        req = FlowCutVideoRepairRequest.model_validate(payload)
    except Exception as e:
        return FlowCutResponse(code=1, message=f"invalid request: {e}")

    if not req.task_id:
        return FlowCutResponse(code=1, message="taskId is required")
    if not req.callback_url:
        return FlowCutResponse(code=1, message="callbackUrl is required")
    if not req.video_input_path and not req.video_url:
        return FlowCutResponse(
            code=1,
            message="video_input_path or video_url is required",
        )

    server_args = get_global_server_args()
    if not _is_vividvr_video_repair_pipeline(server_args):
        return FlowCutResponse(
            code=1,
            message="FlowCut repair endpoint requires Vivid-VR pipeline",
        )
    if _VIDEOEDIT_SEMAPHORE.locked():
        return FlowCutResponse(code=2, message="A task is running.")

    await _VIDEOEDIT_SEMAPHORE.acquire()
    request_id = req.task_id
    reporter = VividVRFlowCutProgressReporter(
        task_id=request_id,
        callback_url=req.callback_url,
        post_callback=_post_stage_callback,
    )
    storage = None
    temp_base_dir = None
    scheduled = False

    try:
        await _send_stage_safely(reporter, "accepted", task_id=request_id)

        base_dir, base_dir_is_temp = _flowcut_work_base_dir(server_args)
        temp_base_dir = base_dir if base_dir_is_temp else None
        storage = VividVRFlowCutStorage(base_dir=base_dir, request_id=request_id)
        source = req.video_url or req.video_input_path
        video_input_path = await storage.materialize_video(
            source,
            filename_hint="input.mp4",
        )
        await _send_stage_safely(reporter, "input_ready", task_id=request_id)

        output_file_name = _flowcut_output_file_name(req, request_id)
        output_file_path = storage.output_file_path(output_file_name)
        output_dir = str(Path(output_file_path).parent)
        try:
            caption_file_path = await _ensure_vividvr_caption_file(
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

        req_for_sampling = _copy_video_repair_request_with_caption(
            req,
            caption_file_path,
        )
        vividvr_kwargs = _build_vividvr_repair_kwargs(
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
        job = _video_repair_job_from_sampling(request_id, req, sampling_params)
        job["model"] = _resolve_video_repair_model_name(req, server_args, "VividVR")
        await VIDEO_STORE.upsert(request_id, job)
        logger.info(
            "FlowCut video repair accepted task_id=%s output_path=%s callback_url=%s",
            request_id,
            job.get("file_path"),
            req.callback_url,
        )
        batch = prepare_request(server_args=server_args, sampling_params=sampling_params)
        asyncio.create_task(
            _dispatch_vividvr_flowcut_video_repair_job_async(
                request_id,
                batch,
                callback_url=req.callback_url,
                storage=storage,
                minio_config=req.minio_config,
                timeout=req.timeout,
            )
        )
        scheduled = True
        return FlowCutResponse(code=0, message="ok")
    except Exception as e:
        detail = e.detail if isinstance(e, HTTPException) else str(e)
        await _send_failed_safely(reporter, detail, task_id=request_id)
        return FlowCutResponse(code=1, message=detail)
    finally:
        if not scheduled:
            _VIDEOEDIT_SEMAPHORE.release()
            if storage is not None:
                storage.cleanup()
            if temp_base_dir is not None:
                shutil.rmtree(temp_base_dir, ignore_errors=True)
