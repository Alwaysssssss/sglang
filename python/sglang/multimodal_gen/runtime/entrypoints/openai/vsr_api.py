# SPDX-License-Identifier: Apache-2.0
"""Asynchronous VSR jobs using the shared video status/content/callback API."""

import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sglang.multimodal_gen.configs.pipeline_configs.vsr import WanVSRPipelineConfig
from sglang.multimodal_gen.configs.sample.sampling_params import generate_request_id
from sglang.multimodal_gen.configs.sample.vsr import WanVRSamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoRepairMinioConfig,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.storage import RequestCloudStorage
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import VIDEO_STORE
from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import (
    _clear_video_cancel_marker,
    _create_registered_video_task,
    _dispatch_job_async,
    _normalize_video_repair_payload,
    _save_video_source_to_path,
    _video_cancel_path,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.request_timeout import request_timeout_deadline
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.vsr.geometry import (
    parse_resolution,
    resize_to_long_edge,
)
from sglang.multimodal_gen.runtime.vsr.video_io import probe_video

router = APIRouter(prefix="/v1/videos", tags=["vsr"])
# VAE caches are mutable. Never dispatch concurrent requests to this model.
_VSR_SEMAPHORE = asyncio.Semaphore(1)


class VideoRestorationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,128}$")
    video_input_path: str | None = None
    video_url: str | None = None
    callback_url: str | None = None
    minio_config: VideoRepairMinioConfig | None = None
    output_object_key: str | None = None
    output_bucket: str | None = None
    timeout: int = -1
    target_resolution: str | None = None
    long_edge: int | None = Field(default=None, gt=0)
    tile_t: int | None = Field(default=None, gt=0)
    tile_h: int | None = Field(default=None, gt=0)
    tile_w: int | None = Field(default=None, gt=0)
    temporal_overlap: int | None = Field(default=None, ge=0)
    spatial_overlap: int | None = Field(default=None, ge=0)
    color_ref: Literal["global", "chunk", "none"] | None = None
    color_ref_samples: int | None = Field(default=None, ge=0)
    crf: int | None = Field(default=None, ge=0, le=51)
    read_queue: int | None = Field(default=None, ge=1, le=16)
    write_queue: int | None = Field(default=None, ge=1, le=16)
    gpu_postprocess: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def aliases(cls, value):
        return _normalize_video_repair_payload(value)

    @model_validator(mode="after")
    def validate_input(self):
        if bool(self.video_input_path) == bool(self.video_url):
            raise ValueError("Provide exactly one of video_input_path or videoUrl")
        if self.target_resolution:
            h, w = parse_resolution(self.target_resolution)
            if h <= 0 or w <= 0 or h % 2 or w % 2:
                raise ValueError("target_resolution must be positive even HxW")
        if self.callback_url and not self.callback_url.startswith(
            ("http://", "https://")
        ):
            raise ValueError("callbackUrl must use http or https")
        if self.output_object_key is not None and self.minio_config is None:
            raise ValueError("output_object_key requires minioConfig")
        return self


def _validate_tiling(req, cfg):
    def effective(name, default):
        value = getattr(req, name)
        return getattr(cfg, default) if value is None else value

    t, h, w = [effective(name, name) for name in ("tile_t", "tile_h", "tile_w")]
    temporal = effective("temporal_overlap", "temporal_overlap")
    spatial = effective("spatial_overlap", "spatial_overlap")
    if (t - 1) % 4 or h % 32 or w % 32:
        raise ValueError("tile_t must be 4n+1; tile_h/tile_w must be multiples of 32")
    if temporal >= t or spatial >= min(h, w):
        raise ValueError("overlap must be smaller than the corresponding tile")


async def _dispatch_vsr(job_id, batch, **kwargs):
    try:
        await VIDEO_STORE.update_fields(job_id, {"status": "running", "progress": 1})
        await _dispatch_job_async(job_id, batch, **kwargs)
    finally:
        _VSR_SEMAPHORE.release()


@router.post("/restorations")
async def create_video_restoration(request: Request):
    server_args = get_global_server_args()
    if not isinstance(server_args.pipeline_config, WanVSRPipelineConfig):
        return {"code": 1, "message": "This endpoint requires WanVSRPipeline"}
    try:
        req = VideoRestorationRequest.model_validate(await request.json())
        _validate_tiling(req, server_args.pipeline_config)
    except (ValueError, TypeError) as error:
        return {"code": 1, "message": str(error)}
    if _VSR_SEMAPHORE.locked():
        return {"code": 2, "message": "A task is running."}
    await _VSR_SEMAPHORE.acquire()
    temp_dirs = []
    handed_off = False
    try:
        job_id = req.task_id or generate_request_id()
        if await VIDEO_STORE.get(job_id) is not None:
            return {"code": 1, "message": "taskId already exists", "id": job_id}
        storage = RequestCloudStorage(req.minio_config) if req.minio_config else None
        input_path = req.video_input_path
        if req.video_url:
            temp_dir = tempfile.mkdtemp(prefix="sglang_vsr_input_")
            temp_dirs.append(temp_dir)
            target = os.path.join(temp_dir, "input.mp4")
            if storage:
                input_path = await storage.download_source(
                    req.video_url, target, default_ext=".mp4"
                )
            else:
                input_path = await _save_video_source_to_path(req.video_url, target)
        if not input_path or not Path(input_path).is_file():
            raise ValueError("Input video does not exist")
        fps, frames, src_h, src_w = await asyncio.to_thread(probe_video, input_path)
        cfg = server_args.pipeline_config
        target = req.target_resolution
        if target is None and req.long_edge is None:
            target = cfg.target_resolution
        h, w = (
            parse_resolution(target)
            if target
            else resize_to_long_edge(src_h, src_w, req.long_edge or cfg.long_edge)
        )
        output_dir = Path(server_args.output_path or "outputs") / "vsr"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (output_dir / f"{job_id}.mp4").resolve()
        cancel_path = _video_cancel_path(job_id)
        _clear_video_cancel_marker(cancel_path)
        kwargs = req.model_dump(
            exclude_none=True,
            exclude={
                "task_id",
                "video_url",
                "video_input_path",
                "callback_url",
                "minio_config",
                "output_object_key",
                "output_bucket",
                "timeout",
            },
        )
        sampling = WanVRSamplingParams.from_user_kwargs(
            server_args,
            **kwargs,
            request_id=job_id,
            video_input_path=str(Path(input_path).resolve()),
            output_path=str(output_path.parent),
            output_file_name=output_path.name,
            save_output=True,
            return_file_paths_only=True,
            height=h,
            width=w,
            num_frames=frames,
            # Generic sampling metadata requires int FPS. VSRRestoreStage probes
            # the source again and preserves its actual (possibly fractional) FPS.
            fps=max(1, round(fps)),
            num_inference_steps=1,
            request_cancel_path=cancel_path,
            request_timeout_deadline=request_timeout_deadline(req.timeout),
        )
        batch = prepare_request(server_args=server_args, sampling_params=sampling)
        await VIDEO_STORE.upsert(
            job_id,
            {
                "id": job_id,
                "object": "video",
                "model": "vsr",
                "status": "queued",
                "progress": 0,
                "created_at": int(time.time()),
                "size": f"{w}x{h}",
                "seconds": str(frames / fps),
                "file_path": None,
                "request_cancel_path": cancel_path,
            },
        )
        await _create_registered_video_task(
            job_id,
            _dispatch_vsr,
            job_id,
            batch,
            temp_dirs=temp_dirs,
            output_persistent=True,
            callback_url=req.callback_url,
            request_storage=storage,
            output_object_key=req.output_object_key
            or (f"{job_id}.mp4" if storage else None),
            output_bucket=req.output_bucket,
        )
        handed_off = True
        return {"code": 0, "message": "Task submitted", "id": job_id}
    except Exception as error:  # noqa: BLE001 - return submission failure and release admission
        return {"code": 1, "message": str(error)}
    finally:
        if not handed_off:
            _VSR_SEMAPHORE.release()
            for path in temp_dirs:
                shutil.rmtree(path, ignore_errors=True)
