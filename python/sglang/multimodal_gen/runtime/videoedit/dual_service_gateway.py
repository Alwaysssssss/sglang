# SPDX-License-Identifier: Apache-2.0
"""Persistent single-dispatcher gateway for normal/DMD VideoEdit services."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import VideoRepairRequest
from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import (
    _normalize_video_repair_payload,
    _validate_video_repair_request,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.videoedit.dual_service_store import (
    ACTIVE_STATUSES,
    DuplicateTaskError,
    DualServiceStore,
)

logger = init_logger(__name__)

_NORMAL_MODELS = {None, "", "videoedit", "videoedit-normal", "normal"}
_DMD_MODELS = {"videoedit-dmd", "dmd"}
_BACKEND_TERMINAL = {"completed", "failed", "deleted"}


@dataclass(frozen=True)
class GatewayConfig:
    queue_db: str
    normal_url: str = "http://127.0.0.1:31100"
    dmd_url: str = "http://127.0.0.1:32100"
    poll_interval: float = 1.0
    health_timeout: float = 2.0

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        return cls(
            queue_db=os.environ.get(
                "VIDEOEDIT_DUAL_QUEUE_DB",
                "runtime/videoedit-dual/queue.sqlite3",
            ),
            normal_url=os.environ.get(
                "VIDEOEDIT_NORMAL_BACKEND", "http://127.0.0.1:31100"
            ),
            dmd_url=os.environ.get("VIDEOEDIT_DMD_BACKEND", "http://127.0.0.1:32100"),
            poll_interval=max(
                0.1, float(os.environ.get("VIDEOEDIT_GATEWAY_POLL_INTERVAL", "1"))
            ),
            health_timeout=max(
                0.1, float(os.environ.get("VIDEOEDIT_GATEWAY_HEALTH_TIMEOUT", "2"))
            ),
        )

    def backend_url(self, variant: str) -> str:
        if variant == "normal":
            return self.normal_url.rstrip("/")
        if variant == "dmd":
            return self.dmd_url.rstrip("/")
        raise ValueError(f"Unknown VideoEdit variant: {variant}")


def resolve_variant(model: Any) -> str:
    value = None if model is None else str(model).strip().lower()
    if value in _NORMAL_MODELS:
        return "normal"
    if value in _DMD_MODELS:
        return "dmd"
    raise ValueError(f"Unsupported VideoEdit model: {model}")


def _apply_variant_sampling_policy(
    request_model: VideoRepairRequest, variant: str
) -> VideoRepairRequest:
    if variant != "dmd":
        return request_model
    return request_model.model_copy(
        update={
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "dynamic_cfg": False,
            "negative_prompt": None,
        }
    )


class GatewayRuntime:
    def __init__(self, config: GatewayConfig):
        self.config = config
        self.store = DualServiceStore(config.queue_db)
        self.client = httpx.AsyncClient(follow_redirects=True, timeout=None)
        self._dispatcher_task: asyncio.Task | None = None
        self._wake = asyncio.Event()
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        if self._dispatcher_task is None:
            self._dispatcher_task = asyncio.create_task(
                self._dispatcher_loop(), name="videoedit-dual-dispatcher"
            )

    async def close(self) -> None:
        self._stopping.set()
        self._wake.set()
        if self._dispatcher_task is not None:
            self._dispatcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._dispatcher_task
            self._dispatcher_task = None
        await self.client.aclose()

    def wake(self) -> None:
        self._wake.set()

    async def _store_call(self, method_name: str, *args, **kwargs):
        method = getattr(self.store, method_name)
        return await asyncio.to_thread(method, *args, **kwargs)

    async def backend_health(self, backend_url: str) -> bool:
        try:
            response = await self.client.get(
                f"{backend_url.rstrip('/')}/health",
                timeout=self.config.health_timeout,
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def health_snapshot(self) -> dict[str, Any]:
        normal, dmd = await asyncio.gather(
            self.backend_health(self.config.normal_url),
            self.backend_health(self.config.dmd_url),
        )
        if normal and dmd:
            status = "ok"
        elif normal:
            status = "degraded_normal_only"
        else:
            status = "unavailable"
        counts = await self._store_call("counts")
        return {
            "status": status,
            "backends": {"normal": normal, "dmd": dmd},
            "queue": counts,
        }

    async def enqueue(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = _normalize_video_repair_payload(payload)
        task_id = str(
            normalized.get("task_id") or f"videoedit-{uuid.uuid4().hex}"
        ).strip()
        normalized["task_id"] = task_id
        try:
            request_model = VideoRepairRequest(**normalized)
            _validate_video_repair_request(request_model)
            variant = resolve_variant(request_model.model)
            request_model = _apply_variant_sampling_policy(request_model, variant)
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        backend_url = self.config.backend_url(variant)
        if not await self.backend_health(backend_url):
            raise HTTPException(
                status_code=503,
                detail=f"VideoEdit {variant} backend is unavailable",
            )

        request_payload = request_model.model_dump(mode="json")
        try:
            task = await self._store_call(
                "enqueue",
                task_id=task_id,
                variant=variant,
                backend_url=backend_url,
                request_payload=request_payload,
            )
        except DuplicateTaskError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        self.wake()
        return task

    async def _dispatcher_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                active = await self._store_call("get_active")
                if active is None:
                    active = await self._store_call("claim_next")
                if active is None:
                    self._wake.clear()
                    try:
                        await asyncio.wait_for(
                            self._wake.wait(), timeout=self.config.poll_interval
                        )
                    except asyncio.TimeoutError:
                        pass
                    continue

                await self._advance(active)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("VideoEdit gateway dispatcher iteration failed")
            await asyncio.sleep(self.config.poll_interval)

    async def _advance(self, task: dict[str, Any]) -> None:
        status = task["status"]
        if status == "dispatching" and task.get("submitted_at") is None:
            await self._submit(task)
            return
        await self._refresh_backend_task(task)

    async def _submit(self, task: dict[str, Any]) -> None:
        task_id = task["task_id"]
        endpoint = f"{task['backend_url']}/v1/videos/repairs"
        try:
            response = await self.client.post(endpoint, json=task["request_json"])
            response.raise_for_status()
            body = response.json()
        except httpx.ConnectError as error:
            await self._store_call("update_task", task_id, error=str(error))
            return
        except (httpx.HTTPError, ValueError) as error:
            # The backend may have accepted the request before the response was lost.
            await self._store_call(
                "update_task",
                task_id,
                submitted_at=time.time(),
                error=f"Uncertain backend submission: {error}",
            )
            return

        code = int(body.get("code", 1)) if isinstance(body, dict) else 1
        if code == 0:
            await self._store_call(
                "update_task",
                task_id,
                status="running",
                submitted_at=time.time(),
                backend_response=body,
                error=None,
            )
        elif code == 1:
            await self._store_call(
                "mark_terminal",
                task_id,
                "failed",
                backend_response=body,
                error=str(
                    body.get("reason") or body.get("message") or "submission failed"
                ),
            )
        else:
            # code=2 means the backend is busy. Keep the global active slot until
            # the task can be reconciled; never dispatch a second variant.
            await self._store_call(
                "update_task",
                task_id,
                submitted_at=time.time(),
                backend_response=body,
                error="Backend reported a running task; awaiting reconciliation",
            )

    async def _backend_task(self, task: dict[str, Any]) -> dict[str, Any] | None:
        endpoint = f"{task['backend_url']}/v1/videos/{task['task_id']}"
        response = await self.client.get(endpoint, timeout=10.0)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    async def _refresh_backend_task(self, task: dict[str, Any]) -> None:
        task_id = task["task_id"]
        try:
            body = await self._backend_task(task)
        except (httpx.HTTPError, ValueError) as error:
            await self._store_call("update_task", task_id, error=str(error))
            return

        if body is None:
            if task.get("submitted_at") is None:
                return
            await self._store_call(
                "update_task",
                task_id,
                error=(
                    "Backend has no task record after submission; queue is paused "
                    "to avoid duplicate execution"
                ),
            )
            return

        backend_status = str(body.get("status", "")).lower()
        if backend_status in _BACKEND_TERMINAL:
            terminal = "completed" if backend_status == "completed" else "failed"
            await self._store_call(
                "mark_terminal",
                task_id,
                terminal,
                backend_response=body,
                error=None if terminal == "completed" else _backend_error(body),
            )
            self.wake()
            return

        next_status = "cancelling" if task["status"] == "cancelling" else "running"
        await self._store_call(
            "update_task",
            task_id,
            status=next_status,
            backend_response=body,
            error=None,
        )

    async def refresh_task(self, task: dict[str, Any]) -> dict[str, Any]:
        if task["status"] in ACTIVE_STATUSES:
            await self._refresh_backend_task(task)
            refreshed = await self._store_call("get", task["task_id"])
            if refreshed is not None:
                return refreshed
        return task

    async def cancel(self, task: dict[str, Any]) -> dict[str, Any]:
        task_id = task["task_id"]
        if task["status"] == "queued":
            await self._store_call("cancel_queued", task_id)
            result = await self._store_call("get", task_id)
            assert result is not None
            return result
        if task["status"] not in ACTIVE_STATUSES:
            return task

        task = await self._store_call(
            "update_task", task_id, status="cancelling", error=None
        )
        try:
            response = await self.client.delete(
                f"{task['backend_url']}/v1/videos/{task_id}"
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as error:
            return await self._store_call("update_task", task_id, error=str(error))

        backend_status = str(body.get("status", "")).lower()
        if backend_status in _BACKEND_TERMINAL:
            terminal = "completed" if backend_status == "completed" else "failed"
            result = await self._store_call(
                "mark_terminal",
                task_id,
                terminal,
                backend_response=body,
                error=None if terminal == "completed" else _backend_error(body),
            )
            self.wake()
            return result
        return await self._store_call("update_task", task_id, backend_response=body)


def _backend_error(body: dict[str, Any]) -> str:
    error = body.get("error")
    if isinstance(error, dict):
        error = error.get("message")
    return str(body.get("reason") or error or "Backend task failed")


def _task_response(task: dict[str, Any], store: DualServiceStore) -> dict[str, Any]:
    backend = task.get("backend_response")
    response = dict(backend) if isinstance(backend, dict) else {}
    request_payload = task.get("request_json") or {}
    response.update(
        {
            "id": task["task_id"],
            "object": response.get("object", "video"),
            "model": request_payload.get("model") or "videoedit",
            "variant": task["variant"],
            "status": task["status"],
            "created_at": int(task["created_at"]),
            "started_at": task.get("started_at"),
            "completed_at": task.get("completed_at"),
            "error": task.get("error") or response.get("error"),
        }
    )
    if task["status"] == "queued":
        response["progress"] = 0
        response["queue_position"] = store.queue_position(task["task_id"])
    elif task["status"] == "cancelled":
        response["progress"] = 0
    return response


def _admin_task(task: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in task.items() if key not in {"request_json"}}


def create_app(config: GatewayConfig | None = None) -> FastAPI:
    gateway_config = config or GatewayConfig.from_env()
    runtime = GatewayRuntime(gateway_config)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await runtime.start()
        try:
            yield
        finally:
            await runtime.close()

    app = FastAPI(title="VideoEdit dual-service gateway", lifespan=lifespan)
    app.state.runtime = runtime

    @app.get("/health")
    async def health():
        return await runtime.health_snapshot()

    @app.post("/v1/videos/repairs")
    async def submit(request: Request):
        try:
            payload = await request.json()
        except Exception as error:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from error
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object")
        task = await runtime.enqueue(payload)
        return {
            "code": 0,
            "message": "queued",
            "task_id": task["task_id"],
            "status": "queued",
            "variant": task["variant"],
        }

    @app.get("/v1/videos/{task_id}")
    async def retrieve(task_id: str):
        task = await runtime._store_call("get", task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Video not found")
        task = await runtime.refresh_task(task)
        return await asyncio.to_thread(_task_response, task, runtime.store)

    @app.get("/v1/videos/{task_id}/progress")
    async def progress(task_id: str):
        task = await runtime._store_call("get", task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Video not found")
        task = await runtime.refresh_task(task)
        response = await asyncio.to_thread(_task_response, task, runtime.store)
        return {
            key: response.get(key)
            for key in (
                "id",
                "variant",
                "status",
                "progress",
                "queue_position",
                "file_path",
                "url",
                "error",
                "reason",
                "callback_status",
                "callback_error",
                "callback_attempts",
            )
            if key in response
        }

    @app.delete("/v1/videos/{task_id}")
    async def cancel(task_id: str):
        task = await runtime._store_call("get", task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Video not found")
        task = await runtime.cancel(task)
        return await asyncio.to_thread(_task_response, task, runtime.store)

    @app.get("/admin/queue")
    async def admin_queue(
        status: str | None = None,
        limit: int = Query(100, ge=1, le=1000),
    ):
        tasks = await runtime._store_call("list_tasks", status=status, limit=limit)
        return {
            "counts": await runtime._store_call("counts"),
            "tasks": [_admin_task(task) for task in tasks],
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--queue-db")
    parser.add_argument("--normal-url")
    parser.add_argument("--dmd-url")
    args = parser.parse_args()

    env_config = GatewayConfig.from_env()
    config = GatewayConfig(
        queue_db=args.queue_db or env_config.queue_db,
        normal_url=args.normal_url or env_config.normal_url,
        dmd_url=args.dmd_url or env_config.dmd_url,
        poll_interval=env_config.poll_interval,
        health_timeout=env_config.health_timeout,
    )
    uvicorn.run(create_app(config), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
