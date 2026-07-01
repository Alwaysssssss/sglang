import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.multimodal_gen.runtime.entrypoints import http_server
from sglang.multimodal_gen.runtime.entrypoints.openai import video_repair_shared
from sglang.multimodal_gen.runtime.entrypoints.openai import vividvr_flowcut_api
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import VideoRepairRequest
from sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_protocol import (
    FlowCutMinIOConfig,
)


def _make_test_client():
    app = FastAPI()
    app.include_router(vividvr_flowcut_api.router)
    return TestClient(app)


def _make_vivid_server_args(tmp_path, *, prompt_file=None, **overrides):
    prompt_file = prompt_file or tmp_path / "prompt.txt"
    prompt_file.write_text("restore the video", encoding="utf-8")
    values = {
        "input_save_path": str(tmp_path / "flowcut_work"),
        "output_path": str(tmp_path / "outputs"),
        "prompt_file_path": str(prompt_file),
        "pipeline_config": SimpleNamespace(default_prompt_file_path=str(prompt_file)),
        "model_id": "vividvr",
        "pipeline_class_name": "CogVideoXVividVRControlNetPipeline",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_create_app_registers_flowcut_route_from_dedicated_router(tmp_path):
    app = http_server.create_app(_make_vivid_server_args(tmp_path))

    flowcut_routes = [
        route
        for route in app.routes
        if getattr(route, "path", None) == "/v1/videos/repairs/flowcut"
    ]
    cancel_routes = [
        route
        for route in app.routes
        if getattr(route, "path", None) == "/v1/videos/repairs/flowcut/{video_id}"
    ]
    progress_routes = [
        route
        for route in app.routes
        if getattr(route, "path", None)
        == "/v1/videos/repairs/flowcut/{video_id}/progress"
    ]

    assert len(flowcut_routes) == 1
    assert flowcut_routes[0].endpoint.__module__.endswith("vividvr_flowcut_api")
    assert len(cancel_routes) == 2
    assert {
        tuple(sorted(getattr(route, "methods", set())))
        for route in cancel_routes
    } == {("DELETE",), ("GET",)}
    assert all(
        route.endpoint.__module__.endswith("vividvr_flowcut_api")
        for route in cancel_routes
    )
    assert len(progress_routes) == 1
    assert progress_routes[0].endpoint.__module__.endswith("vividvr_flowcut_api")


def _patch_sampling(monkeypatch, tmp_path, *, captured_kwargs=None):
    captured_kwargs = captured_kwargs if captured_kwargs is not None else {}

    monkeypatch.setattr(
        vividvr_flowcut_api.VividVRSamplingParams,
        "from_user_kwargs",
        staticmethod(
            lambda server_args, **kwargs: captured_kwargs.update(kwargs)
            or SimpleNamespace(
                output_file_path=lambda: str(
                    Path(kwargs["output_path"]) / kwargs["output_file_name"]
                )
            )
        ),
    )
    monkeypatch.setattr(
        vividvr_flowcut_api,
        "prepare_request",
        lambda server_args, sampling_params: "prepared-batch",
    )
    return captured_kwargs


def test_build_vividvr_kwargs_keeps_phase_e_defaults_optional(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("restore the video", encoding="utf-8")
    server_args = SimpleNamespace(
        prompt_file_path=str(prompt_file),
        pipeline_config=SimpleNamespace(default_prompt_file_path=str(prompt_file)),
    )
    req = VideoRepairRequest(
        task_id="job-1",
        video_input_path="/tmp/input.mp4",
        caption_file_path="/tmp/caption.txt",
        num_inference_steps=20,
        seed=42,
    )

    kwargs = video_repair_shared.build_vividvr_repair_kwargs(
        request_id="job-1",
        req=req,
        server_args=server_args,
        video_input_path="/tmp/input.mp4",
        output_dir=str(tmp_path),
        output_file_name="job-1.mp4",
    )

    assert kwargs["request_id"] == "job-1"
    assert kwargs["video_input_path"] == "/tmp/input.mp4"
    assert kwargs["caption_source"] == "caption_file"
    assert kwargs["caption_file_path"] == "/tmp/caption.txt"
    assert "prompt" not in kwargs
    assert "prompt_file_path" not in kwargs
    assert "reference_video_path" not in kwargs
    assert kwargs["num_inference_steps"] == 20
    assert kwargs["seed"] == 42


def test_build_vividvr_kwargs_caption_file_mode_does_not_require_prompt_file(tmp_path):
    server_args = SimpleNamespace(
        prompt_file_path=None,
        pipeline_config=SimpleNamespace(default_prompt_file_path=None),
    )
    req = VideoRepairRequest(
        task_id="job-1",
        video_input_path="/tmp/input.mp4",
        caption_file_path="/tmp/caption.txt",
        seed=42,
    )

    kwargs = video_repair_shared.build_vividvr_repair_kwargs(
        request_id="job-1",
        req=req,
        server_args=server_args,
        video_input_path="/tmp/input.mp4",
        output_dir=str(tmp_path),
        output_file_name="job-1.mp4",
    )

    assert kwargs["caption_source"] == "caption_file"
    assert kwargs["caption_file_path"] == "/tmp/caption.txt"
    assert "prompt" not in kwargs
    assert "prompt_file_path" not in kwargs


def test_build_vividvr_kwargs_forwards_original_upscale_contract(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("restore the video", encoding="utf-8")
    server_args = SimpleNamespace(
        prompt_file_path=str(prompt_file),
        pipeline_config=SimpleNamespace(default_prompt_file_path=str(prompt_file)),
    )
    req = VideoRepairRequest(
        task_id="job-1",
        video_input_path="/tmp/input.mp4",
        seed=42,
        upscale=0.0,
    )

    kwargs = video_repair_shared.build_vividvr_repair_kwargs(
        request_id="job-1",
        req=req,
        server_args=server_args,
        video_input_path="/tmp/input.mp4",
        output_dir=str(tmp_path),
        output_file_name="job-1.mp4",
    )

    assert kwargs["upscale"] == 0.0


def test_flowcut_endpoint_returns_code_2_when_queue_full(monkeypatch, tmp_path):
    class LockedSemaphore:
        def locked(self):
            return True

        async def acquire(self):
            raise AssertionError("busy request must not acquire semaphore")

    monkeypatch.setattr(video_repair_shared, "VIDEOEDIT_SEMAPHORE", LockedSemaphore())
    monkeypatch.setattr(
        vividvr_flowcut_api,
        "get_global_server_args",
        lambda: _make_vivid_server_args(tmp_path),
    )
    client = _make_test_client()

    response = client.post(
        "/v1/videos/repairs/flowcut",
        json={
            "taskId": "busy-task",
            "timeout": -1,
            "callbackUrl": "http://127.0.0.1:9000/callback",
            "video_input_path": str(tmp_path / "in.mp4"),
        },
    )

    assert response.status_code == 200
    assert response.json() == {"code": 2, "message": "A task is running."}


def test_flowcut_endpoint_returns_code_1_for_missing_input(monkeypatch):
    class AvailableSemaphore:
        def locked(self):
            return False

        async def acquire(self):
            raise AssertionError("should not acquire semaphore for invalid request")

    monkeypatch.setattr(video_repair_shared, "VIDEOEDIT_SEMAPHORE", AvailableSemaphore())
    client = _make_test_client()

    response = client.post(
        "/v1/videos/repairs/flowcut",
        json={
            "taskId": "invalid-task",
            "timeout": -1,
            "callbackUrl": "http://127.0.0.1:9000/callback",
        },
    )

    assert response.status_code == 200
    assert response.json()["code"] == 1
    assert "video_input_path or video_url is required" in response.json()["message"]


def test_flowcut_endpoint_returns_code_1_for_invalid_json(monkeypatch):
    class AvailableSemaphore:
        def locked(self):
            return False

        async def acquire(self):
            raise AssertionError("should not acquire semaphore for invalid request")

    monkeypatch.setattr(video_repair_shared, "VIDEOEDIT_SEMAPHORE", AvailableSemaphore())
    client = _make_test_client()

    response = client.post(
        "/v1/videos/repairs/flowcut",
        content="{bad json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    assert response.json()["code"] == 1
    assert "invalid request" in response.json()["message"]


@pytest.mark.parametrize("field_name", ["reference_video_path", "referenceVideoPath"])
def test_flowcut_endpoint_rejects_reference_video_path_field(field_name):
    client = _make_test_client()

    response = client.post(
        "/v1/videos/repairs/flowcut",
        json={
            "taskId": "task-with-reference-field",
            "timeout": -1,
            "callbackUrl": "http://127.0.0.1:9000/callback",
            "video_input_path": "/tmp/input.mp4",
            field_name: "/tmp/reference.mp4",
        },
    )

    assert response.status_code == 200
    assert response.json()["code"] == 1
    assert field_name in response.json()["message"]


def test_invalid_flowcut_request_with_task_id_is_persisted_as_failed(tmp_path):
    client = _make_test_client()

    async def run_test():
        vividvr_flowcut_api.VIDEO_STORE._items.clear()
        response = client.post(
            "/v1/videos/repairs/flowcut",
            json={
                "taskId": "bad-task",
                "timeout": -2,
                "callbackUrl": "http://127.0.0.1:9000/callback",
            },
        )
        job = await vividvr_flowcut_api.VIDEO_STORE.get("bad-task")
        return response, job

    response, job = asyncio.run(run_test())

    assert response.status_code == 200
    assert response.json()["code"] == 1
    assert job is not None
    assert job["status"] == "failed"
    assert "timeout" in job["reason"]


def test_flowcut_endpoint_rejects_non_vivid_pipeline_without_dispatch(
    monkeypatch, tmp_path
):
    acquired = {"value": False}
    scheduled = {"value": False}
    input_video = tmp_path / "in.mp4"
    input_video.write_bytes(b"video")

    class AvailableSemaphore:
        def locked(self):
            return False

        async def acquire(self):
            acquired["value"] = True

        def release(self):
            acquired["value"] = False

    monkeypatch.setattr(video_repair_shared, "VIDEOEDIT_SEMAPHORE", AvailableSemaphore())
    monkeypatch.setattr(
        vividvr_flowcut_api,
        "get_global_server_args",
        lambda: _make_vivid_server_args(
            tmp_path,
            model_id="wan",
            pipeline_class_name="WanVideoEditPipeline",
        ),
    )
    monkeypatch.setattr(
        vividvr_flowcut_api.asyncio,
        "create_task",
        lambda coro: scheduled.update(value=True),
    )
    client = _make_test_client()

    response = client.post(
        "/v1/videos/repairs/flowcut",
        json={
            "taskId": "task-non-vivid",
            "timeout": -1,
            "callbackUrl": "http://127.0.0.1:9000/callback",
            "video_input_path": str(input_video),
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "code": 1,
        "message": "FlowCut repair endpoint requires Vivid-VR pipeline",
    }
    assert acquired["value"] is False
    assert scheduled["value"] is False


def test_flowcut_endpoint_accepts_and_schedules_background_job(monkeypatch, tmp_path):
    scheduled = {}
    acquired = {"value": False}
    input_video = tmp_path / "in.mov"
    input_video.write_bytes(b"video")

    class AvailableSemaphore:
        def locked(self):
            return False

        async def acquire(self):
            acquired["value"] = True

        def release(self):
            acquired["value"] = False

    monkeypatch.setattr(video_repair_shared, "VIDEOEDIT_SEMAPHORE", AvailableSemaphore())
    monkeypatch.setattr(
        vividvr_flowcut_api,
        "get_global_server_args",
        lambda: _make_vivid_server_args(tmp_path),
    )
    captured_kwargs = _patch_sampling(monkeypatch, tmp_path)

    def fake_create_task(coro):
        scheduled["coro_name"] = coro.cr_code.co_name
        coro.close()
        return None

    monkeypatch.setattr(vividvr_flowcut_api.asyncio, "create_task", fake_create_task)

    client = _make_test_client()
    response = client.post(
        "/v1/videos/repairs/flowcut",
        json={
            "taskId": "task-1",
            "timeout": -1,
            "callbackUrl": "http://127.0.0.1:9000/tasks/task-1/callback",
            "video_input_path": str(input_video),
            "caption_file_path": "/tmp/caption.txt",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"code": 0, "message": "ok"}
    assert acquired["value"] is True
    assert scheduled["coro_name"] == "_dispatch_vividvr_flowcut_video_repair_job_async"
    assert captured_kwargs["video_input_path"].endswith(
        "flowcut_work/task-1/inputs/input.mov"
    )
    assert captured_kwargs["output_path"].endswith("flowcut_work/task-1/outputs")
    assert captured_kwargs["output_file_name"] == "task-1.mov"
    assert "reference_video_path" not in captured_kwargs


def test_flowcut_endpoint_persists_default_output_object_key_for_minio_request(
    monkeypatch, tmp_path
):
    scheduled = {}
    input_video = tmp_path / "in.mov"
    input_video.write_bytes(b"video")

    class AvailableSemaphore:
        def locked(self):
            return False

        async def acquire(self):
            pass

        def release(self):
            pass

    monkeypatch.setattr(video_repair_shared, "VIDEOEDIT_SEMAPHORE", AvailableSemaphore())
    monkeypatch.setattr(
        vividvr_flowcut_api,
        "get_global_server_args",
        lambda: _make_vivid_server_args(tmp_path),
    )
    _patch_sampling(monkeypatch, tmp_path)

    async def fake_dispatch(*args, **kwargs):
        return None

    def fake_create_task(coro):
        scheduled["coro_name"] = coro.cr_code.co_name
        coro.close()
        return None

    monkeypatch.setattr(
        vividvr_flowcut_api,
        "_dispatch_vividvr_flowcut_video_repair_job_async",
        fake_dispatch,
    )
    monkeypatch.setattr(vividvr_flowcut_api.asyncio, "create_task", fake_create_task)
    client = _make_test_client()

    async def run_test():
        vividvr_flowcut_api.VIDEO_STORE._items.clear()
        response = client.post(
            "/v1/videos/repairs/flowcut",
            json={
                "taskId": "task-minio-default-key",
                "timeout": -1,
                "callbackUrl": "http://127.0.0.1:9000/callback",
                "video_input_path": str(input_video),
                "minioConfig": {
                    "endpoint": "minio.example.com:9000",
                    "bucketName": "flowcut",
                    "accessKey": "ak",
                    "secretKey": "sk",
                },
            },
        )
        job = await vividvr_flowcut_api.VIDEO_STORE.get("task-minio-default-key")
        return response, job

    response, job = asyncio.run(run_test())

    assert response.status_code == 200
    assert response.json() == {"code": 0, "message": "ok"}
    assert scheduled["coro_name"] == "fake_dispatch"
    assert job is not None
    assert job["output_bucket"] == "flowcut"
    assert job["output_object_key"].endswith("_task-minio-default-key.mov")


def test_flowcut_endpoint_uses_temp_workdir_when_input_save_path_is_unset(
    monkeypatch, tmp_path
):
    scheduled = {}
    input_video = tmp_path / "in.mov"
    input_video.write_bytes(b"video")

    class AvailableSemaphore:
        def locked(self):
            return False

        async def acquire(self):
            pass

        def release(self):
            pass

    monkeypatch.setattr(video_repair_shared, "VIDEOEDIT_SEMAPHORE", AvailableSemaphore())
    monkeypatch.setattr(
        vividvr_flowcut_api,
        "get_global_server_args",
        lambda: _make_vivid_server_args(
            tmp_path,
            input_save_path=None,
            output_path=str(tmp_path / "persistent-output"),
        ),
    )
    captured_kwargs = _patch_sampling(monkeypatch, tmp_path)
    monkeypatch.setattr(
        vividvr_flowcut_api.tempfile,
        "mkdtemp",
        lambda prefix: str(tmp_path / "flowcut-temp-base"),
    )

    def fake_create_task(coro):
        scheduled["coro_name"] = coro.cr_code.co_name
        coro.close()
        return None

    monkeypatch.setattr(vividvr_flowcut_api.asyncio, "create_task", fake_create_task)
    client = _make_test_client()

    response = client.post(
        "/v1/videos/repairs/flowcut",
        json={
            "taskId": "task-temp-input",
            "timeout": -1,
            "callbackUrl": "http://127.0.0.1:9000/tasks/task-temp-input/callback",
            "video_input_path": str(input_video),
        },
    )

    assert response.status_code == 200
    assert response.json() == {"code": 0, "message": "ok"}
    assert scheduled["coro_name"] == "_dispatch_vividvr_flowcut_video_repair_job_async"
    assert captured_kwargs["video_input_path"].endswith(
        "flowcut-temp-base/task-temp-input/inputs/input.mov"
    )
    assert captured_kwargs["output_path"].endswith(
        "flowcut-temp-base/task-temp-input/outputs"
    )
    assert captured_kwargs["output_file_name"] == "task-temp-input.mov"


def test_flowcut_endpoint_generates_caption_when_bridge_enabled(monkeypatch, tmp_path):
    scheduled = {}
    input_video = tmp_path / "in.mp4"
    input_video.write_bytes(b"video")

    class AvailableSemaphore:
        def locked(self):
            return False

        async def acquire(self):
            pass

        def release(self):
            pass

    monkeypatch.setattr(video_repair_shared, "VIDEOEDIT_SEMAPHORE", AvailableSemaphore())
    monkeypatch.setattr(
        vividvr_flowcut_api,
        "get_global_server_args",
        lambda: _make_vivid_server_args(
            tmp_path,
            vividvr_caption_bridge=True,
            vividvr_caption_sidecar_url="http://127.0.0.1:31200",
            vividvr_caption_work_dir=str(tmp_path / "caption_sidecars"),
            vividvr_caption_sidecar_timeout=30.0,
        ),
    )
    async def fake_ensure_caption_file(**kwargs):
        caption_path = tmp_path / "caption_sidecars" / f"{kwargs['request_id']}.txt"
        caption_path.parent.mkdir(parents=True, exist_ok=True)
        caption_path.write_text("caption 0\n", encoding="utf-8")
        return str(caption_path)

    monkeypatch.setattr(
        video_repair_shared,
        "ensure_vividvr_caption_file",
        fake_ensure_caption_file,
    )

    captured_kwargs = _patch_sampling(monkeypatch, tmp_path)

    def fake_create_task(coro):
        scheduled["coro_name"] = coro.cr_code.co_name
        coro.close()
        return None

    monkeypatch.setattr(vividvr_flowcut_api.asyncio, "create_task", fake_create_task)

    client = _make_test_client()
    response = client.post(
        "/v1/videos/repairs/flowcut",
        json={
            "taskId": "task-auto",
            "timeout": -1,
            "callbackUrl": "http://127.0.0.1:9000/callback",
            "video_input_path": str(input_video),
        },
    )

    assert response.status_code == 200
    assert response.json()["code"] == 0
    assert captured_kwargs["caption_source"] == "caption_file"
    assert captured_kwargs["caption_file_path"].endswith("task-auto.txt")
    assert scheduled["coro_name"] == "_dispatch_vividvr_flowcut_video_repair_job_async"


def test_flowcut_endpoint_logs_accepted_task(monkeypatch, tmp_path, caplog):
    input_video = tmp_path / "in.mp4"
    input_video.write_bytes(b"video")

    class AvailableSemaphore:
        def locked(self):
            return False

        async def acquire(self):
            pass

        def release(self):
            pass

    monkeypatch.setattr(video_repair_shared, "VIDEOEDIT_SEMAPHORE", AvailableSemaphore())
    monkeypatch.setattr(
        vividvr_flowcut_api,
        "get_global_server_args",
        lambda: _make_vivid_server_args(tmp_path),
    )
    _patch_sampling(monkeypatch, tmp_path)

    def fake_create_task(coro):
        coro.close()
        return None

    monkeypatch.setattr(vividvr_flowcut_api.asyncio, "create_task", fake_create_task)
    client = _make_test_client()

    with caplog.at_level(logging.INFO, logger=vividvr_flowcut_api.logger.name):
        response = client.post(
            "/v1/videos/repairs/flowcut",
            json={
                "taskId": "task-log",
                "timeout": -1,
                "callbackUrl": "http://127.0.0.1:9000/tasks/task-log/callback",
                "video_input_path": str(input_video),
            },
        )

    assert response.json() == {"code": 0, "message": "ok"}
    assert "FlowCut video repair accepted task_id=task-log" in caplog.text


def test_dispatch_vividvr_flowcut_job_posts_stage_and_final_callbacks(
    monkeypatch, tmp_path
):
    callbacks = []
    output_path = tmp_path / "task-1" / "outputs" / "task-1.mp4"
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"result")

    async def fake_post_flowcut_callback(callback_url, payload, **kwargs):
        callbacks.append(payload)

    async def fake_run_video_generation_job(batch):
        return SimpleNamespace(
            save_file_path=str(output_path),
            result=SimpleNamespace(inference_time_s=1.25),
        )

    class ReleaseTrackingSemaphore:
        def __init__(self):
            self.released = False

        def release(self):
            self.released = True

    monkeypatch.setattr(
        vividvr_flowcut_api,
        "post_flowcut_callback",
        fake_post_flowcut_callback,
    )
    monkeypatch.setattr(
        vividvr_flowcut_api,
        "run_video_generation_job",
        fake_run_video_generation_job,
    )
    monkeypatch.setattr(
        video_repair_shared,
        "VIDEOEDIT_SEMAPHORE",
        ReleaseTrackingSemaphore(),
    )
    storage = vividvr_flowcut_api.VividVRFlowCutStorage(
        base_dir=tmp_path,
        request_id="task-1",
    )

    async def run_test():
        await vividvr_flowcut_api.VIDEO_STORE.upsert(
            "task-1",
            {"id": "task-1", "status": "queued", "progress": 0},
        )

        await vividvr_flowcut_api._dispatch_vividvr_flowcut_video_repair_job_async(
            "task-1",
            batch="prepared",
            callback_url="http://127.0.0.1:9000/callback",
            storage=storage,
            timeout=-1,
        )

    asyncio.run(run_test())

    assert [callback["status"] for callback in callbacks] == [
        "running",
        "succeeded",
    ]
    assert [callback["reason"] for callback in callbacks] == [
        "uploading_result",
        "succeeded",
    ]
    final_output = json.loads(callbacks[-1]["output"])
    assert final_output == {
        "result_url": str(output_path),
        "duration": 1.25,
    }
    assert set(final_output) == {"result_url", "duration"}
    assert "file_path" not in final_output
    assert "gen_video_url" not in final_output
    assert video_repair_shared.VIDEOEDIT_SEMAPHORE.released is True


def test_dispatch_vividvr_flowcut_job_records_callback_bookkeeping(
    monkeypatch, tmp_path
):
    output_path = tmp_path / "task-callback" / "outputs" / "task-callback.mp4"
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"result")

    async def fake_post_flowcut_callback(callback_url, payload, **kwargs):
        return 1

    async def fake_run_video_generation_job(batch):
        return SimpleNamespace(
            save_file_path=str(output_path),
            result=SimpleNamespace(inference_time_s=1.25),
        )

    class ReleaseTrackingSemaphore:
        def release(self):
            pass

    monkeypatch.setattr(
        vividvr_flowcut_api,
        "post_flowcut_callback",
        fake_post_flowcut_callback,
    )
    monkeypatch.setattr(
        vividvr_flowcut_api,
        "run_video_generation_job",
        fake_run_video_generation_job,
    )
    monkeypatch.setattr(
        video_repair_shared,
        "VIDEOEDIT_SEMAPHORE",
        ReleaseTrackingSemaphore(),
    )
    storage = vividvr_flowcut_api.VividVRFlowCutStorage(
        base_dir=tmp_path,
        request_id="task-callback",
    )

    async def run_test():
        await vividvr_flowcut_api.VIDEO_STORE.upsert(
            "task-callback",
            {"id": "task-callback", "status": "queued", "progress": 0},
        )
        await vividvr_flowcut_api._dispatch_vividvr_flowcut_video_repair_job_async(
            "task-callback",
            batch="prepared",
            callback_url="http://127.0.0.1:9000/callback",
            storage=storage,
            timeout=-1,
        )
        return await vividvr_flowcut_api.VIDEO_STORE.get("task-callback")

    job = asyncio.run(run_test())

    assert job is not None
    assert job["callback_status"] == "succeeded"
    assert job["callback_error"] is None
    assert job["callback_attempts"] == 1


def test_monitor_vividvr_denoise_progress_posts_runtime_progress(tmp_path):
    callbacks = []

    async def fake_post_flowcut_callback(callback_url, payload, **kwargs):
        callbacks.append(payload)

    reporter = vividvr_flowcut_api.VividVRFlowCutProgressReporter(
        task_id="task-denoise",
        callback_url="http://127.0.0.1:9000/callback",
        post_callback=fake_post_flowcut_callback,
    )
    batch = SimpleNamespace(
        sampling_params=SimpleNamespace(runtime_progress=None),
    )

    async def fake_generation():
        for progress in (0.10, 0.50, 1.0):
            batch.sampling_params.runtime_progress = progress
            await asyncio.sleep(0.01)
        return SimpleNamespace(save_file_path=str(tmp_path / "out.mp4"))

    async def run_test():
        await vividvr_flowcut_api.VIDEO_STORE.upsert(
            "task-denoise",
            {"id": "task-denoise", "status": "queued", "progress": 0},
        )
        generation_task = asyncio.create_task(fake_generation())
        await vividvr_flowcut_api._monitor_vividvr_denoise_progress(
            "task-denoise",
            batch,
            reporter,
            generation_task,
            poll_interval_s=0.001,
        )
        await generation_task

    asyncio.run(run_test())

    assert [callback["reason"] for callback in callbacks] == [
        "denoising",
        "denoising",
        "denoising",
    ]
    assert [callback["progress"] for callback in callbacks] == [14.0, 50.0, 95.0]


def test_monitor_vividvr_denoise_progress_reads_progress_file(tmp_path):
    callbacks = []
    progress_path = tmp_path / "runtime_progress.json"

    async def fake_post_flowcut_callback(callback_url, payload, **kwargs):
        callbacks.append(payload)

    reporter = vividvr_flowcut_api.VividVRFlowCutProgressReporter(
        task_id="task-denoise-file",
        callback_url="http://127.0.0.1:9000/callback",
        post_callback=fake_post_flowcut_callback,
    )
    batch = SimpleNamespace(sampling_params=SimpleNamespace(runtime_progress=None))

    async def fake_generation():
        for progress in (0.10, 0.50, 1.0):
            progress_path.write_text(
                json.dumps({"runtime_progress": progress}),
                encoding="utf-8",
            )
            await asyncio.sleep(0.01)

    async def run_test():
        await vividvr_flowcut_api.VIDEO_STORE.upsert(
            "task-denoise-file",
            {"id": "task-denoise-file", "status": "queued", "progress": 0},
        )
        generation_task = asyncio.create_task(fake_generation())
        await vividvr_flowcut_api._monitor_vividvr_denoise_progress(
            "task-denoise-file",
            batch,
            reporter,
            generation_task,
            progress_path=str(progress_path),
            poll_interval_s=0.001,
        )
        await generation_task

    asyncio.run(run_test())

    assert [callback["reason"] for callback in callbacks] == [
        "denoising",
        "denoising",
        "denoising",
    ]
    assert [callback["progress"] for callback in callbacks] == [14.0, 50.0, 95.0]


def test_dispatch_vividvr_flowcut_job_uses_minio_and_deletes_local_output(
    monkeypatch, tmp_path
):
    callbacks = []
    output_path = tmp_path / "task-minio" / "outputs" / "task-minio.mp4"
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"result")

    async def fake_post_flowcut_callback(callback_url, payload, **kwargs):
        callbacks.append(payload)

    async def fake_run_video_generation_job(batch):
        return SimpleNamespace(
            save_file_path=str(output_path),
            result=SimpleNamespace(inference_time_s=None),
        )

    captured_upload = {}

    async def fake_upload_result(
        local_path,
        minio_config,
        *,
        object_key=None,
        bucket_name=None,
    ):
        captured_upload["object_key"] = object_key
        captured_upload["bucket_name"] = bucket_name
        Path(local_path).unlink()
        return "http://minio/target-bucket/custom/task-minio.mp4"

    monkeypatch.setattr(
        vividvr_flowcut_api,
        "post_flowcut_callback",
        fake_post_flowcut_callback,
    )
    monkeypatch.setattr(
        vividvr_flowcut_api,
        "run_video_generation_job",
        fake_run_video_generation_job,
    )
    storage = vividvr_flowcut_api.VividVRFlowCutStorage(
        base_dir=tmp_path,
        request_id="task-minio",
    )
    monkeypatch.setattr(storage, "upload_result", fake_upload_result)

    class ReleaseTrackingSemaphore:
        def release(self):
            pass

    monkeypatch.setattr(
        video_repair_shared,
        "VIDEOEDIT_SEMAPHORE",
        ReleaseTrackingSemaphore(),
    )

    async def run_test():
        await vividvr_flowcut_api._dispatch_vividvr_flowcut_video_repair_job_async(
            "task-minio",
            batch="prepared",
            callback_url="http://127.0.0.1:9000/callback",
            storage=storage,
            minio_config=FlowCutMinIOConfig(
                endpoint="minio:9000",
                bucket_name="flowcut",
                access_key="ak",
                secret_key="sk",
            ),
            output_object_key="custom/task-minio.mp4",
            output_bucket="target-bucket",
            timeout=-1,
        )

    asyncio.run(run_test())

    assert callbacks[-1]["status"] == "succeeded"
    assert json.loads(callbacks[-1]["output"]) == {
        "result_url": "http://minio/target-bucket/custom/task-minio.mp4"
    }
    assert captured_upload == {
        "object_key": "custom/task-minio.mp4",
        "bucket_name": "target-bucket",
    }
    assert not output_path.exists()


def test_dispatch_vividvr_flowcut_job_cleans_temp_workdir_after_externalized_result(
    monkeypatch, tmp_path
):
    callbacks = []
    base_dir = tmp_path / "temp-base"
    storage = vividvr_flowcut_api.VividVRFlowCutStorage(
        base_dir=base_dir,
        request_id="task-cleanup",
    )
    output_path = Path(storage.output_file_path("task-cleanup.mp4"))
    output_path.write_bytes(b"result")

    async def fake_post_flowcut_callback(callback_url, payload, **kwargs):
        callbacks.append(payload)

    async def fake_run_video_generation_job(batch):
        return SimpleNamespace(
            save_file_path=str(output_path),
            result=SimpleNamespace(inference_time_s=1.0),
        )

    async def fake_upload_result(local_path, minio_config, **kwargs):
        Path(local_path).unlink()
        return "http://minio/flowcut/task-cleanup.mp4"

    class ReleaseTrackingSemaphore:
        def release(self):
            pass

    monkeypatch.setattr(
        vividvr_flowcut_api,
        "post_flowcut_callback",
        fake_post_flowcut_callback,
    )
    monkeypatch.setattr(
        vividvr_flowcut_api,
        "run_video_generation_job",
        fake_run_video_generation_job,
    )
    monkeypatch.setattr(storage, "upload_result", fake_upload_result)
    monkeypatch.setattr(
        video_repair_shared,
        "VIDEOEDIT_SEMAPHORE",
        ReleaseTrackingSemaphore(),
    )

    async def run_test():
        await vividvr_flowcut_api._dispatch_vividvr_flowcut_video_repair_job_async(
            "task-cleanup",
            batch="prepared",
            callback_url="http://127.0.0.1:9000/callback",
            storage=storage,
            minio_config=FlowCutMinIOConfig(
                endpoint="minio:9000",
                bucket_name="flowcut",
                access_key="ak",
                secret_key="sk",
            ),
            cleanup_workdir_on_finish=True,
            cleanup_base_dir_on_finish=True,
            timeout=-1,
        )

    asyncio.run(run_test())

    assert callbacks[-1]["status"] == "succeeded"
    assert not storage.workdir.exists()
    assert not storage.base_dir.exists()


def test_dispatch_vividvr_flowcut_job_keeps_temp_workdir_for_local_only_result(
    monkeypatch, tmp_path
):
    callbacks = []
    base_dir = tmp_path / "temp-base-local"
    storage = vividvr_flowcut_api.VividVRFlowCutStorage(
        base_dir=base_dir,
        request_id="task-keep-local",
    )
    output_path = Path(storage.output_file_path("task-keep-local.mp4"))
    output_path.write_bytes(b"result")

    async def fake_post_flowcut_callback(callback_url, payload, **kwargs):
        callbacks.append(payload)

    async def fake_run_video_generation_job(batch):
        return SimpleNamespace(
            save_file_path=str(output_path),
            result=SimpleNamespace(inference_time_s=1.0),
        )

    class ReleaseTrackingSemaphore:
        def release(self):
            pass

    monkeypatch.setattr(
        vividvr_flowcut_api,
        "post_flowcut_callback",
        fake_post_flowcut_callback,
    )
    monkeypatch.setattr(
        vividvr_flowcut_api,
        "run_video_generation_job",
        fake_run_video_generation_job,
    )
    monkeypatch.setattr(
        video_repair_shared,
        "VIDEOEDIT_SEMAPHORE",
        ReleaseTrackingSemaphore(),
    )

    async def run_test():
        await vividvr_flowcut_api._dispatch_vividvr_flowcut_video_repair_job_async(
            "task-keep-local",
            batch="prepared",
            callback_url="http://127.0.0.1:9000/callback",
            storage=storage,
            cleanup_workdir_on_finish=False,
            cleanup_base_dir_on_finish=False,
            timeout=-1,
        )

    asyncio.run(run_test())

    assert callbacks[-1]["status"] == "succeeded"
    assert storage.workdir.exists()
    assert output_path.exists()


def test_dispatch_vividvr_flowcut_job_posts_failed_callback_and_keeps_local_output(
    monkeypatch, tmp_path
):
    callbacks = []
    output_path = tmp_path / "task-fail" / "outputs" / "task-fail.mp4"
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"result")

    async def fake_post_flowcut_callback(callback_url, payload, **kwargs):
        callbacks.append(payload)

    async def fake_run_video_generation_job(batch):
        return SimpleNamespace(
            save_file_path=str(output_path),
            result=SimpleNamespace(inference_time_s=2.0),
        )

    async def fake_upload_result(local_path, minio_config, **kwargs):
        raise RuntimeError("upload failed")

    monkeypatch.setattr(
        vividvr_flowcut_api,
        "post_flowcut_callback",
        fake_post_flowcut_callback,
    )
    monkeypatch.setattr(
        vividvr_flowcut_api,
        "run_video_generation_job",
        fake_run_video_generation_job,
    )
    storage = vividvr_flowcut_api.VividVRFlowCutStorage(
        base_dir=tmp_path,
        request_id="task-fail",
    )
    monkeypatch.setattr(storage, "upload_result", fake_upload_result)

    class ReleaseTrackingSemaphore:
        def release(self):
            pass

    monkeypatch.setattr(
        video_repair_shared,
        "VIDEOEDIT_SEMAPHORE",
        ReleaseTrackingSemaphore(),
    )

    async def run_test():
        await vividvr_flowcut_api._dispatch_vividvr_flowcut_video_repair_job_async(
            "task-fail",
            batch="prepared",
            callback_url="http://127.0.0.1:9000/callback",
            storage=storage,
            minio_config=FlowCutMinIOConfig(
                endpoint="minio:9000",
                bucket_name="flowcut",
                access_key="ak",
                secret_key="sk",
            ),
            timeout=-1,
        )

    asyncio.run(run_test())

    assert callbacks[-1] == {
        "status": "failed",
        "progress": 98.0,
        "reason": "upload failed",
        "output": "",
    }
    assert output_path.exists()


def test_dispatch_vividvr_flowcut_timeout_keeps_semaphore_until_generation_finishes(
    monkeypatch, tmp_path
):
    callbacks = []
    release_events = []
    generation_started = asyncio.Event()
    generation_can_finish = asyncio.Event()
    failed_callback_seen = asyncio.Event()

    async def fake_post_flowcut_callback(callback_url, payload, **kwargs):
        callbacks.append(payload)
        if payload["status"] == "failed":
            failed_callback_seen.set()

    async def fake_run_video_generation_job(batch):
        generation_started.set()
        await generation_can_finish.wait()
        return SimpleNamespace(
            save_file_path=str(tmp_path / "late-result.mp4"),
            result=SimpleNamespace(inference_time_s=10.0),
        )

    class ReleaseTrackingSemaphore:
        def release(self):
            release_events.append("released")

    monkeypatch.setattr(
        vividvr_flowcut_api,
        "post_flowcut_callback",
        fake_post_flowcut_callback,
    )
    monkeypatch.setattr(
        vividvr_flowcut_api,
        "run_video_generation_job",
        fake_run_video_generation_job,
    )
    monkeypatch.setattr(
        video_repair_shared,
        "VIDEOEDIT_SEMAPHORE",
        ReleaseTrackingSemaphore(),
    )
    storage = vividvr_flowcut_api.VividVRFlowCutStorage(
        base_dir=tmp_path,
        request_id="task-timeout",
    )

    async def run_test():
        await vividvr_flowcut_api.VIDEO_STORE.upsert(
            "task-timeout",
            {"id": "task-timeout", "status": "queued", "progress": 0},
        )
        task = asyncio.create_task(
            vividvr_flowcut_api._dispatch_vividvr_flowcut_video_repair_job_async(
                "task-timeout",
                batch="prepared",
                callback_url="http://127.0.0.1:9000/callback",
                storage=storage,
                timeout=0.01,
            )
        )
        await asyncio.wait_for(generation_started.wait(), timeout=1.0)
        await asyncio.wait_for(failed_callback_seen.wait(), timeout=1.0)
        assert release_events == []
        generation_can_finish.set()
        await asyncio.wait_for(task, timeout=1.0)

    asyncio.run(run_test())

    failed_callbacks = [
        callback for callback in callbacks if callback["status"] == "failed"
    ]
    assert len(failed_callbacks) == 1
    assert failed_callbacks[0]["reason"] == vividvr_flowcut_api.TASK_TIMEOUT_MESSAGE
    assert release_events == ["released"]


def test_delete_flowcut_job_marks_failed_and_cancels_registered_task(
    monkeypatch, tmp_path
):
    callbacks = []
    request_cancel_path = str(tmp_path / "cancel" / "task-cancelled.cancel")

    class DummyTask:
        def __init__(self):
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

    dummy_task = DummyTask()

    async def fake_post_flowcut_callback(callback_url, payload, **kwargs):
        callbacks.append((callback_url, payload))
        return 1

    client = _make_test_client()
    vividvr_flowcut_api.VIDEO_STORE._items.clear()
    vividvr_flowcut_api._FLOWCUT_TASKS.clear()
    vividvr_flowcut_api._FLOWCUT_TASKS["task-cancelled"] = dummy_task
    monkeypatch.setattr(
        vividvr_flowcut_api,
        "_flowcut_cancel_path",
        lambda task_id: request_cancel_path,
    )
    monkeypatch.setattr(
        vividvr_flowcut_api,
        "post_flowcut_callback",
        fake_post_flowcut_callback,
    )

    async def run_test():
        await vividvr_flowcut_api.VIDEO_STORE.upsert(
            "task-cancelled",
            {
                "id": "task-cancelled",
                "object": "video",
                "model": "VividVR",
                "status": "running",
                "progress": 50,
                "created_at": 1,
                "callback_url": "http://127.0.0.1:9000/callback",
                "request_cancel_path": request_cancel_path,
            },
        )
        response = client.delete("/v1/videos/repairs/flowcut/task-cancelled")
        read_response = client.get("/v1/videos/repairs/flowcut/task-cancelled")
        progress_response = client.get(
            "/v1/videos/repairs/flowcut/task-cancelled/progress"
        )
        job = await vividvr_flowcut_api.VIDEO_STORE.get("task-cancelled")
        return response, read_response, progress_response, job

    response, read_response, progress_response, job = asyncio.run(run_test())

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["error"]["message"] == vividvr_flowcut_api.TASK_TIMEOUT_MESSAGE
    assert response.json()["reason"] == vividvr_flowcut_api.TASK_TIMEOUT_MESSAGE
    assert read_response.status_code == 200
    assert read_response.json()["status"] == "failed"
    assert read_response.json()["reason"] == vividvr_flowcut_api.TASK_TIMEOUT_MESSAGE
    assert progress_response.status_code == 200
    assert progress_response.json()["status"] == "failed"
    assert progress_response.json()["reason"] == vividvr_flowcut_api.TASK_TIMEOUT_MESSAGE
    assert job is not None
    assert job["status"] == "failed"
    assert job["reason"] == vividvr_flowcut_api.TASK_TIMEOUT_MESSAGE
    assert "cancelled_at" in job
    assert job["callback_status"] == "succeeded"
    assert dummy_task.cancelled is True
    assert Path(request_cancel_path).exists()
    assert callbacks == [
        (
            "http://127.0.0.1:9000/callback",
            {
                "status": "failed",
                "progress": 98.0,
                "reason": vividvr_flowcut_api.TASK_TIMEOUT_MESSAGE,
                "output": "",
            },
        )
    ]
