import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.multimodal_gen.runtime.entrypoints.openai import video_api
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import VideoRepairRequest


def _make_test_client():
    app = FastAPI()
    app.include_router(video_api.router)
    return TestClient(app)


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
        reference_video_path="/tmp/reference.mp4",
        num_inference_steps=20,
        seed=42,
    )

    kwargs = video_api._build_vividvr_repair_kwargs(
        request_id="job-1",
        req=req,
        server_args=server_args,
        video_input_path="/tmp/input.mp4",
        output_dir=str(tmp_path),
        output_file_name="job-1.mp4",
    )

    assert kwargs["request_id"] == "job-1"
    assert kwargs["video_input_path"] == "/tmp/input.mp4"
    assert kwargs["prompt"] == "restore the video"
    assert kwargs["caption_source"] == "caption_file"
    assert kwargs["caption_file_path"] == "/tmp/caption.txt"
    assert kwargs["reference_video_path"] == "/tmp/reference.mp4"
    assert kwargs["num_inference_steps"] == 20
    assert kwargs["seed"] == 42


def test_flowcut_endpoint_returns_code_2_when_queue_full(monkeypatch):
    client = _make_test_client()

    class LockedSemaphore:
        def locked(self):
            return True

    monkeypatch.setattr(video_api, "_VIDEOEDIT_SEMAPHORE", LockedSemaphore())

    response = client.post(
        "/v1/videos/repairs/flowcut",
        json={
            "taskId": "busy-task",
            "timeout": -1,
            "callbackUrl": "http://127.0.0.1:9000/callback",
            "video_input_path": "/tmp/in.mp4",
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

    monkeypatch.setattr(video_api, "_VIDEOEDIT_SEMAPHORE", AvailableSemaphore())
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

    monkeypatch.setattr(video_api, "_VIDEOEDIT_SEMAPHORE", AvailableSemaphore())
    client = _make_test_client()

    response = client.post(
        "/v1/videos/repairs/flowcut",
        content="{bad json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    assert response.json()["code"] == 1
    assert "invalid request" in response.json()["message"]


def test_flowcut_endpoint_accepts_and_schedules_background_job(monkeypatch, tmp_path):
    scheduled = {}
    acquired = {"value": False}

    class AvailableSemaphore:
        def locked(self):
            return False

        async def acquire(self):
            acquired["value"] = True

        def release(self):
            acquired["value"] = False

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("restore the video", encoding="utf-8")

    monkeypatch.setattr(video_api, "_VIDEOEDIT_SEMAPHORE", AvailableSemaphore())
    monkeypatch.setattr(
        video_api,
        "get_global_server_args",
        lambda: SimpleNamespace(
            input_save_path=str(tmp_path / "inputs"),
            output_path=str(tmp_path / "outputs"),
            prompt_file_path=str(prompt_file),
            pipeline_config=SimpleNamespace(
                default_prompt_file_path=str(prompt_file),
            ),
            model_id="vividvr",
            pipeline_class_name="CogVideoXVividVRControlNetPipeline",
        ),
    )
    monkeypatch.setattr(
        video_api.VividVRSamplingParams,
        "from_user_kwargs",
        staticmethod(
            lambda server_args, **kwargs: SimpleNamespace(
                output_file_path=lambda: str(tmp_path / "outputs" / "task-1.mp4")
            )
        ),
    )
    monkeypatch.setattr(
        video_api,
        "prepare_request",
        lambda server_args, sampling_params: "prepared-batch",
    )

    def fake_create_task(coro):
        scheduled["coro_name"] = coro.cr_code.co_name
        coro.close()
        return None

    monkeypatch.setattr(video_api.asyncio, "create_task", fake_create_task)

    client = _make_test_client()
    response = client.post(
        "/v1/videos/repairs/flowcut",
        json={
            "taskId": "task-1",
            "timeout": -1,
            "callbackUrl": "http://127.0.0.1:9000/tasks/task-1/callback",
            "video_input_path": "/tmp/in.mp4",
            "caption_file_path": "/tmp/caption.txt",
            "reference_video_path": "/tmp/ref.mp4",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"code": 0, "message": "ok"}
    assert acquired["value"] is True
    assert scheduled["coro_name"] == "_dispatch_flowcut_video_repair_job_async"


def test_flowcut_endpoint_generates_caption_when_bridge_enabled(monkeypatch, tmp_path):
    scheduled = {}

    class AvailableSemaphore:
        def locked(self):
            return False

        async def acquire(self):
            pass

        def release(self):
            pass

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("restore the video", encoding="utf-8")

    monkeypatch.setattr(video_api, "_VIDEOEDIT_SEMAPHORE", AvailableSemaphore())
    monkeypatch.setattr(
        video_api,
        "get_global_server_args",
        lambda: SimpleNamespace(
            input_save_path=str(tmp_path / "inputs"),
            output_path=str(tmp_path / "outputs"),
            prompt_file_path=str(prompt_file),
            pipeline_config=SimpleNamespace(default_prompt_file_path=str(prompt_file)),
            model_id="vividvr",
            pipeline_class_name="CogVideoXVividVRControlNetPipeline",
            vividvr_caption_bridge=True,
            vividvr_caption_sidecar_url="http://127.0.0.1:31200",
            vividvr_caption_work_dir=str(tmp_path / "caption_sidecars"),
            vividvr_caption_sidecar_timeout=30.0,
        ),
    )
    monkeypatch.setattr(
        video_api,
        "build_vividvr_caption_manifest_for_video_path",
        lambda **kwargs: SimpleNamespace(
            expected_caption_count=1,
            write_json=lambda path: Path(path).write_text("{}", encoding="utf-8"),
        ),
    )

    async def fake_request_caption_sidecar(**kwargs):
        Path(kwargs["output_caption_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_caption_path"]).write_text("caption 0\n", encoding="utf-8")
        return SimpleNamespace(
            caption_file_path=kwargs["output_caption_path"],
            caption_count=1,
        )

    monkeypatch.setattr(
        video_api,
        "request_vividvr_caption_sidecar",
        fake_request_caption_sidecar,
    )

    captured_kwargs = {}
    monkeypatch.setattr(
        video_api.VividVRSamplingParams,
        "from_user_kwargs",
        staticmethod(
            lambda server_args, **kwargs: captured_kwargs.update(kwargs)
            or SimpleNamespace(
                output_file_path=lambda: str(tmp_path / "outputs" / "task-auto.mp4")
            )
        ),
    )
    monkeypatch.setattr(
        video_api,
        "prepare_request",
        lambda server_args, sampling_params: "prepared-batch",
    )

    def fake_create_task(coro):
        scheduled["coro_name"] = coro.cr_code.co_name
        coro.close()
        return None

    monkeypatch.setattr(video_api.asyncio, "create_task", fake_create_task)

    client = _make_test_client()
    response = client.post(
        "/v1/videos/repairs/flowcut",
        json={
            "taskId": "task-auto",
            "timeout": -1,
            "callbackUrl": "http://127.0.0.1:9000/callback",
            "video_input_path": "/tmp/in.mp4",
        },
    )

    assert response.status_code == 200
    assert response.json()["code"] == 0
    assert captured_kwargs["caption_source"] == "caption_file"
    assert captured_kwargs["caption_file_path"].endswith("task-auto.txt")
    assert scheduled["coro_name"] == "_dispatch_flowcut_video_repair_job_async"


def test_flowcut_endpoint_logs_accepted_task(monkeypatch, tmp_path, caplog):
    scheduled = {}

    class AvailableSemaphore:
        def locked(self):
            return False

        async def acquire(self):
            pass

        def release(self):
            pass

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("restore the video", encoding="utf-8")

    monkeypatch.setattr(video_api, "_VIDEOEDIT_SEMAPHORE", AvailableSemaphore())
    monkeypatch.setattr(
        video_api,
        "get_global_server_args",
        lambda: SimpleNamespace(
            input_save_path=str(tmp_path / "inputs"),
            output_path=str(tmp_path / "outputs"),
            prompt_file_path=str(prompt_file),
            pipeline_config=SimpleNamespace(
                default_prompt_file_path=str(prompt_file),
            ),
            model_id="vividvr",
            pipeline_class_name="CogVideoXVividVRControlNetPipeline",
        ),
    )
    monkeypatch.setattr(
        video_api.VividVRSamplingParams,
        "from_user_kwargs",
        staticmethod(
            lambda server_args, **kwargs: SimpleNamespace(
                output_file_path=lambda: str(tmp_path / "outputs" / "task-log.mp4")
            )
        ),
    )
    monkeypatch.setattr(
        video_api,
        "prepare_request",
        lambda server_args, sampling_params: "prepared-batch",
    )

    def fake_create_task(coro):
        scheduled["coro_name"] = coro.cr_code.co_name
        coro.close()
        return None

    monkeypatch.setattr(video_api.asyncio, "create_task", fake_create_task)
    client = _make_test_client()

    with caplog.at_level(logging.INFO, logger=video_api.logger.name):
        response = client.post(
            "/v1/videos/repairs/flowcut",
            json={
                "taskId": "task-log",
                "timeout": -1,
                "callbackUrl": "http://127.0.0.1:9000/tasks/task-log/callback",
                "video_input_path": "/tmp/in.mp4",
            },
        )

    assert response.json() == {"code": 0, "message": "ok"}
    assert "FlowCut video repair accepted task_id=task-log" in caplog.text


def test_retrieve_video_logs_missing_task(caplog):
    client = _make_test_client()

    with caplog.at_level(logging.INFO, logger=video_api.logger.name):
        response = client.get("/v1/videos/missing-task/progress")

    assert response.status_code == 404
    assert "Video task not found task_id=missing-task" in caplog.text


def test_dispatch_flowcut_job_posts_running_and_final_callbacks(monkeypatch, tmp_path):
    callbacks = []

    async def fake_post_flowcut_callback(callback_url, payload, **kwargs):
        callbacks.append(payload)

    async def fake_dispatch_job_async(job_id, batch, **kwargs):
        await video_api.VIDEO_STORE.update_fields(
            job_id,
            {
                "status": "completed",
                "progress": 100,
                "file_path": str(tmp_path / "out.mp4"),
                "url": None,
                "inference_time_s": 1.25,
            },
        )

    monkeypatch.setattr(video_api, "post_flowcut_callback", fake_post_flowcut_callback)
    monkeypatch.setattr(video_api, "_dispatch_job_async", fake_dispatch_job_async)
    monkeypatch.setattr(
        video_api,
        "report_flowcut_running_until_done",
        lambda task_id, callback_url, done_event, **kwargs: fake_post_flowcut_callback(
            callback_url,
            {
                "status": "running",
                "progress": 1,
                "reason": "accepted",
                "output": "",
            },
        ),
    )

    class ReleaseTrackingSemaphore:
        def __init__(self):
            self.released = False

        def release(self):
            self.released = True

    semaphore = ReleaseTrackingSemaphore()
    monkeypatch.setattr(video_api, "_VIDEOEDIT_SEMAPHORE", semaphore)

    async def run_test():
        await video_api.VIDEO_STORE.upsert(
            "task-1",
            {
                "id": "task-1",
                "object": "video",
                "model": "VividVR",
                "status": "queued",
                "progress": 0,
                "created_at": 1,
                "size": "",
                "seconds": "",
                "quality": "standard",
                "file_path": str(tmp_path / "out.mp4"),
            },
        )

        await video_api._dispatch_flowcut_video_repair_job_async(
            "task-1",
            batch="prepared",
            callback_url="http://127.0.0.1:9000/callback",
        )

    asyncio.run(run_test())

    assert callbacks[0]["status"] == "running"
    assert callbacks[-1]["status"] == "succeeded"
    assert callbacks[-1]["progress"] == 100.0
    assert json.loads(callbacks[-1]["output"]) == {
        "result_url": str(tmp_path / "out.mp4"),
        "duration": 1.25,
    }
    assert semaphore.released is True


def test_dispatch_flowcut_job_posts_failed_callback(monkeypatch, tmp_path):
    callbacks = []

    async def fake_post_flowcut_callback(callback_url, payload, **kwargs):
        callbacks.append(payload)

    async def fake_dispatch_job_async(job_id, batch, **kwargs):
        await video_api.VIDEO_STORE.update_fields(
            job_id,
            {
                "status": "failed",
                "error": {"message": "GPU out of memory during inference"},
            },
        )

    monkeypatch.setattr(video_api, "post_flowcut_callback", fake_post_flowcut_callback)
    monkeypatch.setattr(video_api, "_dispatch_job_async", fake_dispatch_job_async)
    monkeypatch.setattr(
        video_api,
        "report_flowcut_running_until_done",
        lambda task_id, callback_url, done_event, **kwargs: fake_post_flowcut_callback(
            callback_url,
            {
                "status": "running",
                "progress": 1,
                "reason": "accepted",
                "output": "",
            },
        ),
    )

    class ReleaseTrackingSemaphore:
        def release(self):
            pass

    monkeypatch.setattr(video_api, "_VIDEOEDIT_SEMAPHORE", ReleaseTrackingSemaphore())

    async def run_test():
        await video_api.VIDEO_STORE.upsert(
            "task-fail",
            {
                "id": "task-fail",
                "object": "video",
                "model": "VividVR",
                "status": "queued",
                "progress": 0,
                "created_at": 1,
                "size": "",
                "seconds": "",
                "quality": "standard",
                "file_path": str(tmp_path / "out.mp4"),
            },
        )

        await video_api._dispatch_flowcut_video_repair_job_async(
            "task-fail",
            batch="prepared",
            callback_url="http://127.0.0.1:9000/callback",
        )

    asyncio.run(run_test())

    assert callbacks[-1] == {
        "status": "failed",
        "progress": 0.0,
        "reason": "GPU out of memory during inference",
        "output": "",
    }
