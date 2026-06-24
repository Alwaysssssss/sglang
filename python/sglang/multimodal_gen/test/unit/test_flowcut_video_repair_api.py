import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.multimodal_gen.runtime.entrypoints.openai import video_api
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


def test_flowcut_endpoint_returns_code_2_when_queue_full(monkeypatch, tmp_path):
    class LockedSemaphore:
        def locked(self):
            return True

        async def acquire(self):
            raise AssertionError("busy request must not acquire semaphore")

    monkeypatch.setattr(vividvr_flowcut_api, "_VIDEOEDIT_SEMAPHORE", LockedSemaphore())
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

    monkeypatch.setattr(vividvr_flowcut_api, "_VIDEOEDIT_SEMAPHORE", AvailableSemaphore())
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

    monkeypatch.setattr(vividvr_flowcut_api, "_VIDEOEDIT_SEMAPHORE", AvailableSemaphore())
    client = _make_test_client()

    response = client.post(
        "/v1/videos/repairs/flowcut",
        content="{bad json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    assert response.json()["code"] == 1
    assert "invalid request" in response.json()["message"]


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

    monkeypatch.setattr(vividvr_flowcut_api, "_VIDEOEDIT_SEMAPHORE", AvailableSemaphore())
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
    input_video = tmp_path / "in.mp4"
    input_video.write_bytes(b"video")

    class AvailableSemaphore:
        def locked(self):
            return False

        async def acquire(self):
            acquired["value"] = True

        def release(self):
            acquired["value"] = False

    monkeypatch.setattr(vividvr_flowcut_api, "_VIDEOEDIT_SEMAPHORE", AvailableSemaphore())
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
            "reference_video_path": "/tmp/ref.mp4",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"code": 0, "message": "ok"}
    assert acquired["value"] is True
    assert scheduled["coro_name"] == "_dispatch_vividvr_flowcut_video_repair_job_async"
    assert captured_kwargs["video_input_path"].endswith(
        "flowcut_work/task-1/inputs/input.mp4"
    )
    assert captured_kwargs["output_path"].endswith("flowcut_work/task-1/outputs")


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

    monkeypatch.setattr(vividvr_flowcut_api, "_VIDEOEDIT_SEMAPHORE", AvailableSemaphore())
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
        vividvr_flowcut_api,
        "_ensure_vividvr_caption_file",
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

    monkeypatch.setattr(vividvr_flowcut_api, "_VIDEOEDIT_SEMAPHORE", AvailableSemaphore())
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
        vividvr_flowcut_api,
        "_VIDEOEDIT_SEMAPHORE",
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
        "running",
        "succeeded",
    ]
    assert [callback["reason"] for callback in callbacks] == [
        "editing",
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
    assert vividvr_flowcut_api._VIDEOEDIT_SEMAPHORE.released is True


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

    async def fake_upload_result(local_path, minio_config):
        Path(local_path).unlink()
        return "http://minio/flowcut/outputs/task-minio.mp4"

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
        vividvr_flowcut_api,
        "_VIDEOEDIT_SEMAPHORE",
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
            timeout=-1,
        )

    asyncio.run(run_test())

    assert callbacks[-1]["status"] == "succeeded"
    assert json.loads(callbacks[-1]["output"]) == {
        "result_url": "http://minio/flowcut/outputs/task-minio.mp4"
    }
    assert not output_path.exists()


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

    async def fake_upload_result(local_path, minio_config):
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
        vividvr_flowcut_api,
        "_VIDEOEDIT_SEMAPHORE",
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
        "progress": 90.0,
        "reason": "upload failed",
        "output": "",
    }
    assert output_path.exists()
