import asyncio
import json

from sglang.multimodal_gen.runtime.entrypoints.openai import flowcut
from sglang.multimodal_gen.runtime.entrypoints.openai.flowcut import (
    build_flowcut_final_callback_payload,
    build_flowcut_running_callback_payload,
    build_minio_result_url,
    post_flowcut_callback,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    FlowCutMinIOConfig,
    FlowCutResponse,
    FlowCutVideoRepairRequest,
)


def test_flowcut_request_accepts_camel_case_system_fields():
    req = FlowCutVideoRepairRequest.model_validate(
        {
            "taskId": "task-1",
            "timeout": -1,
            "callbackUrl": "http://127.0.0.1:9000/tasks/task-1/callback",
            "minioConfig": {
                "endpoint": "minio.example.com:9000",
                "bucket_name": "flowcut",
                "access_key": "ak",
                "secret_key": "sk",
                "secure": False,
                "region": "us-east-1",
            },
            "video_url": "https://example.com/in.mp4",
            "caption_file_path": "/tmp/caption.txt",
            "num_inference_steps": 20,
        }
    )

    assert req.task_id == "task-1"
    assert req.timeout == -1
    assert req.callback_url == "http://127.0.0.1:9000/tasks/task-1/callback"
    assert req.minio_config is not None
    assert req.minio_config.bucket_name == "flowcut"
    assert req.video_url == "https://example.com/in.mp4"
    assert req.caption_file_path == "/tmp/caption.txt"
    assert req.num_inference_steps == 20


def test_flowcut_response_uses_numeric_code():
    accepted = FlowCutResponse(code=0, message="ok")
    busy = FlowCutResponse(code=2, message="A task is running.")

    assert accepted.model_dump() == {"code": 0, "message": "ok"}
    assert busy.model_dump()["code"] == 2
    assert isinstance(busy.model_dump()["code"], int)


def test_flowcut_running_callback_payload():
    payload = build_flowcut_running_callback_payload(
        task_id="task-1",
        progress=45.5,
        reason="processing",
    )

    assert payload == {
        "status": "running",
        "progress": 45.5,
        "reason": "processing",
        "output": "",
    }


def test_flowcut_final_callback_payload_success_output_is_json_string():
    payload = build_flowcut_final_callback_payload(
        status="succeeded",
        progress=100,
        reason="",
        output={"result_url": "http://storage/out.mp4", "duration": 12.5},
    )

    assert payload["status"] == "succeeded"
    assert payload["progress"] == 100.0
    assert payload["reason"] == ""
    assert json.loads(payload["output"]) == {
        "result_url": "http://storage/out.mp4",
        "duration": 12.5,
    }


def test_flowcut_final_callback_payload_failed_omits_output_data():
    payload = build_flowcut_final_callback_payload(
        status="failed",
        progress=0,
        reason="invalid video input",
        output=None,
    )

    assert payload == {
        "status": "failed",
        "progress": 0.0,
        "reason": "invalid video input",
        "output": "",
    }


def test_build_minio_result_url_http():
    cfg = FlowCutMinIOConfig(
        endpoint="minio.example.com:9000",
        bucket_name="flowcut",
        access_key="ak",
        secret_key="sk",
        secure=False,
        region="us-east-1",
    )

    assert (
        build_minio_result_url(cfg, "outputs/task-1.mp4")
        == "http://minio.example.com:9000/flowcut/outputs/task-1.mp4"
    )


def test_build_minio_result_url_https():
    cfg = FlowCutMinIOConfig(
        endpoint="minio.example.com",
        bucket_name="flowcut",
        access_key="ak",
        secret_key="sk",
        secure=True,
        region="us-east-1",
    )

    assert (
        build_minio_result_url(cfg, "outputs/task-1.mp4")
        == "https://minio.example.com/flowcut/outputs/task-1.mp4"
    )


def test_post_flowcut_callback_ignores_environment_proxy(monkeypatch):
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            return None

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, callback_url, json):
            return FakeResponse()

    monkeypatch.setattr(flowcut.httpx, "AsyncClient", FakeAsyncClient)

    asyncio.run(
        post_flowcut_callback(
            "http://127.0.0.1:39090/tasks/task-1/callback",
            {"status": "running", "progress": 1, "reason": "accepted", "output": ""},
            max_retries=1,
        )
    )

    assert calls
    assert calls[0]["trust_env"] is False
