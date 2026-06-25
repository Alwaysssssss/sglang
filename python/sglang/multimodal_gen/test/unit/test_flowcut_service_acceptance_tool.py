from __future__ import annotations

import httpx
import pytest

from sglang.multimodal_gen.tools.run_flowcut_vividvr_service_acceptance import (
    FlowCutAcceptanceError,
    _FlowCutCallbackRecorder,
    _LocalFlowCutCallbackServer,
    _build_payload,
    _validate_final_callback_payload,
    poll_accepted_task,
    parse_args,
    submit_flowcut_task_with_retry,
)


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = str(self._payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}: {self.text}")


class _FakeClient:
    def __init__(self, *, post_responses=None, get_responses=None):
        self.post_responses = list(post_responses or [])
        self.get_responses = list(get_responses or [])
        self.posts = []
        self.gets = []

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return self.post_responses.pop(0)

    def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        return self.get_responses.pop(0)


def test_submit_flowcut_task_retries_code_2_before_polling(monkeypatch):
    sleeps = []
    client = _FakeClient(
        post_responses=[
            _FakeResponse(payload={"code": 2, "message": "A task is running."}),
            _FakeResponse(payload={"code": 0, "message": "ok"}),
        ]
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.tools.run_flowcut_vividvr_service_acceptance.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )

    result = submit_flowcut_task_with_retry(
        client=client,
        base_url="http://127.0.0.1:31191",
        payload={"taskId": "task-1"},
        submit_timeout_s=123.0,
        retry_interval_seconds=0.01,
        max_submit_attempts=2,
    )

    assert result == {"code": 0, "message": "ok"}
    assert len(client.posts) == 2
    assert [kwargs["timeout"] for _, kwargs in client.posts] == [123.0, 123.0]
    assert client.gets == []
    assert sleeps == [0.01]


def test_submit_flowcut_task_rejects_code_1_without_polling():
    client = _FakeClient(
        post_responses=[
            _FakeResponse(payload={"code": 1, "message": "bad request"}),
        ]
    )

    with pytest.raises(FlowCutAcceptanceError, match="bad request"):
        submit_flowcut_task_with_retry(
            client=client,
            base_url="http://127.0.0.1:31191",
            payload={"taskId": "task-1"},
            retry_interval_seconds=0,
            max_submit_attempts=1,
        )

    assert client.gets == []


def test_poll_accepted_task_reports_404_as_stale_or_restarted_service():
    client = _FakeClient(get_responses=[_FakeResponse(status_code=404)])

    with pytest.raises(FlowCutAcceptanceError, match="service may have restarted"):
        poll_accepted_task(
            client=client,
            base_url="http://127.0.0.1:31191",
            task_id="old-task",
            poll_interval_seconds=0,
            max_polls=1,
        )

    assert client.gets == [
        ("http://127.0.0.1:31191/v1/videos/old-task/progress", {"timeout": 60.0})
    ]


def test_build_payload_omits_optional_caption_and_reference_for_bridge_path(tmp_path):
    callback_log = tmp_path / "callback.jsonl"
    args = parse_args(
        [
            "--task-id",
            "task-bridge",
            "--callback-log",
            str(callback_log),
            "--video-input-path",
            "/tmp/input.mp4",
            "--output-path",
            "/tmp/output.mp4",
            "--poll-timeout-s",
            "2400",
        ]
    )

    payload = _build_payload(
        args,
        callback_url="http://127.0.0.1:39090/tasks/task-bridge/callback",
    )

    assert payload["video_input_path"] == "/tmp/input.mp4"
    assert payload["callbackUrl"].endswith("/tasks/task-bridge/callback")
    assert payload["upscale"] == 1.0
    assert "caption_file_path" not in payload
    assert "reference_video_path" not in payload


def test_parse_args_accepts_long_submit_timeout_for_caption_bridge(tmp_path):
    callback_log = tmp_path / "callback.jsonl"
    args = parse_args(
        [
            "--task-id",
            "task-bridge",
            "--callback-log",
            str(callback_log),
            "--video-input-path",
            "/tmp/input.mp4",
            "--submit-timeout-s",
            "2400",
        ]
    )

    assert args.submit_timeout_s == 2400.0


def test_parse_args_accepts_original_upscale_flag_for_service_request(tmp_path):
    callback_log = tmp_path / "callback.jsonl"
    args = parse_args(
        [
            "--task-id",
            "task-upscale",
            "--callback-log",
            str(callback_log),
            "--video-input-path",
            "/tmp/input.mp4",
            "--upscale",
            "2.0",
        ]
    )

    assert args.upscale == 2.0


def test_validate_final_callback_requires_result_url_only():
    _validate_final_callback_payload(
        {
            "status": "succeeded",
            "progress": 100,
            "reason": "",
            "output": '{"result_url":"http://storage.example.com/out.mp4","duration":12.5}',
        }
    )


@pytest.mark.parametrize(
    "output",
    [
        "",
        "{}",
        '{"gen_video_url":"http://storage.example.com/out.mp4"}',
        '{"result_url":"http://storage.example.com/out.mp4","file_path":"/tmp/out.mp4"}',
        '{"result_url":"http://storage.example.com/out.mp4","unexpected":"x"}',
    ],
)
def test_validate_final_callback_rejects_non_flowcut_output(output):
    with pytest.raises(FlowCutAcceptanceError):
        _validate_final_callback_payload(
            {
                "status": "succeeded",
                "progress": 100,
                "reason": "",
                "output": output,
            }
        )


def test_local_callback_server_records_final_payload(tmp_path):
    callback_log = tmp_path / "callback.jsonl"
    recorder = _FlowCutCallbackRecorder(str(callback_log))

    with _LocalFlowCutCallbackServer(
        host="127.0.0.1",
        port=0,
        task_id="task-1",
        recorder=recorder,
    ) as server:
        with httpx.Client(trust_env=False) as client:
            response = client.post(
                server.callback_url,
                json={
                    "status": "succeeded",
                    "progress": 100,
                    "reason": "done",
                    "output": '{"result_url":"http://storage.example.com/out.mp4"}',
                },
            )
            response.raise_for_status()

        final_payload = recorder.wait_for_final(timeout=1.0)

    assert final_payload["status"] == "succeeded"
    _validate_final_callback_payload(final_payload)
    assert callback_log.read_text(encoding="utf-8").strip()
