from __future__ import annotations

import pytest

from sglang.multimodal_gen.tools.run_flowcut_vividvr_service_acceptance import (
    FlowCutAcceptanceError,
    poll_accepted_task,
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
        retry_interval_seconds=0.01,
        max_submit_attempts=2,
    )

    assert result == {"code": 0, "message": "ok"}
    assert len(client.posts) == 2
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
