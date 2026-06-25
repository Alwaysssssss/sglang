import asyncio
import json

from sglang.multimodal_gen.runtime.entrypoints.openai import flowcut
from sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_progress import (
    FLOWCUT_STAGE_PROGRESS,
    VividVRFlowCutProgressReporter,
    flowcut_denoise_progress,
)


def test_stage_callbacks_use_stable_running_payloads():
    calls = []

    async def fake_post_callback(callback_url, payload, **kwargs):
        calls.append((callback_url, payload))

    reporter = VividVRFlowCutProgressReporter(
        task_id="task-1",
        callback_url="http://callback/tasks/task-1",
        post_callback=fake_post_callback,
    )

    for stage in (
        "accepted",
        "input_ready",
        "caption_ready",
        "uploading_result",
    ):
        asyncio.run(reporter.send_stage(stage))

    progresses = [payload["progress"] for _, payload in calls]
    assert progresses == [1.0, 3.0, 5.0, 98.0]
    assert progresses == sorted(progresses)
    assert [payload["reason"] for _, payload in calls] == [
        "accepted",
        "input_ready",
        "caption_ready",
        "uploading_result",
    ]
    assert all(payload["status"] == "running" for _, payload in calls)
    assert all(payload["output"] == "" for _, payload in calls)
    assert all(url == "http://callback/tasks/task-1" for url, _ in calls)


def test_stage_progress_mapping_is_monotonic_and_fixed():
    assert FLOWCUT_STAGE_PROGRESS == {
        "accepted": 1.0,
        "input_ready": 3.0,
        "caption_ready": 5.0,
        "uploading_result": 98.0,
        "succeeded": 100.0,
    }

    values = list(FLOWCUT_STAGE_PROGRESS.values())
    assert values == sorted(values)


def test_succeeded_callback_allows_only_result_url_and_duration():
    calls = []

    async def fake_post_callback(callback_url, payload, **kwargs):
        calls.append(payload)

    reporter = VividVRFlowCutProgressReporter(
        task_id="task-1",
        callback_url="http://callback/tasks/task-1",
        post_callback=fake_post_callback,
    )

    asyncio.run(reporter.send_succeeded("http://storage/out.mp4", duration=12.5))

    assert calls[-1]["status"] == "succeeded"
    assert calls[-1]["progress"] == 100.0
    assert json.loads(calls[-1]["output"]) == {
        "result_url": "http://storage/out.mp4",
        "duration": 12.5,
    }
    assert set(json.loads(calls[-1]["output"])) == {"result_url", "duration"}


def test_failed_callback_defaults_to_last_stage_progress_and_empty_output():
    calls = []

    async def fake_post_callback(callback_url, payload, **kwargs):
        calls.append(payload)

    reporter = VividVRFlowCutProgressReporter(
        task_id="task-1",
        callback_url="http://callback/tasks/task-1",
        post_callback=fake_post_callback,
    )

    asyncio.run(reporter.send_stage("caption_ready"))
    asyncio.run(reporter.send_failed("caption model failed"))

    assert calls[-1] == {
        "status": "failed",
        "progress": 5.0,
        "reason": "caption model failed",
        "output": "",
    }


def test_denoise_progress_maps_to_5_to_95_percent():
    assert flowcut_denoise_progress(0.0) == 5.0
    assert flowcut_denoise_progress(0.5) == 50.0
    assert flowcut_denoise_progress(1.0) == 95.0
    assert flowcut_denoise_progress(-1.0) == 5.0
    assert flowcut_denoise_progress(2.0) == 95.0


def test_denoise_progress_callbacks_only_send_increasing_values():
    calls = []

    async def fake_post_callback(callback_url, payload, **kwargs):
        calls.append(payload)

    reporter = VividVRFlowCutProgressReporter(
        task_id="task-1",
        callback_url="http://callback/tasks/task-1",
        post_callback=fake_post_callback,
    )

    assert asyncio.run(reporter.send_denoise_progress(0.10)) is True
    assert asyncio.run(reporter.send_denoise_progress(0.10)) is False
    assert asyncio.run(reporter.send_denoise_progress(0.09)) is False
    assert asyncio.run(reporter.send_denoise_progress(0.12)) is True
    assert asyncio.run(reporter.send_denoise_progress(0.16)) is True

    assert [payload["progress"] for payload in calls] == [14.0, 15.8, 19.4]
    assert [payload["reason"] for payload in calls] == [
        "denoising",
        "denoising",
        "denoising",
    ]


def test_failed_callback_uses_explicit_progress_without_prior_stage():
    calls = []

    async def fake_post_callback(callback_url, payload, **kwargs):
        calls.append(payload)

    reporter = VividVRFlowCutProgressReporter(
        task_id="task-1",
        callback_url="http://callback/tasks/task-1",
        post_callback=fake_post_callback,
    )

    asyncio.run(reporter.send_failed("invalid input", progress=7))

    assert calls[-1]["progress"] == 7.0
    assert calls[-1]["output"] == ""


def test_progress_reporter_does_not_use_elapsed_time_helper(monkeypatch):
    calls = []

    def fail_if_called(*args, **kwargs):
        raise AssertionError("stage progress must not use elapsed-time helper")

    async def fake_post_callback(callback_url, payload, **kwargs):
        calls.append(payload)

    monkeypatch.setattr(flowcut, "progress_from_elapsed", fail_if_called)
    reporter = VividVRFlowCutProgressReporter(
        task_id="task-1",
        callback_url="http://callback/tasks/task-1",
        post_callback=fake_post_callback,
    )

    asyncio.run(reporter.send_stage("caption_ready"))
    asyncio.run(reporter.send_denoise_progress(0.5))
    asyncio.run(reporter.send_succeeded("http://storage/out.mp4"))

    assert [payload["progress"] for payload in calls] == [5.0, 50.0, 100.0]
