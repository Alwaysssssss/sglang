import asyncio
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.entrypoints.openai import video_api
from sglang.multimodal_gen.runtime.entrypoints.openai.video_job_runner import (
    run_video_generation_job,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch


def test_run_video_generation_job_returns_first_save_path(monkeypatch):
    batch = SimpleNamespace(prompt="make a video")
    result = OutputBatch(output_file_paths=["/tmp/first.mp4"])
    calls = {}

    async def fake_process_generation_batch(scheduler_client, received_batch):
        calls["scheduler_client"] = scheduler_client
        calls["batch"] = received_batch
        return ["/tmp/first.mp4", "/tmp/second.mp4"], result

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.openai.video_job_runner.process_generation_batch",
        fake_process_generation_batch,
    )

    job_result = asyncio.run(run_video_generation_job(batch))

    assert job_result.save_file_path == "/tmp/first.mp4"
    assert job_result.result is result
    assert calls["batch"] is batch


def test_run_video_generation_job_propagates_generation_error(monkeypatch):
    batch = SimpleNamespace(prompt="make a video")

    async def fake_process_generation_batch(scheduler_client, received_batch):
        raise RuntimeError("scheduler failed")

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.openai.video_job_runner.process_generation_batch",
        fake_process_generation_batch,
    )

    with pytest.raises(RuntimeError, match="scheduler failed"):
        asyncio.run(run_video_generation_job(batch))


def test_dispatch_job_async_uses_runner_success_path(monkeypatch):
    job_id = "job-1"
    batch = SimpleNamespace(prompt="make a video")
    result = OutputBatch()
    updates = []

    async def fake_run_video_generation_job(received_batch):
        assert received_batch is batch
        return SimpleNamespace(save_file_path="/tmp/generated.mp4", result=result)

    async def fake_upload_and_cleanup(path):
        assert path == "/tmp/generated.mp4"
        return "https://cdn.example/generated.mp4"

    async def fake_update_fields(received_job_id, fields):
        assert received_job_id == job_id
        updates.append(fields)

    async def fake_get(received_job_id):
        assert received_job_id == job_id
        return None

    monkeypatch.setattr(
        video_api, "run_video_generation_job", fake_run_video_generation_job
    )
    monkeypatch.setattr(
        video_api.cloud_storage, "upload_and_cleanup", fake_upload_and_cleanup
    )
    monkeypatch.setattr(video_api.VIDEO_STORE, "update_fields", fake_update_fields)
    monkeypatch.setattr(video_api.VIDEO_STORE, "get", fake_get)

    asyncio.run(video_api._dispatch_job_async(job_id, batch))

    assert updates == [
        {
            "status": "completed",
            "progress": 100,
            "completed_at": updates[0]["completed_at"],
            "url": "https://cdn.example/generated.mp4",
            "file_path": None,
            "id": job_id,
        }
    ]


def test_dispatch_job_async_records_failure_when_runner_fails(monkeypatch):
    job_id = "job-1"
    batch = SimpleNamespace(prompt="make a video")
    updates = []

    async def fake_run_video_generation_job(received_batch):
        assert received_batch is batch
        raise RuntimeError("generation failed")

    async def fake_update_fields(received_job_id, fields):
        assert received_job_id == job_id
        updates.append(fields)

    async def fake_get(received_job_id):
        assert received_job_id == job_id
        return None

    monkeypatch.setattr(
        video_api, "run_video_generation_job", fake_run_video_generation_job
    )
    monkeypatch.setattr(video_api.VIDEO_STORE, "update_fields", fake_update_fields)
    monkeypatch.setattr(video_api.VIDEO_STORE, "get", fake_get)

    asyncio.run(video_api._dispatch_job_async(job_id, batch))

    assert updates == [
        {"status": "failed", "error": {"message": "generation failed"}}
    ]
