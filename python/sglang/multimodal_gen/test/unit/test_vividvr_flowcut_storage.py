import asyncio
from pathlib import Path

import pytest

from sglang.multimodal_gen.runtime.entrypoints.openai import vividvr_flowcut_storage
from sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_protocol import (
    VividVRFlowCutMinIOConfig,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_storage import (
    VividVRFlowCutStorage,
)


def test_materialize_video_copies_local_file_into_request_inputs(tmp_path):
    source = tmp_path / "source.mov"
    source.write_bytes(b"video-bytes")

    storage = VividVRFlowCutStorage(base_dir=tmp_path / "work", request_id="task-1")
    materialized = asyncio.run(
        storage.materialize_video(str(source), filename_hint="camera")
    )

    assert materialized == str(tmp_path / "work" / "task-1" / "inputs" / "camera.mp4")
    assert Path(materialized).read_bytes() == b"video-bytes"
    assert storage.inputs_dir == tmp_path / "work" / "task-1" / "inputs"
    assert storage.outputs_dir == tmp_path / "work" / "task-1" / "outputs"
    assert storage.manifests_dir == tmp_path / "work" / "task-1" / "manifests"


def test_materialize_video_returns_same_path_when_source_matches_target(tmp_path):
    storage = VividVRFlowCutStorage(base_dir=tmp_path, request_id="task-2")
    target = storage.inputs_dir / "input.mp4"
    target.write_bytes(b"already-here")

    materialized = asyncio.run(storage.materialize_video(str(target)))

    assert materialized == str(target)
    assert target.read_bytes() == b"already-here"


def test_materialize_video_rejects_missing_local_file(tmp_path):
    storage = VividVRFlowCutStorage(base_dir=tmp_path, request_id="task-3")

    with pytest.raises(FileNotFoundError):
        asyncio.run(storage.materialize_video(str(tmp_path / "missing.mp4")))


def test_materialize_video_downloads_http_source(tmp_path, monkeypatch):
    calls = []

    class FakeResponse:
        content = b"http-video"

        def raise_for_status(self):
            return None

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url):
            calls.append({"url": url})
            return FakeResponse()

    monkeypatch.setattr(vividvr_flowcut_storage.httpx, "AsyncClient", FakeAsyncClient)

    storage = VividVRFlowCutStorage(base_dir=tmp_path, request_id="task-4")
    materialized = asyncio.run(
        storage.materialize_video("https://example.test/video", filename_hint="clip.webm")
    )

    assert materialized == str(tmp_path / "task-4" / "inputs" / "clip.webm")
    assert Path(materialized).read_bytes() == b"http-video"
    assert calls[0]["follow_redirects"] is True
    assert calls[0]["timeout"] > 0
    assert calls[1] == {"url": "https://example.test/video"}


def test_output_file_path_is_request_scoped(tmp_path):
    storage = VividVRFlowCutStorage(base_dir=tmp_path, request_id="task-5")

    assert storage.output_file_path() == str(tmp_path / "task-5" / "outputs" / "task-5.mp4")
    assert storage.output_file_path("custom.mov") == str(
        tmp_path / "task-5" / "outputs" / "custom.mov"
    )


def test_upload_result_without_minio_returns_local_path_and_keeps_file(tmp_path):
    storage = VividVRFlowCutStorage(base_dir=tmp_path, request_id="task-6")
    output_path = Path(storage.output_file_path())
    output_path.write_bytes(b"result")

    result = asyncio.run(storage.upload_result(str(output_path), minio_config=None))

    assert result == str(output_path)
    assert output_path.exists()


def test_upload_result_with_minio_deletes_local_file_after_success(tmp_path, monkeypatch):
    uploaded = {}

    async def fake_upload_to_flowcut_minio(*, local_path, object_key, config):
        uploaded["local_path"] = local_path
        uploaded["object_key"] = object_key
        uploaded["config"] = config
        return "http://minio/flowcut/outputs/task-7.mp4"

    monkeypatch.setattr(
        vividvr_flowcut_storage,
        "upload_to_flowcut_minio",
        fake_upload_to_flowcut_minio,
    )
    config = VividVRFlowCutMinIOConfig(
        endpoint="minio:9000",
        bucket_name="flowcut",
        access_key="ak",
        secret_key="sk",
    )
    storage = VividVRFlowCutStorage(base_dir=tmp_path, request_id="task-7")
    output_path = Path(storage.output_file_path())
    output_path.write_bytes(b"result")

    result = asyncio.run(storage.upload_result(str(output_path), config))

    assert result == "http://minio/flowcut/outputs/task-7.mp4"
    assert uploaded == {
        "local_path": str(output_path),
        "object_key": "outputs/task-7.mp4",
        "config": config,
    }
    assert not output_path.exists()


def test_upload_result_with_minio_keeps_local_file_after_failure(tmp_path, monkeypatch):
    async def fake_upload_to_flowcut_minio(*, local_path, object_key, config):
        raise RuntimeError("upload failed")

    monkeypatch.setattr(
        vividvr_flowcut_storage,
        "upload_to_flowcut_minio",
        fake_upload_to_flowcut_minio,
    )
    config = VividVRFlowCutMinIOConfig(
        endpoint="minio:9000",
        bucket_name="flowcut",
        access_key="ak",
        secret_key="sk",
    )
    storage = VividVRFlowCutStorage(base_dir=tmp_path, request_id="task-8")
    output_path = Path(storage.output_file_path())
    output_path.write_bytes(b"result")

    with pytest.raises(RuntimeError, match="upload failed"):
        asyncio.run(storage.upload_result(str(output_path), config))

    assert output_path.exists()


def test_cleanup_deletes_request_workdir(tmp_path):
    storage = VividVRFlowCutStorage(base_dir=tmp_path, request_id="task-9")
    (storage.manifests_dir / "caption.txt").write_text("caption")

    storage.cleanup()

    assert not storage.workdir.exists()
