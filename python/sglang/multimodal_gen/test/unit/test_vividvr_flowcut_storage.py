import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

from sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_protocol import (
    FlowCutMinIOConfig,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_storage import (
    VividVRFlowCutStorage,
)


def test_materialize_video_preserves_local_source_extension(tmp_path):
    storage = VividVRFlowCutStorage(base_dir=tmp_path, request_id="task-ext")
    source = tmp_path / "input.mov"
    source.write_bytes(b"video")

    result = asyncio.run(storage.materialize_video(source))

    result_path = Path(result)
    assert result_path.name == "input.mov"
    assert result_path.read_bytes() == b"video"


def test_upload_result_uses_explicit_output_object_key_and_bucket(tmp_path):
    storage = VividVRFlowCutStorage(base_dir=tmp_path, request_id="task-1")
    local_path = Path(storage.output_file_path("task-1.mp4"))
    local_path.write_bytes(b"video")

    with patch(
        "sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_storage.upload_to_flowcut_minio",
        new=AsyncMock(return_value="http://minio/bucket-b/custom/key.mp4"),
    ) as mock_upload:
        result = asyncio.run(
            storage.upload_result(
                local_path,
                FlowCutMinIOConfig(
                    endpoint="minio.example.com:9000",
                    bucket_name="bucket-a",
                    access_key="ak",
                    secret_key="sk",
                ),
                object_key="custom/key.mp4",
                bucket_name="bucket-b",
            )
        )

    assert result == "http://minio/bucket-b/custom/key.mp4"
    mock_upload.assert_awaited_once_with(
        local_path=str(local_path.resolve()),
        object_key="custom/key.mp4",
        config=FlowCutMinIOConfig(
            endpoint="minio.example.com:9000",
            bucket_name="bucket-a",
            access_key="ak",
            secret_key="sk",
        ),
        bucket_name="bucket-b",
    )
