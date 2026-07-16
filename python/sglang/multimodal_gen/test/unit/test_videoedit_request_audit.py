import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sglang.multimodal_gen.runtime.entrypoints.openai import video_api
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.videoedit.request_audit import (
    VideoEditRequestAudit,
    sanitize_videoedit_request_data,
)
from sglang.multimodal_gen.utils import FlexibleArgumentParser


def _request_payload():
    return {
        "taskId": "audit-task",
        "bbox_expand_scale": 0.3,
        "dilate_px": 8,
        "feather_px": 8,
        "minioConfig": {
            "provider": "cos",
            "prefix": "/flowcut",
            "endpoint": "cos.ap-beijing.myqcloud.com",
            "bucket": "vrs-mms-1258229344",
            "rootUser": "AKID1234567890",
            "rootPass": "secret-value",
            "region": "ap-beijing",
            "useSSL": True,
        },
    }


def test_sanitize_videoedit_request_data_redacts_storage_credentials():
    sanitized = sanitize_videoedit_request_data(_request_payload())
    minio_config = sanitized["minioConfig"]

    assert minio_config["rootUser"] == "AKID***7890"
    assert len(minio_config["rootUser_sha256"]) == 12
    assert minio_config["rootPass"] == "***"
    assert len(minio_config["rootPass_sha256"]) == 12
    assert sanitized["bbox_expand_scale"] == 0.3


def test_sanitize_videoedit_request_data_can_include_sensitive_values():
    sanitized = sanitize_videoedit_request_data(
        _request_payload(), include_sensitive_values=True
    )

    assert sanitized["minioConfig"]["rootUser"] == "AKID1234567890"
    assert sanitized["minioConfig"]["rootPass"] == "secret-value"


def test_request_audit_writes_unique_atomic_json_with_private_permissions(tmp_path):
    audit = VideoEditRequestAudit(
        str(tmp_path), task_id="../../audit-task", include_sensitive_values=False
    )
    first_path = audit.update(status="validated", raw_request=_request_payload())

    assert first_path is not None
    path = Path(first_path)
    assert path.parent == tmp_path
    assert path.name.endswith(".request.json")
    assert os.stat(path).st_mode & 0o777 == 0o600

    audit.update(
        status="queued",
        effective_request={"bbox_expand_scale": 0.3, "dilate_px": 8},
        resolved={"output_object_key": "flowcut/2026/07/15/result.mov"},
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["status"] == "queued"
    assert record["raw_request"]["minioConfig"]["rootPass"] == "***"
    assert record["effective_request"]["dilate_px"] == 8
    assert record["resolved"]["output_object_key"].startswith("flowcut/")

    another = VideoEditRequestAudit(str(tmp_path), task_id="../../audit-task")
    another_path = another.update(raw_request={})
    assert another_path != first_path


def test_request_audit_is_disabled_without_log_directory(tmp_path):
    audit = VideoEditRequestAudit(None, task_id="audit-task")

    assert not audit.enabled
    assert audit.update(raw_request=_request_payload()) is None
    assert list(tmp_path.iterdir()) == []


def test_server_args_expose_videoedit_request_audit_flags():
    parser = FlexibleArgumentParser()
    ServerArgs.add_cli_args(parser)
    args, unknown = parser.parse_known_args(
        [
            "--model-path",
            "test-model",
            "--videoedit-request-log-dir",
            "/tmp/videoedit-audit",
            "--videoedit-request-log-sensitive-values",
        ]
    )

    assert unknown == []
    assert args.videoedit_request_log_dir == "/tmp/videoedit-audit"
    assert args.videoedit_request_log_sensitive_values is True


def test_invalid_videoedit_request_is_recorded(tmp_path):
    class FakeRequest:
        method = "POST"
        url = SimpleNamespace(path="/v1/videos/repairs")
        client = SimpleNamespace(host="10.1.2.3")

        async def json(self):
            return {
                "bbox_expand_scale": 0.3,
                "dilate_px": 8,
                "minioConfig": _request_payload()["minioConfig"],
            }

    server_args = SimpleNamespace(
        videoedit_request_log_dir=str(tmp_path),
        videoedit_request_log_sensitive_values=False,
    )
    with (
        patch.object(video_api, "get_global_server_args", return_value=server_args),
        patch.object(
            video_api,
            "_store_failed_video_repair_submission",
            new=AsyncMock(),
        ),
    ):
        response = asyncio.run(video_api.create_video_repair(FakeRequest()))

    assert response["code"] == 1
    audit_files = list(tmp_path.glob("*.request.json"))
    assert len(audit_files) == 1
    record = json.loads(audit_files[0].read_text(encoding="utf-8"))
    assert record["status"] == "rejected_invalid"
    assert record["raw_request"]["bbox_expand_scale"] == 0.3
    assert record["raw_request"]["dilate_px"] == 8
    assert record["raw_request"]["minioConfig"]["rootPass"] == "***"
