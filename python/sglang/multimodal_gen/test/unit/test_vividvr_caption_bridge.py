import asyncio

import pytest

from sglang.multimodal_gen.runtime.vividvr.caption_bridge import (
    VividVRCaptionBridgeConfig,
    request_vividvr_caption_sidecar,
    validate_caption_sidecar_file,
)


def test_validate_caption_sidecar_file_requires_exact_line_count(tmp_path):
    caption_file = tmp_path / "caption.txt"
    caption_file.write_text("clip a\nclip b\n", encoding="utf-8")

    validate_caption_sidecar_file(caption_file, expected_count=2)

    with pytest.raises(ValueError, match="expected 3 captions"):
        validate_caption_sidecar_file(caption_file, expected_count=3)


def test_request_caption_sidecar_posts_manifest(monkeypatch, tmp_path):
    calls = {}
    output = tmp_path / "caption.txt"
    output.write_text("caption 0\n", encoding="utf-8")

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"caption_file_path": str(output), "caption_count": 1}

    class FakeClient:
        def __init__(self, **kwargs):
            calls["client_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json):
            calls["url"] = url
            calls["json"] = json
            return FakeResponse()

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.vividvr.caption_bridge.httpx.AsyncClient",
        FakeClient,
    )

    config = VividVRCaptionBridgeConfig(
        enabled=True,
        base_url="http://127.0.0.1:31200",
        timeout_s=30.0,
    )

    result = asyncio.run(
        request_vividvr_caption_sidecar(
            config=config,
            manifest_path=str(tmp_path / "manifest.json"),
            output_caption_path=str(output),
            expected_caption_count=1,
        )
    )

    assert result.caption_file_path == str(output)
    assert calls["url"] == "http://127.0.0.1:31200/v1/vividvr/captions"
    assert calls["json"] == {
        "manifest_path": str(tmp_path / "manifest.json"),
        "output_caption_path": str(output),
        "expected_caption_count": 1,
    }
    assert calls["client_kwargs"]["trust_env"] is False
    assert calls["client_kwargs"]["timeout"] == 30.0
