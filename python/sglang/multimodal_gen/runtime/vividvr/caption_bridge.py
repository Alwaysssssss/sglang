# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass(frozen=True)
class VividVRCaptionBridgeConfig:
    enabled: bool = False
    base_url: str | None = None
    timeout_s: float = 1800.0


@dataclass(frozen=True)
class VividVRCaptionBridgeResult:
    caption_file_path: str
    caption_count: int


def validate_caption_sidecar_file(path: str | Path, *, expected_count: int) -> list[str]:
    caption_path = Path(path).expanduser()
    if expected_count <= 0:
        raise ValueError(f"expected_count must be positive, got {expected_count}")
    if not caption_path.exists():
        raise FileNotFoundError(f"caption sidecar file does not exist: {caption_path}")
    captions = [
        line.strip()
        for line in caption_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(captions) != expected_count:
        raise ValueError(
            f"caption sidecar expected {expected_count} captions, "
            f"got {len(captions)}: {caption_path}"
        )
    return captions


async def request_vividvr_caption_sidecar(
    *,
    config: VividVRCaptionBridgeConfig,
    manifest_path: str,
    output_caption_path: str,
    expected_caption_count: int,
) -> VividVRCaptionBridgeResult:
    if not config.enabled:
        raise RuntimeError("VividVR caption bridge is disabled")
    if not config.base_url:
        raise RuntimeError("VividVR caption sidecar URL is not configured")

    url = f"{config.base_url.rstrip('/')}/v1/vividvr/captions"
    payload = {
        "manifest_path": manifest_path,
        "output_caption_path": output_caption_path,
        "expected_caption_count": expected_caption_count,
    }
    try:
        async with httpx.AsyncClient(
            timeout=config.timeout_s,
            trust_env=False,
        ) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        raise RuntimeError(
            "VividVR caption sidecar request failed "
            f"url={url} manifest_path={manifest_path} "
            f"output_caption_path={output_caption_path}: {exc}"
        ) from exc

    caption_file_path = str(data.get("caption_file_path") or output_caption_path)
    validate_caption_sidecar_file(
        caption_file_path,
        expected_count=expected_caption_count,
    )
    return VividVRCaptionBridgeResult(
        caption_file_path=caption_file_path,
        caption_count=int(data.get("caption_count") or expected_caption_count),
    )
