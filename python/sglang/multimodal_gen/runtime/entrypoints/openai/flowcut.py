import asyncio
import json
import os
import time
from typing import Any, Dict, Optional

import httpx

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    FlowCutMinIOConfig,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

FLOWCUT_PROGRESS_INTERVAL_SECONDS = float(
    os.environ.get("SGLANG_FLOWCUT_PROGRESS_INTERVAL_SECONDS", "30")
)


def build_flowcut_running_callback_payload(
    *, task_id: str, progress: float, reason: str
) -> Dict[str, Any]:
    return {
        "status": "running",
        "progress": float(progress),
        "reason": reason,
        "output": "",
    }


def build_flowcut_final_callback_payload(
    *,
    status: str,
    progress: float,
    reason: str,
    output: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if status not in {"succeeded", "failed"}:
        raise ValueError(f"Unsupported FlowCut final status: {status}")
    return {
        "status": status,
        "progress": float(progress),
        "reason": reason,
        "output": json.dumps(output, ensure_ascii=False) if output else "",
    }


async def post_flowcut_callback(
    callback_url: str,
    payload: Dict[str, Any],
    *,
    timeout: float = 10.0,
    max_retries: int = 3,
) -> None:
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=timeout,
                trust_env=False,
            ) as client:
                response = await client.post(callback_url, json=payload)
                response.raise_for_status()
            return
        except Exception as e:
            last_error = str(e)
            logger.warning(
                "FlowCut callback failed attempt=%s/%s url=%s: %s",
                attempt,
                max_retries,
                callback_url,
                last_error,
            )
            if attempt < max_retries:
                await asyncio.sleep(min(2 ** (attempt - 1), 5))
    raise RuntimeError(
        f"FlowCut callback failed after {max_retries} attempts: {last_error}"
    )


def progress_from_elapsed(started_at: float) -> float:
    elapsed = max(0.0, time.monotonic() - started_at)
    if elapsed < 1:
        return 1.0
    return min(89.0, 5.0 + elapsed / 30.0)


async def report_flowcut_running_until_done(
    *,
    task_id: str,
    callback_url: str,
    done_event: asyncio.Event,
    interval_seconds: float = FLOWCUT_PROGRESS_INTERVAL_SECONDS,
    send_initial: bool = True,
) -> None:
    started_at = time.monotonic()
    if send_initial:
        await post_flowcut_callback(
            callback_url,
            build_flowcut_running_callback_payload(
                task_id=task_id,
                progress=1,
                reason="accepted",
            ),
        )
    while not done_event.is_set():
        try:
            await asyncio.wait_for(done_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            await post_flowcut_callback(
                callback_url,
                build_flowcut_running_callback_payload(
                    task_id=task_id,
                    progress=progress_from_elapsed(started_at),
                    reason="running",
                ),
            )


def build_minio_result_url(config: FlowCutMinIOConfig, object_key: str) -> str:
    scheme = "https" if config.secure else "http"
    endpoint = config.endpoint.rstrip("/")
    return f"{scheme}://{endpoint}/{config.bucket_name}/{object_key.lstrip('/')}"


async def upload_to_flowcut_minio(
    *,
    local_path: str,
    object_key: str,
    config: FlowCutMinIOConfig,
) -> str:
    import boto3

    endpoint_url = f"{'https' if config.secure else 'http'}://{config.endpoint.rstrip('/')}"

    def _sync_upload() -> None:
        client = boto3.client(
            "s3",
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
            endpoint_url=endpoint_url,
            region_name=config.region,
        )
        client.upload_file(local_path, config.bucket_name, object_key)

    await asyncio.get_running_loop().run_in_executor(None, _sync_upload)
    return build_minio_result_url(config, object_key)
