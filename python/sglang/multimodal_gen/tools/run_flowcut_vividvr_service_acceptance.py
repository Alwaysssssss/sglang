from __future__ import annotations

import argparse
import json
import time
from typing import Any

import httpx


class FlowCutAcceptanceError(RuntimeError):
    pass


def _url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def submit_flowcut_task_with_retry(
    *,
    client: Any,
    base_url: str,
    payload: dict[str, Any],
    retry_interval_seconds: float = 30.0,
    max_submit_attempts: int = 60,
) -> dict[str, Any]:
    endpoint = _url(base_url, "/v1/videos/repairs/flowcut")
    for attempt in range(1, max_submit_attempts + 1):
        response = client.post(endpoint, json=payload, timeout=60.0)
        response.raise_for_status()
        body = response.json()
        code = body.get("code")
        message = body.get("message", "")
        if code == 0:
            return body
        if code == 2:
            if attempt >= max_submit_attempts:
                raise FlowCutAcceptanceError(
                    f"FlowCut service stayed busy after {attempt} submit attempts: {message}"
                )
            time.sleep(retry_interval_seconds)
            continue
        raise FlowCutAcceptanceError(
            f"FlowCut service rejected task with code={code}: {message}"
        )

    raise FlowCutAcceptanceError("FlowCut submit retry loop ended unexpectedly")


def poll_accepted_task(
    *,
    client: Any,
    base_url: str,
    task_id: str,
    poll_interval_seconds: float = 30.0,
    max_polls: int | None = None,
) -> dict[str, Any]:
    endpoint = _url(base_url, f"/v1/videos/{task_id}/progress")
    polls = 0
    while max_polls is None or polls < max_polls:
        polls += 1
        response = client.get(endpoint, timeout=60.0)
        if response.status_code == 404:
            raise FlowCutAcceptanceError(
                f"Task {task_id!r} is not in the server store; service may have restarted "
                "or the task was never accepted by this process."
            )
        response.raise_for_status()
        body = response.json()
        status = body.get("status")
        if status in {"completed", "failed", "cancelled"}:
            return body
        print(json.dumps(body, ensure_ascii=False), flush=True)
        if max_polls is None or polls < max_polls:
            time.sleep(poll_interval_seconds)

    raise FlowCutAcceptanceError(
        f"Task {task_id!r} did not finish after {max_polls} progress polls"
    )


def _build_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "taskId": args.task_id,
        "timeout": -1,
        "callbackUrl": args.callback_url,
        "video_input_path": args.input_video,
        "caption_file_path": args.caption_file,
        "reference_video_path": args.reference_video,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "num_temporal_process_frames": args.num_temporal_process_frames,
    }
    if args.output_path:
        payload["output_path"] = args.output_path
    if args.perf_dump_path:
        payload["perf_dump_path"] = args.perf_dump_path
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit and poll a FlowCut-compatible Vivid-VR service task."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:31191")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--callback-url", required=True)
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--caption-file", required=True)
    parser.add_argument("--reference-video", required=True)
    parser.add_argument("--output-path")
    parser.add_argument("--perf-dump-path")
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-temporal-process-frames", type=int, default=121)
    parser.add_argument("--submit-retry-interval-seconds", type=float, default=30.0)
    parser.add_argument("--max-submit-attempts", type=int, default=60)
    parser.add_argument("--poll-interval-seconds", type=float, default=30.0)
    parser.add_argument("--max-polls", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = _build_payload(args)
    with httpx.Client(follow_redirects=True, trust_env=False) as client:
        submit_result = submit_flowcut_task_with_retry(
            client=client,
            base_url=args.base_url,
            payload=payload,
            retry_interval_seconds=args.submit_retry_interval_seconds,
            max_submit_attempts=args.max_submit_attempts,
        )
        print(json.dumps(submit_result, ensure_ascii=False), flush=True)
        final_progress = poll_accepted_task(
            client=client,
            base_url=args.base_url,
            task_id=args.task_id,
            poll_interval_seconds=args.poll_interval_seconds,
            max_polls=args.max_polls,
        )
        print(json.dumps(final_progress, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
