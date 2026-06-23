from __future__ import annotations

import argparse
import json
import math
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from typing import Any

import httpx


class FlowCutAcceptanceError(RuntimeError):
    pass


class _FlowCutCallbackRecorder:
    def __init__(self, log_path: str | None):
        self.log_path = Path(log_path) if log_path else None
        self.payloads: list[dict[str, Any]] = []
        self.final_payload: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._final_event = threading.Event()

    def record(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self.payloads.append(payload)
            if self.log_path is not None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as fout:
                    fout.write(json.dumps(payload, ensure_ascii=False))
                    fout.write("\n")
            if payload.get("status") in {"succeeded", "failed"}:
                self.final_payload = payload
                self._final_event.set()

    def wait_for_final(self, timeout: float) -> dict[str, Any]:
        if not self._final_event.wait(timeout):
            raise FlowCutAcceptanceError(
                f"Timed out after {timeout}s waiting for final FlowCut callback"
            )
        return self.final_payload or {}


class _FlowCutCallbackHandler(BaseHTTPRequestHandler):
    recorder: _FlowCutCallbackRecorder

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception as exc:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({"error": f"invalid callback payload: {exc}"}).encode(
                    "utf-8"
                )
            )
            return

        self.recorder.record(payload)
        response = b'{"code":0}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: Any) -> None:
        return


class _LocalFlowCutCallbackServer:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        task_id: str,
        recorder: _FlowCutCallbackRecorder,
    ):
        self.host = host
        self.port = port
        self.task_id = task_id
        self.recorder = recorder
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.callback_url: str | None = None

    def __enter__(self) -> "_LocalFlowCutCallbackServer":
        handler = type(
            "FlowCutCallbackHandler",
            (_FlowCutCallbackHandler,),
            {"recorder": self.recorder},
        )
        self.server = ThreadingHTTPServer((self.host, self.port), handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="flowcut-callback-server",
            daemon=True,
        )
        self.thread.start()
        listen_host, listen_port = self.server.server_address[:2]
        if listen_host in {"0.0.0.0", "::"}:
            listen_host = "127.0.0.1"
        self.callback_url = (
            f"http://{listen_host}:{listen_port}/tasks/{self.task_id}/callback"
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5.0)


def _url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def submit_flowcut_task_with_retry(
    *,
    client: Any,
    base_url: str,
    payload: dict[str, Any],
    submit_timeout_s: float = 1800.0,
    retry_interval_seconds: float = 30.0,
    max_submit_attempts: int = 60,
) -> dict[str, Any]:
    endpoint = _url(base_url, "/v1/videos/repairs/flowcut")
    for attempt in range(1, max_submit_attempts + 1):
        response = client.post(endpoint, json=payload, timeout=submit_timeout_s)
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


def _build_payload(
    args: argparse.Namespace, *, callback_url: str
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "taskId": args.task_id,
        "timeout": -1,
        "callbackUrl": callback_url,
        "video_input_path": args.input_video,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "num_temporal_process_frames": args.num_temporal_process_frames,
    }
    if args.caption_file:
        payload["caption_file_path"] = args.caption_file
    if args.reference_video:
        payload["reference_video_path"] = args.reference_video
    if args.output_path:
        payload["output_path"] = args.output_path
    if args.perf_dump_path:
        payload["perf_dump_path"] = args.perf_dump_path
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit and poll a FlowCut-compatible Vivid-VR service task."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:31191")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--callback-url")
    parser.add_argument("--callback-log")
    parser.add_argument("--callback-host", default="127.0.0.1")
    parser.add_argument("--callback-port", type=int, default=0)
    parser.add_argument("--input-video", "--video-input-path", dest="input_video", required=True)
    parser.add_argument("--caption-file", "--caption-file-path", dest="caption_file")
    parser.add_argument(
        "--reference-video",
        "--reference-video-path",
        dest="reference_video",
    )
    parser.add_argument("--output-path")
    parser.add_argument("--perf-dump-path")
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-temporal-process-frames", type=int, default=121)
    parser.add_argument("--submit-timeout-s", type=float, default=1800.0)
    parser.add_argument("--submit-retry-interval-seconds", type=float, default=30.0)
    parser.add_argument("--max-submit-attempts", type=int, default=60)
    parser.add_argument("--poll-interval-seconds", type=float, default=30.0)
    parser.add_argument("--max-polls", type=int)
    parser.add_argument("--poll-timeout-s", type=float)
    parser.add_argument("--final-callback-timeout-s", type=float, default=60.0)
    args = parser.parse_args(argv)
    if args.callback_url and args.callback_log:
        parser.error("--callback-url and --callback-log are mutually exclusive")
    if not args.callback_url and not args.callback_log:
        parser.error("either --callback-url or --callback-log is required")
    return args


def main() -> int:
    args = parse_args()
    max_polls = args.max_polls
    if max_polls is None and args.poll_timeout_s is not None:
        max_polls = max(
            1,
            math.ceil(
                args.poll_timeout_s / max(args.poll_interval_seconds, 0.001)
            ),
        )

    callback_server = None
    callback_recorder = None
    callback_url = args.callback_url
    if args.callback_log:
        callback_recorder = _FlowCutCallbackRecorder(args.callback_log)
        callback_server = _LocalFlowCutCallbackServer(
            host=args.callback_host,
            port=args.callback_port,
            task_id=args.task_id,
            recorder=callback_recorder,
        )

    with (callback_server or threading.Lock()):
        if callback_server is not None:
            callback_url = callback_server.callback_url
        payload = _build_payload(args, callback_url=callback_url)
        print(
            json.dumps(
                {
                    "task_id": args.task_id,
                    "callback_url": callback_url,
                    "payload": payload,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        with httpx.Client(follow_redirects=True, trust_env=False) as client:
            submit_result = submit_flowcut_task_with_retry(
                client=client,
                base_url=args.base_url,
                payload=payload,
                submit_timeout_s=args.submit_timeout_s,
                retry_interval_seconds=args.submit_retry_interval_seconds,
                max_submit_attempts=args.max_submit_attempts,
            )
            print(json.dumps(submit_result, ensure_ascii=False), flush=True)
            final_progress = poll_accepted_task(
                client=client,
                base_url=args.base_url,
                task_id=args.task_id,
                poll_interval_seconds=args.poll_interval_seconds,
                max_polls=max_polls,
            )
            print(json.dumps(final_progress, ensure_ascii=False), flush=True)
        if final_progress.get("status") != "completed":
            raise FlowCutAcceptanceError(
                f"FlowCut progress ended with status={final_progress.get('status')}: "
                f"{final_progress}"
            )
        if callback_recorder is not None:
            final_callback = callback_recorder.wait_for_final(
                args.final_callback_timeout_s
            )
            print(json.dumps(final_callback, ensure_ascii=False), flush=True)
            if final_callback.get("status") != "succeeded":
                raise FlowCutAcceptanceError(
                    f"FlowCut callback ended with status={final_callback.get('status')}: "
                    f"{final_callback}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
