#!/usr/bin/env python3
"""Submit normal/DMD requests together and assert global queue serialization."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

ACTIVE = {"dispatching", "running", "cancelling"}
TERMINAL = {"completed", "failed", "cancelled"}


def request_json(method: str, url: str, payload: dict[str, Any] | None = None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"{method} {url}: {error.read().decode()}") from error


def load_payload(path: str, model: str, prefix: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["task_id"] = f"{prefix}-{uuid.uuid4().hex[:12]}"
    payload["model"] = model
    return payload


def run_pair(
    base_url: str,
    first: dict[str, Any],
    second: dict[str, Any],
    timeout: int,
) -> list[dict[str, Any]]:
    submit_url = f"{base_url}/v1/videos/repairs"
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(request_json, "POST", submit_url, payload)
            for payload in (first, second)
        ]
        submit_results = [future.result() for future in futures]
    task_ids = [result["task_id"] for result in submit_results]

    observations = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        states = [
            request_json("GET", f"{base_url}/v1/videos/{task_id}")
            for task_id in task_ids
        ]
        active_count = sum(state.get("status") in ACTIVE for state in states)
        if active_count > 1:
            raise AssertionError(f"More than one active task observed: {states}")
        observations.append(
            {
                "timestamp": time.time(),
                "states": [
                    {
                        "id": state.get("id"),
                        "variant": state.get("variant"),
                        "status": state.get("status"),
                        "progress": state.get("progress"),
                    }
                    for state in states
                ],
            }
        )
        if all(state.get("status") in TERMINAL for state in states):
            return states
        time.sleep(1)
    raise TimeoutError(f"Tasks did not finish within {timeout}s: {observations[-1:]}")


def queued_cancel_test(base_url: str, template: dict[str, Any]) -> None:
    first = copy.deepcopy(template)
    first["task_id"] = f"cancel-blocker-{uuid.uuid4().hex[:12]}"
    first["model"] = "videoedit-normal"
    second = copy.deepcopy(template)
    second["task_id"] = f"cancel-queued-{uuid.uuid4().hex[:12]}"
    second["model"] = "videoedit-dmd"

    request_json("POST", f"{base_url}/v1/videos/repairs", first)
    request_json("POST", f"{base_url}/v1/videos/repairs", second)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        state = request_json("GET", f"{base_url}/v1/videos/{second['task_id']}")
        if state.get("status") == "queued":
            cancelled = request_json(
                "DELETE", f"{base_url}/v1/videos/{second['task_id']}"
            )
            if cancelled.get("status") != "cancelled":
                raise AssertionError(f"Queued cancellation failed: {cancelled}")
            return
        time.sleep(0.5)
    raise AssertionError("Second task never reached queued state")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--normal-request", required=True)
    parser.add_argument("--dmd-request", required=True)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--test-queued-cancel", action="store_true")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    normal = load_payload(args.normal_request, "videoedit-normal", "normal")
    dmd = load_payload(args.dmd_request, "videoedit-dmd", "dmd")
    first_result = run_pair(base_url, normal, dmd, args.timeout)

    normal_2 = load_payload(args.normal_request, "videoedit-normal", "normal-reverse")
    dmd_2 = load_payload(args.dmd_request, "videoedit-dmd", "dmd-reverse")
    second_result = run_pair(base_url, dmd_2, normal_2, args.timeout)

    if args.test_queued_cancel:
        queued_cancel_test(base_url, normal)
    print(json.dumps({"ok": True, "runs": [first_result, second_result]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
