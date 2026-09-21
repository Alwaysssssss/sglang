"""Warm the full acceptance shape before collecting any performance numbers."""

import argparse
import json
import os
import time
import uuid
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--resolution", default="3840x2160", help="Height x width")
    args = parser.parse_args()
    base = "http://127.0.0.1:" + os.environ.get("VSR_PORT", "30176")
    opener = build_opener(ProxyHandler({}))

    def call(path, payload=None):
        request = Request(
            base + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with opener.open(request, timeout=30) as response:
            return json.load(response)

    deadline = time.monotonic() + 3600
    while True:
        try:
            if call("/health").get("status") == "ok":
                break
        except (URLError, ConnectionError, TimeoutError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError("Server did not become ready")
        time.sleep(5)
    job = call(
        "/v1/videos/restorations",
        {
            "taskId": "docker_warmup_" + uuid.uuid4().hex,
            "video_input_path": args.input,
            "target_resolution": args.resolution,
        },
    )
    if job.get("code") != 0:
        raise RuntimeError(job)
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        state = call("/v1/videos/" + job["id"])
        if state["status"] == "failed":
            raise RuntimeError(state)
        if state["status"] == "completed":
            print(
                "Full-resolution warmup completed; its latency is excluded.", flush=True
            )
            return
        time.sleep(5)
    raise TimeoutError("Full-resolution warmup did not complete")


if __name__ == "__main__":
    main()
