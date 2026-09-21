# SPDX-License-Identifier: Apache-2.0
"""Real HTTP acceptance: URL/local input, callback, warm repeats and download."""

import argparse
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

urlopen = build_opener(ProxyHandler({})).open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30176")
    parser.add_argument("--input", default="../vsr/input/input.mp4")
    parser.add_argument("--output-dir", default="output_results/vsr/server_api_test")
    parser.add_argument("--warmup-resolution", default="320x640")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.input).resolve()
    callbacks = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            callbacks.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def do_GET(self):
            data = source.read_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    callback = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=callback.serve_forever, daemon=True).start()
    callback_url = f"http://127.0.0.1:{callback.server_port}"

    def call(path, payload=None):
        request = Request(
            args.base_url + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=120) as response:
            return json.load(response)

    def wait(job_id):
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            job = call("/v1/videos/" + job_id)
            if job["status"] in ("completed", "failed"):
                return job
            time.sleep(2)
        raise TimeoutError(job_id)

    deadline = time.monotonic() + 900
    while True:
        try:
            assert call("/health")["status"] == "ok"
            break
        except (URLError, ConnectionError):
            if time.monotonic() > deadline:
                raise
            time.sleep(5)
    endpoint = "/v1/videos/restorations"
    tag = str(int(time.time()))
    results = []
    try:
        bad = call(endpoint, {"video_input_path": "/nonexistent.mp4"})
        assert bad["code"] == 1, bad
        for i, resolution in enumerate(
            [args.warmup_resolution] * 3 + ["3840x2160"] * 2
        ):
            job_id = f"api_{tag}_{i}"
            payload = {
                "taskId": job_id,
                "video_input_path": str(source),
                "target_resolution": resolution,
                "callbackUrl": callback_url,
            }
            if i == 1:
                payload.pop("video_input_path")
                payload["videoUrl"] = callback_url + "/input.mp4"
            started = time.perf_counter()
            submitted = call(endpoint, payload)
            assert submitted["code"] == 0, submitted
            if i == 3:
                busy = call(endpoint, {**payload, "taskId": job_id + "_busy"})
                assert busy["code"] == 2, busy
            job = wait(job_id)
            assert job["status"] == "completed", job
            elapsed = time.perf_counter() - started
            with urlopen(
                args.base_url + "/v1/videos/" + job_id + "/content"
            ) as response:
                data = response.read()
            path = output / f"{i}.mp4"
            path.write_bytes(data)
            record = {
                "index": i,
                "warmup": i < 3,
                "request_wall_s": elapsed,
                "response": job,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            results.append(record)
            (output / "results.json").write_text(json.dumps(results, indent=2))
            print(json.dumps(record), flush=True)
        assert results[3]["sha256"] == results[4]["sha256"]
        duplicate = call(
            endpoint,
            {"taskId": results[4]["response"]["id"], "video_input_path": str(source)},
        )
        assert duplicate["code"] == 1
        # A timed-out request must release admission and not poison VAE caches.
        timed = call(
            endpoint,
            {
                "taskId": f"api_{tag}_timeout",
                "video_input_path": str(source),
                "target_resolution": "3840x2160",
                "timeout": 1,
            },
        )
        assert timed["code"] == 0, timed
        failure = wait(timed["id"])
        assert failure["status"] == "failed", failure
        recovery = call(
            endpoint,
            {
                "taskId": f"api_{tag}_recovery",
                "video_input_path": str(source),
                "target_resolution": args.warmup_resolution,
            },
        )
        assert recovery["code"] == 0, recovery
        recovered = wait(recovery["id"])
        assert recovered["status"] == "completed", recovered
        for _ in range(20):
            if len(callbacks) >= 5:
                break
            time.sleep(1)
        assert len(callbacks) == 5, callbacks
        summary = {
            "passed": True,
            "callbacks": callbacks,
            "timeout": failure,
            "recovery": recovered,
            "warm_mean_s": sum(r["request_wall_s"] for r in results[3:]) / 2,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        print("HTTP_ACCEPTANCE_PASSED", flush=True)
    finally:
        callback.shutdown()
        callback.server_close()


if __name__ == "__main__":
    main()
