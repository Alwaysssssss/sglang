#!/usr/bin/env python3
"""Compare running VideoEdit services using an isolated local request."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

from worker import ROOT, decoded_validation, write_json


def run(args) -> None:
    out = Path(args.out).resolve()
    out.relative_to(ROOT)
    out.mkdir(parents=True, exist_ok=False)
    (out / "outputs").mkdir()

    template_path = Path(args.request_template).resolve()
    template_path.relative_to(ROOT)
    template = json.loads(template_path.read_text())
    for key in ("task_id", "output_path", "perf_dump_path"):
        template.pop(key, None)
    if args.model:
        template["model"] = args.model
    if template.get("model") == "videoedit-dmd":
        template.update(
            num_inference_steps=4,
            guidance_scale=1.0,
            dynamic_cfg=False,
            negative_prompt=None,
            enable_teacache=False,
        )
    if args.geometry == "full":
        template.update(chunk_bbox_mode="global", bbox_expand_scale=100.0)
    template["timeout"] = args.request_timeout
    write_json(out / "request-template.json", template)

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def api(path: str, payload=None):
        request = urllib.request.Request(
            args.base_url + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with opener.open(request, timeout=60) as response:
            return json.load(response)

    write_json(
        out / "service.json",
        {
            "base_url": args.base_url,
            "gpu": args.gpu,
            "geometry": args.geometry,
            "container": json.loads(
                subprocess.check_output(
                    ["docker", "inspect", args.container], text=True
                )
            )[0],
            "started_at": time.time(),
        },
    )
    log_path = Path(args.service_log)
    log_start = log_path.stat().st_size if log_path.exists() else None
    write_json(out / "log-start.json", {"path": str(log_path), "bytes": log_start})

    run_id = uuid.uuid4().hex[:8]
    for index in range(args.requests):
        phase = "warmup" if index == 0 else f"timed{index}"
        health = api("/health")
        write_json(out / f"{phase}-health.json", health)
        queue = health.get("queue", {})
        if any(
            queue.get(key, 0)
            for key in ("queued", "running", "dispatching", "cancelling")
        ):
            raise RuntimeError(
                "Online service is busy; no benchmark request was submitted"
            )

        task_id = f"review-gpu{args.gpu.replace(',', '-')}-{run_id}-{phase}"
        output = out / "outputs" / f"{task_id}.mov"
        payload = dict(template, task_id=task_id, output_path=str(output))
        write_json(out / f"{phase}-request.json", payload)
        started = time.monotonic()
        admission = api("/v1/videos/repairs", payload)
        write_json(out / f"{phase}-admission.json", admission)
        if admission.get("code") != 0:
            raise RuntimeError(f"Online service rejected {phase}: {admission}")
        print(json.dumps({"phase": phase, "state": "submitted"}), flush=True)

        with (out / f"{phase}-samples.jsonl").open("w") as samples:
            while True:
                state = api("/v1/videos/" + task_id)
                now = time.monotonic()
                gpu = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        str(args.gpu),
                        "--query-gpu=uuid,memory.used,utilization.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=10,
                ).strip()
                samples.write(
                    json.dumps({"elapsed_s": now - started, "gpu": gpu}) + "\n"
                )
                samples.flush()
                if state.get("status") in ("completed", "failed", "cancelled"):
                    break
                if now - started > args.request_timeout:
                    raise TimeoutError(task_id)
                time.sleep(1)

        record = {
            "phase": phase,
            "status": state.get("status"),
            "response": state,
            "client_seconds": now - started,
        }
        write_json(out / f"{phase}-result.json", record)
        if state.get("status") != "completed":
            raise RuntimeError(f"Online {phase} failed: {state}")
        subprocess.run(
            ["docker", "exec", args.container, "chmod", "a+r", str(output)],
            check=True,
        )
        record["validation"] = decoded_validation(output, out, phase)
        write_json(out / f"{phase}-result.json", record)
        print(
            json.dumps(
                {"phase": phase, "state": "completed", "seconds": now - started}
            ),
            flush=True,
        )

    write_json(out / "health-after.json", api("/health"))
    if log_start is not None:
        with log_path.open("rb") as source:
            source.seek(log_start)
            (out / "service-request.log").write_bytes(source.read())
    write_json(
        out / "completion.json", {"status": "completed", "finished_at": time.time()}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--request-template", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:5402")
    parser.add_argument(
        "--gpu", default="7", help="Physical GPU index or comma-separated indices"
    )
    parser.add_argument("--model", choices=("videoedit-normal", "videoedit-dmd"))
    parser.add_argument("--container", default="videoedit_l40s")
    parser.add_argument(
        "--service-log",
        default=str(ROOT / ".local/videoedit-l40s/dual/logs/normal.log"),
    )
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--geometry", choices=("crop", "full"), default="crop")
    run(parser.parse_args())
