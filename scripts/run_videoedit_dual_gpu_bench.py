#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener


TERMINAL_STATUSES = {"completed", "failed", "deleted"}


class HttpClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}))

    def json_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with self.opener.open(req, timeout=timeout or self.timeout) as resp:
                body = resp.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} {method} {path}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"{method} {path} failed: {exc}") from exc
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))


@dataclass(frozen=True)
class GpuPoint:
    elapsed_s: float
    wall_time_s: float
    stage: str
    progress: Any
    gpu_index: int
    memory_used_mb: int
    memory_free_mb: int
    memory_total_mb: int
    gpu_util_percent: int


class GpuMonitor:
    def __init__(self, gpus: str, interval_s: float) -> None:
        self.gpus = gpus
        self.interval_s = interval_s
        self.points: list[GpuPoint] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._stage = "unknown"
        self._progress: Any = None
        self._started_monotonic = 0.0
        self._thread: threading.Thread | None = None

    def set_progress(self, stage: str, progress: Any) -> None:
        with self._lock:
            self._stage = stage or "unknown"
            self._progress = progress

    def start(self) -> None:
        self._started_monotonic = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="gpu-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_s * 4))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample_once()
            except Exception:
                # Keep the request running even if one nvidia-smi sample fails.
                pass
            self._stop.wait(self.interval_s)

    def _sample_once(self) -> None:
        cmd = [
            "nvidia-smi",
            "-i",
            self.gpus,
            "--query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        elapsed = time.monotonic() - self._started_monotonic
        wall = time.time()
        with self._lock:
            stage = self._stage
            progress = self._progress
        for raw_line in output.strip().splitlines():
            parts = [part.strip() for part in raw_line.split(",")]
            if len(parts) != 5:
                continue
            self.points.append(
                GpuPoint(
                    elapsed_s=elapsed,
                    wall_time_s=wall,
                    stage=stage,
                    progress=progress,
                    gpu_index=int(parts[0]),
                    memory_used_mb=int(parts[1]),
                    memory_free_mb=int(parts[2]),
                    memory_total_mb=int(parts[3]),
                    gpu_util_percent=int(parts[4]),
                )
            )


def sanitize_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = value.strip("._-")
    return value or "videoedit_case"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def query_gpu_memory(gpus: str) -> list[dict[str, int]]:
    cmd = [
        "nvidia-smi",
        "-i",
        gpus,
        "--query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    rows: list[dict[str, int]] = []
    for raw_line in output.strip().splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "gpu_index": int(parts[0]),
                "memory_used_mb": int(parts[1]),
                "memory_free_mb": int(parts[2]),
                "memory_total_mb": int(parts[3]),
                "gpu_util_percent": int(parts[4]),
            }
        )
    return rows


def write_gpu_samples_csv(path: Path, points: list[GpuPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "elapsed_s",
                "wall_time_s",
                "stage",
                "progress",
                "gpu_index",
                "memory_used_mb",
                "memory_free_mb",
                "memory_total_mb",
                "gpu_util_percent",
            ],
        )
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "elapsed_s": round(point.elapsed_s, 3),
                    "wall_time_s": round(point.wall_time_s, 3),
                    "stage": point.stage,
                    "progress": point.progress,
                    "gpu_index": point.gpu_index,
                    "memory_used_mb": point.memory_used_mb,
                    "memory_free_mb": point.memory_free_mb,
                    "memory_total_mb": point.memory_total_mb,
                    "gpu_util_percent": point.gpu_util_percent,
                }
            )


def snapshot_totals(points: list[GpuPoint], stage: str | None = None) -> list[dict[str, Any]]:
    grouped: dict[float, list[GpuPoint]] = {}
    for point in points:
        if stage is not None and point.stage != stage:
            continue
        grouped.setdefault(point.wall_time_s, []).append(point)
    rows = []
    for wall_time_s, group in grouped.items():
        rows.append(
            {
                "wall_time_s": wall_time_s,
                "stage": group[0].stage,
                "progress": group[0].progress,
                "total_memory_used_mb": sum(item.memory_used_mb for item in group),
                "max_gpu_memory_used_mb": max(item.memory_used_mb for item in group),
                "per_gpu_memory_used_mb": {
                    str(item.gpu_index): item.memory_used_mb for item in group
                },
            }
        )
    return rows


def peak_from_points(points: list[GpuPoint], stage: str | None = None) -> dict[str, Any]:
    totals = snapshot_totals(points, stage=stage)
    if not totals:
        return {
            "available": False,
            "total_memory_used_mb": None,
            "max_gpu_memory_used_mb": None,
            "per_gpu_memory_used_mb": {},
        }
    best = max(totals, key=lambda item: item["total_memory_used_mb"])
    return {
        "available": True,
        "stage": best["stage"],
        "progress": best["progress"],
        "wall_time_s": round(float(best["wall_time_s"]), 3),
        "total_memory_used_mb": int(best["total_memory_used_mb"]),
        "max_gpu_memory_used_mb": int(best["max_gpu_memory_used_mb"]),
        "per_gpu_memory_used_mb": best["per_gpu_memory_used_mb"],
    }


def idle_summary(rows: list[dict[str, int]]) -> dict[str, Any]:
    return {
        "total_memory_used_mb": sum(row["memory_used_mb"] for row in rows),
        "per_gpu_memory_used_mb": {
            str(row["gpu_index"]): row["memory_used_mb"] for row in rows
        },
        "raw": rows,
    }


def extract_perf_summary(perf_path: Path) -> dict[str, Any]:
    data = read_json(perf_path)
    if not data:
        return {"available": False}
    stages = data.get("steps") or data.get("stages") or []
    stage_ms: dict[str, float] = {}
    if isinstance(stages, list):
        for item in stages:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            duration = item.get("duration_ms", item.get("execution_time_ms"))
            if name is not None and duration is not None:
                stage_ms[str(name)] = float(duration)
    return {
        "available": True,
        "total_duration_ms": data.get("total_duration_ms"),
        "videoedit_denoising_stage_ms": stage_ms.get("VideoEditDenoisingStage"),
        "videoedit_decoding_stage_ms": stage_ms.get("VideoEditDecodingStage"),
        "stage_ms": stage_ms,
        "memory_checkpoints": data.get("memory_checkpoints", {}),
    }


def build_payload(
    args: argparse.Namespace,
    *,
    task_id: str,
    output_path: Path,
    perf_path: Path,
    enable_teacache: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task_id": task_id,
        "timeout": args.server_task_timeout,
        "prompt": args.prompt,
        "video_input_path": str(args.video_input_path),
        "mask_input_path": str(args.mask_input_path),
        "output_storage": "local",
        "output_path": str(output_path),
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "enable_teacache": enable_teacache,
        "perf_dump_path": str(perf_path),
    }
    optional_fields = (
        "dynamic_cfg",
        "num_frames",
        "infer_len",
        "overlap",
        "seed",
        "dtype",
        "drop_reference_frame",
        "bbox_expand_scale",
        "dilate_px",
        "mask_scale",
        "feather_px",
        "adain_boundary_dilate",
        "decode_mode",
        "mask_downsample_mode",
    )
    for field in optional_fields:
        value = getattr(args, field)
        if value is not None:
            payload[field] = value
    if args.reference_image_url:
        payload["reference_image_url"] = args.reference_image_url
    if args.extra_json:
        extra = json.loads(args.extra_json)
        if not isinstance(extra, dict):
            raise ValueError("--extra-json must be a JSON object")
        payload.update(extra)
    return payload


def submit_when_available(
    client: HttpClient,
    payload: dict[str, Any],
    *,
    busy_sleep_s: float,
) -> dict[str, Any]:
    while True:
        response = client.json_request("POST", "/v1/videos/repairs", payload)
        code = int(response.get("code", -1))
        if code == 0:
            return response
        message = str(response.get("message") or response.get("reason") or response)
        if code == 2 or "A task is running" in message:
            print(f"[busy] task is running, retry in {busy_sleep_s:.1f}s")
            time.sleep(busy_sleep_s)
            continue
        raise RuntimeError(f"submit failed: code={code}, message={message}")


def poll_until_done(
    client: HttpClient,
    task_id: str,
    *,
    poll_interval_s: float,
    monitor: GpuMonitor,
    progress_path: Path,
) -> dict[str, Any]:
    last = None
    while True:
        status = client.json_request(
            "GET", f"/v1/videos/{quote(task_id, safe='')}/progress"
        )
        state = str(status.get("status") or "")
        progress = status.get("progress")
        progress_payload = read_json(progress_path) or {}
        stage = str(status.get("stage") or progress_payload.get("stage") or "unknown")
        progress = progress_payload.get("progress", progress)
        monitor.set_progress(stage, progress)
        current = (state, progress, stage)
        if current != last:
            print(f"[poll] {task_id}: status={state} progress={progress} stage={stage}")
            last = current
        if state in TERMINAL_STATUSES:
            return status
        time.sleep(poll_interval_s)


def run_variant(
    args: argparse.Namespace,
    client: HttpClient,
    *,
    variant_name: str,
    enable_teacache: bool,
) -> dict[str, Any]:
    task_id = f"{args.task_prefix}{args.run_id}_{variant_name}"
    output_path = args.output_dir / f"{task_id}.mp4"
    progress_path = output_path.parent / f"{task_id}.progress.json"
    perf_path = args.output_dir / "perf" / f"{task_id}.json"
    payload_path = args.output_dir / "payloads" / f"{task_id}.json"
    gpu_samples_path = args.output_dir / "gpu_samples" / f"{task_id}.csv"

    payload = build_payload(
        args,
        task_id=task_id,
        output_path=output_path,
        perf_path=perf_path,
        enable_teacache=enable_teacache,
    )
    write_json(payload_path, payload)

    if args.cooldown_s > 0:
        time.sleep(args.cooldown_s)
    idle = idle_summary(query_gpu_memory(args.gpus))
    print(
        f"[run] {variant_name}: idle_total={idle['total_memory_used_mb']} MB "
        f"task_id={task_id}"
    )

    monitor = GpuMonitor(args.gpus, args.sample_interval_s)
    monitor.set_progress("submitting", 0)
    monitor.start()
    started = time.monotonic()
    final_status: dict[str, Any] | None = None
    error: str | None = None
    try:
        submit_when_available(client, payload, busy_sleep_s=args.busy_sleep_s)
        final_status = poll_until_done(
            client,
            task_id,
            poll_interval_s=args.poll_interval_s,
            monitor=monitor,
            progress_path=progress_path,
        )
        if final_status.get("status") != "completed":
            error = str(final_status.get("reason") or final_status.get("error") or "failed")
    except Exception as exc:
        error = str(exc)
    finally:
        monitor.stop()
    elapsed_s = time.monotonic() - started

    write_gpu_samples_csv(gpu_samples_path, monitor.points)
    perf_summary = extract_perf_summary(perf_path)
    run_peak = peak_from_points(monitor.points)
    dit_peak = peak_from_points(monitor.points, stage="denoising")

    record = {
        "variant": variant_name,
        "task_id": task_id,
        "status": "failed" if error else "completed",
        "error": error,
        "enable_teacache": enable_teacache,
        "elapsed_s": round(elapsed_s, 3),
        "idle_gpu_memory": idle,
        "peak_run_gpu_memory": run_peak,
        "peak_dit_gpu_memory": dit_peak,
        "output_path": str(output_path),
        "progress_path": str(progress_path),
        "perf_path": str(perf_path),
        "payload_path": str(payload_path),
        "gpu_samples_path": str(gpu_samples_path),
        "final_status": final_status,
        "perf_summary": perf_summary,
    }
    write_json(args.output_dir / "records" / f"{task_id}.json", record)
    if error:
        print(f"[failed] {variant_name}: {error}")
    else:
        print(
            f"[done] {variant_name}: elapsed={elapsed_s:.3f}s "
            f"peak_run={run_peak.get('total_memory_used_mb')} MB "
            f"peak_dit={dit_peak.get('total_memory_used_mb')} MB"
        )
    return record


def write_summary_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "task_id",
                "status",
                "enable_teacache",
                "elapsed_s",
                "idle_total_memory_used_mb",
                "peak_run_total_memory_used_mb",
                "peak_dit_total_memory_used_mb",
                "perf_total_duration_ms",
                "perf_videoedit_denoising_stage_ms",
                "perf_videoedit_decoding_stage_ms",
                "output_path",
                "error",
            ],
        )
        writer.writeheader()
        for record in records:
            perf = record.get("perf_summary") or {}
            writer.writerow(
                {
                    "variant": record.get("variant"),
                    "task_id": record.get("task_id"),
                    "status": record.get("status"),
                    "enable_teacache": record.get("enable_teacache"),
                    "elapsed_s": record.get("elapsed_s"),
                    "idle_total_memory_used_mb": (
                        record.get("idle_gpu_memory") or {}
                    ).get("total_memory_used_mb"),
                    "peak_run_total_memory_used_mb": (
                        record.get("peak_run_gpu_memory") or {}
                    ).get("total_memory_used_mb"),
                    "peak_dit_total_memory_used_mb": (
                        record.get("peak_dit_gpu_memory") or {}
                    ).get("total_memory_used_mb"),
                    "perf_total_duration_ms": perf.get("total_duration_ms"),
                    "perf_videoedit_denoising_stage_ms": perf.get(
                        "videoedit_denoising_stage_ms"
                    ),
                    "perf_videoedit_decoding_stage_ms": perf.get(
                        "videoedit_decoding_stage_ms"
                    ),
                    "output_path": record.get("output_path"),
                    "error": record.get("error"),
                }
            )


def write_stage_summary_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage_names = sorted(
        {
            stage_name
            for record in records
            for stage_name in (
                (record.get("perf_summary") or {}).get("stage_ms") or {}
            ).keys()
        }
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "task_id",
                "stage",
                "duration_ms",
                "duration_s",
            ],
        )
        writer.writeheader()
        for record in records:
            stage_ms = (record.get("perf_summary") or {}).get("stage_ms") or {}
            for stage_name in stage_names:
                duration_ms = stage_ms.get(stage_name)
                if duration_ms is None:
                    continue
                writer.writerow(
                    {
                        "variant": record.get("variant"),
                        "task_id": record.get("task_id"),
                        "stage": stage_name,
                        "duration_ms": duration_ms,
                        "duration_s": round(float(duration_ms) / 1000.0, 3),
                    }
                )


def wait_for_server(client: HttpClient, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            client.json_request("GET", "/health", timeout=5)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"server is not healthy after {timeout_s}s: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one VideoEdit sample twice against an existing dual-GPU service: "
            "no TeaCache and TeaCache, with wall-time and nvidia-smi memory sampling."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--video-input-path", type=Path, required=True)
    parser.add_argument("--mask-input-path", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-name", default="videoedit")
    parser.add_argument("--run-id")
    parser.add_argument("--task-prefix", default="videoedit_bench_")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--sample-interval-s", type=float, default=0.5)
    parser.add_argument("--poll-interval-s", type=float, default=1.0)
    parser.add_argument("--busy-sleep-s", type=float, default=10.0)
    parser.add_argument("--cooldown-s", type=float, default=5.0)
    parser.add_argument("--http-timeout-s", type=float, default=120.0)
    parser.add_argument("--wait-server-timeout-s", type=float, default=120.0)

    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--dynamic-cfg", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--infer-len", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--drop-reference-frame", action=argparse.BooleanOptionalAction)
    parser.add_argument("--reference-image-url")
    parser.add_argument("--bbox-expand-scale", type=float)
    parser.add_argument("--dilate-px", type=int)
    parser.add_argument("--mask-scale", type=float)
    parser.add_argument("--feather-px", type=int)
    parser.add_argument("--adain-boundary-dilate", type=int)
    parser.add_argument("--decode-mode", choices=("eager", "stream"))
    parser.add_argument("--mask-downsample-mode", choices=("nearest", "nearest-exact"))
    parser.add_argument("--server-task-timeout", type=int, default=-1)
    parser.add_argument(
        "--extra-json",
        help="JSON object merged into both request payloads after regular fields.",
    )
    args = parser.parse_args()

    args.video_input_path = args.video_input_path.expanduser().resolve()
    args.mask_input_path = args.mask_input_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.run_id is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.run_id = f"{sanitize_id(args.case_name)}_{stamp}"
    else:
        args.run_id = sanitize_id(args.run_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def main() -> int:
    args = parse_args()
    client = HttpClient(args.base_url, timeout=args.http_timeout_s)
    wait_for_server(client, args.wait_server_timeout_s)

    records = [
        run_variant(
            args,
            client,
            variant_name="dual_gpu",
            enable_teacache=False,
        ),
        run_variant(
            args,
            client,
            variant_name="dual_gpu_teacache",
            enable_teacache=True,
        ),
    ]
    write_json(args.output_dir / "summary.json", records)
    write_summary_csv(args.output_dir / "summary.csv", records)
    write_stage_summary_csv(args.output_dir / "stage_summary.csv", records)
    failures = sum(1 for record in records if record.get("status") != "completed")
    print(f"[summary] {args.output_dir / 'summary.csv'}")
    print(f"[stage-summary] {args.output_dir / 'stage_summary.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
