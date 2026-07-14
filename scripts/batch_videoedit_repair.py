#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
TERMINAL_STATUSES = {"completed", "failed", "deleted"}


def _int_or_float(value: str) -> int | float:
    if any(marker in value for marker in (".", "e", "E")):
        return float(value)
    return int(value)


@dataclass(frozen=True)
class RepairJob:
    video_id: str
    prompt: str
    video_path: Path
    mask_path: Path
    output_path: Path


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
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with self.opener.open(request, timeout=timeout or self.timeout) as response:
                body = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} {method} {path}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"{method} {path} failed: {exc}") from exc
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            if isinstance(item, dict):
                rows.append(item)
    return rows


def resolve_existing_path(base_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    return path if path.exists() else None


def find_video_path(base_dir: Path, video_id: str, caption_row: dict[str, Any]) -> Path | None:
    explicit = resolve_existing_path(base_dir, caption_row.get("video"))
    if explicit is not None:
        return explicit
    for dirname in ("videos", "video"):
        directory = base_dir / dirname
        if not directory.is_dir():
            continue
        for ext in VIDEO_EXTENSIONS:
            candidate = directory / f"{video_id}{ext}"
            if candidate.exists():
                return candidate.resolve()
    return None


def find_mask_path(
    base_dir: Path,
    video_id: str,
    index_by_id: dict[str, dict[str, Any]],
) -> Path | None:
    index_row = index_by_id.get(video_id, {})
    for key in ("mask_video", "mask", "mask_input_path"):
        explicit = resolve_existing_path(base_dir, index_row.get(key))
        if explicit is not None:
            return explicit
    for dirname in ("masks", "video_masks"):
        directory = base_dir / dirname
        if not directory.is_dir():
            continue
        for suffix in ("_mask", ""):
            for ext in VIDEO_EXTENSIONS:
                candidate = directory / f"{video_id}{suffix}{ext}"
                if candidate.exists():
                    return candidate.resolve()
    return None


def build_jobs(args: argparse.Namespace) -> list[RepairJob]:
    base_dir = args.data_dir.expanduser().resolve()
    captions_path = args.captions_path.expanduser()
    if not captions_path.is_absolute():
        captions_path = base_dir / captions_path
    index_path = args.index_path.expanduser()
    if not index_path.is_absolute():
        index_path = base_dir / index_path

    caption_rows = read_jsonl(captions_path)
    index_by_id = {
        str(row["id"]): row for row in read_jsonl(index_path) if row.get("id") is not None
    }
    selected_ids = set(args.ids or [])
    jobs: list[RepairJob] = []
    for row in caption_rows:
        video_id = str(row.get("id") or "").strip()
        prompt = str(row.get("caption") or row.get("prompt") or "").strip()
        if not video_id:
            print(f"[skip] caption row without id: {row}", file=sys.stderr)
            continue
        if selected_ids and video_id not in selected_ids:
            continue
        if not prompt:
            print(f"[skip] {video_id}: empty prompt", file=sys.stderr)
            continue
        video_path = find_video_path(base_dir, video_id, row)
        mask_path = find_mask_path(base_dir, video_id, index_by_id)
        if video_path is None:
            print(f"[skip] {video_id}: video not found", file=sys.stderr)
            continue
        if mask_path is None:
            print(f"[skip] {video_id}: mask not found", file=sys.stderr)
            continue
        jobs.append(
            RepairJob(
                video_id=video_id,
                prompt=prompt,
                video_path=video_path,
                mask_path=mask_path,
                output_path=(args.output_dir.expanduser().resolve() / f"{video_id}.mp4"),
            )
        )
        if args.limit and len(jobs) >= args.limit:
            break
    return jobs


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def extract_first_frame(video_path: Path, reference_path: Path) -> Path:
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    if reference_path.exists():
        return reference_path
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(reference_path),
    ]
    subprocess.run(cmd, check=True)
    return reference_path


def build_payload(args: argparse.Namespace, job: RepairJob) -> dict[str, Any]:
    task_id = f"{args.task_prefix}{job.video_id}"
    payload: dict[str, Any] = {
        "taskId": task_id,
        "timeout": args.server_task_timeout,
        "prompt": job.prompt,
        "video_input_path": str(job.video_path),
        "mask_input_path": str(job.mask_path),
        "output_path": str(job.output_path),
        "num_frames": args.num_frames,
        "infer_len": args.infer_len,
        "overlap": args.overlap,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "dtype": args.dtype,
        "bbox_expand_scale": args.bbox_expand_scale,
        "dilate_px": args.dilate_px,
        "mask_scale": args.mask_scale,
        "enable_teacache": False,
        "enable_frame_interpolation": False,
        "enable_upscaling": False,
    }
    if args.negative_prompt is not None:
        payload["negative_prompt"] = args.negative_prompt
    if args.generator_device is not None:
        payload["generator_device"] = args.generator_device
    if not args.no_denoise_trace:
        payload["denoise_trace_path"] = str(
            args.denoise_trace_dir / f"{job.video_id}.jsonl"
        )
    if args.teacache_residual_trace_dir is not None:
        payload["teacache_residual_trace_path"] = str(
            args.teacache_residual_trace_dir / f"{job.video_id}.jsonl"
        )
        payload["teacache_thresh"] = args.teacache_thresh
        payload["teacache_start_skipping"] = args.teacache_start_skipping
        payload["teacache_end_skipping"] = args.teacache_end_skipping
    if args.reference_mode == "none":
        payload["drop_reference_frame"] = False
    elif args.reference_mode == "first-frame":
        reference_path = extract_first_frame(
            job.video_path,
            args.reference_dir.expanduser().resolve() / f"{job.video_id}.png",
        )
        payload["referenceImageUrl"] = str(reference_path)
        payload["drop_reference_frame"] = True
    if args.perf_dir is not None:
        perf_dir = args.perf_dir.expanduser().resolve()
        perf_dir.mkdir(parents=True, exist_ok=True)
        payload["perf_dump_path"] = str(perf_dir / f"{job.video_id}.json")
    return payload


def wait_for_server(client: HttpClient, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            client.json_request("GET", "/health", timeout=5)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"Server did not become healthy within {timeout_s}s: {last_error}")


def submit_when_available(
    client: HttpClient,
    payload: dict[str, Any],
    *,
    busy_sleep: float,
) -> None:
    while True:
        response = client.json_request("POST", "/v1/videos/repairs", payload)
        code = int(response.get("code", -1))
        if code == 0:
            return
        message = str(response.get("message") or response.get("reason") or response)
        if code == 2 or "A task is running" in message:
            print(f"[busy] server is running another task, retry in {busy_sleep:.0f}s")
            time.sleep(busy_sleep)
            continue
        raise RuntimeError(f"submit failed: code={code}, message={message}")


def poll_until_done(
    client: HttpClient,
    task_id: str,
    *,
    poll_interval: float,
) -> dict[str, Any]:
    last_progress = None
    while True:
        quoted_id = quote(task_id, safe="")
        status = client.json_request("GET", f"/v1/videos/{quoted_id}/progress")
        state = str(status.get("status") or "")
        progress = status.get("progress")
        if progress != last_progress:
            print(f"[poll] {task_id}: status={state} progress={progress}")
            last_progress = progress
        if state in TERMINAL_STATUSES:
            return status
        time.sleep(poll_interval)


def run_job(args: argparse.Namespace, client: HttpClient, job: RepairJob) -> dict[str, Any]:
    task_id = f"{args.task_prefix}{job.video_id}"
    default_denoise_trace_path = None
    if not args.no_denoise_trace:
        default_denoise_trace_path = str(args.denoise_trace_dir / f"{job.video_id}.jsonl")
    payload = build_payload(args, job)
    denoise_trace_path = payload.get("denoise_trace_path")
    residual_trace_path = payload.get("teacache_residual_trace_path")
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return {
            "id": job.video_id,
            "task_id": task_id,
            "status": "dry_run",
            "output_path": str(job.output_path),
            "denoise_trace_path": denoise_trace_path,
            "teacache_residual_trace_path": residual_trace_path,
        }

    if job.output_path.exists() and not args.force:
        print(f"[skip] {job.video_id}: output exists: {job.output_path}")
        return {
            "id": job.video_id,
            "task_id": task_id,
            "status": "skipped",
            "output_path": str(job.output_path),
            "denoise_trace_path": default_denoise_trace_path,
            "teacache_residual_trace_path": residual_trace_path,
            "reason": "output exists",
        }

    print(f"[submit] {job.video_id}: {job.video_path.name} -> {job.output_path}")
    if denoise_trace_path:
        trace_path = Path(str(denoise_trace_path))
        if trace_path.exists():
            trace_path.unlink()
    if residual_trace_path:
        trace_path = Path(str(residual_trace_path))
        if trace_path.exists():
            trace_path.unlink()
    submit_when_available(client, payload, busy_sleep=args.busy_sleep)
    final_status = poll_until_done(
        client,
        task_id,
        poll_interval=args.poll_interval,
    )
    status = final_status.get("status")
    if status != "completed":
        reason = final_status.get("reason") or final_status.get("error") or "failed"
        raise RuntimeError(f"{task_id} failed: {reason}")

    return {
        "id": job.video_id,
        "task_id": task_id,
        "status": "completed",
        "output_path": str(job.output_path),
        "denoise_trace_path": denoise_trace_path,
        "teacache_residual_trace_path": residual_trace_path,
        "server_file_path": final_status.get("file_path"),
        "url": final_status.get("url"),
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Batch submit VideoEdit repair requests to a running SGLang server."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("~/workspace/case/erase_data_case"),
    )
    parser.add_argument("--captions-path", type=Path, default=Path("captions.jsonl"))
    parser.add_argument("--index-path", type=Path, default=Path("index.jsonl"))
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--perf-dir", type=Path)
    parser.add_argument("--denoise-trace-dir", type=Path)
    parser.add_argument("--no-denoise-trace", action="store_true")
    parser.add_argument("--teacache-residual-trace-dir", type=Path)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument(
        "--reference-mode",
        choices=("none", "first-frame"),
        default="none",
        help="Use no-reference API path by default; first-frame extracts a png and passes it as referenceImageUrl.",
    )
    parser.add_argument("--task-prefix", default="erase_data_case_")
    parser.add_argument("--ids", nargs="*", help="Only run these ids.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--num-frames", type=int, default=-1)
    parser.add_argument("--infer-len", type=int, default=81)
    parser.add_argument("--overlap", type=int, default=9)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--bbox-expand-scale", type=float, default=1.6)
    parser.add_argument("--dilate-px", type=int, default=15)
    parser.add_argument("--mask-scale", type=float, default=1.2)
    parser.add_argument("--generator-device")
    parser.add_argument("--negative-prompt")
    parser.add_argument("--teacache-thresh", type=float, default=0.3)
    parser.add_argument("--teacache-start-skipping", type=_int_or_float, default=5)
    parser.add_argument("--teacache-end-skipping", type=_int_or_float, default=1.0)
    parser.add_argument("--server-task-timeout", type=int, default=-1)

    parser.add_argument("--http-timeout", type=float, default=120.0)
    parser.add_argument("--wait-server-timeout", type=float, default=300.0)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--busy-sleep", type=float, default=30.0)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = repo_root / "outputs" / "erase_data_case_repair"
    else:
        args.output_dir = args.output_dir.expanduser().resolve()
    if args.manifest_path is None:
        args.manifest_path = args.output_dir / "manifest.jsonl"
    else:
        args.manifest_path = args.manifest_path.expanduser().resolve()
    if args.denoise_trace_dir is None:
        args.denoise_trace_dir = args.output_dir / "denoise_traces"
    else:
        args.denoise_trace_dir = args.denoise_trace_dir.expanduser().resolve()
    if args.reference_dir is None:
        args.reference_dir = args.output_dir / "reference_frames"
    else:
        args.reference_dir = args.reference_dir.expanduser().resolve()
    if args.teacache_residual_trace_dir is not None:
        args.teacache_residual_trace_dir = (
            args.teacache_residual_trace_dir.expanduser().resolve()
        )
    return args


def main() -> int:
    args = parse_args()
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    client = HttpClient(args.base_url, timeout=args.http_timeout)
    if not args.dry_run:
        wait_for_server(client, args.wait_server_timeout)

    jobs = build_jobs(args)
    print(f"[start] jobs={len(jobs)} output_dir={args.output_dir}")
    if not jobs:
        return 1

    failures = 0
    for index, job in enumerate(jobs, start=1):
        print(f"[job] {index}/{len(jobs)} id={job.video_id}")
        started_at = time.time()
        try:
            record = run_job(args, client, job)
        except Exception as exc:
            failures += 1
            record = {
                "id": job.video_id,
                "task_id": f"{args.task_prefix}{job.video_id}",
                "status": "failed",
                "output_path": str(job.output_path),
                "denoise_trace_path": None
                if args.no_denoise_trace
                else str(args.denoise_trace_dir / f"{job.video_id}.jsonl"),
                "teacache_residual_trace_path": None
                if args.teacache_residual_trace_dir is None
                else str(args.teacache_residual_trace_dir / f"{job.video_id}.jsonl"),
                "reason": str(exc),
            }
            print(f"[failed] {job.video_id}: {exc}", file=sys.stderr)
        record["elapsed_s"] = round(time.time() - started_at, 3)
        if not args.dry_run:
            write_jsonl(args.manifest_path, record)

    print(f"[done] total={len(jobs)} failures={failures} manifest={args.manifest_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
