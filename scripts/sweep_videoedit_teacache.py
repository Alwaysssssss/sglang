#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from batch_videoedit_repair import (
    HttpClient,
    RepairJob,
    build_jobs,
    extract_first_frame,
    poll_until_done,
    submit_when_available,
    wait_for_server,
    write_jsonl,
)


TERMINAL_STATUSES = {"completed", "failed", "deleted"}


def _int_or_float(value: str) -> int | float:
    if any(marker in value for marker in (".", "e", "E")):
        return float(value)
    return int(value)


def _fmt_value(value: int | float | None) -> str:
    if value is None:
        return "none"
    if isinstance(value, int):
        return str(value)
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def _safe_mean(values: list[float]) -> float | None:
    return None if not values else float(statistics.mean(values))


def _safe_median(values: list[float]) -> float | None:
    return None if not values else float(statistics.median(values))


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


@dataclass(frozen=True)
class TeaCacheConfig:
    name: str
    enable_teacache: bool
    teacache_thresh: float | None = None
    teacache_start_skipping: int | float | None = None
    teacache_end_skipping: int | float | None = None

    @property
    def label(self) -> str:
        if not self.enable_teacache:
            return self.name
        return (
            f"{self.name}_thr{_fmt_value(self.teacache_thresh)}"
            f"_start{_fmt_value(self.teacache_start_skipping)}"
            f"_end{_fmt_value(self.teacache_end_skipping)}"
        )


def build_config_grid(args: argparse.Namespace) -> list[TeaCacheConfig]:
    configs: list[TeaCacheConfig] = []
    if args.include_baseline:
        configs.append(TeaCacheConfig(name="baseline", enable_teacache=False))

    for thresh in args.teacache_thresh_values:
        for start in args.teacache_start_skipping_values:
            for end in args.teacache_end_skipping_values:
                configs.append(
                    TeaCacheConfig(
                        name="teacache",
                        enable_teacache=True,
                        teacache_thresh=float(thresh),
                        teacache_start_skipping=start,
                        teacache_end_skipping=end,
                    )
                )
    return configs


def config_paths(
    run_root: Path,
    config: TeaCacheConfig,
) -> dict[str, Path]:
    config_root = run_root / "runs" / config.label
    return {
        "root": config_root,
        "videos": config_root / "videos",
        "perf": config_root / "perf",
        "denoise": config_root / "denoise_traces",
    }


def build_payload(
    args: argparse.Namespace,
    job: RepairJob,
    task_id: str,
    config: TeaCacheConfig,
    paths: dict[str, Path],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "taskId": task_id,
        "timeout": args.server_task_timeout,
        "prompt": job.prompt,
        "video_input_path": str(job.video_path),
        "mask_input_path": str(job.mask_path),
        "output_storage": "local",
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
        "enable_teacache": config.enable_teacache,
        "enable_frame_interpolation": False,
        "enable_upscaling": False,
        "perf_dump_path": str(paths["perf"] / f"{job.video_id}.json"),
        "denoise_trace_path": str(paths["denoise"] / f"{job.video_id}.jsonl"),
    }
    if args.negative_prompt is not None:
        payload["negative_prompt"] = args.negative_prompt
    if args.generator_device is not None:
        payload["generator_device"] = args.generator_device
    if config.enable_teacache:
        payload["teacache_thresh"] = config.teacache_thresh
        payload["teacache_start_skipping"] = config.teacache_start_skipping
        payload["teacache_end_skipping"] = config.teacache_end_skipping
    if args.reference_mode == "none":
        payload["drop_reference_frame"] = False
    elif args.reference_mode == "first-frame":
        reference_path = extract_first_frame(
            job.video_path,
            args.reference_dir / f"{job.video_id}.png",
        )
        payload["referenceImageUrl"] = str(reference_path)
        payload["drop_reference_frame"] = True
    return payload


def run_job(
    args: argparse.Namespace,
    client: HttpClient,
    base_job: RepairJob,
    *,
    config: TeaCacheConfig,
    config_index: int,
    job_index: int,
    run_root: Path,
) -> dict[str, Any]:
    paths = config_paths(run_root, config)
    for path in paths.values():
        if path.suffix:
            continue
        path.mkdir(parents=True, exist_ok=True)

    output_path = paths["videos"] / f"{base_job.video_id}.mp4"
    job = replace(base_job, output_path=output_path)
    task_id = (
        f"{args.task_prefix}{args.run_name}_cfg{config_index:02d}_"
        f"job{job_index:04d}_{config.label}_{base_job.video_id}"
    )
    payload = build_payload(args, job, task_id, config, paths)

    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return {
            "id": base_job.video_id,
            "task_id": task_id,
            "config": asdict(config),
            "config_label": config.label,
            "status": "dry_run",
            "output_path": str(output_path),
            "perf_dump_path": str(paths["perf"] / f"{base_job.video_id}.json"),
            "denoise_trace_path": str(paths["denoise"] / f"{base_job.video_id}.jsonl"),
            "payload": payload,
        }

    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --force or a different --run-name."
        )

    for clean_path in (
        output_path,
        paths["perf"] / f"{base_job.video_id}.json",
        paths["denoise"] / f"{base_job.video_id}.jsonl",
    ):
        if clean_path.exists():
            clean_path.unlink()

    print(
        f"[submit] config={config.label} video={base_job.video_id} "
        f"task_id={task_id}"
    )
    started_at = time.time()
    submit_when_available(client, payload, busy_sleep=args.busy_sleep)
    final_status = poll_until_done(
        client,
        task_id,
        poll_interval=args.poll_interval,
    )
    status = str(final_status.get("status") or "")
    record: dict[str, Any] = {
        "id": base_job.video_id,
        "task_id": task_id,
        "config": asdict(config),
        "config_label": config.label,
        "status": status,
        "output_path": str(output_path),
        "perf_dump_path": str(paths["perf"] / f"{base_job.video_id}.json"),
        "denoise_trace_path": str(paths["denoise"] / f"{base_job.video_id}.jsonl"),
        "elapsed_s": round(time.time() - started_at, 3),
        "server_file_path": final_status.get("file_path"),
        "url": final_status.get("url"),
        "peak_memory_mb": final_status.get("peak_memory_mb"),
        "inference_time_s": final_status.get("inference_time_s"),
    }
    if status != "completed":
        reason = final_status.get("reason") or final_status.get("error") or "failed"
        record["reason"] = str(reason)
        return record

    return record


def analyze_teacache_trace(
    trace_path: Path,
    task_ids: set[str],
) -> dict[str, Any]:
    if not trace_path.exists():
        return {
            "available": False,
            "reason": f"trace file not found: {trace_path}",
            "per_request": {},
        }

    per_request_records: dict[str, list[dict[str, Any]]] = {}
    with trace_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {trace_path}:{line_no}: {exc}"
                ) from exc
            if record.get("event") != "teacache_decision":
                continue
            request_id = str(record.get("request_id") or "")
            if request_id not in task_ids:
                continue
            per_request_records.setdefault(request_id, []).append(record)

    per_request: dict[str, Any] = {}
    all_candidate: list[float] = []
    all_rel_l1: list[float] = []
    all_rescaled_l1: list[float] = []
    replay_matches = 0
    replay_total = 0
    total_decisions = 0
    total_skipped = 0
    total_boundary = 0
    total_non_boundary = 0

    for request_id, records in per_request_records.items():
        candidate_values: list[float] = []
        rel_l1_values: list[float] = []
        rescaled_l1_values: list[float] = []
        skipped_steps: list[int] = []
        computed_steps: list[int] = []
        boundary_steps: list[int] = []
        request_replay_matches = 0
        request_replay_total = 0
        skipped = 0
        boundary = 0
        for record in records:
            total_decisions += 1
            denoise_step = int(record["denoise_step"])
            is_boundary_step = bool(record.get("is_boundary_step"))
            skipped_flag = bool(record.get("skipped"))
            threshold = record.get("threshold")
            candidate = record.get("candidate_accumulated")
            rel_l1 = record.get("rel_l1")
            rescaled_l1 = record.get("rescaled_l1")

            if is_boundary_step:
                boundary += 1
                total_boundary += 1
                boundary_steps.append(denoise_step)
            else:
                total_non_boundary += 1
            if skipped_flag:
                skipped += 1
                total_skipped += 1
                skipped_steps.append(denoise_step)
            else:
                computed_steps.append(denoise_step)

            expected_skip = False
            if not is_boundary_step and candidate is not None and threshold is not None:
                expected_skip = float(candidate) < float(threshold)
            request_replay_total += 1
            replay_total += 1
            if expected_skip == skipped_flag:
                request_replay_matches += 1
                replay_matches += 1

            if candidate is not None:
                candidate_values.append(float(candidate))
                all_candidate.append(float(candidate))
            if rel_l1 is not None:
                rel_l1_values.append(float(rel_l1))
                all_rel_l1.append(float(rel_l1))
            if rescaled_l1 is not None:
                rescaled_l1_values.append(float(rescaled_l1))
                all_rescaled_l1.append(float(rescaled_l1))

        per_request[request_id] = {
            "num_decisions": len(records),
            "num_skipped": skipped,
            "skip_ratio": (skipped / len(records)) if records else None,
            "boundary_ratio": (boundary / len(records)) if records else None,
            "replay_accuracy": (
                request_replay_matches / request_replay_total
                if request_replay_total
                else None
            ),
            "skipped_steps": sorted(set(skipped_steps)),
            "computed_steps": sorted(set(computed_steps)),
            "boundary_steps": sorted(set(boundary_steps)),
            "candidate_accumulated_p50": _safe_median(candidate_values),
            "candidate_accumulated_p90": _quantile(candidate_values, 0.9),
            "rel_l1_p50": _safe_median(rel_l1_values),
            "rel_l1_p90": _quantile(rel_l1_values, 0.9),
            "rescaled_l1_p50": _safe_median(rescaled_l1_values),
            "rescaled_l1_p90": _quantile(rescaled_l1_values, 0.9),
        }

    return {
        "available": True,
        "trace_path": str(trace_path),
        "matched_requests": len(per_request),
        "per_request": per_request,
        "aggregate": {
            "num_decisions": total_decisions,
            "num_skipped": total_skipped,
            "skip_ratio": (total_skipped / total_decisions) if total_decisions else None,
            "boundary_ratio": (
                total_boundary / total_decisions if total_decisions else None
            ),
            "non_boundary_ratio": (
                total_non_boundary / total_decisions if total_decisions else None
            ),
            "replay_accuracy": (
                replay_matches / replay_total if replay_total else None
            ),
            "candidate_accumulated_p50": _safe_median(all_candidate),
            "candidate_accumulated_p90": _quantile(all_candidate, 0.9),
            "rel_l1_p50": _safe_median(all_rel_l1),
            "rel_l1_p90": _quantile(all_rel_l1, 0.9),
            "rescaled_l1_p50": _safe_median(all_rescaled_l1),
            "rescaled_l1_p90": _quantile(all_rescaled_l1, 0.9),
        },
    }


def summarize_records(
    records: list[dict[str, Any]],
    *,
    trace_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["config_label"]), []).append(record)

    baseline_mean = None
    if "baseline" in grouped:
        baseline_times = [
            float(r["inference_time_s"])
            for r in grouped["baseline"]
            if r.get("status") == "completed" and r.get("inference_time_s") is not None
        ]
        baseline_mean = _safe_mean(baseline_times)

    per_config: list[dict[str, Any]] = []
    for config_label, config_records in grouped.items():
        inference_times = [
            float(r["inference_time_s"])
            for r in config_records
            if r.get("status") == "completed" and r.get("inference_time_s") is not None
        ]
        elapsed_times = [
            float(r["elapsed_s"])
            for r in config_records
            if r.get("status") == "completed" and r.get("elapsed_s") is not None
        ]
        peak_memories = [
            float(r["peak_memory_mb"])
            for r in config_records
            if r.get("status") == "completed" and r.get("peak_memory_mb") is not None
        ]
        task_ids = {str(r["task_id"]) for r in config_records}
        config_summary = {
            "config_label": config_label,
            "config": config_records[0]["config"],
            "num_jobs": len(config_records),
            "num_completed": sum(
                1 for r in config_records if r.get("status") == "completed"
            ),
            "num_failed": sum(
                1 for r in config_records if r.get("status") != "completed"
            ),
            "mean_inference_time_s": _safe_mean(inference_times),
            "median_inference_time_s": _safe_median(inference_times),
            "mean_elapsed_s": _safe_mean(elapsed_times),
            "median_elapsed_s": _safe_median(elapsed_times),
            "mean_peak_memory_mb": _safe_mean(peak_memories),
            "median_peak_memory_mb": _safe_median(peak_memories),
            "task_ids": sorted(task_ids),
        }
        if baseline_mean and config_label != "baseline" and config_summary["mean_inference_time_s"]:
            config_summary["speedup_vs_baseline"] = (
                baseline_mean / config_summary["mean_inference_time_s"]
            )
        if trace_summary and trace_summary.get("available"):
            per_request = trace_summary.get("per_request", {})
            matched = [
                per_request[task_id]
                for task_id in task_ids
                if task_id in per_request
            ]
            if matched:
                skip_ratios = [
                    float(item["skip_ratio"])
                    for item in matched
                    if item.get("skip_ratio") is not None
                ]
                replay = [
                    float(item["replay_accuracy"])
                    for item in matched
                    if item.get("replay_accuracy") is not None
                ]
                config_summary["trace"] = {
                    "matched_jobs": len(matched),
                    "mean_skip_ratio": _safe_mean(skip_ratios),
                    "mean_replay_accuracy": _safe_mean(replay),
                }
        per_config.append(config_summary)

    sortable = [
        item
        for item in per_config
        if item.get("mean_inference_time_s") is not None
    ]
    ranking = sorted(sortable, key=lambda item: float(item["mean_inference_time_s"]))
    return {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "num_records": len(records),
        "trace_summary": trace_summary,
        "per_config": per_config,
        "ranking": ranking,
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Sweep VideoEdit TeaCache parameters against a running SGLang service "
            "and summarize runtime plus optional trace-based proxy statistics."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("~/workspace/case/erase_data_case"),
    )
    parser.add_argument("--captions-path", type=Path, default=Path("captions.jsonl"))
    parser.add_argument("--index-path", type=Path, default=Path("index.jsonl"))
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--summary-path", type=Path)
    parser.add_argument(
        "--teacache-trace-path",
        type=Path,
        default=None,
        help=(
            "Global TeaCache decision trace written by the server. "
            "This requires the service to be started with SGLANG_TEACACHE_TRACE_PATH."
        ),
    )
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument(
        "--reference-mode",
        choices=("none", "first-frame"),
        default="none",
    )
    parser.add_argument("--task-prefix", default="teacache_sweep_")
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-baseline", action="store_true")

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
    parser.add_argument("--server-task-timeout", type=int, default=-1)

    parser.add_argument(
        "--teacache-thresh-values",
        nargs="+",
        type=float,
        default=[0.3],
    )
    parser.add_argument(
        "--teacache-start-skipping-values",
        nargs="+",
        type=_int_or_float,
        default=[5],
    )
    parser.add_argument(
        "--teacache-end-skipping-values",
        nargs="+",
        type=_int_or_float,
        default=[1.0],
    )

    parser.add_argument("--http-timeout", type=float, default=120.0)
    parser.add_argument("--wait-server-timeout", type=float, default=300.0)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--busy-sleep", type=float, default=30.0)
    args = parser.parse_args()

    if args.output_root is None:
        args.output_root = repo_root / "outputs" / "teacache_sweep"
    else:
        args.output_root = args.output_root.expanduser().resolve()
    if args.run_name is None:
        args.run_name = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_root = args.output_root / args.run_name
    args.run_root = run_root
    if args.manifest_path is None:
        args.manifest_path = run_root / "manifest.jsonl"
    else:
        args.manifest_path = args.manifest_path.expanduser().resolve()
    if args.summary_path is None:
        args.summary_path = run_root / "summary.json"
    else:
        args.summary_path = args.summary_path.expanduser().resolve()
    if args.reference_dir is None:
        args.reference_dir = run_root / "reference_frames"
    else:
        args.reference_dir = args.reference_dir.expanduser().resolve()
    if args.teacache_trace_path is not None:
        args.teacache_trace_path = args.teacache_trace_path.expanduser().resolve()
    return args


def main() -> int:
    args = parse_args()
    configs = build_config_grid(args)
    if not configs:
        print("No configs to run.", file=sys.stderr)
        return 1

    args.output_dir = args.run_root / "job_staging"
    jobs = build_jobs(args)
    print(
        f"[start] jobs={len(jobs)} configs={len(configs)} "
        f"run_root={args.run_root}"
    )
    if not jobs:
        return 1

    if not args.dry_run:
        args.run_root.mkdir(parents=True, exist_ok=True)
        args.reference_dir.mkdir(parents=True, exist_ok=True)

    client = HttpClient(args.base_url, timeout=args.http_timeout)
    if not args.dry_run:
        wait_for_server(client, args.wait_server_timeout)

    records: list[dict[str, Any]] = []
    failures = 0
    for config_index, config in enumerate(configs, start=1):
        print(f"[config] {config_index}/{len(configs)} {config.label}")
        for job_index, job in enumerate(jobs, start=1):
            print(f"[job] {job_index}/{len(jobs)} id={job.video_id}")
            try:
                record = run_job(
                    args,
                    client,
                    job,
                    config=config,
                    config_index=config_index,
                    job_index=job_index,
                    run_root=args.run_root,
                )
            except Exception as exc:
                failures += 1
                task_id = (
                    f"{args.task_prefix}{args.run_name}_cfg{config_index:02d}_"
                    f"job{job_index:04d}_{config.label}_{job.video_id}"
                )
                record = {
                    "id": job.video_id,
                    "task_id": task_id,
                    "config": asdict(config),
                    "config_label": config.label,
                    "status": "failed",
                    "reason": str(exc),
                }
                print(f"[failed] {config.label} {job.video_id}: {exc}", file=sys.stderr)
            records.append(record)
            if not args.dry_run:
                write_jsonl(args.manifest_path, record)

    trace_summary = None
    if not args.dry_run and args.teacache_trace_path is not None:
        successful_task_ids = {
            str(record["task_id"])
            for record in records
            if record.get("status") == "completed"
        }
        trace_summary = analyze_teacache_trace(
            args.teacache_trace_path,
            successful_task_ids,
        )

    summary = summarize_records(records, trace_summary=trace_summary)
    if not args.dry_run:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    ranking = summary.get("ranking", [])
    if ranking:
        print("[ranking]")
        for item in ranking[:10]:
            parts = [
                item["config_label"],
                f"mean_inference_time_s={item['mean_inference_time_s']:.3f}",
            ]
            if item.get("speedup_vs_baseline") is not None:
                parts.append(f"speedup_vs_baseline={item['speedup_vs_baseline']:.3f}")
            trace = item.get("trace")
            if trace and trace.get("mean_skip_ratio") is not None:
                parts.append(f"skip_ratio={trace['mean_skip_ratio']:.3f}")
            if trace and trace.get("mean_replay_accuracy") is not None:
                parts.append(
                    f"proxy_replay_accuracy={trace['mean_replay_accuracy']:.3f}"
                )
            print("  " + " ".join(parts))

    if not args.dry_run:
        print(
            f"[done] records={len(records)} failures={failures} "
            f"manifest={args.manifest_path} summary={args.summary_path}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
