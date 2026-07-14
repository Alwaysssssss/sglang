#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRIC_FIELDS = (
    "has_previous",
    "relative_l1",
    "relative_l2",
    "relative_norm",
    "cosine_distance",
    "mean_abs_delta",
    "rmse",
)

METRIC_GROUPS = {
    "latent_model_input": "latent_model_input_change",
    "latents_before_scheduler": "latents_before_scheduler_change",
    "noise_pred_cond": "noise_pred_cond_change",
    "noise_pred_uncond": "noise_pred_uncond_change",
    "noise_pred_guided": "noise_pred_guided_change",
    "latents_after_scheduler": "latents_after_scheduler_change",
}


def iter_trace_paths(paths: list[Path]) -> list[Path]:
    trace_paths: list[Path] = []
    for path in paths:
        path = path.expanduser().resolve()
        if path.is_dir():
            trace_paths.extend(sorted(path.glob("*.jsonl")))
        elif path.exists():
            trace_paths.append(path)
    return trace_paths


def flatten_record(trace_path: Path, record: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "trace_file": str(trace_path),
        "request_id": record.get("request_id"),
        "window_index": record.get("window_index"),
        "window_start_index": record.get("window_start_index"),
        "window_end_index": record.get("window_end_index"),
        "step": record.get("step"),
        "num_inference_steps": record.get("num_inference_steps"),
        "timestep": record.get("timestep"),
        "do_cfg": record.get("do_cfg"),
        "guidance_scale": record.get("guidance_scale"),
    }
    for prefix, key in METRIC_GROUPS.items():
        metrics = record.get(key)
        if not isinstance(metrics, dict):
            metrics = {}
        for field in METRIC_FIELDS:
            row[f"{prefix}_{field}"] = metrics.get(field)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Flatten VideoEdit denoise trace JSONL files into a CSV."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Trace JSONL file(s) or directories containing *.jsonl traces.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/erase_data_case_repair/denoise_summary.csv"),
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for trace_path in iter_trace_paths(args.paths):
        with trace_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("event") != "videoedit_denoise_step":
                    continue
                rows.append(flatten_record(trace_path, record))

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(flatten_record(Path("trace.jsonl"), {}).keys())
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
