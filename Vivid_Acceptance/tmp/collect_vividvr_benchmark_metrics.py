from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


INDICATOR_DIR = Path("/home/zhiheng/sglang/Vivid_Acceptance/indicator")
TASK_PREFIX = "vividvr-service-benchmark-long-130f-20step"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect the latest VividVR service benchmark metrics by label."
    )
    parser.add_argument("labels", nargs="+")
    parser.add_argument("--indicator-dir", type=Path, default=INDICATOR_DIR)
    parser.add_argument("--as-json", action="store_true")
    return parser.parse_args()


def find_latest_report(indicator_dir: Path, label: str) -> Path:
    pattern = str(indicator_dir / f"{TASK_PREFIX}-{label}-*.json")
    candidates = [
        Path(path)
        for path in glob.glob(pattern)
        if not path.endswith("_perf.json") and not path.endswith("_framewise_ssim.json")
    ]
    if not candidates:
        raise FileNotFoundError(f"No report found for label={label!r}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def get_stage_seconds(perf_dump: dict[str, Any] | None, stage_name: str) -> float | None:
    if not perf_dump:
        return None
    for entry in perf_dump.get("steps", []):
        if entry.get("name") == stage_name:
            duration_ms = entry.get("duration_ms")
            return None if duration_ms is None else float(duration_ms) / 1000.0
    return None


def build_record(report_path: Path) -> dict[str, Any]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    perf_dump = payload.get("perf_dump")
    mem = None
    if perf_dump:
        mem = (
            perf_dump.get("memory_checkpoints", {})
            .get("after_forward", {})
            .get("peak_allocated_mb")
        )

    return {
        "label": payload["benchmark_label"],
        "report_path": str(report_path),
        "task_id": payload["task_id"],
        "total_runtime_seconds": payload.get("total_runtime_seconds"),
        "model_inference_runtime_seconds": payload.get("model_inference_runtime_seconds"),
        "warmup_inference_time_s": (
            payload.get("warmup_detail_response", {}) or {}
        ).get("inference_time_s"),
        "peak_allocated_mb_after_forward": mem,
        "ssim_mean": payload.get("ssim_mean"),
        "ssim_min": payload.get("ssim_min"),
        "mse_mean": payload.get("mse_mean"),
        "mae_mean": payload.get("mae_mean"),
        "psnr_mean": payload.get("psnr_mean"),
        "max_abs_diff": payload.get("max_abs_diff"),
        "pass_compare": payload.get("pass_compare"),
        "prep_seconds": get_stage_seconds(
            perf_dump, "VividVRLongClipPreparationStage"
        ),
        "denoise_seconds": get_stage_seconds(
            perf_dump, "VividVRMultiClipDenoisingStage"
        ),
        "decode_trim_seconds": get_stage_seconds(
            perf_dump, "VividVRMultiClipDecodeTrimStage"
        ),
        "postprocess_seconds": get_stage_seconds(
            perf_dump, "VividVRTemporalStitchPostprocessStage"
        ),
    }


def main() -> int:
    args = parse_args()
    records = [
        build_record(find_latest_report(args.indicator_dir, label)) for label in args.labels
    ]
    if args.as_json:
        print(json.dumps(records, indent=2))
        return 0

    headers = [
        "label",
        "total_runtime_seconds",
        "model_inference_runtime_seconds",
        "warmup_inference_time_s",
        "peak_allocated_mb_after_forward",
        "ssim_mean",
        "ssim_min",
        "mse_mean",
        "mae_mean",
        "psnr_mean",
        "max_abs_diff",
        "pass_compare",
        "prep_seconds",
        "denoise_seconds",
        "decode_trim_seconds",
        "postprocess_seconds",
    ]
    print("\t".join(headers))
    for record in records:
        print("\t".join(str(record.get(header)) for header in headers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
