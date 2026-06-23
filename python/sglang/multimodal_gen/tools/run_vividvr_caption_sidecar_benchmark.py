# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from sglang.multimodal_gen.runtime.vividvr.caption_bridge import (
    VividVRCaptionBridgeConfig,
    request_vividvr_caption_sidecar,
    validate_caption_sidecar_file,
)
from sglang.multimodal_gen.runtime.vividvr.caption_manifest import (
    build_vividvr_caption_manifest_for_video_path,
)


@dataclass(frozen=True)
class CaptionSidecarBenchmarkResult:
    video_path: str
    manifest_path: str
    output_caption_path: str
    baseline_caption_path: str
    expected_caption_count: int
    generated_caption_count: int
    baseline_caption_count: int
    captions_match: bool
    first_mismatch_index: int | None
    elapsed_seconds: float
    sidecar_mode: str | None
    sidecar_worker_count: int | None
    sidecar_fallback_used: bool | None
    sidecar_request_id: str | None
    sidecar_total_clip_count: int | None
    sidecar_assigned_clip_indices_by_worker: dict[str, list[int]] | None
    sidecar_timing: dict[str, object] | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _first_mismatch_index(
    baseline_captions: list[str],
    generated_captions: list[str],
) -> int | None:
    for index, (baseline_caption, generated_caption) in enumerate(
        zip(baseline_captions, generated_captions)
    ):
        if baseline_caption != generated_caption:
            return index
    if len(baseline_captions) != len(generated_captions):
        return min(len(baseline_captions), len(generated_captions))
    return None


def _read_utf8_text(path: str) -> str:
    return Path(path).expanduser().read_text(encoding="utf-8")


def _validate_parallel_assignments(
    result: CaptionSidecarBenchmarkResult,
) -> list[str]:
    failure_reasons: list[str] = []
    if result.sidecar_worker_count != 2:
        failure_reasons.append(
            "caption sidecar response did not report the expected dual-worker configuration"
        )
        return failure_reasons
    if result.sidecar_total_clip_count != result.expected_caption_count:
        failure_reasons.append(
            "caption sidecar response did not report the expected total clip count"
        )
        return failure_reasons

    assignments = result.sidecar_assigned_clip_indices_by_worker
    if not assignments:
        failure_reasons.append(
            "caption sidecar response did not report worker clip assignments"
        )
        return failure_reasons

    covered_clip_indices: list[int] = []
    non_empty_workers = 0
    for worker_index in range(result.sidecar_worker_count):
        clip_indices = assignments.get(str(worker_index))
        if clip_indices is None:
            failure_reasons.append(
                "caption sidecar response omitted one or more worker assignment lists"
            )
            return failure_reasons
        if clip_indices:
            non_empty_workers += 1
        covered_clip_indices.extend(clip_indices)

    expected_clip_indices = list(range(result.expected_caption_count))
    if sorted(covered_clip_indices) != expected_clip_indices:
        failure_reasons.append(
            "caption sidecar worker assignments did not cover every clip exactly once"
        )
    if len(set(covered_clip_indices)) != len(covered_clip_indices):
        failure_reasons.append(
            "caption sidecar worker assignments reported duplicate clip indices"
        )
    if result.expected_caption_count >= 2 and non_empty_workers != result.sidecar_worker_count:
        failure_reasons.append(
            "caption sidecar benchmark did not exercise both workers on the dual-clip request"
        )
    return failure_reasons


def benchmark_caption_sidecar(
    *,
    video_path: str,
    baseline_caption_path: str,
    sidecar_base_url: str,
    sidecar_timeout_s: float,
    num_temporal_process_frames: int,
    tile_size: int,
    tile_stride: int,
    manifest_path: str,
    output_caption_path: str,
    metrics_json_path: str | None = None,
) -> CaptionSidecarBenchmarkResult:
    manifest = build_vividvr_caption_manifest_for_video_path(
        video_path=video_path,
        num_temporal_process_frames=num_temporal_process_frames,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )
    manifest.write_json(manifest_path)

    start_time = time.perf_counter()
    sidecar_result = asyncio.run(
        request_vividvr_caption_sidecar(
            config=VividVRCaptionBridgeConfig(
                enabled=True,
                base_url=sidecar_base_url,
                timeout_s=sidecar_timeout_s,
            ),
            manifest_path=manifest_path,
            output_caption_path=output_caption_path,
            expected_caption_count=manifest.expected_caption_count,
        )
    )
    elapsed_seconds = time.perf_counter() - start_time

    generated_captions = validate_caption_sidecar_file(
        sidecar_result.caption_file_path,
        expected_count=manifest.expected_caption_count,
    )
    baseline_captions = validate_caption_sidecar_file(
        baseline_caption_path,
        expected_count=manifest.expected_caption_count,
    )
    generated_raw_text = _read_utf8_text(sidecar_result.caption_file_path)
    baseline_raw_text = _read_utf8_text(baseline_caption_path)
    mismatch_index = _first_mismatch_index(baseline_captions, generated_captions)
    result = CaptionSidecarBenchmarkResult(
        video_path=video_path,
        manifest_path=manifest_path,
        output_caption_path=sidecar_result.caption_file_path,
        baseline_caption_path=baseline_caption_path,
        expected_caption_count=manifest.expected_caption_count,
        generated_caption_count=len(generated_captions),
        baseline_caption_count=len(baseline_captions),
        captions_match=generated_raw_text == baseline_raw_text,
        first_mismatch_index=mismatch_index,
        elapsed_seconds=elapsed_seconds,
        sidecar_mode=getattr(sidecar_result, "mode", None),
        sidecar_worker_count=getattr(sidecar_result, "worker_count", None),
        sidecar_fallback_used=getattr(sidecar_result, "fallback_used", None),
        sidecar_request_id=getattr(sidecar_result, "request_id", None),
        sidecar_total_clip_count=getattr(sidecar_result, "total_clip_count", None),
        sidecar_assigned_clip_indices_by_worker=getattr(
            sidecar_result,
            "assigned_clip_indices_by_worker",
            None,
        ),
        sidecar_timing=getattr(sidecar_result, "timing", None),
    )
    if metrics_json_path:
        metrics_path = Path(metrics_json_path).expanduser()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a standalone VividVR caption sidecar benchmark."
    )
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--baseline-caption-path", required=True)
    parser.add_argument("--sidecar-base-url", default="http://127.0.0.1:31200")
    parser.add_argument("--sidecar-timeout-s", type=float, default=1800.0)
    parser.add_argument("--num-temporal-process-frames", type=int, default=121)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--tile-stride", type=int, default=64)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("Vivid_Acceptance/caption_sidecar_benchmark"),
    )
    parser.add_argument("--manifest-path")
    parser.add_argument("--output-caption-path")
    parser.add_argument("--metrics-json-path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    work_dir = args.work_dir.expanduser()
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_path or str(work_dir / "manifest.json")
    output_caption_path = args.output_caption_path or str(work_dir / "captions.txt")

    result = benchmark_caption_sidecar(
        video_path=args.video_path,
        baseline_caption_path=args.baseline_caption_path,
        sidecar_base_url=args.sidecar_base_url,
        sidecar_timeout_s=args.sidecar_timeout_s,
        num_temporal_process_frames=args.num_temporal_process_frames,
        tile_size=args.tile_size,
        tile_stride=args.tile_stride,
        manifest_path=manifest_path,
        output_caption_path=output_caption_path,
        metrics_json_path=args.metrics_json_path,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    failure_reasons: list[str] = []
    if not result.captions_match:
        failure_reasons.append(
            "caption sidecar output does not exactly match the baseline caption file"
        )
    if result.sidecar_mode != "parallel":
        failure_reasons.append(
            "caption sidecar request did not remain on the parallel path"
        )
    if result.sidecar_fallback_used is not False:
        failure_reasons.append(
            "caption sidecar request fell back to serial mode instead of staying on the parallel path"
        )
    failure_reasons.extend(_validate_parallel_assignments(result))
    if failure_reasons:
        raise SystemExit("; ".join(failure_reasons))


if __name__ == "__main__":
    main()
