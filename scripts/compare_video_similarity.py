#!/usr/bin/env python3
"""Compare two videos frame by frame, with optional SGLang candidate discovery."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DEFAULT_SGLANG_DIRS = (
    "/home/tyx/workspace/zhouhao6/sglang/output_tyx",
    "/home/tyx/workspace/zhouhao6/sglang/outputs",
)


@dataclass
class VideoMeta:
    path: str
    mtime: str
    mtime_ts: float
    width: int
    height: int
    fps: float
    frames: int
    duration_sec: float
    size_bytes: int


@dataclass
class FrameMetric:
    index: int
    ssim: float
    mse: float
    mae: float
    psnr: float
    max_abs_diff: int


def parse_time(value: str | None) -> float | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    return datetime.fromisoformat(text).timestamp()


def iso_from_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def get_video_meta(path: str | Path) -> VideoMeta:
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    stat = path.stat()
    duration = float(frames / fps) if fps > 0 else 0.0
    return VideoMeta(
        path=str(path),
        mtime=iso_from_ts(stat.st_mtime),
        mtime_ts=stat.st_mtime,
        width=width,
        height=height,
        fps=fps,
        frames=frames,
        duration_sec=duration,
        size_bytes=stat.st_size,
    )


def iter_mp4s(paths: Iterable[str]) -> Iterable[Path]:
    for raw in paths:
        root = Path(raw).expanduser()
        if root.is_file() and root.suffix.lower() == ".mp4":
            yield root
        elif root.is_dir():
            yield from root.rglob("*.mp4")


def find_candidate(
    candidate_dirs: list[str],
    *,
    around_ts: float | None,
    tolerance_hours: float | None,
    expected_frames: int | None,
    name_regex: str | None,
) -> tuple[VideoMeta, list[VideoMeta]]:
    pattern = re.compile(name_regex) if name_regex else None
    candidates: list[VideoMeta] = []
    for path in iter_mp4s(candidate_dirs):
        if pattern and not pattern.search(str(path)):
            continue
        try:
            meta = get_video_meta(path)
        except Exception:
            continue
        if around_ts is not None and tolerance_hours is not None:
            max_delta = tolerance_hours * 3600.0
            if abs(meta.mtime_ts - around_ts) > max_delta:
                continue
        candidates.append(meta)

    if not candidates:
        raise FileNotFoundError("No candidate mp4 matched the discovery filters")

    def score(meta: VideoMeta) -> tuple[float, int, float, str]:
        time_delta = abs(meta.mtime_ts - around_ts) if around_ts is not None else 0.0
        frame_delta = (
            abs(meta.frames - expected_frames) if expected_frames is not None else 0
        )
        name_bonus = -1.0 if pattern and pattern.search(meta.path) else 0.0
        return (float(frame_delta), time_delta + name_bonus, -meta.mtime_ts, meta.path)

    candidates.sort(key=score)
    return candidates[0], candidates


def read_rgb(cap: cv2.VideoCapture) -> np.ndarray | None:
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def align_frames(
    ref: np.ndarray, cand: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    if ref.shape == cand.shape:
        return ref, cand
    if mode == "resize":
        resized = cv2.resize(cand, (ref.shape[1], ref.shape[0]))
        return ref, resized
    if mode == "crop":
        height = min(ref.shape[0], cand.shape[0])
        width = min(ref.shape[1], cand.shape[1])
        return ref[:height, :width], cand[:height, :width]
    raise ValueError(f"Unsupported align mode: {mode}")


def ssim_gray(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    b = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    sigma_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    sigma_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    score = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
        (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2)
    )
    return float(score.mean())


def compare_frames(ref: np.ndarray, cand: np.ndarray) -> tuple[float, float, float, int]:
    diff = ref.astype(np.float32) - cand.astype(np.float32)
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    max_abs = int(np.max(np.abs(diff)))
    psnr = float("inf") if mse == 0 else float(20 * math.log10(255.0 / math.sqrt(mse)))
    return mse, mae, psnr, max_abs


def save_diff_image(
    path: Path,
    ref: np.ndarray,
    cand: np.ndarray,
    *,
    scale: float = 4.0,
) -> None:
    diff = np.abs(ref.astype(np.float32) - cand.astype(np.float32))
    diff = np.clip(diff * scale, 0, 255).astype(np.uint8)
    separator = np.full((ref.shape[0], 8, 3), 255, dtype=np.uint8)
    canvas = np.concatenate([ref, separator, cand, separator, diff], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def compare_videos(
    reference: str,
    candidate: str,
    *,
    align_mode: str,
    max_frames: int | None,
    diff_dir: str | None,
    save_worst: int,
) -> dict:
    ref_meta = get_video_meta(reference)
    cand_meta = get_video_meta(candidate)
    ref_cap = cv2.VideoCapture(reference)
    cand_cap = cv2.VideoCapture(candidate)

    frames: list[FrameMetric] = []
    saved_frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    index = 0
    while True:
        if max_frames is not None and index >= max_frames:
            break
        ref = read_rgb(ref_cap)
        cand = read_rgb(cand_cap)
        if ref is None or cand is None:
            break
        ref, cand = align_frames(ref, cand, align_mode)
        mse, mae, psnr, max_abs = compare_frames(ref, cand)
        metric = FrameMetric(index, ssim_gray(ref, cand), mse, mae, psnr, max_abs)
        frames.append(metric)
        if diff_dir and save_worst > 0:
            saved_frames[index] = (ref.copy(), cand.copy())
        index += 1

    ref_cap.release()
    cand_cap.release()

    if not frames:
        raise ValueError("No comparable frames")

    ssim_values = [m.ssim for m in frames]
    mse_values = [m.mse for m in frames]
    mae_values = [m.mae for m in frames]
    finite_psnr = [m.psnr for m in frames if math.isfinite(m.psnr)]
    worst = sorted(frames, key=lambda m: (m.ssim, -m.mae))[:save_worst]

    if diff_dir:
        root = Path(diff_dir)
        for metric in worst:
            ref, cand = saved_frames[metric.index]
            save_diff_image(root / f"frame_{metric.index:04d}_ssim_{metric.ssim:.4f}.jpg", ref, cand)

    summary = {
        "reference": asdict(ref_meta),
        "candidate": asdict(cand_meta),
        "align_mode": align_mode,
        "compared_frames": len(frames),
        "reference_uncompared_tail_frames": max(ref_meta.frames - len(frames), 0),
        "candidate_uncompared_tail_frames": max(cand_meta.frames - len(frames), 0),
        "ssim_mean": float(np.mean(ssim_values)),
        "ssim_min": float(np.min(ssim_values)),
        "ssim_p05": float(np.percentile(ssim_values, 5)),
        "mse_mean": float(np.mean(mse_values)),
        "mse_max": float(np.max(mse_values)),
        "mae_mean": float(np.mean(mae_values)),
        "mae_max": float(np.max(mae_values)),
        "psnr_mean": float(np.mean(finite_psnr)) if finite_psnr else float("inf"),
        "max_abs_diff": int(np.max([m.max_abs_diff for m in frames])),
        "worst_frames": [asdict(m) for m in worst],
    }
    return {"summary": summary, "frames": [asdict(m) for m in frames]}


def write_frame_csv(report: dict, path: str) -> None:
    rows = report["frames"]
    if not rows:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Reference video path")
    parser.add_argument("--candidate", help="Candidate video path")
    parser.add_argument(
        "--candidate-dir",
        action="append",
        default=[],
        help="Directory/file to search when --candidate is omitted. Can repeat.",
    )
    parser.add_argument(
        "--around",
        help="UTC mtime target for auto discovery, e.g. '2026-05-14 08:50' or ISO.",
    )
    parser.add_argument("--tolerance-hours", type=float, default=None)
    parser.add_argument("--expected-frames", type=int, default=None)
    parser.add_argument("--name-regex", default=None)
    parser.add_argument(
        "--align-mode",
        choices=("crop", "resize"),
        default="crop",
        help="How to handle size mismatch. crop is usually better for 1088 vs 1080 padding.",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--report-json")
    parser.add_argument("--frame-csv")
    parser.add_argument("--diff-dir")
    parser.add_argument("--save-worst", type=int, default=8)
    parser.add_argument(
        "--print-candidates",
        action="store_true",
        help="Print discovery candidates before comparison.",
    )
    args = parser.parse_args()

    candidate = args.candidate
    candidates_report = None
    if candidate is None:
        search_roots = args.candidate_dir or list(DEFAULT_SGLANG_DIRS)
        selected, candidates = find_candidate(
            search_roots,
            around_ts=parse_time(args.around),
            tolerance_hours=args.tolerance_hours,
            expected_frames=args.expected_frames,
            name_regex=args.name_regex,
        )
        candidate = selected.path
        candidates_report = [asdict(c) for c in candidates[:20]]
        print(f"Selected candidate: {candidate}")
        if args.print_candidates:
            print(json.dumps(candidates_report, indent=2))

    report = compare_videos(
        args.reference,
        candidate,
        align_mode=args.align_mode,
        max_frames=args.max_frames,
        diff_dir=args.diff_dir,
        save_worst=args.save_worst,
    )
    if candidates_report is not None:
        report["discovery_candidates"] = candidates_report

    if args.report_json:
        output = Path(args.report_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.frame_csv:
        write_frame_csv(report, args.frame_csv)

    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
