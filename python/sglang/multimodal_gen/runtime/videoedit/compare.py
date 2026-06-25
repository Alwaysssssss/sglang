# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from sglang.multimodal_gen.runtime.videoedit.frame_cache import cache_video_frames


@dataclass
class FrameMetrics:
    index: int
    ssim: float
    mse: float
    mae: float
    psnr: float
    max_abs_diff: int
    pass_frame: bool


def _read_video(path: str) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frames in video: {path}")
    cache_video_frames(path, frames, fps)
    return frames, float(fps)


def _ssim(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
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


def compare_videos(
    reference: str,
    candidate: str,
    *,
    min_ssim: float = 0.90,
    max_mse: float = 150.0,
    max_mae: float = 8.0,
    allow_frame_count_delta: int = 1,
    max_failed_frame_ratio: float = 0.05,
    drop_reference_first_frame: bool = False,
    drop_candidate_first_frame: bool = False,
) -> dict:
    ref_frames, _ = _read_video(reference)
    cand_frames, _ = _read_video(candidate)
    reference_frame_count = len(ref_frames)
    candidate_frame_count = len(cand_frames)
    if drop_reference_first_frame:
        ref_frames = ref_frames[1:]
    if drop_candidate_first_frame:
        cand_frames = cand_frames[1:]
    frame_delta = abs(len(ref_frames) - len(cand_frames))
    if frame_delta > allow_frame_count_delta:
        raise ValueError(
            f"Frame count mismatch: reference={len(ref_frames)}, "
            f"candidate={len(cand_frames)}, allowed_delta={allow_frame_count_delta}"
        )
    compared = min(len(ref_frames), len(cand_frames))
    if compared <= 0:
        raise ValueError("No comparable frames")

    frame_reports: list[FrameMetrics] = []
    for idx in range(compared):
        ref = ref_frames[idx]
        cand = cand_frames[idx]
        if ref.shape != cand.shape:
            cand = cv2.resize(cand, (ref.shape[1], ref.shape[0]))
        diff = ref.astype(np.float32) - cand.astype(np.float32)
        mse = float(np.mean(diff * diff))
        mae = float(np.mean(np.abs(diff)))
        max_abs = int(np.max(np.abs(diff)))
        psnr = float("inf") if mse == 0 else float(20 * math.log10(255.0 / math.sqrt(mse)))
        ssim = _ssim(ref, cand)
        pass_frame = ssim >= min_ssim and mse <= max_mse and mae <= max_mae
        frame_reports.append(FrameMetrics(idx, ssim, mse, mae, psnr, max_abs, pass_frame))

    failed = [m.index for m in frame_reports if not m.pass_frame]
    failed_ratio = len(failed) / compared
    summary = {
        "compared_frames": compared,
        "ssim_mean": float(np.mean([m.ssim for m in frame_reports])),
        "ssim_min": float(np.min([m.ssim for m in frame_reports])),
        "mse_mean": float(np.mean([m.mse for m in frame_reports])),
        "mse_max": float(np.max([m.mse for m in frame_reports])),
        "mae_mean": float(np.mean([m.mae for m in frame_reports])),
        "mae_max": float(np.max([m.mae for m in frame_reports])),
        "psnr_mean": float(np.mean([m.psnr for m in frame_reports if math.isfinite(m.psnr)]))
        if any(math.isfinite(m.psnr) for m in frame_reports)
        else float("inf"),
        "max_abs_diff": int(np.max([m.max_abs_diff for m in frame_reports])),
        "reference_frame_count": reference_frame_count,
        "candidate_frame_count": candidate_frame_count,
        "frame_count_delta": abs(reference_frame_count - candidate_frame_count),
        "failed_frames": failed,
        "pass_compare": failed_ratio <= max_failed_frame_ratio,
        "thresholds": {
            "min_ssim": min_ssim,
            "max_mse": max_mse,
            "max_mae": max_mae,
            "allow_frame_count_delta": allow_frame_count_delta,
            "max_failed_frame_ratio": max_failed_frame_ratio,
        },
    }
    return {"summary": summary, "frames": [asdict(m) for m in frame_reports]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two videos frame-by-frame.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--report-json")
    parser.add_argument("--min-ssim", type=float, default=0.90)
    parser.add_argument("--max-mse", type=float, default=150.0)
    parser.add_argument("--max-mae", type=float, default=8.0)
    parser.add_argument("--allow-frame-count-delta", type=int, default=1)
    parser.add_argument("--max-failed-frame-ratio", type=float, default=0.05)
    parser.add_argument("--drop-reference-first-frame", action="store_true")
    parser.add_argument("--drop-candidate-first-frame", action="store_true")
    args = parser.parse_args()
    report = compare_videos(
        args.reference,
        args.candidate,
        min_ssim=args.min_ssim,
        max_mse=args.max_mse,
        max_mae=args.max_mae,
        allow_frame_count_delta=args.allow_frame_count_delta,
        max_failed_frame_ratio=args.max_failed_frame_ratio,
        drop_reference_first_frame=args.drop_reference_first_frame,
        drop_candidate_first_frame=args.drop_candidate_first_frame,
    )
    if args.report_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_json)), exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["pass_compare"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
