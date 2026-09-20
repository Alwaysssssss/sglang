# SPDX-License-Identifier: Apache-2.0
"""Frame-level comparison of pre-encode uint8 dumps.

This is the algorithm-layer gate from ``requirements.md`` §4.2.1: it compares
the frames *before* the H.264 encoder touches them, so the difference it
reports is the algorithm's, with no codec noise on top.

The per-frame metrics deliberately reuse ``videoedit.compare``'s SSIM
implementation so the two layers agree on what "SSIM" means; only the summary
bookkeeping is local. Note one difference from ``compare_videos``: frames here
are already RGB (``to_uint8_hwc`` emits ``[T, H, W, C]`` straight from the
pipeline's ``[1, C, T, H, W]``), so no BGR->RGB conversion happens.

A shape mismatch is an error rather than a silent ``cv2.resize``: structural
consistency is a separate, zero-tolerance gate (``requirements.md`` §4.3).

Usage:
    python -m sglang.multimodal_gen.runtime.vsr.verify.compare_frames \\
        --reference-dir DUMP_A --candidate-dir DUMP_B [--report-json R.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from sglang.multimodal_gen.runtime.videoedit.compare import FrameMetrics, _ssim
from sglang.multimodal_gen.runtime.vsr.verify.dumps import load_frames


def compare_frames(
    ref: torch.Tensor,
    cand: torch.Tensor,
    *,
    min_ssim: float = 0.97,
    max_mse: float = 25.0,
    max_mae: float = 2.5,
    max_failed_frame_ratio: float = 0.0,
) -> dict:
    if ref.shape != cand.shape:
        raise ValueError(
            f"frame shape mismatch: reference={tuple(ref.shape)} candidate={tuple(cand.shape)}"
        )
    if ref.dtype != torch.uint8 or cand.dtype != torch.uint8:
        raise ValueError(f"expected uint8 frames, got {ref.dtype} and {cand.dtype}")

    ref_np = ref.numpy()
    cand_np = cand.numpy()

    reports: List[FrameMetrics] = []
    for idx in range(ref_np.shape[0]):
        a, b = ref_np[idx], cand_np[idx]
        diff = a.astype(np.float32) - b.astype(np.float32)
        mse = float(np.mean(diff * diff))
        mae = float(np.mean(np.abs(diff)))
        max_abs = int(np.max(np.abs(diff)))
        psnr = float("inf") if mse == 0 else float(20 * math.log10(255.0 / math.sqrt(mse)))
        ssim = _ssim(a, b)
        reports.append(FrameMetrics(
            idx, ssim, mse, mae, psnr, max_abs,
            ssim >= min_ssim and mse <= max_mse and mae <= max_mae,
        ))

    failed = [m.index for m in reports if not m.pass_frame]
    finite_psnr = [m.psnr for m in reports if math.isfinite(m.psnr)]
    summary = {
        "compared_frames": len(reports),
        "ssim_mean": float(np.mean([m.ssim for m in reports])),
        "ssim_min": float(np.min([m.ssim for m in reports])),
        "mse_mean": float(np.mean([m.mse for m in reports])),
        "mse_max": float(np.max([m.mse for m in reports])),
        "mae_mean": float(np.mean([m.mae for m in reports])),
        "mae_max": float(np.max([m.mae for m in reports])),
        "psnr_mean": float(np.mean(finite_psnr)) if finite_psnr else float("inf"),
        "max_abs_diff": int(np.max([m.max_abs_diff for m in reports])),
        "failed_frames": failed,
        "pass_compare": (len(failed) / len(reports)) <= max_failed_frame_ratio,
        "thresholds": {
            "min_ssim": min_ssim,
            "max_mse": max_mse,
            "max_mae": max_mae,
            "max_failed_frame_ratio": max_failed_frame_ratio,
        },
    }
    return {"summary": summary, "frames": [asdict(m) for m in reports]}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare pre-encode frame dumps.")
    parser.add_argument("--reference-dir", required=True, help="Dump dir of the reference run")
    parser.add_argument("--candidate-dir", required=True, help="Dump dir of the candidate run")
    parser.add_argument("--report-json")
    parser.add_argument("--min-ssim", type=float, default=0.97)
    parser.add_argument("--max-mse", type=float, default=25.0)
    parser.add_argument("--max-mae", type=float, default=2.5)
    parser.add_argument("--max-failed-frame-ratio", type=float, default=0.0)
    args = parser.parse_args(argv)

    report = compare_frames(
        load_frames(Path(args.reference_dir)),
        load_frames(Path(args.candidate_dir)),
        min_ssim=args.min_ssim,
        max_mse=args.max_mse,
        max_mae=args.max_mae,
        max_failed_frame_ratio=args.max_failed_frame_ratio,
    )
    if args.report_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_json)), exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["pass_compare"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
