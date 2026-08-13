#!/usr/bin/env python3
"""Compare two aligned VideoEdit outputs with full-frame and masked metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-video", type=Path, required=True)
    parser.add_argument("--candidate-video", type=Path, required=True)
    parser.add_argument("--mask-video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="candidate_vs_reference")
    parser.add_argument(
        "--boundary-px",
        type=int,
        default=16,
        help="Width used to evaluate a band around the mask boundary.",
    )
    return parser.parse_args()


def open_video(path: Path) -> cv2.VideoCapture:
    if not path.is_file():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return capture


def video_metadata(capture: cv2.VideoCapture) -> dict[str, Any]:
    return {
        "reported_frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }


def summarize(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)])
    if finite.size == 0:
        return {
            "count": len(values),
            "finite_count": 0,
            "mean": None,
            "median": None,
            "p05": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p05": float(np.percentile(finite, 5)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def weighted_errors(
    reference: np.ndarray, candidate: np.ndarray, weights: np.ndarray
) -> tuple[float, float, float]:
    weight_sum = float(np.sum(weights))
    if weight_sum == 0:
        return math.nan, math.nan, math.nan
    difference = candidate - reference
    mse = float(np.sum(np.mean(difference**2, axis=2) * weights) / weight_sum)
    mae = float(np.sum(np.mean(np.abs(difference), axis=2) * weights) / weight_sum)
    rmse = math.sqrt(mse)
    psnr = math.inf if mse == 0 else 10.0 * math.log10(1.0 / mse)
    return psnr, mae, rmse


def weighted_ssim(ssim_map: np.ndarray, weights: np.ndarray) -> float:
    if ssim_map.ndim == 3:
        ssim_map = np.mean(ssim_map, axis=2)
    weight_sum = float(np.sum(weights))
    if weight_sum == 0:
        return math.nan
    return float(np.sum(ssim_map * weights) / weight_sum)


def boundary_weights(mask: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        return np.zeros_like(mask)
    kernel_size = width * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    binary = (mask > 0.5).astype(np.uint8)
    dilated = cv2.dilate(binary, kernel)
    eroded = cv2.erode(binary, kernel)
    return (dilated != eroded).astype(np.float32)


def main() -> int:
    args = parse_args()
    if args.boundary_px < 0:
        raise ValueError("--boundary-px must be non-negative")

    reference_capture = open_video(args.reference_video)
    candidate_capture = open_video(args.candidate_video)
    mask_capture = open_video(args.mask_video) if args.mask_video else None

    reference_metadata = video_metadata(reference_capture)
    candidate_metadata = video_metadata(candidate_capture)
    if (
        reference_metadata["width"],
        reference_metadata["height"],
    ) != (candidate_metadata["width"], candidate_metadata["height"]):
        raise ValueError("Reference and candidate resolutions do not match")

    region_names = ["full_frame"]
    if mask_capture:
        region_names.extend(["mask", "background", "boundary"])
    frame_metrics: dict[str, dict[str, list[float]]] = {
        region: {metric: [] for metric in ("psnr_db", "ssim", "mae", "rmse")}
        for region in region_names
    }
    temporal_delta_mae: list[float] = []
    mask_fraction: list[float] = []
    previous_reference: np.ndarray | None = None
    previous_candidate: np.ndarray | None = None
    decoded_frames = 0

    while True:
        reference_ok, reference_frame = reference_capture.read()
        candidate_ok, candidate_frame = candidate_capture.read()
        if reference_ok != candidate_ok:
            raise RuntimeError("Reference and candidate decoded frame counts differ")
        if not reference_ok:
            break

        reference = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2RGB).astype(
            np.float32
        ) / 255.0
        candidate = cv2.cvtColor(candidate_frame, cv2.COLOR_BGR2RGB).astype(
            np.float32
        ) / 255.0
        if reference.shape != candidate.shape:
            raise RuntimeError(f"Frame {decoded_frames}: decoded shapes differ")

        ssim_score, ssim_map = structural_similarity(
            reference,
            candidate,
            channel_axis=2,
            data_range=1.0,
            full=True,
        )
        weights_by_region = {
            "full_frame": np.ones(reference.shape[:2], dtype=np.float32)
        }

        if mask_capture:
            mask_ok, mask_frame = mask_capture.read()
            if not mask_ok:
                raise RuntimeError(
                    f"Mask ended before output videos at frame {decoded_frames}"
                )
            mask = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
            if mask.shape != reference.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (reference.shape[1], reference.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask = (mask > 127).astype(np.float32)
            mask_fraction.append(float(np.mean(mask)))
            weights_by_region.update(
                {
                    "mask": mask,
                    "background": 1.0 - mask,
                    "boundary": boundary_weights(mask, args.boundary_px),
                }
            )

        for region, weights in weights_by_region.items():
            psnr, mae, rmse = weighted_errors(reference, candidate, weights)
            frame_metrics[region]["psnr_db"].append(psnr)
            frame_metrics[region]["ssim"].append(
                float(ssim_score)
                if region == "full_frame"
                else weighted_ssim(ssim_map, weights)
            )
            frame_metrics[region]["mae"].append(mae)
            frame_metrics[region]["rmse"].append(rmse)

        if previous_reference is not None and previous_candidate is not None:
            reference_delta = reference - previous_reference
            candidate_delta = candidate - previous_candidate
            temporal_delta_mae.append(
                float(np.mean(np.abs(candidate_delta - reference_delta)))
            )
        previous_reference = reference
        previous_candidate = candidate
        decoded_frames += 1

    reference_capture.release()
    candidate_capture.release()
    if mask_capture:
        mask_capture.release()

    if decoded_frames == 0:
        raise RuntimeError("No frames were decoded")

    summarized_metrics = {
        region: {
            metric: summarize(values) for metric, values in metrics.items()
        }
        for region, metrics in frame_metrics.items()
    }
    result = {
        "schema_version": 1,
        "label": args.label,
        "reference_video": str(args.reference_video.resolve()),
        "candidate_video": str(args.candidate_video.resolve()),
        "mask_video": str(args.mask_video.resolve()) if args.mask_video else None,
        "boundary_px": args.boundary_px,
        "decoded_frames": decoded_frames,
        "reference_metadata": reference_metadata,
        "candidate_metadata": candidate_metadata,
        "mask_fraction": summarize(mask_fraction) if mask_fraction else None,
        "metrics": summarized_metrics,
        "temporal_delta_mae": summarize(temporal_delta_mae),
        "notes": {
            "reference_role": "The reference is a BF16 output, not ground truth.",
            "ssim_method": "skimage structural_similarity, RGB, data_range=1.0",
            "masked_ssim_method": "Mean of the SSIM similarity map weighted by the binary mask.",
            "temporal_delta_mae_method": "MAE between consecutive-frame residuals of candidate and reference.",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"[quality] {args.label}: {decoded_frames} frames")
    for region in region_names:
        psnr = summarized_metrics[region]["psnr_db"]["mean"]
        ssim = summarized_metrics[region]["ssim"]["mean"]
        mae = summarized_metrics[region]["mae"]["mean"]
        print(
            f"  {region:12s} PSNR={psnr:.4f} dB  SSIM={ssim:.6f}  MAE={mae:.6f}"
        )
    print(f"[output] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
