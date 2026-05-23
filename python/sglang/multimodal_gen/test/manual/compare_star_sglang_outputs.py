from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio
import imageio.v3 as iio
import numpy as np
from skimage.metrics import structural_similarity


@dataclass(frozen=True)
class ThresholdProfile:
    name: str
    min_ssim: float
    max_mse: float
    max_mae: float
    allow_frame_count_delta: int
    max_failed_frame_ratio: float
    drop_tail_frames: bool = True


PROFILES = {
    "smoke": ThresholdProfile(
        name="smoke",
        min_ssim=0.80,
        max_mse=400.0,
        max_mae=15.0,
        allow_frame_count_delta=1,
        max_failed_frame_ratio=0.10,
    ),
    "baseline": ThresholdProfile(
        name="baseline",
        min_ssim=0.90,
        max_mse=150.0,
        max_mae=8.0,
        allow_frame_count_delta=1,
        max_failed_frame_ratio=0.05,
    ),
    "strict": ThresholdProfile(
        name="strict",
        min_ssim=0.95,
        max_mse=60.0,
        max_mae=5.0,
        allow_frame_count_delta=0,
        max_failed_frame_ratio=0.0,
        drop_tail_frames=False,
    ),
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frame-by-frame parity checker for STAR_mg vs SGLang outputs."
    )
    parser.add_argument("--reference", required=True, help="Reference mp4 or frame dir.")
    parser.add_argument("--candidate", required=True, help="Candidate mp4 or frame dir.")
    parser.add_argument(
        "--mode",
        choices=sorted(PROFILES.keys()),
        default="baseline",
        help="Threshold profile to use.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON output path. Defaults to <candidate>.parity.json.",
    )
    parser.add_argument(
        "--allow-frame-count-delta",
        type=int,
        default=None,
        help="Override the selected profile frame-count delta threshold.",
    )
    parser.add_argument(
        "--min-ssim",
        type=float,
        default=None,
        help="Override the selected profile minimum SSIM.",
    )
    parser.add_argument(
        "--max-mse",
        type=float,
        default=None,
        help="Override the selected profile maximum MSE.",
    )
    parser.add_argument(
        "--max-mae",
        type=float,
        default=None,
        help="Override the selected profile maximum MAE.",
    )
    parser.add_argument(
        "--max-failed-frame-ratio",
        type=float,
        default=None,
        help="Override the selected profile failed frame ratio.",
    )
    parser.add_argument(
        "--drop-tail-frames",
        dest="drop_tail_frames",
        action="store_true",
        help="Allow truncating trailing unmatched frames when the count delta is acceptable.",
    )
    parser.add_argument(
        "--no-drop-tail-frames",
        dest="drop_tail_frames",
        action="store_false",
        help="Disallow truncating trailing unmatched frames.",
    )
    parser.set_defaults(drop_tail_frames=None)
    return parser.parse_args()


def _resolve_profile(args: argparse.Namespace) -> ThresholdProfile:
    profile = PROFILES[args.mode]
    return ThresholdProfile(
        name=profile.name,
        min_ssim=args.min_ssim if args.min_ssim is not None else profile.min_ssim,
        max_mse=args.max_mse if args.max_mse is not None else profile.max_mse,
        max_mae=args.max_mae if args.max_mae is not None else profile.max_mae,
        allow_frame_count_delta=(
            args.allow_frame_count_delta
            if args.allow_frame_count_delta is not None
            else profile.allow_frame_count_delta
        ),
        max_failed_frame_ratio=(
            args.max_failed_frame_ratio
            if args.max_failed_frame_ratio is not None
            else profile.max_failed_frame_ratio
        ),
        drop_tail_frames=(
            args.drop_tail_frames
            if args.drop_tail_frames is not None
            else profile.drop_tail_frames
        ),
    )


def _ensure_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[..., :3]
    elif frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB frame, got shape={frame.shape}")

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _read_frames(path: Path) -> tuple[list[np.ndarray], dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_dir():
        frame_paths = sorted(
            p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )
        frames = [_ensure_rgb_uint8(iio.imread(frame_path)) for frame_path in frame_paths]
        metadata = {
            "type": "frame_dir",
            "path": str(path.resolve()),
            "num_frames": len(frames),
            "fps": None,
        }
        return frames, metadata

    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        frame = _ensure_rgb_uint8(iio.imread(path))
        return [frame], {
            "type": "image",
            "path": str(path.resolve()),
            "num_frames": 1,
            "fps": None,
        }

    with imageio.get_reader(path) as reader:
        metadata = reader.get_meta_data()
    frames = [_ensure_rgb_uint8(frame) for frame in iio.imiter(path)]
    return frames, {
        "type": "video",
        "path": str(path.resolve()),
        "num_frames": len(frames),
        "fps": metadata.get("fps"),
    }


def _compute_frame_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    reference = _ensure_rgb_uint8(reference)
    candidate = _ensure_rgb_uint8(candidate)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Frame shape mismatch: {reference.shape} vs {candidate.shape}"
        )

    reference_f = reference.astype(np.float32)
    candidate_f = candidate.astype(np.float32)
    diff = candidate_f - reference_f
    mse = float(np.mean(np.square(diff)))
    mae = float(np.mean(np.abs(diff)))
    max_abs_diff = float(np.max(np.abs(diff)))
    psnr = float("inf") if mse == 0.0 else 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)
    ssim = float(
        structural_similarity(reference, candidate, channel_axis=2, data_range=255)
    )
    return {
        "ssim": ssim,
        "mse": mse,
        "mae": mae,
        "psnr": psnr,
        "max_abs_diff": max_abs_diff,
    }


def _summarize_metrics(
    frame_metrics: list[dict[str, float]],
    failed_frames: list[dict[str, object]],
) -> dict[str, object]:
    ssim_values = [metric["ssim"] for metric in frame_metrics]
    mse_values = [metric["mse"] for metric in frame_metrics]
    mae_values = [metric["mae"] for metric in frame_metrics]
    psnr_values = [metric["psnr"] for metric in frame_metrics]
    max_abs_values = [metric["max_abs_diff"] for metric in frame_metrics]

    return {
        "num_compared_frames": len(frame_metrics),
        "ssim_mean": float(np.mean(ssim_values)),
        "ssim_min": float(np.min(ssim_values)),
        "mse_mean": float(np.mean(mse_values)),
        "mse_max": float(np.max(mse_values)),
        "mae_mean": float(np.mean(mae_values)),
        "mae_max": float(np.max(mae_values)),
        "psnr_mean": float(np.mean(psnr_values)),
        "psnr_min": float(np.min(psnr_values)),
        "max_abs_diff_max": float(np.max(max_abs_values)),
        "failed_frames": failed_frames,
        "num_failed_frames": len(failed_frames),
        "failed_frame_ratio": (
            float(len(failed_frames)) / float(len(frame_metrics)) if frame_metrics else 1.0
        ),
    }


def _compare_frames(
    reference_frames: list[np.ndarray],
    candidate_frames: list[np.ndarray],
    profile: ThresholdProfile,
) -> dict[str, object]:
    frame_delta = abs(len(reference_frames) - len(candidate_frames))
    if frame_delta > profile.allow_frame_count_delta:
        raise ValueError(
            "Frame-count delta exceeds threshold: "
            f"reference={len(reference_frames)}, candidate={len(candidate_frames)}, "
            f"allow={profile.allow_frame_count_delta}"
        )

    trimmed = False
    if len(reference_frames) != len(candidate_frames):
        if not profile.drop_tail_frames:
            raise ValueError(
                "Frame counts differ and tail truncation is disabled: "
                f"reference={len(reference_frames)}, candidate={len(candidate_frames)}"
            )
        trimmed = True
        compared_frames = min(len(reference_frames), len(candidate_frames))
        reference_frames = reference_frames[:compared_frames]
        candidate_frames = candidate_frames[:compared_frames]

    if not reference_frames:
        raise ValueError("No frames available for comparison.")

    failed_frames: list[dict[str, object]] = []
    frame_metrics: list[dict[str, float]] = []
    for frame_index, (reference_frame, candidate_frame) in enumerate(
        zip(reference_frames, candidate_frames, strict=True)
    ):
        if reference_frame.shape != candidate_frame.shape:
            raise ValueError(
                "Resolution mismatch at frame "
                f"{frame_index}: {reference_frame.shape} vs {candidate_frame.shape}"
            )

        metrics = _compute_frame_metrics(reference_frame, candidate_frame)
        frame_metrics.append(metrics)

        failure_reasons: list[str] = []
        if metrics["ssim"] < profile.min_ssim:
            failure_reasons.append(
                f"ssim<{profile.min_ssim:.4f} ({metrics['ssim']:.4f})"
            )
        if metrics["mse"] > profile.max_mse:
            failure_reasons.append(
                f"mse>{profile.max_mse:.4f} ({metrics['mse']:.4f})"
            )
        if metrics["mae"] > profile.max_mae:
            failure_reasons.append(
                f"mae>{profile.max_mae:.4f} ({metrics['mae']:.4f})"
            )

        if failure_reasons:
            failed_frames.append(
                {
                    "frame_index": frame_index,
                    "failure_reasons": failure_reasons,
                    **metrics,
                }
            )

    summary = _summarize_metrics(frame_metrics, failed_frames)
    summary["frame_count_trimmed"] = trimmed
    summary["reference_frame_count"] = len(reference_frames)
    summary["candidate_frame_count"] = len(candidate_frames)
    summary["max_allowed_failed_frame_ratio"] = profile.max_failed_frame_ratio
    summary["passed"] = summary["failed_frame_ratio"] <= profile.max_failed_frame_ratio
    return summary


def _default_output_json(candidate_path: Path) -> Path:
    if candidate_path.is_dir():
        return candidate_path / "star_parity_report.json"
    return candidate_path.with_suffix(candidate_path.suffix + ".parity.json")


def main() -> int:
    args = _parse_args()
    profile = _resolve_profile(args)

    reference_path = Path(args.reference).expanduser().resolve()
    candidate_path = Path(args.candidate).expanduser().resolve()
    output_json_path = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else _default_output_json(candidate_path)
    )

    reference_frames, reference_meta = _read_frames(reference_path)
    candidate_frames, candidate_meta = _read_frames(candidate_path)
    comparison = _compare_frames(reference_frames, candidate_frames, profile)

    report = {
        "reference": reference_meta,
        "candidate": candidate_meta,
        "threshold_profile": asdict(profile),
        "comparison": comparison,
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"reference: {reference_path}")
    print(f"candidate: {candidate_path}")
    print(f"profile:   {profile.name}")
    print(f"ssim_mean: {comparison['ssim_mean']:.6f}")
    print(f"ssim_min:  {comparison['ssim_min']:.6f}")
    print(f"mse_mean:  {comparison['mse_mean']:.6f}")
    print(f"mae_mean:  {comparison['mae_mean']:.6f}")
    print(f"failed:    {comparison['num_failed_frames']}/{comparison['num_compared_frames']}")
    print(f"report:    {output_json_path}")

    return 0 if comparison["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
