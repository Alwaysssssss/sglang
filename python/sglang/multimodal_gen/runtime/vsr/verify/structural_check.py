# SPDX-License-Identifier: Apache-2.0
"""Structural consistency gate, run *before* the frame comparator.

``requirements.md`` §4.3 requires resolution, frame count, frame rate and frame
order to be checked as a separate, zero-tolerance gate. It is deliberately not
delegated to ``videoedit.compare``:

* a shape mismatch there triggers a silent ``cv2.resize`` of the candidate, so
  a wrong output resolution would be smoothed into a high SSIM instead of
  failing;
* it reads frames with ``cv2.VideoCapture`` and never looks at the container's
  frame rate, so an fps mismatch passes unnoticed;
* a frame-count delta above ``allow_frame_count_delta`` raises ``ValueError``
  inside ``compare_videos`` -- a traceback and exit code 1, indistinguishable
  from "thresholds failed".

Usage:
    python -m sglang.multimodal_gen.runtime.vsr.verify.structural_check \\
        --reference A.mp4 --candidate B.mp4 [--report-json R.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

#: Frame-order search window; a real ordering bug shows up as a non-zero lag.
LAG_RANGE = range(-2, 3)


def probe(path: str) -> Dict[str, object]:
    """Container-level facts. Uses ffprobe, falling back to OpenCV."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,avg_frame_rate,nb_frames",
             "-of", "json", path],
            capture_output=True, text=True, check=True, timeout=60,
        ).stdout
        stream = json.loads(out)["streams"][0]
        num, _, den = stream.get("avg_frame_rate", "0/1").partition("/")
        fps = float(num) / float(den) if float(den or 0) else float("nan")
        return {
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "fps": fps,
            "nb_frames": int(stream.get("nb_frames", 0) or 0),
        }
    except Exception:
        cap = cv2.VideoCapture(path)
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "nb_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        cap.release()
        return info


def read_frames(path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)  # keep BGR; only the geometry and ordering matter
    cap.release()
    return frames


def _best_lag(ref: List[float], cand: List[float]) -> tuple:
    """Smallest |lag| that minimises the mean absolute difference of the means."""
    n = min(len(ref), len(cand))
    best = (0, float("inf"))
    for lag in LAG_RANGE:
        diffs = [
            abs(ref[i] - cand[i + lag])
            for i in range(n)
            if 0 <= i + lag < n
        ]
        if not diffs:
            continue
        score = float(np.mean(diffs))
        # Strictly better, or equal score with a smaller |lag|.
        if score < best[1] - 1e-12 or (abs(score - best[1]) <= 1e-12 and abs(lag) < abs(best[0])):
            best = (lag, score)
    return best


def check(reference: str, candidate: str) -> dict:
    ref_info, cand_info = probe(reference), probe(candidate)
    ref_frames, cand_frames = read_frames(reference), read_frames(candidate)

    checks: Dict[str, dict] = {}

    # Resolution: compare the decoder's view, which is what a viewer sees.
    # Spelled out as W x H because the pipeline's own parameter uses H x W and
    # the two are exactly the confusion requirements.md §3 warns about.
    res_ref = (ref_info["width"], ref_info["height"])
    res_cand = (cand_info["width"], cand_info["height"])
    checks["resolution"] = {
        "reference": res_ref, "candidate": res_cand, "pass": res_ref == res_cand,
        "units": "width_height",
    }

    frame_shapes = {f.shape[:2] for f in ref_frames} | {f.shape[:2] for f in cand_frames}
    checks["frame_geometry_uniform"] = {
        "distinct_shapes": sorted(str(s) for s in frame_shapes),
        "pass": len(frame_shapes) == 1,
        "units": "numpy_frame_array_height_width",
    }

    checks["frame_count"] = {
        "reference": len(ref_frames), "candidate": len(cand_frames),
        "delta": abs(len(ref_frames) - len(cand_frames)),
        "pass": len(ref_frames) == len(cand_frames),
    }

    fps_ref, fps_cand = ref_info["fps"], cand_info["fps"]
    checks["fps"] = {
        "reference": fps_ref, "candidate": fps_cand,
        "pass": math.isfinite(fps_ref) and math.isfinite(fps_cand)
        and abs(fps_ref - fps_cand) < 1e-6,
    }

    if ref_frames and cand_frames:
        lag, score = _best_lag(
            [float(f.mean()) for f in ref_frames],
            [float(f.mean()) for f in cand_frames],
        )
        checks["frame_order"] = {
            "best_lag": lag, "mean_abs_mean_diff": score, "pass": lag == 0,
        }
    else:
        checks["frame_order"] = {"best_lag": None, "pass": False}

    passed = all(c["pass"] for c in checks.values())
    return {
        "reference": reference,
        "candidate": candidate,
        "checks": checks,
        "pass_structural": passed,
        "notes": [
            "drop_reference_first_frame / drop_candidate_first_frame must stay off "
            "(requirements.md §4.3); this check never drops frames.",
        ],
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Structural consistency check (requirements §4.3).")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--report-json")
    args = parser.parse_args(argv)

    report = check(args.reference, args.candidate)
    if args.report_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_json)), exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    for name, c in report["checks"].items():
        mark = "PASS" if c["pass"] else "FAIL"
        detail = {k: v for k, v in c.items() if k != "pass"}
        print(f"  [{mark}] {name}: {detail}")
    print(f"pass_structural = {report['pass_structural']}")
    return 0 if report["pass_structural"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
