# SPDX-License-Identifier: Apache-2.0
"""Per-stage tensor comparison for the §4.1.3 sampling points.

Two kinds of mismatch are reported separately, because they call for different
responses (``requirements.md`` §4.1.1):

* **structural** -- shape and dtype. Zero tolerance: a mismatch is a defect,
  not drift.
* **numerical** -- the values. Measured, and optionally gated.

With no ``--atol``/``--rtol`` the tool only *measures*: that is how the
per-stage ``ε_torch`` floor gets established in the first place. Supplying them
turns it into a gate.

The ``retired`` point (pre-encode uint8 frames) is deliberately skipped -- it is
``compare_frames.py``'s job, and it has its own report.

Usage:
    python -m sglang.multimodal_gen.runtime.vsr.verify.compare_tensors \\
        --reference-dir DUMP_A --candidate-dir DUMP_B [--report-json R.json] \\
        [--atol X --rtol Y]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from sglang.multimodal_gen.runtime.vsr.verify.dumps import list_parts  # noqa: F401

#: Points compared here, in call-chain order (requirements.md §4.1.3).
TENSOR_POINTS = [
    "chunk_input",
    "window",
    "latent",
    "velocity",
    "decoded",
    "spatial_fused",
]


def _load(dump_dir: Path, point: str) -> dict[str, torch.Tensor]:
    """All files of one dump point, keyed by file name."""
    d = Path(dump_dir) / point
    if not d.is_dir():
        return {}
    return {p.name: torch.load(p, map_location="cpu") for p in sorted(d.glob("*.pt"))}


def _errors(ref: torch.Tensor, cand: torch.Tensor) -> dict[str, float]:
    """Absolute errors, plus errors normalised by the reference's range.

    A per-element ``|diff| / |ref|`` is not reported: it is dominated by
    elements whose reference value is near zero, which says nothing about the
    drift. ``rel_mean`` / ``rel_max`` divide by ``|ref|.max()`` instead, so they
    are stable and directly comparable across stages with different scales.
    """
    a = ref.float()
    b = cand.float()
    diff = (a - b).abs()
    ref_abs_max = float(a.abs().max()) if a.numel() else 0.0
    scale = ref_abs_max if ref_abs_max > 0 else 1.0
    return {
        "max_abs": float(diff.max()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean()) if diff.numel() else 0.0,
        "rel_mean": (float(diff.mean()) / scale) if diff.numel() else 0.0,
        "rel_max": (float(diff.max()) / scale) if diff.numel() else 0.0,
        "ref_abs_max": ref_abs_max,
    }


def compare(
    ref_dir: Path,
    cand_dir: Path,
    atol: float | None = None,
    rtol: float | None = None,
) -> dict:
    gating = atol is not None or rtol is not None
    atol = 0.0 if atol is None else atol
    rtol = 0.0 if rtol is None else rtol

    points: dict[str, dict] = {}
    for point in TENSOR_POINTS:
        ref_files = _load(ref_dir, point)
        cand_files = _load(cand_dir, point)
        if not ref_files and not cand_files:
            continue

        entry: dict[str, object] = {
            "reference_files": sorted(ref_files),
            "candidate_files": sorted(cand_files),
        }

        # Structural first: any asymmetry is a defect regardless of values.
        missing = sorted(set(ref_files) - set(cand_files))
        extra = sorted(set(cand_files) - set(ref_files))
        per_file = []
        structural_ok = not missing and not extra
        worst = None

        for name in sorted(set(ref_files) & set(cand_files)):
            a, b = ref_files[name], cand_files[name]
            if (
                a.shape != b.shape
                or a.dtype != b.dtype
                or not torch.isfinite(a).all()
                or not torch.isfinite(b).all()
            ):
                per_file.append(
                    {
                        "file": name,
                        "shape_reference": list(a.shape),
                        "shape_candidate": list(b.shape),
                        "dtype_reference": str(a.dtype),
                        "dtype_candidate": str(b.dtype),
                        "structural_pass": False,
                    }
                )
                structural_ok = False
                continue
            err = _errors(a, b)
            entry_obj = {
                "file": name,
                "shape": list(a.shape),
                "dtype": str(a.dtype),
                "structural_pass": True,
                **err,
            }
            if gating:
                entry_obj["within_tolerance"] = bool(
                    torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)
                )
                structural_ok = structural_ok and entry_obj["within_tolerance"]
            per_file.append(entry_obj)
            if worst is None or err["max_abs"] > worst["max_abs"]:
                worst = entry_obj

        entry["missing_in_candidate"] = missing
        entry["extra_in_candidate"] = extra
        entry["n_compared"] = len(per_file)
        entry["worst"] = (
            {
                k: worst[k]
                for k in (
                    "file",
                    "max_abs",
                    "mean_abs",
                    "rel_mean",
                    "rel_max",
                    "ref_abs_max",
                )
            }
            if worst
            else None
        )
        entry["structural_pass"] = structural_ok
        points[point] = entry

    all_structural = bool(points) and all(p["structural_pass"] for p in points.values())
    return {
        "reference_dir": str(ref_dir),
        "candidate_dir": str(cand_dir),
        "gating": gating,
        "atol": atol,
        "rtol": rtol,
        "points": points,
        "structural_pass": all_structural,
        "pass_compare": all_structural if gating else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare per-stage tensor dumps.")
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--report-json")
    parser.add_argument(
        "--atol", type=float, default=None, help="Gate; omit to measure only"
    )
    parser.add_argument(
        "--rtol", type=float, default=None, help="Gate; omit to measure only"
    )
    args = parser.parse_args(argv)

    report = compare(
        Path(args.reference_dir), Path(args.candidate_dir), args.atol, args.rtol
    )
    if args.report_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_json)), exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    for point, p in report["points"].items():
        mark = "PASS" if p["structural_pass"] else "FAIL"
        w = p["worst"]
        if w is None:
            print(f"  [{mark}] {point}: {p['n_compared']} compared (no value metrics)")
        else:
            print(
                f"  [{mark}] {point}: n={p['n_compared']} worst={w['file']} "
                f"max_abs={w['max_abs']:.3e} mean_abs={w['mean_abs']:.3e} "
                f"rel_mean={w['rel_mean']:.3e} rel_max={w['rel_max']:.3e} "
                f"(|ref|max={w['ref_abs_max']:.3f})"
            )
    print(f"structural_pass = {report['structural_pass']}")
    return 0 if report["structural_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
