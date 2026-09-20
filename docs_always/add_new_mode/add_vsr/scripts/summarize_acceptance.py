# SPDX-License-Identifier: Apache-2.0
"""Validate frozen gates after exact identity checks; never loosen thresholds."""

import json
from pathlib import Path

root = Path("output_results/vsr/migration_20260920")
thresholds = {
    "A1": (0.993550, 1.33, 1.04),
    "A2": (0.993415, 1.32, 1.04),
    "A3": (0.993580, 1.43, 1.07),
    "B1": (0.993763, 3.47, 1.65),
    "B2": (0.991861, 4.18, 1.82),
    "B3": (0.987439, 7.88, 2.50),
    "C1": (0.991078, 4.18, 1.87),
    "C2": (0.990769, 4.42, 1.88),
    "C3": (0.990601, 2.05, 1.32),
}
rows = json.loads((root / "results.json").read_text())
assert {row["case"] for row in rows} == set(thresholds)
for row in rows:
    row["reference_reports"] = {
        layer: json.loads((root / f"{row['case']}_{layer}.json").read_text())
        for layer in ("frames", "mp4", "structural")
    }
    assert row["frames_bitwise_equal"]
    assert row["mp4_sha256"] == row["reference_mp4_sha256"]
    for layer, gate in [("frames", thresholds[row["case"]]), ("mp4", (0.97, 25, 2.5))]:
        report = row["reference_reports"][layer]
        assert report["summary"]["compared_frames"] == row["frames"]
        assert all(
            f["ssim"] >= gate[0] and f["mse"] <= gate[1] and f["mae"] <= gate[2]
            for f in report["frames"]
        )
    assert row["reference_reports"]["structural"]["pass_structural"]
edges = json.loads((root / "edge_results.json").read_text())
assert {r["case"] for r in edges} == {"single", "three", "pad", "short", "one"}
assert (root / "native.mp4").read_bytes() == (
    root / "single_candidate.mp4"
).read_bytes()
summary = {
    "pass": True,
    "gpu": 7,
    "scope": "phase 1: diffusers models, SGLang streaming core and native pipeline",
    "metric_provenance": "Fresh frame, mp4 and structural comparisons of newly generated candidate artifacts against swiftvr reference artifacts; same-environment exact identity independently checked.",
    "cases": [
        {
            "case": r["case"],
            "frames": r["frames"],
            "exact_identity": True,
            "frames_metrics": r["reference_reports"]["frames"]["summary"],
            "mp4_metrics": r["reference_reports"]["mp4"]["summary"],
            "frame_thresholds": thresholds[r["case"]],
        }
        for r in rows
    ],
    "edge_cases": edges,
    "native_pipeline_exact_identity": True,
}
path = Path("output_results/vsr/reports/migration_20260920_acceptance.json")
path.write_text(json.dumps(summary, indent=2))
print(path, "PASS", sum(r["frames"] for r in rows), "matrix frames")
