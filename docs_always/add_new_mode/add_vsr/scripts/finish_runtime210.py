# SPDX-License-Identifier: Apache-2.0
"""Require all runtime-switch checks before emitting a passing report."""

import hashlib
import json
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[4]
run = root / "output_results/vsr/runtime210_validation"
rows = json.loads((run / "matrix/results.json").read_text())
assert {r["case"] for r in rows} == {
    "A1",
    "A2",
    "A3",
    "B1",
    "B2",
    "B3",
    "C1",
    "C2",
    "C3",
}
assert sum(r["frames"] for r in rows) == 649
for row in rows:
    assert row["torch"] == "2.10.0+cu126"
    assert row["reference_environment"] == "swiftvr"
    assert row["frames_bitwise_equal"]
    assert row["mp4_sha256"] == row["reference_mp4_sha256"]
    video = run / "matrix" / f"{row['case']}.mp4"
    assert hashlib.sha256(video.read_bytes()).hexdigest() == row["mp4_sha256"]
ref, cand = run / "stages_reference", run / "stages_candidate"
ref_manifest = json.loads((ref / "manifest.json").read_text())
cand_manifest = json.loads((cand / "manifest.json").read_text())
assert ref_manifest["shapes"] == cand_manifest["shapes"]
assert ref_manifest["color_stats"] == cand_manifest["color_stats"]
assert ref_manifest["frames_written"] == cand_manifest["frames_written"] == 33
ref_files = sorted(ref.rglob("*.pt"))
assert len(ref_files) == len(list(cand.rglob("*.pt"))) == 7
for path in ref_files:
    a = torch.load(path, map_location="cpu")
    b = torch.load(cand / path.relative_to(ref), map_location="cpu")
    assert (
        a.shape == b.shape
        and a.dtype == b.dtype
        and torch.isfinite(a).all()
        and torch.equal(a, b)
    )
assert (run / "native.mp4").read_bytes() == (run / "native_reference.mp4").read_bytes()
assert (
    "Ran 6 tests" in (run / "tests.log").read_text()
    and "\nOK\n" in (run / "tests.log").read_text()
)
assert "61/61 comparisons bit-identical" in (run / "ops.log").read_text()
env = json.loads((run / "environment.json").read_text())
assert env["torch"] == "2.10.0+cu126" and env["cuda"] == "12.6"
benchmark = json.loads((run / "benchmark.json").read_text())
old = json.loads(
    (root / "output_results/vsr/performance_20260920/candidate_sglang.json").read_text()
)
report = {
    "pass": True,
    "environment": env,
    "matrix": rows,
    "matrix_frames": 649,
    "reference": "Original swiftvr artifacts",
    "quality": "Exact pixel and mp4 identity; zero numerical tolerance required.",
    "stage_tensor_files_exact": len(ref_files),
    "native_pipeline_exact": True,
    "cpu_regressions": 6,
    "pure_function_comparisons": 61,
    "benchmark": benchmark,
    "single_tile_speedup_over_previous_runtime": old["median"]["wall_s"]
    / benchmark["median"]["wall_s"],
    "timing_scope": "Warm single 33x320x640 tile, not whole-video speedup.",
}
(run / "report.json").write_text(json.dumps(report, indent=2))
print("PASS: runtime210, 9 cases / 649 frames, stages and native pipeline exact")
