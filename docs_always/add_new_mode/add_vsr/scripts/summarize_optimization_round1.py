# SPDX-License-Identifier: Apache-2.0
"""Fail closed unless every optimization acceptance artifact is present and passes."""

import argparse
import json
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-dir", type=Path, default=Path("output_results/vsr/optimization_round1")
)
options = parser.parse_args()
out = options.output_dir
cases = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
reports = []
for case in cases:
    folder = out / "production" / f"both_{case}"
    frames = json.loads((folder / "frames.json").read_text())["summary"]
    mp4 = json.loads((folder / "mp4.json").read_text())["summary"]
    structure = json.loads((folder / "structural.json").read_text())
    assert (
        frames["pass_compare"] and mp4["pass_compare"] and structure["pass_structural"]
    ), case
    reports.append({"case": case, "frames": frames, "mp4": mp4, "structure_pass": True})
native = json.loads((out / "native/report.json").read_text())
assert all(s["pass"] for s in native["stages"])
assert native["native_structure"]["pass_structural"]
assert native["native_vs_direct"]["summary"]["pass_compare"]
if "native_vs_original" in native:
    assert native["native_vs_original"]["summary"]["pass_compare"]
samples = json.loads((out / "warm_video/results.json").read_text())
assert [s["mode"] for s in samples] == ["baseline", "both", "both", "baseline"]
for sample in samples:
    if sample.get("compile_decoder"):
        assert sample["compile_stats_before"] == sample["compile_stats_after"], sample
        assert sample["compile_stats_before"].get("unique_graphs", 0) > 0, sample
baseline = statistics.median(
    s["warm_video_s"] for s in samples if s["mode"] == "baseline"
)
optimized = statistics.median(s["warm_video_s"] for s in samples if s["mode"] == "both")
result = {
    "pass": True,
    "cases": reports,
    "total_frames": sum(r["frames"]["compared_frames"] for r in reports),
    "native": native,
    "warm_video": {
        "samples": samples,
        "baseline_median_s": baseline,
        "optimized_median_s": optimized,
        "latency_reduction": 1 - optimized / baseline,
        "speedup": baseline / optimized,
    },
}
(out / "report.json").write_text(json.dumps(result, indent=2))
print(
    json.dumps(
        {
            "pass": True,
            "total_frames": result["total_frames"],
            "warm_video": result["warm_video"],
        },
        indent=2,
    )
)
