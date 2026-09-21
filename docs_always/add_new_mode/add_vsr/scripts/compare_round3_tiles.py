# SPDX-License-Identifier: Apache-2.0
"""Quick per-frame screen of each real-window candidate against original numerics."""

import json
from pathlib import Path

import torch
from sglang.multimodal_gen.runtime.vsr.verify.compare_frames import compare_frames
from sglang.multimodal_gen.runtime.vsr.video_io import to_uint8_hwc

root = Path.cwd()
folder = root / "output_results/vsr/optimization_round3"
reference = torch.load(
    root / "output_results/vsr/optimization_round1/baseline.pt", map_location="cpu"
)
ref = torch.from_numpy(to_uint8_hwc(reference))
reports = {}
for path in folder.glob("*.pt"):
    value = torch.load(path, map_location="cpu")
    if not bool(value.isfinite().all()):
        reports[path.stem] = {"finite": False, "pass": False}
        continue
    report = compare_frames(
        ref,
        torch.from_numpy(to_uint8_hwc(value)),
        min_ssim=0.989,
        max_mse=36,
        max_mae=6,
    )
    reports[path.stem] = report
    print(path.stem, report["summary"], flush=True)
(folder / "tile_quality.json").write_text(json.dumps(reports, indent=2))
