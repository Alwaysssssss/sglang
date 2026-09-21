# SPDX-License-Identifier: Apache-2.0
"""Screen real-window candidates against round-four baseline and original output."""

import json
from pathlib import Path

import torch
from sglang.multimodal_gen.runtime.vsr.verify.compare_frames import compare_frames
from sglang.multimodal_gen.runtime.vsr.video_io import to_uint8_hwc

folder = Path("output_results/vsr/optimization_round4")
baseline = torch.load(folder / "baseline.pt", map_location="cpu")
reference = torch.from_numpy(
    to_uint8_hwc(
        torch.load(
            "output_results/vsr/optimization_round1/baseline.pt", map_location="cpu"
        )
    )
)
reports = {}
for path in sorted(folder.glob("*.pt")):
    value = torch.load(path, map_location="cpu")
    assert value.shape == baseline.shape and value.isfinite().all(), path
    report = compare_frames(
        reference,
        torch.from_numpy(to_uint8_hwc(value)),
        min_ssim=0.989,
        max_mse=36,
        max_mae=6,
    )
    reports[path.stem] = {
        "baseline_equal": torch.equal(value, baseline),
        "baseline_max_abs": (value.float() - baseline.float()).abs().max().item(),
        "original": report,
    }
    print(
        path.stem, reports[path.stem]["baseline_equal"], report["summary"], flush=True
    )
(folder / "tile_quality.json").write_text(json.dumps(reports, indent=2))
