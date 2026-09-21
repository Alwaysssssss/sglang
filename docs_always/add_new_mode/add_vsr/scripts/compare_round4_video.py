# SPDX-License-Identifier: Apache-2.0
"""Compare timed round-four A1 videos without affecting their timing."""

import json
from pathlib import Path

from sglang.multimodal_gen.runtime.videoedit.compare import compare_videos
from sglang.multimodal_gen.runtime.vsr.verify.structural_check import check

root = Path("output_results/vsr/optimization_round4/video")
ref = "output_results/vsr/EPS_A1_swiftvr.mp4"
for name in ("1_padding", "2_combined"):
    video = str(root / (name + ".mp4"))
    report = {
        "structure": check(ref, video),
        "video": compare_videos(
            ref,
            video,
            min_ssim=0.989,
            max_mse=36,
            max_mae=6,
            allow_frame_count_delta=0,
            max_failed_frame_ratio=0,
        ),
    }
    (root / (name + "_quality.json")).write_text(json.dumps(report, indent=2))
    print(name, report["video"]["summary"], flush=True)
