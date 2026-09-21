# SPDX-License-Identifier: Apache-2.0
"""Quality comparisons kept outside performance timing."""

import json
from pathlib import Path

from sglang.multimodal_gen.runtime.videoedit.compare import compare_videos
from sglang.multimodal_gen.runtime.vsr.verify.structural_check import check


def main():
    root = Path("output_results/vsr/parallel_round1")
    report = {}
    for label, reference, candidate in [
        ("dual_vs_single", root / "0_single.mp4", root / "1_dual.mp4"),
        (
            "dual_vs_original",
            Path("output_results/vsr/EPS_A1_swiftvr.mp4"),
            root / "1_dual.mp4",
        ),
    ]:
        result = {
            "structure": check(str(reference), str(candidate)),
            "video": compare_videos(
                str(reference),
                str(candidate),
                min_ssim=0.985,
                max_mse=36,
                max_mae=6,
                allow_frame_count_delta=0,
                max_failed_frame_ratio=0,
            ),
        }
        report[label] = result
        print(label, result["video"]["summary"], flush=True)
        (root / "quality.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
