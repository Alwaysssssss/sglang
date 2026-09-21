# SPDX-License-Identifier: Apache-2.0
"""Compare completed dual-GPU HTTP artifacts, outside timing."""

import json
from pathlib import Path

from sglang.multimodal_gen.runtime.videoedit.compare import compare_videos
from sglang.multimodal_gen.runtime.vsr.verify.structural_check import check


def main():
    root = Path("output_results/vsr/server_api_dual_test")
    candidate = str(root / "3.mp4")
    reports = {}
    for label, reference in (
        ("original", "output_results/vsr/EPS_A1_swiftvr.mp4"),
        ("single_api", "output_results/vsr/server_api_test/3.mp4"),
    ):
        report = {
            "structure": check(reference, candidate),
            "video": compare_videos(
                reference,
                candidate,
                min_ssim=0.985,
                max_mse=36,
                max_mae=6,
                allow_frame_count_delta=0,
                max_failed_frame_ratio=0,
            ),
        }
        reports[label] = report
        (root / "quality.json").write_text(json.dumps(reports, indent=2))
        print(label, report["video"]["summary"], flush=True)
        assert report["structure"]["pass_structural"]
        assert report["video"]["summary"]["pass_compare"]


if __name__ == "__main__":
    main()
