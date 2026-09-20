# SPDX-License-Identifier: Apache-2.0
"""Compare completed migration runs directly against the swiftvr artifacts."""

import json
import time
from pathlib import Path

import cv2
from sglang.multimodal_gen.runtime.videoedit.compare import compare_videos
from sglang.multimodal_gen.runtime.vsr.verify.compare_frames import compare_frames
from sglang.multimodal_gen.runtime.vsr.verify.dumps import load_frames
from sglang.multimodal_gen.runtime.vsr.verify.structural_check import check

cv2.setNumThreads(4)
out = Path("output_results/vsr")
root = out / "migration_20260920"
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
done = set()
deadline = time.monotonic() + 10800
while len(done) < 9:
    if time.monotonic() > deadline:
        raise TimeoutError("Matrix inference has not completed in three hours")
    try:
        rows = json.loads((root / "results.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        time.sleep(20)
        continue
    for row in rows:
        name = row["case"]
        if name in done:
            continue
        print(time.strftime("%H:%M:%S"), name, "fresh comparison", flush=True)
        ref_video, cand_video = (
            str(out / f"EPS_{name}_swiftvr.mp4"),
            str(root / f"{name}.mp4"),
        )
        structural = check(ref_video, cand_video)
        assert structural["pass_structural"], name
        gate = thresholds[name]
        frames = compare_frames(
            load_frames(out / "dumps" / f"EPS_{name}_swiftvr"),
            load_frames(root / name),
            min_ssim=gate[0],
            max_mse=gate[1],
            max_mae=gate[2],
        )
        assert frames["summary"]["pass_compare"], name
        mp4 = compare_videos(
            ref_video,
            cand_video,
            min_ssim=0.97,
            max_mse=25,
            max_mae=2.5,
            allow_frame_count_delta=0,
            max_failed_frame_ratio=0,
        )
        assert mp4["summary"]["pass_compare"], name
        for layer, data in [
            ("structural", structural),
            ("frames", frames),
            ("mp4", mp4),
        ]:
            data["measurement"] = "fresh comparison against swiftvr reference artifacts"
            (root / f"{name}_{layer}.json").write_text(json.dumps(data, indent=2))
        done.add(name)
        print(name, "PASS fresh frames/mp4/structure", flush=True)
    if len(done) < 9:
        time.sleep(20)
print("ALL FRESH COMPARISONS PASS", flush=True)
