# SPDX-License-Identifier: Apache-2.0
"""Paired GPU7 A1 video timing with warm models, no acceptance dumps."""

import json
import time
from pathlib import Path

import torch
from sglang.multimodal_gen.runtime.vsr.model import VSRRestorer
from sglang.multimodal_gen.runtime.vsr.stream import stream_restore

root = Path.cwd()
out = root / "output_results/vsr/optimization_round1/warm_video"
out.mkdir(exist_ok=True)
model = VSRRestorer.from_pretrained(
    checkpoint_dir="/mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/outputs/best/checkpoint-1300",
    wan_root="/mnt/shanhai-ai/liuh/Wan2.2-TI2V-5B-Diffusers",
)
window = torch.load(
    root
    / "output_results/vsr/migration_20260920/single_reference/window/tile_00000.pt",
    map_location="cpu",
)
results = []
for index, mode in enumerate(["baseline", "both", "both", "baseline"]):
    model.cudnn_benchmark = mode == "both"
    torch.nn.utils.convert_conv3d_weight_memory_format(
        model.vae, torch.channels_last_3d if mode == "both" else torch.contiguous_format
    )
    for _ in range(2):
        model.restore_window(window)
    torch.cuda.synchronize()
    started = time.perf_counter()
    frames = stream_restore(
        model,
        root.parent / "vsr/input/input.mp4",
        out / f"{index}_{mode}.mp4",
        target_h=3840,
        target_w=2160,
        read_queue=2,
        show_progress=False,
    )
    torch.cuda.synchronize()
    result = {
        "mode": mode,
        "frames": frames,
        "warm_video_s": time.perf_counter() - started,
    }
    results.append(result)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(result, flush=True)
