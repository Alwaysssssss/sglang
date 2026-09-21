# SPDX-License-Identifier: Apache-2.0
"""GPU7 warm A/B/B/A video timing: round-one optimized vs compiled decoder."""

import json
import time
from pathlib import Path

import torch
from sglang.multimodal_gen.runtime.vsr.compile import compile_decoder
from sglang.multimodal_gen.runtime.vsr.model import VSRRestorer
from sglang.multimodal_gen.runtime.vsr.stream import stream_restore

root = Path.cwd()
out = root / "output_results/vsr/optimization_round2/warm_video"
out.mkdir(exist_ok=True)
model = VSRRestorer.from_pretrained(
    checkpoint_dir="/mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/outputs/best/checkpoint-1300",
    wan_root="/mnt/shanhai-ai/liuh/Wan2.2-TI2V-5B-Diffusers",
    cudnn_benchmark=True,
    channels_last_3d=True,
)
decoder = model.vae.vae.decoder
eager = decoder.forward
compile_decoder(model.vae.vae)
compiled = decoder.forward
window = torch.load(
    root
    / "output_results/vsr/migration_20260920/single_reference/window/tile_00000.pt",
    map_location="cpu",
)
results = []
for index, mode in enumerate(["baseline", "both", "both", "baseline"]):
    decoder.forward = eager if mode == "baseline" else compiled
    model.cudnn_benchmark = True
    for _ in range(4):
        model.restore_window(window)
    torch.cuda.synchronize()
    before = dict(torch._dynamo.utils.counters["stats"])
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
        "compile_decoder": mode == "both",
        "cudnn_benchmark": model.cudnn_benchmark,
        "frames": frames,
        "warm_video_s": time.perf_counter() - started,
        "compile_stats_before": before,
        "compile_stats_after": dict(torch._dynamo.utils.counters["stats"]),
    }
    results.append(result)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(result, flush=True)
