# SPDX-License-Identifier: Apache-2.0
"""Warm GPU7 video A/B/C/C/B/A comparison of GPU spatial fusion and prefetch."""

import argparse
import json
import time
from pathlib import Path

import torch
from io_experiment import spatial_gpu
from sglang.multimodal_gen.runtime.vsr.model import VSRRestorer
from sglang.multimodal_gen.runtime.vsr.stream import stream_restore

root = Path.cwd()
parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-dir",
    type=Path,
    default=root / "output_results/vsr/optimization_round3/io",
)
parser.add_argument(
    "--modes", default="baseline,fusion,prefetch,prefetch,fusion,baseline"
)
args = parser.parse_args()
out = args.output_dir
out.mkdir(exist_ok=True)
model = VSRRestorer.from_pretrained(
    checkpoint_dir="/mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/outputs/best/checkpoint-1300",
    wan_root="/mnt/shanhai-ai/liuh/Wan2.2-TI2V-5B-Diffusers",
    cudnn_benchmark=True,
    channels_last_3d=True,
    compile_decoder=True,
)
window = torch.load(
    root
    / "output_results/vsr/migration_20260920/single_reference/window/tile_00000.pt",
    map_location="cpu",
)
results = []
for index, mode in enumerate(args.modes.split(",")):
    with spatial_gpu(mode, model.device):
        for _ in range(4):
            model.restore_window(window)
        for attempt in range(3):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = dict(torch._dynamo.utils.counters["stats"])
            start = time.perf_counter()
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
            elapsed = time.perf_counter() - start
            after = dict(torch._dynamo.utils.counters["stats"])
            if before == after:
                break
            print(
                mode, "additional warmup due to compilation", before, after, flush=True
            )
        else:
            raise RuntimeError("Compiler did not stabilize")
    result = {
        "mode": mode,
        "warm_video_s": elapsed,
        "frames": frames,
        "compile_stats_before": before,
        "compile_stats_after": after,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    results.append(result)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(result, flush=True)
