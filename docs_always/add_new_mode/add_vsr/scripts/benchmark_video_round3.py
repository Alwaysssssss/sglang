# SPDX-License-Identifier: Apache-2.0
"""Warm ABBA video benchmark for encoder and optional DiT compilation."""

import argparse
import json
import time
from pathlib import Path

import torch
from sglang.multimodal_gen.runtime.vsr.compile import compile_encoder
from sglang.multimodal_gen.runtime.vsr.model import VSRRestorer
from sglang.multimodal_gen.runtime.vsr.stream import stream_restore

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--compile-dit", action="store_true")
parser.add_argument("--gpu-postprocess", action="store_true")
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
model = VSRRestorer.from_pretrained(
    checkpoint_dir="/mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/outputs/best/checkpoint-1300",
    wan_root="/mnt/shanhai-ai/liuh/Wan2.2-TI2V-5B-Diffusers",
    cudnn_benchmark=True,
    channels_last_3d=True,
    compile_decoder=True,
)
encoder = model.vae.vae.encoder
eager_encoder, eager_dit = encoder.forward, model.dit.forward
compile_encoder(model.vae.vae)
compiled_encoder = encoder.forward
compiled_dit = (
    torch.compile(eager_dit, fullgraph=True, dynamic=False)
    if args.compile_dit
    else eager_dit
)
window = torch.load(
    "output_results/vsr/migration_20260920/single_reference/window/tile_00000.pt",
    map_location="cpu",
)
results = []
for index, mode in enumerate(["baseline", "candidate", "candidate", "baseline"]):
    encoder.forward = eager_encoder if mode == "baseline" else compiled_encoder
    model.dit.forward = eager_dit if mode == "baseline" else compiled_dit
    for _ in range(4):
        model.restore_window(window)
    for attempt in range(3):
        torch.cuda.synchronize()
        before = dict(torch._dynamo.utils.counters["stats"])
        start = time.perf_counter()
        frames = stream_restore(
            model,
            Path("../vsr/input/input.mp4"),
            args.output_dir / f"{index}_{mode}.mp4",
            target_h=3840,
            target_w=2160,
            read_queue=2,
            show_progress=False,
            gpu_postprocess=args.gpu_postprocess and mode == "candidate",
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        after = dict(torch._dynamo.utils.counters["stats"])
        if before == after:
            break
        print("Discard compilation sample", before, after, flush=True)
    else:
        raise RuntimeError("Compilation did not stabilize")
    result = {
        "mode": mode,
        "warm_video_s": elapsed,
        "frames": frames,
        "compile_dit": args.compile_dit,
        "gpu_postprocess": args.gpu_postprocess and mode == "candidate",
        "compile_stats_before": before,
        "compile_stats_after": after,
    }
    results.append(result)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(result, flush=True)
