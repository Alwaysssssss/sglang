# SPDX-License-Identifier: Apache-2.0
"""Warm ABC CBA comparison; baseline is round-three encoder + GPU postprocessing."""

import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d
from round4_experiments import apply
from round4_postprocess import fused_postprocess
from sglang.multimodal_gen.runtime.vsr.model import VSRRestorer
from sglang.multimodal_gen.runtime.vsr.stream import stream_restore

out = Path("output_results/vsr/optimization_round4/video")
out.mkdir(parents=True, exist_ok=True)
model = VSRRestorer.from_pretrained(
    checkpoint_dir="/mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/outputs/best/checkpoint-1300",
    wan_root="/mnt/shanhai-ai/liuh/Wan2.2-TI2V-5B-Diffusers",
    cudnn_benchmark=True,
    channels_last_3d=True,
    compile_encoder=True,
    compile_decoder=True,
)
originals = [
    (m, m.forward)
    for m in model.vae.vae.decoder.modules()
    if isinstance(m, WanCausalConv3d)
]
originals += [(model.dit.condition_embedder, model.dit.condition_embedder.forward)]
originals += [(block.attn2, block.attn2.forward) for block in model.dit.blocks]
window = torch.load(
    "output_results/vsr/migration_20260920/single_reference/window/tile_00000.pt",
    map_location="cpu",
)
results = []
for index, mode in enumerate(
    ["baseline", "padding", "combined", "combined", "padding", "baseline"]
):
    for module, fn in originals:
        module.forward = fn
    if mode != "baseline":
        apply(model, "implicit_pad" if mode == "padding" else "combined")
    context = fused_postprocess() if mode == "combined" else nullcontext()
    with context:
        for _ in range(4):
            model.restore_window(window)
        for attempt in range(3):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = dict(torch._dynamo.utils.counters["stats"])
            started = time.perf_counter()
            frames = stream_restore(
                model,
                Path("../vsr/input/input.mp4"),
                out / f"{index}_{mode}.mp4",
                target_h=3840,
                target_w=2160,
                read_queue=2,
                show_progress=False,
                gpu_postprocess=True,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            after = dict(torch._dynamo.utils.counters["stats"])
            if before == after:
                break
            print(
                mode, "discard compilation sample", elapsed, before, after, flush=True
            )
        else:
            raise RuntimeError("Compiler failed to stabilize")
    record = {
        "mode": mode,
        "warm_video_s": elapsed,
        "frames": frames,
        "compile_before": before,
        "compile_after": after,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    results.append(record)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(record, flush=True)
