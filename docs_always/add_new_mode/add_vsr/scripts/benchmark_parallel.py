# SPDX-License-Identifier: Apache-2.0
"""Warmed, matched single/dual GPU full-video comparison."""

import json
import time
from pathlib import Path

import torch
from sglang.multimodal_gen.runtime.vsr.parallel import ParallelVSRRestorer
from sglang.multimodal_gen.runtime.vsr.stream import stream_restore


def main():
    out = Path("output_results/vsr/parallel_round1")
    out.mkdir(parents=True, exist_ok=True)
    with ParallelVSRRestorer(
        ["cuda:0", "cuda:1"],
        checkpoint_dir="/mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/outputs/best/checkpoint-1300",
        wan_root="/mnt/shanhai-ai/liuh/Wan2.2-TI2V-5B-Diffusers",
        cudnn_benchmark=True,
        channels_last_3d=True,
        compile_encoder=True,
        compile_decoder=True,
        decoder_implicit_padding=True,
    ) as model:
        window = torch.load(
            "output_results/vsr/migration_20260920/single_reference/window/tile_00000.pt",
            map_location="cpu",
        )
        model.warmup(window, iterations=6)
        # Compare identical inputs on both replicas before full-video timing.
        outputs = list(model.restore_windows([window, window]))
        difference = (outputs[0].float() - outputs[1].float()).abs()
        print(
            "REPLICA_DIFF",
            difference.max().item(),
            difference.mean().item(),
            flush=True,
        )
        torch.save([x.cpu() for x in outputs], out / "replica_outputs.pt")
        del outputs, difference
        records = []
        for index, mode in enumerate(["single", "dual", "dual", "single"]):
            restorer = model.local if mode == "single" else model
            before = model.compile_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            frames = stream_restore(
                restorer,
                Path("../vsr/input/input.mp4"),
                out / f"{index}_{mode}.mp4",
                target_h=3840,
                target_w=2160,
                gpu_postprocess=True,
                read_queue=2,
                show_progress=False,
            )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            after = model.compile_stats()
            if before != after:
                raise RuntimeError(f"Timed compilation: {before} -> {after}")
            record = {
                "mode": mode,
                "seconds": seconds,
                "frames": frames,
                "compile_before": before,
                "compile_after": after,
            }
            records.append(record)
            (out / "results.json").write_text(json.dumps(records, indent=2))
            print("RESULT", record, flush=True)


if __name__ == "__main__":
    main()
