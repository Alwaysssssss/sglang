# SPDX-License-Identifier: Apache-2.0
"""Measure one real tile in a fresh process per residency mode."""

import argparse
import json
import resource
import time
from pathlib import Path

import torch
from sglang.multimodal_gen.runtime.vsr.model import VSRRestorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--wan-root", required=True)
    parser.add_argument("--window", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode", choices=["resident", "whole", "layerwise"], required=True
    )
    parser.add_argument("--compile-vae", action="store_true")
    args = parser.parse_args()
    model = VSRRestorer.from_pretrained(
        args.checkpoint_dir,
        args.wan_root,
        vae_cpu_offload=args.mode != "resident",
        dit_cpu_offload=args.mode == "whole",
        dit_layerwise_offload=args.mode == "layerwise",
        compile_encoder=args.compile_vae,
        compile_decoder=args.compile_vae,
        cache_dit_condition=True,
    )
    window = torch.load(args.window, map_location="cpu", weights_only=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize()
    init_peak = torch.cuda.max_memory_allocated()
    times = []
    peak = 0
    for i in range(3):
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = model.restore_window(window)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        peak = max(peak, torch.cuda.max_memory_allocated())
        if i == 2:
            torch.save(result.cpu(), output_dir / f"{args.mode}.pt")
        del result
        print(f"{args.mode} iteration {i}: {times[-1]:.3f}s", flush=True)
    torch.cuda.synchronize()
    report = {
        "mode": args.mode,
        "compile_vae": args.compile_vae,
        "seconds": times,
        "gpu_init_peak_gib": init_peak / 2**30,
        "gpu_inference_peak_gib": peak / 2**30,
        "gpu_idle_allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "gpu_reserved_gib": torch.cuda.memory_reserved() / 2**30,
        "cpu_process_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 2**20,
    }
    (output_dir / f"{args.mode}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
