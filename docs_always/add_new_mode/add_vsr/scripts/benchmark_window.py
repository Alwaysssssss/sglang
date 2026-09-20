# SPDX-License-Identifier: Apache-2.0
"""Diagnostic benchmark: one real training-size tile, no video IO or dumps."""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    "--implementation", choices=["reference", "candidate"], required=True
)
parser.add_argument("--report", required=True)
parser.add_argument("--profile", action="store_true")
args = parser.parse_args()
root = Path.cwd()
sys.path.insert(0, str(root.parent / "vsr"))
if args.implementation == "reference":
    from infer.models.stage3 import Stage3Pipeline as Model
else:
    from sglang.multimodal_gen.runtime.vsr.model import VSRRestorer as Model
import diffusers

model = Model.from_pretrained(
    checkpoint_dir="/mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/outputs/best/checkpoint-1300",
    wan_root="/mnt/shanhai-ai/liuh/Wan2.2-TI2V-5B-Diffusers",
)
window = torch.load(
    root
    / "output_results/vsr/migration_20260920/single_reference/window/tile_00000.pt",
    map_location="cpu",
)
forward = (
    model._restore_window
    if args.implementation == "reference"
    else model.restore_window
)
for _ in range(2):
    result = forward(window)
    torch.cuda.synchronize()
    del result

stages = {}
for owner, attr, name in [
    (model.vae, "encode", "encode"),
    (model.dit, "forward", "dit"),
    (model.vae, "decode", "decode"),
]:
    original = getattr(owner, attr)

    def instrument(*a, _fn=original, _name=name, **kw):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        value = _fn(*a, **kw)
        end.record()
        stages[_name] = (start, end)
        return value

    setattr(owner, attr, instrument)

samples = []
for i in range(6):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = forward(window)
    torch.cuda.synchronize()
    sample = {
        "wall_s": time.perf_counter() - started,
        **{name + "_s": a.elapsed_time(b) / 1000 for name, (a, b) in stages.items()},
    }
    samples.append(sample)
    del result
    print(i, sample, flush=True)

ops = []
if args.profile:
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        result = forward(window)
        torch.cuda.synchronize()
    ops = sorted(
        profiler.key_averages(), key=lambda e: e.self_device_time_total, reverse=True
    )
report = {
    "implementation": args.implementation,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "diffusers": diffusers.__version__,
    "python": sys.executable,
    "shape": list(window.shape),
    "dtype": str(model.dtype),
    "threads": torch.get_num_threads(),
    "cudnn": torch.backends.cudnn.version(),
    "flags": {
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp": torch.backends.cuda.math_sdp_enabled(),
        "matmul_precision": torch.get_float32_matmul_precision(),
    },
    "samples": samples,
    "median": {key: statistics.median(s[key] for s in samples) for key in samples[0]},
    "top_cuda_ops": [
        {
            "name": e.key,
            "self_cuda_ms": e.self_device_time_total / 1000,
            "calls": e.count,
        }
        for e in ops[:20]
    ],
    "gpu": subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            "7",
            "--query-gpu=uuid,name,memory.used,utilization.gpu,clocks.sm,power.draw",
            "--format=csv",
        ],
        text=True,
    ),
}
Path(args.report).write_text(json.dumps(report, indent=2))
print("MEDIAN", report["median"], flush=True)
