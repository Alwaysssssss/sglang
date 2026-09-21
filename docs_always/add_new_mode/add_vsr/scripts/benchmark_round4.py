# SPDX-License-Identifier: Apache-2.0
"""Diagnostic benchmark: one real training-size tile, no video IO or dumps."""

import argparse
import json
import os
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
parser.add_argument("--cudnn-benchmark", action="store_true")
parser.add_argument("--channels-last-3d", action="store_true")
parser.add_argument("--dump-output", help="Save final tensor after timed runs")
parser.add_argument("--compile-decoder", action="store_true")
parser.add_argument("--warmup", type=int, default=2)
parser.add_argument("--compile-encoder", action="store_true")
parser.add_argument("--compile-dit", action="store_true")
parser.add_argument(
    "--decoder-mode",
    default="default",
    choices=["default", "max-autotune-no-cudagraphs"],
)
parser.add_argument("--vae-fp16", action="store_true")
parser.add_argument(
    "--experiment",
    default="baseline",
    choices=[
        "baseline",
        "cat_once",
        "outer_compile",
        "condition",
        "cross_constant",
        "graph",
        "implicit_pad",
        "combined",
    ],
)
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
if args.implementation == "candidate":
    model.cudnn_benchmark = args.cudnn_benchmark
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
torch.backends.cudnn.benchmark = args.cudnn_benchmark
if args.channels_last_3d:
    torch.nn.utils.convert_conv3d_weight_memory_format(
        model.vae, torch.channels_last_3d
    )
if args.compile_decoder:
    from sglang.multimodal_gen.runtime.vsr.compile import compile_decoder

    compile_decoder(model.vae.vae)
if args.vae_fp16:
    model.vae.to(dtype=torch.float16)
    original_encode = model.vae.encode

    def mixed_encode(frames):
        return original_encode(frames).to(dtype=model.dtype)

    model.vae.encode = mixed_encode
if args.compile_encoder:
    model.vae.vae.encoder.forward = torch.compile(
        model.vae.vae.encoder.forward, fullgraph=True, dynamic=False
    )
if args.compile_dit:
    model.dit.forward = torch.compile(model.dit.forward, fullgraph=True, dynamic=False)
if args.compile_decoder and args.decoder_mode != "default":
    eager_decoder = model.vae.vae.decoder.forward._torchdynamo_orig_callable
    model.vae.vae.decoder.forward = torch.compile(
        eager_decoder, fullgraph=True, dynamic=False, mode=args.decoder_mode
    )
from round4_experiments import apply, capture

apply(model, args.experiment)
if args.experiment == "graph":
    capture(model, window)
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
warmup_s = []
for _ in range(args.warmup):
    started = time.perf_counter()
    result = forward(window)
    torch.cuda.synchronize()
    warmup_s.append(time.perf_counter() - started)
    print("WARMUP", len(warmup_s), warmup_s[-1], flush=True)
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

compile_stats_before = dict(torch._dynamo.utils.counters["stats"])
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

compile_stats_after = dict(torch._dynamo.utils.counters["stats"])
assert compile_stats_before == compile_stats_after, (
    "Compilation occurred during timed runs"
)
peak_allocated = torch.cuda.max_memory_allocated()
peak_reserved = torch.cuda.max_memory_reserved()
if args.dump_output:
    torch.save(forward(window).cpu(), args.dump_output)

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
    profiler.export_chrome_trace(str(Path(args.report).with_suffix(".trace.json")))
    ops = sorted(
        profiler.key_averages(), key=lambda e: e.self_device_time_total, reverse=True
    )
report = {
    "implementation": args.implementation,
    "experiment_args": vars(args),
    "compile_stats_before": compile_stats_before,
    "compile_stats_after": compile_stats_after,
    "seconds_per_tile": statistics.median(s["wall_s"] for s in samples),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "diffusers": diffusers.__version__,
    "python": sys.executable,
    "shape": list(window.shape),
    "dtype": str(model.dtype),
    "threads": torch.get_num_threads(),
    "channels_last_3d": args.channels_last_3d,
    "compile_decoder": args.compile_decoder,
    "compile_counters": {
        key: dict(value) for key, value in torch._dynamo.utils.counters.items()
    }
    if args.compile_decoder
    else {},
    "warmup_s": warmup_s,
    "peak_allocated_bytes": peak_allocated,
    "peak_reserved_bytes": peak_reserved,
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
            os.environ.get("CUDA_VISIBLE_DEVICES", "7"),
            "--query-gpu=uuid,name,memory.used,utilization.gpu,clocks.sm,power.draw",
            "--format=csv",
        ],
        text=True,
    ),
}
Path(args.report).write_text(json.dumps(report, indent=2))
print("MEDIAN", report["median"], flush=True)
