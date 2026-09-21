# SPDX-License-Identifier: Apache-2.0
"""Check fixed-condition output reuse on different inputs and warm timing."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from round4_experiments import fixed_condition
from sglang.multimodal_gen.runtime.vsr.condition_cache import cache_fixed_vsr_condition
from sglang.multimodal_gen.runtime.vsr.model import VSRRestorer

parser = argparse.ArgumentParser()
parser.add_argument("--production", action="store_true")
args = parser.parse_args()
out = Path("output_results/vsr/optimization_round4")
model = VSRRestorer.from_pretrained(
    checkpoint_dir="/mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/outputs/best/checkpoint-1300",
    wan_root="/mnt/shanhai-ai/liuh/Wan2.2-TI2V-5B-Diffusers",
    cudnn_benchmark=True,
    channels_last_3d=True,
    compile_decoder=True,
    compile_encoder=True,
    decoder_implicit_padding=True,
)
window = torch.load(
    "output_results/vsr/migration_20260920/single_reference/window/tile_00000.pt",
    map_location="cpu",
)
inputs = [window, window.flip(-1), window * 0.5, window]
for _ in range(4):
    model.restore_window(window)
references = [model.restore_window(x).cpu() for x in inputs]
modules = [model.dit.condition_embedder] + [block.attn2 for block in model.dit.blocks]
original = [m.forward for m in modules]
if args.production:
    cache_fixed_vsr_condition(model.dit)
else:
    fixed_condition(model, collapse=True, exact=True)
cached = [m.forward for m in modules]
checks = []
for x, ref in zip(inputs, references):
    actual = model.restore_window(x).cpu()
    checks.append(
        {
            "equal": torch.equal(actual, ref),
            "max_abs": (actual.float() - ref.float()).abs().max().item(),
        }
    )
print("OUTPUT_CHECKS", checks, flush=True)
results = []
for name, forwards in [
    ("baseline", original),
    ("cached", cached),
    ("cached", cached),
    ("baseline", original),
]:
    for m, fn in zip(modules, forwards):
        m.forward = fn
    for _ in range(3):
        model.restore_window(window)
    torch.cuda.synchronize()
    before = dict(torch._dynamo.utils.counters["stats"])
    times = []
    for _ in range(6):
        start = time.perf_counter()
        model.restore_window(window)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    assert before == dict(torch._dynamo.utils.counters["stats"])
    record = {"mode": name, "median_s": statistics.median(times), "samples": times}
    results.append(record)
    print(record, flush=True)
(
    out / ("constant_production.json" if args.production else "constant_exact.json")
).write_text(
    json.dumps({"different_input_checks": checks, "results": results}, indent=2)
)
assert all(c["equal"] for c in checks)
