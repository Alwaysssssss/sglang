# SPDX-License-Identifier: Apache-2.0
"""Isolate full-resolution GPU correction/quantization; excludes model and video IO."""

import json
import statistics
import time
from pathlib import Path

import torch
from round4_postprocess import correct, quantize
from sglang.multimodal_gen.runtime.vsr.color import match_color_to_stats

folder = Path("output_results/vsr/optimization_round4")
x = torch.load(folder / "baseline.pt", map_location="cpu").cuda().float()
x = (
    torch.nn.functional.interpolate(
        x.squeeze(0).permute(1, 0, 2, 3),
        size=(3840, 2160),
        mode="bilinear",
        align_corners=False,
    )
    .permute(1, 0, 2, 3)
    .unsqueeze(0)
)
mean = torch.tensor([0.01, -0.16, -0.30], device="cuda").view(1, 3, 1, 1, 1)
std = torch.tensor([0.53, 0.51, 0.49], device="cuda").view(1, 3, 1, 1, 1)


def original(x, mean, std):
    return quantize(match_color_to_stats(x, mean, std))


def candidate(x, mean, std):
    return quantize(correct(x, mean, std))


compiled = torch.compile(candidate, fullgraph=True, dynamic=True)
results = []
outputs = {}
for name, fn in [
    ("baseline", original),
    ("fused", compiled),
    ("fused", compiled),
    ("baseline", original),
]:
    for _ in range(3):
        y = fn(x, mean, std)
    torch.cuda.synchronize()
    before = dict(torch._dynamo.utils.counters["stats"])
    timings = []
    for _ in range(6):
        start = time.perf_counter()
        y = fn(x, mean, std)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    assert before == dict(torch._dynamo.utils.counters["stats"])
    outputs[name] = y
    record = {"mode": name, "median_s": statistics.median(timings), "samples": timings}
    results.append(record)
    print(record, flush=True)
diff = outputs["baseline"].float() - outputs["fused"].float()
quality = {
    "max_abs": diff.abs().max().item(),
    "mse": diff.square().mean().item(),
    "mae": diff.abs().mean().item(),
    "mismatch_ratio": (diff != 0).float().mean().item(),
}
(folder / "post_micro.json").write_text(
    json.dumps({"results": results, "quality_vs_baseline": quality}, indent=2)
)
print(quality, flush=True)
