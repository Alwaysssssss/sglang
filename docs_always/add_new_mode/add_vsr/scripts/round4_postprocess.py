# SPDX-License-Identifier: Apache-2.0
"""GPU-only fused postprocessing experiment with unchanged correction semantics."""

from contextlib import contextmanager

import torch
from sglang.multimodal_gen.runtime.vsr import color, stream


def correct(out, ref_mean, ref_std, eps=1e-5):
    dims = (0, 2, 3, 4)
    # Same unbiased variance as the original; reduce mean and variance together.
    variance, mean = torch.var_mean(out, dim=dims, correction=1, keepdim=True)
    return (out - mean) / (variance.sqrt() + eps) * ref_std.to(out) + ref_mean.to(out)


def quantize(video):
    value = video.squeeze(0).permute(1, 2, 3, 0).float()
    value = (value * 0.5 + 0.5).clamp(0, 1)
    return (value * 255).clamp(0, 255).to(torch.uint8)


@contextmanager
def fused_postprocess():
    old_color = color.match_color_to_stats
    old_stream = stream.match_color_to_stats
    old_uint8 = stream.to_uint8_hwc
    compiled_color = torch.compile(correct, fullgraph=True, dynamic=True)
    compiled_uint8 = torch.compile(quantize, fullgraph=True, dynamic=True)
    color.match_color_to_stats = compiled_color
    stream.match_color_to_stats = compiled_color
    stream.to_uint8_hwc = lambda video: compiled_uint8(video).cpu().numpy()
    try:
        yield
    finally:
        color.match_color_to_stats = old_color
        stream.match_color_to_stats = old_stream
        stream.to_uint8_hwc = old_uint8
