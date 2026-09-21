# SPDX-License-Identifier: Apache-2.0
"""Process-local IO experiments; original stream ordering is preserved."""

from contextlib import contextmanager

import torch
from sglang.multimodal_gen.runtime.vsr import stream


@contextmanager
def spatial_gpu(mode, device):
    if mode == "baseline":
        yield
        return
    original_tiles = stream.tiled_restore_rect
    original_reader = stream._reader_worker
    original_weight = stream.temporal_weight
    original_uint8 = stream.to_uint8_hwc
    postprocess = mode in {"postprocess", "postprocess_prefetch"}
    pending = {}
    copy_stream = (
        torch.cuda.Stream(device=device)
        if mode in {"prefetch", "postprocess_prefetch"}
        else None
    )

    class UploadQueue:
        def __init__(self, target):
            self.target = target

        def put(self, item):
            if isinstance(item, tuple):
                cpu = item[3]
                pinned = cpu.pin_memory()
                with torch.cuda.stream(copy_stream):
                    gpu = pinned.to(device, non_blocking=True)
                    ready = torch.cuda.Event()
                    ready.record()
                pending[id(cpu)] = (gpu, ready, pinned)
            self.target.put(item)

    def reader(*args):
        args = list(args)
        args[-2] = UploadQueue(args[-2])
        return original_reader(*args)

    def tiles(frames, *args, **kwargs):
        if copy_stream is not None:
            gpu, ready, _pinned = pending.pop(id(frames))
            torch.cuda.current_stream(device).wait_event(ready)
            gpu.record_stream(torch.cuda.current_stream(device))
        else:
            gpu = frames.to(device)
        # Crop/normalization and all spatial accumulation execute on GPU.
        # One blocking D2H per chunk leaves existing colour/temporal IO intact.
        result = original_tiles(gpu, *args, **kwargs)
        return result if postprocess else result.cpu()

    def weight(*args, **kwargs):
        return original_weight(*args, **kwargs).to(device)

    def uint8(video):
        value = video.squeeze(0).permute(1, 2, 3, 0).float()
        value = (value * 0.5 + 0.5).clamp(0, 1)
        return (value * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

    stream.tiled_restore_rect = tiles
    if postprocess:
        stream.temporal_weight = weight
        stream.to_uint8_hwc = uint8
    if copy_stream is not None:
        stream._reader_worker = reader
    try:
        yield
    finally:
        if copy_stream is not None:
            copy_stream.synchronize()
        pending.clear()
        stream.tiled_restore_rect = original_tiles
        stream._reader_worker = original_reader
        stream.temporal_weight = original_weight
        stream.to_uint8_hwc = original_uint8
