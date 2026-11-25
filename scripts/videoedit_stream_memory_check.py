"""Measure streaming pixel I/O memory without loading a generative model."""

import argparse
import json
import os
import threading
import time
import tracemalloc

import psutil

from sglang.multimodal_gen.configs.sample.videoedit_wan import (
    WanVideoEditSamplingParams,
)
from sglang.multimodal_gen.runtime.videoedit.streaming import run_streaming_edit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--frames", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    params = WanVideoEditSamplingParams(
        video_input_path=args.video,
        mask_input_path=args.mask,
        reference_image_path=args.reference,
        num_frames=args.frames,
    )
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = baseline
    stop = threading.Event()

    def sample():
        nonlocal peak
        while not stop.wait(0.05):
            peak = max(peak, process.memory_info().rss)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    tracemalloc.start()
    start = time.monotonic()
    try:
        metadata = run_streaming_edit(
            params, lambda frames, masks, spec, chunk: frames, args.output
        )
        _, traced_peak = tracemalloc.get_traced_memory()
    finally:
        stop.set()
        monitor.join()
        tracemalloc.stop()
    metadata.pop("frames")
    metadata.update(
        scope="pixel I/O only; identity model callback; no neural inference",
        elapsed_seconds=time.monotonic() - start,
        baseline_rss_bytes=baseline,
        peak_rss_bytes=peak,
        incremental_rss_bytes=peak - baseline,
        traced_peak_bytes=traced_peak,
    )
    path = os.path.splitext(args.output)[0] + ".memory.json"
    with open(path, "w") as handle:
        json.dump(metadata, handle, indent=2)
    print(
        json.dumps(
            {
                key: metadata[key]
                for key in (
                    "num_output_frames",
                    "elapsed_seconds",
                    "incremental_rss_bytes",
                    "traced_peak_bytes",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
