# SPDX-License-Identifier: Apache-2.0
"""Full-video warm benchmark of original VSR and all CLI optimizations."""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=["reference", "candidate"], required=True)
    parser.add_argument("--gpus", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.implementation == "reference" and args.gpus != 1:
        parser.error("The original reference is single-GPU")
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    import diffusers
    import torch

    root = Path(__file__).resolve().parents[4]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        checkpoint_dir="/mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/outputs/best/checkpoint-1300",
        wan_root="/mnt/shanhai-ai/liuh/Wan2.2-TI2V-5B-Diffusers",
        dtype=torch.bfloat16,
        tile_t=33, tile_h=320, tile_w=640, t_overlap=5, s_overlap=32,
    )
    optimized = args.implementation == "candidate"
    if optimized:
        from sglang.multimodal_gen.runtime.vsr.model import VSRRestorer as Model
        from sglang.multimodal_gen.runtime.vsr.stream import stream_restore

        kwargs.update(
            cudnn_benchmark=True, channels_last_3d=True,
            compile_encoder=True, compile_decoder=True,
            decoder_implicit_padding=True, cache_dit_condition=True,
        )
    else:
        sys.path.insert(0, str(root.parent / "vsr"))
        from infer.models.stage3 import Stage3Pipeline as Model
        from infer.stream import stream_restore

    def gpu_snapshot():
        return subprocess.check_output([
            "nvidia-smi", "-i", os.environ["CUDA_VISIBLE_DEVICES"],
            "--query-gpu=index,uuid,name,memory.used,utilization.gpu,clocks.sm,power.draw",
            "--format=csv",
        ], text=True)

    report = dict(
        implementation=args.implementation, gpus=args.gpus,
        visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
        diffusers=diffusers.__version__, cudnn=torch.backends.cudnn.version(),
        model_options={k: str(v) if k == "dtype" else v for k, v in kwargs.items()},
        input=str(root.parent / "vsr/input/input.mp4"),
        gpu_before=gpu_snapshot(), warmups=[], samples=[], discarded=[],
    )

    def save():
        path = args.output_dir / "results.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(path)

    save()
    with ExitStack() as stack:
        if args.gpus > 1:
            from sglang.multimodal_gen.runtime.vsr.parallel import ParallelVSRRestorer

            model = stack.enter_context(ParallelVSRRestorer(
                [f"cuda:{i}" for i in range(args.gpus)], **kwargs
            ))
        else:
            model = Model.from_pretrained(device="cuda:0", **kwargs)

        def compile_stats():
            if args.gpus > 1:
                return model.compile_stats()
            return [dict(torch._dynamo.utils.counters["stats"])]

        video_kwargs = dict(
            target_h=3840, target_w=2160, color_ref="global", color_samples=64,
            crf=5, read_queue=2, write_queue=4, show_progress=False,
        )
        if optimized:
            video_kwargs["gpu_postprocess"] = True
        report["video_options"] = video_kwargs

        def run(label):
            before = compile_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            frames = stream_restore(
                model, Path(report["input"]), args.output_dir / f"{label}.mp4",
                **video_kwargs,
            )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            record = dict(seconds=seconds, frames=frames,
                          compile_before=before, compile_after=compile_stats())
            print(label, json.dumps(record), flush=True)
            return record

        print("FULL_VIDEO_WARMUP", flush=True)
        report["warmups"].append(run("warmup"))
        save()
        torch.cuda.reset_peak_memory_stats()
        for attempt in range(args.repeats + 3):
            record = run(f"sample_{attempt}")
            record["output"] = str(args.output_dir / f"sample_{attempt}.mp4")
            if record["compile_before"] != record["compile_after"]:
                report["discarded"].append(record)
            else:
                report["samples"].append(record)
            save()
            if len(report["samples"]) == args.repeats:
                break
        if len(report["samples"]) != args.repeats:
            raise RuntimeError("Compilation did not stabilize")
        if len({s["frames"] for s in report["samples"] + report["warmups"]}) != 1:
            raise RuntimeError("Frame counts differ")
        report["median_seconds"] = statistics.median(s["seconds"] for s in report["samples"])
        report["owner_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["gpu_after"] = gpu_snapshot()
        report["complete"] = True
        save()
        print("COMPLETE", report["median_seconds"], flush=True)


if __name__ == "__main__":
    main()
