"""NCCL microbenchmark; run with torchrun on explicitly selected idle GPUs.

Uses PyTorch SDPA to isolate communication scheduling. This is not an
end-to-end VideoEdit or production FlashAttention backend benchmark.
"""

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from test_overlap_pipeline import ROOT, usp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=1024, help="tokens per rank")
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--chunks", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--trace-dir", type=Path)
    args = parser.parse_args()
    if min(args.sequence_length, args.heads, args.head_dim, args.iterations, args.warmup) < 1:
        parser.error("shape, iteration and warmup values must be positive")
    if any(chunks < 2 for chunks in args.chunks):
        parser.error("chunk counts must be at least 2")
    if args.trace_dir:
        args.trace_dir = args.trace_dir.resolve()
        if not args.trace_dir.is_relative_to(ROOT):
            parser.error("traces must be written inside the repository")
        args.trace_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        world = dist.get_world_size()
        if args.heads % world:
            raise ValueError("head count must be divisible by world size")
        torch.manual_seed(42 + dist.get_rank())
        values = [torch.randn(1, args.sequence_length, args.heads, args.head_dim,
                              device="cuda", dtype=torch.bfloat16) for _ in range(3)]

        def attention(q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            ).transpose(1, 2)

        def separate():
            q, k, v = [usp._usp_input_all_to_all(x, 2) for x in values]
            return usp._usp_output_all_to_all(attention(q, k, v), 2)

        def fused():
            q, k, v = usp._usp_qkv_input_all_to_all(*values)
            return usp._usp_output_all_to_all(attention(q, k, v), 2)

        variants = [("separate", separate), ("fused", fused)]
        for chunks in args.chunks:
            if args.heads % (world * chunks) == 0:
                variants.append((f"chunks{chunks}", lambda chunks=chunks:
                    usp.ulysses_attention_pipelined(*values, attention, chunks)))
            elif dist.get_rank() == 0:
                print(json.dumps({"skipped_chunks": chunks, "reason": "indivisible heads"}))

        with torch.inference_mode():
            expected = separate()
            for name, run in variants:
                actual = run()
                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
                error = (actual.float() - expected.float()).abs().max()
                dist.all_reduce(error, op=dist.ReduceOp.MAX)
                max_error = error.item()
                del actual, error
                for _ in range(args.warmup):
                    run()
                torch.cuda.synchronize()
                # Release cached blocks between variants; never inside timed work.
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                times = []
                for _ in range(args.iterations):
                    dist.barrier()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    result = run()
                    end.record()
                    end.synchronize()
                    elapsed = torch.tensor(start.elapsed_time(end), device="cuda")
                    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                    times.append(elapsed.item())
                    del result, elapsed
                memory = torch.tensor([torch.cuda.max_memory_allocated(),
                                       torch.cuda.max_memory_reserved()], device="cuda")
                dist.all_reduce(memory, op=dist.ReduceOp.MAX)
                if dist.get_rank() == 0:
                    print(json.dumps({"variant": name, "world": world,
                        "shape_per_rank": list(values[0].shape), "median_ms": statistics.median(times),
                        "min_ms": min(times), "max_ms": max(times),
                        "max_rank_peak_allocated_bytes": memory[0].item(),
                        "max_rank_peak_reserved_bytes": memory[1].item(),
                        "max_abs_error": max_error}), flush=True)
                if args.trace_dir:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
                        run()
                        torch.cuda.synchronize()
                    prof.export_chrome_trace(str(args.trace_dir / f"{name}-rank{dist.get_rank()}.json"))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
