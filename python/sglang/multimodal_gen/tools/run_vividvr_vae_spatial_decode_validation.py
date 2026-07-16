# SPDX-License-Identifier: Apache-2.0
"""Validate CogVideoX VAE spatial tile parallel decode with fixed latents."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from sglang.multimodal_gen.configs.pipeline_configs.vividvr import (
    VividVRPipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    destroy_model_parallel,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    PipelineComponentLoader,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


TOPOLOGIES = {
    "sp2": (1, 2),
    "sp4": (1, 4),
    "cfg2_sp2": (2, 2),
}


def compare_serial_and_parallel_decode(
    serial_a: torch.Tensor,
    serial_b: torch.Tensor,
    parallel: torch.Tensor,
) -> dict[str, Any]:
    serial_shape_match = serial_a.shape == serial_b.shape
    parallel_shape_match = serial_a.shape == parallel.shape
    shape_match = serial_shape_match and parallel_shape_match
    deterministic = serial_shape_match and torch.equal(serial_a, serial_b)
    parallel_exact = parallel_shape_match and torch.equal(serial_a, parallel)

    serial_repeat_max = None
    serial_repeat_mean = None
    parallel_max = None
    parallel_mean = None
    passed = False
    if shape_match:
        serial_repeat = (serial_a.float() - serial_b.float()).abs()
        parallel_error = (serial_a.float() - parallel.float()).abs()
        serial_repeat_max = float(serial_repeat.max().item())
        serial_repeat_mean = float(serial_repeat.mean().item())
        parallel_max = float(parallel_error.max().item())
        parallel_mean = float(parallel_error.mean().item())
        passed = (
            parallel_exact
            if deterministic
            else (
                parallel_error.max() <= serial_repeat.max()
                and parallel_error.mean() <= serial_repeat.mean()
            )
        )

    return {
        "serial_a_shape": list(serial_a.shape),
        "serial_b_shape": list(serial_b.shape),
        "parallel_shape": list(parallel.shape),
        "shape_match": shape_match,
        "serial_deterministic": bool(deterministic),
        "parallel_exact": bool(parallel_exact),
        "serial_repeat_max_abs_error": serial_repeat_max,
        "serial_repeat_mean_abs_error": serial_repeat_mean,
        "parallel_max_abs_error": parallel_max,
        "parallel_mean_abs_error": parallel_mean,
        "passed": bool(passed),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate serial and VAE-SP CogVideoX tiled decode outputs."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/home/zhiheng/ckpts/CogVideoX1.5-5B"),
    )
    parser.add_argument("--topology", choices=tuple(TOPOLOGIES), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-frames", type=int, default=5)
    parser.add_argument("--latent-height", type=int, default=65)
    parser.add_argument("--latent-width", type=int, default=97)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def initialize_topology(
    topology: str, rank: int, local_rank: int, world_size: int
):
    cfg_degree, sp_degree = TOPOLOGIES[topology]
    if world_size != cfg_degree * sp_degree:
        raise ValueError(
            f"topology {topology} requires {cfg_degree * sp_degree} ranks, "
            f"got {world_size}"
        )
    torch.cuda.set_device(local_rank)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        backend="nccl",
        device_id=torch.device("cuda", local_rank),
    )
    initialize_model_parallel(
        classifier_free_guidance_degree=cfg_degree,
        sequence_parallel_degree=sp_degree,
        ulysses_degree=sp_degree,
        ring_degree=1,
    )
    return get_sp_group(), cfg_degree, sp_degree


def build_server_args(
    model_path: Path, *, world_size: int, cfg_degree: int, sp_degree: int
) -> ServerArgs:
    return ServerArgs(
        model_path=str(model_path),
        pipeline_class_name="CogVideoXVividVRControlNetPipeline",
        pipeline_config=VividVRPipelineConfig(),
        num_gpus=world_size,
        tp_size=1,
        dp_size=1,
        dp_degree=1,
        sp_degree=sp_degree,
        ulysses_degree=sp_degree,
        ring_degree=1,
        enable_cfg_parallel=cfg_degree > 1,
        vae_cpu_offload=False,
    )


def timed_decode(vae, latent: torch.Tensor):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    sample = vae.decode(latent).sample
    end.record()
    return sample, start, end


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def maximum_rank_metric(
    rank_payloads: list[dict[str, Any]], key: str
) -> float | None:
    values = [item[key] for item in rank_payloads]
    if not all(isinstance(value, (int, float)) for value in values):
        return None
    return max(float(value) for value in values)


def validate_subgroup_results(
    rank_payloads: list[dict[str, Any]], cfg_degree: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for item in rank_payloads:
        group_ranks = tuple(item["sp_subgroup_ranks"])
        groups.setdefault(group_ranks, []).append(item)

    checks = []
    for group_ranks, items in sorted(groups.items()):
        seeds = {item["latent_seed"] for item in items}
        latent_sums = {item["latent_sum"] for item in items}
        serial_sums = {item["serial_output_sum"] for item in items}
        parallel_sums = {item["parallel_output_sum"] for item in items}
        divergent_latent_sums = {
            item["rank_divergent_latent_sum"] for item in items
        }
        divergent_parallel_sums = {
            item["rank_divergent_parallel_output_sum"] for item in items
        }
        checks.append(
            {
                "sp_subgroup_ranks": list(group_ranks),
                "latent_seed": next(iter(seeds)) if len(seeds) == 1 else None,
                "same_latent_within_subgroup": len(latent_sums) == 1,
                "same_serial_output_within_subgroup": len(serial_sums) == 1,
                "same_parallel_output_within_subgroup": len(parallel_sums) == 1,
                "rank_divergent_inputs_detected": len(divergent_latent_sums)
                == len(items),
                "same_canonicalized_output_within_subgroup": len(
                    divergent_parallel_sums
                )
                == 1,
                "passed": (
                    len(seeds) == 1
                    and len(latent_sums) == 1
                    and len(serial_sums) == 1
                    and len(parallel_sums) == 1
                    and len(divergent_latent_sums) == len(items)
                    and len(divergent_parallel_sums) == 1
                    and all(
                        item["rank_divergent_input_comparison"]["passed"]
                        for item in items
                    )
                ),
            }
        )
    distinct_seeds = {item["latent_seed"] for item in checks}
    if len(checks) != cfg_degree or len(distinct_seeds) != cfg_degree:
        for item in checks:
            item["passed"] = False
    return checks


def main() -> int:
    process_start = time.perf_counter()
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for VAE spatial decode validation")
    if not (args.model_path / "vae").is_dir():
        raise SystemExit(f"VAE directory not found: {args.model_path / 'vae'}")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    model_parallel_initialized = False
    vae = None
    try:
        sp_group, cfg_degree, sp_degree = initialize_topology(
            args.topology, rank, local_rank, world_size
        )
        model_parallel_initialized = True
        device = torch.device("cuda", local_rank)
        server_args = build_server_args(
            args.model_path,
            world_size=world_size,
            cfg_degree=cfg_degree,
            sp_degree=sp_degree,
        )
        vae, _memory_usage = PipelineComponentLoader.load_component(
            "vae", str(args.model_path / "vae"), "diffusers", server_args
        )
        vae.eval()
        torch.cuda.reset_peak_memory_stats(device)

        cfg_group_index = rank // sp_degree
        latent_seed = args.seed + cfg_group_index
        latent_channels = int(vae.config.latent_channels)
        generator = torch.Generator(device=device).manual_seed(latent_seed)
        latent = torch.randn(
            (
                1,
                latent_channels,
                args.latent_frames,
                args.latent_height,
                args.latent_width,
            ),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )

        with torch.inference_mode():
            vae.configure_spatial_tile_parallel(requested=False)
            serial_a, serial_a_start, serial_a_end = timed_decode(vae, latent)
            serial_b, serial_b_start, serial_b_end = timed_decode(vae, latent)
            vae.configure_spatial_tile_parallel(requested=True)
            parallel, parallel_start, parallel_end = timed_decode(vae, latent)
            rank_divergent_latent = latent + float(sp_group.rank_in_group)
            rank_divergent_parallel, divergent_start, divergent_end = timed_decode(
                vae, rank_divergent_latent
            )

        torch.cuda.synchronize(device)
        model_seconds = sum(
            start.elapsed_time(end)
            for start, end in (
                (serial_a_start, serial_a_end),
                (serial_b_start, serial_b_end),
                (parallel_start, parallel_end),
                (divergent_start, divergent_end),
            )
        ) / 1000.0
        comparison = compare_serial_and_parallel_decode(
            serial_a, serial_b, parallel
        )
        rank_divergent_comparison = compare_serial_and_parallel_decode(
            serial_a, serial_b, rank_divergent_parallel
        )
        vae_stats = vae.get_last_spatial_decode_stats().to_debug_dict()
        rank_payload = {
            "rank": rank,
            "local_rank": local_rank,
            "cfg_group_index": cfg_group_index,
            "sp_subgroup_ranks": list(sp_group.ranks),
            "latent_seed": latent_seed,
            "latent_sum": float(latent.float().sum().item()),
            "serial_output_sum": float(serial_a.float().sum().item()),
            "parallel_output_sum": float(parallel.float().sum().item()),
            "rank_divergent_latent_sum": float(
                rank_divergent_latent.float().sum().item()
            ),
            "rank_divergent_parallel_output_sum": float(
                rank_divergent_parallel.float().sum().item()
            ),
            "rank_divergent_input_comparison": rank_divergent_comparison,
            "model_inference_runtime_seconds": model_seconds,
            "total_runtime_seconds_before_gather": time.perf_counter()
            - process_start,
            "peak_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            **comparison,
            **vae_stats,
        }
        gathered: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(gathered, rank_payload)
        rank_payloads = [item for item in gathered if item is not None]
        subgroup_checks = validate_subgroup_results(rank_payloads, cfg_degree)
        overall_pass = all(
            item["passed"]
            and item["rank_divergent_input_comparison"]["passed"]
            for item in rank_payloads
        ) and all(item["passed"] for item in subgroup_checks)

        if rank == 0:
            subgroup_seeds = {
                str(item["cfg_group_index"]): item["latent_seed"]
                for item in rank_payloads
            }
            payload = {
                "schema_version": 1,
                "topology": args.topology,
                "cfg_degree": cfg_degree,
                "sp_degree": sp_degree,
                "world_size": world_size,
                "sp_subgroups": sorted(
                    {tuple(item["sp_subgroup_ranks"]) for item in rank_payloads}
                ),
                "latent_shape": [
                    1,
                    latent_channels,
                    args.latent_frames,
                    args.latent_height,
                    args.latent_width,
                ],
                "latent_dtype": "torch.bfloat16",
                "seed": args.seed,
                "cfg_subgroup_seeds": subgroup_seeds,
                "serial_deterministic": all(
                    item["serial_deterministic"] for item in rank_payloads
                ),
                "serial_repeat_max_abs_error": maximum_rank_metric(
                    rank_payloads, "serial_repeat_max_abs_error"
                ),
                "serial_repeat_mean_abs_error": maximum_rank_metric(
                    rank_payloads, "serial_repeat_mean_abs_error"
                ),
                "parallel_max_abs_error": maximum_rank_metric(
                    rank_payloads, "parallel_max_abs_error"
                ),
                "parallel_mean_abs_error": maximum_rank_metric(
                    rank_payloads, "parallel_mean_abs_error"
                ),
                "ranks": rank_payloads,
                "subgroup_checks": subgroup_checks,
                "rank_pass": {
                    str(item["rank"]): item["passed"] for item in rank_payloads
                },
                "overall_pass": overall_pass,
                "total_runtime_seconds": time.perf_counter() - process_start,
                "model_inference_runtime_seconds": max(
                    item["model_inference_runtime_seconds"]
                    for item in rank_payloads
                ),
                "timing_scope": {
                    "total_runtime_seconds": (
                        "rank 0 process main entry through immediately before "
                        "atomic result write"
                    ),
                    "model_inference_runtime_seconds": (
                        "maximum per-rank CUDA-event sum of the four "
                        "vae.decode calls"
                    ),
                },
            }
            atomic_write_json(args.output_json, payload)
            print(
                f"{'PASS' if payload['overall_pass'] else 'FAIL'}: "
                f"VAE spatial decode topology={args.topology} "
                f"output={args.output_json}",
                flush=True,
            )
        dist.barrier()
        return 0 if overall_pass else 1
    finally:
        if vae is not None:
            del vae
        if model_parallel_initialized:
            destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
