# SPDX-License-Identifier: Apache-2.0
"""Validate CogVideoX VAE spatial tiled encode parallelism bitwise."""

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


def _error_metrics(left: torch.Tensor, right: torch.Tensor):
    if left.shape != right.shape:
        return None, None
    error = (left.float() - right.float()).abs()
    return float(error.max().item()), float(error.mean().item())


def compare_serial_and_parallel_encode(
    serial_moments: torch.Tensor,
    parallel_moments: torch.Tensor,
    serial_latents: torch.Tensor,
    parallel_latents: torch.Tensor,
) -> dict[str, Any]:
    moments_exact = serial_moments.shape == parallel_moments.shape and torch.equal(
        serial_moments, parallel_moments
    )
    sampled_latents_exact = (
        serial_latents.shape == parallel_latents.shape
        and torch.equal(serial_latents, parallel_latents)
    )
    moments_max, moments_mean = _error_metrics(serial_moments, parallel_moments)
    latents_max, latents_mean = _error_metrics(serial_latents, parallel_latents)
    return {
        "moments_exact": bool(moments_exact),
        "sampled_latents_exact": bool(sampled_latents_exact),
        "passed": bool(moments_exact and sampled_latents_exact),
        "serial_moments_shape": list(serial_moments.shape),
        "parallel_moments_shape": list(parallel_moments.shape),
        "serial_latents_shape": list(serial_latents.shape),
        "parallel_latents_shape": list(parallel_latents.shape),
        "moments_max_abs_error": moments_max,
        "moments_mean_abs_error": moments_mean,
        "sampled_latents_max_abs_error": latents_max,
        "sampled_latents_mean_abs_error": latents_mean,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate serial and SP CogVideoX tiled encode outputs."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/home/zhiheng/ckpts/CogVideoX1.5-5B"),
    )
    parser.add_argument("--topology", choices=tuple(TOPOLOGIES), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-frames", type=int, default=17)
    parser.add_argument("--sample-height", type=int, default=720)
    parser.add_argument("--sample-width", type=int, default=960)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def initialize_topology(topology: str, rank: int, local_rank: int, world_size: int):
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


def timed_encode(vae, sample: torch.Tensor):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    encoded = vae.encode(sample).latent_dist
    end.record()
    return encoded, start, end


def sample_latents(distribution, *, device: torch.device, seed: int):
    generator = torch.Generator(device=device).manual_seed(seed)
    return distribution.sample(generator=generator)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def validate_subgroup_results(
    rank_payloads: list[dict[str, Any]], cfg_degree: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for item in rank_payloads:
        groups.setdefault(tuple(item["sp_subgroup_ranks"]), []).append(item)
    checks = []
    for group_ranks, items in sorted(groups.items()):
        input_seeds = {item["input_seed"] for item in items}
        root_input_sums = {item["root_input_sum"] for item in items}
        checks.append(
            {
                "sp_subgroup_ranks": list(group_ranks),
                "same_seed_within_subgroup": len(input_seeds) == 1,
                "same_root_input_within_subgroup": len(root_input_sums) == 1,
                "rank_divergent_inputs_detected": len(
                    {item["rank_divergent_input_sum"] for item in items}
                )
                == len(items),
                "passed": (
                    len(input_seeds) == 1
                    and len(root_input_sums) == 1
                    and all(item["passed"] for item in items)
                    and all(item["rank_divergent_passed"] for item in items)
                ),
            }
        )
    if len(checks) != cfg_degree:
        for item in checks:
            item["passed"] = False
    return checks


def main() -> int:
    process_start = time.perf_counter()
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for VAE spatial encode validation")
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
        vae.enable_tiling(
            tile_sample_min_height=240,
            tile_sample_min_width=360,
            tile_overlap_factor_height=1 / 6,
            tile_overlap_factor_width=1 / 5,
        )
        torch.cuda.reset_peak_memory_stats(device)

        cfg_group_index = rank // sp_degree
        input_seed = args.seed + cfg_group_index
        generator = torch.Generator(device=device).manual_seed(input_seed)
        sample_storage = torch.randn(
            (
                1,
                int(vae.config.in_channels),
                args.sample_frames,
                args.sample_width,
                args.sample_height,
            ),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        sample = sample_storage.transpose(-1, -2)
        latent_seed = args.seed + 1000 + cfg_group_index

        with torch.inference_mode():
            vae.configure_spatial_tile_encode_parallel(False)
            serial_dist, serial_start, serial_end = timed_encode(vae, sample)
            serial_moments = serial_dist.parameters
            serial_latents = sample_latents(
                serial_dist, device=device, seed=latent_seed
            )

            vae.configure_spatial_tile_encode_parallel(True)
            parallel_dist, parallel_start, parallel_end = timed_encode(vae, sample)
            parallel_moments = parallel_dist.parameters
            parallel_latents = sample_latents(
                parallel_dist, device=device, seed=latent_seed
            )

            rank_divergent_sample = sample + float(sp_group.rank_in_group)
            divergent_dist, divergent_start, divergent_end = timed_encode(
                vae, rank_divergent_sample
            )
            divergent_moments = divergent_dist.parameters
            divergent_latents = sample_latents(
                divergent_dist, device=device, seed=latent_seed
            )

        torch.cuda.synchronize(device)
        model_seconds = (
            sum(
                start.elapsed_time(end)
                for start, end in (
                    (serial_start, serial_end),
                    (parallel_start, parallel_end),
                    (divergent_start, divergent_end),
                )
            )
            / 1000.0
        )
        comparison = compare_serial_and_parallel_encode(
            serial_moments,
            parallel_moments,
            serial_latents,
            parallel_latents,
        )
        divergent_comparison = compare_serial_and_parallel_encode(
            serial_moments,
            divergent_moments,
            serial_latents,
            divergent_latents,
        )
        vae_stats = vae.get_last_spatial_encode_stats().to_debug_dict()
        rank_payload = {
            "rank": rank,
            "local_rank": local_rank,
            "cfg_group_index": cfg_group_index,
            "sp_subgroup_ranks": list(sp_group.ranks),
            "input_seed": input_seed,
            "latent_seed": latent_seed,
            "input_shape": list(sample.shape),
            "input_noncontiguous": not sample.is_contiguous(),
            "root_input_sum": float(sample.float().sum().item()),
            "rank_divergent_input_sum": float(
                rank_divergent_sample.float().sum().item()
            ),
            "rank_divergent_passed": divergent_comparison["passed"],
            "rank_divergent_comparison": divergent_comparison,
            "model_inference_runtime_seconds": model_seconds,
            "total_runtime_seconds_before_gather": time.perf_counter() - process_start,
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            **comparison,
            **vae_stats,
        }
        gathered: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(gathered, rank_payload)
        rank_payloads = [item for item in gathered if item is not None]
        subgroup_checks = validate_subgroup_results(rank_payloads, cfg_degree)
        overall_pass = (
            all(item["passed"] for item in rank_payloads)
            and all(item["rank_divergent_passed"] for item in rank_payloads)
            and all(item["input_noncontiguous"] for item in rank_payloads)
            and all(item["passed"] for item in subgroup_checks)
        )

        if rank == 0:
            payload = {
                "schema_version": 1,
                "topology": args.topology,
                "cfg_degree": cfg_degree,
                "sp_degree": sp_degree,
                "world_size": world_size,
                "sp_subgroups": sorted(
                    {tuple(item["sp_subgroup_ranks"]) for item in rank_payloads}
                ),
                "input_shape": list(sample.shape),
                "moments_shape": list(serial_moments.shape),
                "latents_shape": list(serial_latents.shape),
                "input_dtype": str(sample.dtype),
                "noncontiguous_inputs_exercised": all(
                    item["input_noncontiguous"] for item in rank_payloads
                ),
                "seed": args.seed,
                "ranks": rank_payloads,
                "subgroup_checks": subgroup_checks,
                "rank_pass": {
                    str(item["rank"]): item["passed"] for item in rank_payloads
                },
                "rank_divergent_pass": {
                    str(item["rank"]): item["rank_divergent_passed"]
                    for item in rank_payloads
                },
                "overall_pass": overall_pass,
                "total_runtime_seconds": time.perf_counter() - process_start,
                "model_inference_runtime_seconds": max(
                    item["model_inference_runtime_seconds"] for item in rank_payloads
                ),
                "timing_scope": {
                    "total_runtime_seconds": (
                        "rank 0 process main entry through immediately before "
                        "atomic result write"
                    ),
                    "model_inference_runtime_seconds": (
                        "maximum per-rank CUDA-event sum of the three "
                        "vae.encode calls"
                    ),
                },
            }
            atomic_write_json(args.output_json, payload)
            print(
                f"{'PASS' if overall_pass else 'FAIL'}: "
                f"VAE spatial encode topology={args.topology} "
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
