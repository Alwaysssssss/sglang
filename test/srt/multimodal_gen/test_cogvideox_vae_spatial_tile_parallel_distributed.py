# SPDX-License-Identifier: Apache-2.0
"""Direct torchrun coverage for CogVideoX VAE SP subgroup transport."""

import argparse
import os

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.distributed import (
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    destroy_model_parallel,
)
from sglang.multimodal_gen.runtime.models.vaes.cogvideox import (
    _all_gather_decoded_tiles,
)


TOPOLOGIES = {
    "sp2": (1, 2),
    "sp4": (1, 4),
    "cfg2_sp2": (2, 2),
}


def initialize_topology(
    topology: str, rank: int, local_rank: int, world_size: int
):
    cfg_degree, sp_degree = TOPOLOGIES[topology]
    assert world_size == cfg_degree * sp_degree
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", choices=tuple(TOPOLOGIES), required=True)
    return parser.parse_args()


def make_tile(index: int, marker: int, device: torch.device) -> torch.Tensor:
    shape = (1, 1, 1 + index % 2, 1 + index % 3, 1 + index)
    return torch.full(
        shape,
        float(marker + index),
        dtype=torch.float32,
        device=device,
    )


def main() -> int:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    model_parallel_initialized = False
    try:
        sp_group, _cfg_degree, sp_degree = initialize_topology(
            args.topology, rank, local_rank, world_size
        )
        model_parallel_initialized = True
        device = torch.device("cuda", local_rank)
        marker = (rank // sp_degree) * 10000
        local_tiles = {
            index: make_tile(index, marker, device)
            for index in range(7)
            if index % sp_degree == sp_group.rank_in_group
        }

        recovered = _all_gather_decoded_tiles(
            get_sp_group(),
            local_tiles,
            7,
            payload_dtype=torch.float32,
            payload_device=device,
        )

        assert tuple(recovered) == tuple(range(7))
        for index, tile in recovered.items():
            expected = make_tile(index, marker, device)
            assert tile.shape == expected.shape
            assert torch.equal(tile, expected)

        dist.barrier()
        if rank == 0:
            print(
                "PASS: CogVideoX VAE tile transport "
                f"topology={args.topology} subgroup isolation verified",
                flush=True,
            )
        return 0
    finally:
        if model_parallel_initialized:
            destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
