import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

import sglang.multimodal_gen.runtime.layers.usp as usp


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2
    device = torch.device("cuda", local_rank)

    usp.get_ulysses_parallel_world_size = lambda: world_size
    usp.get_sp_group = lambda: SimpleNamespace(ulysses_group=dist.group.WORLD)

    base = torch.arange(
        1 * 7 * 8 * 16,
        device=device,
        dtype=torch.bfloat16,
    ).reshape(1, 7, 8, 16)
    q = base + rank * 10000
    k = base + rank * 10000 + 1000
    v = base + rank * 10000 + 2000

    legacy = tuple(usp._usp_input_all_to_all(x, head_dim=2) for x in (q, k, v))
    packed = usp._usp_input_all_to_all_qkv(q, k, v)
    for actual, expected in zip(packed, legacy, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    out_rep = packed[0][:, :3].contiguous()
    gathered_list = [torch.empty_like(out_rep) for _ in range(world_size)]
    dist.all_gather(gathered_list, out_rep, group=dist.group.WORLD)
    expected_prefix = torch.cat(gathered_list, dim=2)
    actual_prefix = usp._usp_prefix_all_gather(out_rep)
    torch.testing.assert_close(actual_prefix, expected_prefix, rtol=0, atol=0)

    dist.barrier()
    if rank == 0:
        print("PASS: packed QKV A2A and prefix gather are bitwise exact")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
