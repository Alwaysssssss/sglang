"""Run with torchrun --standalone --nproc_per_node=2; no GPU allocations."""
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.layers import usp


def main():
    dist.init_process_group("gloo")
    try:
        with patch.object(usp, "get_ulysses_parallel_world_size", dist.get_world_size), \
             patch.object(usp, "get_sp_group", lambda: SimpleNamespace(ulysses_group=dist.group.WORLD)):
            for batch in (1, 2):
                for head_dim in (1, 2):
                    for kv_heads in (8, 4):
                        values = []
                        for i, heads in enumerate((8, kv_heads, kv_heads)):
                            x = torch.arange(batch * 5 * heads * 8, dtype=torch.float32).reshape(batch, 5, heads, 8)
                            x = x + (10000 * dist.get_rank() + i * 1000)
                            x = x[..., ::2]
                            if head_dim == 1:
                                x = x.transpose(1, 2)
                            values.append(x)
                        expected = [usp._usp_input_all_to_all(x, head_dim) for x in values]
                        with patch.object(usp, "_usp_all_to_all_single", wraps=usp._usp_all_to_all_single) as calls:
                            actual = usp._usp_qkv_input_all_to_all(*values, head_dim=head_dim)
                            assert calls.call_count == (1 if kv_heads == 8 else 3)
                        for a, b in zip(actual, expected):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
        print(f"rank {dist.get_rank()}: packed QKV matches separate exchanges, including GQA fallback", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
