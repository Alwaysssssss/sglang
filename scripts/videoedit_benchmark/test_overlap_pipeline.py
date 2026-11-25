"""Standalone tests requiring only PyTorch, not the serving dependencies.

CPU: python -m unittest scripts.videoedit_benchmark.test_overlap_pipeline -v
GPU: torchrun --standalone --nproc_per_node=2 \
     scripts/videoedit_benchmark/test_overlap_pipeline.py --backend nccl
The distributed entrypoint also supports --backend gloo for CPU validation.
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "python/sglang/multimodal_gen/runtime"


def load_module(name, relative, dependencies):
    spec = importlib.util.spec_from_file_location(name, RUNTIME / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = {key: sys.modules.get(key) for key in dependencies}
    sys.modules.update(dependencies)
    try:
        spec.loader.exec_module(module)
    finally:
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module


memory = load_module("memory_under_test", "utils/videoedit_memory.py", {})
offload = load_module("offload_under_test", "utils/layerwise_offload.py", {
    "sglang.multimodal_gen.runtime.utils.videoedit_memory": memory,
    "sglang.multimodal_gen.runtime.platforms": SimpleNamespace(
        current_platform=SimpleNamespace(device_type="cuda")),
    "sglang.multimodal_gen.runtime.server_args": SimpleNamespace(ServerArgs=object),
    "sglang.multimodal_gen.runtime.utils.logging_utils": SimpleNamespace(
        init_logger=logging.getLogger),
})
usp = load_module("usp_under_test", "layers/usp.py", {
    "sglang.multimodal_gen.runtime.distributed.parallel_state": SimpleNamespace(
        get_sp_group=lambda: SimpleNamespace(ulysses_group=dist.group.WORLD),
        get_ulysses_parallel_world_size=dist.get_world_size),
    "sglang.srt.utils.common": SimpleNamespace(torch_release=(2, 6)),
})


class Event:
    def record(self, stream):
        self.recorded = True


class Stream:
    def wait_stream(self, stream):
        pass

    def wait_event(self, event):
        assert event.recorded

    def synchronize(self):
        pass


class Model(torch.nn.Module):
    def __init__(self, count=9, strided=False):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            torch.nn.Linear(16, 16) for _ in range(count)
        ])
        if strided:
            for block in self.blocks:
                block.weight.data = block.weight.data.t()

    def forward(self, x):
        for block in self.blocks:
            x = block(x).tanh()
        return x


class SchedulingTests(unittest.TestCase):
    def test_rolling_window_reuses_buffers_and_reloads_after_release(self):
        # CPU arithmetic with simulated streams exercises the real manager,
        # including parameter rebinding, transfer counts and storage reuse.
        # It does not establish CUDA event/allocator lifetime correctness.
        for prefetch in (1, 2, 4, 9):
            for resident in (0, 3, 9):
                for strided in (False, True):
                    with self.subTest(prefetch=prefetch, resident=resident, strided=strided):
                        self.check_manager(prefetch, resident, strided)

    def check_manager(self, prefetch, resident, strided, budget=None):
        torch.manual_seed(123)
        reference = Model(strided=strided).eval()
        model = Model(strided=strided).eval()
        model.load_state_dict(reference.state_dict())
        stream = Stream()
        device = SimpleNamespace(
            is_available=lambda: True, current_device=lambda: 0,
            Stream=Stream, Event=Event, current_stream=lambda: stream,
            stream=lambda unused: nullcontext(),
        )
        with patch.dict(os.environ, SGLANG_VIDEOEDIT_OVERLAP_PREFETCH="1",
                        SGLANG_VIDEOEDIT_RESIDENT_LAYERS=str(resident)), \
             patch.object(torch, "get_device_module", return_value=device), \
             patch.object(torch.Tensor, "record_stream", return_value=None), \
             patch.object(offload.current_platform, "device_type", "cpu"):
            # Enable the CUDA-only policy while using CPU tensors for this test.
            with patch.object(offload.LayerwiseOffloadManager, "_initialize"):
                manager = offload.LayerwiseOffloadManager(
                    model, layers_attr_str="blocks", num_layers=9,
                    enabled=True, prefetch_size=prefetch, pin_cpu_memory=False,
                )
            manager.overlap_prefetch = True
            manager._initialize()
            if budget is not None:
                manager.configure_memory_budget(budget)
                prefetch, resident = manager.prefetch_size, manager.resident_layers
            pointers = set()
            hooks = []
            for index, block in enumerate(model.blocks):
                def inspect(module, args, index=index):
                    self.assertIn(index, manager._gpu_layers)
                    pointers.update(t.data_ptr() for buffers in manager._gpu_buffers.values()
                                    for t in buffers.values())
                    self.assertLessEqual(len(manager._gpu_layers), resident + prefetch + 1)
                hooks.append(block.register_forward_pre_hook(inspect))
            with torch.inference_mode():
                x = torch.randn(8, 16)
                expected = reference(x)
                for iteration in range(3):
                    before = manager.h2d_layer_loads
                    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
                    if iteration:
                        self.assertEqual(manager.h2d_layer_loads - before, 9 - resident)
                    self.assertEqual(manager._gpu_layers, set(range(resident)))
                self.assertLessEqual(len(pointers), resident + prefetch + 1)
                manager.release_all()
                self.assertFalse(manager._gpu_buffers)
                self.assertFalse(manager._retired_gpu_buffers)
                pointers.clear()
                torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
                manager.release_all()
            for hook in hooks:
                hook.remove()
            manager.remove_forward_hooks()


class PipelineLayoutTests(unittest.TestCase):
    def test_all_heads_and_sequence_positions_match_global_attention(self):
        # In-process all-to-all transport permits layout/numeric validation
        # without sockets. Real NCCL overlap is tested by the entrypoint below.
        torch.set_num_threads(1)
        for world in (2, 4):
            for batch in (1, 2):
                for chunks in (2, 4):
                    for causal in (False, True):
                        with self.subTest(world=world, batch=batch, chunks=chunks, causal=causal):
                            self.check_pipeline(world, batch, chunks, causal)

    def check_pipeline(self, world, batch, chunks, causal):
        torch.manual_seed(43)
        values = [[torch.randn(batch, 5, 8 * world, 16)[..., ::2]
                   for _ in range(3)] for _ in range(world)]
        rank_context = threading.local()
        lock = threading.Lock()
        exchanges = {}

        def exchange(tensor):
            rank, index = rank_context.rank, rank_context.index
            rank_context.index += 1
            with lock:
                entry = exchanges.setdefault(index, {
                    "inputs": {}, "ready": threading.Event(), "outputs": None,
                })
                entry["inputs"][rank] = tensor.flatten()
                if len(entry["inputs"]) == world:
                    parts = [entry["inputs"][r].chunk(world) for r in range(world)]
                    entry["outputs"] = [torch.cat([p[r] for p in parts]) for r in range(world)]
                    entry["ready"].set()

            def wait():
                if not entry["ready"].wait(10):
                    raise TimeoutError(f"rank {rank}: mismatched collective order {index}")
                return entry["outputs"][rank]
            return SimpleNamespace(wait=wait)

        def attention(q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=causal,
            ).transpose(1, 2)

        expected = attention(*[torch.cat([v[i] for v in values], dim=1) for i in range(3)])

        def run(rank):
            rank_context.rank, rank_context.index = rank, 0
            with torch.inference_mode():
                return usp.ulysses_attention_pipelined(*values[rank], attention, chunks)

        with patch.object(usp, "get_ulysses_parallel_world_size", return_value=world), \
             patch.object(usp, "_usp_all_to_all_single_async", side_effect=exchange), \
             patch.object(usp, "_maybe_wait", side_effect=lambda pending: pending.wait()), \
             ThreadPoolExecutor(max_workers=world) as pool:
            futures = [pool.submit(run, rank) for rank in range(world)]
            for rank, future in enumerate(futures):
                actual = future.result(timeout=15)
                torch.testing.assert_close(actual, expected[:, rank * 5 : (rank + 1) * 5],
                                           rtol=1e-5, atol=1e-6)
        self.assertEqual(len(exchanges), 2 * chunks)


def distributed_check(backend):
    if backend == "nccl":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend)
    device = torch.device("cuda" if backend == "nccl" else "cpu")
    try:
        if backend == "nccl":
            check_cuda_buffer_lifetimes()
        for batch in (1, 2):
            for chunks in (2, 4):
                for causal in (False, True):
                    torch.manual_seed(17 + dist.get_rank())
                    values = [torch.randn(batch, 7, 8 * dist.get_world_size(), 16,
                                          device=device)[..., ::2] for _ in range(3)]

                    def attention(q, k, v):
                        return torch.nn.functional.scaled_dot_product_attention(
                            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                            is_causal=causal,
                        ).transpose(1, 2)

                    with torch.inference_mode():
                        q, k, v = [usp._usp_input_all_to_all(x, 2) for x in values]
                        expected = usp._usp_output_all_to_all(attention(q, k, v), 2)
                        for _ in range(3):
                            actual = usp.ulysses_attention_pipelined(*values, attention, chunks)
                            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        print(json.dumps({"rank": dist.get_rank(), "backend": backend,
                          "pipelined_attention": "passed"}), flush=True)
    finally:
        dist.destroy_process_group()


def check_cuda_buffer_lifetimes():
    """Queue repeated forwards and release before synchronization on a side stream."""
    for prefetch in (1, 4, 9):
        for resident in (0, 3, 9):
            for strided in (False, True):
                torch.manual_seed(7)
                reference = Model(strided=strided).cuda().eval()
                model = Model(strided=strided).eval()
                model.load_state_dict(reference.state_dict())
                with patch.dict(os.environ, SGLANG_VIDEOEDIT_OVERLAP_PREFETCH="1",
                                SGLANG_VIDEOEDIT_RESIDENT_LAYERS=str(resident)):
                    manager = offload.LayerwiseOffloadManager(
                        model, layers_attr_str="blocks", num_layers=9,
                        enabled=True, prefetch_size=prefetch,
                    )
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                results = []
                with torch.inference_mode(), torch.cuda.stream(stream):
                    for _ in range(20):
                        x = torch.randn(8, 16, device="cuda")
                        results.append((model(x), reference(x)))
                    manager.release_all()
                    self_check = torch.empty(1024 * 1024, device="cuda").fill_(123)
                stream.synchronize()
                for actual, expected in results:
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert not manager._gpu_layers and not manager._retired_gpu_buffers
                manager.remove_forward_hooks()
                del self_check


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "nccl"), required=True)
    distributed_check(parser.parse_args().backend)
