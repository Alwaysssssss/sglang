"""Scheduling regression and CUDA lifetime checks for layerwise prefetch."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.multimodal_gen.runtime.utils.layerwise_offload import LayerwiseOffloadManager
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.videoedit_wan import VideoEditDenoisingStage


class OverlapTests(unittest.TestCase):
    def test_stage_cache_is_released_on_success_and_failure(self):
        for failure in (False, True):
            manager = Mock(release_after_denoise=True)
            stage = SimpleNamespace(
                transformer=SimpleNamespace(layerwise_offload_managers=[manager]),
                _forward_denoising=Mock(return_value="result", side_effect=RuntimeError("simulated failure") if failure else None))
            if failure:
                with self.assertRaisesRegex(RuntimeError, "simulated failure"):
                    VideoEditDenoisingStage.forward(stage, None, None)
            else:
                self.assertEqual(VideoEditDenoisingStage.forward(stage, None, None), "result")
            manager.release_all.assert_called_once_with()

    def test_first_block_waits_only_for_its_own_weights(self):
        manager = LayerwiseOffloadManager.__new__(LayerwiseOffloadManager)
        blocks = [Mock() for _ in range(8)]
        manager.model = SimpleNamespace(blocks=blocks)
        manager.layers_attr_str = "blocks"
        manager.enabled = True
        manager.overlap_prefetch = True
        manager.resident_layers = 0
        manager.num_layers = 8
        manager.prefetch_size = 4
        manager._forward_hooks = []
        manager._prefetch_events = {0: object()}
        manager._gpu_buffers = {0: {torch.float32: Mock()}}
        manager._strided_gpu_buffers = {}
        manager.prepare_for_next_req = Mock()
        manager.prefetch_layer = Mock()
        manager.release_layer = Mock()
        with patch.object(torch, "get_device_module") as device:
            manager.register_forward_hooks()
            hook = blocks[0].register_forward_pre_hook.call_args.args[0]
            hook(None, None)
            manager.prepare_for_next_req.assert_called_once_with(non_blocking=True)
            device.return_value.current_stream.return_value.wait_event.assert_called_once_with(manager._prefetch_events[0])
            manager._gpu_buffers[0][torch.float32].record_stream.assert_called_once()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_weights_survive_copy_stream_reuse_and_repeated_forwards(self):
        self.check_repeated_forwards(resident_layers=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_resident_cache_survives_forwards_but_explicit_release_frees_it(self):
        for overlap in (False, True):
            for count in (3, 9):
                with self.subTest(overlap=overlap, resident_layers=count):
                    self.check_repeated_forwards(resident_layers=count, overlap=overlap)

    def check_repeated_forwards(self, resident_layers, overlap=True):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([
                    torch.nn.Linear(128, 128, bias=True) for _ in range(9)])

            def forward(self, x):
                for block in self.blocks:
                    x = torch.tanh(block(x))
                return x

        torch.manual_seed(7)
        reference = Model().cuda().eval()
        offloaded = Model().eval()
        offloaded.load_state_dict(reference.state_dict())
        with patch.dict(os.environ, SGLANG_VIDEOEDIT_OVERLAP_PREFETCH=str(int(overlap)),
                        SGLANG_VIDEOEDIT_RESIDENT_LAYERS=str(resident_layers)):
            manager = LayerwiseOffloadManager(offloaded, layers_attr_str="blocks",
                num_layers=9, enabled=True, prefetch_size=4)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), torch.cuda.stream(stream):
            for _ in range(20):
                x = torch.randn(32, 128, device="cuda")
                expected = reference(x)
                actual = offloaded(x)
                # Queue work before releasing the buffers to exercise cross-stream lifetimes.
                scratch = torch.empty(128 * 128 * 8, device="cuda").fill_(123)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                del scratch
        stream.synchronize()
        self.assertTrue(set(range(resident_layers)).issubset(manager._gpu_layers))
        if resident_layers:
            before = manager.h2d_layer_loads
            with torch.inference_mode():
                offloaded(torch.randn(32, 128, device="cuda"))
            torch.cuda.synchronize()
            self.assertEqual(manager.h2d_layer_loads - before, 9 - resident_layers)
        manager.release_all()
        self.assertFalse(manager._gpu_layers)
        self.assertFalse(manager._gpu_buffers)
        self.assertFalse(manager._retired_gpu_buffers)
        if resident_layers == 9:
            before = manager.h2d_layer_loads
            with torch.inference_mode():
                x = torch.randn(32, 128, device="cuda")
                torch.testing.assert_close(offloaded(x), reference(x), rtol=0, atol=0)
            self.assertEqual(manager.h2d_layer_loads - before, 9)
            manager.release_all()
        manager.remove_forward_hooks()


if __name__ == "__main__":
    unittest.main()
