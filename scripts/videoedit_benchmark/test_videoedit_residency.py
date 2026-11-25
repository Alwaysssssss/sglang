"""Shape budget, bounded observations, scheduling transitions and rank consensus."""
import json
import os
import sys
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist

from scripts.videoedit_benchmark.test_overlap_pipeline import memory, offload, Model

GiB = 1024**3
SMALL = (1, 16, 13, 60, 68)
FULL = (1, 16, 13, 136, 240)


class ResidencyPolicyTests(unittest.TestCase):
    def test_small_full_small_uses_shape_not_last_request(self):
        planner = memory.VideoEditResidencyPlanner()
        small_key, small = planner.plan(SMALL, 43 * GiB)
        full_key, full = planner.plan(FULL, 43 * GiB)
        self.assertEqual(small['estimated_activation_bytes'], 4 * GiB)
        self.assertEqual(full['estimated_activation_bytes'], 16 * GiB)
        layers = [3 * GiB // 4] * 40
        self.assertEqual(memory.select_weight_budget(layers, small['local_weight_budget_bytes'], 1)[0], 40)
        self.assertLess(memory.select_weight_budget(layers, full['local_weight_budget_bytes'], 1)[0], 40)
        planner.observe(full_key, peak_allocated_bytes=43 * GiB,
                        initial_allocated_bytes=1 * GiB, weight_capacity_bytes=24 * GiB)
        _, grown = planner.plan(FULL, 43 * GiB)
        self.assertGreater(grown['estimated_activation_bytes'], full['estimated_activation_bytes'])
        planner.observe(full_key, peak_allocated_bytes=30 * GiB,
                        initial_allocated_bytes=1 * GiB, weight_capacity_bytes=24 * GiB)
        self.assertEqual(planner.plan(FULL, 43 * GiB)[1], grown)
        self.assertEqual(planner.plan(SMALL, 43 * GiB), (small_key, small))

    def test_headroom_limits_and_out_of_profile_shapes(self):
        planner = memory.VideoEditResidencyPlanner()
        _, limited = planner.plan(SMALL, 43 * GiB, weight_limit_bytes=10 * GiB)
        self.assertEqual(limited['local_weight_budget_bytes'], 10 * GiB)
        # An explicit weight cap can still leave a budget no layer fits.
        _, capped = planner.plan(FULL, 43 * GiB, weight_limit_bytes=1)
        self.assertEqual(capped['local_weight_budget_bytes'], 1)
        with self.assertRaises(ValueError):
            memory.select_weight_budget([GiB] * 40, capped['local_weight_budget_bytes'], 1)
        # Headroom below the reserve plus margin cannot serve the window at all.
        with self.assertRaisesRegex(ValueError, '49-frame window'):
            planner.plan(FULL, 10 * GiB)
        for shape in ((2, 16, 13, 60, 68), (1, 16, 13, 61, 68),
                      (1, 16, 13, 240, 240), (1, 16, 0, 60, 68)):
            with self.assertRaises(ValueError):
                planner.plan(shape, 43 * GiB)
        with self.assertRaises(ValueError):
            planner.plan(SMALL, 43 * GiB, sp_degree=4)
        # Portrait full-frame has the same budget as landscape.
        self.assertEqual(planner.plan((1, 16, 13, 240, 136), 43 * GiB)[1]['local_weight_budget_bytes'],
                         planner.plan(FULL, 43 * GiB)[1]['local_weight_budget_bytes'])

    def test_longer_windows_trade_activation_reserve_for_weight_budget(self):
        planner = memory.VideoEditResidencyPlanner()
        largest_layer = 3 * GiB // 4
        budgets = []
        for latent_frames in (13, 26, 40):
            _, plan = planner.plan((1, 16, latent_frames, 78, 218), 36 * GiB,
                                   minimum_weight_bytes=largest_layer)
            budgets.append(plan['local_weight_budget_bytes'])
            # The reserve scales with the window's latent volume, not with a cap.
            self.assertEqual(plan['estimated_activation_bytes'],
                             (16 * GiB * latent_frames * 78 * 218 + 13 * 136 * 240 - 1)
                             // (13 * 136 * 240))
        self.assertGreater(budgets[0], budgets[1])
        self.assertGreater(budgets[1], budgets[2])
        self.assertEqual(len(planner.history), 0)

    def test_infeasible_window_reports_the_supported_frame_count(self):
        planner = memory.VideoEditResidencyPlanner()
        largest_layer = 3 * GiB // 4
        headroom, hint = 36 * GiB, memory.feasible_window_frames(
            78, 218, 36 * GiB, minimum_weight_bytes=largest_layer)
        self.assertGreater(hint, 0)
        self.assertLess(hint, 209)
        with self.assertRaises(ValueError) as caught:
            planner.plan((1, 16, 53, 78, 218), headroom, minimum_weight_bytes=largest_layer)
        self.assertIn('209-frame window', str(caught.exception))
        self.assertIn(f'about {hint} frames', str(caught.exception))
        # The reported frame count is the largest window that still fits.
        hint_latent = (hint - 1) // 4 + 1
        self.assertEqual(
            planner.plan((1, 16, hint_latent, 78, 218), headroom,
                         minimum_weight_bytes=largest_layer)[1]['latent_shape'],
            [1, 16, hint_latent, 78, 218])
        with self.assertRaises(ValueError):
            planner.plan((1, 16, hint_latent + 1, 78, 218), headroom,
                         minimum_weight_bytes=largest_layer)

    def test_observations_are_bounded_and_cfg_isolated(self):
        planner = memory.VideoEditResidencyPlanner()
        key, _ = planner.plan(SMALL, 43 * GiB, cfg=True)
        planner.observe(key, peak_allocated_bytes=20 * GiB,
                        initial_allocated_bytes=0, weight_capacity_bytes=0)
        self.assertEqual(planner.plan(SMALL, 43 * GiB, cfg=False)[1]['historical_activation_bytes'], 0)
        for latent_frames in range(1, 200):
            key, _ = planner.plan((1, 16, latent_frames, 2, 2), 43 * GiB)
            planner.observe(key, peak_allocated_bytes=GiB,
                            initial_allocated_bytes=0, weight_capacity_bytes=0)
        self.assertEqual(len(planner.history), 128)

    def test_window_validation_bounds_geometry_not_a_fixed_frame_count(self):
        from types import SimpleNamespace
        from scripts.videoedit_benchmark.test_videoedit_memory import stages

        args = SimpleNamespace(dit_layerwise_offload=True, sp_degree=2)

        def validate(size, frames):
            window = [SimpleNamespace(size=size)] * frames
            params = SimpleNamespace(runtime_window_frames=window,
                                     runtime_window_masks=window, infer_len=frames)
            with patch.dict(os.environ, SGLANG_VIDEOEDIT_AUTO_RESIDENCY='1'):
                stages.VideoEditWindowValidationStage().forward(
                    SimpleNamespace(sampling_params=params), args)
            return params

        # infer_len varies per request: a longer window only costs activation
        # reserve, so it validates as long as the crop stays inside the profile.
        self.assertTrue(validate((1920, 1088), 53).runtime_window_validated)
        self.assertTrue(validate((1920, 1088), 105).runtime_window_validated)
        with self.assertRaisesRegex(ValueError, 'aligned crops'):
            validate((3840, 2160), 53)
        with self.assertRaisesRegex(ValueError, 'divisible by 16'):
            validate((1920, 1080), 53)
        # Without CUDA the pre-flight has no device bound and still validates.
        params = validate((1920, 1088), 305)
        self.assertTrue(params.runtime_window_validated)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_full_partial_full_manager_transitions_preserve_outputs(self):
        with patch.dict(os.environ, SGLANG_VIDEOEDIT_AUTO_RESIDENCY='1',
                        SGLANG_VIDEOEDIT_RESIDENT_LAYERS='0',
                        SGLANG_VIDEOEDIT_OVERLAP_PREFETCH='0',
                        SGLANG_VIDEOEDIT_SHARED_CPU_DIR='',
                        SGLANG_VIDEOEDIT_OFFLOAD_PINNED_MB='0'):
            torch.manual_seed(19)
            reference, model = Model().cuda().eval(), Model().eval()
            model.load_state_dict(reference.state_dict())
            manager = offload.LayerwiseOffloadManager(model, layers_attr_str='blocks', num_layers=9,
                                                       enabled=True, prefetch_size=4)
            try:
                per_layer = manager.cpu_weight_bytes // 9
                for budget, residents, prefetch, overlap in (
                    (manager.cpu_weight_bytes, 9, 4, False),
                    (per_layer * 5, 3, 1, True),
                    (per_layer, 0, 0, True),
                    (manager.cpu_weight_bytes, 9, 4, False),
                ):
                    manager.configure_memory_budget(budget)
                    self.assertEqual((manager.resident_layers, manager.prefetch_size,
                                      manager.overlap_prefetch), (residents, prefetch, overlap))
                    self.assertLessEqual(manager.weight_capacity_bytes, budget)
                    with torch.inference_mode():
                        for _ in range(3):
                            x = torch.randn(8, 16, device='cuda')
                            torch.testing.assert_close(model(x), reference(x), rtol=0, atol=0)
                    manager.release_all()
                    torch.cuda.synchronize()
                    self.assertFalse(manager._gpu_layers)
                    self.assertFalse(manager._retired_gpu_buffers)
            finally:
                manager.remove_forward_hooks()
                manager.release_all()


def distributed_check():
    dist.init_process_group('gloo')
    try:
        rank = dist.get_rank()
        for local in (43 * GiB - rank * 5 * GiB, 0 if rank else 43 * GiB):
            common = memory.common_weight_budget(local, dist.group.WORLD)
            values = [None] * dist.get_world_size()
            dist.all_gather_object(values, local)
            assert common == min(values)
        print(json.dumps({'rank': rank, 'minimum_budget_consensus': 'passed'}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    if '--distributed' in sys.argv:
        distributed_check()
    else:
        unittest.main()
