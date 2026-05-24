import math
import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.configs.pipeline_configs.star_cogvideox_sr import (
    StarCogVideoXSRPipelineConfig,
)


class TestStarDynamicCFG(unittest.TestCase):
    def test_dynamic_cfg_matches_star_schedule_shape(self):
        config = StarCogVideoXSRPipelineConfig()
        batch = SimpleNamespace(num_inference_steps=50)

        first = config.get_classifier_free_guidance_scale_for_step(batch, 6.0, 0)
        middle = config.get_classifier_free_guidance_scale_for_step(batch, 6.0, 25)
        last = config.get_classifier_free_guidance_scale_for_step(batch, 6.0, 49)

        self.assertTrue(math.isclose(first, 7.0, rel_tol=0.0, abs_tol=1e-6))
        self.assertGreater(first, middle)
        self.assertGreater(middle, last)
        self.assertGreaterEqual(last, 1.0)

    def test_star_timestep_hook_uses_remaining_step_index(self):
        config = StarCogVideoXSRPipelineConfig()
        batch = SimpleNamespace(num_inference_steps=50)
        t_device = torch.tensor(999, dtype=torch.int64)

        self.assertIsNone(
            config.expand_timestep_before_forward_for_step(
                batch=batch,
                t_device=t_device,
                target_dtype=torch.bfloat16,
                seq_len=None,
                reserved_frames_mask=None,
                batch_size=2,
                timestep_index=0,
            )
        )

        config.use_step_index_timestep = True

        timestep = config.expand_timestep_before_forward_for_step(
            batch=batch,
            t_device=t_device,
            target_dtype=torch.bfloat16,
            seq_len=None,
            reserved_frames_mask=None,
            batch_size=2,
            timestep_index=0,
        )
        last_timestep = config.expand_timestep_before_forward_for_step(
            batch=batch,
            t_device=t_device,
            target_dtype=torch.bfloat16,
            seq_len=None,
            reserved_frames_mask=None,
            batch_size=2,
            timestep_index=49,
        )

        self.assertTrue(torch.equal(timestep, torch.tensor([50, 50])))
        self.assertTrue(torch.equal(last_timestep, torch.tensor([1, 1])))

    def test_batched_cfg_prefers_request_override(self):
        config = StarCogVideoXSRPipelineConfig()
        server_args = SimpleNamespace()

        self.assertTrue(
            config.should_use_batched_cfg(
                SimpleNamespace(enable_batched_cfg=None), server_args
            )
        )
        self.assertFalse(
            config.should_use_batched_cfg(
                SimpleNamespace(enable_batched_cfg=False), server_args
            )
        )
        self.assertTrue(
            config.should_use_batched_cfg(
                SimpleNamespace(enable_batched_cfg=True), server_args
            )
        )

    def test_phase7_pipeline_overrides_update_resident_strategy_flags(self):
        config = StarCogVideoXSRPipelineConfig()

        config.apply_integration_config(
            {
                "release_text_encoder_after_prompt_encode": False,
                "temporarily_offload_transformer_during_condition_vae_encode": True,
                "condition_video_vae_peak_memory_mode": "transformer_only",
            }
        )

        self.assertFalse(config.release_text_encoder_after_prompt_encode)
        self.assertTrue(
            config.temporarily_offload_transformer_during_condition_vae_encode
        )
        self.assertEqual(
            config.resolve_condition_video_vae_peak_memory_mode(),
            "transformer_only",
        )


if __name__ == "__main__":
    unittest.main()
