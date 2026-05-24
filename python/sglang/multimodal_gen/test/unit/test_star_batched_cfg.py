import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)


class _DummyStarCFGModel(nn.Module):
    def forward(
        self,
        hidden_states,
        timestep,
        guidance,
        encoder_hidden_states,
        encoder_attention_mask=None,
    ):
        batch = hidden_states.shape[0]
        text_term = encoder_hidden_states.mean(dim=(1, 2)).view(batch, 1, 1, 1, 1)
        time_term = timestep.float().view(batch, 1, 1, 1, 1)
        guide_term = guidance.float().view(batch, 1, 1, 1, 1)
        if encoder_attention_mask is None:
            mask_term = 0.0
        else:
            mask_term = encoder_attention_mask.float().mean(dim=1).view(batch, 1, 1, 1, 1)
        return hidden_states + text_term + time_term + guide_term + mask_term


class TestStarBatchedCFG(unittest.TestCase):
    def _build_stage(self):
        stage = object.__new__(DenoisingStage)
        stage.server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(slice_noise_pred=lambda noise, latents: noise)
        )
        return stage

    def test_concat_cfg_branch_values_for_tensors_and_lists(self):
        tensor_a = torch.randn(1, 2, 3)
        tensor_b = torch.randn(1, 2, 3)
        merged, ok = DenoisingStage._concat_cfg_branch_values(tensor_a, tensor_b)
        self.assertTrue(ok)
        self.assertEqual(tuple(merged.shape), (2, 2, 3))

        list_a = [torch.randn(1, 4), torch.randn(1, 5)]
        list_b = [torch.randn(1, 4), torch.randn(1, 5)]
        merged_list, ok = DenoisingStage._concat_cfg_branch_values(list_a, list_b)
        self.assertTrue(ok)
        self.assertEqual(tuple(merged_list[0].shape), (2, 4))
        self.assertEqual(tuple(merged_list[1].shape), (2, 5))

    def test_batched_cfg_prediction_matches_serial_reference(self):
        stage = self._build_stage()
        model = _DummyStarCFGModel()
        batch = SimpleNamespace(is_cfg_negative=False)

        latent_model_input = torch.randn(1, 4, 2, 3, 3)
        latents = latent_model_input.clone()
        timestep = torch.tensor([7], dtype=torch.int64)
        guidance = torch.tensor([6.0], dtype=torch.float32)
        pos_cond_kwargs = {
            "encoder_hidden_states": torch.randn(1, 5, 8),
            "encoder_attention_mask": torch.ones(1, 5),
        }
        neg_cond_kwargs = {
            "encoder_hidden_states": torch.randn(1, 5, 8),
            "encoder_attention_mask": torch.zeros(1, 5),
        }

        batched = stage._predict_noise_with_batched_cfg(
            current_model=model,
            latent_model_input=latent_model_input,
            timestep=timestep,
            batch=batch,
            timestep_index=3,
            attn_metadata=None,
            target_dtype=torch.bfloat16,
            image_kwargs={},
            pos_cond_kwargs=pos_cond_kwargs,
            neg_cond_kwargs=neg_cond_kwargs,
            guidance=guidance,
            latents=latents,
        )
        self.assertIsNotNone(batched)
        noise_pred_cond, noise_pred_uncond = batched

        serial_cond = model(
            hidden_states=latent_model_input,
            timestep=timestep,
            guidance=guidance,
            **pos_cond_kwargs,
        )
        serial_uncond = model(
            hidden_states=latent_model_input,
            timestep=timestep,
            guidance=guidance,
            **neg_cond_kwargs,
        )
        self.assertTrue(torch.allclose(noise_pred_cond, serial_cond))
        self.assertTrue(torch.allclose(noise_pred_uncond, serial_uncond))


if __name__ == "__main__":
    unittest.main()
