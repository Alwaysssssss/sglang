import unittest

import torch

from sglang.multimodal_gen.configs.models.dits.star_cogvideox_sr import (
    StarCogVideoXSRArchConfig,
    StarCogVideoXSRDiTConfig,
)
from sglang.multimodal_gen.runtime.models.dits.star_cogvideox_sr import (
    StarCogVideoXSRTransformer3DModel,
)


class TestStarTransformerShapes(unittest.TestCase):
    def test_forward_accepts_concat_latents_and_text_states(self):
        config = StarCogVideoXSRDiTConfig(
            arch_config=StarCogVideoXSRArchConfig(
                hidden_size=64,
                num_attention_heads=4,
                num_layers=2,
                in_channels=4,
                out_channels=4,
                num_channels_latents=4,
                patch_size=2,
                text_hidden_size=32,
                text_length=6,
                latent_width=8,
                latent_height=8,
                num_frames=5,
                time_compressed_rate=2,
                time_embed_dim=32,
            )
        )
        model = StarCogVideoXSRTransformer3DModel(
            config=config,
            hf_config={},
        )

        hidden_states = torch.randn(2, 8, 3, 8, 8)
        encoder_hidden_states = [torch.randn(2, 6, 32)]
        timestep = torch.tensor([31, 31], dtype=torch.long)

        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
        )

        self.assertEqual(tuple(output.shape), (2, 4, 3, 8, 8))
        self.assertEqual(output.dtype, hidden_states.dtype)

    def test_final_layernorm_participates_in_forward(self):
        config = StarCogVideoXSRDiTConfig(
            arch_config=StarCogVideoXSRArchConfig(
                hidden_size=64,
                num_attention_heads=4,
                num_layers=1,
                in_channels=4,
                out_channels=4,
                num_channels_latents=4,
                patch_size=2,
                text_hidden_size=32,
                text_length=6,
                latent_width=8,
                latent_height=8,
                num_frames=5,
                time_compressed_rate=2,
                time_embed_dim=32,
            )
        )
        model = StarCogVideoXSRTransformer3DModel(
            config=config,
            hf_config={},
        )
        model.eval()

        hidden_states = torch.randn(1, 8, 3, 8, 8)
        encoder_hidden_states = [torch.randn(1, 6, 32)]
        timestep = torch.tensor([31], dtype=torch.long)

        with torch.no_grad():
            baseline = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
            )
            model.transformer.final_layernorm.weight.zero_()
            model.transformer.final_layernorm.bias.zero_()
            changed = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
            )

        self.assertFalse(torch.allclose(baseline, changed))


if __name__ == "__main__":
    unittest.main()
