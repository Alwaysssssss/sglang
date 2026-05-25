import unittest
from pathlib import Path

import torch

from sglang.multimodal_gen.configs.models.vaes.star_cogvideox_vae import (
    StarCogVideoXSRVAEArchConfig,
    StarCogVideoXSRVAEConfig,
)
from sglang.multimodal_gen.runtime.models.vaes.star_cogvideox_vae import (
    StarCogVideoXSRVAE,
    _resolve_star_sat_root,
)


class TestStarVaeShapes(unittest.TestCase):
    def test_vendored_star_sat_root_lives_inside_sglang_repo(self):
        sat_root = _resolve_star_sat_root()
        self.assertIsNotNone(sat_root)
        resolved = Path(sat_root).resolve()
        self.assertIn("star_sat_vendor", resolved.parts)
        self.assertNotIn("STAR_mg", resolved.parts)

    def test_encode_and_decode_keep_expected_video_shape_contract(self):
        config = StarCogVideoXSRVAEConfig(
            arch_config=StarCogVideoXSRVAEArchConfig(
                ch=32,
                ch_mult=[1, 1, 1, 1],
                num_res_blocks=1,
                z_channels=4,
                latent_channels=4,
                temporal_compression_ratio=2,
                spatial_compression_ratio=8,
                scaling_factor=0.7,
            )
        )
        vae = StarCogVideoXSRVAE(config)

        video = torch.randn(1, 3, 5, 16, 16)
        posterior = vae.encode(video).latent_dist
        latents = posterior.mode()

        self.assertEqual(tuple(latents.shape), (1, 4, 3, 2, 2))

        decoded = vae.decode(latents).sample
        self.assertEqual(tuple(decoded.shape), (1, 3, 5, 16, 16))


if __name__ == "__main__":
    unittest.main()
