import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.star_latent_preparation import (
    STARLatentPreparationStage,
)


class TestSTARLatentPreparationStage(unittest.TestCase):
    def test_adjust_video_length_uses_latent_timeline_directly(self):
        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args",
            return_value=MagicMock(),
        ):
            stage = STARLatentPreparationStage(
                scheduler=object(),
                transformer=object(),
            )
        batch = SimpleNamespace(num_frames=7)
        server_args = SimpleNamespace()

        latent_num_frames = stage.adjust_video_length(batch, server_args)

        self.assertEqual(latent_num_frames, 7)

    def test_forward_uses_star_cpu_initial_noise_generator(self):
        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args",
            return_value=MagicMock(),
        ):
            stage = STARLatentPreparationStage(
                scheduler=object(),
                transformer=object(),
            )
        batch = SimpleNamespace(
            batch_size=1,
            num_frames=3,
            height=16,
            width=16,
            prompt_embeds=[torch.zeros(1, dtype=torch.float32)],
            generator=[torch.Generator("cpu").manual_seed(999)],
            latents=None,
            extra={
                "star_initial_noise_generator": [
                    torch.Generator("cpu").manual_seed(1234)
                ]
            },
        )
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(
                latent_channels=4,
                dit_config=SimpleNamespace(
                    arch_config=SimpleNamespace(num_channels_latents=4)
                ),
                vae_config=SimpleNamespace(
                    arch_config=SimpleNamespace(spatial_compression_ratio=8)
                ),
                get_latent_dtype=lambda _dtype: torch.float32,
            )
        )

        batch = stage.forward(batch, server_args)

        expected = torch.randn(
            (1, 3, 4, 2, 2),
            generator=torch.Generator("cpu").manual_seed(1234),
            dtype=torch.float32,
        ).permute(0, 2, 1, 3, 4)
        self.assertTrue(torch.equal(batch.latents.cpu(), expected))
        self.assertEqual(tuple(batch.raw_latent_shape), (1, 4, 3, 2, 2))


if __name__ == "__main__":
    unittest.main()
