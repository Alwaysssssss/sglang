import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.configs.models.dits.star_cogvideox_sr import (
    StarCogVideoXSRArchConfig,
    StarCogVideoXSRDiTConfig,
)
from sglang.multimodal_gen.configs.models.vaes.star_cogvideox_vae import (
    StarCogVideoXSRVAEArchConfig,
    StarCogVideoXSRVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.star_cogvideox_sr import (
    StarCogVideoXSRPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.star_cogvideox_sr import (
    StarCogVideoXSRSamplingParams,
)
from sglang.multimodal_gen.runtime.models.vaes.star_cogvideox_vae import (
    StarCogVideoXSRVAE,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.video_condition_vae_encoding import (
    STARConditionVideoVAEEncodingStage,
)

_GLOBAL_ARGS_PATCH = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args"
)


class TestStarConditionVideoVaeEncodingStage(unittest.TestCase):
    def test_stage_encodes_condition_video_to_image_latent(self):
        vae_config = StarCogVideoXSRVAEConfig(
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
        dit_config = StarCogVideoXSRDiTConfig(
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
                latent_width=2,
                latent_height=2,
                num_frames=5,
                time_compressed_rate=2,
                time_embed_dim=32,
            )
        )
        pipeline_config = StarCogVideoXSRPipelineConfig(
            dit_config=dit_config,
            vae_config=vae_config,
            width=16,
            height=16,
            num_frames=5,
            latent_channels=4,
            vae_precision="fp32",
            vae_tiling=False,
        )
        vae = StarCogVideoXSRVAE(vae_config)

        with patch(_GLOBAL_ARGS_PATCH, return_value=MagicMock(vae_cpu_offload=False)):
            stage = STARConditionVideoVAEEncodingStage(vae=vae)
        stage.server_args = SimpleNamespace(vae_cpu_offload=False)

        sampling_params = StarCogVideoXSRSamplingParams(
            prompt="test",
            condition_video_path="/tmp/unused.mp4",
        )
        batch = Req(sampling_params=sampling_params)
        batch.condition_video = torch.randn(1, 5, 3, 16, 16)
        batch.generator = torch.Generator(device="cpu").manual_seed(0)
        batch.height = 16
        batch.width = 16
        batch.num_frames = 5

        server_args = SimpleNamespace(
            pipeline_config=pipeline_config,
            disable_autocast=True,
        )

        result = stage.forward(batch, server_args)

        self.assertIsNotNone(result.image_latent)
        self.assertEqual(tuple(result.image_latent.shape), (1, 4, 3, 2, 2))

    def test_expected_latent_num_frames_uses_condition_video_frame_count(self):
        pipeline_config = StarCogVideoXSRPipelineConfig()
        pipeline_config.vae_config.use_temporal_scaling_frames = True
        pipeline_config.vae_config.arch_config.temporal_compression_ratio = 4

        batch = Req(
            sampling_params=StarCogVideoXSRSamplingParams(
                prompt="test",
                condition_video_path="/tmp/unused.mp4",
                condition_video_num_frames=25,
            )
        )
        batch.num_frames = 7
        batch.condition_video_num_frames = 25

        expected = STARConditionVideoVAEEncodingStage._expected_latent_num_frames(
            batch,
            SimpleNamespace(pipeline_config=pipeline_config),
        )
        self.assertEqual(expected, 7)


if __name__ == "__main__":
    unittest.main()
