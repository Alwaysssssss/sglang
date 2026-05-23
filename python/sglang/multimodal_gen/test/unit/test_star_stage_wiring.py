import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.multimodal_gen.configs.pipeline_configs.star_cogvideox_sr import (
    StarCogVideoXSRPipelineConfig,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines.star_cogvideox_sr_pipeline import (
    StarCogVideoXSRPipeline,
)

_GLOBAL_ARGS_PATCH = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args"
)
_DENOISING_ATTN_PATCH = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising.get_attn_backend"
)


class TestStarPipelineWiring(unittest.TestCase):
    @staticmethod
    def _make_pipeline(model_path: str) -> StarCogVideoXSRPipeline:
        pipeline = object.__new__(StarCogVideoXSRPipeline)
        pipeline.model_path = model_path
        pipeline.modules = {
            "text_encoder": object(),
            "tokenizer": object(),
            "vae": MagicMock(),
            "transformer": object(),
            "scheduler": object(),
        }
        pipeline._disagg_role = RoleType.MONOLITHIC
        pipeline._stages = []
        pipeline._stage_name_mapping = {}
        return pipeline

    def test_initialize_pipeline_applies_integration_defaults(self):
        with tempfile.TemporaryDirectory() as tempdir:
            integration_path = Path(tempdir) / "star_integration_config.json"
            payload = {
                "latent_channels": 32,
                "default_sampling_num_frames": 9,
                "latent_scale_factor": 0.7,
                "transformer_summary": {
                    "latent_width": 80,
                    "latent_height": 40,
                },
            }
            with open(integration_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

            pipeline = self._make_pipeline(tempdir)
            config = StarCogVideoXSRPipelineConfig()
            server_args = SimpleNamespace(pipeline_config=config)

            StarCogVideoXSRPipeline.initialize_pipeline(pipeline, server_args)

            self.assertEqual(config.width, 640)
            self.assertEqual(config.height, 320)
            self.assertEqual(config.num_frames, 9)
            self.assertEqual(config.latent_channels, 32)
            self.assertEqual(config.dit_config.arch_config.num_channels_latents, 32)
            self.assertEqual(config.vae_config.arch_config.scaling_factor, 0.7)

    def test_stage_order_matches_phase3_plan(self):
        with patch(_GLOBAL_ARGS_PATCH, return_value=MagicMock()), patch(
            _DENOISING_ATTN_PATCH, return_value=MagicMock()
        ):
            pipeline = self._make_pipeline("/tmp/fake-star-model")
            server_args = SimpleNamespace(
                pipeline_config=StarCogVideoXSRPipelineConfig()
            )

            StarCogVideoXSRPipeline.create_pipeline_stages(pipeline, server_args)

        stage_names = [type(stage).__name__ for stage in pipeline.stages]
        self.assertEqual(
            stage_names,
            [
                "InputValidationStage",
                "STARConditionVideoLoadingStage",
                "TextEncodingStage",
                "STARConditionVideoVAEEncodingStage",
                "STARLatentPreparationStage",
                "TimestepPreparationStage",
                "DenoisingStage",
                "STARCogVideoXSRDecodingStage",
            ],
        )


if __name__ == "__main__":
    unittest.main()
