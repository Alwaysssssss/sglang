import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.multimodal_gen.configs.sample import (
    SamplingParams,
    StarCogVideoXSRSamplingParams,
)
from sglang.multimodal_gen.configs.sample.sampling_params import DataType


class TestStarCogVideoXSamplingParams(unittest.TestCase):
    @staticmethod
    def _make_server_args():
        task_type = MagicMock()
        task_type.requires_image_input.return_value = False
        task_type.accepts_image_input.return_value = False
        task_type.is_image_gen.return_value = False
        task_type.data_type.return_value = DataType.VIDEO

        pipeline_config = MagicMock()
        pipeline_config.task_type = task_type
        pipeline_config.adjust_num_frames.side_effect = lambda value: value
        pipeline_config.vae_config = SimpleNamespace(
            use_temporal_scaling_frames=False,
            arch_config=SimpleNamespace(temporal_compression_ratio=1),
        )

        server_args = MagicMock()
        server_args.backend = "sglang"
        server_args.model_id = None
        server_args.pipeline_config = pipeline_config
        server_args.num_gpus = 1
        server_args.comfyui_mode = True
        server_args.output_path = None
        return server_args

    def test_condition_video_path_is_preserved(self):
        params = StarCogVideoXSRSamplingParams(
            prompt="test",
            condition_video_path="/tmp/input.mp4",
        )
        self.assertEqual(params.condition_video_path, "/tmp/input.mp4")

    def test_explicit_fields_track_condition_video_and_dimensions(self):
        server_args = self._make_server_args()

        with patch.object(
            SamplingParams,
            "from_pretrained",
            side_effect=lambda *args, **kwargs: StarCogVideoXSRSamplingParams(),
        ):
            params = SamplingParams.from_user_sampling_params_args(
                "dummy-model",
                server_args=server_args,
                prompt="p",
                condition_video_path="/tmp/in.mp4",
                width=768,
                height=512,
                num_frames=9,
            )

        explicit_fields = set(params.build_request_extra()["explicit_fields"])
        self.assertIn("condition_video_path", explicit_fields)
        self.assertIn("width", explicit_fields)
        self.assertIn("height", explicit_fields)
        self.assertIn("num_frames", explicit_fields)

    def test_star_specific_fields_do_not_pollute_base_sampling_params(self):
        self.assertFalse(hasattr(SamplingParams(), "condition_video_path"))
        self.assertTrue(hasattr(StarCogVideoXSRSamplingParams(), "condition_video_path"))


if __name__ == "__main__":
    unittest.main()
