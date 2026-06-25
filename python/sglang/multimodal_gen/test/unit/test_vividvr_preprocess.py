import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from sglang.multimodal_gen.configs.sample.vividvr import VividVRSamplingParams
from sglang.multimodal_gen.runtime.pipelines.vividvr_pipeline import VividVRPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.vividvr import (
    VividVRConditionEncodingStage,
)
from sglang.multimodal_gen.runtime.vividvr.preprocess import (
    load_control_video,
    plan_generation_resolution,
)


def _make_test_frames(*, width: int = 8, height: int = 4, count: int = 2) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for frame_idx in range(count):
        tensor = torch.zeros(height, width, 3, dtype=torch.uint8)
        tensor[..., 0] = (frame_idx + 1) * 10
        tensor[..., 1] = torch.arange(width, dtype=torch.uint8).unsqueeze(0).expand(height, width)
        tensor[..., 2] = torch.arange(height, dtype=torch.uint8).unsqueeze(1).expand(height, width)
        frames.append(Image.fromarray(tensor.numpy(), mode="RGB"))
    return frames


class TestVividVRPreprocess(unittest.TestCase):
    def test_load_control_video_preserves_resolution_for_upscale_one(self):
        with patch(
            "sglang.multimodal_gen.runtime.vividvr.preprocess.load_control_video_frames",
            return_value=(_make_test_frames(), 24.0),
        ):
            info = load_control_video("/tmp/input.mp4", upscale=1.0)

        self.assertEqual(int(info["original_height"]), 4)
        self.assertEqual(int(info["original_width"]), 8)
        self.assertEqual(tuple(info["reference_video"].shape), (2, 3, 4, 8))
        self.assertEqual(int(info["original_num_frames"]), 2)
        self.assertEqual(int(info["num_padding_frames"]), 7)
        self.assertEqual(int(info["video"].shape[0]), 9)

    def test_load_control_video_applies_original_upscale_multiplier(self):
        with patch(
            "sglang.multimodal_gen.runtime.vividvr.preprocess.load_control_video_frames",
            return_value=(_make_test_frames(), 24.0),
        ):
            info = load_control_video("/tmp/input.mp4", upscale=2.0)

        self.assertEqual(int(info["original_height"]), 8)
        self.assertEqual(int(info["original_width"]), 16)
        self.assertEqual(tuple(info["reference_video"].shape), (2, 3, 8, 16))

    def test_load_control_video_matches_original_short_side_1024_behavior(self):
        with patch(
            "sglang.multimodal_gen.runtime.vividvr.preprocess.load_control_video_frames",
            return_value=(_make_test_frames(), 24.0),
        ):
            info = load_control_video("/tmp/input.mp4", upscale=0.0)

        self.assertEqual(int(info["original_height"]), 1024)
        self.assertEqual(int(info["original_width"]), 2048)
        self.assertEqual(tuple(info["reference_video"].shape), (2, 3, 1024, 2048))

    def test_plan_generation_resolution_matches_official_origin_formula(self):
        gen_height, gen_width = plan_generation_resolution(
            raw_height=1024,
            raw_width=1365,
            tile_size=128,
            vae_scale_factor_spatial=8,
        )

        self.assertEqual(gen_height, 1024)
        self.assertEqual(gen_width, 1365)

    def test_core_stage_passes_upscale_into_preprocess(self):
        stage = object.__new__(VividVRConditionEncodingStage)
        stage.vae = SimpleNamespace(
            config=SimpleNamespace(block_out_channels=[128, 256, 256, 512])
        )
        batch = SimpleNamespace(extra={})
        params = VividVRSamplingParams(video_input_path="/tmp/input.mp4", upscale=2.0)

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.vividvr.load_control_video",
            return_value={
                "video": torch.zeros(1, 3, 4, 8),
                "original_height": 4,
                "original_width": 8,
            },
        ) as load_mock:
            stage._resolve_control_video_info(batch, params)

        load_mock.assert_called_once_with("/tmp/input.mp4", upscale=2.0)

    def test_core_stage_syncs_runtime_resolution_from_generation_dims(self):
        stage = object.__new__(VividVRConditionEncodingStage)
        params = VividVRSamplingParams(video_input_path="/tmp/input.mp4", upscale=0.0)

        stage._sync_runtime_resolution(
            params,
            {
                "original_height": 1024,
                "original_width": 1365,
                "gen_height": 1024,
                "gen_width": 1365,
            },
        )

        self.assertEqual(params.height, 1024)
        self.assertEqual(params.width, 1365)

    def test_legacy_pipeline_cache_key_distinguishes_upscale(self):
        pipeline = object.__new__(VividVRPipeline)
        pipeline._cached_control_video_cache_key = None
        pipeline._cached_control_video_info = None

        fake_stat = SimpleNamespace(st_mtime_ns=123, st_size=456)
        with (
            patch("os.stat", return_value=fake_stat),
            patch(
                "sglang.multimodal_gen.runtime.pipelines.vividvr_pipeline.load_control_video",
                side_effect=[
                    {"marker": "one"},
                    {"marker": "two"},
                ],
            ) as load_mock,
        ):
            first = pipeline._resolve_input_video_info("/tmp/input.mp4", upscale=1.0)
            second = pipeline._resolve_input_video_info("/tmp/input.mp4", upscale=2.0)

        self.assertEqual(first["marker"], "one")
        self.assertEqual(second["marker"], "two")
        self.assertEqual(load_mock.call_args_list[0].kwargs["upscale"], 1.0)
        self.assertEqual(load_mock.call_args_list[1].kwargs["upscale"], 2.0)


if __name__ == "__main__":
    unittest.main()
