import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from sglang.multimodal_gen.configs.sample.star_cogvideox_sr import (
    StarCogVideoXSRSamplingParams,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.star_cogvideox_sr_decoding import (
    STARCogVideoXSRDecodingStage,
)

_GLOBAL_ARGS_PATCH = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args"
)


class _FakeVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decode_calls: list[dict[str, float | int | bool]] = []
        self.enable_tiling_called = False

    def to(self, *args, **kwargs):
        del args, kwargs
        return self

    def enable_tiling(self, use_tiling: bool = True) -> None:
        self.enable_tiling_called = use_tiling

    def decode(
        self,
        latents: torch.Tensor,
        *,
        target_num_frames: int | None = None,
        clear_fake_cp_cache: bool = False,
        **kwargs,
    ):
        del kwargs
        self.decode_calls.append(
            {
                "num_frames": int(latents.shape[2]),
                "target_num_frames": (
                    None if target_num_frames is None else int(target_num_frames)
                ),
                "clear_fake_cp_cache": bool(clear_fake_cp_cache),
                "first_value": float(latents[0, 0, 0, 0, 0].item()),
                "last_value": float(latents[0, 0, -1, 0, 0].item()),
            }
        )
        if target_num_frames is None:
            expanded = latents
            for _ in range(2):
                if expanded.shape[2] <= 1:
                    break
                if expanded.shape[2] % 2 == 1:
                    first_frame = expanded[:, :, :1]
                    rest_frames = expanded[:, :, 1:].repeat_interleave(2, dim=2)
                    expanded = torch.cat([first_frame, rest_frames], dim=2)
                else:
                    expanded = expanded.repeat_interleave(2, dim=2)
        else:
            expanded = latents.repeat_interleave(target_num_frames, dim=2)
        return SimpleNamespace(sample=expanded.repeat(1, 3, 1, 1, 1))


class _PipelineConfig:
    vae_precision = "fp32"
    vae_tiling = True
    enable_color_fix = False
    color_fix_mode = None

    def get_decode_scale_and_shift(self, device, dtype, vae):
        del device, dtype, vae
        return 2.0, None

    def preprocess_decoding(self, latents, server_args=None, vae=None):
        del server_args, vae
        return latents + 0.25

    def post_decoding(self, frames, server_args):
        del server_args
        return frames + 0.1


class TestStarDecodingStage(unittest.TestCase):
    @staticmethod
    def _make_stage() -> tuple[STARCogVideoXSRDecodingStage, _FakeVAE]:
        fake_vae = _FakeVAE()
        with patch(
            _GLOBAL_ARGS_PATCH,
            return_value=MagicMock(vae_cpu_offload=False, comfyui_mode=False),
        ):
            stage = STARCogVideoXSRDecodingStage(vae=fake_vae)
        stage.server_args = SimpleNamespace(vae_cpu_offload=False, comfyui_mode=False)
        return stage, fake_vae

    def test_decode_applies_scale_shift_and_window_concat(self):
        stage, fake_vae = self._make_stage()
        server_args = SimpleNamespace(
            pipeline_config=_PipelineConfig(),
            disable_autocast=True,
        )
        latents = torch.linspace(-1.0, 1.0, steps=7).reshape(1, 1, 7, 1, 1)

        decoded = stage.decode(
            latents,
            server_args,
            batch=SimpleNamespace(enable_color_fix=False, color_fix_mode=None),
        )

        expected_values = torch.tensor(
            [
                0.375,
                0.45833334,
                0.45833334,
                0.45833334,
                0.45833334,
                0.5416667,
                0.5416667,
                0.5416667,
                0.5416667,
                0.625,
                0.625,
                0.625,
                0.625,
                0.7083334,
                0.7083334,
                0.7083334,
                0.7083334,
                0.7916666,
                0.7916666,
                0.7916666,
                0.7916666,
                0.875,
                0.875,
                0.875,
                0.875,
            ],
            device=decoded.device,
        )
        self.assertEqual(tuple(decoded.shape), (1, 3, 25, 1, 1))
        self.assertTrue(torch.allclose(decoded[0, 0, :, 0, 0], expected_values, atol=1e-6))
        expected_calls = [
            {
                "num_frames": 3,
                "target_num_frames": None,
                "clear_fake_cp_cache": False,
                "first_value": -0.25,
                "last_value": 0.0833333432674408,
            },
            {
                "num_frames": 2,
                "target_num_frames": None,
                "clear_fake_cp_cache": False,
                "first_value": 0.25,
                "last_value": 0.4166666865348816,
            },
            {
                "num_frames": 2,
                "target_num_frames": None,
                "clear_fake_cp_cache": True,
                "first_value": 0.5833333134651184,
                "last_value": 0.75,
            },
        ]
        self.assertEqual(len(fake_vae.decode_calls), len(expected_calls))
        for actual, expected in zip(fake_vae.decode_calls, expected_calls, strict=True):
            self.assertEqual(actual["num_frames"], expected["num_frames"])
            self.assertEqual(actual["target_num_frames"], expected["target_num_frames"])
            self.assertEqual(
                actual["clear_fake_cp_cache"], expected["clear_fake_cp_cache"]
            )
            self.assertAlmostEqual(actual["first_value"], expected["first_value"], places=6)
            self.assertAlmostEqual(actual["last_value"], expected["last_value"], places=6)
        self.assertTrue(fake_vae.enable_tiling_called)

    def test_forward_runs_post_decoding_hook(self):
        stage, _ = self._make_stage()
        server_args = SimpleNamespace(
            pipeline_config=_PipelineConfig(),
            disable_autocast=True,
        )
        sampling_params = StarCogVideoXSRSamplingParams(
            prompt="test",
            condition_video_path="/tmp/unused.mp4",
            enable_color_fix=False,
        )
        latents = torch.linspace(-1.0, 1.0, steps=7).reshape(1, 1, 7, 1, 1)
        batch = Req(sampling_params=sampling_params)
        batch.latents = latents
        batch.return_trajectory_decoded = False

        with patch.object(stage, "load_model", return_value=None), patch.object(
            stage, "offload_model", return_value=None
        ):
            output_batch = stage.forward(batch, server_args)

        expected = stage.decode(latents, server_args, batch=batch) + 0.1
        self.assertTrue(torch.allclose(output_batch.output, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
