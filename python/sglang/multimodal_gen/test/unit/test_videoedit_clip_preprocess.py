import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.videoedit_wan import (
    WanVideoEditPipelineConfig,
    videoedit_prompt_clean,
)
from sglang.multimodal_gen.configs.pipeline_configs.wan import t5_postprocess_text
from sglang.multimodal_gen.configs.sample.videoedit_wan import (
    WanVideoEditSamplingParams,
)
from sglang.multimodal_gen.runtime.models.dits.wanvideo import (
    _explicit_norm_scale_shift,
    _videoedit_apply_rotary_emb,
    _videoedit_rms_norm,
)
from sglang.multimodal_gen.runtime.pipelines.wan_videoedit_pipeline import (
    _load_videoedit_clip_encoder,
    _load_videoedit_text_encoder,
    _load_videoedit_vae,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.videoedit_wan import (
    _prepare_clip_pixel_values,
)


class _SentinelImageProcessor:
    def __init__(self, pixel_values: torch.Tensor):
        self.pixel_values = pixel_values
        self.images = None

    def __call__(self, *, images, return_tensors):
        self.images = images
        if return_tensors != "pt":
            raise AssertionError(return_tensors)
        return {"pixel_values": self.pixel_values}


class TestVideoEditClipPreprocess(unittest.TestCase):
    def test_strict_videoedit_rope_matches_original_eager_pair_math(self):
        hidden = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]],
            dtype=torch.bfloat16,
        )
        cos = torch.tensor([[0.75, 0.25]], dtype=torch.float32)
        sin = torch.tensor([[0.5, -0.5]], dtype=torch.float32)

        x1, x2 = hidden.unflatten(-1, (-1, 2)).unbind(-1)
        expected = torch.empty_like(hidden)
        expected[..., 0::2] = x1 * cos[None, :, None, :] - x2 * sin[
            None, :, None, :
        ]
        expected[..., 1::2] = x1 * sin[None, :, None, :] + x2 * cos[
            None, :, None, :
        ]

        actual = _videoedit_apply_rotary_emb(hidden, cos, sin)

        self.assertTrue(torch.equal(actual, expected))

    def test_strict_videoedit_rms_norm_uses_native_reference_order(self):
        class _RecordingNorm:
            def __init__(self):
                self.native_called = False

            def forward_native(self, value):
                self.native_called = True
                return value + 1

            def __call__(self, value):
                raise AssertionError("fused RMSNorm must not run in strict mode")

        norm = _RecordingNorm()
        hidden = torch.zeros(1, 2, 4, dtype=torch.bfloat16)

        actual = _videoedit_rms_norm(norm, hidden)

        self.assertTrue(norm.native_called)
        self.assertTrue(torch.equal(actual, hidden + 1))

    def test_explicit_videoedit_norm_uses_reference_fp32_order(self):
        hidden = torch.tensor([[[1.25, -0.5, 2.0, -3.0]]], dtype=torch.bfloat16)
        shift = torch.tensor([[[0.1, -0.2, 0.3, -0.4]]], dtype=torch.float32)
        scale = torch.tensor([[[0.2, 0.3, -0.1, 0.4]]], dtype=torch.float32)
        norm = torch.nn.LayerNorm(4, eps=1e-6, elementwise_affine=False)

        actual = _explicit_norm_scale_shift(norm, hidden, shift, scale)
        expected = (norm(hidden.float()) * (1 + scale) + shift).to(torch.bfloat16)

        self.assertTrue(torch.equal(actual, expected))

    def test_prompt_clean_matches_original_ftfy_normalization(self):
        self.assertEqual(videoedit_prompt_clean("  中文，test！\nnext  "), "中文,test! next")

    def test_image_encoder_precision_matches_fp32_algorithm_baseline(self):
        self.assertEqual(WanVideoEditPipelineConfig().image_encoder_precision, "fp32")

    def test_videoedit_uses_native_hf_clip_encoder(self):
        sentinel = object()
        with patch(
            "transformers.CLIPVisionModel.from_pretrained", return_value=sentinel
        ) as load:
            actual = _load_videoedit_clip_encoder("/tmp/image_encoder", torch.float32)

        self.assertIs(actual, sentinel)
        load.assert_called_once_with("/tmp/image_encoder", dtype=torch.float32)

    def test_videoedit_uses_native_hf_text_encoder(self):
        sentinel = object()
        with patch(
            "transformers.UMT5EncoderModel.from_pretrained", return_value=sentinel
        ) as load:
            actual = _load_videoedit_text_encoder(
                "/tmp/text_encoder", torch.bfloat16
            )

        self.assertIs(actual, sentinel)
        load.assert_called_once_with(
            "/tmp/text_encoder", dtype=torch.bfloat16, low_cpu_mem_usage=True
        )

    def test_videoedit_uses_native_diffusers_vae(self):
        sentinel = object()
        with patch(
            "diffusers.AutoencoderKLWan.from_pretrained", return_value=sentinel
        ) as load:
            actual = _load_videoedit_vae("/tmp/vae", torch.bfloat16)

        self.assertIs(actual, sentinel)
        load.assert_called_once_with(
            "/tmp/vae", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )

    def test_t5_postprocess_accepts_native_hf_output(self):
        hidden = torch.arange(24, dtype=torch.float32).view(1, 3, 8)
        attention_mask = torch.tensor([[1, 1, 0]])

        actual = t5_postprocess_text(
            SimpleNamespace(last_hidden_state=hidden),
            {"attention_mask": attention_mask},
        )

        self.assertEqual(tuple(actual.shape), (1, 512, 8))
        self.assertTrue(torch.equal(actual[:, :2], hidden[:, :2]))
        self.assertEqual(int(torch.count_nonzero(actual[:, 2:])), 0)

    def test_diffuser_mode_uses_checkpoint_image_processor(self):
        image = Image.fromarray(np.full((8, 12, 3), 127, dtype=np.uint8))
        expected = torch.arange(12, dtype=torch.float32).view(1, 3, 2, 2)
        processor = _SentinelImageProcessor(expected)

        actual = _prepare_clip_pixel_values(
            image,
            clip_preprocess="diffuser",
            image_processor=processor,
            device=torch.device("cpu"),
        )

        self.assertIs(processor.images, image)
        self.assertTrue(torch.equal(actual, expected))

    def test_diffsynth_mode_does_not_require_image_processor(self):
        image = Image.fromarray(np.full((8, 12, 3), 127, dtype=np.uint8))

        actual = _prepare_clip_pixel_values(
            image,
            clip_preprocess="diffsynth",
            image_processor=None,
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(actual.shape), (1, 3, 224, 224))
        self.assertEqual(actual.dtype, torch.float32)

    def test_sampling_params_reject_unknown_clip_preprocess(self):
        with self.assertRaisesRegex(ValueError, "clip_preprocess"):
            WanVideoEditSamplingParams(clip_preprocess="unknown")

    def test_sampling_params_remove_mask_downsample_request_knob(self):
        with self.assertRaisesRegex(TypeError, "mask_downsample_mode"):
            WanVideoEditSamplingParams(mask_downsample_mode="nearest-exact")

    def test_sampling_params_remove_pure_cpu_noise_knobs(self):
        removed = (
            ({"strength": 0.5}, "strength"),
            ({"vary_seed_by_window": True}, "vary_seed_by_window"),
            ({"init_latent_mode": "add_noise"}, "init_latent_mode"),
            ({"generator_device": "cuda"}, "generator_device"),
        )
        for kwargs, message in removed:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(TypeError, message):
                    WanVideoEditSamplingParams(**kwargs)

    def test_sampling_params_enable_teacache_by_default(self):
        self.assertTrue(WanVideoEditSamplingParams().enable_teacache)

    def test_sampling_params_require_reference_for_videoedit_request(self):
        params = WanVideoEditSamplingParams(
            num_frames=1,
            video_input_path="/tmp/video.mp4",
            mask_input_path="/tmp/mask.mp4",
        )
        with self.assertRaisesRegex(ValueError, "reference_image_path"):
            params._validate_with_pipeline_config(WanVideoEditPipelineConfig())


if __name__ == "__main__":
    unittest.main()
