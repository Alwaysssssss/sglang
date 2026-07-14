import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from sglang.multimodal_gen.configs.sample.videoedit_wan import (
    WanVideoEditSamplingParams,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.videoedit_wan import (
    VideoEditImageEncodingStage,
)


class _FakeImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            hidden_states=[
                torch.zeros(1, 1, 1),
                torch.full((1, 2, 3), 2.0),
                torch.full((1, 2, 3), 3.0),
            ]
        )


class _FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))


class _FakeBatch(dict):
    def __init__(self, pixel_values):
        super().__init__(pixel_values=pixel_values)
        self.to_device = None

    def to(self, device):
        self.to_device = device
        return self


class _FakeImageProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, *, images, return_tensors):
        self.calls.append((images.size, return_tensors))
        return _FakeBatch(torch.ones(1, 3, 5, 7))


class _ExplodingImageProcessor:
    def __call__(self, **kwargs):
        raise AssertionError("image_processor should not be called")


def _server_args():
    return SimpleNamespace(
        image_encoder_cpu_offload=False,
        disable_autocast=False,
        pipeline_config=SimpleNamespace(
            dit_precision="fp32",
            image_encoder_extra_args={"output_hidden_states": True},
            postprocess_image=lambda outputs: outputs.hidden_states[-2],
        ),
    )


def _batch(clip_preprocess):
    params = WanVideoEditSamplingParams(clip_preprocess=clip_preprocess)
    params.runtime_window_frames = [Image.new("RGB", (4, 6), color=(10, 20, 30))]
    params.runtime_height = 16
    params.runtime_width = 32
    return Req(sampling_params=params)


def _stage(*, image_encoder, image_processor, transformer):
    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args",
        return_value=SimpleNamespace(comfyui_mode=False),
    ):
        return VideoEditImageEncodingStage(
            image_encoder=image_encoder,
            image_processor=image_processor,
            transformer=transformer,
        )


class TestVideoEditClipPreprocess(unittest.TestCase):
    def test_diffuser_clip_preprocess_uses_image_processor(self):
        image_encoder = _FakeImageEncoder()
        image_processor = _FakeImageProcessor()
        stage = _stage(
            image_encoder=image_encoder,
            image_processor=image_processor,
            transformer=_FakeTransformer(),
        )
        batch = _batch("diffuser")

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.videoedit_wan.get_local_torch_device",
            return_value=torch.device("cpu"),
        ):
            stage.forward(batch, _server_args())

        self.assertEqual(image_processor.calls, [((32, 16), "pt")])
        self.assertEqual(
            tuple(image_encoder.calls[0]["pixel_values"].shape), (1, 3, 5, 7)
        )
        self.assertTrue(image_encoder.calls[0]["output_hidden_states"])
        self.assertTrue(
            torch.equal(
                batch.sampling_params.runtime_image_embeds,
                torch.full((1, 2, 3), 2.0),
            )
        )

    def test_diffsynth_clip_preprocess_uses_hand_rolled_pixels(self):
        image_encoder = _FakeImageEncoder()
        stage = _stage(
            image_encoder=image_encoder,
            image_processor=_ExplodingImageProcessor(),
            transformer=_FakeTransformer(),
        )
        batch = _batch("diffsynth")

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.videoedit_wan.get_local_torch_device",
            return_value=torch.device("cpu"),
        ):
            stage.forward(batch, _server_args())

        pixel_values = image_encoder.calls[0]["pixel_values"]
        self.assertEqual(tuple(pixel_values.shape), (1, 3, 224, 224))
        self.assertTrue(image_encoder.calls[0]["output_hidden_states"])
        self.assertTrue(
            torch.equal(
                batch.sampling_params.runtime_image_embeds,
                torch.full((1, 2, 3), 2.0),
            )
        )

    def test_diffuser_clip_preprocess_requires_image_processor(self):
        stage = _stage(
            image_encoder=_FakeImageEncoder(),
            image_processor=None,
            transformer=_FakeTransformer(),
        )
        batch = _batch("diffuser")

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.videoedit_wan.get_local_torch_device",
            return_value=torch.device("cpu"),
        ):
            with self.assertRaisesRegex(ValueError, "requires an image_processor"):
                stage.forward(batch, _server_args())


if __name__ == "__main__":
    unittest.main()
