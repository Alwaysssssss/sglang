import unittest
from types import SimpleNamespace

import torch
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0
from torch import nn

from sglang.multimodal_gen.runtime.pipelines.vividvr_pipeline import VividVRPipeline
from sglang.multimodal_gen.runtime.models.dits.cogvideox_attention_backend import (
    CogVideoXFlashAttnProcessor,
    CogVideoXNativeAttnProcessor,
    inspect_cogvideox_attention_backend,
    normalize_cogvideox_attention_backend,
    set_cogvideox_attention_backend,
)


class _DummyCogVideoXAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention(
            query_dim=128,
            dim_head=64,
            heads=2,
            qk_norm="layer_norm",
            eps=1e-6,
            bias=True,
            out_bias=True,
            processor=CogVideoXAttnProcessor2_0(),
        )


class _PipelineHookModule(_DummyCogVideoXAttentionModule):
    def set_attention_backend(self, backend: str) -> None:
        set_cogvideox_attention_backend(self, backend)


class TestVividVRAttentionBackend(unittest.TestCase):
    def test_normalize_attention_backend_aliases(self):
        self.assertEqual(normalize_cogvideox_attention_backend("fa3"), "fa")
        self.assertEqual(normalize_cogvideox_attention_backend("flash"), "fa")
        self.assertEqual(normalize_cogvideox_attention_backend("torch_sdpa"), "native")
        self.assertEqual(
            normalize_cogvideox_attention_backend("sage_attn"), "sage_attn"
        )

    def test_set_attention_backend_replaces_processors(self):
        module = _DummyCogVideoXAttentionModule()

        set_cogvideox_attention_backend(module, "fa")
        self.assertEqual(inspect_cogvideox_attention_backend(module), "fa")
        self.assertIsInstance(module.attn.processor, CogVideoXFlashAttnProcessor)

        set_cogvideox_attention_backend(module, "torch_sdpa")
        self.assertEqual(inspect_cogvideox_attention_backend(module), "native")
        self.assertIsInstance(module.attn.processor, CogVideoXNativeAttnProcessor)

    def test_unsupported_attention_backend_raises(self):
        module = _DummyCogVideoXAttentionModule()
        with self.assertRaisesRegex(ValueError, "not supported yet"):
            set_cogvideox_attention_backend(module, "sage_attn")

    def test_vividvr_pipeline_applies_backend_to_runtime_modules(self):
        pipeline = object.__new__(VividVRPipeline)
        transformer = _PipelineHookModule()
        controlnet = _PipelineHookModule()
        pipeline.modules = {
            "transformer": transformer,
            "controlnet": controlnet,
        }

        pipeline._apply_attention_backend(SimpleNamespace(attention_backend="fa"))
        debug = pipeline._build_runtime_acceleration_debug(
            SimpleNamespace(attention_backend="fa")
        )

        self.assertEqual(inspect_cogvideox_attention_backend(transformer), "fa")
        self.assertEqual(inspect_cogvideox_attention_backend(controlnet), "fa")
        self.assertEqual(debug["attention_backend_transformer"], "fa")
        self.assertEqual(debug["attention_backend_controlnet"], "fa")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required for flash attention parity")
    def test_flash_attention_processor_matches_native(self):
        device = torch.device("cuda:0")
        dtype = torch.bfloat16
        attn = _DummyCogVideoXAttentionModule().attn.to(device=device, dtype=dtype).eval()

        hidden_states = torch.randn(1, 8, 128, device=device, dtype=dtype)
        encoder_hidden_states = torch.randn(1, 4, 128, device=device, dtype=dtype)
        image_rotary_emb = (
            torch.ones(8, 64, device=device, dtype=torch.float32),
            torch.zeros(8, 64, device=device, dtype=torch.float32),
        )

        native_hidden, native_encoder = CogVideoXNativeAttnProcessor()(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        flash_hidden, flash_encoder = CogVideoXFlashAttnProcessor()(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        torch.testing.assert_close(
            flash_hidden.float(),
            native_hidden.float(),
            rtol=1e-2,
            atol=1e-2,
        )
        torch.testing.assert_close(
            flash_encoder.float(),
            native_encoder.float(),
            rtol=1e-2,
            atol=1e-2,
        )


if __name__ == "__main__":
    unittest.main()
