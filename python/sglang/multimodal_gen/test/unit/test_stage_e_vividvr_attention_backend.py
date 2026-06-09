import unittest
from copy import deepcopy
from os import environ
from types import SimpleNamespace

import torch
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock
from torch import nn

from sglang.multimodal_gen.runtime.models.dits.cogvideox_attention_backend import (
    CogVideoXFlashAttnProcessor,
    CogVideoXNativeAttnProcessor,
    enable_cogvideox_qk_norm_rope_fusion,
    _prepare_cogvideox_qkv,
    inspect_cogvideox_qk_norm_rope_fusion,
    enable_cogvideox_qkv_fusion,
    inspect_cogvideox_attention_backend,
    inspect_cogvideox_qkv_fusion,
    normalize_cogvideox_attention_backend,
    set_cogvideox_attention_backend,
)
from sglang.multimodal_gen.runtime.models.dits.cogvideox_operator_fusion import (
    CogVideoXModulationFusedBlock,
    enable_cogvideox_modulation_fusion,
    inspect_cogvideox_modulation_fusion,
)
from sglang.multimodal_gen.runtime.pipelines.vividvr_pipeline import VividVRPipeline
from sglang.multimodal_gen.runtime.pipelines.vividvr_pipeline import (
    _maybe_torch_compile_module,
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


class _CompileTrackingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.compile_invocations: list[dict[str, object]] = []

    def compile(self, **kwargs):
        self.compile_invocations.append(dict(kwargs))


class _DummyCogVideoXBlockModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=128,
                    num_attention_heads=2,
                    attention_head_dim=64,
                    time_embed_dim=64,
                    attention_bias=True,
                    norm_elementwise_affine=True,
                )
            ]
        )


class TestVividVRAttentionBackend(unittest.TestCase):
    @staticmethod
    def _make_image_rotary_emb(
        seq_len: int = 8, head_dim: int = 64
    ) -> tuple[torch.Tensor, torch.Tensor]:
        half_dim = head_dim // 2
        positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.linspace(
            0.0,
            1.0,
            half_dim,
            dtype=torch.float32,
        ).unsqueeze(0)
        phase = positions * frequencies
        return phase.cos().repeat_interleave(2, dim=1), phase.sin().repeat_interleave(
            2, dim=1
        )

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
            SimpleNamespace(
                attention_backend="fa",
                enable_torch_compile=False,
                enable_cogvideox_qkv_fusion=False,
                cogvideox_qkv_fusion_targets="transformer",
            )
        )

        self.assertEqual(inspect_cogvideox_attention_backend(transformer), "fa")
        self.assertEqual(inspect_cogvideox_attention_backend(controlnet), "fa")
        self.assertEqual(debug["attention_backend_transformer"], "fa")
        self.assertEqual(debug["attention_backend_controlnet"], "fa")

    def test_qkv_fusion_matches_unfused_projections(self):
        module = _DummyCogVideoXAttentionModule()
        attn = module.attn.eval()

        hidden_states = torch.randn(1, 8, 128)
        encoder_hidden_states = torch.randn(1, 4, 128)

        expected = _prepare_cogvideox_qkv(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )

        enable_cogvideox_qkv_fusion(module)
        actual = _prepare_cogvideox_qkv(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )

        self.assertEqual(inspect_cogvideox_qkv_fusion(module), "diffusers_fused_linear")
        self.assertIsInstance(attn.to_qkv, nn.Linear)
        self.assertEqual(expected[0], actual[0])
        torch.testing.assert_close(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2])
        torch.testing.assert_close(actual[3], expected[3])

    def test_qk_norm_rope_fusion_matches_unfused_path(self):
        module = _DummyCogVideoXAttentionModule()
        attn = module.attn.eval()

        hidden_states = torch.randn(1, 8, 128)
        encoder_hidden_states = torch.randn(1, 4, 128)
        image_rotary_emb = self._make_image_rotary_emb()

        expected = _prepare_cogvideox_qkv(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        enable_cogvideox_qk_norm_rope_fusion(module)
        actual = _prepare_cogvideox_qkv(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        self.assertEqual(
            inspect_cogvideox_qk_norm_rope_fusion(module),
            "sglang_layernorm+rope_accel",
        )
        self.assertEqual(expected[0], actual[0])
        torch.testing.assert_close(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2])
        torch.testing.assert_close(actual[3], expected[3])

    def test_vividvr_pipeline_applies_qkv_fusion_to_requested_runtime_modules(self):
        pipeline = object.__new__(VividVRPipeline)
        transformer = _PipelineHookModule()
        controlnet = _PipelineHookModule()
        pipeline.modules = {
            "transformer": transformer,
            "controlnet": controlnet,
        }

        pipeline._apply_qkv_fusion(
            SimpleNamespace(
                enable_cogvideox_qkv_fusion=True,
                attention_backend="fa",
                cogvideox_qkv_fusion_targets="transformer",
            )
        )
        debug = pipeline._build_runtime_acceleration_debug(
            SimpleNamespace(
                attention_backend="fa",
                enable_torch_compile=False,
                enable_cogvideox_qkv_fusion=True,
                cogvideox_qkv_fusion_targets="transformer",
            )
        )

        self.assertEqual(
            inspect_cogvideox_qkv_fusion(transformer),
            "diffusers_fused_linear",
        )
        self.assertIsNone(inspect_cogvideox_qkv_fusion(controlnet))
        self.assertEqual(
            debug["qkv_fusion_transformer"], "diffusers_fused_linear"
        )
        self.assertIsNone(debug["qkv_fusion_controlnet"])
        self.assertEqual(debug["qkv_fusion_targets"], ["transformer"])

    def test_vividvr_pipeline_can_apply_qkv_fusion_to_both_runtime_modules(self):
        pipeline = object.__new__(VividVRPipeline)
        transformer = _PipelineHookModule()
        controlnet = _PipelineHookModule()
        pipeline.modules = {
            "transformer": transformer,
            "controlnet": controlnet,
        }

        pipeline._apply_qkv_fusion(
            SimpleNamespace(
                enable_cogvideox_qkv_fusion=True,
                attention_backend="fa",
                cogvideox_qkv_fusion_targets="transformer,controlnet",
            )
        )
        debug = pipeline._build_runtime_acceleration_debug(
            SimpleNamespace(
                attention_backend="fa",
                enable_torch_compile=False,
                enable_cogvideox_qkv_fusion=True,
                cogvideox_qkv_fusion_targets="transformer,controlnet",
            )
        )

        self.assertEqual(
            inspect_cogvideox_qkv_fusion(transformer),
            "diffusers_fused_linear",
        )
        self.assertEqual(
            inspect_cogvideox_qkv_fusion(controlnet),
            "diffusers_fused_linear",
        )
        self.assertEqual(
            debug["qkv_fusion_transformer"], "diffusers_fused_linear"
        )
        self.assertEqual(
            debug["qkv_fusion_controlnet"], "diffusers_fused_linear"
        )
        self.assertEqual(
            debug["qkv_fusion_targets"], ["transformer", "controlnet"]
        )

    def test_vividvr_pipeline_applies_qk_norm_rope_fusion_to_requested_runtime_modules(self):
        pipeline = object.__new__(VividVRPipeline)
        transformer = _PipelineHookModule()
        controlnet = _PipelineHookModule()
        pipeline.modules = {
            "transformer": transformer,
            "controlnet": controlnet,
        }

        pipeline._apply_qk_norm_rope_fusion(
            SimpleNamespace(
                enable_cogvideox_qk_norm_rope_fusion=True,
                attention_backend="fa",
                cogvideox_qk_norm_rope_fusion_targets="transformer",
            )
        )
        debug = pipeline._build_runtime_acceleration_debug(
            SimpleNamespace(
                attention_backend="fa",
                enable_torch_compile=False,
                enable_cogvideox_qk_norm_rope_fusion=True,
                cogvideox_qk_norm_rope_fusion_targets="transformer",
                enable_cogvideox_qkv_fusion=False,
                cogvideox_qkv_fusion_targets="transformer",
                enable_cogvideox_modulation_fusion=False,
                cogvideox_modulation_fusion_targets="transformer",
            )
        )

        self.assertEqual(
            inspect_cogvideox_qk_norm_rope_fusion(transformer),
            "sglang_layernorm+rope_accel",
        )
        self.assertIsNone(inspect_cogvideox_qk_norm_rope_fusion(controlnet))
        self.assertEqual(
            debug["qk_norm_rope_fusion_transformer"],
            "sglang_layernorm+rope_accel",
        )
        self.assertIsNone(debug["qk_norm_rope_fusion_controlnet"])
        self.assertEqual(debug["qk_norm_rope_fusion_targets"], ["transformer"])

    def test_modulation_fused_block_matches_reference(self):
        torch.manual_seed(0)
        device = torch.device("cpu")
        reference_block = CogVideoXBlock(
            dim=128,
            num_attention_heads=2,
            attention_head_dim=64,
            time_embed_dim=64,
            attention_bias=True,
            norm_elementwise_affine=True,
        ).to(device=device).eval()
        fused_block = CogVideoXModulationFusedBlock(deepcopy(reference_block)).to(
            device=device
        ).eval()
        fused_block.norm1_modulation._forward_method = (
            fused_block.norm1_modulation.forward_native
        )
        fused_block.norm2_residual_modulation._forward_method = (
            fused_block.norm2_residual_modulation.forward_native
        )
        fused_block.ff_residual._forward_method = fused_block.ff_residual.forward_native

        hidden_states = torch.randn(1, 8, 128, device=device)
        encoder_hidden_states = torch.randn(1, 4, 128, device=device)
        temb = torch.randn(1, 64, device=device)

        expected_hidden, expected_encoder = reference_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
        )
        actual_hidden, actual_encoder = fused_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
        )

        torch.testing.assert_close(actual_hidden, expected_hidden, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(
            actual_encoder, expected_encoder, rtol=1e-4, atol=1e-5
        )

    def test_vividvr_pipeline_applies_modulation_fusion_to_requested_runtime_modules(self):
        pipeline = object.__new__(VividVRPipeline)
        transformer = _DummyCogVideoXBlockModule()
        controlnet = _DummyCogVideoXBlockModule()
        pipeline.modules = {
            "transformer": transformer,
            "controlnet": controlnet,
        }

        pipeline._apply_modulation_fusion(
            SimpleNamespace(
                enable_cogvideox_modulation_fusion=True,
                cogvideox_modulation_fusion_targets="transformer",
            )
        )
        debug = pipeline._build_runtime_acceleration_debug(
            SimpleNamespace(
                attention_backend=None,
                enable_torch_compile=False,
                enable_cogvideox_modulation_fusion=True,
                cogvideox_modulation_fusion_targets="transformer",
                enable_cogvideox_qkv_fusion=False,
                cogvideox_qkv_fusion_targets="transformer",
            )
        )

        self.assertIsInstance(
            transformer.transformer_blocks[0], CogVideoXModulationFusedBlock
        )
        self.assertIsInstance(controlnet.transformer_blocks[0], CogVideoXBlock)
        self.assertEqual(
            inspect_cogvideox_modulation_fusion(transformer),
            "sglang_modulation_fused_ops",
        )
        self.assertIsNone(inspect_cogvideox_modulation_fusion(controlnet))
        self.assertEqual(
            debug["modulation_fusion_transformer"], "sglang_modulation_fused_ops"
        )
        self.assertIsNone(debug["modulation_fusion_controlnet"])
        self.assertEqual(debug["modulation_fusion_targets"], ["transformer"])

    def test_torch_compile_helper_marks_module_and_is_idempotent(self):
        module = _CompileTrackingModule()
        original_mode = environ.get("SGLANG_TORCH_COMPILE_MODE")
        environ["SGLANG_TORCH_COMPILE_MODE"] = "reduce-overhead"
        try:
            compiled_module = _maybe_torch_compile_module(
                module,
                enabled=True,
                module_name="unit_test_transformer",
            )
            self.assertIs(compiled_module, module)
            self.assertTrue(getattr(module, "_sglang_torch_compile_enabled"))
            self.assertEqual(len(module.compile_invocations), 1)
            self.assertEqual(
                module.compile_invocations[0]["mode"], "reduce-overhead"
            )

            compiled_again = _maybe_torch_compile_module(
                compiled_module,
                enabled=True,
                module_name="unit_test_transformer",
            )
            self.assertIs(compiled_again, compiled_module)
            self.assertEqual(len(module.compile_invocations), 1)
        finally:
            if original_mode is None:
                environ.pop("SGLANG_TORCH_COMPILE_MODE", None)
            else:
                environ["SGLANG_TORCH_COMPILE_MODE"] = original_mode

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required for flash attention parity")
    def test_flash_attention_processor_matches_native(self):
        device = torch.device("cuda:0")
        dtype = torch.bfloat16
        attn = _DummyCogVideoXAttentionModule().attn.to(device=device, dtype=dtype).eval()

        hidden_states = torch.randn(1, 8, 128, device=device, dtype=dtype)
        encoder_hidden_states = torch.randn(1, 4, 128, device=device, dtype=dtype)
        image_rotary_emb = tuple(
            tensor.to(device=device) for tensor in self._make_image_rotary_emb()
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

    @unittest.skipIf(
        not torch.cuda.is_available(), "CUDA is required for flash attention parity"
    )
    def test_flash_attention_processor_matches_native_with_qk_norm_rope_fusion(self):
        device = torch.device("cuda:0")
        dtype = torch.bfloat16
        module = _DummyCogVideoXAttentionModule().to(device=device, dtype=dtype).eval()
        enable_cogvideox_qk_norm_rope_fusion(module)
        attn = module.attn

        hidden_states = torch.randn(1, 8, 128, device=device, dtype=dtype)
        encoder_hidden_states = torch.randn(1, 4, 128, device=device, dtype=dtype)
        image_rotary_emb = tuple(
            tensor.to(device=device) for tensor in self._make_image_rotary_emb()
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
