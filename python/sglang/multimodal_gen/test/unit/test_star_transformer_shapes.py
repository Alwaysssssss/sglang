import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.models.dits.star_cogvideox_sr import (
    StarCogVideoXSRArchConfig,
    StarCogVideoXSRDiTConfig,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.models.dits.star_cogvideox_sr import (
    _SpatialLocalEnhancer,
    _TemporalLocalEnhancer,
    _StarTransformerLayer,
    _build_flashinfer_rotary_cache,
    StarCogVideoXSRTransformer3DModel,
)


class _FakeQuantMethod(QuantizeMethodBase):
    def __init__(self, seen_prefixes: list[str]) -> None:
        self.seen_prefixes = seen_prefixes
        self._fallback = UnquantizedLinearMethod()

    def create_weights(self, layer, *weight_args, **extra_weight_attrs):
        self.seen_prefixes.append(layer.prefix)
        self._fallback.create_weights(layer, *weight_args, **extra_weight_attrs)

    def apply(self, layer, *args, **kwargs):
        return self._fallback.apply(layer, *args, **kwargs)


class _FakeQuantConfig(QuantizationConfig):
    def __init__(self) -> None:
        super().__init__()
        self.seen_prefixes: list[str] = []

    @classmethod
    def get_name(cls):
        return "fake_star_quant"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        del config
        return cls()

    def get_quant_method(self, layer, prefix: str):
        del layer, prefix
        return _FakeQuantMethod(self.seen_prefixes)


class TestStarTransformerShapes(unittest.TestCase):
    def setUp(self):
        self._attn_args_patch = patch(
            "sglang.multimodal_gen.runtime.layers.attention.selector.get_global_server_args",
            return_value=SimpleNamespace(attention_backend="torch_sdpa"),
        )
        self._attn_args_patch.start()

    def tearDown(self):
        self._attn_args_patch.stop()

    def test_forward_accepts_concat_latents_and_text_states(self):
        config = StarCogVideoXSRDiTConfig(
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
                latent_width=8,
                latent_height=8,
                num_frames=5,
                time_compressed_rate=2,
                time_embed_dim=32,
            )
        )
        model = StarCogVideoXSRTransformer3DModel(
            config=config,
            hf_config={},
        )

        hidden_states = torch.randn(2, 8, 3, 8, 8)
        encoder_hidden_states = [torch.randn(2, 6, 32)]
        timestep = torch.tensor([31, 31], dtype=torch.long)

        with set_forward_context(current_timestep=0, attn_metadata=None):
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
            )

        self.assertEqual(tuple(output.shape), (2, 4, 3, 8, 8))
        self.assertEqual(output.dtype, hidden_states.dtype)

    def test_final_layernorm_participates_in_forward(self):
        config = StarCogVideoXSRDiTConfig(
            arch_config=StarCogVideoXSRArchConfig(
                hidden_size=64,
                num_attention_heads=4,
                num_layers=1,
                in_channels=4,
                out_channels=4,
                num_channels_latents=4,
                patch_size=2,
                text_hidden_size=32,
                text_length=6,
                latent_width=8,
                latent_height=8,
                num_frames=5,
                time_compressed_rate=2,
                time_embed_dim=32,
            )
        )
        model = StarCogVideoXSRTransformer3DModel(
            config=config,
            hf_config={},
        )
        model.eval()

        hidden_states = torch.randn(1, 8, 3, 8, 8)
        encoder_hidden_states = [torch.randn(1, 6, 32)]
        timestep = torch.tensor([31], dtype=torch.long)

        with torch.no_grad():
            with set_forward_context(current_timestep=0, attn_metadata=None):
                baseline = model(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                )
            model.transformer.final_layernorm.weight.zero_()
            model.transformer.final_layernorm.bias.zero_()
            with set_forward_context(current_timestep=0, attn_metadata=None):
                changed = model(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                )

        self.assertFalse(torch.allclose(baseline, changed))

    def test_local_enhancers_5d_path_matches_legacy_semantics(self):
        torch.manual_seed(0)
        spatial = _SpatialLocalEnhancer(kernel_size=3)
        temporal = _TemporalLocalEnhancer()

        batch_size, channels, num_frames, grid_h, grid_w = 2, 8, 3, 4, 5
        video_hidden = torch.randn(batch_size, channels, num_frames, grid_h, grid_w)

        optimized = temporal(spatial(video_hidden))

        legacy_spatial = video_hidden.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, grid_h, grid_w
        )
        legacy_spatial = spatial(legacy_spatial)
        legacy_temporal = legacy_spatial.view(
            batch_size, num_frames, channels, grid_h, grid_w
        ).permute(0, 3, 4, 1, 2).reshape(batch_size * grid_h * grid_w, num_frames, channels)
        legacy_temporal = temporal(legacy_temporal)
        legacy = legacy_temporal.view(
            batch_size, grid_h, grid_w, num_frames, channels
        ).permute(0, 4, 3, 1, 2)

        self.assertTrue(torch.allclose(optimized, legacy, atol=1e-6, rtol=1e-6))

    def test_flashinfer_rotary_cache_uses_half_dim_cos_sin_layout(self):
        freqs_sin = torch.tensor([[1.0, 1.0, 2.0, 2.0]])
        freqs_cos = torch.tensor([[3.0, 3.0, 4.0, 4.0]])
        cache = _build_flashinfer_rotary_cache(freqs_sin, freqs_cos)
        expected = torch.tensor([[3.0, 4.0, 1.0, 2.0]], dtype=torch.float32)
        self.assertTrue(torch.equal(cache, expected))

    def test_apply_local_enhancers_fused_5d_matches_legacy_path(self):
        torch.manual_seed(0)
        layer = SimpleNamespace(
            spa_local=_SpatialLocalEnhancer(kernel_size=3),
            temp_local=_TemporalLocalEnhancer(),
            local_enhancer_mode="fused_5d",
        )
        hidden = torch.randn(2, 3 * 4 * 5, 8)

        fused = _StarTransformerLayer._apply_local_enhancers(
            hidden,
            layer,
            num_frames=3,
            grid_h=4,
            grid_w=5,
        )
        layer.local_enhancer_mode = "legacy"
        legacy = _StarTransformerLayer._apply_local_enhancers(
            hidden,
            layer,
            num_frames=3,
            grid_h=4,
            grid_w=5,
        )

        self.assertTrue(torch.allclose(fused, legacy, atol=1e-6, rtol=1e-6))

    def test_single_gpu_quant_config_reaches_replicated_linear_hot_path(self):
        quant_config = _FakeQuantConfig()
        config = StarCogVideoXSRDiTConfig(
            arch_config=StarCogVideoXSRArchConfig(
                hidden_size=64,
                num_attention_heads=4,
                num_layers=1,
                in_channels=4,
                out_channels=4,
                num_channels_latents=4,
                patch_size=2,
                text_hidden_size=32,
                text_length=6,
                latent_width=8,
                latent_height=8,
                num_frames=5,
                time_compressed_rate=2,
                time_embed_dim=32,
            )
        )

        model = StarCogVideoXSRTransformer3DModel(
            config=config,
            hf_config={},
            quant_config=quant_config,
        )

        self.assertIn("time_embed.0", quant_config.seen_prefixes)
        self.assertIn(
            "transformer.layers.0.attention.query_key_value.original",
            quant_config.seen_prefixes,
        )
        self.assertIn(
            "transformer.layers.0.mlp.dense_h_to_4h",
            quant_config.seen_prefixes,
        )
        self.assertIn("mixins.final_layer.linear", quant_config.seen_prefixes)
        self.assertNotIsInstance(
            model.transformer.layers[0].attention.query_key_value.original.quant_method,
            UnquantizedLinearMethod,
        )


if __name__ == "__main__":
    unittest.main()
