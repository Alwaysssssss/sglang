import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.configs.models.dits.star_cogvideox_sr import (
    StarCogVideoXSRArchConfig,
    StarCogVideoXSRDiTConfig,
)
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.models.dits.star_cogvideox_sr import (
    StarCogVideoXSRTransformer3DModel,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


class TestStarPhase6Integrations(unittest.TestCase):
    def setUp(self):
        self._attn_args_patch = patch(
            "sglang.multimodal_gen.runtime.layers.attention.selector.get_global_server_args",
            return_value=SimpleNamespace(attention_backend="fa"),
        )
        self._attn_args_patch.start()

    def tearDown(self):
        self._attn_args_patch.stop()

    def _build_model(self) -> StarCogVideoXSRTransformer3DModel:
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
        return StarCogVideoXSRTransformer3DModel(config=config, hf_config={})

    def test_supported_attention_backends_include_fa(self):
        self.assertIn(
            AttentionBackendEnum.FA,
            StarCogVideoXSRTransformer3DModel._supported_attention_backends,
        )
        self.assertIn(
            AttentionBackendEnum.TORCH_SDPA,
            StarCogVideoXSRTransformer3DModel._supported_attention_backends,
        )

    def test_cfg_teacache_support_is_enabled_for_star_prefix(self):
        model = self._build_model()
        self.assertTrue(model._supports_cfg_cache)

    def test_single_gpu_attention_uses_local_attention_wrapper(self):
        model = self._build_model()
        attn = model.transformer.layers[0].attention.attn
        self.assertIsInstance(attn, LocalAttention)

    def test_cache_dit_adapter_is_registered_when_available(self):
        try:
            from cache_dit.caching.block_adapters import BlockAdapterRegister
        except Exception:
            self.skipTest("cache_dit is not available")

        model = self._build_model()
        self.assertTrue(BlockAdapterRegister.is_supported(model))
        adapter = BlockAdapterRegister.get_adapter(model, skip_post_init=True)
        self.assertIsNotNone(adapter)
        assert adapter is not None
        self.assertTrue(adapter.has_separate_cfg)
        self.assertIs(adapter.blocks, model.transformer.layers)


if __name__ == "__main__":
    unittest.main()
