import inspect
import unittest
from unittest.mock import patch

import torch

from cache_dit import ForwardPattern

from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
    CacheDitConfig,
    _is_wan_videoedit_transformer,
    enable_cache_on_transformer,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    videoedit_wan,
)

CACHE_DIT_INTEGRATION = (
    "sglang.multimodal_gen.runtime.cache.cache_dit_integration"
)


class WanVideoEditTransformer3DModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Identity()])


class NotVideoEditTransformer(torch.nn.Module):
    pass


class FakeBlockAdapter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class TestVideoEditCacheDit(unittest.TestCase):
    def test_videoedit_transformer_helper_matches_exact_class_name(self):
        self.assertTrue(_is_wan_videoedit_transformer(WanVideoEditTransformer3DModel()))
        self.assertFalse(_is_wan_videoedit_transformer(NotVideoEditTransformer()))

    def test_cache_dit_register_supports_wan_videoedit_prefix(self):
        from cache_dit.caching.block_adapters import BlockAdapterRegister

        self.assertTrue(
            BlockAdapterRegister.is_supported(WanVideoEditTransformer3DModel())
        )

    def test_fallback_adapter_uses_wan_single_transformer_settings(self):
        transformer = WanVideoEditTransformer3DModel()
        config = CacheDitConfig(enabled=True, num_inference_steps=4)

        with (
            patch(
                f"{CACHE_DIT_INTEGRATION}.BlockAdapterRegister.is_supported",
                return_value=False,
            ),
            patch(
                f"{CACHE_DIT_INTEGRATION}.BlockAdapter",
                FakeBlockAdapter,
            ),
            patch(f"{CACHE_DIT_INTEGRATION}.cache_dit.enable_cache") as enable_cache,
        ):
            self.assertIs(enable_cache_on_transformer(transformer, config), transformer)

        adapter = enable_cache.call_args.args[0]
        self.assertIsInstance(adapter, FakeBlockAdapter)
        self.assertIs(adapter.kwargs["transformer"], transformer)
        self.assertIs(adapter.kwargs["blocks"], transformer.blocks)
        self.assertIs(adapter.kwargs["forward_pattern"], ForwardPattern.Pattern_2)
        self.assertTrue(adapter.kwargs["check_forward_pattern"])
        self.assertTrue(adapter.kwargs["has_separate_cfg"])
        self.assertIn("params_modifiers", adapter.kwargs)
        self.assertNotIn("params_modifier", adapter.kwargs)
        self.assertIsNone(enable_cache.call_args.kwargs["parallelism_config"])

    def test_videoedit_sets_cfg_state_before_cache_dit(self):
        source = inspect.getsource(videoedit_wan.VideoEditDenoisingStage.forward)

        cache_dit_index = source.index("self._maybe_enable_cache_dit")
        self.assertLess(
            source.index("batch.do_classifier_free_guidance"), cache_dit_index
        )
        self.assertLess(source.index("batch.is_cfg_negative = False"), cache_dit_index)


if __name__ == "__main__":
    unittest.main()
