import os
import sys
import unittest
from unittest.mock import patch

from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.registry import _get_config_info
from sglang.multimodal_gen.runtime.server_args import Backend, ServerArgs
from sglang.multimodal_gen.utils import FlexibleArgumentParser


class TestServerArgsPathExpansion(unittest.TestCase):
    def _from_dict_without_model_resolution(self, kwargs):
        with patch.object(
            PipelineConfig, "from_kwargs", return_value=QwenImagePipelineConfig()
        ):
            return ServerArgs.from_dict(kwargs)

    def test_tilde_model_path_is_expanded(self):
        args = self._from_dict_without_model_resolution(
            {"model_path": "~/fake/local/model"}
        )
        expected = os.path.expanduser("~/fake/local/model")
        self.assertEqual(args.model_path, expected)
        self.assertFalse(args.model_path.startswith("~"))

    def test_absolute_path_is_unchanged(self):
        args = self._from_dict_without_model_resolution(
            {"model_path": "/data/my-model"}
        )
        self.assertEqual(args.model_path, "/data/my-model")

    def test_component_paths_are_expanded_before_pipeline_resolution(self):
        args = self._from_dict_without_model_resolution(
            {
                "model_path": "/data/my-model",
                "component_paths": {"vae": "~/fake/local/vae"},
            }
        )

        self.assertEqual(
            args.component_paths["vae"], os.path.expanduser("~/fake/local/vae")
        )


class TestModelIdResolution(unittest.TestCase):
    def setUp(self):
        _get_config_info.cache_clear()

    def test_model_id_overrides_arbitrary_local_path(self):
        # a local path whose directory name does not match any HF repo name;
        # --model-id tells the engine which config to use
        info = _get_config_info("/data/my-custom-qwen", model_id="Qwen-Image")
        self.assertIsNotNone(info)

        self.assertIs(info.pipeline_config_cls, QwenImagePipelineConfig)

    def test_model_id_works_after_tilde_expansion(self):
        # simulate the full flow: user passes ~/..., engine expands and resolves
        expanded = os.path.expanduser("~/.cache/huggingface/hub/bbb/snapshots/ccc")
        _get_config_info.cache_clear()
        info = _get_config_info(expanded, model_id="Qwen-Image")
        self.assertIsNotNone(info)

    def test_hf_cache_snapshot_path_resolves_registered_nvfp4_model(self):
        path = (
            "/root/.cache/huggingface/hub/"
            "models--black-forest-labs--FLUX.2-dev-NVFP4/"
            "snapshots/142b87e70bc3006937b7093d89ff287b5f59f071"
        )
        info = _get_config_info(path)
        self.assertIsNotNone(info)

    def test_model_id_unknown_falls_back_without_crash(self):
        # unrecognized model_id: should warn and fall back to path-based detection
        # with an unresolvable path, expect RuntimeError from the detector step
        with self.assertRaises((RuntimeError, Exception)):
            _get_config_info("/data/no-such-model", model_id="NonExistentModelXYZ")


class TestPipelineResolutionCliOverride(unittest.TestCase):
    def setUp(self):
        _get_config_info.cache_clear()

    def test_resolution_flag_overrides_qwen_image_layered_pipeline_config(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        argv = [
            "--model-path",
            "Qwen/Qwen-Image-Layered",
            "--resolution",
            "768",
        ]

        with patch.object(sys, "argv", ["sglang"] + argv):
            args, unknown_args = parser.parse_known_args(argv)
            server_args = ServerArgs.from_cli_args(args, unknown_args)

        self.assertEqual(server_args.pipeline_config.resolution, 768)


class TestComponentPathParsing(unittest.TestCase):
    def test_extract_component_paths_accepts_config_expanded_keys(self):
        component_paths, remaining = ServerArgs._extract_component_paths(
            [
                "--component-paths.spatial-upsampler",
                "/tmp/latent_upsampler",
                "--component_paths.distilled-lora=/tmp/distilled.safetensors",
            ]
        )

        self.assertEqual(
            component_paths,
            {
                "spatial_upsampler": "/tmp/latent_upsampler",
                "distilled_lora": "/tmp/distilled.safetensors",
            },
        )
        self.assertEqual(remaining, [])


class TestOffloadValidation(unittest.TestCase):
    def test_cache_dit_conflicts_with_dit_layerwise_offload(self):
        args = object.__new__(ServerArgs)
        args.dit_layerwise_offload = True
        args.dit_offload_prefetch_size = 0
        args.use_fsdp_inference = False
        args.dit_cpu_offload = False

        with patch.dict(os.environ, {"SGLANG_CACHE_DIT_ENABLED": "true"}):
            with self.assertRaisesRegex(ValueError, "cache-dit"):
                args._validate_offload()


class TestVideoEditAttentionBackend(unittest.TestCase):
    @staticmethod
    def _args(self_backend, cross_backend=None):
        args = object.__new__(ServerArgs)
        args.attention_backend = None
        args.attention_backend_config = None
        args.videoedit_self_attention_backend = self_backend
        args.videoedit_cross_attention_backend = cross_backend
        args.ring_degree = 1
        args.backend = Backend.DIFFUSERS
        return args

    def test_sage_self_attention_defaults_cross_attention_to_flash(self):
        args = self._args("SAGE_ATTN")

        args._adjust_attention_backend()

        self.assertEqual(args.videoedit_self_attention_backend, "sage_attn")
        self.assertEqual(args.videoedit_cross_attention_backend, "fa")

    def test_rejects_unknown_videoedit_attention_backend(self):
        args = self._args("unknown")

        with self.assertRaisesRegex(ValueError, "must be one of"):
            args._adjust_attention_backend()


if __name__ == "__main__":
    unittest.main()
