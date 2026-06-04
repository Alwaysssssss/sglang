import json
import os
import tempfile
import unittest

from sglang.multimodal_gen.configs.pipeline_configs.vividvr import (
    VividVRPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.vividvr import VividVRSamplingParams
from sglang.multimodal_gen.registry import _get_config_info


class TestVividVRRegistry(unittest.TestCase):
    def setUp(self):
        _get_config_info.cache_clear()

    def test_vividvr_keyword_path_resolves_vividvr_config(self):
        with tempfile.TemporaryDirectory(prefix="vivid-vr-local-model-") as model_dir:
            config_path = os.path.join(model_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as fout:
                json.dump({}, fout)

            info = _get_config_info(model_dir)

        self.assertIsNotNone(info)
        self.assertIs(info.pipeline_config_cls, VividVRPipelineConfig)
        self.assertIs(info.sampling_param_cls, VividVRSamplingParams)

    def test_vividvr_pipeline_class_keyword_resolves_vividvr_config(self):
        with tempfile.TemporaryDirectory(prefix="native-cogvideox-stage-a-") as model_dir:
            config_path = os.path.join(model_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as fout:
                json.dump({"_class_name": "CogVideoXVividVRControlNetPipeline"}, fout)

            info = _get_config_info(model_dir)

        self.assertIsNotNone(info)
        self.assertIs(info.pipeline_config_cls, VividVRPipelineConfig)
        self.assertIs(info.sampling_param_cls, VividVRSamplingParams)

    def test_existing_wan_family_still_resolves(self):
        info = _get_config_info("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

        self.assertIsNotNone(info)
        self.assertEqual(info.pipeline_config_cls.__name__, "WanT2V480PConfig")
        self.assertEqual(info.sampling_param_cls.__name__, "WanT2V_1_3B_SamplingParams")


if __name__ == "__main__":
    unittest.main()
