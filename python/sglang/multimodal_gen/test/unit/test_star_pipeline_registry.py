import json
import tempfile
import unittest
from pathlib import Path

from sglang.multimodal_gen.configs.pipeline_configs.star_cogvideox_sr import (
    StarCogVideoXSRPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.star_cogvideox_sr import (
    StarCogVideoXSRSamplingParams,
)
from sglang.multimodal_gen.registry import (
    get_model_info,
    get_pipeline_config_classes,
)


def _write_model_index(path: Path) -> None:
    payload = {
        "_class_name": "StarCogVideoXSRPipeline",
        "_diffusers_version": "0.0.0",
        "transformer": ["sglang", "StarCogVideoXSRTransformer3DModel"],
        "vae": ["sglang", "StarCogVideoXSRVAE"],
        "text_encoder": ["transformers", "T5EncoderModel"],
        "tokenizer": ["transformers", "T5Tokenizer"],
        "scheduler": ["sglang", "StarVPSDEDPMPP2MScheduler"],
    }
    with open(path / "model_index.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)
    for component_name in (
        "transformer",
        "vae",
        "text_encoder",
        "tokenizer",
        "scheduler",
    ):
        (path / component_name).mkdir(parents=True, exist_ok=True)


class TestStarPipelineRegistry(unittest.TestCase):
    def test_pipeline_config_classes_are_discoverable(self):
        config_classes = get_pipeline_config_classes("StarCogVideoXSRPipeline")
        self.assertIsNotNone(config_classes)
        pipeline_config_cls, sampling_params_cls = config_classes
        self.assertIs(pipeline_config_cls, StarCogVideoXSRPipelineConfig)
        self.assertIs(sampling_params_cls, StarCogVideoXSRSamplingParams)

    def test_get_model_info_resolves_converted_star_directory(self):
        with tempfile.TemporaryDirectory() as tempdir:
            model_dir = Path(tempdir)
            _write_model_index(model_dir)

            model_info = get_model_info(str(model_dir), backend="sglang")

            self.assertIsNotNone(model_info)
            self.assertEqual(
                model_info.pipeline_cls.pipeline_name,
                "StarCogVideoXSRPipeline",
            )
            self.assertIs(model_info.pipeline_config_cls, StarCogVideoXSRPipelineConfig)
            self.assertIs(model_info.sampling_param_cls, StarCogVideoXSRSamplingParams)


if __name__ == "__main__":
    unittest.main()
