# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

from sglang.multimodal_gen.configs.pipeline_configs.star_cogvideox_sr import (
    StarCogVideoXSRPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.star_cogvideox_sr import (
    StarCogVideoXSRSamplingParams,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    InputValidationStage,
    STARConditionVideoLoadingStage,
    STARConditionVideoVAEEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.star_cogvideox_sr_decoding import (
    STARCogVideoXSRDecodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

INTEGRATION_CONFIG_NAME = "star_integration_config.json"


class StarCogVideoXSRPipeline(LoRAPipeline, ComposedPipelineBase):
    """Modular STAR CogVideoX-SR pipeline."""

    pipeline_name = "StarCogVideoXSRPipeline"
    pipeline_config_cls = StarCogVideoXSRPipelineConfig
    sampling_params_cls = StarCogVideoXSRSamplingParams
    is_video_pipeline = True

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def initialize_pipeline(self, server_args: ServerArgs):
        integration_path = Path(self.model_path) / INTEGRATION_CONFIG_NAME
        if not integration_path.exists():
            return

        try:
            with open(integration_path, encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            logger.warning(
                "Failed to read STAR integration config %s: %s",
                integration_path,
                exc,
            )
            return

        if hasattr(server_args.pipeline_config, "apply_integration_config"):
            server_args.pipeline_config.apply_integration_config(payload)

    def create_pipeline_stages(self, server_args: ServerArgs):
        del server_args
        self.add_stage(InputValidationStage())
        self.add_stage(STARConditionVideoLoadingStage())
        self.add_standard_text_encoding_stage()
        self.add_stage(
            STARConditionVideoVAEEncodingStage(vae=self.get_module("vae"))
        )
        self.add_standard_latent_preparation_stage()
        self.add_standard_timestep_preparation_stage()
        self.add_standard_denoising_stage()
        self.add_stage(
            STARCogVideoXSRDecodingStage(
                vae=self.get_module("vae"),
                pipeline=self,
            )
        )


EntryClass = StarCogVideoXSRPipeline
