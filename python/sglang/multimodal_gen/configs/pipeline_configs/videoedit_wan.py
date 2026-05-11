# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.wan_videoedit import (
    WanVideoEditConfig,
)
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.t5 import T5Config
from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.wan import t5_postprocess_text


@dataclass
class WanVideoEditPipelineConfig(PipelineConfig):
    task_type: ModelTaskType = ModelTaskType.VIDEO_EDIT
    dit_config: DiTConfig = field(default_factory=WanVideoEditConfig)
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)

    flow_shift: float | None = 5.0
    precision: str = "bf16"
    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    vae_tiling: bool = True
    vae_sp: bool = False
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = (
        field(default_factory=lambda: (t5_postprocess_text,))
    )

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        self.dit_config.arch_config.in_channels = 36
        self.dit_config.arch_config.out_channels = 16
        self.dit_config.arch_config.num_channels_latents = 16
        self.dit_config.arch_config.image_dim = 1280
        self.dit_config.arch_config.added_kv_proj_dim = 5120

    def adjust_num_frames(self, num_frames):
        return num_frames

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        latent_num_frames = (num_frames - 1) // 4 + 1
        return (
            batch_size,
            16,
            latent_num_frames,
            batch.height // 8,
            batch.width // 8,
        )
