# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.t5 import T5Config
from sglang.multimodal_gen.configs.models.dits.star_cogvideox_sr import (
    StarCogVideoXSRDiTConfig,
)
from sglang.multimodal_gen.configs.models.encoders.base import EncoderConfig
from sglang.multimodal_gen.configs.models.vaes.star_cogvideox_vae import (
    StarCogVideoXSRVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)


def star_t5_postprocess_text(
    outputs: BaseEncoderOutput,
    _text_inputs: dict[str, Any],
) -> torch.Tensor:
    return outputs.last_hidden_state


@dataclass
class StarCogVideoXSRPipelineConfig(PipelineConfig):
    """Pipeline configuration for STAR CogVideoX video super-resolution."""

    task_type: ModelTaskType = ModelTaskType.T2V
    should_use_guidance: bool = False

    dit_config: StarCogVideoXSRDiTConfig = field(
        default_factory=StarCogVideoXSRDiTConfig
    )
    vae_config: StarCogVideoXSRVAEConfig = field(
        default_factory=StarCogVideoXSRVAEConfig
    )

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(),)
    )
    postprocess_text_funcs: tuple[
        Callable[[BaseEncoderOutput, dict[str, Any]], torch.Tensor], ...
    ] = field(default_factory=lambda: (star_t5_postprocess_text,))
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp32",))

    precision: str = "bf16"
    vae_precision: str = "fp32"
    vae_tiling: bool = True
    vae_sp: bool = False

    width: int = 720
    height: int = 480
    num_frames: int = 7
    condition_video_num_frames: int = 25
    latent_channels: int = 16

    condition_video_resize_mode: str = "crop"
    enable_color_fix: bool = False
    color_fix_mode: str | None = None
    dynamic_cfg_enabled: bool = True
    dynamic_cfg_exp: float = 5.0
    use_step_index_timestep: bool = False

    integration_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        self._sync_latent_channel_defaults()

    def _sync_latent_channel_defaults(self) -> None:
        if getattr(self.dit_config.arch_config, "num_channels_latents", 0) in (0, None):
            self.dit_config.arch_config.num_channels_latents = self.latent_channels
        if getattr(self.vae_config.arch_config, "z_channels", None) is None:
            self.vae_config.arch_config.z_channels = self.latent_channels

    def apply_integration_config(self, payload: dict[str, Any]) -> None:
        self.integration_metadata = dict(payload)

        latent_channels = payload.get("latent_channels")
        if isinstance(latent_channels, int) and latent_channels > 0:
            self.latent_channels = latent_channels
            self.dit_config.arch_config.num_channels_latents = latent_channels
            self.vae_config.arch_config.z_channels = latent_channels

        default_num_frames = payload.get("default_sampling_num_frames")
        if isinstance(default_num_frames, int) and default_num_frames > 0:
            self.num_frames = default_num_frames

        latent_scale_factor = payload.get("latent_scale_factor")
        if latent_scale_factor not in (None, 0):
            self.vae_config.arch_config.scaling_factor = latent_scale_factor

        dynamic_cfg_exp = payload.get("dynamic_cfg_exp")
        if isinstance(dynamic_cfg_exp, (int, float)):
            self.dynamic_cfg_exp = float(dynamic_cfg_exp)

        transformer_summary = payload.get("transformer_summary") or {}
        latent_width = transformer_summary.get("latent_width")
        latent_height = transformer_summary.get("latent_height")
        spatial_ratio = (
            getattr(self.vae_config.arch_config, "spatial_compression_ratio", None)
            or 8
        )
        if isinstance(latent_width, int) and latent_width > 0:
            self.width = latent_width * spatial_ratio
        if isinstance(latent_height, int) and latent_height > 0:
            self.height = latent_height * spatial_ratio

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        height = batch.height // self.vae_config.arch_config.spatial_compression_ratio
        width = batch.width // self.vae_config.arch_config.spatial_compression_ratio
        num_channels = (
            getattr(self.dit_config.arch_config, "num_channels_latents", 0)
            or self.latent_channels
        )
        return (batch_size, num_channels, num_frames, height, width)

    def postprocess_image_latent(self, latent_condition, batch):
        del batch
        return latent_condition

    def get_decode_scale_and_shift(self, device, dtype, vae):
        del device, dtype
        scaling_factor = getattr(self.vae_config.arch_config, "scaling_factor", None)
        if scaling_factor in (None, 0):
            scaling_factor = self.integration_metadata.get("latent_scale_factor")
        if scaling_factor in (None, 0):
            scaling_factor = getattr(vae, "scaling_factor", None)
        if scaling_factor in (None, 0):
            scaling_factor = 1.0

        shift_factor = getattr(self.vae_config.arch_config, "shift_factor", None)
        if shift_factor is None:
            shift_factor = getattr(vae, "shift_factor", None)
        return scaling_factor, shift_factor

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        del rotary_emb, dtype
        kwargs = {}
        if batch.prompt_attention_mask:
            kwargs["encoder_attention_mask"] = batch.prompt_attention_mask[0].to(
                device=device
            )
        return kwargs

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        del rotary_emb, dtype
        kwargs = {}
        if batch.negative_attention_mask:
            kwargs["encoder_attention_mask"] = batch.negative_attention_mask[0].to(
                device=device
            )
        return kwargs

    def get_text_encoder_attention_mask(self, text_inputs, encoder_index):
        del text_inputs, encoder_index
        # STAR's FrozenT5Embedder calls T5EncoderModel with input_ids only.
        return None

    def should_force_zero_unconditional_text_embeddings(self) -> bool:
        return True

    def get_classifier_free_guidance_scale_for_step(
        self,
        batch,
        guidance_scale: float,
        timestep_index: int,
        scheduler_timestep: int | None = None,
    ) -> float:
        if not self.dynamic_cfg_enabled:
            return guidance_scale
        num_steps = max(int(batch.num_inference_steps), 1)
        if scheduler_timestep is None:
            scheduler_timestep = int(timestep_index)
        progress = (num_steps - int(scheduler_timestep)) / num_steps
        return 1.0 + guidance_scale * (
            1.0 - math.cos(math.pi * (progress**self.dynamic_cfg_exp))
        ) / 2.0

    def expand_timestep_before_forward_for_step(
        self,
        batch,
        t_device,
        target_dtype,
        seq_len,
        reserved_frames_mask,
        batch_size: int,
        timestep_index: int,
    ):
        if not self.use_step_index_timestep:
            return None
        del target_dtype, seq_len, reserved_frames_mask
        remaining_steps = max(int(batch.num_inference_steps) - int(timestep_index), 1)
        return torch.full(
            (batch_size,),
            remaining_steps,
            dtype=t_device.dtype,
            device=t_device.device,
        )
