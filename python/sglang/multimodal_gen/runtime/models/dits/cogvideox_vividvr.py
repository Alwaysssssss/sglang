# SPDX-License-Identifier: Apache-2.0
import math
from typing import Any

import torch
import torch.nn as nn
from diffusers.models.embeddings import CogVideoXPatchEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

from sglang.multimodal_gen.configs.models.dits.cogvideox import CogVideoXConfig
from sglang.multimodal_gen.runtime.models.dits.cogvideox import CogVideoXTransformer3DModel
from sglang.multimodal_gen.runtime.models.utils import set_weight_attrs
from sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common import (
    Connector,
    build_control_feat_proj,
    zero_module,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _mark_sidecar_params_for_late_load(module: nn.Module) -> None:
    for parameter in module.parameters():
        set_weight_attrs(parameter, {"missing_param_init": "zeros"})


class CogVideoXVividVRTransformer3DModel(CogVideoXTransformer3DModel):
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        config: CogVideoXConfig,
        hf_config: dict[str, Any],
        quant_config=None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config, quant_config=quant_config)
        arch = config.arch_config
        self.connectors = nn.ModuleList(
            [
                Connector(arch.hidden_size, arch.num_attention_heads)
                for _ in range(arch.num_layers)
            ]
        )
        self.control_feat_proj = build_control_feat_proj(
            arch.in_channels,
            arch.time_embed_dim,
        )
        self.control_patch_embed = zero_module(
            CogVideoXPatchEmbed(
                patch_size=arch.patch_size,
                patch_size_t=arch.patch_size_t,
                in_channels=arch.in_channels,
                embed_dim=arch.hidden_size,
                text_embed_dim=arch.text_embed_dim,
                bias=arch.patch_bias,
                sample_width=arch.sample_width,
                sample_height=arch.sample_height,
                sample_frames=arch.sample_frames,
                temporal_compression_ratio=arch.temporal_compression_ratio,
                max_text_seq_length=arch.max_text_seq_length,
                spatial_interpolation_scale=arch.spatial_interpolation_scale,
                temporal_interpolation_scale=arch.temporal_interpolation_scale,
                use_positional_embeddings=False,
                use_learned_positional_embeddings=False,
            )
        )
        _mark_sidecar_params_for_late_load(self.connectors)
        _mark_sidecar_params_for_late_load(self.control_feat_proj)
        _mark_sidecar_params_for_late_load(self.control_patch_embed)

    def load_connectors(self, path: str) -> None:
        state_dict = torch.load(path, map_location="cpu")
        self.connectors.load_state_dict(state_dict, strict=True)

    def load_control_feat_proj(self, path: str) -> None:
        state_dict = torch.load(path, map_location="cpu")
        self.control_feat_proj.load_state_dict(state_dict, strict=True)

    def load_control_patch_embed(self, path: str) -> None:
        state_dict = torch.load(path, map_location="cpu")
        self.control_patch_embed.load_state_dict(state_dict, strict=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: int | float | torch.LongTensor,
        timestep_cond: torch.Tensor | None = None,
        ofs: int | float | torch.LongTensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        control_hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
        return_dict: bool = True,
    ) -> tuple[torch.Tensor] | Transformer2DModelOutput:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        elif attention_kwargs is not None and attention_kwargs.get("scale") is not None:
            logger.warning(
                "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        timesteps = timestep
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        if self.control_patch_embed is not None:
            if channels % 2 != 0:
                raise ValueError(
                    "VividVR transformer expects concatenated noisy/control latents with an even channel count"
                )

            hidden_states, control_states = hidden_states.split(channels // 2, dim=2)
            hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
            hidden_states = self.embedding_dropout(hidden_states)

            batch_size, num_control_frames, control_channels, control_height, control_width = (
                control_states.shape
            )
            control_states = control_states.reshape(
                batch_size * num_control_frames,
                control_channels,
                control_height,
                control_width,
            )
            # The original Vivid-VR code used `emb.repeat(B * F, 1)`, which only
            # behaves correctly for batch size 1. Expand over frames explicitly so
            # multi-item batches keep the intended timestep-to-frame mapping.
            res_emb = emb[:, None, :].expand(batch_size, num_control_frames, -1).reshape(
                batch_size * num_control_frames, -1
            )
            image_only_indicator = torch.ones(
                (batch_size, num_control_frames),
                device=control_states.device,
                dtype=torch.bool,
            )
            for module in self.control_feat_proj:
                control_states = module(control_states, res_emb, image_only_indicator)
            control_states = control_states.reshape(
                batch_size,
                num_control_frames,
                control_channels,
                control_height,
                control_width,
            )
            control_states = self.control_patch_embed(
                encoder_hidden_states,
                control_states,
            )
            hidden_states = hidden_states + control_states
        else:
            hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
            hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    attention_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                )

            if control_hidden_states is not None:
                control_interval = math.ceil(
                    len(self.transformer_blocks) / len(control_hidden_states)
                )
                hidden_states = self.connectors[i](
                    control_hidden_states[i // control_interval],
                    hidden_states,
                ).to(hidden_states.dtype)

        if not self.config.use_rotary_positional_embeddings:
            hidden_states = self.norm_final(hidden_states)
        else:
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(
                batch_size,
                num_frames,
                height // p,
                width // p,
                -1,
                p,
                p,
            )
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size,
                (num_frames + p_t - 1) // p_t,
                height // p,
                width // p,
                -1,
                p_t,
                p,
                p,
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


EntryClass = CogVideoXVividVRTransformer3DModel
