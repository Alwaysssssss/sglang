# SPDX-License-Identifier: Apache-2.0
import math
from typing import Any

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

from sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr import (
    CogVideoXVividVRTransformer3DModel,
)
from sglang.multimodal_gen.runtime.models.dits.cogvideox_attention_backend import (
    inspect_cogvideox_attention_backend,
    set_cogvideox_attention_backend,
)
from sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common import (
    build_vividvr_connector_control_states,
    build_control_feat_proj,
    shard_vividvr_video_tokens,
    zero_module,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CogVideoXVividVRControlNetModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: int | None = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        ofs_embed_dim: int | None = None,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        patch_size_t: int | None = None,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        patch_bias: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disabled rotary embeddings and learned positional embeddings."
            )

        self.hidden_size = inner_dim
        self.num_attention_heads = num_attention_heads
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels

        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(
            inner_dim,
            time_embed_dim,
            timestep_activation_fn,
        )

        self.ofs_proj = None
        self.ofs_embedding = None
        if ofs_embed_dim:
            self.ofs_proj = Timesteps(ofs_embed_dim, flip_sin_to_cos, freq_shift)
            self.ofs_embedding = TimestepEmbedding(
                ofs_embed_dim,
                ofs_embed_dim,
                timestep_activation_fn,
            )

        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=patch_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.control_feat_proj = build_control_feat_proj(in_channels, time_embed_dim)
        self.control_patch_embed = zero_module(
            CogVideoXPatchEmbed(
                patch_size=patch_size,
                patch_size_t=patch_size_t,
                in_channels=in_channels,
                embed_dim=inner_dim,
                text_embed_dim=text_embed_dim,
                bias=patch_bias,
                sample_width=sample_width,
                sample_height=sample_height,
                sample_frames=sample_frames,
                temporal_compression_ratio=temporal_compression_ratio,
                max_text_seq_length=max_text_seq_length,
                spatial_interpolation_scale=spatial_interpolation_scale,
                temporal_interpolation_scale=temporal_interpolation_scale,
                use_positional_embeddings=False,
                use_learned_positional_embeddings=False,
            )
        )

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value: bool = False) -> None:
        del module
        self.gradient_checkpointing = value

    def post_load_weights(self) -> None:
        return None

    def set_attention_backend(self, backend: str) -> None:
        set_cogvideox_attention_backend(self, backend)

    @property
    def attention_backend(self):
        return inspect_cogvideox_attention_backend(self)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        control_states: torch.Tensor,
        timestep: int | float | torch.LongTensor,
        conditioning_scale: float = 1.0,
        timestep_cond: torch.Tensor | None = None,
        ofs: int | float | torch.LongTensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> tuple[tuple[list[torch.Tensor], ...]] | Transformer2DModelOutput:
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

        batch_size, num_frames, _, _, _ = hidden_states.shape

        timesteps = timestep
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

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

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]
        hidden_states, image_rotary_emb, sequence_shard_state = shard_vividvr_video_tokens(
            hidden_states,
            image_rotary_emb,
        )

        controlnet_inter_states: tuple[torch.Tensor, ...] = ()
        for block in self.transformer_blocks:
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
            controlnet_inter_states = controlnet_inter_states + (hidden_states,)

        controlnet_hidden_states = build_vividvr_connector_control_states(
            controlnet_inter_states,
            sequence_shard_state,
            conditioning_scale=conditioning_scale,
        )

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (controlnet_hidden_states,)
        return Transformer2DModelOutput(sample=controlnet_hidden_states)

    @classmethod
    def from_transformer(
        cls,
        transformer: CogVideoXVividVRTransformer3DModel,
        num_layers: int | None = 6,
        load_weights_from_transformer: bool = True,
        load_transformer_weights_interval: bool = False,
    ) -> "CogVideoXVividVRControlNetModel":
        config = dict(transformer.config)
        config["num_layers"] = num_layers or config["num_layers"]
        config["use_learned_positional_embeddings"] = False
        config["use_rotary_positional_embeddings"] = True
        controlnet = cls(**config)

        if load_weights_from_transformer:
            controlnet.patch_embed.load_state_dict(
                transformer.patch_embed.state_dict()
            )
            controlnet.time_embedding.load_state_dict(
                transformer.time_embedding.state_dict()
            )
            if controlnet.ofs_embedding is not None and transformer.ofs_embedding is not None:
                controlnet.ofs_embedding.load_state_dict(
                    transformer.ofs_embedding.state_dict()
                )

            if load_transformer_weights_interval:
                control_interval = math.ceil(
                    transformer.config.num_layers / config["num_layers"]
                )
                logger.info(
                    "Loading VividVR controlnet blocks from transformer with interval %s",
                    control_interval,
                )
                for i in range(config["num_layers"]):
                    controlnet.transformer_blocks[i].load_state_dict(
                        transformer.transformer_blocks[control_interval * i].state_dict(),
                        strict=False,
                    )
            else:
                controlnet.transformer_blocks.load_state_dict(
                    transformer.transformer_blocks.state_dict(),
                    strict=False,
                )

        return controlnet


EntryClass = CogVideoXVividVRControlNetModel
