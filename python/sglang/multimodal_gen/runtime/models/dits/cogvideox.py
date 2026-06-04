# SPDX-License-Identifier: Apache-2.0
from typing import Any

from diffusers.models.transformers.cogvideox_transformer_3d import (
    CogVideoXTransformer3DModel as DiffusersCogVideoXTransformer3DModel,
)

from sglang.multimodal_gen.configs.models.dits.cogvideox import CogVideoXConfig


class CogVideoXTransformer3DModel(DiffusersCogVideoXTransformer3DModel):
    _fsdp_shard_conditions = CogVideoXConfig()._fsdp_shard_conditions
    _compile_conditions = CogVideoXConfig()._compile_conditions
    _supported_attention_backends = CogVideoXConfig()._supported_attention_backends
    param_names_mapping = CogVideoXConfig().param_names_mapping
    reverse_param_names_mapping = CogVideoXConfig().reverse_param_names_mapping
    lora_param_names_mapping = CogVideoXConfig().lora_param_names_mapping

    def __init__(
        self,
        config: CogVideoXConfig,
        hf_config: dict[str, Any],
        quant_config=None,
    ) -> None:
        arch = config.arch_config
        super().__init__(
            num_attention_heads=arch.num_attention_heads,
            attention_head_dim=arch.attention_head_dim,
            in_channels=arch.in_channels,
            out_channels=arch.out_channels,
            flip_sin_to_cos=arch.flip_sin_to_cos,
            freq_shift=arch.freq_shift,
            time_embed_dim=arch.time_embed_dim,
            ofs_embed_dim=arch.ofs_embed_dim,
            text_embed_dim=arch.text_embed_dim,
            num_layers=arch.num_layers,
            dropout=arch.dropout,
            attention_bias=arch.attention_bias,
            sample_width=arch.sample_width,
            sample_height=arch.sample_height,
            sample_frames=arch.sample_frames,
            patch_size=arch.patch_size,
            patch_size_t=arch.patch_size_t,
            temporal_compression_ratio=arch.temporal_compression_ratio,
            max_text_seq_length=arch.max_text_seq_length,
            activation_fn=arch.activation_fn,
            timestep_activation_fn=arch.timestep_activation_fn,
            norm_elementwise_affine=arch.norm_elementwise_affine,
            norm_eps=arch.norm_eps,
            spatial_interpolation_scale=arch.spatial_interpolation_scale,
            temporal_interpolation_scale=arch.temporal_interpolation_scale,
            use_rotary_positional_embeddings=arch.use_rotary_positional_embeddings,
            use_learned_positional_embeddings=arch.use_learned_positional_embeddings,
            patch_bias=arch.patch_bias,
        )
        self.sglang_config = config
        self.hf_config = hf_config
        self.quant_config = quant_config
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.num_channels_latents
        self.in_channels = arch.in_channels
        self.out_channels = arch.out_channels

    def post_load_weights(self) -> None:
        return None

    @property
    def supported_attention_backends(self):
        return self._supported_attention_backends

    @property
    def device(self):
        return next(self.parameters()).device


EntryClass = CogVideoXTransformer3DModel
