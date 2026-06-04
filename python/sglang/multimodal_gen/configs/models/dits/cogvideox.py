# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class CogVideoXArchConfig(DiTArchConfig):
    activation_fn: str = "gelu-approximate"
    attention_bias: bool = True
    attention_head_dim: int = 64
    dropout: float = 0.0
    flip_sin_to_cos: bool = True
    freq_shift: int = 0
    in_channels: int = 16
    max_text_seq_length: int = 226
    norm_elementwise_affine: bool = True
    norm_eps: float = 1e-5
    num_attention_heads: int = 48
    num_layers: int = 42
    ofs_embed_dim: int | None = None
    out_channels: int = 16
    patch_bias: bool = False
    patch_size: int = 2
    patch_size_t: int | None = 2
    sample_frames: int = 81
    sample_height: int = 96
    sample_width: int = 170
    spatial_interpolation_scale: float = 1.875
    temporal_compression_ratio: int = 4
    temporal_interpolation_scale: float = 1.0
    text_embed_dim: int = 4096
    time_embed_dim: int = 512
    timestep_activation_fn: str = "silu"
    use_learned_positional_embeddings: bool = False
    use_rotary_positional_embeddings: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels or self.in_channels


@dataclass
class CogVideoXConfig(DiTConfig):
    arch_config: CogVideoXArchConfig = field(default_factory=CogVideoXArchConfig)
    prefix: str = "CogVideoX"
