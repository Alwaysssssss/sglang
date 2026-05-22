# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


def _is_star_transformer_layer(name: str, _module) -> bool:
    return "transformer.layers." in name and name.split(".")[-1].isdigit()


@dataclass
class StarCogVideoXSRArchConfig(DiTArchConfig):
    """Architecture config for the STAR CogVideoX-SR transformer."""

    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [_is_star_transformer_layer]
    )

    time_embed_dim: int = 512
    hidden_size: int = 3072
    num_attention_heads: int = 48
    num_layers: int = 42

    in_channels: int = 16
    out_channels: int = 16
    num_channels_latents: int = 16

    patch_size: int = 2
    text_hidden_size: int = 4096
    text_length: int = 226

    latent_width: int = 90
    latent_height: int = 60
    num_frames: int = 49
    time_compressed_rate: int = 4

    elementwise_affine: bool = True
    qk_ln: bool = True
    mlp_ratio: float = 4.0
    local_spatial_kernel_size: int = 7

    def __post_init__(self) -> None:
        super().__post_init__()
        self.num_channels_latents = self.out_channels


@dataclass
class StarCogVideoXSRDiTConfig(DiTConfig):
    arch_config: StarCogVideoXSRArchConfig = field(
        default_factory=StarCogVideoXSRArchConfig
    )

    prefix: str = "star_cogvideox_sr"
