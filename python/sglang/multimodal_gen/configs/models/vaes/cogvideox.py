# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class CogVideoXVAEArchConfig(VAEArchConfig):
    act_fn: str = "silu"
    block_out_channels: tuple[int, ...] = (128, 256, 256, 512)
    down_block_types: tuple[str, ...] = (
        "CogVideoXDownBlock3D",
        "CogVideoXDownBlock3D",
        "CogVideoXDownBlock3D",
        "CogVideoXDownBlock3D",
    )
    force_upcast: bool = True
    in_channels: int = 3
    invert_scale_latents: bool = True
    latent_channels: int = 16
    latents_mean: tuple[float, ...] | None = None
    latents_std: tuple[float, ...] | None = None
    layers_per_block: int = 3
    norm_eps: float = 1e-6
    norm_num_groups: int = 32
    out_channels: int = 3
    sample_height: int = 480
    sample_width: int = 720
    scaling_factor: float = 0.7
    shift_factor: float | None = None
    temporal_compression_ratio: int = 4
    up_block_types: tuple[str, ...] = (
        "CogVideoXUpBlock3D",
        "CogVideoXUpBlock3D",
        "CogVideoXUpBlock3D",
        "CogVideoXUpBlock3D",
    )
    use_post_quant_conv: bool = False
    use_quant_conv: bool = False

    def __post_init__(self) -> None:
        self.spatial_compression_ratio = 2 ** (len(self.block_out_channels) - 1)


@dataclass
class CogVideoXVAEConfig(VAEConfig):
    arch_config: CogVideoXVAEArchConfig = field(
        default_factory=CogVideoXVAEArchConfig
    )
