# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class StarCogVideoXSRVAEArchConfig(VAEArchConfig):
    """Architecture config for the STAR CogVideoX-SR 3D VAE."""

    in_channels: int = 3
    out_channels: int = 3
    z_channels: int = 16
    latent_channels: int = 16

    ch: int = 128
    ch_mult: list[int] = field(default_factory=lambda: [1, 2, 2, 4])
    num_res_blocks: int = 3
    dropout: float = 0.0
    resolution: int = 256

    temporal_compression_ratio: int = 4
    spatial_compression_ratio: int = 8
    scaling_factor: float = 0.7


@dataclass
class StarCogVideoXSRVAEConfig(VAEConfig):
    arch_config: StarCogVideoXSRVAEArchConfig = field(
        default_factory=StarCogVideoXSRVAEArchConfig
    )

    def encode_sample_mode(self):
        # STAR's reference pipeline samples from the VAE posterior during
        # condition-video encoding via DiagonalGaussianRegularizer(sample=True).
        return "sample"
