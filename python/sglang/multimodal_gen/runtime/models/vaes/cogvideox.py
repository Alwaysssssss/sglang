# SPDX-License-Identifier: Apache-2.0
from diffusers.models.autoencoders.autoencoder_kl_cogvideox import (
    AutoencoderKLCogVideoX as DiffusersAutoencoderKLCogVideoX,
)

from sglang.multimodal_gen.configs.models.vaes.cogvideox import CogVideoXVAEConfig


class AutoencoderKLCogVideoX(DiffusersAutoencoderKLCogVideoX):
    def __init__(self, config: CogVideoXVAEConfig) -> None:
        arch = config.arch_config
        super().__init__(
            in_channels=arch.in_channels,
            out_channels=arch.out_channels,
            down_block_types=tuple(arch.down_block_types),
            up_block_types=tuple(arch.up_block_types),
            block_out_channels=tuple(arch.block_out_channels),
            latent_channels=arch.latent_channels,
            layers_per_block=arch.layers_per_block,
            act_fn=arch.act_fn,
            norm_eps=arch.norm_eps,
            norm_num_groups=arch.norm_num_groups,
            temporal_compression_ratio=arch.temporal_compression_ratio,
            sample_height=arch.sample_height,
            sample_width=arch.sample_width,
            scaling_factor=arch.scaling_factor,
            shift_factor=arch.shift_factor,
            latents_mean=arch.latents_mean,
            latents_std=arch.latents_std,
            force_upcast=arch.force_upcast,
            use_quant_conv=arch.use_quant_conv,
            use_post_quant_conv=arch.use_post_quant_conv,
            invert_scale_latents=arch.invert_scale_latents,
        )
        self.sglang_config = config
        for name in (
            "tile_sample_min_height",
            "tile_sample_min_width",
            "tile_sample_min_num_frames",
            "tile_sample_stride_height",
            "tile_sample_stride_width",
            "tile_sample_stride_num_frames",
            "blend_num_frames",
            "use_tiling",
            "use_temporal_tiling",
            "use_parallel_tiling",
            "use_temporal_scaling_frames",
            "load_encoder",
            "load_decoder",
        ):
            setattr(self, name, getattr(config, name))


EntryClass = AutoencoderKLCogVideoX
