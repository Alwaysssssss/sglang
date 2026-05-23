# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
    LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class STARLatentPreparationStage(LatentPreparationStage):
    """Prepare STAR SR denoising latents on the latent timeline directly."""

    def adjust_video_length(self, batch: Req, server_args: ServerArgs) -> int:
        del server_args
        return int(batch.num_frames)

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        latent_num_frames = self.adjust_video_length(batch, server_args)
        batch_size = batch.batch_size
        dtype = self._get_latent_dtype(batch, server_args)
        device = get_local_torch_device()
        generator = batch.generator
        initial_noise_generator = batch.extra.get("star_initial_noise_generator")
        latents = batch.latents

        if latents is None:
            channels = (
                getattr(
                    server_args.pipeline_config.dit_config.arch_config,
                    "num_channels_latents",
                    0,
                )
                or server_args.pipeline_config.latent_channels
            )
            height = (
                batch.height
                // server_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio
            )
            width = (
                batch.width
                // server_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio
            )
            # STAR's reference sampler draws noise in [B, T, C, H, W] layout.
            # We then permute to the runtime's internal [B, C, T, H, W] layout.
            star_shape = (batch_size, latent_num_frames, channels, height, width)
            if (
                initial_noise_generator is not None
                and isinstance(initial_noise_generator, list)
                and len(initial_noise_generator) != batch_size
            ):
                raise ValueError(
                    "STAR initial-noise generator list length must match batch size."
                )
            # Mirror STAR's reference sampler exactly:
            # 1. sample the initial latent noise on CPU
            # 2. move it to the runtime device/dtype
            # 3. permute into the internal [B, C, T, H, W] layout
            latents_btchw = randn_tensor(
                star_shape,
                generator=initial_noise_generator or generator,
                device="cpu" if initial_noise_generator is not None else device,
                dtype=torch.float32 if initial_noise_generator is not None else dtype,
            )
            if initial_noise_generator is not None:
                latents_btchw = latents_btchw.to(device=device, dtype=dtype)
            latents = latents_btchw.permute(0, 2, 1, 3, 4).contiguous()
        else:
            latents = latents.to(device)

        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma

        batch.latents = latents
        batch.raw_latent_shape = latents.shape
        return batch
