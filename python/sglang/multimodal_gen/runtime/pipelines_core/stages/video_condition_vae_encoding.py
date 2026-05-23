# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageVAEEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE


class STARConditionVideoVAEEncodingStage(ImageVAEEncodingStage):
    """Encode a full condition video into STAR-compatible conditioning latents."""

    @staticmethod
    def _expected_latent_num_frames(batch: Req, server_args: ServerArgs) -> int:
        source_num_frames = batch.condition_video_num_frames or (
            batch.condition_video.shape[1]
            if isinstance(batch.condition_video, torch.Tensor)
            else batch.num_frames
        )
        temporal_ratio = (
            server_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        )
        if server_args.pipeline_config.vae_config.use_temporal_scaling_frames:
            return int((source_num_frames - 1) // temporal_ratio + 1)
        return int(source_num_frames)

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        if batch.condition_video is None:
            return batch

        self.load_model()

        condition_video = batch.condition_video
        if isinstance(condition_video, list):
            condition_video = torch.stack(condition_video, dim=0)
        if condition_video.ndim != 5:
            raise ValueError(
                f"condition_video must have shape [B, T, C, H, W], got {tuple(condition_video.shape)}"
            )

        # [B, T, C, H, W] -> [B, C, T, H, W]
        condition_video = condition_video.permute(0, 2, 1, 3, 4).contiguous()
        condition_video = condition_video.to(
            device=get_local_torch_device(),
            dtype=torch.float32,
        )

        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not server_args.disable_autocast

        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=vae_dtype,
            enabled=vae_autocast_enabled,
        ):
            if server_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            if not vae_autocast_enabled:
                condition_video = condition_video.to(vae_dtype)
            latent_dist: DiagonalGaussianDistribution = self.vae.encode(
                condition_video
            )
            if isinstance(latent_dist, AutoencoderKLOutput):
                latent_dist = latent_dist.latent_dist

        sample_mode = server_args.pipeline_config.vae_config.encode_sample_mode()
        latent_condition = self.retrieve_latents(
            latent_dist,
            batch.generator,
            sample_mode=sample_mode,
        )
        latent_condition = server_args.pipeline_config.postprocess_vae_encode(
            latent_condition,
            self.vae,
        )
        normalized_latent_condition = server_args.pipeline_config.normalize_vae_encode(
            latent_condition,
            self.vae,
        )

        if normalized_latent_condition is None:
            scaling_factor, shift_factor = (
                server_args.pipeline_config.get_decode_scale_and_shift(
                    device=latent_condition.device,
                    dtype=latent_condition.dtype,
                    vae=self.vae,
                )
            )
            if shift_factor is not None:
                if isinstance(shift_factor, torch.Tensor):
                    shift_factor = shift_factor.to(
                        latent_condition.device, latent_condition.dtype
                    )
                latent_condition = latent_condition - shift_factor
            if scaling_factor is not None:
                if isinstance(scaling_factor, torch.Tensor):
                    scaling_factor = scaling_factor.to(
                        latent_condition.device, latent_condition.dtype
                    )
                latent_condition = latent_condition * scaling_factor
        else:
            latent_condition = normalized_latent_condition

        if latent_condition.ndim != 5:
            raise ValueError(
                "Encoded STAR condition video latent must have shape [B, C, T, H, W], "
                f"got {tuple(latent_condition.shape)}"
            )

        expected_channels = (
            getattr(server_args.pipeline_config.dit_config.arch_config, "num_channels_latents", 0)
            or getattr(server_args.pipeline_config, "latent_channels", 0)
        )
        if expected_channels and latent_condition.shape[1] != expected_channels:
            raise ValueError(
                "Encoded STAR condition video latent channel mismatch: "
                f"expected {expected_channels}, got {latent_condition.shape[1]}"
            )

        expected_num_frames = self._expected_latent_num_frames(batch, server_args)
        if latent_condition.shape[2] != expected_num_frames:
            raise ValueError(
                "Encoded STAR condition video latent time dimension mismatch: "
                f"expected {expected_num_frames}, got {latent_condition.shape[2]}"
            )

        spatial_ratio = (
            server_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio
        )
        expected_height = batch.height // spatial_ratio
        expected_width = batch.width // spatial_ratio
        if latent_condition.shape[3:] != (expected_height, expected_width):
            raise ValueError(
                "Encoded STAR condition video latent spatial mismatch: expected "
                f"({expected_height}, {expected_width}), got {tuple(latent_condition.shape[3:])}"
            )

        batch.image_latent = server_args.pipeline_config.postprocess_image_latent(
            latent_condition,
            batch,
        )

        self.offload_model()
        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("condition_video", batch.condition_video, V.not_none)
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("image_latent", batch.image_latent, V.none_or_tensor)
        return result
