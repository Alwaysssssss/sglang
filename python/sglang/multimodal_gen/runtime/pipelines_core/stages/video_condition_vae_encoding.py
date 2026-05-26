# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import gc

from torch import nn
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

    _AUTO_OFFLOAD_WORKSPACE_MULTIPLIER = 32.0

    @staticmethod
    def _tensor_summary(tensor: torch.Tensor) -> dict[str, object]:
        tensor_f = tensor.detach().cpu().float()
        return {
            "shape": list(tensor_f.shape),
            "mean": float(tensor_f.mean()),
            "std": float(tensor_f.std()),
            "min": float(tensor_f.min()),
            "max": float(tensor_f.max()),
        }

    @staticmethod
    def _resolve_sample_rng_mode(server_args: ServerArgs) -> str:
        pipeline_config = server_args.pipeline_config
        resolver = getattr(
            pipeline_config, "resolve_condition_video_vae_sample_rng_mode", None
        )
        if callable(resolver):
            return str(resolver())
        return str(
            getattr(
                pipeline_config,
                "condition_video_vae_sample_rng_mode",
                "generator",
            )
        )

    def __init__(
        self,
        vae,
        transformer: nn.Module | None = None,
        text_encoders: list[nn.Module] | None = None,
    ) -> None:
        super().__init__(vae=vae)
        self.transformer = transformer
        self.text_encoders = list(text_encoders or [])

    @staticmethod
    def _module_device(module: nn.Module | None) -> str | None:
        if module is None:
            return None
        try:
            return next(module.parameters()).device.type
        except StopIteration:
            return None

    @staticmethod
    def _move_module(module: nn.Module | None, device: str | torch.device) -> bool:
        if module is None or not hasattr(module, "to"):
            return False
        current_device = STARConditionVideoVAEEncodingStage._module_device(module)
        if current_device == torch.device(device).type:
            return False
        module.to(device)
        return True

    @staticmethod
    def _resolve_peak_memory_mode(server_args: ServerArgs) -> str:
        pipeline_config = server_args.pipeline_config
        resolver = getattr(
            pipeline_config, "resolve_condition_video_vae_peak_memory_mode", None
        )
        if callable(resolver):
            return str(resolver())
        if getattr(
            pipeline_config,
            "temporarily_offload_transformer_during_condition_vae_encode",
            False,
        ):
            return "text_encoder_and_transformer"
        if getattr(pipeline_config, "release_text_encoder_after_prompt_encode", False):
            return "text_encoder_only"
        return "off"

    @staticmethod
    def _get_free_gpu_memory_bytes() -> int | None:
        if not torch.cuda.is_available():
            return None
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
        except Exception:
            return None
        return int(free_bytes)

    @staticmethod
    def _estimate_condition_video_encode_bytes(
        batch: Req,
        server_args: ServerArgs,
    ) -> int:
        condition_video = batch.condition_video
        if isinstance(condition_video, torch.Tensor):
            numel = int(condition_video.numel())
        elif isinstance(condition_video, list) and condition_video:
            numel = sum(int(frame.numel()) for frame in condition_video)
        else:
            numel = int(
                batch.batch_size
                * (batch.condition_video_num_frames or batch.num_frames)
                * 3
                * batch.height
                * batch.width
            )

        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        element_size = torch.empty((), dtype=vae_dtype).element_size()
        return int(
            numel
            * element_size
            * STARConditionVideoVAEEncodingStage._AUTO_OFFLOAD_WORKSPACE_MULTIPLIER
        )

    def _should_auto_offload_transformer(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> bool:
        free_bytes = self._get_free_gpu_memory_bytes()
        if free_bytes is None:
            return False

        target_headroom_gb = float(
            getattr(
                server_args.pipeline_config,
                "condition_video_vae_target_headroom_gb",
                6.0,
            )
        )
        target_headroom_bytes = int(target_headroom_gb * (1024**3))
        estimated_encode_bytes = self._estimate_condition_video_encode_bytes(
            batch, server_args
        )
        decision = free_bytes < (target_headroom_bytes + estimated_encode_bytes)
        batch.extra["star_condition_vae_peak_memory"] = {
            "mode": "auto",
            "free_bytes": free_bytes,
            "estimated_encode_bytes": estimated_encode_bytes,
            "target_headroom_bytes": target_headroom_bytes,
            "offloaded_transformer": decision,
        }
        return decision

    def _prepare_peak_memory(self, batch: Req, server_args: ServerArgs) -> None:
        released_any = False

        mode = self._resolve_peak_memory_mode(server_args)

        if mode in (
            "text_encoder_only",
            "text_encoder_and_transformer",
            "auto",
        ):
            for text_encoder in self.text_encoders:
                released_any = self._move_module(text_encoder, "cpu") or released_any

        should_temporarily_offload_transformer = mode in (
            "transformer_only",
            "text_encoder_and_transformer",
        )
        if mode == "auto":
            should_temporarily_offload_transformer = (
                self._should_auto_offload_transformer(batch, server_args)
            )
        if should_temporarily_offload_transformer and self._move_module(
            self.transformer, "cpu"
        ):
            if not bool(getattr(server_args, "dit_cpu_offload", False)):
                batch.extra["star_reload_transformer_before_denoise"] = True
            released_any = True

        if released_any and torch.cuda.is_initialized():
            gc.collect()
            torch.cuda.empty_cache()

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

        self._prepare_peak_memory(batch, server_args)
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
        sample_rng_mode = self._resolve_sample_rng_mode(server_args)
        latent_generator = batch.generator
        if sample_mode == "sample" and sample_rng_mode == "global_seed":
            torch.manual_seed(int(batch.seed))
            latent_generator = None
        latent_condition = self.retrieve_latents(
            latent_dist,
            latent_generator,
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
        if batch.return_trajectory_latents and batch.metrics is not None:
            batch.metrics.record_annotation(
                "condition_video_vae_summary",
                {
                    "sample_mode": sample_mode,
                    "sample_rng_mode": sample_rng_mode,
                    "image_latent_summary": self._tensor_summary(batch.image_latent),
                },
            )
        batch.condition_video = None

        self.offload_model()
        if torch.cuda.is_initialized():
            gc.collect()
            torch.cuda.empty_cache()
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
