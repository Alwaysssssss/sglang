# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from contextlib import nullcontext
import math
import weakref
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.configs.sample.videoedit_wan import (
    WanVideoEditSamplingParams,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    V,
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.request_timeout import check_request_timeout
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.videoedit.preprocess import prepare_window_inputs
from sglang.multimodal_gen.runtime.videoedit.progress import (
    build_window_progress_payload,
    write_videoedit_progress,
)
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE


def _videoedit_params(batch: Req) -> WanVideoEditSamplingParams:
    params = batch.sampling_params
    if not isinstance(params, WanVideoEditSamplingParams):
        raise TypeError(
            "VideoEdit stages require WanVideoEditSamplingParams, "
            f"got {type(params).__name__}"
        )
    return params


def _module_dtype(module: torch.nn.Module, default: torch.dtype) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return default


def _retrieve_latents(encoder_output: Any) -> torch.Tensor:
    if hasattr(encoder_output, "latent_dist"):
        encoder_output = encoder_output.latent_dist
    if hasattr(encoder_output, "mode"):
        return encoder_output.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    if isinstance(encoder_output, torch.Tensor):
        return encoder_output
    raise AttributeError("Could not access latents of VAE encoder output")


def _vae_latent_mean_std(
    vae: torch.nn.Module, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    vae_config = getattr(vae, "config", None)
    z_dim = int(getattr(vae_config, "z_dim", None) or getattr(vae, "z_dim", 16))
    latents_mean = getattr(vae_config, "latents_mean", None)
    if latents_mean is None:
        latents_mean = getattr(vae, "latents_mean")
    latents_std = getattr(vae_config, "latents_std", None)
    if latents_std is None:
        latents_std = getattr(vae, "latents_std")
    mean = torch.tensor(
        latents_mean,
        device=device,
        dtype=dtype,
    ).view(1, z_dim, 1, 1, 1)
    std = torch.tensor(
        latents_std,
        device=device,
        dtype=dtype,
    ).view(1, z_dim, 1, 1, 1)
    return mean, std


def _normalize_vae_latents(latents: torch.Tensor, vae: torch.nn.Module) -> torch.Tensor:
    mean, std = _vae_latent_mean_std(vae, latents.device, latents.dtype)
    return (latents - mean) / std


def _denormalize_vae_latents(
    latents: torch.Tensor, vae: torch.nn.Module
) -> torch.Tensor:
    mean, std = _vae_latent_mean_std(vae, latents.device, latents.dtype)
    return latents * std + mean


def _ensure_tensor_decode_output(decode_output: Any) -> torch.Tensor:
    if isinstance(decode_output, tuple):
        return decode_output[0]
    if hasattr(decode_output, "sample"):
        return decode_output.sample
    return decode_output


def _move_vae_for_videoedit(
    vae: torch.nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    if getattr(vae, "_sglang_videoedit_native_vae", False):
        return vae.to(device=device)
    return vae.to(device=device, dtype=dtype)


def _decode_vae_for_videoedit(
    vae: torch.nn.Module,
    latents: torch.Tensor,
) -> torch.Tensor:
    try:
        decoded = vae.decode(latents, return_dict=False)
    except TypeError:
        decoded = vae.decode(latents)
    return _ensure_tensor_decode_output(decoded)


def _ensure_tensor_transformer_output(transformer_output: Any) -> torch.Tensor:
    if isinstance(transformer_output, torch.Tensor):
        return transformer_output
    if isinstance(transformer_output, tuple):
        return transformer_output[0]
    if hasattr(transformer_output, "sample"):
        return transformer_output.sample
    raise TypeError(
        "VideoEdit transformer output must be a tensor, tuple, or object with .sample; "
        f"got {type(transformer_output).__name__}"
    )


def _transformer_cache_context(transformer: torch.nn.Module, name: str):
    if hasattr(transformer, "cache_context"):
        return transformer.cache_context(name)
    return nullcontext()


def calc_current_cfg(
    max_cfg: float,
    current_step: int,
    max_step: int = 15,
    min_cfg: float = 1.0,
    dynamic_cfg: bool = True,
) -> tuple[float, bool]:
    if dynamic_cfg:
        if current_step < max_step:
            add_cfg = max(
                (max_cfg - min_cfg)
                / (math.pow((max_cfg - min_cfg), 1.0 / max_step) ** current_step),
                0,
            )
        else:
            add_cfg = 0.0
        current_cfg = min_cfg + add_cfg
        return current_cfg, current_cfg > 1.0
    return max_cfg, max_cfg != 1.0


class VideoEditWindowValidationStage(PipelineStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        params = _videoedit_params(batch)
        frames = params.runtime_window_frames
        masks = params.runtime_window_masks
        if frames is None or masks is None:
            raise ValueError("VideoEdit window frames and masks must be materialized")
        if len(frames) != params.infer_len:
            raise ValueError(
                f"VideoEdit window must contain {params.infer_len} frames, "
                f"got {len(frames)}"
            )
        if len(masks) != params.infer_len:
            raise ValueError(
                f"VideoEdit window must contain {params.infer_len} masks, got {len(masks)}"
            )
        if params.infer_len != 81 or (params.infer_len - 1) % 4 != 0:
            raise ValueError("VideoEdit stages require infer_len=81 and (infer_len-1)%4=0")

        width, height = frames[0].size
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"VideoEdit window size must be divisible by 16, got {width}x{height}"
            )

        params.runtime_height = height
        params.runtime_width = width
        params.runtime_num_frames = len(frames)
        params.runtime_window_validated = True
        batch.height = height
        batch.width = width
        batch.num_frames = len(frames)
        return batch


class VideoEditTextEncodingStage(PipelineStage):
    def __init__(self, text_encoder: torch.nn.Module, tokenizer: Any, transformer: torch.nn.Module):
        super().__init__()
        self.text_stage = TextEncodingStage([text_encoder], [tokenizer])
        self.transformer = transformer

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _videoedit_params(batch)
        target_dtype = _module_dtype(
            self.transformer, PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]
        )
        autocast_enabled = target_dtype != torch.float32 and not server_args.disable_autocast
        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=target_dtype,
            enabled=autocast_enabled,
        ):
            prompt_embeds, _, _ = self.text_stage.encode_text(
                params.prompt or " ",
                server_args,
                encoder_index=[0],
                return_attention_mask=True,
                dtype=target_dtype,
            )
        params.runtime_prompt_embeds = prompt_embeds[0]

        params.runtime_do_cfg = float(params.guidance_scale) > 1.0
        if params.runtime_do_cfg:
            with torch.autocast(
                device_type=current_platform.device_type,
                dtype=target_dtype,
                enabled=autocast_enabled,
            ):
                neg_embeds, _, _ = self.text_stage.encode_text(
                    params.negative_prompt or "",
                    server_args,
                    encoder_index=[0],
                    return_attention_mask=True,
                    dtype=target_dtype,
                )
            params.runtime_negative_prompt_embeds = neg_embeds[0]
        else:
            params.runtime_negative_prompt_embeds = None
        return batch


class VideoEditImageEncodingStage(PipelineStage):
    def __init__(
        self,
        image_encoder: torch.nn.Module | None,
        image_processor: Any | None,
        transformer: torch.nn.Module,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.image_processor = image_processor
        self.transformer = transformer

    def _prepare_diffsynth_pixel_values(
        self,
        image,
        device: torch.device,
    ) -> torch.Tensor:
        pixel_values = (
            torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        pixel_values = pixel_values.mul(2.0).sub(1.0).to(device=device, dtype=torch.float32)
        pixel_values = F.interpolate(
            pixel_values,
            size=(224, 224),
            mode="bicubic",
            align_corners=False,
        )
        pixel_values = pixel_values.mul(0.5).add(0.5)
        mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            device=device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            device=device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        return (pixel_values - mean) / std

    def _prepare_diffuser_image_inputs(self, image, device: torch.device):
        if self.image_processor is None:
            raise ValueError(
                "VideoEdit clip_preprocess='diffuser' requires an image_processor component. "
                "Provide --image-processor-path/--component_paths.image_processor "
                "or set clip_preprocess='diffsynth'."
            )
        return self.image_processor(images=image, return_tensors="pt").to(device)

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _videoedit_params(batch)
        params.runtime_image_embeds = None
        if not params.use_clip:
            return batch
        if self.image_encoder is None:
            raise ValueError(
                "VideoEdit use_clip=True requires an image_encoder component. "
                "Provide --image-encoder-path/--component_paths.image_encoder or set use_clip=False."
            )
        if params.runtime_window_frames is None or not params.runtime_window_frames:
            raise ValueError("VideoEdit window frames must be materialized before image encoding")
        if params.runtime_height is None or params.runtime_width is None:
            raise ValueError("VideoEdit window must be validated before image encoding")

        device = get_local_torch_device()
        if server_args.image_encoder_cpu_offload and hasattr(self.image_encoder, "to"):
            self.image_encoder = self.image_encoder.to(device)

        image = params.runtime_window_frames[0].convert("RGB")
        image = image.resize((params.runtime_width, params.runtime_height))

        image_dtype = _module_dtype(self.image_encoder, torch.float32)
        target_dtype = _module_dtype(
            self.transformer, PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]
        )
        autocast_enabled = target_dtype != torch.float32 and not server_args.disable_autocast
        if params.clip_preprocess == "diffuser":
            image_inputs = self._prepare_diffuser_image_inputs(image, device)
            pixel_values = None
        else:
            image_inputs = None
            pixel_values = self._prepare_diffsynth_pixel_values(image, device)
        try:
            with torch.autocast(
                device_type=current_platform.device_type,
                dtype=target_dtype,
                enabled=autocast_enabled,
            ):
                if image_inputs is not None:
                    outputs = self.image_encoder(
                        **image_inputs,
                        **server_args.pipeline_config.image_encoder_extra_args,
                    )
                else:
                    outputs = self.image_encoder(
                        pixel_values=pixel_values.to(dtype=image_dtype),
                        **server_args.pipeline_config.image_encoder_extra_args,
                    )
                image_embeds = server_args.pipeline_config.postprocess_image(outputs)
            params.runtime_image_embeds = image_embeds.to(device=device, dtype=target_dtype)
        finally:
            if server_args.image_encoder_cpu_offload and hasattr(self.image_encoder, "to"):
                self.image_encoder = self.image_encoder.to("cpu")
        return batch


class VideoEditConditionEncodingStage(PipelineStage):
    def __init__(self, vae: torch.nn.Module):
        super().__init__()
        self.vae = vae

    @torch.no_grad()
    def _encode_video_latents(
        self,
        video_tensor: torch.Tensor,
        *,
        vae_dtype: torch.dtype,
        output_dtype: torch.dtype,
        autocast_enabled: bool,
    ) -> torch.Tensor:
        device = get_local_torch_device()
        self.vae = _move_vae_for_videoedit(
            self.vae,
            device=device,
            dtype=vae_dtype,
        )
        video_tensor = video_tensor.to(device=device, dtype=vae_dtype)
        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=output_dtype,
            enabled=autocast_enabled,
        ):
            encoded = self.vae.encode(video_tensor)
            latents = _retrieve_latents(encoded)
            latents = _normalize_vae_latents(latents, self.vae)
        return latents.to(device=device, dtype=output_dtype)

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _videoedit_params(batch)
        if not params.runtime_window_validated:
            raise ValueError("VideoEdit window must be validated before condition encoding")

        device = get_local_torch_device()
        tensor_dtype = PRECISION_TO_TYPE[params.dtype]
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        prepared = prepare_window_inputs(
            params.runtime_window_frames,
            params.runtime_window_masks,
            device=device,
            dtype=tensor_dtype,
            mask_downsample_mode=params.mask_downsample_mode,
            preserve_first_frame=(params.runtime_window_index in (None, 0)),
        )
        params.runtime_masked_video_tensor = prepared["masked_video_tensor"]
        params.runtime_raw_video_tensor = prepared["video_tensor"]
        params.runtime_mask_video_tensor = prepared["mask_video_tensor"]
        params.runtime_cond_masks = prepared["cond_masks"]

        if server_args.pipeline_config.vae_tiling and hasattr(self.vae, "enable_tiling"):
            self.vae.enable_tiling()
        elif hasattr(self.vae, "disable_tiling"):
            self.vae.disable_tiling()
        autocast_enabled = tensor_dtype != torch.float32 and not server_args.disable_autocast

        masked_video = params.runtime_masked_video_tensor.permute(1, 0, 2, 3).unsqueeze(0)
        raw_video = params.runtime_raw_video_tensor.permute(1, 0, 2, 3).unsqueeze(0)

        params.runtime_cond_latents = self._encode_video_latents(
            masked_video,
            vae_dtype=vae_dtype,
            output_dtype=tensor_dtype,
            autocast_enabled=autocast_enabled,
        )
        params.runtime_video_latents = self._encode_video_latents(
            raw_video,
            vae_dtype=vae_dtype,
            output_dtype=tensor_dtype,
            autocast_enabled=autocast_enabled,
        )
        params.runtime_condition_latent = torch.cat(
            [params.runtime_cond_masks, params.runtime_cond_latents], dim=1
        )
        return batch


class VideoEditLatentPreparationStage(PipelineStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _videoedit_params(batch)
        shape_source = (
            params.runtime_video_latents
            if params.runtime_video_latents is not None
            else params.runtime_cond_latents
        )
        if shape_source is None:
            raise ValueError("VideoEdit latent shape source must be prepared before noise")
        device = shape_source.device
        seed = int(params.seed)
        if params.vary_seed_by_window and params.runtime_window_index is not None:
            seed += int(params.runtime_window_index)
        generator_device = (
            params.generator_device
            or getattr(server_args.pipeline_config, "generator_device", None)
            or device.type
        )
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        params.runtime_generator = generator
        if generator_device == "cpu":
            params.runtime_noise = torch.randn(
                shape_source.shape,
                generator=generator,
                device="cpu",
                dtype=torch.float32,
            ).to(device=device)
        else:
            params.runtime_noise = torch.randn(
                shape_source.shape,
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
        params.runtime_latents = params.runtime_noise
        batch.generator = generator
        batch.latents = params.runtime_latents
        batch.raw_latent_shape = tuple(params.runtime_latents.shape)
        return batch


class VideoEditTimestepPreparationStage(PipelineStage):
    def __init__(self, scheduler: Any):
        super().__init__()
        self.scheduler = scheduler

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _videoedit_params(batch)
        device = get_local_torch_device()
        self.scheduler.set_timesteps(
            params.num_inference_steps,
            shift=server_args.pipeline_config.flow_shift or 5.0,
            device=device,
        )
        if params.strength < 1.0:
            timesteps, effective_steps = self.scheduler.get_timesteps(
                params.num_inference_steps, self.scheduler.timesteps, params.strength
            )
        else:
            timesteps = self.scheduler.timesteps
            effective_steps = params.num_inference_steps

        params.runtime_timesteps = timesteps.to(device)
        params.runtime_effective_num_inference_steps = int(effective_steps)
        params.runtime_num_warmup_steps = len(timesteps) - int(effective_steps)
        batch.timesteps = params.runtime_timesteps
        return batch


class VideoEditLatentInitStage(PipelineStage):
    def __init__(self, scheduler: Any):
        super().__init__()
        self.scheduler = scheduler

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        params = _videoedit_params(batch)
        if params.runtime_timesteps is None:
            raise ValueError("VideoEdit timesteps must be prepared before latent init")
        params.runtime_initial_timestep = params.runtime_timesteps[:1]
        if params.init_latent_mode == "noise":
            params.runtime_latents = params.runtime_noise
            batch.latents = params.runtime_latents
            batch.raw_latent_shape = tuple(params.runtime_latents.shape)
            return batch
        if params.runtime_video_latents is None:
            raise ValueError("VideoEdit video latents are required for add_noise init")
        params.runtime_latents = self.scheduler.add_noise(
            params.runtime_video_latents.to(dtype=torch.float32),
            params.runtime_noise,
            params.runtime_initial_timestep,
        )
        batch.latents = params.runtime_latents
        batch.raw_latent_shape = tuple(params.runtime_latents.shape)
        return batch


class VideoEditDenoisingStage(DenoisingStage):
    def __init__(self, transformer: torch.nn.Module, scheduler: Any, pipeline=None):
        super().__init__(transformer=transformer, scheduler=scheduler, pipeline=pipeline)
        self.pipeline = weakref.ref(pipeline) if pipeline else None

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        del server_args
        params = _videoedit_params(batch)
        result = VerificationResult()
        result.add_check("runtime_timesteps", params.runtime_timesteps, V.is_tensor)
        result.add_check(
            "runtime_effective_num_inference_steps",
            params.runtime_effective_num_inference_steps,
            V.positive_int,
        )
        result.add_check(
            "runtime_latents", params.runtime_latents, [V.is_tensor, V.with_dims(5)]
        )
        result.add_check(
            "runtime_cond_masks",
            params.runtime_cond_masks,
            [V.is_tensor, V.with_dims(5)],
        )
        result.add_check(
            "runtime_cond_latents",
            params.runtime_cond_latents,
            [V.is_tensor, V.with_dims(5)],
        )
        result.add_check("runtime_prompt_embeds", params.runtime_prompt_embeds, V.is_tensor)
        result.add_check(
            "runtime_image_embeds",
            params.runtime_image_embeds,
            lambda value: (not params.use_clip) or V.is_tensor(value),
        )
        result.add_check("runtime_generator", params.runtime_generator, V.generator_or_list_generators)
        result.add_check("runtime_do_cfg", params.runtime_do_cfg, V.bool_value)
        result.add_check(
            "runtime_negative_prompt_embeds",
            params.runtime_negative_prompt_embeds,
            lambda value: not params.runtime_do_cfg or V.is_tensor(value),
        )
        return result

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if server_args.enable_cfg_parallel:
            raise NotImplementedError("VideoEdit MVP does not support CFG parallel yet")

        params = _videoedit_params(batch)
        if params.runtime_latents is None or params.runtime_timesteps is None:
            raise ValueError("VideoEdit latents and timesteps must be prepared")
        if params.runtime_prompt_embeds is None:
            raise ValueError("VideoEdit prompt embeds must be prepared")

        target_dtype = _module_dtype(
            self.transformer, PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]
        )
        autocast_enabled = target_dtype != torch.float32 and not server_args.disable_autocast
        timesteps = params.runtime_timesteps
        timesteps_cpu = timesteps.detach().cpu()
        latents = params.runtime_latents
        image_embeds = params.runtime_image_embeds

        self._manage_device_placement(self.transformer, None, server_args)
        batch.do_classifier_free_guidance = bool(params.runtime_do_cfg)
        batch.is_cfg_negative = False
        self._maybe_enable_cache_dit(params.runtime_effective_num_inference_steps, batch)

        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=target_dtype,
            enabled=autocast_enabled,
        ):
            with self.progress_bar(
                total=params.runtime_effective_num_inference_steps
            ) as progress_bar:
                for i, t_host in enumerate(timesteps_cpu):
                    check_request_timeout(batch)
                    t_device = timesteps[i]
                    current_cfg, do_cfg = calc_current_cfg(
                        max_cfg=float(params.guidance_scale),
                        current_step=i,
                        max_step=int(params.dynamic_cfg_max_step),
                        min_cfg=float(params.dynamic_cfg_min),
                        dynamic_cfg=bool(params.dynamic_cfg),
                    )
                    params.runtime_current_step = i
                    params.runtime_current_timestep = t_device
                    params.runtime_current_cfg = float(current_cfg)

                    latent_model_input = torch.cat(
                        [latents, params.runtime_cond_masks, params.runtime_cond_latents],
                        dim=1,
                    ).to(target_dtype)
                    timestep = t_device.to(dtype=target_dtype).expand(latents.shape[0])
                    attn_metadata = self._build_attn_metadata(
                        i,
                        batch,
                        server_args,
                        timestep_value=int(t_host.item()),
                        timesteps=timesteps_cpu,
                    )
                    batch.is_cfg_negative = False
                    with set_forward_context(
                        current_timestep=i,
                        attn_metadata=attn_metadata,
                        forward_batch=batch,
                    ):
                        with _transformer_cache_context(self.transformer, "cond"):
                            noise_pred = _ensure_tensor_transformer_output(
                                self.transformer(
                                    hidden_states=latent_model_input,
                                    timestep=timestep,
                                    encoder_hidden_states=params.runtime_prompt_embeds,
                                    encoder_hidden_states_image=image_embeds,
                                    attention_kwargs=None,
                                    return_dict=False,
                                )
                            )
                    check_request_timeout(batch)

                    if do_cfg:
                        if params.runtime_negative_prompt_embeds is None:
                            raise ValueError("Negative prompt embeds are required for CFG")
                        batch.is_cfg_negative = True
                        with set_forward_context(
                            current_timestep=i,
                            attn_metadata=attn_metadata,
                            forward_batch=batch,
                        ):
                            with _transformer_cache_context(self.transformer, "uncond"):
                                noise_uncond = _ensure_tensor_transformer_output(
                                    self.transformer(
                                        hidden_states=latent_model_input,
                                        timestep=timestep,
                                        encoder_hidden_states=params.runtime_negative_prompt_embeds,
                                        encoder_hidden_states_image=image_embeds,
                                        attention_kwargs=None,
                                        return_dict=False,
                                    )
                                )
                        check_request_timeout(batch)
                        noise_pred = noise_uncond + current_cfg * (
                            noise_pred - noise_uncond
                        )
                        batch.is_cfg_negative = False

                    latents = self.scheduler.step(noise_pred, t_device, latents)
                    params.runtime_latents = latents
                    batch.latents = latents
                    params.runtime_progress = float(i + 1) / float(len(timesteps))
                    write_videoedit_progress(
                        params.progress_path,
                        build_window_progress_payload(
                            stage="denoising",
                            total_frames=params.runtime_num_input_frames,
                            infer_len=params.infer_len,
                            overlap=params.overlap,
                            total_windows=len(params.runtime_window_specs or [None]),
                            current_window_index=params.runtime_window_index,
                            current_step_index=i,
                            steps_per_window=len(timesteps),
                        ),
                    )
                    if progress_bar is not None:
                        progress_bar.update()

        return batch


class VideoEditDecodingStage(PipelineStage):
    def __init__(self, vae: torch.nn.Module):
        super().__init__()
        self.vae = vae

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _videoedit_params(batch)
        if params.runtime_latents is None:
            raise ValueError("VideoEdit denoised latents must be prepared")
        device = get_local_torch_device()
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        self.vae = _move_vae_for_videoedit(
            self.vae,
            device=device,
            dtype=vae_dtype,
        )
        if server_args.pipeline_config.vae_tiling:
            self.vae.enable_tiling()
        latents = params.runtime_latents.to(device=device).to(dtype=vae_dtype)
        autocast_enabled = vae_dtype != torch.float32 and not server_args.disable_autocast
        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=vae_dtype,
            enabled=autocast_enabled,
        ):
            latents = _denormalize_vae_latents(latents, self.vae)
            decoded = _decode_vae_for_videoedit(self.vae, latents)
        decoded_frames = (decoded / 2 + 0.5).clamp(0, 1)[0].permute(1, 0, 2, 3)
        params.runtime_decoded_video_tensor = decoded_frames

        video = decoded_frames.detach().float().cpu()
        frames = []
        for frame in video.permute(0, 2, 3, 1):
            arr = (frame.clamp(0, 1).numpy() * 255.0).astype("uint8")
            from PIL import Image

            frames.append(Image.fromarray(arr))
        params.runtime_window_output_frames = frames
        return batch


class VideoEditWindowPostprocessStage(PipelineStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        params = _videoedit_params(batch)
        frames = params.runtime_window_output_frames
        if frames is None:
            raise ValueError("VideoEdit window output frames are missing")
        if len(frames) != params.infer_len:
            raise ValueError(
                f"VideoEdit decoded window must contain {params.infer_len} frames, "
                f"got {len(frames)}"
            )
        params.runtime_window_metadata = {
            "window_index": params.runtime_window_index,
            "num_frames": len(frames),
            "height": frames[0].height if frames else None,
            "width": frames[0].width if frames else None,
        }
        return batch
