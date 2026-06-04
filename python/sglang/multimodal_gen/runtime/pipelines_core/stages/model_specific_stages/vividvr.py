# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import inspect
from typing import Any

import torch
import torch.nn.functional as F
from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import (
    retrieve_latents,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from tqdm.auto import tqdm

from sglang.multimodal_gen.configs.sample.vividvr import VividVRSamplingParams
from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_world_group,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.vividvr import (
    adaptive_instance_normalization,
    compose_positive_prompt,
    load_control_video,
    prepare_rotary_positional_embeddings,
    prepare_tiling_infos_generator,
    read_prompt_file,
    resolve_negative_prompt,
    resolve_prompt_file_path,
)
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE


def _vividvr_params(batch: Req) -> VividVRSamplingParams:
    params = batch.sampling_params
    if not isinstance(params, VividVRSamplingParams):
        raise TypeError(
            "VividVR stages require VividVRSamplingParams, "
            f"got {type(params).__name__}"
        )
    return params


def _module_dtype(module: torch.nn.Module, default: torch.dtype) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return default


def _module_device(
    module: torch.nn.Module,
    default: torch.device,
) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        try:
            return next(module.buffers()).device
        except StopIteration:
            return default


def _ensure_tensor_decode_output(decode_output: Any) -> torch.Tensor:
    if isinstance(decode_output, tuple):
        return decode_output[0]
    if hasattr(decode_output, "sample"):
        return decode_output.sample
    return decode_output


def _prepare_extra_step_kwargs(
    scheduler: Any,
    generator: torch.Generator | list[torch.Generator] | None,
    eta: float,
) -> dict[str, Any]:
    params = set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs: dict[str, Any] = {}
    if "eta" in params:
        extra_step_kwargs["eta"] = eta
    if "generator" in params:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs


class _VividVRLatentMixin:
    @property
    def _vae_scale_factor_spatial(self) -> int:
        return 2 ** (len(self.vae.config.block_out_channels) - 1)

    @property
    def _vae_scale_factor_temporal(self) -> int:
        return int(self.vae.config.temporal_compression_ratio)

    @torch.no_grad()
    def _encode_control_latents(
        self,
        *,
        control_video: torch.Tensor,
        dtype: torch.dtype,
        generator: torch.Generator,
    ) -> torch.Tensor:
        control_latents = [
            retrieve_latents(self.vae.encode(video.unsqueeze(0)), generator)
            for video in control_video
        ]
        control_latents = (
            torch.cat(control_latents, dim=0).to(dtype=dtype).permute(0, 2, 1, 3, 4)
        )
        return self.vae.config.scaling_factor * control_latents

    @torch.no_grad()
    def _prepare_latent_noise(
        self,
        *,
        control_video: torch.Tensor,
        control_latents: torch.Tensor,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator,
        scheduler: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        num_frames = (control_video.size(2) - 1) // self._vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_frames,
            num_channels_latents,
            height // self._vae_scale_factor_spatial,
            width // self._vae_scale_factor_spatial,
        )

        patch_size_t = self.transformer.config.patch_size_t
        if patch_size_t is not None:
            shape = shape[:1] + (shape[1] + shape[1] % patch_size_t,) + shape[2:]

        num_latent_padding_frames = shape[1] - control_latents.size(1)
        if num_latent_padding_frames > 0:
            first_frame = control_latents[
                :,
                : control_latents.size(1) % patch_size_t,
                ...,
            ]
            control_latents = torch.cat([first_frame, control_latents], dim=1)

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = latents * scheduler.init_noise_sigma
        return latents, control_latents, num_latent_padding_frames


class VividVRInputValidationStage(PipelineStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _vividvr_params(batch)
        params.reset_runtime()
        params._validate_with_pipeline_config(server_args.pipeline_config)
        return batch


class VividVRPromptPreparationStage(PipelineStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _vividvr_params(batch)
        pipeline_config = server_args.pipeline_config

        prompt_file_path = resolve_prompt_file_path(params, pipeline_config)
        prompt_text = read_prompt_file(prompt_file_path)
        model_prompt_text = compose_positive_prompt(prompt_text, pipeline_config)
        negative_prompt_text = resolve_negative_prompt(params, pipeline_config)

        params.runtime_prompt_file_path = prompt_file_path
        params.runtime_raw_prompt_text = prompt_text
        params.runtime_model_prompt_text = model_prompt_text
        params.runtime_negative_prompt_text = negative_prompt_text

        batch.prompt = model_prompt_text
        batch.negative_prompt = negative_prompt_text
        return batch


class VividVRTextEncodingStage(PipelineStage):
    def __init__(
        self,
        text_encoder: torch.nn.Module,
        tokenizer: Any,
        transformer: torch.nn.Module,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.transformer = transformer
        self.text_stage = TextEncodingStage([text_encoder], [tokenizer])

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _vividvr_params(batch)
        if params.runtime_model_prompt_text is None:
            raise ValueError("VividVR prompt text must be prepared before text encoding")

        device = get_local_torch_device()
        target_dtype = _module_dtype(
            self.transformer,
            PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision],
        )

        self.text_encoder = self.text_encoder.to(device=device)

        prompt_embeds_list, _, _ = self.text_stage.encode_text(
            params.runtime_model_prompt_text,
            server_args,
            encoder_index=[0],
            return_attention_mask=True,
            device=device,
            dtype=target_dtype,
        )
        params.runtime_prompt_embeds = prompt_embeds_list[0]

        params.runtime_do_cfg = float(params.guidance_scale) > 1.0
        if params.runtime_do_cfg:
            negative_prompt_embeds_list, _, _ = self.text_stage.encode_text(
                params.runtime_negative_prompt_text or "",
                server_args,
                encoder_index=[0],
                return_attention_mask=True,
                device=device,
                dtype=target_dtype,
            )
            params.runtime_negative_prompt_embeds = negative_prompt_embeds_list[0]
        else:
            params.runtime_negative_prompt_embeds = None

        batch.do_classifier_free_guidance = bool(params.runtime_do_cfg)
        return batch


class VividVRConditionEncodingStage(_VividVRLatentMixin, PipelineStage):
    def __init__(
        self,
        vae: torch.nn.Module,
        transformer: torch.nn.Module,
        video_processor: VideoProcessor,
    ):
        super().__init__()
        self.vae = vae
        self.transformer = transformer
        self.video_processor = video_processor

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _vividvr_params(batch)
        device = get_local_torch_device()

        target_dtype = _module_dtype(
            self.transformer,
            PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision],
        )
        vae_dtype = _module_dtype(
            self.vae,
            PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision],
        )

        self.vae = self.vae.to(device=device, dtype=vae_dtype)

        # Keep generator creation before VAE latent retrieval so control latents and
        # noise latents consume RNG in the same order as the accepted Phase C path.
        generator = torch.Generator(device=device.type).manual_seed(int(params.seed))
        params.runtime_generator = generator
        batch.generator = generator

        control_video_info = load_control_video(params.video_input_path)
        control_video = self.video_processor.preprocess_video(
            control_video_info["video"],
            height=params.height,
            width=params.width,
        ).to(device=device, dtype=target_dtype)
        control_latents = self._encode_control_latents(
            control_video=control_video,
            dtype=target_dtype,
            generator=generator,
        )

        params.runtime_control_video = control_video
        params.runtime_reference_video = control_video_info["reference_video"]
        params.runtime_control_latents = control_latents
        params.runtime_original_height = int(control_video_info["original_height"])
        params.runtime_original_width = int(control_video_info["original_width"])
        params.runtime_original_num_frames = int(control_video_info["original_num_frames"])
        params.runtime_num_padding_frames = int(control_video_info["num_padding_frames"])
        params.runtime_padded_input_frames = int(control_video.shape[2])
        params.runtime_fps = max(1, int(round(float(control_video_info["fps"]))))

        debug = batch.extra.setdefault("vividvr_debug", {})
        debug["padded_input_frames"] = params.runtime_padded_input_frames

        batch.height = int(params.height)
        batch.width = int(params.width)
        batch.num_frames = params.runtime_original_num_frames
        batch.fps = params.runtime_fps
        return batch


class VividVRLatentPreparationStage(_VividVRLatentMixin, PipelineStage):
    def __init__(
        self,
        vae: torch.nn.Module,
        transformer: torch.nn.Module,
        scheduler: Any,
    ):
        super().__init__()
        self.vae = vae
        self.transformer = transformer
        self.scheduler = scheduler

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        params = _vividvr_params(batch)
        if params.runtime_control_video is None:
            raise ValueError("VividVR control video must be prepared before latent init")
        if params.runtime_control_latents is None:
            raise ValueError("VividVR control latents must be prepared before latent init")
        if params.runtime_generator is None:
            raise ValueError("VividVR generator must be prepared before latent init")

        default_device = get_local_torch_device()
        device = _module_device(self.transformer, default_device)
        target_dtype = _module_dtype(
            self.transformer,
            params.runtime_control_video.dtype,
        )

        control_video = params.runtime_control_video.to(device=device, dtype=target_dtype)
        control_latents = params.runtime_control_latents.to(device=device, dtype=target_dtype)
        latents, control_latents, num_latent_padding_frames = self._prepare_latent_noise(
            control_video=control_video,
            control_latents=control_latents,
            batch_size=1,
            num_channels_latents=int(self.transformer.config.in_channels),
            height=params.height,
            width=params.width,
            dtype=target_dtype,
            device=device,
            generator=params.runtime_generator,
            scheduler=self.scheduler,
        )

        params.runtime_control_video = control_video
        params.runtime_control_latents = control_latents
        params.runtime_latents = latents
        params.runtime_num_latent_padding_frames = num_latent_padding_frames

        debug = batch.extra.setdefault("vividvr_debug", {})
        debug["control_latent_shape"] = tuple(control_latents.shape)
        debug["latents_shape"] = tuple(latents.shape)

        batch.latents = latents
        batch.raw_latent_shape = tuple(latents.shape)
        return batch


class VividVRTilingPreparationStage(PipelineStage):
    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        params = _vividvr_params(batch)
        if params.runtime_latents is None:
            raise ValueError("VividVR latents must be prepared before tiling")
        if params.runtime_prompt_embeds is None:
            raise ValueError("VividVR prompt embeds must be prepared before tiling")

        tiling_infos = list(
            prepare_tiling_infos_generator(
                latents=params.runtime_latents,
                enable_spatial_tiling=params.enable_spatial_tiling,
                enable_temporal_tiling=params.enable_temporal_tiling,
                tile_size=params.tile_size,
                tile_stride=params.tile_stride,
            )
        )
        tile_count = len(tiling_infos)

        params.runtime_tiling_infos = tiling_infos
        params.runtime_tile_count = tile_count
        params.runtime_tiled_prompt_embeds = params.runtime_prompt_embeds.repeat(
            tile_count, 1, 1
        )
        if params.runtime_negative_prompt_embeds is not None:
            params.runtime_tiled_negative_prompt_embeds = (
                params.runtime_negative_prompt_embeds.repeat(tile_count, 1, 1)
            )
        else:
            params.runtime_tiled_negative_prompt_embeds = None

        debug = batch.extra.setdefault("vividvr_debug", {})
        debug["prompt_embed_shape"] = tuple(params.runtime_tiled_prompt_embeds.shape)
        debug["tile_count"] = tile_count
        return batch


class VividVRTimestepPreparationStage(PipelineStage):
    def __init__(self, scheduler: Any, transformer: torch.nn.Module):
        super().__init__()
        self.scheduler = scheduler
        self.transformer = transformer

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _vividvr_params(batch)
        default_device = get_local_torch_device()
        device = _module_device(self.transformer, default_device)
        timesteps, _ = retrieve_timesteps(
            self.scheduler,
            params.num_inference_steps,
            device,
            None,
        )
        params.runtime_timesteps = timesteps
        params.runtime_timestep_count = len(timesteps)

        debug = batch.extra.setdefault("vividvr_debug", {})
        debug["timestep_count"] = len(timesteps)

        batch.timesteps = timesteps
        return batch


class VividVRDenoisingStage(PipelineStage):
    def __init__(
        self,
        transformer: torch.nn.Module,
        controlnet: torch.nn.Module,
        scheduler: Any,
    ):
        super().__init__()
        self.transformer = transformer
        self.controlnet = controlnet
        self.scheduler = scheduler

    def progress_bar(self, total: int) -> tqdm:
        try:
            local_rank = get_world_group().local_rank
        except AssertionError:
            local_rank = 0
        return tqdm(total=total, disable=local_rank != 0, desc="VividVR denoising")

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _vividvr_params(batch)
        if params.runtime_latents is None:
            raise ValueError("VividVR latents must be prepared before denoising")
        if params.runtime_control_latents is None:
            raise ValueError("VividVR control latents must be prepared before denoising")
        if params.runtime_tiled_prompt_embeds is None:
            raise ValueError("VividVR tiled prompt embeds must be prepared before denoising")
        if params.runtime_timesteps is None:
            raise ValueError("VividVR timesteps must be prepared before denoising")

        latents = params.runtime_latents
        control_latents = params.runtime_control_latents
        prompt_embeds = params.runtime_tiled_prompt_embeds
        negative_prompt_embeds = params.runtime_tiled_negative_prompt_embeds
        do_classifier_free_guidance = bool(params.runtime_do_cfg)
        timesteps = params.runtime_timesteps

        default_device = get_local_torch_device()
        device = _module_device(self.transformer, default_device)
        target_dtype = _module_dtype(
            self.transformer,
            PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision],
        )

        self.transformer = self.transformer.to(device=device, dtype=target_dtype)
        self.controlnet = self.controlnet.to(device=device, dtype=target_dtype)
        latents = latents.to(device=device, dtype=target_dtype)
        control_latents = control_latents.to(device=device, dtype=target_dtype)
        prompt_embeds = prompt_embeds.to(device=device, dtype=target_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                device=device,
                dtype=target_dtype,
            )

        extra_step_kwargs = _prepare_extra_step_kwargs(
            self.scheduler,
            batch.generator,
            batch.eta,
        )
        ofs_emb = (
            None
            if self.transformer.config.ofs_embed_dim is None
            else latents.new_full((1,), fill_value=2.0)
        )
        image_rotary_emb = (
            prepare_rotary_positional_embeddings(
                latent_height=latents.shape[-2],
                latent_width=latents.shape[-1],
                num_frames=latents.shape[1],
                patch_size=self.transformer.config.patch_size,
                patch_size_t=self.transformer.config.patch_size_t,
                attention_head_dim=self.transformer.config.attention_head_dim,
                device=device,
                sample_height=self.transformer.config.sample_height,
                sample_width=self.transformer.config.sample_width,
            )
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        old_pred_original_sample = None
        tiling_infos = params.runtime_tiling_infos or []
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for timestep_index, timestep in enumerate(timesteps):
                latents_meshgrid = torch.zeros_like(latents)
                old_pred_original_sample_meshgrid = torch.zeros_like(latents)
                weights_meshgrid = torch.zeros_like(latents)

                for tile_index, (tile_slice, tile_weights) in enumerate(tiling_infos):
                    prompt_slice = slice(tile_index, tile_index + 1)
                    tile_slice_tuple = tuple(tile_slice)

                    tile_latents = latents[tile_slice_tuple]
                    tile_control_latents = control_latents[tile_slice_tuple]
                    tile_old_pred_original_sample = (
                        old_pred_original_sample[tile_slice_tuple]
                        if old_pred_original_sample is not None
                        else None
                    )
                    timestep_expand = timestep.expand(tile_latents.shape[0])

                    latent_model_input = (
                        torch.cat([tile_latents] * 2)
                        if do_classifier_free_guidance
                        else tile_latents
                    )
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input,
                        timestep,
                    )

                    control_model_input = (
                        torch.cat([tile_control_latents] * 2)
                        if do_classifier_free_guidance
                        else tile_control_latents
                    )
                    tile_prompt_embeds = prompt_embeds[prompt_slice]
                    if do_classifier_free_guidance:
                        if negative_prompt_embeds is None:
                            raise ValueError(
                                "VividVR negative prompt embeds are required for CFG"
                            )
                        tile_prompt_embeds = torch.cat(
                            [negative_prompt_embeds[prompt_slice], tile_prompt_embeds],
                            dim=0,
                        )

                    concat_latent_model_input = torch.cat(
                        [latent_model_input, control_model_input],
                        dim=2,
                    )

                    with set_forward_context(
                        current_timestep=timestep_index,
                        attn_metadata=None,
                        forward_batch=batch,
                    ):
                        control_hidden_states = self.controlnet(
                            hidden_states=latent_model_input,
                            encoder_hidden_states=tile_prompt_embeds,
                            control_states=control_model_input,
                            image_rotary_emb=image_rotary_emb,
                            timestep=timestep_expand,
                            ofs=ofs_emb,
                            return_dict=False,
                        )[0]
                        control_hidden_states = tuple(
                            [state_tensor.to(tile_prompt_embeds.dtype) for state_tensor in state]
                            for state in control_hidden_states
                        )
                        noise_pred = self.transformer(
                            hidden_states=concat_latent_model_input,
                            encoder_hidden_states=tile_prompt_embeds,
                            control_hidden_states=control_hidden_states,
                            image_rotary_emb=image_rotary_emb,
                            timestep=timestep_expand,
                            ofs=ofs_emb,
                            return_dict=False,
                        )[0]

                    noise_pred = noise_pred.float()
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + float(params.guidance_scale) * (
                            noise_pred_text - noise_pred_uncond
                        )

                    tile_latents, tile_old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        tile_old_pred_original_sample if timestep_index > 0 else None,
                        timestep,
                        timesteps[timestep_index - 1] if timestep_index > 0 else None,
                        tile_latents,
                        **extra_step_kwargs,
                        return_dict=False,
                        restoration_guidance_scale=params.restoration_guidance_scale,
                        restoration_ori_latent=tile_control_latents,
                    )
                    tile_latents = tile_latents.to(target_dtype)
                    tile_old_pred_original_sample = tile_old_pred_original_sample.to(
                        target_dtype
                    )

                    latents_meshgrid[tile_slice_tuple] += tile_latents * tile_weights
                    old_pred_original_sample_meshgrid[tile_slice_tuple] += (
                        tile_old_pred_original_sample * tile_weights
                    )
                    weights_meshgrid[tile_slice_tuple] += tile_weights

                latents = latents_meshgrid / weights_meshgrid.clamp_min(1e-6)
                old_pred_original_sample = (
                    old_pred_original_sample_meshgrid / weights_meshgrid.clamp_min(1e-6)
                )
                params.runtime_progress = float(timestep_index + 1) / float(len(timesteps))
                if progress_bar is not None:
                    progress_bar.update()

        params.runtime_control_latents = control_latents
        params.runtime_latents = latents
        batch.latents = latents
        return batch


class VividVRDecodingStage(PipelineStage):
    def __init__(self, vae: torch.nn.Module):
        super().__init__()
        self.vae = vae

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _vividvr_params(batch)
        if params.runtime_latents is None:
            raise ValueError("VividVR denoised latents must be prepared before decoding")

        latents = params.runtime_latents
        device = latents.device
        vae_dtype = _module_dtype(
            self.vae,
            PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision],
        )
        self.vae = self.vae.to(device=device, dtype=vae_dtype)

        num_latent_padding_frames = int(params.runtime_num_latent_padding_frames or 0)
        if num_latent_padding_frames > 0:
            latents = latents[:, num_latent_padding_frames:]

        decode_latents = latents.permute(0, 2, 1, 3, 4)
        decode_latents = decode_latents / self.vae.config.scaling_factor
        decoded = _ensure_tensor_decode_output(
            self.vae.decode(decode_latents.to(dtype=vae_dtype))
        )

        params.runtime_decoded_video = decoded
        return batch


class VividVROutputPostprocessStage(PipelineStage):
    def __init__(self, video_processor: VideoProcessor):
        super().__init__()
        self.video_processor = video_processor

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        params = _vividvr_params(batch)
        if params.runtime_decoded_video is None:
            raise ValueError("VividVR decoded video must be prepared before postprocess")
        if params.runtime_original_height is None or params.runtime_original_width is None:
            raise ValueError("VividVR original video size must be available for postprocess")

        debug = batch.extra.setdefault("vividvr_debug", {})
        decoded = params.runtime_decoded_video

        resized_video = [
            F.interpolate(
                sample.permute(1, 0, 2, 3),
                size=(params.runtime_original_height, params.runtime_original_width),
                mode="bilinear",
                align_corners=False,
            )
            for sample in decoded
        ]
        resized_video = torch.stack(resized_video, dim=0).permute(0, 2, 1, 3, 4)
        processed = self.video_processor.postprocess_video(
            video=resized_video.float(),
            output_type="pt",
        )
        if isinstance(processed, list):
            processed = torch.stack(processed, dim=0)

        output_video = processed[0]
        if output_video.shape[0] % 4 == 0:
            output_video = output_video[3:]

        num_padding_frames = int(params.runtime_num_padding_frames or 0)
        if num_padding_frames > 0:
            output_video = output_video[:-num_padding_frames]

        reference_video = params.runtime_reference_video
        if reference_video is not None:
            reference_video = reference_video.to(
                device=output_video.device,
                dtype=output_video.dtype,
            )
            if reference_video.shape[-2:] != output_video.shape[-2:]:
                reference_video = F.interpolate(
                    reference_video,
                    size=output_video.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            if reference_video.shape[0] != output_video.shape[0]:
                frame_count = min(reference_video.shape[0], output_video.shape[0])
                reference_video = reference_video[:frame_count]
                output_video = output_video[:frame_count]
            output_video = adaptive_instance_normalization(
                output_video,
                reference_video,
            ).clamp_(0.0, 1.0)

        params.runtime_output_video = output_video
        batch.output = output_video.permute(1, 0, 2, 3).contiguous()
        batch.fps = int(params.runtime_fps or batch.fps)
        debug["output_shape"] = tuple(batch.output.shape)
        debug["output_num_frames"] = int(batch.output.shape[1])
        return batch
