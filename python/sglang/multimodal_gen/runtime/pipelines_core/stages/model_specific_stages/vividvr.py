# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import inspect
from typing import Any

import torch
from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import (
    retrieve_latents,
)
from diffusers.video_processor import VideoProcessor
from tqdm.auto import tqdm

from sglang.multimodal_gen.configs.sample.vividvr import VividVRSamplingParams
from sglang.multimodal_gen.runtime.distributed import (
    cfg_model_parallel_all_reduce,
    get_local_torch_device,
    get_sp_group,
    get_world_group,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import get_attn_backend
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common import (
    get_vividvr_connector_sp_context_mode,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.request_timeout import check_request_timeout
from sglang.multimodal_gen.runtime.server_args import ServerArgs, get_global_server_args
from sglang.multimodal_gen.runtime.utils.common import (
    randn_tensor_with_generator_device,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler
from sglang.multimodal_gen.runtime.vividvr import (
    apply_reference_color_fix,
    attach_generation_resolution,
    build_vividvr_caption_prompt_lists,
    build_vividvr_temporal_latent_merge_plan,
    build_vividvr_temporal_window_plan,
    build_vividvr_tiled_prompt_lists,
    decoded_video_to_frame_tensor,
    load_control_video,
    merge_vividvr_temporal_latent_states,
    prepare_rotary_positional_embeddings,
    prepare_tiling_infos_generator,
    prepare_vividvr_prompt_context,
    run_optional_postprocess_modules,
    stitch_vividvr_temporal_output_clips,
    trim_vividvr_temporal_output_clip,
)
from sglang.multimodal_gen.runtime.vividvr.progress_file import (
    write_vividvr_runtime_progress,
)
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


def _vividvr_params(batch: Req) -> VividVRSamplingParams:
    params = batch.sampling_params
    if not isinstance(params, VividVRSamplingParams):
        raise TypeError(
            "VividVR stages require VividVRSamplingParams, "
            f"got {type(params).__name__}"
        )
    return params


def _flowcut_progress_path(batch: Req) -> str | None:
    extra = getattr(batch, "extra", None)
    if isinstance(extra, dict):
        return extra.get("vividvr_flowcut_progress_path")
    return None


def _module_dtype(module: torch.nn.Module, default: torch.dtype) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return default


def _runtime_compute_device() -> torch.device:
    # CPU offload may temporarily park modules on CPU, but VividVR denoising
    # should still prepare compute tensors on the active local accelerator.
    return get_local_torch_device()


def _resolve_vividvr_parallel_mode(server_args: ServerArgs) -> str:
    requested = getattr(server_args, "vividvr_parallel_mode", "auto")
    enable_cfg_parallel = bool(getattr(server_args, "enable_cfg_parallel", False))
    sp_degree = int(getattr(server_args, "sp_degree", 1) or 1)

    if requested == "auto":
        if enable_cfg_parallel and sp_degree > 1:
            return "cfg_sp"
        if enable_cfg_parallel:
            return "cfg"
        if sp_degree > 1:
            return "sp"
        return "single"

    expected_by_mode = {
        "single": (False, False),
        "sp": (False, True),
        "cfg": (True, False),
        "cfg_sp": (True, True),
    }
    if requested not in expected_by_mode:
        raise ValueError(
            "Invalid vividvr_parallel_mode configuration: "
            f"vividvr_parallel_mode={requested!r}"
        )

    expected = expected_by_mode[requested]
    actual = (enable_cfg_parallel, sp_degree > 1)
    if actual != expected:
        raise ValueError(
            "Invalid vividvr_parallel_mode configuration: "
            f"vividvr_parallel_mode={requested!r}, "
            f"enable_cfg_parallel={enable_cfg_parallel}, sp_degree={sp_degree}"
        )
    return requested


def _resolve_attn_backend_cls(
    *,
    head_size: int,
    dtype: torch.dtype,
) -> Any:
    try:
        return get_attn_backend(head_size=head_size, dtype=dtype)
    except ValueError as exc:
        if str(exc) != "Global sgl_diffusion args is not set.":
            raise
        return current_platform.get_attn_backend(None, head_size, dtype)


def _ensure_tensor_decode_output(decode_output: Any) -> torch.Tensor:
    if isinstance(decode_output, tuple):
        return decode_output[0]
    if hasattr(decode_output, "sample"):
        return decode_output.sample
    return decode_output


def _aggregate_vae_spatial_decode_stats(
    clip_stats: list[dict[str, object]],
) -> dict[str, object]:
    if not clip_stats:
        return {}
    topologies = {
        (
            int(item["vae_sp_world_size"]),
            str(item["vae_sp_group_type"]),
            len(item["vae_local_tiles_per_rank"]),
        )
        for item in clip_stats
    }
    if len(topologies) != 1:
        raise RuntimeError(f"VAE SP clip topology mismatch: {sorted(topologies)}")
    reasons = {str(item["vae_sp_fallback_reason"]) for item in clip_stats}
    local_width = max(
        len(item["vae_local_tiles_per_rank"]) for item in clip_stats
    )
    local_totals = [0] * local_width
    for item in clip_stats:
        for rank, count in enumerate(item["vae_local_tiles_per_rank"]):
            local_totals[rank] += int(count)
    return {
        "vae_sp_requested": all(
            bool(item["vae_sp_requested"]) for item in clip_stats
        ),
        "vae_sp_effective": all(
            bool(item["vae_sp_effective"]) for item in clip_stats
        ),
        "vae_sp_fallback_reason": (
            next(iter(reasons)) if len(reasons) == 1 else "mixed"
        ),
        "vae_sp_world_size": int(clip_stats[0]["vae_sp_world_size"]),
        "vae_sp_group_type": str(clip_stats[0]["vae_sp_group_type"]),
        "vae_total_tiles": sum(
            int(item["vae_total_tiles"]) for item in clip_stats
        ),
        "vae_local_tiles_per_rank": local_totals,
        "vae_tile_decode_seconds": sum(
            float(item["vae_tile_decode_seconds"]) for item in clip_stats
        ),
        "vae_tile_gather_seconds": sum(
            float(item["vae_tile_gather_seconds"]) for item in clip_stats
        ),
        "vae_tile_merge_seconds": sum(
            float(item["vae_tile_merge_seconds"]) for item in clip_stats
        ),
        "vae_decode_seconds": sum(
            float(item["vae_decode_seconds"]) for item in clip_stats
        ),
        "vae_sp_clips": [dict(item) for item in clip_stats],
    }


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


def _clip_spec_record(clip_spec) -> dict[str, int]:
    return {
        "clip_index": int(clip_spec.clip_index),
        "start_frame": int(clip_spec.start_frame),
        "end_frame": int(clip_spec.end_frame),
        "original_num_frames": int(clip_spec.original_num_frames),
        "padded_num_frames": int(clip_spec.padded_num_frames),
        "num_padding_frames": int(clip_spec.num_padding_frames),
        "trim_front_frames": int(clip_spec.trim_front_frames),
        "trim_back_frames": int(clip_spec.trim_back_frames),
    }


def _require_vividvr_input_video_info(batch: Req) -> dict[str, Any]:
    input_video_info = batch.extra.get("vividvr_input_video_info")
    if input_video_info is None:
        raise ValueError("VividVR input video info must be prepared before long-video stages")
    return input_video_info


def _get_long_video_runtime(batch: Req) -> dict[str, Any]:
    return batch.extra.setdefault("vividvr_long_video_runtime", {})


def _build_temporal_clip_video_info(
    input_video_info: dict[str, Any],
    clip_spec,
) -> dict[str, Any]:
    reference_video = input_video_info["reference_video"]
    clip_reference_video = reference_video[clip_spec.start_frame : clip_spec.end_frame]
    clip_video = clip_reference_video
    if clip_spec.num_padding_frames > 0:
        padding = clip_reference_video[-1:].repeat(
            clip_spec.num_padding_frames,
            1,
            1,
            1,
        )
        clip_video = torch.cat([clip_reference_video, padding], dim=0)
    return {
        "video": clip_video,
        "reference_video": clip_reference_video,
        "fps": input_video_info["fps"],
        "original_height": input_video_info["original_height"],
        "original_width": input_video_info["original_width"],
        "gen_height": input_video_info["gen_height"],
        "gen_width": input_video_info["gen_width"],
        "original_num_frames": clip_spec.original_num_frames,
        "num_padding_frames": clip_spec.num_padding_frames,
    }


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

        latents = randn_tensor_with_generator_device(
            shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        latents = latents * scheduler.init_noise_sigma
        return latents, control_latents, num_latent_padding_frames


class VividVRInputValidationStage(PipelineStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _vividvr_params(batch)
        params.reset_runtime()
        params._validate_with_pipeline_config(server_args.pipeline_config)
        debug = batch.extra.setdefault("vividvr_debug", {})
        debug["vividvr_parallel_mode"] = _resolve_vividvr_parallel_mode(server_args)
        return batch


class VividVRPromptPreparationStage(PipelineStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _vividvr_params(batch)
        debug = batch.extra.setdefault("vividvr_debug", {})

        prompt_context = prepare_vividvr_prompt_context(
            params,
            server_args.pipeline_config,
            debug=debug,
        )

        params.runtime_prompt_file_path = prompt_context["prompt_file_path"]
        params.runtime_caption_file_path = prompt_context.get("caption_file_path")
        caption_texts = prompt_context.get("caption_texts")
        if caption_texts is not None:
            params.runtime_caption_texts = list(caption_texts)
        params.runtime_raw_prompt_text = prompt_context["prompt_text"]
        params.runtime_model_prompt_text = prompt_context["model_prompt_text"]
        params.runtime_negative_prompt_text = prompt_context["negative_prompt_text"]
        debug["caption_backend"] = prompt_context["caption_backend"]
        if params.runtime_caption_file_path is not None:
            debug["caption_file_path"] = params.runtime_caption_file_path
            debug["caption_entry_count"] = len(params.runtime_caption_texts or [])
        if "optional_module_warnings" in debug:
            params.runtime_optional_module_warnings = list(
                debug["optional_module_warnings"]
            )

        batch.prompt = params.runtime_model_prompt_text
        batch.negative_prompt = params.runtime_negative_prompt_text
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
    def encode_prompt_pair(
        self,
        *,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None,
        do_classifier_free_guidance: bool,
        server_args: ServerArgs,
    ) -> dict[str, torch.Tensor | None]:
        device = get_local_torch_device()
        target_dtype = _module_dtype(
            self.transformer,
            PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision],
        )

        self.text_encoder = self.text_encoder.to(device=device)

        prompt_embeds_list, _, _ = self.text_stage.encode_text(
            prompt,
            server_args,
            encoder_index=[0],
            return_attention_mask=True,
            device=device,
            dtype=target_dtype,
        )
        prompt_embeds = prompt_embeds_list[0]
        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            if isinstance(prompt, list):
                empty_negative_prompt = [""] * len(prompt)
            else:
                empty_negative_prompt = ""
            negative_prompt_embeds_list, _, _ = self.text_stage.encode_text(
                negative_prompt if negative_prompt is not None else empty_negative_prompt,
                server_args,
                encoder_index=[0],
                return_attention_mask=True,
                device=device,
                dtype=target_dtype,
            )
            negative_prompt_embeds = negative_prompt_embeds_list[0]
        if server_args.text_encoder_cpu_offload:
            self.text_encoder = self.text_encoder.to("cpu")
        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
        }

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _vividvr_params(batch)
        if params.runtime_model_prompt_text is None:
            raise ValueError("VividVR prompt text must be prepared before text encoding")

        params.runtime_do_cfg = float(params.guidance_scale) > 1.0
        encoded = self.encode_prompt_pair(
            prompt=params.runtime_model_prompt_text,
            negative_prompt=params.runtime_negative_prompt_text,
            do_classifier_free_guidance=bool(params.runtime_do_cfg),
            server_args=server_args,
        )
        params.runtime_prompt_embeds = encoded["prompt_embeds"]
        params.runtime_negative_prompt_embeds = encoded["negative_prompt_embeds"]

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

    def _resolve_control_video_info(
        self,
        batch: Req,
        params: VividVRSamplingParams,
        control_video_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved = control_video_info
        if resolved is None:
            preloaded = batch.extra.pop("vividvr_input_video_info", None)
            if preloaded is not None:
                resolved = preloaded
        if resolved is None:
            resolved = load_control_video(
                params.video_input_path,
                upscale=float(params.upscale),
            )
        return attach_generation_resolution(
            resolved,
            tile_size=int(params.tile_size),
            vae_scale_factor_spatial=int(self._vae_scale_factor_spatial),
        )

    def _sync_runtime_resolution(
        self,
        params: VividVRSamplingParams,
        control_video_info: dict[str, Any],
    ) -> None:
        params.height = int(control_video_info["gen_height"])
        params.width = int(control_video_info["gen_width"])

    def _resolve_generator(
        self,
        batch: Req,
        params: VividVRSamplingParams,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Generator:
        if generator is not None:
            batch.generator = generator
            return generator
        if isinstance(params.runtime_generator, torch.Generator):
            batch.generator = params.runtime_generator
            return params.runtime_generator
        if isinstance(batch.generator, torch.Generator):
            return batch.generator
        generator = torch.Generator(device=device.type).manual_seed(int(params.seed))
        batch.generator = generator
        return generator

    @torch.no_grad()
    def prepare_condition_inputs(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        control_video_info: dict[str, Any] | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
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

        generator = self._resolve_generator(batch, params, device, generator)
        control_video_info = self._resolve_control_video_info(
            batch,
            params,
            control_video_info,
        )
        self._sync_runtime_resolution(params, control_video_info)

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
        if server_args.vae_cpu_offload:
            self.vae = self.vae.to("cpu")

        return {
            "generator": generator,
            "control_video": control_video,
            "reference_video": control_video_info["reference_video"],
            "control_latents": control_latents,
            "original_height": int(control_video_info["original_height"]),
            "original_width": int(control_video_info["original_width"]),
            "gen_height": int(control_video_info["gen_height"]),
            "gen_width": int(control_video_info["gen_width"]),
            "original_num_frames": int(control_video_info["original_num_frames"]),
            "num_padding_frames": int(control_video_info["num_padding_frames"]),
            "padded_input_frames": int(control_video.shape[2]),
            "fps": max(1, int(round(float(control_video_info["fps"])))),
        }

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _vividvr_params(batch)
        prepared = self.prepare_condition_inputs(batch, server_args)

        params.runtime_generator = prepared["generator"]
        params.runtime_control_video = prepared["control_video"]
        params.runtime_reference_video = prepared["reference_video"]
        params.runtime_control_latents = prepared["control_latents"]
        params.runtime_original_height = prepared["original_height"]
        params.runtime_original_width = prepared["original_width"]
        params.runtime_original_num_frames = prepared["original_num_frames"]
        params.runtime_num_padding_frames = prepared["num_padding_frames"]
        params.runtime_padded_input_frames = prepared["padded_input_frames"]
        params.runtime_fps = prepared["fps"]

        debug = batch.extra.setdefault("vividvr_debug", {})
        debug["padded_input_frames"] = params.runtime_padded_input_frames
        debug["gen_height"] = int(prepared["gen_height"])
        debug["gen_width"] = int(prepared["gen_width"])

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
    def prepare_latents(
        self,
        *,
        control_video: torch.Tensor,
        control_latents: torch.Tensor,
        generator: torch.Generator,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        device = _runtime_compute_device()
        target_dtype = _module_dtype(
            self.transformer,
            control_video.dtype,
        )

        control_video = control_video.to(device=device, dtype=target_dtype)
        control_latents = control_latents.to(device=device, dtype=target_dtype)
        return self._prepare_latent_noise(
            control_video=control_video,
            control_latents=control_latents,
            batch_size=1,
            num_channels_latents=int(self.transformer.config.in_channels),
            height=height,
            width=width,
            dtype=target_dtype,
            device=device,
            generator=generator,
            scheduler=self.scheduler,
        )

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

        latents, control_latents, num_latent_padding_frames = self.prepare_latents(
            control_video=params.runtime_control_video,
            control_latents=params.runtime_control_latents,
            generator=params.runtime_generator,
            height=params.height,
            width=params.width,
        )

        params.runtime_control_video = params.runtime_control_video.to(
            device=latents.device,
            dtype=latents.dtype,
        )
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
    @staticmethod
    @torch.no_grad()
    def build_tiling_infos(
        *,
        latents: torch.Tensor,
        enable_spatial_tiling: bool,
        enable_temporal_tiling: bool,
        tile_size: int,
        tile_stride: int,
    ) -> list[Any]:
        return list(
            prepare_tiling_infos_generator(
                latents=latents,
                enable_spatial_tiling=enable_spatial_tiling,
                enable_temporal_tiling=enable_temporal_tiling,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        )

    @staticmethod
    def _align_prompt_embeds_to_tile_count(
        prompt_embeds: torch.Tensor | None,
        *,
        tile_count: int,
        tensor_name: str,
    ) -> torch.Tensor | None:
        if prompt_embeds is None:
            return None
        batch_size = int(prompt_embeds.shape[0])
        if batch_size == tile_count:
            return prompt_embeds
        if batch_size == 1:
            return prompt_embeds.repeat(tile_count, 1, 1)
        raise ValueError(
            f"{tensor_name} batch size must be 1 or tile_count={tile_count}, "
            f"got {batch_size}"
        )

    @classmethod
    @torch.no_grad()
    def prepare_tiling_state(
        cls,
        *,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        enable_spatial_tiling: bool,
        enable_temporal_tiling: bool,
        tile_size: int,
        tile_stride: int,
        tiling_infos: list[Any] | None = None,
    ) -> dict[str, Any]:
        resolved_tiling_infos = tiling_infos
        if resolved_tiling_infos is None:
            resolved_tiling_infos = cls.build_tiling_infos(
                latents=latents,
                enable_spatial_tiling=enable_spatial_tiling,
                enable_temporal_tiling=enable_temporal_tiling,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        tile_count = len(resolved_tiling_infos)
        tiled_prompt_embeds = cls._align_prompt_embeds_to_tile_count(
            prompt_embeds,
            tile_count=tile_count,
            tensor_name="prompt_embeds",
        )
        tiled_negative_prompt_embeds = cls._align_prompt_embeds_to_tile_count(
            negative_prompt_embeds,
            tile_count=tile_count,
            tensor_name="negative_prompt_embeds",
        )
        return {
            "tiling_infos": resolved_tiling_infos,
            "tile_count": tile_count,
            "tiled_prompt_embeds": tiled_prompt_embeds,
            "tiled_negative_prompt_embeds": tiled_negative_prompt_embeds,
        }

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        params = _vividvr_params(batch)
        if params.runtime_latents is None:
            raise ValueError("VividVR latents must be prepared before tiling")
        if params.runtime_prompt_embeds is None:
            raise ValueError("VividVR prompt embeds must be prepared before tiling")

        tiling_state = self.prepare_tiling_state(
            latents=params.runtime_latents,
            prompt_embeds=params.runtime_prompt_embeds,
            negative_prompt_embeds=params.runtime_negative_prompt_embeds,
            enable_spatial_tiling=params.enable_spatial_tiling,
            enable_temporal_tiling=params.enable_temporal_tiling,
            tile_size=params.tile_size,
            tile_stride=params.tile_stride,
        )

        params.runtime_tiling_infos = tiling_state["tiling_infos"]
        params.runtime_tile_count = tiling_state["tile_count"]
        params.runtime_tiled_prompt_embeds = tiling_state["tiled_prompt_embeds"]
        params.runtime_tiled_negative_prompt_embeds = tiling_state[
            "tiled_negative_prompt_embeds"
        ]

        debug = batch.extra.setdefault("vividvr_debug", {})
        debug["prompt_embed_shape"] = tuple(params.runtime_tiled_prompt_embeds.shape)
        debug["tile_count"] = params.runtime_tile_count
        return batch


class VividVRTimestepPreparationStage(PipelineStage):
    def __init__(self, scheduler: Any, transformer: torch.nn.Module):
        super().__init__()
        self.scheduler = scheduler
        self.transformer = transformer

    @torch.no_grad()
    def prepare_timesteps(self, num_inference_steps: int) -> torch.Tensor:
        device = _runtime_compute_device()
        timesteps, _ = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            None,
        )
        return timesteps

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        params = _vividvr_params(batch)
        timesteps = self.prepare_timesteps(params.num_inference_steps)
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
        attn_head_size = int(self.transformer.config.attention_head_dim)
        self.attn_backend = _resolve_attn_backend_cls(
            head_size=attn_head_size,
            dtype=torch.float16,
        )
        self.attn_metadata_builder_cls = None
        self.attn_metadata_builder = None
        self._cached_fa_attn_metadata = None

    @property
    def parallelism_type(self) -> StageParallelismType:
        if get_global_server_args().enable_cfg_parallel:
            return StageParallelismType.CFG_PARALLEL
        return StageParallelismType.REPLICATED

    @staticmethod
    def _select_cfg_model_input(
        tensor: torch.Tensor,
        *,
        do_classifier_free_guidance: bool,
        enable_cfg_parallel: bool,
    ) -> torch.Tensor:
        if do_classifier_free_guidance and not enable_cfg_parallel:
            return torch.cat([tensor] * 2)
        return tensor

    @staticmethod
    def _select_cfg_prompt_embeds(
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        prompt_slice: slice,
        *,
        do_classifier_free_guidance: bool,
        enable_cfg_parallel: bool,
        cfg_rank: int,
    ) -> tuple[torch.Tensor, str]:
        tile_prompt_embeds = prompt_embeds[prompt_slice]
        if not do_classifier_free_guidance:
            return tile_prompt_embeds, "none"
        if negative_prompt_embeds is None:
            raise ValueError("VividVR negative prompt embeds are required for CFG")
        if enable_cfg_parallel:
            if cfg_rank == 0:
                return tile_prompt_embeds, "cond"
            if cfg_rank == 1:
                return negative_prompt_embeds[prompt_slice], "uncond"
            raise ValueError(
                f"VividVR CFG parallel only supports cfg_rank 0/1, got {cfg_rank}"
            )
        return (
            torch.cat([negative_prompt_embeds[prompt_slice], tile_prompt_embeds], dim=0),
            "serial",
        )

    @staticmethod
    def _combine_cfg_noise_pred(
        noise_pred: torch.Tensor,
        *,
        guidance_scale: float,
        do_classifier_free_guidance: bool,
        enable_cfg_parallel: bool,
        cfg_rank: int,
        all_reduce_fn=cfg_model_parallel_all_reduce,
    ) -> torch.Tensor:
        if not do_classifier_free_guidance:
            return noise_pred
        if enable_cfg_parallel:
            if cfg_rank == 0:
                partial = float(guidance_scale) * noise_pred
            elif cfg_rank == 1:
                partial = (1.0 - float(guidance_scale)) * noise_pred
            else:
                raise ValueError(
                    f"VividVR CFG parallel only supports cfg_rank 0/1, got {cfg_rank}"
                )
            return all_reduce_fn(partial)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        return noise_pred_uncond + float(guidance_scale) * (
            noise_pred_text - noise_pred_uncond
        )

    def _prepare_runtime_module(
        self,
        module: torch.nn.Module,
        *,
        device: torch.device,
        target_dtype: torch.dtype,
        server_args: ServerArgs,
    ) -> torch.nn.Module:
        module = module.to(dtype=target_dtype)
        if getattr(server_args, "dit_cpu_offload", False):
            DenoisingStage._manage_device_placement(self, module, None, server_args)
        if next(module.parameters()).device != device:
            module = module.to(device=device)
        return module

    def progress_bar(self, total: int) -> tqdm:
        try:
            local_rank = get_world_group().local_rank
        except AssertionError:
            local_rank = 0
        return tqdm(total=total, disable=local_rank != 0, desc="VividVR denoising")

    def _prepare_image_rotary_emb(
        self,
        latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.transformer.config.use_rotary_positional_embeddings:
            return None
        return prepare_rotary_positional_embeddings(
            latent_height=latents.shape[-2],
            latent_width=latents.shape[-1],
            num_frames=latents.shape[1],
            patch_size=self.transformer.config.patch_size,
            patch_size_t=self.transformer.config.patch_size_t,
            attention_head_dim=self.transformer.config.attention_head_dim,
            device=latents.device,
            sample_height=self.transformer.config.sample_height,
            sample_width=self.transformer.config.sample_width,
        )

    def _build_runtime_attn_metadata(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        timestep_index: int,
    ) -> Any | None:
        if self.attn_backend.get_enum() == AttentionBackendEnum.FA:
            if self.attn_metadata_builder_cls is None:
                try:
                    self.attn_metadata_builder_cls = self.attn_backend.get_builder_cls()
                except NotImplementedError:
                    self.attn_metadata_builder_cls = None
            if (
                self.attn_metadata_builder_cls is not None
                and self.attn_metadata_builder is None
            ):
                self.attn_metadata_builder = self.attn_metadata_builder_cls()
            if (
                self.attn_metadata_builder is not None
                and self._cached_fa_attn_metadata is None
            ):
                self._cached_fa_attn_metadata = self.attn_metadata_builder.build(
                    raw_latent_shape=batch.raw_latent_shape
                )
            attn_metadata = self._cached_fa_attn_metadata
        else:
            attn_metadata = DenoisingStage._build_attn_metadata(
                self,
                timestep_index,
                batch,
                server_args,
            )
        debug = batch.extra.setdefault("vividvr_debug", {})
        debug["attn_metadata_enabled"] = attn_metadata is not None
        debug["attn_metadata_backend"] = str(self.attn_backend.get_enum())
        debug["attn_metadata_builder"] = (
            None
            if self.attn_metadata_builder is None
            else self.attn_metadata_builder.__class__.__name__
        )
        return attn_metadata

    @torch.no_grad()
    def prepare_denoising_state(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        latents: torch.Tensor,
        control_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        do_classifier_free_guidance: bool,
        timesteps: torch.Tensor,
        tiling_infos: list[Any],
    ) -> dict[str, Any]:
        params = _vividvr_params(batch)
        device = _runtime_compute_device()
        target_dtype = _module_dtype(
            self.transformer,
            PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision],
        )
        autocast_enabled = False

        self.transformer = self._prepare_runtime_module(
            self.transformer,
            device=device,
            target_dtype=target_dtype,
            server_args=server_args,
        )
        self.controlnet = self._prepare_runtime_module(
            self.controlnet,
            device=device,
            target_dtype=target_dtype,
            server_args=server_args,
        )

        latents = latents.to(device=device, dtype=target_dtype)
        control_latents = control_latents.to(device=device, dtype=target_dtype)
        prompt_embeds = prompt_embeds.to(device=device, dtype=target_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                device=device,
                dtype=target_dtype,
            )

        ofs_emb = (
            None
            if self.transformer.config.ofs_embed_dim is None
            else latents.new_full((1,), fill_value=2.0)
        )
        image_rotary_emb = self._prepare_image_rotary_emb(latents)

        debug = batch.extra.setdefault("vividvr_debug", {})
        debug.update(
            {
                "denoising_target_dtype": str(target_dtype).removeprefix("torch."),
                "denoising_device_type": device.type,
                "denoising_autocast_enabled": bool(autocast_enabled),
                "device_placement_helper": "DenoisingStage._manage_device_placement",
                "denoising_step_profile_helper": "DenoisingStage.step_profile",
                "attn_metadata_enabled": False,
                "attn_metadata_backend": None,
                "attn_metadata_builder": None,
                "vae_tiling_enabled": False,
            }
        )
        try:
            sp_group = get_sp_group()
        except AssertionError:
            sp_group = None
        try:
            world_group = get_world_group()
        except AssertionError:
            world_group = None
        total_video_tokens = (
            None if image_rotary_emb is None else int(image_rotary_emb[0].shape[0])
        )
        sequence_shard_enabled = bool(getattr(params, "enable_sequence_shard", False))
        sequence_shard_pad = 0
        sequence_shard_local_tokens = None
        if (
            sequence_shard_enabled
            and sp_group is not None
            and int(sp_group.world_size) > 1
            and total_video_tokens is not None
        ):
            sequence_shard_pad = (-total_video_tokens) % int(sp_group.world_size)
            sequence_shard_local_tokens = (
                total_video_tokens + sequence_shard_pad
            ) // int(sp_group.world_size)
        debug.update(
            {
                "distributed_world_size": (
                    None if world_group is None else int(world_group.world_size)
                ),
                "distributed_rank": (
                    None if world_group is None else int(world_group.rank)
                ),
                "distributed_local_rank": (
                    None if world_group is None else int(world_group.local_rank)
                ),
                "sp_world_size": None if sp_group is None else int(sp_group.world_size),
                "sp_rank": None if sp_group is None else int(sp_group.rank_in_group),
                "enable_sequence_shard": sequence_shard_enabled,
                "sp_sequence_shard_strategy": (
                    "model_native_video_token_shard"
                    if sequence_shard_enabled
                    else None
                ),
                "sp_sequence_tokens_global": total_video_tokens,
                "sp_sequence_tokens_local": sequence_shard_local_tokens,
                "sp_sequence_tokens_pad": sequence_shard_pad,
                "sp_video_token_layout": "contiguous_flat_video_token_sequence",
                "runtime_num_timesteps": len(timesteps),
                "connector_context_mode": (
                    (
                        "sp_exact_local_attention"
                        if get_vividvr_connector_sp_context_mode() == "deferred_global"
                        else (
                            "sp_exact_global_control_attention"
                            if get_vividvr_connector_sp_context_mode() == "eager_global"
                            else "sp_exact_distributed_control_attention"
                        )
                    )
                    if sequence_shard_enabled
                    else "single_rank_full_sequence"
                ),
                "control_context_shape_local": None,
                "control_context_shape_global": None,
            }
        )

        return {
            "latents": latents,
            "control_latents": control_latents,
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "do_classifier_free_guidance": do_classifier_free_guidance,
            "timesteps": timesteps,
            "tiling_infos": tiling_infos,
            "extra_step_kwargs": _prepare_extra_step_kwargs(
                self.scheduler,
                batch.generator,
                batch.eta,
            ),
            "ofs_emb": ofs_emb,
            "old_pred_original_sample": None,
            "target_dtype": target_dtype,
            "autocast_enabled": autocast_enabled,
            "raw_latent_shape": tuple(latents.shape),
        }

    @torch.no_grad()
    def run_denoising_step(
        self,
        batch: Req,
        server_args: ServerArgs,
        denoising_state: dict[str, Any],
        timestep_index: int,
        *,
        guidance_scale: float,
        restoration_guidance_scale: float,
    ) -> None:
        latents = denoising_state["latents"]
        control_latents = denoising_state["control_latents"]
        prompt_embeds = denoising_state["prompt_embeds"]
        negative_prompt_embeds = denoising_state["negative_prompt_embeds"]
        do_classifier_free_guidance = denoising_state["do_classifier_free_guidance"]
        timesteps = denoising_state["timesteps"]
        tiling_infos = denoising_state["tiling_infos"]
        extra_step_kwargs = denoising_state["extra_step_kwargs"]
        ofs_emb = denoising_state["ofs_emb"]
        old_pred_original_sample = denoising_state["old_pred_original_sample"]
        target_dtype = denoising_state["target_dtype"]
        debug = batch.extra.setdefault("vividvr_debug", {})
        sequence_shard_enabled = bool(debug.get("enable_sequence_shard", False))
        enable_cfg_parallel = bool(getattr(server_args, "enable_cfg_parallel", False))
        cfg_rank = 0
        cfg_world_size = 1
        if enable_cfg_parallel:
            cfg_rank = int(get_classifier_free_guidance_rank())
            cfg_world_size = int(get_classifier_free_guidance_world_size())
            if cfg_world_size != 2:
                raise ValueError(
                    "VividVR CFG parallel requires cfg world size 2, "
                    f"got {cfg_world_size}"
                )
        cfg_branch_for_debug = None
        if enable_cfg_parallel:
            cfg_branch_for_debug = "cond" if cfg_rank == 0 else "uncond"
        debug.update(
            {
                "vividvr_parallel_mode": _resolve_vividvr_parallel_mode(server_args),
                "cfg_parallel_enabled": enable_cfg_parallel,
                "cfg_rank": cfg_rank if enable_cfg_parallel else None,
                "cfg_world_size": cfg_world_size if enable_cfg_parallel else None,
                "cfg_branch": cfg_branch_for_debug,
                "cfg_combine_formula": (
                    "guidance_scale * cond + (1 - guidance_scale) * uncond"
                    if enable_cfg_parallel and do_classifier_free_guidance
                    else "serial_uncond_plus_guidance_delta"
                ),
            }
        )

        timestep = timesteps[timestep_index]
        attn_metadata = self._build_runtime_attn_metadata(
            batch,
            server_args,
            timestep_index=timestep_index,
        )
        del server_args
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
            image_rotary_emb = self._prepare_image_rotary_emb(tile_latents)
            timestep_expand = timestep.expand(tile_latents.shape[0])

            latent_model_input = self._select_cfg_model_input(
                tile_latents,
                do_classifier_free_guidance=do_classifier_free_guidance,
                enable_cfg_parallel=enable_cfg_parallel,
            )
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input,
                timestep,
            )

            control_model_input = self._select_cfg_model_input(
                tile_control_latents,
                do_classifier_free_guidance=do_classifier_free_guidance,
                enable_cfg_parallel=enable_cfg_parallel,
            )
            tile_prompt_embeds, cfg_branch = self._select_cfg_prompt_embeds(
                prompt_embeds,
                negative_prompt_embeds,
                prompt_slice,
                do_classifier_free_guidance=do_classifier_free_guidance,
                enable_cfg_parallel=enable_cfg_parallel,
                cfg_rank=cfg_rank,
            )
            debug["cfg_branch"] = cfg_branch

            concat_latent_model_input = torch.cat(
                [latent_model_input, control_model_input],
                dim=2,
            )
            with set_forward_context(
                current_timestep=timestep_index,
                attn_metadata=attn_metadata,
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
                    tuple(
                        state_tensor.to(tile_prompt_embeds.dtype)
                        for state_tensor in state
                    )
                    for state in control_hidden_states
                )
                if control_hidden_states:
                    first_control_state = control_hidden_states[0]
                    debug["connector_context_mode"] = (
                        "sp_exact_global_control_attention"
                        if len(first_control_state) >= 2
                        else (
                            (
                                "sp_exact_local_attention"
                                if get_vividvr_connector_sp_context_mode()
                                == "deferred_global"
                                else "sp_exact_distributed_control_attention"
                            )
                            if sequence_shard_enabled
                            else "single_rank_full_sequence"
                        )
                    )
                    debug["control_context_shape_local"] = tuple(
                        first_control_state[0].shape
                    )
                    debug["control_context_shape_global"] = (
                        tuple(first_control_state[-1].shape)
                        if len(first_control_state) >= 2
                        else None
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
            noise_pred = self._combine_cfg_noise_pred(
                noise_pred,
                guidance_scale=guidance_scale,
                do_classifier_free_guidance=do_classifier_free_guidance,
                enable_cfg_parallel=enable_cfg_parallel,
                cfg_rank=cfg_rank,
            )

            tile_latents, tile_old_pred_original_sample = self.scheduler.step(
                noise_pred,
                tile_old_pred_original_sample if timestep_index > 0 else None,
                timestep,
                timesteps[timestep_index - 1] if timestep_index > 0 else None,
                tile_latents,
                **extra_step_kwargs,
                return_dict=False,
                restoration_guidance_scale=restoration_guidance_scale,
                restoration_ori_latent=tile_control_latents,
            )
            tile_latents = tile_latents.to(target_dtype)
            tile_old_pred_original_sample = tile_old_pred_original_sample.to(target_dtype)

            latents_meshgrid[tile_slice_tuple] += tile_latents * tile_weights
            old_pred_original_sample_meshgrid[tile_slice_tuple] += (
                tile_old_pred_original_sample * tile_weights
            )
            weights_meshgrid[tile_slice_tuple] += tile_weights

        denoising_state["latents"] = latents_meshgrid / weights_meshgrid.clamp_min(1e-6)
        denoising_state["old_pred_original_sample"] = (
            old_pred_original_sample_meshgrid / weights_meshgrid.clamp_min(1e-6)
        )

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

        denoising_state = self.prepare_denoising_state(
            batch,
            server_args,
            latents=params.runtime_latents,
            control_latents=params.runtime_control_latents,
            prompt_embeds=params.runtime_tiled_prompt_embeds,
            negative_prompt_embeds=params.runtime_tiled_negative_prompt_embeds,
            do_classifier_free_guidance=bool(params.runtime_do_cfg),
            timesteps=params.runtime_timesteps,
            tiling_infos=params.runtime_tiling_infos or [],
        )
        with self.progress_bar(total=len(params.runtime_timesteps)) as progress_bar:
            for timestep_index, _ in enumerate(params.runtime_timesteps):
                check_request_timeout(batch)
                self.run_denoising_step(
                    batch,
                    server_args,
                    denoising_state,
                    timestep_index,
                    guidance_scale=float(params.guidance_scale),
                    restoration_guidance_scale=float(
                        params.restoration_guidance_scale
                    ),
                )
                DenoisingStage.step_profile(self)
                params.runtime_progress = float(timestep_index + 1) / float(
                    len(params.runtime_timesteps)
                )
                write_vividvr_runtime_progress(
                    _flowcut_progress_path(batch),
                    request_id=batch.request_id,
                    runtime_progress=params.runtime_progress,
                )
                if progress_bar is not None:
                    progress_bar.update()

        params.runtime_control_latents = denoising_state["control_latents"]
        params.runtime_latents = denoising_state["latents"]
        batch.latents = denoising_state["latents"]
        return batch


class VividVRDecodingStage(PipelineStage):
    def __init__(self, vae: torch.nn.Module):
        super().__init__()
        self.vae = vae
        self.last_vae_decode_stats: dict[str, object] = {}

    @torch.no_grad()
    def decode_latents(
        self,
        latents: torch.Tensor,
        num_latent_padding_frames: int,
        server_args: ServerArgs,
    ) -> torch.Tensor:
        device = latents.device
        vae_dtype = _module_dtype(
            self.vae,
            PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision],
        )
        self.vae = self.vae.to(device=device, dtype=vae_dtype)

        if num_latent_padding_frames > 0:
            latents = latents[:, num_latent_padding_frames:]

        decode_latents = latents.permute(0, 2, 1, 3, 4)
        decode_latents = decode_latents / self.vae.config.scaling_factor
        decoded = _ensure_tensor_decode_output(
            self.vae.decode(decode_latents.to(dtype=vae_dtype))
        )
        stats_getter = getattr(self.vae, "get_last_spatial_decode_stats", None)
        self.last_vae_decode_stats = (
            dict(stats_getter().to_debug_dict())
            if stats_getter is not None
            else {}
        )
        if server_args.vae_cpu_offload:
            self.vae = self.vae.to("cpu")
        return decoded

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _vividvr_params(batch)
        if params.runtime_latents is None:
            raise ValueError("VividVR denoised latents must be prepared before decoding")

        decoded = self.decode_latents(
            params.runtime_latents,
            int(params.runtime_num_latent_padding_frames or 0),
            server_args,
        )
        params.runtime_decoded_video = decoded
        debug = batch.extra.setdefault("vividvr_debug", {})
        debug["vae_tiling_enabled"] = bool(getattr(self.vae, "use_tiling", False))
        debug.update(self.last_vae_decode_stats)
        debug["vae_sp_clips"] = (
            [dict(self.last_vae_decode_stats)]
            if self.last_vae_decode_stats
            else []
        )
        return batch


class VividVROutputPostprocessStage(PipelineStage):
    def __init__(self, video_processor: VideoProcessor):
        super().__init__()
        self.video_processor = video_processor

    @torch.no_grad()
    def finalize_output_video(
        self,
        params: VividVRSamplingParams,
        debug: dict[str, Any],
    ) -> torch.Tensor:
        if params.runtime_decoded_video is None:
            raise ValueError("VividVR decoded video must be prepared before postprocess")
        if params.runtime_original_height is None or params.runtime_original_width is None:
            raise ValueError("VividVR original video size must be available for postprocess")

        output_video = decoded_video_to_frame_tensor(
            params.runtime_decoded_video,
            video_processor=self.video_processor,
            original_height=int(params.runtime_original_height),
            original_width=int(params.runtime_original_width),
        )
        if output_video.shape[0] % 4 == 0:
            output_video = output_video[3:]

        num_padding_frames = int(params.runtime_num_padding_frames or 0)
        if num_padding_frames > 0:
            output_video = output_video[:-num_padding_frames]

        output_video = apply_reference_color_fix(
            output_video,
            params.runtime_reference_video,
        )
        output_video = run_optional_postprocess_modules(
            output_video,
            reference_video=params.runtime_reference_video,
            enabled=bool(params.enable_optional_postprocess_module),
            allow_fallback=bool(params.allow_optional_module_fallback),
            debug=debug,
            processor=None,
        )
        if "optional_module_warnings" in debug:
            params.runtime_optional_module_warnings = list(
                debug["optional_module_warnings"]
            )
        return output_video

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        params = _vividvr_params(batch)
        debug = batch.extra.setdefault("vividvr_debug", {})

        output_video = self.finalize_output_video(params, debug)
        params.runtime_output_video = output_video
        batch.output = output_video.permute(1, 0, 2, 3).contiguous()
        batch.fps = int(params.runtime_fps or batch.fps)
        debug["output_shape"] = tuple(batch.output.shape)
        debug["output_num_frames"] = int(batch.output.shape[1])
        return batch


class VividVRTemporalWindowPlanningStage(PipelineStage):
    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        params = _vividvr_params(batch)
        input_video_info = _require_vividvr_input_video_info(batch)
        debug = batch.extra.setdefault("vividvr_debug", {})
        long_runtime = _get_long_video_runtime(batch)

        original_num_frames = int(input_video_info["original_num_frames"])
        window_plan = build_vividvr_temporal_window_plan(
            original_num_frames,
            params.num_temporal_process_frames,
        )
        long_runtime["window_plan"] = window_plan

        params.runtime_execution_mode = "temporal_windowed"
        params.runtime_clip_specs = list(window_plan.clip_specs)
        params.runtime_num_temporal_overlapped_frames = (
            window_plan.num_temporal_overlapped_frames
        )
        params.runtime_temporal_frame_stride = window_plan.temporal_frame_stride
        params.runtime_reference_video = input_video_info["reference_video"]
        params.runtime_original_height = int(input_video_info["original_height"])
        params.runtime_original_width = int(input_video_info["original_width"])
        params.runtime_original_num_frames = original_num_frames
        params.runtime_fps = max(1, int(round(float(input_video_info["fps"]))))
        params.runtime_do_cfg = float(params.guidance_scale) > 1.0
        params.height = int(input_video_info["gen_height"])
        params.width = int(input_video_info["gen_width"])

        batch.height = int(params.height)
        batch.width = int(params.width)
        batch.num_frames = original_num_frames
        batch.fps = params.runtime_fps
        batch.do_classifier_free_guidance = bool(params.runtime_do_cfg)

        debug["execution_mode"] = params.runtime_execution_mode
        debug["num_clips"] = int(window_plan.num_clips)
        debug["clip_specs"] = [_clip_spec_record(clip_spec) for clip_spec in window_plan.clip_specs]
        return batch


class VividVRLongClipPreparationStage(PipelineStage):
    def __init__(
        self,
        *,
        text_encoding_stage: VividVRTextEncodingStage,
        condition_encoding_stage: VividVRConditionEncodingStage,
        latent_preparation_stage: VividVRLatentPreparationStage,
        tiling_preparation_stage: VividVRTilingPreparationStage,
    ):
        super().__init__()
        self.text_encoding_stage = text_encoding_stage
        self.condition_encoding_stage = condition_encoding_stage
        self.latent_preparation_stage = latent_preparation_stage
        self.tiling_preparation_stage = tiling_preparation_stage

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _vividvr_params(batch)
        input_video_info = _require_vividvr_input_video_info(batch)
        debug = batch.extra.setdefault("vividvr_debug", {})
        long_runtime = _get_long_video_runtime(batch)
        window_plan = long_runtime.get("window_plan")
        if window_plan is None:
            raise ValueError("VividVR temporal window plan must be prepared before clip setup")

        generator = torch.Generator(device=get_local_torch_device().type).manual_seed(
            int(params.seed)
        )
        params.runtime_generator = generator
        batch.generator = generator

        clip_states: list[dict[str, Any]] = []
        clip_caption_records: list[dict[str, Any]] = []
        clip_latent_lengths: list[int] = []
        clip_tile_counts: list[int] = []
        caption_cursor = 0
        for clip_spec in window_plan.clip_specs:
            clip_video_info = _build_temporal_clip_video_info(input_video_info, clip_spec)
            prepared_condition = self.condition_encoding_stage.prepare_condition_inputs(
                batch,
                server_args,
                control_video_info=clip_video_info,
                generator=generator,
            )
            latents, control_latents, num_latent_padding_frames = (
                self.latent_preparation_stage.prepare_latents(
                    control_video=prepared_condition["control_video"],
                    control_latents=prepared_condition["control_latents"],
                    generator=prepared_condition["generator"],
                    height=params.height,
                    width=params.width,
                )
            )
            tiling_infos = self.tiling_preparation_stage.build_tiling_infos(
                latents=latents,
                enable_spatial_tiling=params.enable_spatial_tiling,
                enable_temporal_tiling=params.enable_temporal_tiling,
                tile_size=params.tile_size,
                tile_stride=params.tile_stride,
            )
            if params.runtime_caption_texts is not None:
                tiled_prompts = build_vividvr_caption_prompt_lists(
                    caption_texts=params.runtime_caption_texts,
                    start_index=caption_cursor,
                    tile_count=len(tiling_infos),
                    negative_prompt_text=params.runtime_negative_prompt_text,
                    pipeline_config=server_args.pipeline_config,
                )
                caption_cursor = int(tiled_prompts["next_index"])
                clip_caption_records.append(
                    {
                        "clip_index": int(clip_spec.clip_index),
                        "caption_text": str(tiled_prompts["clip_caption_text"]),
                        "tile_count": len(tiling_infos),
                    }
                )
            else:
                tiled_prompts = build_vividvr_tiled_prompt_lists(
                    model_prompt_text=params.runtime_model_prompt_text or "",
                    negative_prompt_text=params.runtime_negative_prompt_text,
                    tile_count=len(tiling_infos),
                )
            encoded_prompts = self.text_encoding_stage.encode_prompt_pair(
                prompt=tiled_prompts["prompt_list"],
                negative_prompt=tiled_prompts["negative_prompt_list"],
                do_classifier_free_guidance=bool(params.runtime_do_cfg),
                server_args=server_args,
            )
            tiling_state = self.tiling_preparation_stage.prepare_tiling_state(
                latents=latents,
                prompt_embeds=encoded_prompts["prompt_embeds"],
                negative_prompt_embeds=encoded_prompts["negative_prompt_embeds"],
                enable_spatial_tiling=params.enable_spatial_tiling,
                enable_temporal_tiling=params.enable_temporal_tiling,
                tile_size=params.tile_size,
                tile_stride=params.tile_stride,
                tiling_infos=tiling_infos,
            )

            clip_state = {
                "clip_spec": clip_spec,
                "control_latents": control_latents,
                "latents": latents,
                "num_latent_padding_frames": num_latent_padding_frames,
                "tiling_infos": tiling_state["tiling_infos"],
                "tiled_prompt_embeds": tiling_state["tiled_prompt_embeds"],
                "tiled_negative_prompt_embeds": tiling_state[
                    "tiled_negative_prompt_embeds"
                ],
                "prompt_embeds": encoded_prompts["prompt_embeds"],
                "negative_prompt_embeds": encoded_prompts["negative_prompt_embeds"],
                "do_classifier_free_guidance": bool(params.runtime_do_cfg),
            }
            clip_states.append(clip_state)
            clip_latent_lengths.append(int(latents.shape[1]))
            tile_count = tiling_state.get("tile_count")
            if tile_count is None:
                tile_count = len(tiling_state["tiling_infos"])
            clip_tile_counts.append(int(tile_count))

        if params.runtime_caption_texts is not None:
            if caption_cursor != len(params.runtime_caption_texts):
                raise ValueError(
                    "caption file entry count does not match temporal clip consumption: "
                    f"consumed {caption_cursor}, available {len(params.runtime_caption_texts)}"
                )
            debug["clip_caption_texts"] = clip_caption_records

        long_runtime["clip_states"] = clip_states
        long_runtime["clip_caption_records"] = clip_caption_records
        long_runtime["clip_latent_lengths"] = clip_latent_lengths
        long_runtime["clip_tile_counts"] = clip_tile_counts

        params.runtime_prompt_embeds = clip_states[0]["prompt_embeds"]
        params.runtime_negative_prompt_embeds = clip_states[0]["negative_prompt_embeds"]
        params.runtime_tiled_prompt_embeds = clip_states[0]["tiled_prompt_embeds"]
        params.runtime_tiled_negative_prompt_embeds = clip_states[0][
            "tiled_negative_prompt_embeds"
        ]
        params.runtime_tiling_infos = clip_states[0]["tiling_infos"]
        params.runtime_tile_count = max(clip_tile_counts)

        debug["padded_input_frames"] = max(
            int(clip_spec.padded_num_frames) for clip_spec in window_plan.clip_specs
        )
        debug["prompt_embed_shape"] = tuple(clip_states[0]["tiled_prompt_embeds"].shape)
        debug["control_latent_shape"] = tuple(clip_states[0]["control_latents"].shape)
        debug["tile_count"] = params.runtime_tile_count
        debug["clip_latent_lengths"] = clip_latent_lengths
        debug["clip_tile_counts"] = clip_tile_counts
        return batch


class VividVRMultiClipDenoisingStage(PipelineStage):
    def __init__(
        self,
        *,
        denoising_stage: VividVRDenoisingStage,
        vae_scale_factor_temporal: int,
    ):
        super().__init__()
        self.denoising_stage = denoising_stage
        self.vae_scale_factor_temporal = vae_scale_factor_temporal

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _vividvr_params(batch)
        debug = batch.extra.setdefault("vividvr_debug", {})
        long_runtime = _get_long_video_runtime(batch)
        clip_states = long_runtime.get("clip_states")
        if not clip_states:
            raise ValueError("VividVR clip states must be prepared before denoising")
        if params.runtime_timesteps is None:
            raise ValueError("VividVR timesteps must be prepared before denoising")

        clip_latent_lengths = long_runtime.get("clip_latent_lengths")
        if clip_latent_lengths is None:
            clip_latent_lengths = [int(clip_state["latents"].shape[1]) for clip_state in clip_states]
            long_runtime["clip_latent_lengths"] = clip_latent_lengths

        merge_plan = None
        if len(clip_states) > 1:
            merge_plan = build_vividvr_temporal_latent_merge_plan(
                clip_latent_lengths,
                num_temporal_process_frames=params.num_temporal_process_frames,
                vae_scale_factor_temporal=self.vae_scale_factor_temporal,
            )
        params.runtime_temporal_merge_plan = merge_plan

        denoising_states: list[dict[str, Any]] = []
        for clip_state in clip_states:
            denoising_state = self.denoising_stage.prepare_denoising_state(
                batch,
                server_args,
                latents=clip_state["latents"],
                control_latents=clip_state["control_latents"],
                prompt_embeds=clip_state["tiled_prompt_embeds"],
                negative_prompt_embeds=clip_state["tiled_negative_prompt_embeds"],
                do_classifier_free_guidance=bool(
                    clip_state["do_classifier_free_guidance"]
                ),
                timesteps=params.runtime_timesteps,
                tiling_infos=clip_state["tiling_infos"],
            )
            denoising_states.append(denoising_state)
        long_runtime["denoising_states"] = denoising_states

        with self.denoising_stage.progress_bar(total=len(params.runtime_timesteps)) as progress_bar:
            for timestep_index, _ in enumerate(params.runtime_timesteps):
                check_request_timeout(batch)
                with StageProfiler(
                    f"denoising_step_{timestep_index}",
                    logger=logger,
                    metrics=batch.metrics,
                    perf_dump_path_provided=batch.perf_dump_path is not None,
                    record_as_step=True,
                ):
                    for denoising_state in denoising_states:
                        batch.raw_latent_shape = tuple(denoising_state["latents"].shape)
                        self.denoising_stage.run_denoising_step(
                            batch,
                            server_args,
                            denoising_state,
                            timestep_index,
                            guidance_scale=float(params.guidance_scale),
                            restoration_guidance_scale=float(
                                params.restoration_guidance_scale
                            ),
                        )
                    if merge_plan is not None:
                        merge_vividvr_temporal_latent_states(
                            denoising_states,
                            merge_plan,
                        )
                DenoisingStage.step_profile(self.denoising_stage)
                params.runtime_progress = float(timestep_index + 1) / float(
                    len(params.runtime_timesteps)
                )
                write_vividvr_runtime_progress(
                    _flowcut_progress_path(batch),
                    request_id=batch.request_id,
                    runtime_progress=params.runtime_progress,
                )
                if progress_bar is not None:
                    progress_bar.update()

        debug["latents_shape"] = tuple(denoising_states[0]["latents"].shape)
        return batch


class VividVRMultiClipDecodeTrimStage(PipelineStage):
    def __init__(
        self,
        *,
        decoding_stage: VividVRDecodingStage,
        video_processor: VideoProcessor,
    ):
        super().__init__()
        self.decoding_stage = decoding_stage
        self.video_processor = video_processor

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _vividvr_params(batch)
        input_video_info = _require_vividvr_input_video_info(batch)
        debug = batch.extra.setdefault("vividvr_debug", {})
        long_runtime = _get_long_video_runtime(batch)
        clip_states = long_runtime.get("clip_states")
        denoising_states = long_runtime.get("denoising_states")
        if not clip_states or not denoising_states:
            raise ValueError("VividVR denoising states must be prepared before decode/trim")

        trimmed_clips: list[torch.Tensor] = []
        clip_vae_stats: list[dict[str, object]] = []
        for clip_state, denoising_state in zip(clip_states, denoising_states, strict=True):
            decoded_video = self.decoding_stage.decode_latents(
                denoising_state["latents"],
                int(clip_state["num_latent_padding_frames"]),
                server_args,
            )
            last_vae_decode_stats = getattr(
                self.decoding_stage, "last_vae_decode_stats", {}
            )
            if last_vae_decode_stats:
                clip_vae_stats.append(dict(last_vae_decode_stats))
            output_video = decoded_video_to_frame_tensor(
                decoded_video,
                video_processor=self.video_processor,
                original_height=int(input_video_info["original_height"]),
                original_width=int(input_video_info["original_width"]),
            )
            trimmed_clips.append(
                trim_vividvr_temporal_output_clip(
                    output_video,
                    clip_state["clip_spec"],
                )
            )

        long_runtime["trimmed_clips"] = trimmed_clips
        long_runtime["denoising_states"] = None
        debug["vae_tiling_enabled"] = bool(getattr(self.decoding_stage.vae, "use_tiling", False))
        debug.update(_aggregate_vae_spatial_decode_stats(clip_vae_stats))
        return batch


class VividVRTemporalStitchPostprocessStage(PipelineStage):
    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        params = _vividvr_params(batch)
        debug = batch.extra.setdefault("vividvr_debug", {})
        long_runtime = _get_long_video_runtime(batch)
        trimmed_clips = long_runtime.get("trimmed_clips")
        if trimmed_clips is None:
            raise ValueError("VividVR trimmed clips must be prepared before final stitch")

        final_output_video = stitch_vividvr_temporal_output_clips(trimmed_clips)
        final_output_video = apply_reference_color_fix(
            final_output_video,
            params.runtime_reference_video,
        )
        final_output_video = run_optional_postprocess_modules(
            final_output_video,
            reference_video=params.runtime_reference_video,
            enabled=bool(params.enable_optional_postprocess_module),
            allow_fallback=bool(params.allow_optional_module_fallback),
            debug=debug,
            processor=None,
        )
        if "optional_module_warnings" in debug:
            params.runtime_optional_module_warnings = list(
                debug["optional_module_warnings"]
            )

        params.runtime_num_padding_frames = 0
        params.runtime_output_video = final_output_video
        batch.output = final_output_video.permute(1, 0, 2, 3).contiguous()
        batch.fps = int(params.runtime_fps or batch.fps)

        debug["output_shape"] = tuple(batch.output.shape)
        debug["output_num_frames"] = int(batch.output.shape[1])
        batch.extra.pop("vividvr_long_video_runtime", None)
        return batch
