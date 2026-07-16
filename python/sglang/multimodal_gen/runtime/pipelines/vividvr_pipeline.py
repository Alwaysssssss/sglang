# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

import torch
from diffusers.video_processor import VideoProcessor
from safetensors.torch import load_file
from torch import nn
from transformers import T5EncoderModel

from sglang.multimodal_gen.configs.models.dits.cogvideox import CogVideoXConfig
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.pipeline_configs.vividvr import VividVRPipelineConfig
from sglang.multimodal_gen.configs.sample.vividvr import VividVRSamplingParams
from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
    resolve_transformer_quant_load_spec,
    resolve_transformer_safetensors_to_load,
)
from sglang.multimodal_gen.runtime.loader.utils import set_default_torch_dtype
from sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr import (
    CogVideoXVividVRTransformer3DModel,
)
from sglang.multimodal_gen.runtime.models.dits.cogvideox_attention_backend import (
    configure_cogvideox_usp_collectives,
    enable_cogvideox_qk_norm_fusion,
    enable_cogvideox_qk_norm_rope_fusion,
    enable_cogvideox_qkv_fusion,
    inspect_cogvideox_attention_backend,
    inspect_cogvideox_qk_norm_fusion,
    inspect_cogvideox_qk_norm_rope_fusion,
    inspect_cogvideox_qkv_fusion,
    inspect_cogvideox_usp_collectives,
    normalize_cogvideox_attention_backend,
    resolve_cogvideox_attention_runtime_choice,
)
from sglang.multimodal_gen.runtime.models.dits.cogvideox_operator_fusion import (
    enable_cogvideox_modulation_fusion,
    inspect_cogvideox_modulation_fusion,
)
from sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_controlnet import (
    CogVideoXVividVRControlNetModel,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.executors.sync_executor import (
    SyncExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.vividvr import (
    VividVRConditionEncodingStage,
    VividVRDecodingStage,
    VividVRDenoisingStage,
    VividVRInputValidationStage,
    VividVRLatentPreparationStage,
    VividVRLongClipPreparationStage,
    VividVRMultiClipDecodeTrimStage,
    VividVRMultiClipDenoisingStage,
    VividVROutputPostprocessStage,
    VividVRPromptPreparationStage,
    VividVRTemporalStitchPostprocessStage,
    VividVRTemporalWindowPlanningStage,
    VividVRTextEncodingStage,
    VividVRTilingPreparationStage,
    VividVRTimestepPreparationStage,
    _aggregate_vae_spatial_decode_stats,
)
from sglang.multimodal_gen.runtime.request_timeout import check_request_timeout
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler
from sglang.multimodal_gen.runtime.vividvr import (
    apply_reference_color_fix,
    attach_generation_resolution,
    build_vividvr_caption_prompt_lists,
    build_vividvr_tiled_prompt_lists,
    build_vividvr_temporal_latent_merge_plan,
    build_vividvr_temporal_window_plan,
    decoded_video_to_frame_tensor,
    load_control_video,
    merge_vividvr_temporal_latent_states,
    run_optional_postprocess_modules,
    stitch_vividvr_temporal_output_clips,
    trim_vividvr_temporal_output_clip,
)
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE
from sglang.srt.utils.common import get_compiler_backend

logger = init_logger(__name__)


def _as_vividvr_params(batch: Req) -> VividVRSamplingParams:
    params = batch.sampling_params
    if not isinstance(params, VividVRSamplingParams):
        raise TypeError(
            "VividVRPipeline requires VividVRSamplingParams, "
            f"got {type(params).__name__}"
        )
    return params


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


def _enum_value_or_none(value: object) -> str | None:
    if isinstance(value, Enum):
        return str(value.value)
    if value is None:
        return None
    return str(value)


def _inspect_module_attention_backend(module: nn.Module | None) -> str | None:
    if module is None:
        return None

    backend = inspect_cogvideox_attention_backend(module)
    if backend is not None:
        return backend

    for child in module.modules():
        processor = getattr(child, "processor", None)
        backend = getattr(processor, "_attention_backend", None)
        if backend is not None:
            return _enum_value_or_none(backend)
    return None


def _inspect_module_torch_compile(module: nn.Module | None) -> bool:
    return bool(
        module is not None and getattr(module, "_sglang_torch_compile_enabled", False)
    )


def _inspect_module_qkv_fusion(module: nn.Module | None) -> str | None:
    if module is None:
        return None
    return inspect_cogvideox_qkv_fusion(module)


def _inspect_module_qk_norm_rope_fusion(module: nn.Module | None) -> str | None:
    if module is None:
        return None
    return inspect_cogvideox_qk_norm_rope_fusion(module)


def _inspect_module_qk_norm_fusion(module: nn.Module | None) -> str | None:
    if module is None:
        return None
    return inspect_cogvideox_qk_norm_fusion(module)


def _inspect_module_modulation_fusion(module: nn.Module | None) -> str | None:
    if module is None:
        return None
    return inspect_cogvideox_modulation_fusion(module)


def _normalize_component_targets(value: object) -> tuple[str, ...]:
    if value is None:
        return ("transformer",)

    if isinstance(value, str):
        raw_targets = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_targets = value
    else:
        raw_targets = [value]

    normalized: list[str] = []
    for raw_target in raw_targets:
        target = str(raw_target).strip().lower()
        if not target or target not in {"transformer", "controlnet"}:
            continue
        if target not in normalized:
            normalized.append(target)
    return tuple(normalized or ("transformer",))


def _normalize_qkv_fusion_targets(value: object) -> tuple[str, ...]:
    return _normalize_component_targets(value)


def _normalize_qk_norm_rope_fusion_targets(value: object) -> tuple[str, ...]:
    return _normalize_component_targets(value)


def _normalize_qk_norm_fusion_targets(value: object) -> tuple[str, ...]:
    return _normalize_component_targets(value)


def _normalize_modulation_fusion_targets(value: object) -> tuple[str, ...]:
    return _normalize_component_targets(value)


def _ensure_single_process_model_parallel_env(server_args: ServerArgs) -> None:
    if model_parallel_is_initialized():
        return

    master_port = int(getattr(server_args, "master_port", 30005))
    tp_size = int(getattr(server_args, "tp_size", 1) or 1)
    sp_degree = int(getattr(server_args, "sp_degree", 1) or 1)
    dp_size = int(getattr(server_args, "dp_size", 1) or 1)
    ulysses_degree = int(getattr(server_args, "ulysses_degree", 1) or 1)
    ring_degree = int(getattr(server_args, "ring_degree", 1) or 1)
    dist_timeout = getattr(server_args, "dist_timeout", 3600)
    enable_cfg_parallel = bool(getattr(server_args, "enable_cfg_parallel", False))

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(master_port))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    maybe_init_distributed_environment_and_model_parallel(
        tp_size=tp_size,
        sp_size=sp_degree,
        enable_cfg_parallel=enable_cfg_parallel,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        dp_size=dp_size,
        dist_timeout=dist_timeout,
    )


def _requires_model_parallel_runtime(server_args: ServerArgs) -> bool:
    return any(
        (
            int(getattr(server_args, "num_gpus", 1) or 1) > 1,
            int(getattr(server_args, "tp_size", 1) or 1) > 1,
            int(getattr(server_args, "sp_degree", 1) or 1) > 1,
            int(getattr(server_args, "dp_size", 1) or 1) > 1,
            bool(getattr(server_args, "enable_cfg_parallel", False)),
        )
    )


def _maybe_initialize_model_parallel_runtime(server_args: ServerArgs) -> None:
    if model_parallel_is_initialized():
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not (world_size > 1 or _requires_model_parallel_runtime(server_args)):
        return

    if world_size == 1:
        _ensure_single_process_model_parallel_env(server_args)
        return

    maybe_init_distributed_environment_and_model_parallel(
        tp_size=int(getattr(server_args, "tp_size", 1) or 1),
        sp_size=int(getattr(server_args, "sp_degree", 1) or 1),
        enable_cfg_parallel=bool(getattr(server_args, "enable_cfg_parallel", False)),
        ulysses_degree=int(getattr(server_args, "ulysses_degree", 1) or 1),
        ring_degree=int(getattr(server_args, "ring_degree", 1) or 1),
        dp_size=int(getattr(server_args, "dp_size", 1) or 1),
        dist_timeout=getattr(server_args, "dist_timeout", 3600),
    )


def _maybe_torch_compile_module(
    module: nn.Module,
    *,
    enabled: bool,
    module_name: str,
) -> nn.Module:
    if not enabled or not isinstance(module, nn.Module):
        return module
    if getattr(module, "_sglang_torch_compile_enabled", False):
        return module

    compile_kwargs: dict[str, object] = {"fullgraph": False, "dynamic": None}
    if current_platform.is_npu():
        backend = get_compiler_backend()
        compile_kwargs["backend"] = backend
        compile_kwargs["dynamic"] = False
        logger.info("Compiling VividVR %s with torchair backend on NPU.", module_name)
    else:
        try:
            import torch._inductor.config as _inductor_cfg

            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                _inductor_cfg.reorder_for_compute_comm_overlap = True
        except ImportError:
            pass
        mode = os.environ.get(
            "SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs"
        )
        # Keep VividVR on static-shape graphs per realized tile shape. This avoids
        # inductor autotune carrying symbolic tile extents into benchmark buffer
        # allocation on edge-tile inputs such as 90x128 latent views.
        compile_kwargs["dynamic"] = False
        compile_kwargs["mode"] = mode
        logger.info("Compiling VividVR %s with mode=%s.", module_name, mode)

    try:
        if hasattr(module, "compile"):
            module.compile(**compile_kwargs)
            compiled_module = module
        else:
            compiled_module = torch.compile(module, **compile_kwargs)
        setattr(compiled_module, "_sglang_torch_compile_enabled", True)
        setattr(compiled_module, "_sglang_torch_compile_kwargs", dict(compile_kwargs))
        logger.info("Applied torch.compile to VividVR %s.", module_name)
        return compiled_module
    except Exception as exc:
        logger.warning(
            "Failed to apply torch.compile to VividVR %s: %s",
            module_name,
            exc,
        )
        return module


def _configure_vividvr_vae_spatial_tile_parallel(
    vae: object, requested: bool
) -> None:
    configure_vae_sp = getattr(vae, "configure_spatial_tile_parallel", None)
    if configure_vae_sp is None and requested:
        raise TypeError("VividVR vae_sp requires the native CogVideoX VAE runtime")
    if configure_vae_sp is not None:
        configure_vae_sp(requested=bool(requested))


def _build_stage_without_global_server_args(stage_cls, *, server_args, **attrs):
    stage = object.__new__(stage_cls)
    stage.server_args = server_args
    for key, value in attrs.items():
        setattr(stage, key, value)
    return stage


class _VividVRT5EncoderWrapper(nn.Module):
    def __init__(self, encoder: T5EncoderModel):
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        # Original Vivid-VR calls T5 with input_ids only. Keep that behavior so
        # prompt embeddings stay numerically aligned with the reference pipeline.
        outputs = self.encoder(input_ids=input_ids, **kwargs)
        return BaseEncoderOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


class VividVRPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "CogVideoXVividVRControlNetPipeline"
    is_video_pipeline = True
    pipeline_config_cls = VividVRPipelineConfig
    sampling_params_cls = VividVRSamplingParams

    _required_config_modules = [
        "tokenizer",
        "vae",
        "scheduler",
    ]

    def build_executor(self, server_args: ServerArgs):
        return SyncExecutor(server_args=server_args)

    def _build_runtime_acceleration_debug(self, server_args: ServerArgs) -> dict[str, object]:
        transformer = self.get_module("transformer")
        controlnet = self.get_module("controlnet")
        requested_backend = server_args.attention_backend
        resolved_backend = normalize_cogvideox_attention_backend(requested_backend)
        sp_degree = max(
            getattr(server_args, "sp_degree", None) or 1,
            getattr(server_args, "ulysses_degree", None) or 1,
        )
        sp_enabled = sp_degree > 1
        runtime_choice = resolve_cogvideox_attention_runtime_choice(
            requested_backend,
            sp_enabled=sp_enabled,
        )
        return {
            "attention_backend_requested": requested_backend,
            "attention_backend_resolved": resolved_backend,
            "attention_backend_semantics": runtime_choice.semantics,
            "attention_backend_kernel": runtime_choice.kernel,
            "attention_backend_transformer": _inspect_module_attention_backend(
                transformer
            ),
            "attention_backend_controlnet": _inspect_module_attention_backend(
                controlnet
            ),
            "usp_packed_qkv_a2a_requested": bool(
                getattr(server_args, "enable_usp_packed_qkv_a2a", False)
            ),
            "usp_prefix_all_gather_into_tensor_requested": bool(
                getattr(
                    server_args,
                    "enable_usp_prefix_all_gather_into_tensor",
                    False,
                )
            ),
            "usp_transformer": inspect_cogvideox_usp_collectives(transformer),
            "usp_controlnet": inspect_cogvideox_usp_collectives(controlnet),
            "torch_compile_requested": bool(server_args.enable_torch_compile),
            "torch_compile_transformer": _inspect_module_torch_compile(transformer),
            "torch_compile_controlnet": _inspect_module_torch_compile(controlnet),
            "qkv_fusion_requested": bool(
                getattr(server_args, "enable_cogvideox_qkv_fusion", False)
            ),
            "qkv_fusion_targets": list(
                _normalize_qkv_fusion_targets(
                    getattr(
                        server_args,
                        "cogvideox_qkv_fusion_targets",
                        "transformer",
                    )
                )
            ),
            "qkv_fusion_transformer": _inspect_module_qkv_fusion(transformer),
            "qkv_fusion_controlnet": _inspect_module_qkv_fusion(controlnet),
            "qk_norm_fusion_requested": bool(
                getattr(server_args, "enable_cogvideox_qk_norm_fusion", False)
            ),
            "qk_norm_fusion_targets": list(
                _normalize_qk_norm_fusion_targets(
                    getattr(
                        server_args,
                        "cogvideox_qk_norm_fusion_targets",
                        "transformer",
                    )
                )
            ),
            "qk_norm_fusion_transformer": _inspect_module_qk_norm_fusion(
                transformer
            ),
            "qk_norm_fusion_controlnet": _inspect_module_qk_norm_fusion(controlnet),
            "qk_norm_rope_fusion_requested": bool(
                getattr(server_args, "enable_cogvideox_qk_norm_rope_fusion", False)
            ),
            "qk_norm_rope_fusion_targets": list(
                _normalize_qk_norm_rope_fusion_targets(
                    getattr(
                        server_args,
                        "cogvideox_qk_norm_rope_fusion_targets",
                        "transformer",
                    )
                )
            ),
            "qk_norm_rope_fusion_transformer": _inspect_module_qk_norm_rope_fusion(
                transformer
            ),
            "qk_norm_rope_fusion_controlnet": _inspect_module_qk_norm_rope_fusion(
                controlnet
            ),
            "modulation_fusion_requested": bool(
                getattr(server_args, "enable_cogvideox_modulation_fusion", False)
            ),
            "modulation_fusion_targets": list(
                _normalize_modulation_fusion_targets(
                    getattr(
                        server_args,
                        "cogvideox_modulation_fusion_targets",
                        "transformer",
                    )
                )
            ),
            "modulation_fusion_transformer": _inspect_module_modulation_fusion(
                transformer
            ),
            "modulation_fusion_controlnet": _inspect_module_modulation_fusion(
                controlnet
            ),
        }

    def _attach_runtime_acceleration_debug(
        self, batch: Req, server_args: ServerArgs
    ) -> None:
        debug = batch.extra.setdefault("vividvr_debug", {})
        debug.update(self._build_runtime_acceleration_debug(server_args))

    def _build_control_video_cache_key(
        self, video_input_path: str, upscale: float
    ) -> tuple[str, int, int, float]:
        resolved_path = os.path.abspath(os.fspath(video_input_path))
        stat = os.stat(resolved_path)
        return resolved_path, int(stat.st_mtime_ns), int(stat.st_size), float(upscale)

    def _resolve_input_video_info(
        self,
        video_input_path: str,
        *,
        upscale: float,
    ) -> dict[str, object]:
        cache_key = self._build_control_video_cache_key(video_input_path, upscale)
        cached_key = getattr(self, "_cached_control_video_cache_key", None)
        cached_info = getattr(self, "_cached_control_video_info", None)
        if cached_key == cache_key and cached_info is not None:
            return cached_info

        # Warmup requests are built via deepcopy(req), so keep the large decoded
        # control video cache on the pipeline instance instead of the request.
        input_video_info = load_control_video(video_input_path, upscale=upscale)
        self._cached_control_video_cache_key = cache_key
        self._cached_control_video_info = input_video_info
        return input_video_info

    def _enrich_input_video_info_with_generation_resolution(
        self,
        input_video_info: dict[str, object],
        *,
        tile_size: int,
    ) -> dict[str, object]:
        if "gen_height" in input_video_info and "gen_width" in input_video_info:
            return input_video_info

        vae = self.get_module("vae")
        vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
        return attach_generation_resolution(
            input_video_info,
            tile_size=int(tile_size),
            vae_scale_factor_spatial=int(vae_scale_factor_spatial),
        )

    def _apply_attention_backend(self, server_args: ServerArgs) -> None:
        requested_backend = server_args.attention_backend

        sp_degree = max(
            getattr(server_args, "sp_degree", None) or 1,
            getattr(server_args, "ulysses_degree", None) or 1,
        )
        sp_enabled = sp_degree > 1
        runtime_choice = resolve_cogvideox_attention_runtime_choice(
            requested_backend,
            sp_enabled=sp_enabled,
        )

        components = {
            "transformer": self.get_module("transformer"),
            "controlnet": self.get_module("controlnet"),
        }
        effective_candidates = [runtime_choice.effective_backend]
        if runtime_choice.kernel == "fa":
            fallback_backend = (
                "sdpa_sp" if runtime_choice.semantics == "ulysses_sp" else "sdpa"
            )
            effective_candidates.append(fallback_backend)

        last_error: Exception | None = None
        for candidate_backend in effective_candidates:
            try:
                for component_name, component in components.items():
                    if component is None:
                        logger.warning(
                            "Skipping attention backend '%s' for %s because the component is not loaded.",
                            candidate_backend,
                            component_name,
                        )
                        continue
                    if not hasattr(component, "set_attention_backend"):
                        logger.warning(
                            "Skipping attention backend '%s' for %s because the component does not expose set_attention_backend().",
                            candidate_backend,
                            component_name,
                        )
                        continue
                    component.set_attention_backend(candidate_backend)
                    applied_backend = _inspect_module_attention_backend(component)
                    logger.info(
                        "Applied VividVR attention backend candidate='%s' "
                        "(requested='%s', semantics='%s', kernel='%s') to %s; effective_backend=%s.",
                        candidate_backend,
                        requested_backend,
                        runtime_choice.semantics,
                        runtime_choice.kernel,
                        component_name,
                        applied_backend,
                    )
                return
            except Exception as exc:
                last_error = exc
                if candidate_backend == effective_candidates[-1]:
                    raise
                logger.warning(
                    "VividVR attention backend candidate '%s' failed for requested='%s' "
                    "(semantics='%s', kernel='%s'); falling back to '%s'. error=%s",
                    candidate_backend,
                    requested_backend,
                    runtime_choice.semantics,
                    runtime_choice.kernel,
                    effective_candidates[effective_candidates.index(candidate_backend) + 1],
                    exc,
                )

        if last_error is not None:
            raise last_error

    def _apply_usp_collective_optimizations(self, server_args: ServerArgs) -> None:
        use_packed_qkv_a2a = bool(
            getattr(server_args, "enable_usp_packed_qkv_a2a", False)
        )
        use_prefix_all_gather_into_tensor = bool(
            getattr(
                server_args,
                "enable_usp_prefix_all_gather_into_tensor",
                False,
            )
        )
        if not use_packed_qkv_a2a and not use_prefix_all_gather_into_tensor:
            return

        ulysses_degree = getattr(server_args, "ulysses_degree", None) or 1
        if ulysses_degree <= 1:
            raise ValueError(
                "USP collective optimizations require a Ulysses degree greater than 1."
            )

        for component_name in ("transformer", "controlnet"):
            component = self.get_module(component_name)
            if component is None:
                logger.warning(
                    "Skipping USP collective optimizations for %s because the component is not loaded.",
                    component_name,
                )
                continue
            applied = configure_cogvideox_usp_collectives(
                component,
                use_packed_qkv_a2a=use_packed_qkv_a2a,
                use_prefix_all_gather_into_tensor=(
                    use_prefix_all_gather_into_tensor
                ),
            )
            logger.info(
                "Applied USP collective optimizations to %s processors=%d "
                "packed_qkv_a2a=%s prefix_all_gather_into_tensor=%s.",
                component_name,
                applied,
                use_packed_qkv_a2a,
                use_prefix_all_gather_into_tensor,
            )

    def _apply_qkv_fusion(self, server_args: ServerArgs) -> None:
        if not getattr(server_args, "enable_cogvideox_qkv_fusion", False):
            return

        _ensure_single_process_model_parallel_env(server_args)
        resolved_backend = normalize_cogvideox_attention_backend(
            server_args.attention_backend
        )
        if resolved_backend != "fa":
            logger.warning(
                "CogVideoX QKV fusion is enabled, but the requested backend resolves to %s. "
                "Phase E3 acceleration is currently consumed by the custom flash-attention path.",
                resolved_backend,
            )
        target_components = set(
            _normalize_qkv_fusion_targets(
                getattr(
                    server_args,
                    "cogvideox_qkv_fusion_targets",
                    "transformer",
                )
            )
        )

        components = {
            "transformer": self.get_module("transformer"),
            "controlnet": self.get_module("controlnet"),
        }
        for component_name, component in components.items():
            if component_name not in target_components:
                logger.info(
                    "Skipping CogVideoX QKV fusion for VividVR %s because it is not in the requested targets=%s.",
                    component_name,
                    sorted(target_components),
                )
                continue
            if component is None:
                logger.warning(
                    "Skipping CogVideoX QKV fusion for %s because the component is not loaded.",
                    component_name,
                )
                continue
            fused_modules = enable_cogvideox_qkv_fusion(component)
            logger.info(
                "Enabled CogVideoX QKV fusion on VividVR %s; attention_modules=%s, effective_impl=%s.",
                component_name,
                fused_modules,
                _inspect_module_qkv_fusion(component),
            )

    def _apply_qk_norm_fusion(self, server_args: ServerArgs) -> None:
        if not getattr(server_args, "enable_cogvideox_qk_norm_fusion", False):
            return

        resolved_backend = normalize_cogvideox_attention_backend(
            server_args.attention_backend
        )
        if resolved_backend != "fa":
            logger.warning(
                "CogVideoX QK-norm fusion is enabled, but the requested backend resolves to %s. "
                "Phase E3 acceleration is currently consumed by the custom flash-attention path.",
                resolved_backend,
            )
        target_components = set(
            _normalize_qk_norm_fusion_targets(
                getattr(
                    server_args,
                    "cogvideox_qk_norm_fusion_targets",
                    "transformer",
                )
            )
        )

        components = {
            "transformer": self.get_module("transformer"),
            "controlnet": self.get_module("controlnet"),
        }
        for component_name, component in components.items():
            if component_name not in target_components:
                logger.info(
                    "Skipping CogVideoX QK-norm fusion for VividVR %s because it is not in the requested targets=%s.",
                    component_name,
                    sorted(target_components),
                )
                continue
            if component is None:
                logger.warning(
                    "Skipping CogVideoX QK-norm fusion for %s because the component is not loaded.",
                    component_name,
                )
                continue
            fused_modules = enable_cogvideox_qk_norm_fusion(component)
            logger.info(
                "Enabled CogVideoX QK-norm fusion on VividVR %s; attention_modules=%s, effective_impl=%s.",
                component_name,
                fused_modules,
                _inspect_module_qk_norm_fusion(component),
            )

    def _apply_qk_norm_rope_fusion(self, server_args: ServerArgs) -> None:
        if not getattr(server_args, "enable_cogvideox_qk_norm_rope_fusion", False):
            return

        resolved_backend = normalize_cogvideox_attention_backend(
            server_args.attention_backend
        )
        if resolved_backend != "fa":
            logger.warning(
                "CogVideoX QK-norm/RoPE fusion is enabled, but the requested backend resolves to %s. "
                "This Phase E3 path is only consumed by the custom flash-attention processor.",
                resolved_backend,
            )

        target_components = set(
            _normalize_qk_norm_rope_fusion_targets(
                getattr(
                    server_args,
                    "cogvideox_qk_norm_rope_fusion_targets",
                    "transformer",
                )
            )
        )
        components = {
            "transformer": self.get_module("transformer"),
            "controlnet": self.get_module("controlnet"),
        }
        for component_name, component in components.items():
            if component_name not in target_components:
                logger.info(
                    "Skipping CogVideoX QK-norm/RoPE fusion for VividVR %s because it is not in the requested targets=%s.",
                    component_name,
                    sorted(target_components),
                )
                continue
            if component is None:
                logger.warning(
                    "Skipping CogVideoX QK-norm/RoPE fusion for %s because the component is not loaded.",
                    component_name,
                )
                continue
            fused_modules = enable_cogvideox_qk_norm_rope_fusion(component)
            logger.info(
                "Enabled CogVideoX QK-norm/RoPE fusion on VividVR %s; attention_modules=%s, effective_impl=%s.",
                component_name,
                fused_modules,
                _inspect_module_qk_norm_rope_fusion(component),
            )

    def _apply_modulation_fusion(self, server_args: ServerArgs) -> None:
        if not getattr(server_args, "enable_cogvideox_modulation_fusion", False):
            return

        target_components = set(
            _normalize_modulation_fusion_targets(
                getattr(
                    server_args,
                    "cogvideox_modulation_fusion_targets",
                    "transformer",
                )
            )
        )
        components = {
            "transformer": self.get_module("transformer"),
            "controlnet": self.get_module("controlnet"),
        }
        for component_name, component in components.items():
            if component_name not in target_components:
                logger.info(
                    "Skipping CogVideoX modulation fusion for VividVR %s because it is not in the requested targets=%s.",
                    component_name,
                    sorted(target_components),
                )
                continue
            if component is None:
                logger.warning(
                    "Skipping CogVideoX modulation fusion for %s because the component is not loaded.",
                    component_name,
                )
                continue
            fused_blocks = enable_cogvideox_modulation_fusion(component)
            logger.info(
                "Enabled CogVideoX modulation fusion on VividVR %s; fused_blocks=%s, effective_impl=%s.",
                component_name,
                fused_blocks,
                _inspect_module_modulation_fusion(component),
            )

    def _apply_torch_compile(self, server_args: ServerArgs) -> None:
        if not server_args.enable_torch_compile:
            return

        for component_name in ("transformer", "controlnet"):
            component = self.get_module(component_name)
            if component is None:
                logger.warning(
                    "Skipping torch.compile for %s because the component is not loaded.",
                    component_name,
                )
                continue
            compiled_component = _maybe_torch_compile_module(
                component,
                enabled=True,
                module_name=component_name,
            )
            self.add_module(component_name, compiled_component)

    def initialize_pipeline(self, server_args: ServerArgs):
        _maybe_initialize_model_parallel_runtime(server_args)

        vivid_root = Path(
            server_args.component_paths.get(
                "vividvr",
                str(Path(self.model_path).resolve().parent / "Vivid-VR"),
            )
        )
        text_encoder_component_path = self._resolve_component_path(
            server_args,
            "text_encoder",
            "text_encoder",
        )
        transformer_component_path = self._resolve_component_path(
            server_args,
            "transformer",
            "transformer",
        )
        controlnet_dir = Path(
            server_args.component_paths.get("controlnet", str(vivid_root / "controlnet"))
        )
        text_encoder_dtype = PRECISION_TO_TYPE[
            server_args.pipeline_config.text_encoder_precisions[0]
        ]

        text_encoder = T5EncoderModel.from_pretrained(
            text_encoder_component_path,
            torch_dtype=text_encoder_dtype,
        )
        text_encoder = _VividVRT5EncoderWrapper(text_encoder).to(
            device=get_local_torch_device(),
            dtype=text_encoder_dtype,
        )
        text_encoder.eval()

        hf_config = get_diffusers_component_config(transformer_component_path)
        server_args.pipeline_config.dit_config.update_model_arch(hf_config)
        vividvr_config = CogVideoXConfig()
        vividvr_config.update_model_arch(hf_config)

        safetensors_list = resolve_transformer_safetensors_to_load(
            server_args,
            transformer_component_path,
        )
        quant_spec = resolve_transformer_quant_load_spec(
            hf_config=hf_config,
            server_args=server_args,
            safetensors_list=safetensors_list,
            component_model_path=transformer_component_path,
            model_cls=CogVideoXVividVRTransformer3DModel,
            cls_name=CogVideoXVividVRTransformer3DModel.__name__,
        )

        transformer = maybe_load_fsdp_model(
            model_cls=CogVideoXVividVRTransformer3DModel,
            init_params={
                "config": vividvr_config,
                "hf_config": hf_config,
                "quant_config": quant_spec.runtime_quant_config,
            },
            weight_dir_list=safetensors_list,
            device=get_local_torch_device(),
            hsdp_replicate_dim=server_args.hsdp_replicate_dim,
            hsdp_shard_dim=server_args.hsdp_shard_dim,
            cpu_offload=server_args.dit_cpu_offload,
            pin_cpu_memory=server_args.pin_cpu_memory,
            fsdp_inference=server_args.use_fsdp_inference,
            param_dtype=quant_spec.param_dtype,
            reduce_dtype=torch.float32,
            output_dtype=None,
            strict=False,
        )
        for post_load_hook in quant_spec.post_load_hooks:
            post_load_hook(transformer)
        transformer.load_connectors(str(vivid_root / "connectors.pt"))
        transformer.load_control_feat_proj(str(vivid_root / "control_feat_proj.pt"))
        transformer.load_control_patch_embed(str(vivid_root / "control_patch_embed.pt"))
        transformer.patch_embed.use_positional_embeddings = False
        transformer.patch_embed.use_learned_positional_embeddings = False
        transformer.config.use_learned_positional_embeddings = False
        transformer.config.use_rotary_positional_embeddings = True
        transformer.eval()

        controlnet_config = get_diffusers_component_config(str(controlnet_dir))
        controlnet_dtype = quant_spec.param_dtype or PRECISION_TO_TYPE[
            server_args.pipeline_config.dit_precision
        ]
        with set_default_torch_dtype(controlnet_dtype), torch.device("meta"):
            controlnet = CogVideoXVividVRControlNetModel(**controlnet_config)
        controlnet_state_dict = load_file(
            str(controlnet_dir / "diffusion_pytorch_model.safetensors"),
            device="cpu",
        )
        controlnet.load_state_dict(controlnet_state_dict, strict=True, assign=True)
        controlnet = controlnet.to(device=get_local_torch_device(), dtype=controlnet_dtype)
        controlnet.eval()

        self.add_module("text_encoder", text_encoder)
        self.add_module("transformer", transformer)
        self.add_module("controlnet", controlnet)
        self._apply_attention_backend(server_args)
        self._apply_usp_collective_optimizations(server_args)
        self._apply_qk_norm_fusion(server_args)
        self._apply_qk_norm_rope_fusion(server_args)
        self._apply_modulation_fusion(server_args)
        self._apply_qkv_fusion(server_args)
        self._apply_torch_compile(server_args)

        vae = self.get_module("vae")
        _configure_vividvr_vae_spatial_tile_parallel(
            vae, bool(server_args.pipeline_config.vae_sp)
        )
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        self.video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor)

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        del server_args
        self.input_validation_stage = VividVRInputValidationStage()
        self.prompt_preparation_stage = VividVRPromptPreparationStage()
        self.text_encoding_stage = VividVRTextEncodingStage(
            text_encoder=self.get_module("text_encoder"),
            tokenizer=self.get_module("tokenizer"),
            transformer=self.get_module("transformer"),
        )
        self.condition_encoding_stage = VividVRConditionEncodingStage(
            vae=self.get_module("vae"),
            transformer=self.get_module("transformer"),
            video_processor=self.video_processor,
        )
        self.latent_preparation_stage = VividVRLatentPreparationStage(
            vae=self.get_module("vae"),
            transformer=self.get_module("transformer"),
            scheduler=self.get_module("scheduler"),
        )
        self.tiling_preparation_stage = VividVRTilingPreparationStage()
        self.timestep_preparation_stage = VividVRTimestepPreparationStage(
            scheduler=self.get_module("scheduler"),
            transformer=self.get_module("transformer"),
        )
        self.denoising_stage = VividVRDenoisingStage(
            transformer=self.get_module("transformer"),
            controlnet=self.get_module("controlnet"),
            scheduler=self.get_module("scheduler"),
        )
        self.decoding_stage = VividVRDecodingStage(
            vae=self.get_module("vae"),
        )
        self.output_postprocess_stage = VividVROutputPostprocessStage(
            video_processor=self.video_processor,
        )
        self.temporal_window_planning_stage = VividVRTemporalWindowPlanningStage()
        self.long_clip_preparation_stage = VividVRLongClipPreparationStage(
            text_encoding_stage=self.text_encoding_stage,
            condition_encoding_stage=self.condition_encoding_stage,
            latent_preparation_stage=self.latent_preparation_stage,
            tiling_preparation_stage=self.tiling_preparation_stage,
        )
        self.multi_clip_denoising_stage = VividVRMultiClipDenoisingStage(
            denoising_stage=self.denoising_stage,
            vae_scale_factor_temporal=int(
                self.get_module("vae").config.temporal_compression_ratio
            ),
        )
        self.multi_clip_decode_trim_stage = VividVRMultiClipDecodeTrimStage(
            decoding_stage=self.decoding_stage,
            video_processor=self.video_processor,
        )
        self.temporal_stitch_postprocess_stage = (
            VividVRTemporalStitchPostprocessStage()
        )
        self.vividvr_stages = [
            self.input_validation_stage,
            self.prompt_preparation_stage,
            self.text_encoding_stage,
            self.condition_encoding_stage,
            self.latent_preparation_stage,
            self.tiling_preparation_stage,
            self.timestep_preparation_stage,
            self.denoising_stage,
            self.decoding_stage,
            self.output_postprocess_stage,
        ]
        self.vividvr_long_video_stages = [
            self.input_validation_stage,
            self.prompt_preparation_stage,
            self.temporal_window_planning_stage,
            self.long_clip_preparation_stage,
            self.timestep_preparation_stage,
            self.multi_clip_denoising_stage,
            self.multi_clip_decode_trim_stage,
            self.temporal_stitch_postprocess_stage,
        ]
        self.add_stages(self.vividvr_stages)

    def _build_temporal_clip_video_info(
        self,
        input_video_info: dict[str, object],
        clip_spec,
    ) -> dict[str, object]:
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

    def _ensure_temporal_windowed_runtime(
        self, server_args: ServerArgs
    ) -> list[object]:
        if getattr(self, "executor", None) is None:
            self.executor = self.build_executor(server_args)

        if getattr(self, "temporal_window_planning_stage", None) is None:
            self.temporal_window_planning_stage = _build_stage_without_global_server_args(
                VividVRTemporalWindowPlanningStage,
                server_args=server_args,
            )
        if getattr(self, "long_clip_preparation_stage", None) is None:
            self.long_clip_preparation_stage = _build_stage_without_global_server_args(
                VividVRLongClipPreparationStage,
                server_args=server_args,
                text_encoding_stage=self.text_encoding_stage,
                condition_encoding_stage=self.condition_encoding_stage,
                latent_preparation_stage=self.latent_preparation_stage,
                tiling_preparation_stage=self.tiling_preparation_stage,
            )
        if getattr(self, "multi_clip_denoising_stage", None) is None:
            self.multi_clip_denoising_stage = _build_stage_without_global_server_args(
                VividVRMultiClipDenoisingStage,
                server_args=server_args,
                denoising_stage=self.denoising_stage,
                vae_scale_factor_temporal=int(
                    self.get_module("vae").config.temporal_compression_ratio
                ),
            )
        if getattr(self, "multi_clip_decode_trim_stage", None) is None:
            self.multi_clip_decode_trim_stage = _build_stage_without_global_server_args(
                VividVRMultiClipDecodeTrimStage,
                server_args=server_args,
                decoding_stage=self.decoding_stage,
                video_processor=self.video_processor,
            )
        if getattr(self, "temporal_stitch_postprocess_stage", None) is None:
            self.temporal_stitch_postprocess_stage = _build_stage_without_global_server_args(
                VividVRTemporalStitchPostprocessStage,
                server_args=server_args,
            )

        stages = getattr(self, "vividvr_long_video_stages", None)
        if stages is None:
            stages = [
                self.input_validation_stage,
                self.prompt_preparation_stage,
                self.temporal_window_planning_stage,
                self.long_clip_preparation_stage,
                self.timestep_preparation_stage,
                self.multi_clip_denoising_stage,
                self.multi_clip_decode_trim_stage,
                self.temporal_stitch_postprocess_stage,
            ]
            self.vividvr_long_video_stages = stages
        return stages

    def _run_temporal_windowed_compat_stage(
        self,
        stage: object,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        if hasattr(stage, "forward"):
            return stage.forward(batch, server_args)
        if callable(stage):
            return stage(batch, server_args)
        raise TypeError(
            "Temporal-windowed compatibility stage is not executable: "
            f"{type(stage).__name__}"
        )

    def _forward_temporal_windowed_compat(
        self,
        batch: Req,
        server_args: ServerArgs,
        input_video_info: dict[str, object],
    ) -> Req:
        input_video_info = self._enrich_input_video_info_with_generation_resolution(
            input_video_info,
            tile_size=int(_as_vividvr_params(batch).tile_size),
        )
        params = _as_vividvr_params(batch)
        batch = self._run_temporal_windowed_compat_stage(
            self.input_validation_stage,
            batch,
            server_args,
        )
        batch = self._run_temporal_windowed_compat_stage(
            self.prompt_preparation_stage,
            batch,
            server_args,
        )

        debug = batch.extra.setdefault("vividvr_debug", {})
        long_runtime = batch.extra.setdefault("vividvr_long_video_runtime", {})

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
        debug["clip_specs"] = [
            _clip_spec_record(clip_spec) for clip_spec in window_plan.clip_specs
        ]

        generator = torch.Generator(device=get_local_torch_device().type).manual_seed(
            int(params.seed)
        )
        params.runtime_generator = generator
        batch.generator = generator

        clip_states: list[dict[str, object]] = []
        clip_caption_records: list[dict[str, object]] = []
        clip_latent_lengths: list[int] = []
        clip_tile_counts: list[int] = []
        caption_cursor = 0
        for clip_spec in window_plan.clip_specs:
            clip_video_info = self._build_temporal_clip_video_info(
                input_video_info,
                clip_spec,
            )
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
            clip_tile_counts.append(
                int(tiling_state.get("tile_count", len(tiling_state["tiling_infos"])))
            )

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
        params.runtime_negative_prompt_embeds = clip_states[0][
            "negative_prompt_embeds"
        ]
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

        timesteps = self.timestep_preparation_stage.prepare_timesteps(
            params.num_inference_steps
        )
        params.runtime_timesteps = timesteps
        params.runtime_timestep_count = len(timesteps)
        batch.timesteps = timesteps
        debug["timestep_count"] = len(timesteps)

        merge_plan = None
        if len(clip_states) > 1:
            merge_plan = build_vividvr_temporal_latent_merge_plan(
                clip_latent_lengths,
                num_temporal_process_frames=params.num_temporal_process_frames,
                vae_scale_factor_temporal=int(
                    self.get_module("vae").config.temporal_compression_ratio
                ),
            )
        params.runtime_temporal_merge_plan = merge_plan

        denoising_states: list[dict[str, object]] = []
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
                if progress_bar is not None:
                    progress_bar.update()

        debug["latents_shape"] = tuple(denoising_states[0]["latents"].shape)

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
        debug["vae_tiling_enabled"] = bool(
            getattr(getattr(self.decoding_stage, "vae", None), "use_tiling", False)
        )
        debug.update(_aggregate_vae_spatial_decode_stats(clip_vae_stats))

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

    def _forward_temporal_windowed(
        self,
        batch: Req,
        server_args: ServerArgs,
        input_video_info: dict[str, object],
    ) -> Req:
        input_video_info = self._enrich_input_video_info_with_generation_resolution(
            input_video_info,
            tile_size=int(_as_vividvr_params(batch).tile_size),
        )
        batch.extra["vividvr_input_video_info"] = input_video_info
        stages = self._ensure_temporal_windowed_runtime(server_args)
        if not hasattr(batch, "profile") or not hasattr(batch, "is_warmup"):
            del stages
            return self._forward_temporal_windowed_compat(
                batch,
                server_args,
                input_video_info,
            )
        return self.executor.execute_with_profiling(
            stages,
            batch,
            server_args,
        )

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs):
        params = _as_vividvr_params(batch)
        raw_input_video_info = self._resolve_input_video_info(
            params.video_input_path,
            upscale=float(params.upscale),
        )
        input_video_info = self._enrich_input_video_info_with_generation_resolution(
            raw_input_video_info,
            tile_size=int(params.tile_size),
        )

        if int(input_video_info["original_num_frames"]) <= params.num_temporal_process_frames:
            batch.extra["vividvr_input_video_info"] = input_video_info
            result = super().forward(batch, server_args)
            self._attach_runtime_acceleration_debug(result, server_args)
            return result

        if self.is_lora_set() and not self.is_lora_effective():
            logger.warning(
                "LoRA adapter is set, but not effective. Please make sure the LoRA weights are merged"
            )
        if not batch.is_warmup and not batch.suppress_logs:
            logger.info(
                "Running pipeline stages: %s",
                [
                    stage.__class__.__name__
                    for stage in self.vividvr_long_video_stages
                ],
                main_process_only=True,
            )
        result = self._forward_temporal_windowed(batch, server_args, input_video_info)
        self._attach_runtime_acceleration_debug(result, server_args)
        return result


EntryClass = VividVRPipeline
