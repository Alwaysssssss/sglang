# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.videoedit_wan import (
    WanVideoEditPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.videoedit_wan import (
    WanVideoEditSamplingParams,
)
from sglang.multimodal_gen.runtime.models.schedulers.videoedit_flow_match import (
    VideoEditFlowMatchScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.videoedit_wan import (
    VideoEditConditionEncodingStage,
    VideoEditDecodingStage,
    VideoEditDenoisingStage,
    VideoEditImageEncodingStage,
    VideoEditLatentInitStage,
    VideoEditLatentPreparationStage,
    VideoEditTextEncodingStage,
    VideoEditTimestepPreparationStage,
    VideoEditWindowPostprocessStage,
    VideoEditWindowValidationStage,
)
from sglang.multimodal_gen.runtime.distributed import get_world_rank
from sglang.multimodal_gen.runtime.request_timeout import check_request_timeout
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.videoedit.ffmpeg_io import (
    save_video_frames_like_reference,
)
from sglang.multimodal_gen.runtime.videoedit.frame_provider import (
    WindowFrameProvider,
)
from sglang.multimodal_gen.runtime.videoedit.io import save_video_frames
from sglang.multimodal_gen.runtime.videoedit.postprocess import paste_back
from sglang.multimodal_gen.runtime.videoedit.preprocess import (
    VideoEditSequence,
    build_videoedit_bridge,
    materialize_videoedit_pass,
    materialize_videoedit_window,
    prepare_global_inputs,
    resize_frames,
    scan_global_bbox,
)
from sglang.multimodal_gen.runtime.videoedit.progress import (
    build_window_progress_payload,
    write_videoedit_progress,
)
from sglang.multimodal_gen.runtime.videoedit.windowing import (
    VideoEditPassPlan,
    build_videoedit_pass_window_specs,
    plan_videoedit_passes,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _load_videoedit_clip_encoder(
    image_encoder_path: str, dtype: torch.dtype
) -> torch.nn.Module:
    # VideoEdit's numerical baseline is the native HF CLIPVisionModel.  The
    # optimized SGLang CLIP implementation currently does not reproduce
    # hidden_states[-2], so using it changes the edit conditioning itself.
    from transformers import CLIPVisionModel

    return CLIPVisionModel.from_pretrained(image_encoder_path, dtype=dtype)


def _load_videoedit_text_encoder(
    text_encoder_path: str, dtype: torch.dtype
) -> torch.nn.Module:
    # Keep VideoEdit's prompt-conditioning boundary numerically identical to
    # the reference pipeline.  The optimized UMT5 implementation diverges
    # from HF before the first transformer block even with identical tokens.
    from transformers import UMT5EncoderModel

    return UMT5EncoderModel.from_pretrained(
        text_encoder_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )


def _load_videoedit_vae(vae_path: str, dtype: torch.dtype) -> torch.nn.Module:
    # VideoEdit's condition and output boundaries are defined by Diffusers'
    # AutoencoderKLWan.  The optimized Wan VAE is not numerically identical.
    from diffusers import AutoencoderKLWan

    return AutoencoderKLWan.from_pretrained(
        vae_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )


def _as_videoedit_params(batch: Req) -> WanVideoEditSamplingParams:
    params = batch.sampling_params
    if not isinstance(params, WanVideoEditSamplingParams):
        raise TypeError(
            "WanVideoEditPipeline requires WanVideoEditSamplingParams, "
            f"got {type(params).__name__}"
        )
    return params


def _is_output_rank() -> bool:
    try:
        return get_world_rank() == 0
    except Exception:
        return True


def _pil_frames_to_video_tensor(frames: list[Image.Image]) -> torch.Tensor:
    arrays = [
        np.array(frame.convert("RGB")).astype(np.float32) / 255.0
        for frame in frames
    ]
    stacked = torch.from_numpy(np.stack(arrays, axis=0))
    return stacked.permute(3, 0, 1, 2).unsqueeze(0).contiguous()


class WanVideoEditPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "WanVideoEditPipeline"
    pipeline_config_cls = WanVideoEditPipelineConfig
    sampling_params_cls = WanVideoEditSamplingParams

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        modules = dict(loaded_modules or {})
        if modules.get("text_encoder") is None:
            text_encoder_path = self._resolve_component_path(
                server_args, "text_encoder", "text_encoder"
            )
            modules["text_encoder"] = _load_videoedit_text_encoder(
                text_encoder_path, torch.bfloat16
            ).eval()
            self.memory_usages["text_encoder"] = sum(
                parameter.numel() * parameter.element_size()
                for parameter in modules["text_encoder"].parameters()
            ) / (1024**3)
        if modules.get("vae") is None:
            vae_path = self._resolve_component_path(server_args, "vae", "vae")
            modules["vae"] = _load_videoedit_vae(vae_path, torch.bfloat16).eval()
            self.memory_usages["vae"] = sum(
                parameter.numel() * parameter.element_size()
                for parameter in modules["vae"].parameters()
            ) / (1024**3)
        return super().load_modules(server_args, loaded_modules=modules)

    def initialize_pipeline(self, server_args: ServerArgs):
        self.modules["scheduler"] = VideoEditFlowMatchScheduler(
            shift=server_args.pipeline_config.flow_shift or 5.0,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self._maybe_load_image_encoder(server_args)

    def _maybe_load_image_encoder(self, server_args: ServerArgs) -> None:
        if self.modules.get("image_encoder") is None:
            override_path = server_args.component_paths.get("image_encoder")
            default_path = os.path.join(self.model_path, "image_encoder")
            image_encoder_path = override_path or default_path
            if not os.path.isdir(image_encoder_path):
                logger.warning(
                    "VideoEdit image_encoder was not found at %s; requests with "
                    "use_clip=True will fail.",
                    image_encoder_path,
                )
            else:
                module = _load_videoedit_clip_encoder(
                    image_encoder_path,
                    torch.float32,
                )
                self.modules["image_encoder"] = module.eval()
                self.memory_usages["image_encoder"] = 0.0

        if self.modules.get("image_processor") is None:
            processor_path = server_args.component_paths.get(
                "image_processor", os.path.join(self.model_path, "image_processor")
            )
            if not os.path.isdir(processor_path):
                logger.warning(
                    "VideoEdit image_processor was not found at %s; "
                    "clip_preprocess='diffuser' requests will fail.",
                    processor_path,
                )
            else:
                from transformers import CLIPImageProcessor

                self.modules["image_processor"] = CLIPImageProcessor.from_pretrained(
                    processor_path
                )

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.videoedit_stages = [
            VideoEditWindowValidationStage(),
            VideoEditTextEncodingStage(
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                transformer=self.get_module("transformer"),
            ),
            VideoEditImageEncodingStage(
                image_encoder=self.get_module("image_encoder", None),
                image_processor=self.get_module("image_processor", None),
                transformer=self.get_module("transformer"),
            ),
            VideoEditConditionEncodingStage(vae=self.get_module("vae")),
            VideoEditLatentPreparationStage(),
            VideoEditTimestepPreparationStage(scheduler=self.get_module("scheduler")),
            VideoEditLatentInitStage(scheduler=self.get_module("scheduler")),
            VideoEditDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                pipeline=self,
            ),
            VideoEditDecodingStage(vae=self.get_module("vae")),
            VideoEditWindowPostprocessStage(),
        ]
        self.add_stages(self.videoedit_stages)

    def _prepare_global_videoedit_context(
        self, params: WanVideoEditSamplingParams, batch: Req
    ) -> Image.Image:
        params.runtime_frame_provider = None
        if params.decode_mode == "stream":
            scanned_geometry = scan_global_bbox(
                input_video=params.video_input_path,
                mask_video=params.mask_input_path,
                num_frames=params.num_frames,
                reference_image=params.reference_image_path,
                bbox_padding=params.bbox_padding,
                dilate_px=params.dilate_px,
                mask_scale=params.mask_scale,
                bbox_expand_scale=params.bbox_expand_scale,
                align=16,
            )
            provider = WindowFrameProvider.from_scanned_geometry(
                video_input_path=params.video_input_path,
                mask_input_path=params.mask_input_path,
                reference_image_path=params.reference_image_path,
                scanned_geometry=scanned_geometry,
                dilate_px=params.dilate_px,
                mask_scale=params.mask_scale,
                infer_len=params.infer_len,
                enable_prefetch=True,
            )
            params.runtime_original_frames = None
            params.runtime_dilated_cropped_masks = None
            params.runtime_resized_frames = None
            params.runtime_resized_masks = None
            params.runtime_frame_provider = provider
            params.runtime_bbox = tuple(scanned_geometry["bbox"])
            params.runtime_crop_h = int(scanned_geometry["crop_h"])
            params.runtime_crop_w = int(scanned_geometry["crop_w"])
            params.runtime_aligned_h = int(scanned_geometry["aligned_h"])
            params.runtime_aligned_w = int(scanned_geometry["aligned_w"])
            params.runtime_fps = float(scanned_geometry["fps"])
            params.runtime_num_input_frames = int(scanned_geometry["num_frames"])
            resized_reference = provider.get_resized_reference_frame()
        else:
            data = prepare_global_inputs(
                input_video=params.video_input_path,
                mask_video=params.mask_input_path,
                num_frames=params.num_frames,
                reference_image=params.reference_image_path,
                bbox_padding=params.bbox_padding,
                dilate_px=params.dilate_px,
                mask_scale=params.mask_scale,
                bbox_expand_scale=params.bbox_expand_scale,
                align=16,
            )
            params.runtime_original_frames = data["original_frames"]
            params.runtime_dilated_cropped_masks = data["dilated_cropped_masks"]
            params.runtime_resized_frames = data["resized_video"]
            params.runtime_resized_masks = data["resized_masks"]
            params.runtime_bbox = data["bbox"]
            params.runtime_crop_h = data["crop_h"]
            params.runtime_crop_w = data["crop_w"]
            params.runtime_aligned_h = data["aligned_h"]
            params.runtime_aligned_w = data["aligned_w"]
            params.runtime_fps = data["fps"]
            params.runtime_num_input_frames = data["num_frames"]
            resized_reference = data["resized_reference"]
        if resized_reference is None:
            raise ValueError("VideoEdit requires an edited reference image")
        params.runtime_prev_window_output_frames = None
        params.runtime_prev_window_index = None
        params.runtime_window_materialize_metadata = []
        batch.height = params.runtime_aligned_h
        batch.width = params.runtime_aligned_w
        batch.fps = float(params.runtime_fps or batch.fps)
        return resized_reference

    def _materialize_pass_window(
        self,
        params: WanVideoEditSamplingParams,
        pass_plan: VideoEditPassPlan,
        window_spec: Any,
        *,
        eager_sequence: VideoEditSequence | None,
        bridge_frames: tuple[Image.Image, ...] | None,
        previous_output_frames: list[Image.Image] | None,
    ) -> None:
        provider = params.runtime_frame_provider
        if provider is not None:
            window = provider.materialize_pass_window(
                pass_plan,
                window_spec,
                bridge_frames=bridge_frames,
                previous_output_frames=previous_output_frames,
            )
        else:
            if eager_sequence is None:
                raise ValueError("Eager VideoEdit pass sequence is missing")
            window = materialize_videoedit_window(
                eager_sequence,
                window_spec,
                previous_output_frames=previous_output_frames,
            )
        params.runtime_window_frames = list(window.frames)
        params.runtime_window_masks = list(window.masks)
        if params.runtime_window_materialize_metadata is not None:
            params.runtime_window_materialize_metadata.append(
                {
                    "pass": pass_plan.name,
                    "direction": pass_plan.direction,
                    "window_index": window_spec.window_index,
                    "start_index": window_spec.start_index,
                    "end_index": window_spec.end_index,
                    "valid_len": window_spec.valid_len,
                    "stride": window_spec.stride,
                    "propagated_overlap": window_spec.overlap_mask_zero_count,
                    "commit_start_local_idx": window_spec.commit_start_local_idx,
                    "global_indices": list(window.global_indices),
                }
            )

    @staticmethod
    def _commit_pass_window(
        pass_plan: VideoEditPassPlan,
        window_spec: Any,
        window_output_frames: list[Image.Image],
        pass_outputs: list[Image.Image | None],
        generated_by_index: dict[int, Image.Image],
    ) -> None:
        if len(window_output_frames) < window_spec.valid_len:
            raise ValueError(
                f"VideoEdit {pass_plan.name} window {window_spec.window_index} "
                f"decoded {len(window_output_frames)} frames, need "
                f"valid_len={window_spec.valid_len}"
            )
        commit_start = window_spec.commit_start_local_idx
        for local_idx in range(commit_start, window_spec.valid_len):
            pass_position = window_spec.start_index + local_idx
            if pass_outputs[pass_position] is not None:
                raise RuntimeError(
                    f"VideoEdit {pass_plan.name} pass position {pass_position} "
                    "was committed more than once"
                )
            pass_outputs[pass_position] = window_output_frames[local_idx]

        for local_idx, global_idx in window_spec.commit_local_to_global.items():
            if global_idx in generated_by_index:
                raise RuntimeError(
                    f"VideoEdit global source index {global_idx} was committed "
                    "more than once"
                )
            generated_by_index[global_idx] = window_output_frames[local_idx]

    def _run_videoedit_pass(
        self,
        params: WanVideoEditSamplingParams,
        batch: Req,
        server_args: ServerArgs,
        pass_plan: VideoEditPassPlan,
        *,
        reference_frame: Image.Image,
        bridge_frames: tuple[Image.Image, ...] | None,
        generated_by_index: dict[int, Image.Image],
    ) -> tuple[list[Image.Image | None], list[Any]]:
        eager_sequence = None
        if params.runtime_frame_provider is None:
            if (
                params.runtime_resized_frames is None
                or params.runtime_resized_masks is None
            ):
                raise ValueError("Eager VideoEdit source frames and masks are missing")
            eager_sequence = materialize_videoedit_pass(
                pass_plan,
                source_frames=params.runtime_resized_frames,
                source_masks=params.runtime_resized_masks,
                reference_frame=reference_frame,
                bridge_frames=bridge_frames,
            )

        window_specs = build_videoedit_pass_window_specs(
            pass_plan.sequence_indices,
            infer_len=params.infer_len,
            overlap=params.overlap,
        )
        params.runtime_window_specs = window_specs
        pass_outputs: list[Image.Image | None] = [None] * len(
            pass_plan.sequence_indices
        )
        previous_output_frames: list[Image.Image] | None = None
        write_videoedit_progress(
            params.progress_path,
            build_window_progress_payload(
                stage="windowing",
                total_frames=params.runtime_num_input_frames,
                infer_len=params.infer_len,
                overlap=params.overlap,
                total_windows=len(window_specs),
            ),
        )

        for window_spec in window_specs:
            check_request_timeout(batch)
            params.reset_window_runtime(window_spec)
            write_videoedit_progress(
                params.progress_path,
                build_window_progress_payload(
                    stage="window_start",
                    total_frames=params.runtime_num_input_frames,
                    infer_len=params.infer_len,
                    overlap=params.overlap,
                    total_windows=len(window_specs),
                    current_window_index=window_spec.window_index,
                    steps_per_window=params.num_inference_steps,
                ),
            )
            self._materialize_pass_window(
                params,
                pass_plan,
                window_spec,
                eager_sequence=eager_sequence,
                bridge_frames=bridge_frames,
                previous_output_frames=previous_output_frames,
            )
            check_request_timeout(batch)
            self.executor.execute_with_profiling(self.stages, batch, server_args)
            check_request_timeout(batch)
            window_output_frames = params.runtime_window_output_frames
            if window_output_frames is None:
                raise ValueError("VideoEdit window output is missing")
            self._commit_pass_window(
                pass_plan,
                window_spec,
                window_output_frames,
                pass_outputs,
                generated_by_index,
            )
            previous_output_frames = window_output_frames
            params.runtime_prev_window_output_frames = window_output_frames
            params.runtime_prev_window_index = window_spec.window_index
            write_videoedit_progress(
                params.progress_path,
                build_window_progress_payload(
                    stage="window_done",
                    total_frames=params.runtime_num_input_frames,
                    infer_len=params.infer_len,
                    overlap=params.overlap,
                    total_windows=len(window_specs),
                    current_window_index=window_spec.window_index,
                    current_step_index=(
                        params.runtime_effective_num_inference_steps - 1
                        if params.runtime_effective_num_inference_steps
                        else None
                    ),
                    steps_per_window=(
                        params.runtime_effective_num_inference_steps
                        or params.num_inference_steps
                    ),
                ),
            )

        missing_positions = [
            index for index, frame in enumerate(pass_outputs) if frame is None
        ]
        if missing_positions:
            raise RuntimeError(
                f"VideoEdit {pass_plan.name} pass has uncommitted positions: "
                f"{missing_positions}"
            )
        return pass_outputs, window_specs

    @staticmethod
    def _finalize_crop_frames(
        params: WanVideoEditSamplingParams,
        generated_by_index: dict[int, Image.Image],
    ) -> list[Image.Image]:
        expected = set(range(params.runtime_num_input_frames))
        actual = set(generated_by_index)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise RuntimeError(
                "VideoEdit generated global indices do not match the source: "
                f"missing={missing}, extra={extra}"
            )
        return [generated_by_index[index] for index in range(len(expected))]

    def _write_metadata(
        self,
        params: WanVideoEditSamplingParams,
        output_video_path: str | None,
        num_output_frames: int | None = None,
        window_records: list[tuple[str, Any]] | None = None,
    ) -> None:
        if not _is_output_rank():
            return
        if output_video_path is None:
            return
        metadata_path = os.path.splitext(output_video_path)[0] + ".videoedit.json"
        metadata = {
            "video_input_path": params.video_input_path,
            "mask_input_path": params.mask_input_path,
            "reference_image_path": params.reference_image_path,
            "bbox": params.runtime_bbox,
            "crop_h": params.runtime_crop_h,
            "crop_w": params.runtime_crop_w,
            "aligned_h": params.runtime_aligned_h,
            "aligned_w": params.runtime_aligned_w,
            "fps": params.runtime_fps,
            "num_input_frames": params.runtime_num_input_frames,
            "num_output_frames": num_output_frames,
            "drop_reference_frame": params.drop_reference_frame,
            "enable_paste_back": params.enable_paste_back,
            "window_specs": [
                {
                    "pass": pass_name,
                    "window_index": spec.window_index,
                    "start_index": spec.start_index,
                    "end_index": spec.end_index,
                    "valid_len": spec.valid_len,
                    "input_indices": spec.input_indices,
                    "reflected_count": spec.reflected_count,
                    "stride": getattr(spec, "stride", None),
                    "reference_prev_local_idx": getattr(
                        spec, "reference_prev_local_idx", None
                    ),
                    "reference_global_index": getattr(
                        spec, "reference_global_index", None
                    ),
                    "overlap_mask_zero_count": getattr(
                        spec, "overlap_mask_zero_count", 0
                    ),
                    "commit_start_local_idx": getattr(
                        spec, "commit_start_local_idx", 0
                    ),
                }
                for pass_name, spec in (
                    window_records
                    or [
                        ("unknown", spec)
                        for spec in (params.runtime_window_specs or [])
                    ]
                )
            ],
            "window_materialize": params.runtime_window_materialize_metadata or [],
        }
        os.makedirs(os.path.dirname(os.path.abspath(metadata_path)), exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        params.runtime_metadata_path = metadata_path

    def _save_crop_sidecar(
        self,
        params: WanVideoEditSamplingParams,
        crop_frames: list[Image.Image],
        output_video_path: str | None,
    ) -> None:
        if not _is_output_rank():
            return
        if not params.save_crop_only or output_video_path is None:
            return
        base, ext = os.path.splitext(output_video_path)
        crop_ext = ext or ".mp4"
        crop_path = f"{base}_crop_only{crop_ext}"
        frames = resize_frames(
            crop_frames, params.runtime_crop_h, params.runtime_crop_w
        )
        if params.drop_reference_frame and len(frames) > 0:
            frames = frames[1:]
        if params.video_input_path:
            save_video_frames_like_reference(
                frames,
                crop_path,
                refer_file=params.video_input_path,
                fps=params.runtime_fps or params.fps,
                quality=None,
                bit_rate=10_000_000,
                copy_color_metadata=False,
            )
        else:
            save_video_frames(frames, crop_path, fps=params.runtime_fps or params.fps)
        params.runtime_crop_video_path = crop_path

    def _finalize_videoedit_output(
        self,
        params: WanVideoEditSamplingParams,
        batch: Req,
        generated_by_index: dict[int, Image.Image],
        window_records: list[tuple[str, Any]],
    ) -> list[Image.Image]:
        crop_frames = self._finalize_crop_frames(params, generated_by_index)
        output_video_path = batch.output_file_path()
        params.runtime_output_video_path = output_video_path
        self._save_crop_sidecar(params, crop_frames, output_video_path)

        if params.enable_paste_back:
            if params.runtime_frame_provider is not None:
                frames = params.runtime_frame_provider.paste_back_frames(
                    crop_frames,
                    feather_px=params.feather_px,
                    adain_boundary_dilate=params.adain_boundary_dilate,
                )
            else:
                frames = paste_back(
                    original_frames=params.runtime_original_frames,
                    generated_frames=crop_frames,
                    mask_frames=params.runtime_dilated_cropped_masks,
                    bbox=params.runtime_bbox,
                    crop_h=params.runtime_crop_h,
                    crop_w=params.runtime_crop_w,
                    feather_px=params.feather_px,
                    adain_boundary_dilate=params.adain_boundary_dilate,
                )
        else:
            frames = resize_frames(
                crop_frames, params.runtime_crop_h, params.runtime_crop_w
            )

        self._write_metadata(
            params,
            output_video_path,
            num_output_frames=len(frames),
            window_records=window_records,
        )
        return frames

    def _cleanup_videoedit_context(self, params: WanVideoEditSamplingParams) -> None:
        if params.runtime_frame_provider is not None:
            params.runtime_frame_provider.close()
            params.runtime_frame_provider = None

    @staticmethod
    def _set_final_batch_output(
        batch: Req,
        params: WanVideoEditSamplingParams,
        output_frames: list[Image.Image],
    ) -> None:
        batch.output = _pil_frames_to_video_tensor(output_frames)
        batch.num_frames = params.runtime_num_input_frames

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs):
        params = _as_videoedit_params(batch)
        if self.executor is None:
            raise RuntimeError("WanVideoEditPipeline requires a pipeline executor")

        if self.is_lora_set() and not self.is_lora_effective():
            logger.warning(
                "LoRA adapter is set, but not effective. Please make sure the "
                "LoRA weights are merged"
            )

        # Execute each stage
        if not batch.is_warmup and not batch.suppress_logs:
            logger.info(
                "Running pipeline stages: %s",
                list(self._stage_name_mapping.keys()),
                main_process_only=True,
            )

        with self.executor.profile_execution(batch, dump_rank=0):
            check_request_timeout(batch)
            try:
                reference_frame = self._prepare_global_videoedit_context(params, batch)
                check_request_timeout(batch)
                sequence_plan = plan_videoedit_passes(
                    params.runtime_num_input_frames,
                    params.ref_frame_idx,
                    params.bridge_overlap,
                )
                generated_by_index: dict[int, Image.Image] = {}
                window_records: list[tuple[str, Any]] = []

                long_outputs, long_specs = self._run_videoedit_pass(
                    params,
                    batch,
                    server_args,
                    sequence_plan.long,
                    reference_frame=reference_frame,
                    bridge_frames=None,
                    generated_by_index=generated_by_index,
                )
                window_records.extend(("long", spec) for spec in long_specs)

                if sequence_plan.short is not None:
                    check_request_timeout(batch)
                    bridge_frames = build_videoedit_bridge(
                        long_outputs,
                        sequence_plan.bridge_length,
                    )
                    _, short_specs = self._run_videoedit_pass(
                        params,
                        batch,
                        server_args,
                        sequence_plan.short,
                        reference_frame=reference_frame,
                        bridge_frames=bridge_frames,
                        generated_by_index=generated_by_index,
                    )
                    window_records.extend(("short", spec) for spec in short_specs)

                check_request_timeout(batch)
                params.runtime_window_specs = [spec for _, spec in window_records]
                output_frames = self._finalize_videoedit_output(
                    params,
                    batch,
                    generated_by_index,
                    window_records,
                )
                check_request_timeout(batch)
                self._set_final_batch_output(batch, params, output_frames)
            finally:
                self._cleanup_videoedit_context(params)

        return batch


EntryClass = WanVideoEditPipeline
