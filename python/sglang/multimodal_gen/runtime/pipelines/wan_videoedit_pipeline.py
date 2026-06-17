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
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    PipelineComponentLoader,
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
from sglang.multimodal_gen.runtime.request_timeout import check_request_timeout
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.videoedit.frame_provider import (
    WindowFrameProvider,
)
from sglang.multimodal_gen.runtime.videoedit.io import save_video_frames
from sglang.multimodal_gen.runtime.videoedit.postprocess import paste_back
from sglang.multimodal_gen.runtime.videoedit.preprocess import (
    prepare_global_inputs,
    resize_frames,
    scan_global_bbox,
)
from sglang.multimodal_gen.runtime.videoedit.progress import (
    build_window_progress_payload,
    write_videoedit_progress,
)
from sglang.multimodal_gen.runtime.videoedit.windowing import (
    build_videoedit_window_specs,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _as_videoedit_params(batch: Req) -> WanVideoEditSamplingParams:
    params = batch.sampling_params
    if not isinstance(params, WanVideoEditSamplingParams):
        raise TypeError(
            "WanVideoEditPipeline requires WanVideoEditSamplingParams, "
            f"got {type(params).__name__}"
        )
    return params


def _image_to_float_array(frame: Image.Image) -> np.ndarray:
    return np.array(frame.convert("RGB")).astype(np.float32)


def _float_array_to_image(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(frame, 0, 255).round().astype(np.uint8))


def _pil_frames_to_video_tensor(frames: list[Image.Image]) -> torch.Tensor:
    arrays = [np.array(frame.convert("RGB")).astype(np.float32) / 255.0 for frame in frames]
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

    def initialize_pipeline(self, server_args: ServerArgs):
        self.modules["scheduler"] = VideoEditFlowMatchScheduler(
            shift=server_args.pipeline_config.flow_shift or 5.0,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self._maybe_load_image_encoder(server_args)

    def _maybe_load_image_encoder(self, server_args: ServerArgs) -> None:
        if self.modules.get("image_encoder") is not None:
            return
        override_path = server_args.component_paths.get("image_encoder")
        default_path = os.path.join(self.model_path, "image_encoder")
        image_encoder_path = override_path or default_path
        if not os.path.isdir(image_encoder_path):
            logger.warning(
                "VideoEdit image_encoder was not found at %s; requests with use_clip=True will fail.",
                image_encoder_path,
            )
            return
        module, memory_usage = PipelineComponentLoader.load_component(
            component_name="image_encoder",
            component_model_path=image_encoder_path,
            transformers_or_diffusers="transformers",
            server_args=server_args,
        )
        self.modules["image_encoder"] = module
        self.memory_usages["image_encoder"] = memory_usage

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
    ) -> None:
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
        params.runtime_accum_frames = [
            np.zeros((params.runtime_aligned_h, params.runtime_aligned_w, 3), dtype=np.float32)
            for _ in range(params.runtime_num_input_frames)
        ]
        params.runtime_accum_weights = np.zeros(
            (params.runtime_num_input_frames,), dtype=np.float32
        )
        params.runtime_prev_window_output_frames = None
        params.runtime_prev_window_index = None
        batch.height = params.runtime_aligned_h
        batch.width = params.runtime_aligned_w
        batch.fps = max(1, int(round(params.runtime_fps or batch.fps)))

    def _materialize_window_inputs(
        self, params: WanVideoEditSamplingParams, window_spec: Any
    ) -> None:
        provider = params.runtime_frame_provider
        if provider is not None:
            source_frames, masks = provider.materialize_window(window_spec.input_indices)
        else:
            source_frames = [params.runtime_resized_frames[idx] for idx in window_spec.input_indices]
            masks = [params.runtime_resized_masks[idx] for idx in window_spec.input_indices]

        frames: list[Image.Image] = []
        for idx, source_frame in zip(window_spec.input_indices, source_frames, strict=True):
            use_repaired = (
                params.use_repaired_context
                and params.runtime_accum_weights is not None
                and idx < len(params.runtime_accum_weights)
                and params.runtime_accum_weights[idx] > 0
            )
            if use_repaired:
                repaired = (
                    params.runtime_accum_frames[idx] / params.runtime_accum_weights[idx]
                )
                frames.append(_float_array_to_image(repaired))
            else:
                frames.append(source_frame)
        if (
            params.overlap_commit_mode == "native_skip"
            and int(params.overlap) > 0
            and window_spec.window_index > 0
        ):
            if (
                params.runtime_prev_window_index == window_spec.window_index - 1
                and params.runtime_prev_window_output_frames is not None
            ):
                stride = params.infer_len - int(params.overlap)
                if 0 <= stride < len(params.runtime_prev_window_output_frames):
                    frames[0] = params.runtime_prev_window_output_frames[stride]
            if masks:
                w, h = masks[0].size
                black = Image.new("L", (w, h), 0)
                for i in range(min(int(params.overlap), len(masks))):
                    masks[i] = black.copy()
        params.runtime_window_frames = frames
        params.runtime_window_masks = masks

    # overlap权重，越靠近窗口边缘权重越小，中心区域权重为1.0，边缘区域权重线性衰减到0.0，最小不低于1e-6
    def _commit_weight(
        self,
        params: WanVideoEditSamplingParams,
        window_spec: Any,
        local_idx: int,
    ) -> float:
        overlap = int(params.overlap)
        if overlap <= 0:
            return 1.0
        weight = 1.0
        if window_spec.window_index > 0 and local_idx < overlap:
            weight = min(weight, float(local_idx + 1) / float(overlap + 1))
        if (
            params.runtime_window_specs is not None
            and window_spec.window_index < len(params.runtime_window_specs) - 1
            and local_idx >= params.infer_len - overlap
        ):
            weight = min(
                weight, float(params.infer_len - local_idx) / float(overlap + 1)
            )
        return max(weight, 1e-6)

    def _commit_window_output(
        self, params: WanVideoEditSamplingParams, window_spec: Any
    ) -> None:
        frames = params.runtime_window_output_frames
        if frames is None:
            raise ValueError("VideoEdit window output is missing")
        for local_idx, global_idx in window_spec.commit_local_to_global.items():
            if global_idx >= params.runtime_num_input_frames:
                continue
            if (
                params.overlap_commit_mode == "native_skip"
                and window_spec.window_index > 0
                and local_idx < int(params.overlap)
            ):
                continue
            weight = (
                1.0
                if params.overlap_commit_mode == "native_skip"
                else self._commit_weight(params, window_spec, local_idx)
            )
            params.runtime_accum_frames[global_idx] += (
                _image_to_float_array(frames[local_idx]) * weight
            )
            params.runtime_accum_weights[global_idx] += weight
        params.runtime_prev_window_output_frames = frames
        params.runtime_prev_window_index = window_spec.window_index

    def _finalize_crop_frames(self, params: WanVideoEditSamplingParams) -> list[Image.Image]:
        crop_frames: list[Image.Image] = []
        provider = params.runtime_frame_provider
        for idx, weight in enumerate(params.runtime_accum_weights):
            if weight > 0:
                crop_frames.append(_float_array_to_image(params.runtime_accum_frames[idx] / weight))
            else:
                if provider is not None:
                    crop_frames.append(provider.get_resized_frame(idx))
                else:
                    crop_frames.append(params.runtime_resized_frames[idx])
        return crop_frames

    def _write_metadata(
        self,
        params: WanVideoEditSamplingParams,
        output_video_path: str | None,
        num_output_frames: int | None = None,
    ) -> None:
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
                    "window_index": spec.window_index,
                    "start_index": spec.start_index,
                    "end_index": spec.end_index,
                    "valid_len": spec.valid_len,
                    "input_indices": spec.input_indices,
                    "reflected_count": spec.reflected_count,
                }
                for spec in (params.runtime_window_specs or [])
            ],
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
        if not params.save_crop_only or output_video_path is None:
            return
        crop_path = os.path.splitext(output_video_path)[0] + "_crop_only.mp4"
        frames = resize_frames(crop_frames, params.runtime_crop_h, params.runtime_crop_w)
        if params.drop_reference_frame and len(frames) > 0:
            frames = frames[1:]
        save_video_frames(frames, crop_path, fps=params.runtime_fps or params.fps)
        params.runtime_crop_video_path = crop_path

    def _finalize_long_video_output(
        self, params: WanVideoEditSamplingParams, batch: Req
    ) -> list[Image.Image]:
        crop_frames = self._finalize_crop_frames(params)
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
            frames = resize_frames(crop_frames, params.runtime_crop_h, params.runtime_crop_w)

        if params.drop_reference_frame and len(frames) > 0:
            frames = frames[1:]
        self._write_metadata(params, output_video_path, num_output_frames=len(frames))
        return frames

    def _cleanup_videoedit_context(self, params: WanVideoEditSamplingParams) -> None:
        if params.runtime_frame_provider is not None:
            params.runtime_frame_provider.close()
            params.runtime_frame_provider = None

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs):
        params = _as_videoedit_params(batch)
        if self.executor is None:
            raise RuntimeError("WanVideoEditPipeline requires a pipeline executor")
        
        if self.is_lora_set() and not self.is_lora_effective():
            logger.warning(
                "LoRA adapter is set, but not effective. Please make sure the LoRA weights are merged"
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
            self._prepare_global_videoedit_context(params, batch)
            check_request_timeout(batch)
            try:
                window_specs = build_videoedit_window_specs(
                    num_frames=params.runtime_num_input_frames,
                    infer_len=params.infer_len,
                    overlap=params.overlap,
                    tail_padding_mode=params.tail_padding_mode,
                )
                params.runtime_window_specs = window_specs
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
                    self._materialize_window_inputs(params, window_spec)
                    check_request_timeout(batch)
                    self.executor.execute_with_profiling(self.stages, batch, server_args)
                    check_request_timeout(batch)
                    self._commit_window_output(params, window_spec)
                    check_request_timeout(batch)
                    write_videoedit_progress(
                        params.progress_path,
                        build_window_progress_payload(
                            stage="window_done",
                            total_frames=params.runtime_num_input_frames,
                            infer_len=params.infer_len,
                            overlap=params.overlap,
                            total_windows=len(window_specs),
                            current_window_index=window_spec.window_index,
                            current_step_index=params.runtime_effective_num_inference_steps - 1
                            if params.runtime_effective_num_inference_steps
                            else None,
                            steps_per_window=params.runtime_effective_num_inference_steps
                            or params.num_inference_steps,
                        ),
                    )

                check_request_timeout(batch)
                output_frames = self._finalize_long_video_output(params, batch)
                check_request_timeout(batch)
                batch.output = _pil_frames_to_video_tensor(output_frames)
            finally:
                self._cleanup_videoedit_context(params)

        return batch


EntryClass = WanVideoEditPipeline
