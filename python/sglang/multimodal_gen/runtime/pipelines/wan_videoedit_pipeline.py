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
    VideoEditLatentInitStage,
    VideoEditLatentPreparationStage,
    VideoEditTextEncodingStage,
    VideoEditTimestepPreparationStage,
    VideoEditWindowPostprocessStage,
    VideoEditWindowValidationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.videoedit.io import save_video_frames
from sglang.multimodal_gen.runtime.videoedit.postprocess import paste_back
from sglang.multimodal_gen.runtime.videoedit.preprocess import (
    prepare_global_inputs,
    resize_frames,
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

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.videoedit_stages = [
            VideoEditWindowValidationStage(),
            VideoEditTextEncodingStage(
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
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
        data = prepare_global_inputs(
            input_video=params.video_input_path,
            mask_video=params.mask_input_path,
            num_frames=params.num_frames,
            reference_image=params.reference_image_path,
            bbox_padding=params.bbox_padding,
            dilate_px=params.dilate_px,
            mask_scale=params.mask_scale,
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
        batch.height = params.runtime_aligned_h
        batch.width = params.runtime_aligned_w
        batch.fps = max(1, int(round(params.runtime_fps or batch.fps)))

    def _materialize_window_inputs(
        self, params: WanVideoEditSamplingParams, window_spec: Any
    ) -> None:
        frames: list[Image.Image] = []
        for idx in window_spec.input_indices:
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
                frames.append(params.runtime_resized_frames[idx])
        masks = [params.runtime_resized_masks[idx] for idx in window_spec.input_indices]
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
            weight = self._commit_weight(params, window_spec, local_idx)
            params.runtime_accum_frames[global_idx] += (
                _image_to_float_array(frames[local_idx]) * weight
            )
            params.runtime_accum_weights[global_idx] += weight

    def _finalize_crop_frames(self, params: WanVideoEditSamplingParams) -> list[Image.Image]:
        crop_frames: list[Image.Image] = []
        for idx, weight in enumerate(params.runtime_accum_weights):
            if weight > 0:
                crop_frames.append(_float_array_to_image(params.runtime_accum_frames[idx] / weight))
            else:
                crop_frames.append(params.runtime_resized_frames[idx])
        return crop_frames

    def _write_metadata(
        self, params: WanVideoEditSamplingParams, output_video_path: str | None
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
            "num_output_frames": None,
            "drop_reference_frame": params.drop_reference_frame,
            "enable_paste_back": params.enable_paste_back,
            "window_specs": [
                {
                    "window_index": spec.window_index,
                    "start_index": spec.start_index,
                    "end_index": spec.end_index,
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
        self._write_metadata(params, output_video_path)
        return frames

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
            self._prepare_global_videoedit_context(params, batch)
            window_specs = build_videoedit_window_specs(
                num_frames=params.runtime_num_input_frames,
                infer_len=params.infer_len,
                overlap=params.overlap,
            )
            params.runtime_window_specs = window_specs
            for window_spec in window_specs:
                params.reset_window_runtime(window_spec)
                self._materialize_window_inputs(params, window_spec)
                # for stage in self.videoedit_stages:
                #     batch = stage(batch, server_args)
                self.executor.execute_with_profiling(self.stages, batch, server_args)
                self._commit_window_output(params, window_spec)

            output_frames = self._finalize_long_video_output(params, batch)
            batch.output = _pil_frames_to_video_tensor(output_frames)

        return batch


EntryClass = WanVideoEditPipeline
