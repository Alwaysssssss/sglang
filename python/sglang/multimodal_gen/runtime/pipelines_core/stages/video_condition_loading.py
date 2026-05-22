# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import imageio
import numpy as np
import PIL.Image
import PIL.ImageOps

from sglang.multimodal_gen.runtime.models.vision_utils import (
    PIL_INTERPOLATION,
    load_video,
    normalize,
    numpy_to_pt,
    pil_to_numpy,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs


@dataclass
class ConditionVideoMetadata:
    width: int
    height: int
    fps: float | None
    num_frames: int


class STARConditionVideoLoadingStage(PipelineStage):
    """Load and preprocess a full condition video for STAR-style pipelines."""

    @staticmethod
    def _validate_positive_int(name: str, value: int | None) -> int | None:
        if value is None:
            return None
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive int, got {value!r}")
        return value

    @staticmethod
    def _validate_non_negative_int(name: str, value: int | None) -> int | None:
        if value is None:
            return None
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative int, got {value!r}")
        return value

    @staticmethod
    def _inspect_video_metadata(
        path: str, frames: list[PIL.Image.Image]
    ) -> ConditionVideoMetadata:
        first_frame = frames[0]
        fps = None

        if path.lower().endswith(".gif"):
            with PIL.Image.open(path) as gif:
                duration_ms = gif.info.get("duration")
            if duration_ms and duration_ms > 0:
                fps = 1000.0 / float(duration_ms)
        else:
            with imageio.get_reader(path) as reader:
                metadata = reader.get_meta_data()
            raw_fps = metadata.get("fps")
            if raw_fps is not None:
                fps = float(raw_fps)

        return ConditionVideoMetadata(
            width=first_frame.width,
            height=first_frame.height,
            fps=fps,
            num_frames=len(frames),
        )

    @staticmethod
    def _resolve_target_size(
        batch: Req,
        pipeline_config,
        source_width: int,
        source_height: int,
    ) -> tuple[int, int]:
        explicit_fields = set(batch.extra.get("explicit_fields", []))

        target_width = batch.width if "width" in explicit_fields else None
        target_height = batch.height if "height" in explicit_fields else None

        if target_width is None:
            target_width = getattr(pipeline_config, "width", None) or batch.width
        if target_height is None:
            target_height = getattr(pipeline_config, "height", None) or batch.height

        if target_width is None and target_height is None:
            target_width = source_width
            target_height = source_height
        elif target_width is None:
            target_width = max(1, round(target_height * source_width / source_height))
        elif target_height is None:
            target_height = max(1, round(target_width * source_height / source_width))

        if target_width <= 0 or target_height <= 0:
            raise ValueError(
                f"Resolved invalid condition video size {target_width}x{target_height}"
            )

        return int(target_width), int(target_height)

    @classmethod
    def _resolve_target_num_frames(
        cls,
        batch: Req,
        metadata: ConditionVideoMetadata,
        pipeline_config,
    ) -> int | None:
        requested_condition_frames = cls._validate_positive_int(
            "condition_video_num_frames", batch.condition_video_num_frames
        )
        if requested_condition_frames is not None:
            return requested_condition_frames

        explicit_fields = set(batch.extra.get("explicit_fields", []))
        if "num_frames" in explicit_fields:
            return cls._validate_positive_int("num_frames", batch.num_frames)

        default_num_frames = getattr(pipeline_config, "num_frames", None)
        if default_num_frames is not None:
            return cls._validate_positive_int("pipeline_config.num_frames", default_num_frames)

        return metadata.num_frames

    @classmethod
    def _select_frame_indices(
        cls,
        batch: Req,
        metadata: ConditionVideoMetadata,
        pipeline_config,
    ) -> list[int]:
        start_frame = cls._validate_non_negative_int(
            "condition_video_start_frame", batch.condition_video_start_frame
        )
        if start_frame is None:
            start_frame = 0
        if start_frame >= metadata.num_frames:
            raise ValueError(
                f"condition_video_start_frame {start_frame} is out of range for a video with {metadata.num_frames} frames"
            )

        indices = list(range(start_frame, metadata.num_frames))

        frame_stride = cls._validate_positive_int(
            "condition_video_frame_stride", batch.condition_video_frame_stride
        )
        sample_fps = cls._validate_positive_int(
            "condition_video_sample_fps", batch.condition_video_sample_fps
        )

        if frame_stride is not None and sample_fps is not None:
            raise ValueError(
                "condition_video_frame_stride and condition_video_sample_fps are mutually exclusive"
            )

        if frame_stride is not None:
            indices = indices[::frame_stride]
        elif sample_fps is not None and metadata.fps is not None:
            effective_stride = max(int(round(metadata.fps / sample_fps)), 1)
            indices = indices[::effective_stride]

        if not indices:
            raise ValueError("No condition video frames remain after frame selection")

        target_num_frames = cls._resolve_target_num_frames(
            batch=batch,
            metadata=metadata,
            pipeline_config=pipeline_config,
        )
        if target_num_frames is None or target_num_frames == len(indices):
            return indices

        sample_positions = np.linspace(
            0,
            len(indices) - 1,
            num=target_num_frames,
        )
        return [indices[int(round(position))] for position in sample_positions]

    @staticmethod
    def _resize_and_crop_frame(
        frame: PIL.Image.Image,
        width: int,
        height: int,
    ) -> PIL.Image.Image:
        return PIL.ImageOps.fit(
            frame.convert("RGB"),
            (width, height),
            method=PIL_INTERPOLATION["lanczos"],
            centering=(0.5, 0.5),
        )

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        condition_video_path = batch.condition_video_path
        if not condition_video_path:
            raise ValueError("condition_video_path is required for STAR condition video loading")

        frames = load_video(condition_video_path)
        if not frames:
            raise ValueError(f"No frames could be loaded from {condition_video_path!r}")

        metadata = self._inspect_video_metadata(condition_video_path, frames)
        target_width, target_height = self._resolve_target_size(
            batch=batch,
            pipeline_config=server_args.pipeline_config,
            source_width=metadata.width,
            source_height=metadata.height,
        )
        frame_indices = self._select_frame_indices(
            batch=batch,
            metadata=metadata,
            pipeline_config=server_args.pipeline_config,
        )

        selected_frames = [
            self._resize_and_crop_frame(frames[index], target_width, target_height)
            for index in frame_indices
        ]
        condition_video = normalize(numpy_to_pt(pil_to_numpy(selected_frames))).unsqueeze(0)
        if batch.num_outputs_per_prompt > 1:
            condition_video = condition_video.repeat(batch.num_outputs_per_prompt, 1, 1, 1, 1)

        batch.condition_video = condition_video
        batch.original_condition_video_size = (metadata.width, metadata.height)
        batch.original_condition_video_fps = metadata.fps
        batch.condition_video_indices = frame_indices
        batch.condition_video_num_frames = len(frame_indices)
        batch.width = target_width
        batch.height = target_height
        return batch
