# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import decord
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import torchvision.transforms as TT

from sglang.multimodal_gen.runtime.models.vision_utils import normalize
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
    def _paired_dataset_preprocess(
        frames: torch.Tensor,
        target_height: int,
        target_width: int,
    ) -> torch.Tensor:
        if frames.shape[-1] > target_width:
            scale_factor = target_width / frames.shape[-1]
            frames = F.interpolate(
                frames,
                scale_factor=scale_factor,
                mode="bilinear",
            )
            frames = TT.functional.center_crop(frames, (target_height, target_width))
        elif frames.shape[-1] < target_width:
            scale_factor = target_width / frames.shape[-1]
            frames = F.interpolate(
                frames,
                scale_factor=scale_factor,
                mode="bicubic",
            )
        return frames

    @staticmethod
    def _load_gif_frames(path: str) -> tuple[torch.Tensor, float | None]:
        pil_frames: list[torch.Tensor] = []
        with PIL.Image.open(path) as gif:
            duration_ms = gif.info.get("duration")
            fps = 1000.0 / float(duration_ms) if duration_ms and duration_ms > 0 else None
            try:
                while True:
                    pil_frames.append(
                        torch.from_numpy(np.array(gif.convert("RGB"), copy=True))
                    )
                    gif.seek(gif.tell() + 1)
            except EOFError:
                pass

        if not pil_frames:
            raise ValueError(f"No frames could be loaded from {path!r}")
        return torch.stack(pil_frames, dim=0), fps

    @staticmethod
    def _load_video_tensor(path: str) -> tuple[torch.Tensor, ConditionVideoMetadata]:
        if path.lower().endswith(".gif"):
            frames, fps = STARConditionVideoLoadingStage._load_gif_frames(path)
            return frames, ConditionVideoMetadata(
                width=int(frames.shape[2]),
                height=int(frames.shape[1]),
                fps=fps,
                num_frames=int(frames.shape[0]),
            )

        decord.bridge.set_bridge("torch")
        reader = decord.VideoReader(uri=path, height=-1, width=-1)
        if len(reader) == 0:
            raise ValueError(f"No frames could be loaded from {path!r}")
        indices = np.arange(len(reader))
        frames = reader.get_batch(indices)
        if not isinstance(frames, torch.Tensor):
            frames = torch.from_numpy(frames)

        fps = float(reader.get_avg_fps()) if hasattr(reader, "get_avg_fps") else None
        return frames, ConditionVideoMetadata(
            width=int(frames.shape[2]),
            height=int(frames.shape[1]),
            fps=fps,
            num_frames=int(frames.shape[0]),
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

        default_condition_frames = getattr(
            pipeline_config, "condition_video_num_frames", None
        )
        if default_condition_frames is not None:
            return cls._validate_positive_int(
                "pipeline_config.condition_video_num_frames",
                default_condition_frames,
            )

        default_num_frames = getattr(pipeline_config, "num_frames", None)
        if default_num_frames is not None:
            return cls._validate_positive_int(
                "pipeline_config.num_frames", default_num_frames
            )

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

        # Match STAR reference inference semantics: without an explicit stride/FPS
        # override, the condition clip is the leading contiguous window rather than
        # a uniformly resampled subset of the full source video.
        if (
            frame_stride is None
            and sample_fps is None
            and target_num_frames < len(indices)
        ):
            return indices[:target_num_frames]

        sample_positions = np.linspace(
            0,
            len(indices) - 1,
            num=target_num_frames,
        )
        return [indices[int(round(position))] for position in sample_positions]

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        condition_video_path = batch.condition_video_path
        if not condition_video_path:
            raise ValueError(
                "condition_video_path is required for STAR condition video loading"
            )

        frames, metadata = self._load_video_tensor(condition_video_path)
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

        selected_frames = frames.index_select(
            0, torch.tensor(frame_indices, dtype=torch.long)
        )
        condition_video = selected_frames.permute(0, 3, 1, 2).contiguous()
        condition_video = self._paired_dataset_preprocess(
            condition_video,
            target_height=target_height,
            target_width=target_width,
        )
        condition_video = normalize(condition_video.to(torch.float32) / 255.0).unsqueeze(0)
        if batch.num_outputs_per_prompt > 1:
            condition_video = condition_video.repeat(
                batch.num_outputs_per_prompt, 1, 1, 1, 1
            )

        batch.condition_video = condition_video
        batch.original_condition_video_size = (metadata.width, metadata.height)
        batch.original_condition_video_fps = metadata.fps
        batch.condition_video_indices = frame_indices
        batch.condition_video_num_frames = len(frame_indices)
        batch.width = target_width
        batch.height = target_height
        if batch.return_trajectory_latents and batch.metrics is not None:
            batch.metrics.record_annotation(
                "condition_video_preprocess_summary",
                {
                    "original_size": [metadata.width, metadata.height],
                    "target_size": [target_width, target_height],
                    "source_num_frames": metadata.num_frames,
                    "selected_indices": frame_indices,
                    "video_fps": metadata.fps,
                    "tensor_summary": self._tensor_summary(condition_video),
                },
            )
        return batch
