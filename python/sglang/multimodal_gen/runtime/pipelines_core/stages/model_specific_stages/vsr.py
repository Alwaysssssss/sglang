# SPDX-License-Identifier: Apache-2.0
"""VSR restore stage: the boundary between SGLang's request model and the
streaming core.

There is exactly one stage. VideoEdit splits its work across ten because it runs
a text encoder, a scheduler and a multi-step denoising loop, one window at a
time. VSR has none of those: no text encoder, no scheduler, one DiT forward per
tile, and a streaming core that owns its own window loop and writes output
incrementally. Splitting it across stages would mean re-shaping the streaming
loop to fit the stage contract, which is exactly the kind of restructuring that
changes semantics -- and ``requirements.md`` §2 exists to prevent that.

So the stage does what a stage is for: read the request, call the core, report
back. The core itself keeps taking plain geometry/queue/IO arguments and never
sees a ``Req`` (``requirements.md`` §5.2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sglang.multimodal_gen.configs.sample.vsr import WanVRSamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.vsr.geometry import (
    parse_resolution,
    resize_to_long_edge,
)
from sglang.multimodal_gen.runtime.vsr.stream import COLOR_REF_MODES, stream_restore
from sglang.multimodal_gen.runtime.vsr.video_io import probe_video

#: Video containers we are willing to write. Anything else in `output_path` is
#: treated as a directory to put the result in.
_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def _vsr_params(batch: Req) -> WanVRSamplingParams:
    params = batch.sampling_params
    if not isinstance(params, WanVRSamplingParams):
        raise TypeError(
            f"WanVSRPipeline requires WanVRSamplingParams, got {type(params).__name__}"
        )
    return params


def resolve_output_path(params: WanVRSamplingParams) -> Path:
    """Work out where the mp4 goes.

    ``output_path`` is used verbatim when it names a file; otherwise it is a
    directory and ``output_file_name`` (default ``vsr_out.mp4``) names the file
    inside it.
    """
    raw = params.output_path
    if not raw:
        raise ValueError("output_path is required")
    path = Path(raw)
    if path.suffix.lower() in _VIDEO_SUFFIXES:
        return path
    return path / (params.output_file_name or "vsr_out.mp4")


class VSRRestoreStage(PipelineStage):
    """Run the streaming restore for one request."""

    def __init__(self, restorer: Any, config: Any):
        super().__init__()
        self.restorer = restorer
        self.config = config

    def _resolve_geometry(
        self, params: WanVRSamplingParams, src_h: int, src_w: int
    ) -> tuple[int, int]:
        """Per-request geometry wins over the pipeline config's default."""
        target = params.target_resolution
        if target is None and params.long_edge is None:
            target = self.config.target_resolution
        if target:
            return parse_resolution(target)
        long_edge = params.long_edge or self.config.long_edge
        return resize_to_long_edge(src_h, src_w, long_edge)

    def _apply_tiling(self, params: WanVRSamplingParams) -> None:
        """Push per-request tiling onto the restorer.

        Tiling has to match training, so the pipeline config supplies the
        defaults. A request may still override them, and since these are plain
        integer attributes rather than anything baked into the weights, applying
        them costs nothing and needs no rebuild.
        """
        restorer = self.restorer
        for attr, value, default in (
            ("tile_t", params.tile_t, self.config.tile_t),
            ("tile_h", params.tile_h, self.config.tile_h),
            ("tile_w", params.tile_w, self.config.tile_w),
            ("t_overlap", params.temporal_overlap, self.config.temporal_overlap),
            ("s_overlap", params.spatial_overlap, self.config.spatial_overlap),
        ):
            setattr(restorer, attr, default if value is None else value)

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _vsr_params(batch)

        input_path = params.video_input_path
        if not input_path:
            raise ValueError("video_input_path is required")
        output_path = resolve_output_path(params)

        fps, total_frames, src_h, src_w = probe_video(input_path)
        target_h, target_w = self._resolve_geometry(params, src_h, src_w)
        self._apply_tiling(params)
        import torch

        precision = params.dtype or self.config.precision
        precision = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(
            precision, precision
        )
        dtype = getattr(torch, precision, None)
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"unsupported VSR dtype: {precision!r}")
        if self.restorer.dtype != dtype:
            raise ValueError(
                "VSR request dtype must match the loaded model precision; "
                "set pipeline_config.precision when constructing the pipeline"
            )

        color_ref = params.color_ref or self.config.color_ref
        if color_ref not in COLOR_REF_MODES:
            raise ValueError(
                f"color_ref must be one of {COLOR_REF_MODES}, got {color_ref!r}"
            )

        params.runtime_target_h = target_h
        params.runtime_target_w = target_w

        written = stream_restore(
            self.restorer,
            input_path,
            output_path,
            target_h=target_h,
            target_w=target_w,
            fps=fps,
            total_frames=total_frames,
            color_ref=color_ref,
            color_samples=(
                params.color_ref_samples
                if params.color_ref_samples is not None
                else self.config.color_ref_samples
            ),
            crf=params.crf if params.crf is not None else self.config.crf,
            read_queue=(
                params.read_queue
                if params.read_queue is not None
                else self.config.read_queue
            ),
            write_queue=(
                params.write_queue
                if params.write_queue is not None
                else self.config.write_queue
            ),
            save_tiles_dir=params.save_tiles_dir,
        )

        params.runtime_frames_written = written
        return batch
