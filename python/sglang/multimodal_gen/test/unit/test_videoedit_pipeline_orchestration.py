# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from sglang.multimodal_gen.configs.sample.videoedit_wan import (
    WanVideoEditSamplingParams,
)
from sglang.multimodal_gen.runtime.pipelines.wan_videoedit_pipeline import (
    WanVideoEditPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.videoedit_wan import (
    VideoEditConditionEncodingStage,
    VideoEditLatentPreparationStage,
    VideoEditWindowValidationStage,
)
from sglang.multimodal_gen.runtime.videoedit.preprocess import build_videoedit_bridge
from sglang.multimodal_gen.runtime.videoedit.windowing import (
    build_videoedit_pass_window_specs,
    plan_videoedit_passes,
)


def _rgb(value: int, size: tuple[int, int] = (16, 16)) -> Image.Image:
    return Image.new("RGB", size, (value, 0, 0))


def _mask(value: int = 255, size: tuple[int, int] = (16, 16)) -> Image.Image:
    return Image.new("L", size, value)


def _red(frame: Image.Image) -> int:
    return int(np.asarray(frame)[0, 0, 0])


class _RecordingExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[list[int], list[int]]] = []

    def execute_with_profiling(self, stages, batch, server_args) -> None:
        del stages, server_args
        params = batch.sampling_params
        self.calls.append(
            (
                [_red(frame) for frame in params.runtime_window_frames],
                [int(np.asarray(mask).sum()) for mask in params.runtime_window_masks],
            )
        )
        params.runtime_window_output_frames = [
            _rgb((_red(frame) + 20) % 256)
            for frame in params.runtime_window_frames
        ]
        params.runtime_effective_num_inference_steps = 1


def test_pipeline_runs_long_bridge_short_and_commits_exact_global_indices():
    params = WanVideoEditSamplingParams(
        num_frames=9,
        infer_len=5,
        overlap=1,
        ref_frame_idx=4,
        bridge_overlap=5,
    )
    params.runtime_num_input_frames = 9
    params.runtime_resized_frames = [_rgb(index) for index in range(9)]
    params.runtime_resized_masks = [_mask() for _ in range(9)]
    params.runtime_frame_provider = None
    params.runtime_window_materialize_metadata = []

    executor = _RecordingExecutor()
    pipeline = object.__new__(WanVideoEditPipeline)
    pipeline.executor = executor
    pipeline._stages = []
    batch = SimpleNamespace(sampling_params=params)
    sequence_plan = plan_videoedit_passes(9, 4, 5)
    generated_by_index: dict[int, Image.Image] = {}

    long_outputs, long_specs = pipeline._run_videoedit_pass(
        params,
        batch,
        SimpleNamespace(),
        sequence_plan.long,
        reference_frame=_rgb(200),
        bridge_frames=None,
        generated_by_index=generated_by_index,
    )
    bridge = build_videoedit_bridge(long_outputs, sequence_plan.bridge_length)
    short_outputs, short_specs = pipeline._run_videoedit_pass(
        params,
        batch,
        SimpleNamespace(),
        sequence_plan.short,
        reference_frame=_rgb(200),
        bridge_frames=bridge,
        generated_by_index=generated_by_index,
    )

    assert len(long_specs) == 2
    assert len(short_specs) == 2
    assert all(frame is not None for frame in long_outputs)
    assert all(frame is not None for frame in short_outputs)
    assert [_red(frame) for frame in bridge] == [28, 27, 26, 25, 24]
    assert executor.calls[1][0][0] == 27
    assert executor.calls[1][1][0] == 0
    assert executor.calls[2][0] == [28, 27, 26, 25, 24]
    assert executor.calls[2][1] == [0, 0, 0, 0, 0]
    assert executor.calls[3][0] == [44, 3, 2, 1, 0]
    assert executor.calls[3][1][0] == 0
    assert set(generated_by_index) == set(range(9))
    assert [_red(generated_by_index[index]) for index in range(9)] == [
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
    ]


def test_pipeline_duplicate_global_commit_fails_loudly():
    plan = plan_videoedit_passes(5, 0, 5).long
    spec = build_videoedit_pass_window_specs(plan.sequence_indices, 5, 1)[0]
    frames = [_rgb(index) for index in range(5)]

    with pytest.raises(RuntimeError, match="global source index 0.*more than once"):
        WanVideoEditPipeline._commit_pass_window(
            plan,
            spec,
            frames,
            [None] * len(plan.sequence_indices),
            {0: _rgb(99)},
        )


def test_pipeline_rejects_missing_or_extra_final_global_indices():
    params = SimpleNamespace(runtime_num_input_frames=3)

    with pytest.raises(RuntimeError, match=r"missing=\[2\], extra=\[4\]"):
        WanVideoEditPipeline._finalize_crop_frames(
            params,
            {0: _rgb(0), 1: _rgb(1), 4: _rgb(4)},
        )


def test_overlap_zero_later_window_still_uses_clean_local_zero_anchor():
    params = WanVideoEditSamplingParams(num_frames=9, infer_len=5, overlap=0)
    params.runtime_window_validated = True
    params.runtime_window_index = 1
    params.runtime_window_frames = [_rgb(index) for index in range(5)]
    params.runtime_window_masks = [_mask() for _ in range(5)]
    sentinel = object()
    params.runtime_raw_video_tensor = sentinel
    params.runtime_video_latents = sentinel
    prepared = {
        "masked_video_tensor": torch.zeros(5, 3, 16, 16),
        "mask_video_tensor": torch.zeros(5, 1, 16, 16),
        "cond_masks": torch.zeros(1, 4, 2, 2, 2),
    }
    stage = object.__new__(VideoEditConditionEncodingStage)
    object.__setattr__(stage, "vae", SimpleNamespace())
    object.__setattr__(
        stage,
        "_encode_video_latents",
        lambda *args, **kwargs: torch.zeros(1, 16, 2, 2, 2),
    )
    batch = SimpleNamespace(sampling_params=params)
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(vae_precision="fp32", vae_tiling=False),
        disable_autocast=True,
    )

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages."
        "model_specific_stages.videoedit_wan.prepare_window_inputs",
        return_value=prepared,
    ) as prepare:
        stage.forward(batch, server_args)

    assert prepare.call_args.kwargs["preserve_first_frame"] is True
    assert params.runtime_raw_video_tensor is sentinel
    assert params.runtime_video_latents is sentinel


def test_stage_accepts_any_structural_4n_plus_one_infer_len():
    params = WanVideoEditSamplingParams(num_frames=9, infer_len=9, overlap=0)
    params.runtime_window_frames = [_rgb(index) for index in range(9)]
    params.runtime_window_masks = [_mask() for _ in range(9)]
    batch = SimpleNamespace(sampling_params=params)

    stage = object.__new__(VideoEditWindowValidationStage)
    stage.forward(batch, SimpleNamespace())

    assert params.runtime_window_validated is True
    assert batch.num_frames == 9


def test_each_window_reseeds_identical_cpu_float32_noise():
    params = WanVideoEditSamplingParams(num_frames=5, infer_len=5, overlap=0, seed=17)
    params.runtime_cond_latents = torch.zeros(1, 16, 2, 2, 2)
    batch = SimpleNamespace(sampling_params=params)
    stage = object.__new__(VideoEditLatentPreparationStage)

    stage.forward(batch, SimpleNamespace())
    first = params.runtime_noise.clone()
    stage.forward(batch, SimpleNamespace())

    assert params.runtime_generator.device.type == "cpu"
    assert params.runtime_noise.dtype == torch.float32
    assert torch.equal(first, params.runtime_noise)


def test_final_output_tensor_metadata_and_batch_frames_match_source(tmp_path):
    params = WanVideoEditSamplingParams(num_frames=3, infer_len=5, overlap=1)
    params.runtime_num_input_frames = 3
    params.runtime_frame_provider = None
    params.runtime_bbox = (0, 0, 16, 16)
    params.runtime_crop_h = 16
    params.runtime_crop_w = 16
    params.runtime_aligned_h = 16
    params.runtime_aligned_w = 16
    params.runtime_fps = 12.5
    params.runtime_window_specs = []
    params.runtime_window_materialize_metadata = []
    params.enable_paste_back = False
    params.save_crop_only = False
    output_path = tmp_path / "result.mp4"
    batch = SimpleNamespace(
        output_file_path=lambda: str(output_path),
        fps=12.5,
        num_frames=5,
    )
    generated = {index: _rgb(index) for index in range(3)}
    pipeline = object.__new__(WanVideoEditPipeline)

    output_frames = pipeline._finalize_videoedit_output(
        params,
        batch,
        generated,
        [],
    )
    pipeline._set_final_batch_output(batch, params, output_frames)

    metadata = json.loads((tmp_path / "result.videoedit.json").read_text())
    assert len(output_frames) == 3
    assert batch.output.shape == (1, 3, 3, 16, 16)
    assert batch.num_frames == 3
    assert batch.fps == 12.5
    assert metadata["num_output_frames"] == 3
    assert metadata["fps"] == 12.5


def test_crop_sidecar_matches_original_videoedit_writer(tmp_path):
    params = WanVideoEditSamplingParams(num_frames=1, infer_len=1, overlap=0)
    params.save_crop_only = True
    params.drop_reference_frame = False
    params.video_input_path = "input.mp4"
    params.runtime_crop_h = 16
    params.runtime_crop_w = 16
    params.runtime_fps = 50.0
    pipeline = object.__new__(WanVideoEditPipeline)
    frames = [_rgb(10)]

    with (
        patch(
            "sglang.multimodal_gen.runtime.pipelines.wan_videoedit_pipeline._is_output_rank",
            return_value=True,
        ),
        patch(
            "sglang.multimodal_gen.runtime.pipelines.wan_videoedit_pipeline.resize_frames",
            return_value=frames,
        ),
        patch(
            "sglang.multimodal_gen.runtime.pipelines.wan_videoedit_pipeline."
            "save_video_frames_like_reference"
        ) as save,
    ):
        pipeline._save_crop_sidecar(
            params,
            frames,
            str(tmp_path / "result.mp4"),
        )

    assert save.call_args.kwargs["bit_rate"] == 10_000_000
    assert save.call_args.kwargs["copy_color_metadata"] is False
