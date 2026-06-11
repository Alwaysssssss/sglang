# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import MutableMapping

import torch


@dataclass(frozen=True)
class VividVRTemporalClipSpec:
    clip_index: int
    start_frame: int
    end_frame: int
    original_num_frames: int
    padded_num_frames: int
    num_padding_frames: int
    trim_front_frames: int
    trim_back_frames: int


@dataclass(frozen=True)
class VividVRTemporalWindowPlan:
    num_frames: int
    num_clips: int
    num_temporal_process_frames: int
    num_temporal_overlapped_frames: int
    temporal_frame_stride: int
    clip_specs: list[VividVRTemporalClipSpec]


@dataclass(frozen=True)
class VividVRTemporalLatentMergePlan:
    non_first_frame_latents_start_index: int
    num_temporal_overlap_latents: int
    temporal_latent_stride: int
    clip_id_to_latent_id_map: dict[int, tuple[int, int]]
    valid_latent_id_to_clip_id_map: dict[int, int]


def build_vividvr_temporal_window_plan(
    num_frames: int,
    num_temporal_process_frames: int,
) -> VividVRTemporalWindowPlan:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if num_temporal_process_frames <= 0:
        raise ValueError(
            "num_temporal_process_frames must be positive, "
            f"got {num_temporal_process_frames}"
        )
    if (num_temporal_process_frames - 1) % 8 != 0:
        raise ValueError(
            "num_temporal_process_frames must satisfy "
            "(num_temporal_process_frames - 1) % 8 == 0"
        )

    num_temporal_overlapped_frames = (num_temporal_process_frames - 1) // 2 + 1
    temporal_frame_stride = num_temporal_process_frames - num_temporal_overlapped_frames

    num_clips = (num_frames - num_temporal_process_frames) // temporal_frame_stride + 1
    if (num_clips - 1) * temporal_frame_stride + num_temporal_process_frames < num_frames:
        num_clips += 1
    num_clips = max(1, num_clips)

    clip_specs: list[VividVRTemporalClipSpec] = []
    for clip_index in range(num_clips):
        start_frame = clip_index * temporal_frame_stride
        end_frame = min(start_frame + num_temporal_process_frames, num_frames)
        original_num_frames = end_frame - start_frame
        num_padding_frames = 0
        if (original_num_frames - 1) % 8 != 0:
            num_padding_frames = 8 - (original_num_frames - 1) % 8
        clip_specs.append(
            VividVRTemporalClipSpec(
                clip_index=clip_index,
                start_frame=start_frame,
                end_frame=end_frame,
                original_num_frames=original_num_frames,
                padded_num_frames=original_num_frames + num_padding_frames,
                num_padding_frames=num_padding_frames,
                trim_front_frames=(num_temporal_overlapped_frames + 1) // 2
                if clip_index > 0
                else 0,
                trim_back_frames=num_temporal_overlapped_frames // 2
                if clip_index < num_clips - 1
                else 0,
            )
        )

    return VividVRTemporalWindowPlan(
        num_frames=num_frames,
        num_clips=num_clips,
        num_temporal_process_frames=num_temporal_process_frames,
        num_temporal_overlapped_frames=num_temporal_overlapped_frames,
        temporal_frame_stride=temporal_frame_stride,
        clip_specs=clip_specs,
    )


def build_vividvr_temporal_latent_merge_plan(
    clip_latent_lengths: list[int],
    *,
    num_temporal_process_frames: int,
    vae_scale_factor_temporal: int,
) -> VividVRTemporalLatentMergePlan:
    if not clip_latent_lengths:
        raise ValueError("clip_latent_lengths must not be empty")
    if vae_scale_factor_temporal <= 0:
        raise ValueError(
            "vae_scale_factor_temporal must be positive, "
            f"got {vae_scale_factor_temporal}"
        )

    num_temporal_overlapped_frames = (num_temporal_process_frames - 1) // 2 + 1
    num_temporal_overlap_latents = (
        num_temporal_overlapped_frames - 1
    ) // vae_scale_factor_temporal
    temporal_latent_stride = (
        (num_temporal_process_frames - 1) // vae_scale_factor_temporal
    ) - num_temporal_overlap_latents
    non_first_frame_latents_start_index = 2

    clip_id_to_latent_id_map: dict[int, tuple[int, int]] = {}
    valid_latent_id_to_clip_id_map: dict[int, int] = {}
    for clip_index, temporal_latent_length in enumerate(clip_latent_lengths):
        if temporal_latent_length < non_first_frame_latents_start_index:
            raise ValueError(
                "clip latent length must include the special first-frame latents, "
                f"got {temporal_latent_length}"
            )
        latent_id_begin = temporal_latent_stride * clip_index + 1
        latent_id_end = latent_id_begin + (
            temporal_latent_length - non_first_frame_latents_start_index
        )
        clip_id_to_latent_id_map[clip_index] = (latent_id_begin, latent_id_end)

        num_valid_latents = (latent_id_end - latent_id_begin) - num_temporal_overlap_latents
        if clip_index == 0 or clip_index == len(clip_latent_lengths) - 1:
            num_valid_latents = (latent_id_end - latent_id_begin) - (
                num_temporal_overlap_latents // 2
            )
        valid_latent_begin = latent_id_begin + (
            num_temporal_overlap_latents // 2 * int(clip_index > 0)
        )
        valid_latent_end = valid_latent_begin + num_valid_latents
        for latent_id in range(valid_latent_begin, valid_latent_end):
            valid_latent_id_to_clip_id_map[latent_id] = clip_index

    return VividVRTemporalLatentMergePlan(
        non_first_frame_latents_start_index=non_first_frame_latents_start_index,
        num_temporal_overlap_latents=num_temporal_overlap_latents,
        temporal_latent_stride=temporal_latent_stride,
        clip_id_to_latent_id_map=clip_id_to_latent_id_map,
        valid_latent_id_to_clip_id_map=valid_latent_id_to_clip_id_map,
    )


def merge_vividvr_temporal_latent_states(
    clip_states: list[MutableMapping[str, torch.Tensor | None]],
    merge_plan: VividVRTemporalLatentMergePlan,
) -> None:
    for clip_index, clip_state in enumerate(clip_states):
        latents = clip_state["latents"]
        old_pred_original_sample = clip_state["old_pred_original_sample"]
        if latents is None or old_pred_original_sample is None:
            raise ValueError("All clip states must provide latents and old_pred_original_sample")

        latent_id_range = merge_plan.clip_id_to_latent_id_map[clip_index]
        latent_id_offset = (
            latent_id_range[0] - merge_plan.non_first_frame_latents_start_index
        )
        for latent_id in range(*latent_id_range):
            target_clip_index = merge_plan.valid_latent_id_to_clip_id_map[latent_id]
            if target_clip_index == clip_index:
                continue

            target_clip_state = clip_states[target_clip_index]
            target_latents = target_clip_state["latents"]
            target_old_pred_original_sample = target_clip_state["old_pred_original_sample"]
            if target_latents is None or target_old_pred_original_sample is None:
                raise ValueError(
                    "Target clip state must provide latents and old_pred_original_sample"
                )

            target_clip_latent_id_offset = (
                merge_plan.clip_id_to_latent_id_map[target_clip_index][0]
                - merge_plan.non_first_frame_latents_start_index
            )
            latents[:, latent_id - latent_id_offset, ...] = target_latents[
                :,
                latent_id - target_clip_latent_id_offset,
                ...,
            ]
            old_pred_original_sample[:, latent_id - latent_id_offset, ...] = (
                target_old_pred_original_sample[
                    :,
                    latent_id - target_clip_latent_id_offset,
                    ...,
                ]
            )

        clip_state["latents"] = latents
        clip_state["old_pred_original_sample"] = old_pred_original_sample


def trim_vividvr_temporal_output_clip(
    video: torch.Tensor,
    clip_spec: VividVRTemporalClipSpec,
) -> torch.Tensor:
    if video.ndim != 4:
        raise ValueError(f"Expected [F, C, H, W] clip video, got shape {tuple(video.shape)}")

    if video.shape[0] % 4 == 0:
        video = video[3:]
    if clip_spec.num_padding_frames > 0:
        video = video[:-clip_spec.num_padding_frames]
    if clip_spec.trim_front_frames > 0:
        video = video[clip_spec.trim_front_frames :]
    if clip_spec.trim_back_frames > 0:
        video = video[: -clip_spec.trim_back_frames]
    return video


def stitch_vividvr_temporal_output_clips(clips: list[torch.Tensor]) -> torch.Tensor:
    if not clips:
        raise ValueError("clips must not be empty")
    return torch.cat(clips, dim=0)
