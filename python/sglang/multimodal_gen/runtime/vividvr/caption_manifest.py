# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class VividVRCaptionTileSpec:
    tile_index: int
    t_start: int
    t_end: int
    h_start: int
    h_end: int
    w_start: int
    w_end: int


@dataclass(frozen=True)
class VividVRCaptionClipSpec:
    clip_index: int
    start_frame: int
    end_frame: int
    original_num_frames: int
    padded_num_frames: int
    tile_count: int
    tiles: list[VividVRCaptionTileSpec]


@dataclass(frozen=True)
class VividVRCaptionManifest:
    version: int
    video_path: str
    fps: float
    num_frames: int
    height: int
    width: int
    num_temporal_process_frames: int
    tile_size: int
    tile_stride: int
    expected_caption_count: int
    clips: list[VividVRCaptionClipSpec]

    def write_json(self, path: str | Path) -> None:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "VividVRCaptionManifest":
        data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        clips = [
            VividVRCaptionClipSpec(
                **{
                    **clip,
                    "tiles": [
                        VividVRCaptionTileSpec(**tile)
                        for tile in clip.get("tiles", [])
                    ],
                }
            )
            for clip in data.get("clips", [])
        ]
        return cls(**{**data, "clips": clips})


def _slice_start_stop(tile_slice: list[slice], dim: int) -> tuple[int, int]:
    value = tile_slice[dim]
    if value.start is None or value.stop is None:
        raise ValueError(f"caption tile slice for dim={dim} is not bounded")
    return int(value.start), int(value.stop)


def build_vividvr_caption_manifest_from_video_info(
    *,
    video_path: str,
    fps: float,
    num_frames: int,
    height: int,
    width: int,
    num_temporal_process_frames: int,
    tile_size: int,
    tile_stride: int,
) -> VividVRCaptionManifest:
    # Keep heavy Vivid-VR runtime imports local so the original caption sidecar
    # environment can import the manifest contract without loading the full
    # sglang inference dependency graph.
    from sglang.multimodal_gen.runtime.vividvr.tiling import (
        prepare_tiling_infos_generator,
    )
    from sglang.multimodal_gen.runtime.vividvr.windowing import (
        build_vividvr_temporal_window_plan,
    )

    window_plan = build_vividvr_temporal_window_plan(
        num_frames,
        num_temporal_process_frames,
    )
    clips: list[VividVRCaptionClipSpec] = []
    # Caption sidecars are consumed as one caption per temporal clip. Spatial
    # tile metadata is still recorded so bridge debugging stays aligned with
    # the accepted Phase D tiling/windowing plan.
    expected_caption_count = len(window_plan.clip_specs)

    for temporal_clip in window_plan.clip_specs:
        latents = torch.empty(
            (1, temporal_clip.padded_num_frames, 3, height, width),
            device="meta",
        )
        tiles: list[VividVRCaptionTileSpec] = []
        for tile_index, (tile_slice, _weights) in enumerate(
            prepare_tiling_infos_generator(
                latents,
                enable_spatial_tiling=True,
                enable_temporal_tiling=False,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        ):
            t_start, t_end = _slice_start_stop(tile_slice, 1)
            h_start, h_end = _slice_start_stop(tile_slice, 3)
            w_start, w_end = _slice_start_stop(tile_slice, 4)
            tiles.append(
                VividVRCaptionTileSpec(
                    tile_index=tile_index,
                    t_start=t_start,
                    t_end=t_end,
                    h_start=h_start,
                    h_end=h_end,
                    w_start=w_start,
                    w_end=w_end,
                )
            )

        clips.append(
            VividVRCaptionClipSpec(
                clip_index=temporal_clip.clip_index,
                start_frame=temporal_clip.start_frame,
                end_frame=temporal_clip.end_frame,
                original_num_frames=temporal_clip.original_num_frames,
                padded_num_frames=temporal_clip.padded_num_frames,
                tile_count=len(tiles),
                tiles=tiles,
            )
        )

    return VividVRCaptionManifest(
        version=1,
        video_path=str(video_path),
        fps=float(fps),
        num_frames=int(num_frames),
        height=int(height),
        width=int(width),
        num_temporal_process_frames=int(num_temporal_process_frames),
        tile_size=int(tile_size),
        tile_stride=int(tile_stride),
        expected_caption_count=expected_caption_count,
        clips=clips,
    )


def build_vividvr_caption_manifest_for_video_path(
    *,
    video_path: str,
    num_temporal_process_frames: int,
    tile_size: int,
    tile_stride: int,
) -> VividVRCaptionManifest:
    from sglang.multimodal_gen.runtime.vividvr.preprocess import (
        load_control_video_frames,
    )

    frames, fps = load_control_video_frames(video_path)
    if not frames:
        raise ValueError(f"No frames found in video file: {video_path}")
    first_frame = frames[0]
    return build_vividvr_caption_manifest_from_video_info(
        video_path=video_path,
        fps=fps,
        num_frames=len(frames),
        height=int(first_frame.height),
        width=int(first_frame.width),
        num_temporal_process_frames=num_temporal_process_frames,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )
