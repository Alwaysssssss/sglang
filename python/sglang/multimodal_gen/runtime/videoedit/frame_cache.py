# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from os.path import abspath

import numpy as np


@dataclass(frozen=True)
class CachedVideoFrames:
    frames: tuple[np.ndarray, ...]
    fps: float


_VIDEO_FRAME_CACHE: dict[str, CachedVideoFrames] = {}


def cache_video_frames(
    video_path: str,
    frames: list[np.ndarray],
    fps: float,
) -> None:
    _VIDEO_FRAME_CACHE[abspath(video_path)] = CachedVideoFrames(
        frames=tuple(frames),
        fps=float(fps),
    )


def get_cached_video_frames(video_path: str) -> CachedVideoFrames | None:
    return _VIDEO_FRAME_CACHE.get(abspath(video_path))


def clear_cached_video_frames() -> None:
    _VIDEO_FRAME_CACHE.clear()
