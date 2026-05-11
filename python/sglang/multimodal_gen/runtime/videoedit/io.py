# SPDX-License-Identifier: Apache-2.0
import os

import imageio
import numpy as np
from PIL import Image


def save_video_frames(
    frames: list[Image.Image] | list[np.ndarray],
    path: str,
    fps: float,
    quality: int | float | None = None,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arrays = [np.array(frame) if isinstance(frame, Image.Image) else frame for frame in frames]
    imageio.mimsave(
        path,
        arrays,
        fps=fps,
        format="mp4",
        codec="libx264",
        quality=quality if quality is not None else 8,
    )
    return path

