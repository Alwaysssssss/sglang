# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.vividvr import VividVRPipelineConfig
from sglang.multimodal_gen.configs.sample.vividvr import VividVRSamplingParams

logger = logging.getLogger(__name__)


def resolve_prompt_file_path(
    params: VividVRSamplingParams,
    pipeline_config: VividVRPipelineConfig,
) -> str:
    prompt_path = (
        params.prompt_file_path
        or params.prompt_path
        or pipeline_config.default_prompt_file_path
    )
    if prompt_path is None:
        raise ValueError("VividVR prompt file path is not configured")
    return str(Path(prompt_path).expanduser())


def read_prompt_file(prompt_file_path: str) -> str:
    path = Path(prompt_file_path).expanduser()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {path}")
    return text


def compose_positive_prompt(
    prompt_text: str,
    pipeline_config: VividVRPipelineConfig,
) -> str:
    suffix = pipeline_config.default_positive_prompt_suffix.strip()
    prompt = prompt_text.strip()
    if not suffix:
        return prompt
    if prompt.endswith(suffix):
        return prompt
    return f"{prompt} {suffix}"


def resolve_negative_prompt(
    params: VividVRSamplingParams,
    pipeline_config: VividVRPipelineConfig,
) -> str:
    if params.negative_prompt is not None:
        return params.negative_prompt
    return pipeline_config.default_negative_prompt


def _load_control_video_frames_cv2(video_path: str) -> tuple[list[Image.Image], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames: list[Image.Image] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames, float(fps)


def _load_control_video_frames_decord(video_path: str) -> tuple[list[Image.Image], float]:
    import decord

    # Match the original Vivid-VR control-video decode path so the model sees
    # the same RGB frames as the upstream implementation.
    video_reader = decord.VideoReader(uri=video_path, num_threads=1)
    total_frames = len(video_reader)
    if total_frames <= 0:
        raise RuntimeError(f"No frames found in video file: {video_path}")

    batch = video_reader.get_batch(list(range(total_frames)))
    arrays = batch.asnumpy() if hasattr(batch, "asnumpy") else np.asarray(batch)
    frames = [Image.fromarray(frame) for frame in arrays]
    fps = video_reader.get_avg_fps() or 24.0
    return frames, float(fps)


def load_control_video_frames(video_path: str) -> tuple[list[Image.Image], float]:
    try:
        return _load_control_video_frames_decord(video_path)
    except Exception as exc:
        logger.warning(
            "VividVR decord control-video decode failed for %s; falling back to OpenCV: %s",
            video_path,
            exc,
        )
        return _load_control_video_frames_cv2(video_path)


def load_control_video(video_path: str) -> dict[str, object]:
    frames, fps = load_control_video_frames(video_path)
    if not frames:
        raise ValueError(f"No frames found in control video: {video_path}")

    arrays = [
        np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0 for frame in frames
    ]
    video = torch.from_numpy(np.stack(arrays, axis=0)).permute(0, 3, 1, 2).contiguous()

    # Preserve the unpadded reference frames. Phase C color alignment depends on
    # using the original control video, not the repeated tail frames.
    reference_video = video
    original_num_frames = int(video.shape[0])
    num_padding_frames = 0
    if (original_num_frames - 1) % 8 != 0:
        num_padding_frames = 8 - (original_num_frames - 1) % 8
        padding = video[-1:].repeat(num_padding_frames, 1, 1, 1)
        video = torch.cat([video, padding], dim=0)

    return {
        "video": video,
        "reference_video": reference_video,
        "fps": float(fps),
        "original_height": int(video.shape[-2]),
        "original_width": int(video.shape[-1]),
        "original_num_frames": original_num_frames,
        "num_padding_frames": num_padding_frames,
    }
