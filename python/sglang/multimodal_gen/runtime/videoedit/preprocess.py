# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sglang.multimodal_gen.runtime.videoedit.mask_io import (
    load_mask_frames,
    probe_mask_frame_count,
)

logger = logging.getLogger(__name__)


def load_video_frames(
    video_path: str, num_frames: int | None = None
) -> tuple[list[Image.Image], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames: list[Image.Image] = []
    while True:
        if num_frames is not None and len(frames) >= num_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames, fps


def probe_video_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if count <= 0:
            count = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                count += 1
    finally:
        cap.release()
    if count <= 0:
        raise RuntimeError(f"No frames found in video file: {video_path}")
    return count


def resolve_videoedit_num_frames(
    requested_num_frames: int,
    video_input_path: str,
    mask_input_path: str,
) -> int:
    if requested_num_frames != -1:
        return requested_num_frames

    video_frames = probe_video_frame_count(video_input_path)
    mask_frames = probe_mask_frame_count(mask_input_path)
    resolved = min(video_frames, mask_frames)
    if resolved <= 0:
        raise RuntimeError(
            "Could not resolve full-video frame count: "
            f"video={video_frames}, mask={mask_frames}"
        )
    if video_frames != mask_frames:
        logger.warning(
            "VideoEdit num_frames=-1 resolved to %s using min(video=%s, mask=%s)",
            resolved,
            video_frames,
            mask_frames,
        )
    return resolved


def get_aligned_size(h: int, w: int, align: int = 16) -> tuple[int, int]:
    return ((h + align - 1) // align) * align, ((w + align - 1) // align) * align


def resize_frames(frames: list[Image.Image], target_h: int, target_w: int) -> list[Image.Image]:
    return [
        Image.fromarray(cv2.resize(np.array(frame), (target_w, target_h)))
        for frame in frames
    ]


def _dilate_single(binary: np.ndarray, dilate_px: int) -> np.ndarray:
    if dilate_px <= 0:
        return binary
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
    )
    return cv2.dilate(binary, kernel, iterations=1)


def _scale_single(binary: np.ndarray, scale: float) -> np.ndarray:
    h, w = binary.shape
    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or scale == 1.0:
        return binary
    cx, cy = float(xs.mean()), float(ys.mean())
    matrix = np.array(
        [[scale, 0, cx * (1 - scale)], [0, scale, cy * (1 - scale)]],
        dtype=np.float64,
    )
    return cv2.warpAffine(
        binary,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def expand_mask_frames(
    mask_frames: list[Image.Image],
    dilate_px: int = 15,
    scale: float = 1.2,
    threshold: float = 0.5,
) -> list[Image.Image]:
    result: list[Image.Image] = []
    for i, mask in enumerate(mask_frames):
        if i == 0:
            result.append(Image.new("L", mask.size, 0))
            continue
        gray = np.array(mask.convert("L")).astype(np.float32) / 255.0
        binary = ((gray > threshold) * 255).astype(np.uint8)
        expanded = _scale_single(_dilate_single(binary, dilate_px), scale)
        result.append(Image.fromarray(expanded, mode="L"))
    return result


def get_mask_bbox(
    mask_frames: list[Image.Image], padding: int = 0
) -> tuple[int, int, int, int] | None:
    all_x_min = all_y_min = float("inf")
    all_x_max = all_y_max = 0
    height = width = None
    for mask in mask_frames:
        mask_np = np.array(mask.convert("L"))
        if height is None:
            height, width = mask_np.shape
        ys, xs = np.where(mask_np > 10)
        if len(ys):
            all_y_min = min(all_y_min, int(ys.min()))
            all_y_max = max(all_y_max, int(ys.max()))
            all_x_min = min(all_x_min, int(xs.min()))
            all_x_max = max(all_x_max, int(xs.max()))
    if all_x_min == float("inf"):
        return None

    crop_w = all_x_max - all_x_min
    crop_h = all_y_max - all_y_min
    cx = (all_x_min + all_x_max) / 2.0
    cy = (all_y_min + all_y_max) / 2.0
    target_w = crop_w + 2 * padding
    target_h = crop_h + 2 * padding
    x_min = int(round(cx - target_w / 2))
    x_max = int(round(cx + target_w / 2))
    y_min = int(round(cy - target_h / 2))
    y_max = int(round(cy + target_h / 2))
    if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
        raise ValueError(
            f"Expanded mask bbox is out of bounds: {(x_min, y_min, x_max, y_max)} "
            f"for frame size {(width, height)}"
        )
    return x_min, y_min, x_max, y_max


def expand_bbox_for_small(
    bbox: tuple[int, int, int, int], height: int, width: int
) -> tuple[int, int, int, int]:
    x_min, y_min, x_max, y_max = bbox
    crop_w = x_max - x_min
    crop_h = y_max - y_min
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    short_side = max(1, min(crop_w, crop_h))
    scale = 480 / short_side
    new_w = min(crop_w * scale, width)
    new_h = min(crop_h * scale, height)
    x_min = int(cx - new_w / 2)
    x_max = int(cx + new_w / 2)
    y_min = int(cy - new_h / 2)
    y_max = int(cy + new_h / 2)
    if x_min < 0:
        x_max -= x_min
        x_min = 0
    if y_min < 0:
        y_max -= y_min
        y_min = 0
    if x_max > width:
        shift = x_max - width
        x_min -= shift
        x_max = width
    if y_max > height:
        shift = y_max - height
        y_min -= shift
        y_max = height
    return max(0, x_min), max(0, y_min), min(width, x_max), min(height, y_max)


def crop_frames(
    frames: list[Image.Image], bbox: tuple[int, int, int, int]
) -> list[Image.Image]:
    x_min, y_min, x_max, y_max = bbox
    return [Image.fromarray(np.array(frame)[y_min:y_max, x_min:x_max]) for frame in frames]


def create_masked_video(
    video_frames: list[Image.Image],
    mask_frames: list[Image.Image],
    binarize_threshold: int = 128,
) -> list[Image.Image]:
    masked: list[Image.Image] = []
    for i, (frame, mask) in enumerate(zip(video_frames, mask_frames, strict=True)):
        if i == 0:
            masked.append(frame.copy())
            continue
        frame_np = np.array(frame)
        mask_np = (np.array(mask.convert("L")) > binarize_threshold).astype(np.float32)
        masked.append(Image.fromarray((frame_np * (1 - mask_np[:, :, None])).astype(np.uint8)))
    return masked


def create_mask_video(mask_frames: list[Image.Image]) -> list[Image.Image]:
    processed: list[Image.Image] = []
    for i, mask in enumerate(mask_frames):
        processed.append(Image.new("L", mask.size, 0) if i == 0 else mask.convert("L"))
    return processed


def frames_to_tensor(frames: list[Image.Image], normalize: bool = True) -> torch.Tensor:
    tensors = []
    for frame in frames:
        tensor = torch.from_numpy(np.array(frame)).float()
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(-1)
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor / 127.5 - 1.0 if normalize else tensor / 255.0
        tensors.append(tensor)
    return torch.stack(tensors)


def prepare_global_inputs(
    input_video: str,
    mask_video: str,
    num_frames: int | None = None,
    reference_image: str | None = None,
    bbox_padding: int = 0,
    dilate_px: int = 15,
    mask_scale: float = 1.2,
    align: int = 16,
    debug_dir: str | None = None,
) -> dict:
    original_frames, fps = load_video_frames(input_video, num_frames)
    if not original_frames:
        raise RuntimeError("No frames loaded from input video")
    raw_mask_frames = load_mask_frames(
        mask_video,
        num_frames=num_frames,
        target_size=original_frames[0].size,
    )
    n = min(len(original_frames), len(raw_mask_frames))
    if n == 0:
        raise RuntimeError("No frames loaded from input or mask video")
    original_frames = original_frames[:n]
    raw_mask_frames = raw_mask_frames[:n]
    if reference_image:
        with Image.open(reference_image) as image:
            reference_frame = image.convert("RGB").resize(original_frames[0].size)
        reference_mask = Image.new("L", original_frames[0].size, 255)
        original_frames = [reference_frame] + original_frames
        raw_mask_frames = [reference_mask] + raw_mask_frames
        n = len(original_frames)
    dilated_masks = expand_mask_frames(
        raw_mask_frames, dilate_px=dilate_px, scale=mask_scale
    )
    bbox = get_mask_bbox(dilated_masks, padding=bbox_padding)
    if bbox is None:
        raise RuntimeError("No mask region detected")
    x_min, y_min, x_max, y_max = bbox
    height, width = original_frames[0].height, original_frames[0].width
    crop_w, crop_h = x_max - x_min, y_max - y_min
    if (crop_w * crop_h) / float(height * width) < 0.2:
        bbox = expand_bbox_for_small(bbox, height, width)
        x_min, y_min, x_max, y_max = bbox
        crop_w, crop_h = x_max - x_min, y_max - y_min
    cropped_video = crop_frames(original_frames, bbox)
    dilated_cropped_masks = crop_frames(dilated_masks, bbox)
    aligned_h, aligned_w = get_aligned_size(crop_h, crop_w, align)
    resized_video = resize_frames(cropped_video, aligned_h, aligned_w)
    resized_masks = resize_frames(dilated_cropped_masks, aligned_h, aligned_w)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        resized_video[0].save(os.path.join(debug_dir, "global_resized_000.png"))
        resized_masks[0].save(os.path.join(debug_dir, "global_mask_000.png"))
    return {
        "original_frames": original_frames,
        "dilated_cropped_masks": dilated_cropped_masks,
        "resized_video": resized_video,
        "resized_masks": resized_masks,
        "bbox": bbox,
        "crop_h": crop_h,
        "crop_w": crop_w,
        "aligned_h": aligned_h,
        "aligned_w": aligned_w,
        "num_frames": n,
        "fps": fps,
    }


def prepare_window_inputs(
    window_video: list[Image.Image],
    window_masks: list[Image.Image],
    device: str | torch.device,
    dtype: torch.dtype,
) -> dict:
    if len(window_video) != len(window_masks):
        raise ValueError("Window video and mask length mismatch")
    masked_video = create_masked_video(window_video, window_masks)
    processed_masks = create_mask_video(window_masks)
    masked_video_tensor = frames_to_tensor(masked_video, normalize=True).to(device=device, dtype=dtype)
    mask_video_tensor = frames_to_tensor(processed_masks, normalize=False).to(device=device, dtype=torch.float32)
    first_frame_mask = mask_video_tensor[0:1].repeat(4, 1, 1, 1)
    expanded_masks = torch.cat([first_frame_mask, mask_video_tensor[1:]], dim=0)
    cond_masks = F.interpolate(expanded_masks, scale_factor=1 / 8, mode="nearest-exact")
    cond_masks = (cond_masks < 0.5).float()
    num_mask_frames, _, latent_height, latent_width = cond_masks.shape
    if num_mask_frames % 4 != 0:
        raise ValueError(f"Packed mask frame count must be divisible by 4: {num_mask_frames}")
    cond_masks = cond_masks.view(1, num_mask_frames // 4, 4, latent_height, latent_width)
    cond_masks = cond_masks.transpose(1, 2).contiguous().to(device=device, dtype=dtype)
    return {
        "masked_video_tensor": masked_video_tensor,
        "mask_video_tensor": mask_video_tensor.to(device=device, dtype=dtype),
        "cond_masks": cond_masks,
        "video_tensor": frames_to_tensor(window_video, normalize=True).to(device=device, dtype=dtype),
        "num_frames": len(window_video),
    }
