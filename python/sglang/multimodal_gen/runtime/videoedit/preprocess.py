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
from sglang.multimodal_gen.runtime.videoedit.frame_cache import (
    get_cached_video_frames,
)

logger = logging.getLogger(__name__)


def load_video_frames(
    video_path: str, num_frames: int | None = None
) -> tuple[list[Image.Image], float]:
    cached = get_cached_video_frames(video_path)
    if cached is not None:
        frames = list(cached.frames)
        if num_frames is not None:
            frames = frames[:num_frames]
        return [Image.fromarray(frame) for frame in frames], cached.fps

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


def probe_video_frame_size_and_fps(video_path: str) -> tuple[tuple[int, int], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read a frame from video file: {video_path}")
            height, width = frame.shape[:2]
    finally:
        cap.release()
    return (width, height), fps


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
    return [resize_frame(frame, target_h, target_w) for frame in frames]


def resize_frame(frame: Image.Image, target_h: int, target_w: int) -> Image.Image:
    return Image.fromarray(cv2.resize(np.array(frame), (target_w, target_h)))


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
        result.append(
            expand_mask_frame(
                mask,
                dilate_px=dilate_px,
                scale=scale,
                threshold=threshold,
                force_zero=i == 0,
            )
        )
    return result


def expand_mask_frame(
    mask: Image.Image,
    dilate_px: int = 15,
    scale: float = 1.2,
    threshold: float = 0.5,
    force_zero: bool = False,
) -> Image.Image:
    if force_zero:
        return Image.new("L", mask.size, 0)
    gray = np.array(mask.convert("L")).astype(np.float32) / 255.0
    binary = ((gray > threshold) * 255).astype(np.uint8)
    expanded = _scale_single(_dilate_single(binary, dilate_px), scale)
    return Image.fromarray(expanded, mode="L")


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


def expand_bbox(
    bbox: tuple[int, int, int, int],
    height: int,
    width: int,
    scale: float = 0.3,
) -> tuple[int, int, int, int]:
    if scale <= 0:
        return bbox
    x_min, y_min, x_max, y_max = bbox
    crop_w = x_max - x_min
    crop_h = y_max - y_min
    return (
        max(0, int(x_min - crop_w * scale)),
        max(0, int(y_min - crop_h * scale)),
        min(width, int(x_max + crop_w * scale)),
        min(height, int(y_max + crop_h * scale)),
    )


def _finalize_bbox_geometry(
    bbox: tuple[int, int, int, int],
    *,
    height: int,
    width: int,
    align: int,
    bbox_expand_scale: float,
) -> dict[str, int | tuple[int, int, int, int]]:
    bbox = expand_bbox(bbox, height, width, scale=bbox_expand_scale)
    x_min, y_min, x_max, y_max = bbox
    crop_w, crop_h = x_max - x_min, y_max - y_min
    area_ratio = (crop_w * crop_h) / float(height * width)
    short_side = min(crop_w, crop_h)
    if area_ratio < 0.2 and short_side < 480:
        bbox = expand_bbox_for_small(bbox, height, width)
        x_min, y_min, x_max, y_max = bbox
        crop_w, crop_h = x_max - x_min, y_max - y_min
    aligned_h, aligned_w = get_aligned_size(crop_h, crop_w, align)
    return {
        "bbox": bbox,
        "crop_h": crop_h,
        "crop_w": crop_w,
        "aligned_h": aligned_h,
        "aligned_w": aligned_w,
    }


def crop_frames(
    frames: list[Image.Image], bbox: tuple[int, int, int, int]
) -> list[Image.Image]:
    return [crop_frame(frame, bbox) for frame in frames]


def crop_frame(frame: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    x_min, y_min, x_max, y_max = bbox
    return Image.fromarray(np.array(frame)[y_min:y_max, x_min:x_max])


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


def scan_global_bbox(
    input_video: str,
    mask_video: str,
    num_frames: int | None = None,
    reference_image: str | None = None,
    bbox_padding: int = 0,
    dilate_px: int = 0,
    mask_scale: float = 1.0,
    bbox_expand_scale: float = 0.3,
    align: int = 16,
) -> dict:
    from sglang.multimodal_gen.runtime.videoedit.stream_decoder import (
        SequentialMaskDecoder,
    )

    frame_size, fps = probe_video_frame_size_and_fps(input_video)
    width, height = frame_size
    video_frame_count = probe_video_frame_count(input_video)
    mask_frame_count = probe_mask_frame_count(mask_video)
    raw_num_frames = min(
        video_frame_count,
        mask_frame_count,
        num_frames if num_frames is not None else min(video_frame_count, mask_frame_count),
    )
    if raw_num_frames <= 0:
        raise RuntimeError("No frames available for VideoEdit bbox scan")
    global_offset = 1 if reference_image else 0
    all_x_min = all_y_min = float("inf")
    all_x_max = all_y_max = 0
    loaded_raw_frames = 0
    decoder = SequentialMaskDecoder(mask_video, target_size=frame_size)
    try:
        for raw_idx in range(raw_num_frames):
            raw_mask = decoder.read_next()
            if raw_mask is None:
                break
            loaded_raw_frames += 1
            expanded_mask = expand_mask_frame(
                raw_mask,
                dilate_px=dilate_px,
                scale=mask_scale,
                force_zero=(raw_idx + global_offset) == 0,
            )
            mask_np = np.array(expanded_mask.convert("L"))
            ys, xs = np.where(mask_np > 10)
            if len(ys):
                all_y_min = min(all_y_min, int(ys.min()))
                all_y_max = max(all_y_max, int(ys.max()))
                all_x_min = min(all_x_min, int(xs.min()))
                all_x_max = max(all_x_max, int(xs.max()))
    finally:
        decoder.close()
    if loaded_raw_frames <= 0:
        raise RuntimeError("No mask frames loaded during VideoEdit bbox scan")
    effective_num_frames = loaded_raw_frames + global_offset
    if all_x_min == float("inf"):
        raise RuntimeError("No mask region detected")

    crop_w = all_x_max - all_x_min
    crop_h = all_y_max - all_y_min
    cx = (all_x_min + all_x_max) / 2.0
    cy = (all_y_min + all_y_max) / 2.0
    target_w = crop_w + 2 * bbox_padding
    target_h = crop_h + 2 * bbox_padding
    x_min = int(round(cx - target_w / 2))
    x_max = int(round(cx + target_w / 2))
    y_min = int(round(cy - target_h / 2))
    y_max = int(round(cy + target_h / 2))
    if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
        raise ValueError(
            f"Expanded mask bbox is out of bounds: {(x_min, y_min, x_max, y_max)} "
            f"for frame size {(width, height)}"
        )
    bbox = (x_min, y_min, x_max, y_max)

    geometry = _finalize_bbox_geometry(
        bbox,
        height=height,
        width=width,
        align=align,
        bbox_expand_scale=bbox_expand_scale,
    )
    geometry.update(
        {
            "fps": fps,
            "num_frames": effective_num_frames,
            "frame_size": frame_size,
            "frame_width": width,
            "frame_height": height,
        }
    )
    return geometry


def prepare_global_inputs(
    input_video: str,
    mask_video: str,
    num_frames: int | None = None,
    reference_image: str | None = None,
    bbox_padding: int = 0,
    dilate_px: int = 0,
    mask_scale: float = 1.0,
    bbox_expand_scale: float = 0.3,
    align: int = 16,
    debug_dir: str | None = None,
    scanned_geometry: dict | None = None,
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
    height, width = original_frames[0].height, original_frames[0].width
    if scanned_geometry is None:
        bbox = get_mask_bbox(dilated_masks, padding=bbox_padding)
        if bbox is None:
            raise RuntimeError("No mask region detected")
        geometry = _finalize_bbox_geometry(
            bbox,
            height=height,
            width=width,
            align=align,
            bbox_expand_scale=bbox_expand_scale,
        )
    else:
        bbox = tuple(scanned_geometry["bbox"])
        crop_h = int(scanned_geometry.get("crop_h", bbox[3] - bbox[1]))
        crop_w = int(scanned_geometry.get("crop_w", bbox[2] - bbox[0]))
        aligned_h = int(scanned_geometry.get("aligned_h", get_aligned_size(crop_h, crop_w, align)[0]))
        aligned_w = int(scanned_geometry.get("aligned_w", get_aligned_size(crop_h, crop_w, align)[1]))
        geometry = {
            "bbox": bbox,
            "crop_h": crop_h,
            "crop_w": crop_w,
            "aligned_h": aligned_h,
            "aligned_w": aligned_w,
        }
        fps = float(scanned_geometry.get("fps", fps))

    bbox = geometry["bbox"]
    crop_h = int(geometry["crop_h"])
    crop_w = int(geometry["crop_w"])
    aligned_h = int(geometry["aligned_h"])
    aligned_w = int(geometry["aligned_w"])
    cropped_video = crop_frames(original_frames, bbox)
    dilated_cropped_masks = crop_frames(dilated_masks, bbox)
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
    mask_downsample_mode: str = "nearest",
) -> dict:
    if len(window_video) != len(window_masks):
        raise ValueError("Window video and mask length mismatch")
    if mask_downsample_mode not in {"nearest", "nearest-exact"}:
        raise ValueError(
            "mask_downsample_mode must be one of nearest/nearest-exact, "
            f"got {mask_downsample_mode!r}"
        )
    masked_video = create_masked_video(window_video, window_masks)
    processed_masks = create_mask_video(window_masks)
    masked_video_tensor = frames_to_tensor(masked_video, normalize=True).to(device=device, dtype=dtype)
    mask_video_tensor = frames_to_tensor(processed_masks, normalize=False).to(device=device, dtype=torch.float32)
    first_frame_mask = mask_video_tensor[0:1].repeat(4, 1, 1, 1)
    expanded_masks = torch.cat([first_frame_mask, mask_video_tensor[1:]], dim=0)
    cond_masks = F.interpolate(
        expanded_masks, scale_factor=1 / 8, mode=mask_downsample_mode
    )
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
