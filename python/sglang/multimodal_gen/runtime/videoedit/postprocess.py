# SPDX-License-Identifier: Apache-2.0
import cv2
import numpy as np
from PIL import Image

from sglang.multimodal_gen.runtime.videoedit.preprocess import resize_frames


def _ensure_2d(mask: np.ndarray) -> np.ndarray:
    return mask[:, :, 0] if mask.ndim == 3 else mask


def _to_float_mask(mask) -> np.ndarray:
    if isinstance(mask, Image.Image):
        arr = np.array(mask.convert("L")).astype(np.float32) / 255.0
    else:
        arr = np.array(mask).astype(np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
    return _ensure_2d(arr)


def _edge_feather_blend(
    orig_crop: np.ndarray,
    gen: np.ndarray,
    mask_bin: np.ndarray,
    feather_px: int = 12,
) -> np.ndarray:
    k = feather_px * 2 + 1
    feather = cv2.GaussianBlur(
        mask_bin.astype(np.float32), (k, k), sigmaX=max(feather_px / 2.0, 0.1)
    )
    feather = np.clip(feather, 0, 1)[:, :, None]
    return np.clip(gen * feather + orig_crop * (1 - feather), 0, 255)


def paste_back(
    original_frames: list[Image.Image],
    generated_frames: list[Image.Image],
    mask_frames: list,
    bbox: tuple[int, int, int, int],
    crop_h: int,
    crop_w: int,
    feather_px: int = 12,
    adain_boundary_dilate: int = 15,
) -> list[Image.Image]:
    del adain_boundary_dilate
    gen_resized = resize_frames(generated_frames, crop_h, crop_w)
    result_frames: list[Image.Image] = []
    for orig, gen, mask in zip(original_frames, gen_resized, mask_frames, strict=False):
        result_frames.append(
            paste_back_frame(
                original_frame=orig,
                generated_frame=gen,
                mask_frame=mask,
                bbox=bbox,
                feather_px=feather_px,
            )
        )
    return result_frames


def paste_back_frame(
    original_frame: Image.Image,
    generated_frame: Image.Image,
    mask_frame,
    bbox: tuple[int, int, int, int],
    feather_px: int = 12,
) -> Image.Image:
    x_min, y_min, _, _ = bbox
    orig_np = np.array(original_frame).astype(np.float32)
    gen_np = np.array(generated_frame).astype(np.float32)
    mask_np = _to_float_mask(mask_frame)
    h_full, w_full = orig_np.shape[:2]
    y_end = min(y_min + gen_np.shape[0], h_full)
    x_end = min(x_min + gen_np.shape[1], w_full)
    h = y_end - y_min
    w = x_end - x_min
    if h <= 0 or w <= 0:
        return original_frame
    gen_np = cv2.resize(gen_np, (w, h))
    mask_np = cv2.resize(mask_np, (w, h))
    mask_bin = (mask_np > 0.5).astype(np.float32)
    orig_crop = orig_np[y_min:y_end, x_min:x_end]
    blended = _edge_feather_blend(orig_crop, gen_np, mask_bin, feather_px=feather_px)
    result_np = orig_np.copy()
    result_np[y_min:y_end, x_min:x_end] = blended
    return Image.fromarray(result_np.astype(np.uint8))
