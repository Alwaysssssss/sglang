# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F
from diffusers.video_processor import VideoProcessor


def _calc_mean_std(feat: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    if feat.ndim != 4:
        raise ValueError(f"Expected [F, C, H, W] video tensor, got shape {tuple(feat.shape)}")
    frames, channels = feat.shape[:2]
    feat_var = feat.reshape(frames, channels, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(frames, channels, 1, 1)
    feat_mean = feat.reshape(frames, channels, -1).mean(dim=2).reshape(frames, channels, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(
    content_feat: torch.Tensor,
    style_feat: torch.Tensor,
) -> torch.Tensor:
    if content_feat.shape != style_feat.shape:
        raise ValueError(
            "Color fix requires generated and reference videos to share the same shape, "
            f"got {tuple(content_feat.shape)} vs {tuple(style_feat.shape)}"
        )
    style_mean, style_std = _calc_mean_std(style_feat)
    content_mean, content_std = _calc_mean_std(content_feat)
    normalized = (content_feat - content_mean) / content_std
    return normalized * style_std + style_mean


def decoded_video_to_frame_tensor(
    decoded_video: torch.Tensor,
    *,
    video_processor: VideoProcessor,
    original_height: int,
    original_width: int,
) -> torch.Tensor:
    resized_video = [
        F.interpolate(
            sample.permute(1, 0, 2, 3),
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False,
        )
        for sample in decoded_video
    ]
    resized_video = torch.stack(resized_video, dim=0).permute(0, 2, 1, 3, 4)
    processed = video_processor.postprocess_video(
        video=resized_video.float(),
        output_type="pt",
    )
    if isinstance(processed, list):
        processed = torch.stack(processed, dim=0)
    return processed[0]


def apply_reference_color_fix(
    output_video: torch.Tensor,
    reference_video: torch.Tensor | None,
) -> torch.Tensor:
    if reference_video is None:
        return output_video

    reference_video = reference_video.to(device=output_video.device, dtype=output_video.dtype)
    if reference_video.shape[-2:] != output_video.shape[-2:]:
        reference_video = F.interpolate(
            reference_video,
            size=output_video.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if reference_video.shape[0] != output_video.shape[0]:
        frame_count = min(reference_video.shape[0], output_video.shape[0])
        reference_video = reference_video[:frame_count]
        output_video = output_video[:frame_count]
    return adaptive_instance_normalization(output_video, reference_video).clamp_(0.0, 1.0)


def run_optional_postprocess_modules(
    output_video: torch.Tensor,
    *,
    reference_video: torch.Tensor | None,
    enabled: bool,
    allow_fallback: bool,
    debug: dict[str, Any] | None,
    processor: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor] | None = None,
) -> torch.Tensor:
    if not enabled or processor is None:
        return output_video

    try:
        return processor(output_video, reference_video)
    except Exception as exc:
        if not allow_fallback:
            raise
        if debug is not None:
            warnings = debug.setdefault("optional_module_warnings", [])
            warnings.append(f"postprocess_module_fallback: {exc}")
        return output_video
