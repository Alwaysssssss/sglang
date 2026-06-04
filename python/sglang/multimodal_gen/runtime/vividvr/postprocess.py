# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch


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
