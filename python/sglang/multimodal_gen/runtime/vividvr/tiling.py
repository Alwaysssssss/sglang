# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import itertools
import math
from typing import Optional

import torch
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.cogvideo.pipeline_cogvideox import (
    get_resize_crop_region_for_grid,
)


def prepare_rotary_positional_embeddings(
    latent_height: int,
    latent_width: int,
    num_frames: int,
    patch_size: int = 2,
    patch_size_t: Optional[int] = None,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    sample_height: int = 60,
    sample_width: int = 90,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid_height = latent_height // patch_size
    grid_width = latent_width // patch_size

    if patch_size_t is None:
        base_size_width = sample_width // patch_size
        base_size_height = sample_height // patch_size
        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width),
            base_size_width,
            base_size_height,
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
        )
    else:
        max_size_width = 300 // patch_size
        max_size_height = 300 // patch_size
        base_num_frames = (num_frames + patch_size_t - 1) // patch_size_t
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(max_size_height, max_size_width),
        )

    return freqs_cos.to(device=device), freqs_sin.to(device=device)


def prepare_tiling_infos_generator(
    latents: torch.Tensor,
    enable_spatial_tiling: bool = False,
    enable_temporal_tiling: bool = False,
    tile_size: int = 128,
    tile_stride: int = 64,
    temporal_tile_size: int = 31,
    temporal_tile_stride: int = 15,
):
    if not enable_spatial_tiling and not enable_temporal_tiling:
        yield (
            [
                slice(None),
                slice(None),
                slice(None),
                slice(None),
                slice(None),
            ],
            torch.ones_like(latents),
        )
        return

    batch_size, num_frames, num_channels, height, width = latents.shape

    if not enable_spatial_tiling:
        tile_size = max(height, width)
    if not enable_temporal_tiling:
        temporal_tile_size = num_frames

    def create_start_indices(size: int, max_tile_size: int, stride: int):
        resolved_tile_size = max_tile_size
        resolved_stride = stride
        if size <= resolved_tile_size:
            resolved_stride = resolved_tile_size
        else:
            num_tiles = (size - resolved_tile_size) // resolved_stride + 1
            if (size - resolved_tile_size) % resolved_stride != 0:
                num_tiles += 1
            resolved_stride = math.ceil((size - resolved_tile_size) / (num_tiles - 1))
        starts = list(
            range(0, max(1, size - resolved_tile_size + 1), resolved_stride)
        )
        if size >= resolved_tile_size and (size - resolved_tile_size) % resolved_stride != 0:
            starts.append(size - resolved_tile_size)
        return starts, resolved_tile_size, resolved_stride

    ti_list, t_tile_size, t_tile_stride = create_start_indices(
        num_frames,
        temporal_tile_size,
        temporal_tile_stride,
    )
    hi_list, h_tile_size, h_tile_stride = create_start_indices(
        height,
        tile_size,
        tile_stride,
    )
    wi_list, w_tile_size, w_tile_stride = create_start_indices(
        width,
        tile_size,
        tile_stride,
    )

    def compute_valid_weights_range(
        start: int,
        end: int,
        total_size: int,
        max_tile_size: int,
        stride: int,
    ) -> slice:
        float_padding = (max_tile_size - stride) / 2
        valid_end = max_tile_size - math.floor(float_padding) if end < total_size else max_tile_size
        valid_start = math.ceil(float_padding) if start > 0 else 0
        remainder = start % stride
        if remainder > 0:
            valid_start = max_tile_size - (math.floor(float_padding) + remainder)
        return slice(valid_start, valid_end)

    for ti, hi, wi in itertools.product(ti_list, hi_list, wi_list):
        ti_end = min(ti + t_tile_size, num_frames)
        hi_end = min(hi + h_tile_size, height)
        wi_end = min(wi + w_tile_size, width)
        tile_slice = [
            slice(None),
            slice(ti, ti_end),
            slice(None),
            slice(hi, hi_end),
            slice(wi, wi_end),
        ]

        t_valid_slice = compute_valid_weights_range(
            ti,
            ti_end,
            num_frames,
            t_tile_size,
            t_tile_stride,
        )
        h_valid_slice = compute_valid_weights_range(
            hi,
            hi_end,
            height,
            h_tile_size,
            h_tile_stride,
        )
        w_valid_slice = compute_valid_weights_range(
            wi,
            wi_end,
            width,
            w_tile_size,
            w_tile_stride,
        )

        weights = torch.zeros((1, ti_end - ti, 1, hi_end - hi, wi_end - wi))
        weights[:, t_valid_slice, :, h_valid_slice, w_valid_slice] = 1
        yield tile_slice, weights.repeat(batch_size, 1, num_channels, 1, 1).to(
            latents.device
        )
