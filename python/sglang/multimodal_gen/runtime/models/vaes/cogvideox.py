# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import torch
from diffusers.models.autoencoders.autoencoder_kl_cogvideox import (
    AutoencoderKLCogVideoX as DiffusersAutoencoderKLCogVideoX,
)

from sglang.multimodal_gen.configs.models.vaes.cogvideox import CogVideoXVAEConfig


@dataclass(frozen=True)
class CogVideoXSpatialTile:
    global_index: int
    row_index: int
    column_index: int
    latent_top: int
    latent_left: int


@dataclass(frozen=True)
class CogVideoXSpatialTilePlan:
    tiles: tuple[CogVideoXSpatialTile, ...]
    num_rows: int
    num_columns: int
    overlap_height: int
    overlap_width: int
    blend_extent_height: int
    blend_extent_width: int
    row_limit_height: int
    row_limit_width: int


def _build_spatial_tile_plan(
    *,
    latent_height: int,
    latent_width: int,
    tile_latent_min_height: int,
    tile_latent_min_width: int,
    tile_sample_min_height: int,
    tile_sample_min_width: int,
    tile_overlap_factor_height: float,
    tile_overlap_factor_width: float,
) -> CogVideoXSpatialTilePlan:
    overlap_height = int(
        tile_latent_min_height * (1 - tile_overlap_factor_height)
    )
    overlap_width = int(tile_latent_min_width * (1 - tile_overlap_factor_width))
    if overlap_height <= 0 or overlap_width <= 0:
        raise ValueError("CogVideoX VAE tile overlap stride must be positive")
    blend_extent_height = int(tile_sample_min_height * tile_overlap_factor_height)
    blend_extent_width = int(tile_sample_min_width * tile_overlap_factor_width)
    coordinates = [
        (row_index, column_index, top, left)
        for row_index, top in enumerate(range(0, latent_height, overlap_height))
        for column_index, left in enumerate(range(0, latent_width, overlap_width))
    ]
    num_rows = len(range(0, latent_height, overlap_height))
    num_columns = len(range(0, latent_width, overlap_width))
    return CogVideoXSpatialTilePlan(
        tiles=tuple(
            CogVideoXSpatialTile(index, row, column, top, left)
            for index, (row, column, top, left) in enumerate(coordinates)
        ),
        num_rows=num_rows,
        num_columns=num_columns,
        overlap_height=overlap_height,
        overlap_width=overlap_width,
        blend_extent_height=blend_extent_height,
        blend_extent_width=blend_extent_width,
        row_limit_height=tile_sample_min_height - blend_extent_height,
        row_limit_width=tile_sample_min_width - blend_extent_width,
    )


def _assign_spatial_tiles(
    tiles: tuple[CogVideoXSpatialTile, ...], rank: int, world_size: int
) -> tuple[CogVideoXSpatialTile, ...]:
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"invalid SP rank/world size: {rank}/{world_size}")
    return tuple(tile for tile in tiles if tile.global_index % world_size == rank)


def _decode_one_spatial_tile(
    vae, z, tile: CogVideoXSpatialTile
):
    frame_batch_size = vae.num_latent_frames_batch_size
    num_frames = z.shape[2]
    num_batches = max(num_frames // frame_batch_size, 1)
    conv_cache = None
    temporal_parts = []
    for batch_index in range(num_batches):
        remaining_frames = num_frames % frame_batch_size
        start_frame = frame_batch_size * batch_index + (
            0 if batch_index == 0 else remaining_frames
        )
        end_frame = frame_batch_size * (batch_index + 1) + remaining_frames
        decoded = z[
            :,
            :,
            start_frame:end_frame,
            tile.latent_top : tile.latent_top + vae.tile_latent_min_height,
            tile.latent_left : tile.latent_left + vae.tile_latent_min_width,
        ]
        if vae.post_quant_conv is not None:
            decoded = vae.post_quant_conv(decoded)
        decoded, conv_cache = vae.decoder(decoded, conv_cache=conv_cache)
        temporal_parts.append(decoded)
    return torch.cat(temporal_parts, dim=2)


def _merge_spatial_tiles(vae, plan, decoded_tiles):
    rows = [
        [
            decoded_tiles[row * plan.num_columns + column]
            for column in range(plan.num_columns)
        ]
        for row in range(plan.num_rows)
    ]
    result_rows = []
    for row_index, row in enumerate(rows):
        result_row = []
        for column_index, tile in enumerate(row):
            if row_index > 0:
                tile = vae.blend_v(
                    rows[row_index - 1][column_index],
                    tile,
                    plan.blend_extent_height,
                )
            if column_index > 0:
                tile = vae.blend_h(
                    row[column_index - 1], tile, plan.blend_extent_width
                )
            result_row.append(
                tile[
                    :, :, :, : plan.row_limit_height, : plan.row_limit_width
                ]
            )
        result_rows.append(torch.cat(result_row, dim=4))
    return torch.cat(result_rows, dim=3)


_DTYPE_CODES = {
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float32: 3,
}


def _build_spatial_decode_descriptor(
    z: torch.Tensor,
    plan: CogVideoXSpatialTilePlan,
    world_size: int,
) -> torch.Tensor:
    if z.ndim != 5:
        raise ValueError(
            f"CogVideoX VAE SP expects a 5D latent tensor, got {z.shape}"
        )
    try:
        dtype_code = _DTYPE_CODES[z.dtype]
    except KeyError as error:
        raise TypeError(
            f"unsupported CogVideoX VAE SP latent dtype: {z.dtype}"
        ) from error
    return torch.tensor(
        [
            *z.shape,
            dtype_code,
            plan.overlap_height,
            plan.overlap_width,
            plan.blend_extent_height,
            plan.blend_extent_width,
            plan.row_limit_height,
            plan.row_limit_width,
            len(plan.tiles),
            world_size,
        ],
        dtype=torch.int64,
        device=z.device,
    )


def _validate_spatial_decode_descriptor(
    sp_group, z: torch.Tensor, plan: CogVideoXSpatialTilePlan
) -> None:
    local = _build_spatial_decode_descriptor(z, plan, sp_group.world_size)
    gathered = sp_group.all_gather(local.unsqueeze(0), dim=0)
    reference = gathered[0]
    mismatch_ranks = [
        rank
        for rank in range(sp_group.world_size)
        if not torch.equal(gathered[rank], reference)
    ]
    if mismatch_ranks:
        raise RuntimeError(
            "CogVideoX VAE SP input descriptor mismatch on ranks "
            f"{mismatch_ranks}"
        )


def _pack_local_tiles(
    local_tiles: dict[int, torch.Tensor],
    slots_per_rank: int,
    *,
    payload_dtype: torch.dtype,
    payload_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(local_tiles) > slots_per_rank:
        raise RuntimeError(
            "CogVideoX VAE SP local tile count exceeds the fixed slot count"
        )
    metadata = torch.zeros(
        (slots_per_rank, 7), dtype=torch.int64, device=payload_device
    )
    metadata[:, 0] = -1
    payload_parts = []
    for slot, (global_index, tile) in enumerate(sorted(local_tiles.items())):
        if tile.ndim != 5:
            raise RuntimeError(
                f"CogVideoX VAE SP decoded tile {global_index} is not 5D"
            )
        packed_tile = tile.to(device=payload_device, dtype=payload_dtype)
        metadata[slot] = torch.tensor(
            [global_index, *packed_tile.shape, packed_tile.numel()],
            dtype=torch.int64,
            device=payload_device,
        )
        payload_parts.append(packed_tile.reshape(-1))
    payload = (
        torch.cat(payload_parts)
        if payload_parts
        else torch.empty(0, dtype=payload_dtype, device=payload_device)
    )
    return metadata, payload


def _unpack_gathered_tiles(
    gathered_metadata: torch.Tensor,
    gathered_payload: torch.Tensor,
    *,
    total_tiles: int,
) -> dict[int, torch.Tensor]:
    if gathered_metadata.ndim != 3 or gathered_metadata.shape[-1] != 7:
        raise RuntimeError("invalid CogVideoX VAE SP gathered tile metadata")
    if gathered_payload.ndim != 2:
        raise RuntimeError("invalid CogVideoX VAE SP gathered tile payload")
    if gathered_metadata.shape[0] != gathered_payload.shape[0]:
        raise RuntimeError("CogVideoX VAE SP metadata/payload rank mismatch")

    recovered = {}
    for source_rank in range(gathered_metadata.shape[0]):
        payload_offset = 0
        for slot in gathered_metadata[source_rank]:
            global_index = int(slot[0].item())
            if global_index == -1:
                continue
            if not 0 <= global_index < total_tiles:
                raise RuntimeError(
                    f"invalid global tile index {global_index} in VAE SP payload"
                )
            if global_index in recovered:
                raise RuntimeError(
                    f"duplicate global tile index {global_index} in VAE SP payload"
                )
            shape = tuple(int(value.item()) for value in slot[1:6])
            numel = int(slot[6].item())
            expected_numel = 1
            for dimension in shape:
                expected_numel *= dimension
            if numel <= 0 or expected_numel != numel:
                raise RuntimeError(
                    f"invalid shape/numel for global tile index {global_index}"
                )
            if payload_offset + numel > gathered_payload.shape[1]:
                raise RuntimeError(
                    f"truncated payload for global tile index {global_index}"
                )
            recovered[global_index] = (
                gathered_payload[source_rank]
                .narrow(0, payload_offset, numel)
                .view(shape)
            )
            payload_offset += numel

    expected_indices = set(range(total_tiles))
    missing_indices = sorted(expected_indices.difference(recovered))
    if missing_indices:
        raise RuntimeError(
            f"missing global tile index values {missing_indices} in VAE SP payload"
        )
    return dict(sorted(recovered.items()))


def _all_gather_decoded_tiles(
    sp_group,
    local_tiles: dict[int, torch.Tensor],
    total_tiles: int,
    *,
    payload_dtype: torch.dtype,
    payload_device: torch.device,
) -> dict[int, torch.Tensor]:
    slots_per_rank = (total_tiles + sp_group.world_size - 1) // sp_group.world_size
    metadata, payload = _pack_local_tiles(
        local_tiles,
        slots_per_rank,
        payload_dtype=payload_dtype,
        payload_device=payload_device,
    )
    gathered_metadata = sp_group.all_gather(metadata, dim=0).reshape(
        sp_group.world_size, slots_per_rank, 7
    )
    rank_numels = gathered_metadata[:, :, 6].clamp_min(0).sum(dim=1)
    max_rank_numel = int(rank_numels.max().item())
    padded_payload = torch.zeros(
        max_rank_numel, dtype=payload.dtype, device=payload.device
    )
    padded_payload[: payload.numel()].copy_(payload)
    gathered_payload = sp_group.all_gather(padded_payload, dim=0).reshape(
        sp_group.world_size, max_rank_numel
    )
    return _unpack_gathered_tiles(
        gathered_metadata, gathered_payload, total_tiles=total_tiles
    )


class AutoencoderKLCogVideoX(DiffusersAutoencoderKLCogVideoX):
    def __init__(self, config: CogVideoXVAEConfig) -> None:
        arch = config.arch_config
        super().__init__(
            in_channels=arch.in_channels,
            out_channels=arch.out_channels,
            down_block_types=tuple(arch.down_block_types),
            up_block_types=tuple(arch.up_block_types),
            block_out_channels=tuple(arch.block_out_channels),
            latent_channels=arch.latent_channels,
            layers_per_block=arch.layers_per_block,
            act_fn=arch.act_fn,
            norm_eps=arch.norm_eps,
            norm_num_groups=arch.norm_num_groups,
            temporal_compression_ratio=arch.temporal_compression_ratio,
            sample_height=arch.sample_height,
            sample_width=arch.sample_width,
            scaling_factor=arch.scaling_factor,
            shift_factor=arch.shift_factor,
            latents_mean=arch.latents_mean,
            latents_std=arch.latents_std,
            force_upcast=arch.force_upcast,
            use_quant_conv=arch.use_quant_conv,
            use_post_quant_conv=arch.use_post_quant_conv,
            invert_scale_latents=arch.invert_scale_latents,
        )
        self.sglang_config = config
        for name in (
            "tile_sample_min_height",
            "tile_sample_min_width",
            "tile_sample_min_num_frames",
            "tile_sample_stride_height",
            "tile_sample_stride_width",
            "tile_sample_stride_num_frames",
            "blend_num_frames",
            "use_tiling",
            "use_temporal_tiling",
            "use_parallel_tiling",
            "use_temporal_scaling_frames",
            "load_encoder",
            "load_decoder",
        ):
            setattr(self, name, getattr(config, name))


EntryClass = AutoencoderKLCogVideoX
