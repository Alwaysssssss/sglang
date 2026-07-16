from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.multimodal_gen.runtime.models.vaes import cogvideox
from sglang.multimodal_gen.runtime.models.vaes.cogvideox import (
    AutoencoderKLCogVideoX,
    CogVideoXSpatialTile,
    CogVideoXSpatialTilePlan,
    DiffusersAutoencoderKLCogVideoX,
    _assign_spatial_tiles,
    _build_spatial_tile_plan,
    _decode_one_spatial_tile,
    _merge_spatial_tiles,
    _unpack_gathered_tiles,
    _validate_spatial_decode_descriptor,
)


class RecordingDecoder:
    def __init__(self):
        self.received_cache = []

    def __call__(self, tensor, *, conv_cache):
        self.received_cache.append(conv_cache)
        next_cache = f"cache-{len(self.received_cache)}"
        return tensor + 10, next_cache


def make_two_by_two_plan(
    *, row_limit_height: int, row_limit_width: int
) -> CogVideoXSpatialTilePlan:
    return CogVideoXSpatialTilePlan(
        tiles=tuple(
            CogVideoXSpatialTile(index, row, column, row, column)
            for index, (row, column) in enumerate(
                ((0, 0), (0, 1), (1, 0), (1, 1))
            )
        ),
        num_rows=2,
        num_columns=2,
        overlap_height=1,
        overlap_width=1,
        blend_extent_height=1,
        blend_extent_width=1,
        row_limit_height=row_limit_height,
        row_limit_width=row_limit_width,
    )


def make_plan_3x3() -> CogVideoXSpatialTilePlan:
    return _build_spatial_tile_plan(
        latent_height=90,
        latent_width=120,
        tile_latent_min_height=30,
        tile_latent_min_width=45,
        tile_sample_min_height=240,
        tile_sample_min_width=360,
        tile_overlap_factor_height=0.0,
        tile_overlap_factor_width=0.0,
    )


class FakeGroup:
    def __init__(self, *, world_size, rank_in_group=0, gathered_descriptors):
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        self.gathered_descriptors = gathered_descriptors
        self.all_gather_calls = 0

    def all_gather(self, _tensor, dim=0):
        assert dim == 0
        self.all_gather_calls += 1
        return self.gathered_descriptors


def simulate_fixed_tensor_gather(rank_tiles, *, total_tiles):
    slots_per_rank = (total_tiles + len(rank_tiles) - 1) // len(rank_tiles)
    metadata_per_rank = []
    payload_per_rank = []
    for local_tiles in rank_tiles:
        metadata = torch.zeros(slots_per_rank, 7, dtype=torch.int64)
        metadata[:, 0] = -1
        payload_parts = []
        for slot, (global_index, tile) in enumerate(sorted(local_tiles.items())):
            metadata[slot] = torch.tensor(
                [global_index, *tile.shape, tile.numel()], dtype=torch.int64
            )
            payload_parts.append(tile.flatten())
        metadata_per_rank.append(metadata)
        payload_per_rank.append(
            torch.cat(payload_parts) if payload_parts else torch.empty(0)
        )
    max_numel = max(payload.numel() for payload in payload_per_rank)
    padded_payloads = []
    for payload in payload_per_rank:
        padded = torch.zeros(max_numel, dtype=torch.float32)
        padded[: payload.numel()] = payload
        padded_payloads.append(padded)
    return torch.stack(metadata_per_rank), torch.stack(padded_payloads)


def simulate_duplicate_index_gather():
    metadata, payload = simulate_fixed_tensor_gather(
        ({0: torch.ones(1, 1, 1, 1, 1)}, {0: torch.ones(1, 1, 1, 1, 1)}),
        total_tiles=2,
    )
    return metadata, payload


def simulate_missing_index_gather():
    metadata, payload = simulate_fixed_tensor_gather(
        ({0: torch.ones(1, 1, 1, 1, 1)}, {}), total_tiles=2
    )
    return metadata, payload


def test_tile_plan_matches_diffusers_row_major_geometry():
    plan = _build_spatial_tile_plan(
        latent_height=65,
        latent_width=97,
        tile_latent_min_height=30,
        tile_latent_min_width=45,
        tile_sample_min_height=240,
        tile_sample_min_width=360,
        tile_overlap_factor_height=1 / 6,
        tile_overlap_factor_width=1 / 5,
    )
    assert plan.overlap_height == 25
    assert plan.overlap_width == 36
    assert plan.blend_extent_height == 40
    assert plan.blend_extent_width == 72
    assert plan.row_limit_height == 200
    assert plan.row_limit_width == 288
    assert [(tile.latent_top, tile.latent_left) for tile in plan.tiles] == [
        (top, left)
        for top in range(0, 65, 25)
        for left in range(0, 97, 36)
    ]
    assert [tile.global_index for tile in plan.tiles] == list(range(9))


@pytest.mark.parametrize(
    ("total_tiles", "world_size", "expected"),
    [
        (2, 4, ((0,), (1,), (), ())),
        (4, 4, ((0,), (1,), (2,), (3,))),
        (7, 3, ((0, 3, 6), (1, 4), (2, 5))),
    ],
)
def test_round_robin_assignment_is_complete_and_balanced(
    total_tiles, world_size, expected
):
    tiles = tuple(
        CogVideoXSpatialTile(index, 0, index, 0, index)
        for index in range(total_tiles)
    )
    actual = tuple(
        tuple(
            tile.global_index
            for tile in _assign_spatial_tiles(tiles, rank, world_size)
        )
        for rank in range(world_size)
    )
    assert actual == expected
    assert sorted(index for rank_tiles in actual for index in rank_tiles) == list(
        range(total_tiles)
    )


def test_one_spatial_tile_keeps_cache_only_across_its_temporal_batches():
    decoder = RecordingDecoder()
    vae = SimpleNamespace(
        num_latent_frames_batch_size=2,
        tile_latent_min_height=3,
        tile_latent_min_width=4,
        post_quant_conv=None,
        decoder=decoder,
    )
    z = torch.arange(1 * 1 * 5 * 5 * 6).reshape(1, 1, 5, 5, 6).float()
    tile = CogVideoXSpatialTile(0, 0, 0, 1, 1)

    first = _decode_one_spatial_tile(vae, z, tile)
    second = _decode_one_spatial_tile(vae, z, tile)

    assert first.shape == (1, 1, 5, 3, 4)
    assert torch.equal(first, second)
    assert decoder.received_cache[:2] == [None, "cache-1"]
    assert decoder.received_cache[2:] == [None, "cache-3"]


def test_merge_is_row_major_vertical_then_horizontal_then_crop():
    calls = []

    class BlendVAE:
        @staticmethod
        def blend_v(above, current, extent):
            calls.append(
                ("v", int(above.flatten()[0]), int(current.flatten()[0]), extent)
            )
            current.add_(100)
            return current

        @staticmethod
        def blend_h(left, current, extent):
            calls.append(
                ("h", int(left.flatten()[0]), int(current.flatten()[0]), extent)
            )
            current.add_(1000)
            return current

    plan = make_two_by_two_plan(row_limit_height=1, row_limit_width=1)
    tiles = {
        index: torch.full((1, 1, 1, 2, 2), float(index + 1))
        for index in range(4)
    }
    actual = _merge_spatial_tiles(BlendVAE(), plan, tiles)

    assert calls == [
        ("h", 1, 2, plan.blend_extent_width),
        ("v", 1, 3, plan.blend_extent_height),
        ("v", 1002, 4, plan.blend_extent_height),
        ("h", 103, 104, plan.blend_extent_width),
    ]
    assert actual.shape == (1, 1, 1, 2, 2)


def test_unpack_tiles_handles_boundary_shapes_and_empty_rank():
    rank_tiles = (
        {
            0: torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 2, 4),
            3: torch.arange(6, dtype=torch.float32).reshape(1, 1, 1, 2, 3),
        },
        {
            1: torch.arange(4, dtype=torch.float32).reshape(1, 1, 1, 1, 4),
            2: torch.arange(3, dtype=torch.float32).reshape(1, 1, 1, 1, 3),
        },
        {},
    )
    gathered_metadata, gathered_payload = simulate_fixed_tensor_gather(
        rank_tiles, total_tiles=4
    )
    recovered = _unpack_gathered_tiles(
        gathered_metadata, gathered_payload, total_tiles=4
    )

    assert tuple(recovered) == (0, 1, 2, 3)
    assert recovered[3].shape == (1, 1, 1, 2, 3)


@pytest.mark.parametrize(
    ("metadata_payload", "match"),
    [
        (simulate_duplicate_index_gather(), "duplicate.*global tile index"),
        (simulate_missing_index_gather(), "missing.*global tile index"),
    ],
)
def test_unpack_tiles_rejects_duplicate_or_missing_global_index(
    metadata_payload, match
):
    metadata, payload = metadata_payload
    with pytest.raises(RuntimeError, match=match):
        _unpack_gathered_tiles(metadata, payload, total_tiles=2)


def test_descriptor_preflight_rejects_rank_mismatch_before_payload_gather():
    plan = make_plan_3x3()
    z = torch.zeros(1, 16, 5, 90, 120)
    descriptor = torch.tensor(
        [
            1,
            16,
            5,
            90,
            120,
            3,
            plan.overlap_height,
            plan.overlap_width,
            plan.blend_extent_height,
            plan.blend_extent_width,
            plan.row_limit_height,
            plan.row_limit_width,
            len(plan.tiles),
            2,
        ],
        dtype=torch.int64,
    )
    mismatched = descriptor.clone()
    mismatched[4] = 121
    group = FakeGroup(
        world_size=2,
        rank_in_group=0,
        gathered_descriptors=torch.stack((descriptor, mismatched)),
    )
    with pytest.raises(RuntimeError, match="SP input descriptor mismatch"):
        _validate_spatial_decode_descriptor(group, z, plan)
    assert group.all_gather_calls == 1
