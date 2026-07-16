from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.multimodal_gen.runtime.models.vaes import cogvideox
from sglang.multimodal_gen.runtime.models.vaes.cogvideox import (
    AutoencoderKLCogVideoX,
    CogVideoXSpatialTile,
    DiffusersAutoencoderKLCogVideoX,
    _assign_spatial_tiles,
    _build_spatial_tile_plan,
)


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
