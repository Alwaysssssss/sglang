from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.models.vaes.cogvideox import (
    CogVideoXSpatialEncodeTile,
    CogVideoXSpatialEncodeTilePlan,
    _all_gather_decoded_tiles,
    _all_gather_spatial_tiles,
    _assign_spatial_tiles,
    _build_spatial_encode_tile_plan,
    _encode_one_spatial_tile,
    _merge_spatial_encode_tiles,
)


def make_formal_encode_plan() -> CogVideoXSpatialEncodeTilePlan:
    return _build_spatial_encode_tile_plan(
        sample_height=720,
        sample_width=960,
        tile_sample_min_height=240,
        tile_sample_min_width=360,
        tile_latent_min_height=30,
        tile_latent_min_width=45,
        tile_overlap_factor_height=1 / 6,
        tile_overlap_factor_width=1 / 5,
    )


def test_encode_plan_matches_formal_720x960_geometry():
    plan = make_formal_encode_plan()

    assert (plan.num_rows, plan.num_columns) == (4, 4)
    assert [
        (tile.global_index, tile.sample_top, tile.sample_left)
        for tile in plan.tiles
    ] == [
        (index, top, left)
        for index, (top, left) in enumerate(
            [
                (top, left)
                for top in (0, 200, 400, 600)
                for left in (0, 288, 576, 864)
            ]
        )
    ]
    assert (plan.blend_extent_height, plan.blend_extent_width) == (5, 9)
    assert (plan.row_limit_height, plan.row_limit_width) == (25, 36)


@pytest.mark.parametrize(
    ("world_size", "expected"), [(2, [8, 8]), (4, [4, 4, 4, 4])]
)
def test_encode_tiles_use_round_robin_ownership(world_size, expected):
    plan = make_formal_encode_plan()

    counts = [
        len(_assign_spatial_tiles(plan.tiles, rank, world_size))
        for rank in range(world_size)
    ]

    assert counts == expected


def test_encode_plan_keeps_partial_edge_tiles_in_row_major_order():
    plan = _build_spatial_encode_tile_plan(
        sample_height=401,
        sample_width=577,
        tile_sample_min_height=240,
        tile_sample_min_width=360,
        tile_latent_min_height=30,
        tile_latent_min_width=45,
        tile_overlap_factor_height=1 / 6,
        tile_overlap_factor_width=1 / 5,
    )

    assert (plan.num_rows, plan.num_columns) == (3, 3)
    assert [(tile.sample_top, tile.sample_left) for tile in plan.tiles] == [
        (top, left)
        for top in (0, 200, 400)
        for left in (0, 288, 576)
    ]


class RecordingEncoder:
    def __init__(self):
        self.input_caches = []
        self.ranges = []

    def __call__(self, tensor, *, conv_cache):
        self.input_caches.append(conv_cache)
        frame_values = tensor[0, 0, :, 0, 0]
        self.ranges.append((int(frame_values[0]), int(frame_values[-1]) + 1))
        return tensor, f"cache-{len(self.input_caches) - 1}"


class RecordingQuantConv:
    def __init__(self):
        self.calls = 0

    def __call__(self, tensor):
        self.calls += 1
        return tensor + 1


def test_encode_one_tile_preserves_temporal_cache_and_quant_conv():
    encoder = RecordingEncoder()
    quant_conv = RecordingQuantConv()
    vae = SimpleNamespace(
        num_sample_frames_batch_size=4,
        tile_sample_min_height=6,
        tile_sample_min_width=8,
        encoder=encoder,
        quant_conv=quant_conv,
    )
    x = torch.arange(9, dtype=torch.float32).view(1, 1, 9, 1, 1).expand(
        1, 3, 9, 6, 8
    )
    tile = CogVideoXSpatialEncodeTile(0, 0, 0, 0, 0)

    encoded = _encode_one_spatial_tile(vae, x, tile)

    assert encoder.ranges == [(0, 5), (5, 9)]
    assert encoder.input_caches == [None, "cache-0"]
    assert quant_conv.calls == 2
    assert encoded.shape[2] == 9


def make_two_by_two_encode_plan() -> CogVideoXSpatialEncodeTilePlan:
    return CogVideoXSpatialEncodeTilePlan(
        tiles=tuple(
            CogVideoXSpatialEncodeTile(index, row, column, row, column)
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
        row_limit_height=1,
        row_limit_width=1,
    )


def test_encode_merge_matches_diffusers_order_and_crop():
    calls = []

    class RecordingBlendVae:
        @staticmethod
        def blend_v(above, current, _extent):
            calls.append(("v", int(above.flatten()[0]), int(current.flatten()[0])))
            return current

        @staticmethod
        def blend_h(left, current, _extent):
            calls.append(("h", int(left.flatten()[0]), int(current.flatten()[0])))
            return current

    plan = make_two_by_two_encode_plan()
    tiles = {
        index: torch.full((1, 1, 1, 2, 2), index, dtype=torch.float32)
        for index in range(4)
    }

    merged = _merge_spatial_encode_tiles(RecordingBlendVae(), plan, tiles)

    assert calls == [
        ("h", 0, 1),
        ("v", 0, 2),
        ("v", 1, 3),
        ("h", 2, 3),
    ]
    assert merged.shape[-2:] == (
        plan.num_rows * plan.row_limit_height,
        plan.num_columns * plan.row_limit_width,
    )


class LocalGatherGroup:
    world_size = 1

    @staticmethod
    def all_gather(tensor, dim=0):
        assert dim == 0
        return tensor


def test_shared_transport_preserves_decode_wrapper_bytes():
    local_tiles = {
        0: torch.arange(8, dtype=torch.bfloat16).reshape(1, 1, 1, 2, 4),
        1: torch.arange(3, dtype=torch.bfloat16).reshape(1, 1, 1, 1, 3),
    }
    kwargs = {
        "payload_dtype": torch.bfloat16,
        "payload_device": torch.device("cpu"),
    }

    shared = _all_gather_spatial_tiles(
        LocalGatherGroup(), local_tiles, 2, **kwargs
    )
    decoded = _all_gather_decoded_tiles(
        LocalGatherGroup(), local_tiles, 2, **kwargs
    )

    assert tuple(shared) == tuple(decoded) == (0, 1)
    for index in shared:
        assert shared[index].shape == decoded[index].shape
        assert shared[index].dtype == decoded[index].dtype == torch.bfloat16
        assert torch.equal(shared[index], decoded[index])
