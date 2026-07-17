from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.multimodal_gen.runtime.models.vaes import cogvideox
from sglang.multimodal_gen.runtime.models.vaes.cogvideox import (
    AutoencoderKLCogVideoX,
    CogVideoXSpatialEncodeTile,
    CogVideoXSpatialEncodeTilePlan,
    DiffusersAutoencoderKLCogVideoX,
    _all_gather_decoded_tiles,
    _all_gather_spatial_tiles,
    _assign_spatial_tiles,
    _build_spatial_encode_tile_plan,
    _canonicalize_spatial_encode_input,
    _encode_one_spatial_tile,
    _merge_spatial_encode_tiles,
    _unpack_gathered_tiles,
    _validate_spatial_encode_descriptor,
)
from sglang.multimodal_gen.tools.run_vividvr_vae_spatial_encode_validation import (
    build_rank_divergent_validation_fields,
    compare_serial_and_parallel_encode,
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
        (tile.global_index, tile.sample_top, tile.sample_left) for tile in plan.tiles
    ] == [
        (index, top, left)
        for index, (top, left) in enumerate(
            [(top, left) for top in (0, 200, 400, 600) for left in (0, 288, 576, 864)]
        )
    ]
    assert (plan.blend_extent_height, plan.blend_extent_width) == (5, 9)
    assert (plan.row_limit_height, plan.row_limit_width) == (25, 36)


@pytest.mark.parametrize(("world_size", "expected"), [(2, [8, 8]), (4, [4, 4, 4, 4])])
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
        (top, left) for top in (0, 200, 400) for left in (0, 288, 576)
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
    x = torch.arange(9, dtype=torch.float32).view(1, 1, 9, 1, 1).expand(1, 3, 9, 6, 8)
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
            for index, (row, column) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1)))
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

    shared = _all_gather_spatial_tiles(LocalGatherGroup(), local_tiles, 2, **kwargs)
    decoded = _all_gather_decoded_tiles(LocalGatherGroup(), local_tiles, 2, **kwargs)

    assert tuple(shared) == tuple(decoded) == (0, 1)
    for index in shared:
        assert shared[index].shape == decoded[index].shape
        assert shared[index].dtype == decoded[index].dtype == torch.bfloat16
        assert torch.equal(shared[index], decoded[index])


class FakeCudaEvent:
    _next_timestamp_ms = 0.0

    def __init__(self, *, enable_timing):
        assert enable_timing is True
        self.timestamp_ms = None

    def record(self):
        self.timestamp_ms = type(self)._next_timestamp_ms
        type(self)._next_timestamp_ms += 10.0

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return other.timestamp_ms - self.timestamp_ms


class FakeGroup:
    def __init__(self, world_size, *, rank_in_group=0, gathered=None):
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        self.ranks = tuple(range(10, 10 + world_size))
        self.gathered = gathered

    def all_gather(self, tensor, dim=0):
        assert dim == 0
        if self.gathered is not None:
            return self.gathered
        return tensor.repeat(self.world_size, *([1] * (tensor.ndim - 1)))

    @staticmethod
    def broadcast(tensor, src=0):
        assert src == 0
        return tensor


class BroadcastFakeGroup(FakeGroup):
    def __init__(self, canonical):
        super().__init__(2, rank_in_group=1)
        self.canonical = canonical
        self.broadcast_sources = []

    def broadcast(self, tensor, src=0):
        self.broadcast_sources.append(src)
        tensor.copy_(self.canonical)
        return tensor


def test_encode_descriptor_rejects_rank_mismatch():
    plan = make_formal_encode_plan()
    x = torch.zeros(1, 3, 9, 720, 960)
    local = cogvideox._build_spatial_encode_descriptor(x, plan, FakeGroup(2))
    mismatch = local.clone()
    mismatch[4] += 1
    group = FakeGroup(2, gathered=torch.stack((local, mismatch)))

    with pytest.raises(RuntimeError, match="encode input descriptor mismatch"):
        _validate_spatial_encode_descriptor(group, x, plan)


def test_encode_input_is_contiguous_and_canonicalized_from_subgroup_root():
    root = torch.arange(24).reshape(1, 3, 2, 2, 2).transpose(-1, -2)
    local = torch.full_like(root, -1)
    group = BroadcastFakeGroup(root.contiguous())

    canonical = _canonicalize_spatial_encode_input(group, local)

    assert group.broadcast_sources == [0]
    assert canonical.is_contiguous()
    assert torch.equal(canonical, root.contiguous())
    assert torch.equal(local, torch.full_like(local, -1))


@pytest.mark.parametrize("dtype", [torch.int64, torch.uint8])
def test_encode_descriptor_rejects_unsupported_dtype(dtype):
    with pytest.raises(TypeError, match="unsupported.*encode.*dtype"):
        cogvideox._build_spatial_encode_descriptor(
            torch.zeros(1, 3, 1, 1, 1, dtype=dtype),
            make_two_by_two_encode_plan(),
            FakeGroup(2),
        )


@pytest.mark.parametrize("duplicate", [True, False])
def test_encode_transport_rejects_duplicate_or_missing_tiles(duplicate):
    metadata = torch.zeros((2, 1, 7), dtype=torch.int64)
    metadata[:, :, 0] = -1
    metadata[0, 0] = torch.tensor([0, 1, 1, 1, 1, 1, 1])
    if duplicate:
        metadata[1, 0] = torch.tensor([0, 1, 1, 1, 1, 1, 1])
    payload = torch.ones((2, 1))
    match = "duplicate" if duplicate else "missing"
    with pytest.raises(RuntimeError, match=match):
        _unpack_gathered_tiles(metadata, payload, total_tiles=2)


class TrackingEncoder:
    def __init__(self):
        self.tile_values = []

    def __call__(self, tensor, *, conv_cache):
        del conv_cache
        self.tile_values.append(int(tensor.flatten()[0]))
        return tensor, None


def make_toy_runtime(*, use_tiling=True):
    vae = object.__new__(AutoencoderKLCogVideoX)
    torch.nn.Module.__init__(vae)
    vae.use_tiling = use_tiling
    vae.tile_sample_min_height = 1
    vae.tile_sample_min_width = 1
    vae.tile_latent_min_height = 1
    vae.tile_latent_min_width = 1
    vae.tile_overlap_factor_height = 0.0
    vae.tile_overlap_factor_width = 0.0
    vae.num_sample_frames_batch_size = 1
    vae.encoder = TrackingEncoder()
    vae.quant_conv = None
    vae.blend_v = lambda _above, current, _extent: current
    vae.blend_h = lambda _left, current, _extent: current
    vae._vae_encode_sp_requested = False
    vae._vae_encode_sp_group = None
    vae._last_spatial_encode_stats = None
    return vae


def make_tiled_input():
    return torch.arange(3, dtype=torch.float32).reshape(1, 1, 1, 1, 3)


def test_encode_parallel_startup_requires_tiling_and_sp_group(monkeypatch):
    vae = make_toy_runtime(use_tiling=False)
    with pytest.raises(ValueError, match="vae_encode_sp requires VAE tiling"):
        vae.configure_spatial_tile_encode_parallel(True)

    vae = make_toy_runtime(use_tiling=True)
    monkeypatch.setattr(
        cogvideox,
        "get_sp_group",
        lambda: (_ for _ in ()).throw(AssertionError("not initialized")),
    )
    with pytest.raises(RuntimeError, match="vae_encode_sp.*not initialized"):
        vae.configure_spatial_tile_encode_parallel(True)


@pytest.mark.parametrize(
    "requested,world_size,tiled,reason",
    [
        (False, 1, True, "not_requested"),
        (True, 1, True, "sp_world_size_one"),
        (True, 2, False, "input_below_tiling_threshold"),
    ],
)
def test_encode_serial_fallback_reasons(
    requested, world_size, tiled, reason, monkeypatch
):
    vae = make_toy_runtime(use_tiling=True)
    group = FakeGroup(world_size)
    monkeypatch.setattr(cogvideox, "get_sp_group", lambda: group)
    monkeypatch.setattr(cogvideox.torch.cuda, "Event", FakeCudaEvent)
    monkeypatch.setattr(
        DiffusersAutoencoderKLCogVideoX,
        "_encode",
        lambda _self, x: x,
    )
    monkeypatch.setattr(
        DiffusersAutoencoderKLCogVideoX,
        "tiled_encode",
        lambda _self, x: x,
    )
    vae.configure_spatial_tile_encode_parallel(requested)
    if tiled:
        x = make_tiled_input()
    else:
        vae.tile_sample_min_height = 240
        vae.tile_sample_min_width = 360
        x = torch.zeros(1, 3, 5, 240, 360)

    vae._encode(x)

    stats = vae.get_last_spatial_encode_stats()
    assert stats.effective is False
    assert stats.fallback_reason == reason
    assert stats.total_tiles == 0
    assert stats.tile_compute_seconds >= 0.0
    assert stats.tile_gather_seconds >= 0.0
    assert stats.tile_merge_seconds >= 0.0
    assert stats.encode_seconds >= 0.0


def test_encode_parallel_encodes_only_owned_tiles_and_reports_stats(monkeypatch):
    vae = make_toy_runtime()
    group = FakeGroup(2)
    monkeypatch.setattr(cogvideox, "get_sp_group", lambda: group)
    monkeypatch.setattr(cogvideox.torch.cuda, "Event", FakeCudaEvent)
    monkeypatch.setattr(
        cogvideox, "_validate_spatial_encode_descriptor", lambda *_args: None
    )

    def gather_all(_group, local_tiles, total_tiles, **_kwargs):
        assert tuple(local_tiles) == (0, 2)
        assert total_tiles == 3
        return {
            index: torch.tensor(float(index)).reshape(1, 1, 1, 1, 1)
            for index in range(total_tiles)
        }

    monkeypatch.setattr(cogvideox, "_all_gather_spatial_tiles", gather_all)
    vae.configure_spatial_tile_encode_parallel(True)

    moments = vae.tiled_encode(make_tiled_input())

    assert vae.encoder.tile_values == [0, 2]
    assert moments.shape == (1, 1, 1, 1, 3)
    stats = vae.get_last_spatial_encode_stats()
    assert stats.effective is True
    assert stats.local_tiles_per_rank == (2, 1)


def test_encode_parallel_does_not_retry_serial_after_collective_failure(
    monkeypatch,
):
    vae = make_toy_runtime()
    group = FakeGroup(2)
    monkeypatch.setattr(cogvideox, "get_sp_group", lambda: group)
    monkeypatch.setattr(cogvideox.torch.cuda, "Event", FakeCudaEvent)
    monkeypatch.setattr(
        cogvideox, "_validate_spatial_encode_descriptor", lambda *_args: None
    )
    monkeypatch.setattr(
        cogvideox,
        "_all_gather_spatial_tiles",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("collective failed")
        ),
    )
    serial = Mock()
    monkeypatch.setattr(DiffusersAutoencoderKLCogVideoX, "tiled_encode", serial)
    vae.configure_spatial_tile_encode_parallel(True)

    with pytest.raises(RuntimeError, match="collective failed"):
        vae.tiled_encode(make_tiled_input())
    serial.assert_not_called()


def test_encode_validation_requires_exact_moments_and_sampled_latents():
    moments = torch.arange(8, dtype=torch.bfloat16)
    latents = torch.arange(4, dtype=torch.bfloat16)
    assert compare_serial_and_parallel_encode(
        moments, moments.clone(), latents, latents.clone()
    )["passed"]

    changed = moments.clone()
    changed[0] += 1
    result = compare_serial_and_parallel_encode(
        moments, changed, latents, latents.clone()
    )

    assert result["moments_exact"] is False
    assert result["passed"] is False


def test_encode_validation_uses_rank_divergent_input_contract_key():
    comparison = {"passed": True, "moments_exact": True}

    assert build_rank_divergent_validation_fields(comparison) == {
        "rank_divergent_passed": True,
        "rank_divergent_input_comparison": comparison,
    }
    DiffusersAutoencoderKLCogVideoX,
    _canonicalize_spatial_encode_input,
