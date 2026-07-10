from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all,
    _usp_input_all_to_all_qkv,
    _usp_prefix_all_gather,
)
from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_packed_qkv_world_size_one_returns_inputs(dtype):
    q = torch.randn(1, 5, 4, 8, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    with patch(
        "sglang.multimodal_gen.runtime.layers.usp.get_ulysses_parallel_world_size",
        return_value=1,
    ):
        actual = _usp_input_all_to_all_qkv(q, k, v)
    assert actual[0] is q
    assert actual[1] is k
    assert actual[2] is v


def test_packed_qkv_matches_three_legacy_calls():
    q = torch.arange(1 * 3 * 4 * 2, dtype=torch.float32).reshape(1, 3, 4, 2)
    k = q + 1000
    v = q + 2000

    def fake_a2a(x):
        return x.reshape(2, 2, *x.shape[1:]).flip(0).reshape_as(x)

    with patch(
        "sglang.multimodal_gen.runtime.layers.usp.get_ulysses_parallel_world_size",
        return_value=2,
    ), patch(
        "sglang.multimodal_gen.runtime.layers.usp._usp_all_to_all_single",
        side_effect=fake_a2a,
    ):
        expected = tuple(
            _usp_input_all_to_all(x, head_dim=2) for x in (q, k, v)
        )
    with patch(
        "sglang.multimodal_gen.runtime.layers.usp.get_ulysses_parallel_world_size",
        return_value=2,
    ), patch(
        "sglang.multimodal_gen.runtime.layers.usp._usp_all_to_all_single",
        side_effect=fake_a2a,
    ):
        actual = _usp_input_all_to_all_qkv(q, k, v)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda q: (q, q[:, :-1], q), "shapes must match"),
        (lambda q: (q, q.float(), q), "dtypes must match"),
        (lambda q: (q.squeeze(0), q, q), "must all be 4D"),
    ],
)
def test_packed_qkv_rejects_incompatible_inputs(mutate, message):
    q = torch.randn(1, 3, 4, 8, dtype=torch.bfloat16)
    with patch(
        "sglang.multimodal_gen.runtime.layers.usp.get_ulysses_parallel_world_size",
        return_value=2,
    ):
        with pytest.raises(ValueError, match=message):
            _usp_input_all_to_all_qkv(*mutate(q))


def test_packed_qkv_requires_divisible_heads():
    q = torch.randn(1, 3, 3, 8)
    with patch(
        "sglang.multimodal_gen.runtime.layers.usp.get_ulysses_parallel_world_size",
        return_value=2,
    ):
        with pytest.raises(ValueError, match="must be divisible"):
            _usp_input_all_to_all_qkv(q, q, q)


def test_usp_input_selector_uses_packed_helper_when_enabled():
    attention = SimpleNamespace(use_packed_qkv_a2a=True)
    q = torch.randn(1, 3, 4, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    with patch(
        "sglang.multimodal_gen.runtime.layers.attention.layer._usp_input_all_to_all_qkv",
        return_value=(q + 1, k + 1, v + 1),
    ) as packed:
        actual = USPAttention._input_all_to_all_qkv(attention, q, k, v)
    packed.assert_called_once_with(q, k, v)
    torch.testing.assert_close(actual[0], q + 1)


def test_usp_input_selector_keeps_three_legacy_calls_when_disabled():
    attention = SimpleNamespace(use_packed_qkv_a2a=False)
    q = torch.randn(1, 3, 4, 8)
    with patch(
        "sglang.multimodal_gen.runtime.layers.attention.layer._usp_input_all_to_all",
        side_effect=lambda x, head_dim: x + 1,
    ) as legacy:
        actual = USPAttention._input_all_to_all_qkv(attention, q, q, q)
    assert legacy.call_count == 3
    assert all(call.kwargs == {"head_dim": 2} for call in legacy.call_args_list)
    assert all(torch.equal(x, q + 1) for x in actual)


def test_prefix_all_gather_uses_functional_collective_on_head_dim():
    x = torch.randn(1, 5, 2, 8)
    expected = torch.randn(1, 5, 4, 8)
    fake_group = MagicMock()
    fake_sp_group = MagicMock(ulysses_group=fake_group)
    with patch(
        "sglang.multimodal_gen.runtime.layers.usp.get_sp_group",
        return_value=fake_sp_group,
    ), patch(
        "sglang.multimodal_gen.runtime.layers.usp.ft_c.all_gather_tensor",
        return_value=expected,
    ) as gather:
        actual = _usp_prefix_all_gather(x)
    gather.assert_called_once_with(x.contiguous(), gather_dim=2, group=fake_group)
    assert actual is expected
