from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all,
    _usp_input_all_to_all_qkv,
)


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
