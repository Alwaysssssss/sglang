import unittest
from os import environ
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common import (
    Connector,
    VividVRSequenceShardState,
    build_vividvr_connector_control_states,
    get_vividvr_connector_sp_context_mode,
    restore_vividvr_connector_global_control_state,
    restore_vividvr_connector_global_control_states,
    gather_vividvr_video_tokens,
    run_vividvr_connector_attention,
    run_vividvr_connector_sequence_parallel_attention,
    shard_vividvr_video_tokens,
    unpack_vividvr_connector_context,
    vividvr_sequence_shard_enabled,
)


class TestVividVRSequenceShardHelpers(unittest.TestCase):
    def test_get_vividvr_connector_sp_context_mode_defaults_to_eager_global(self):
        with patch.dict(environ, {}, clear=True):
            self.assertEqual(
                get_vividvr_connector_sp_context_mode(),
                "eager_global",
            )

    def test_sequence_shard_enabled_requires_forward_batch_and_sp_world(self):
        context = SimpleNamespace(
            forward_batch=SimpleNamespace(enable_sequence_shard=True)
        )

        with (
            patch(
                "sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common.get_forward_context",
                return_value=context,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common.get_sp_world_size",
                return_value=2,
            ),
        ):
            self.assertTrue(vividvr_sequence_shard_enabled())

    def test_shard_video_tokens_slices_rank_local_tokens_and_rope(self):
        hidden_states = torch.arange(10, dtype=torch.float32).view(1, 5, 2)
        cos = torch.arange(20, dtype=torch.float32).view(5, 4)
        sin = cos + 100
        context = SimpleNamespace(
            forward_batch=SimpleNamespace(enable_sequence_shard=True)
        )

        with (
            patch(
                "sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common.get_forward_context",
                return_value=context,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common.get_sp_world_size",
                return_value=2,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common.get_sp_parallel_rank",
                return_value=1,
            ),
        ):
            local_hidden_states, local_rope, shard_state = shard_vividvr_video_tokens(
                hidden_states,
                (cos, sin),
            )

        expected_hidden_states = torch.tensor(
            [[[6.0, 7.0], [8.0, 9.0], [0.0, 0.0]]]
        )
        expected_cos = torch.tensor(
            [[12.0, 13.0, 14.0, 15.0], [16.0, 17.0, 18.0, 19.0], [0.0, 0.0, 0.0, 0.0]]
        )
        expected_sin = torch.tensor(
            [
                [112.0, 113.0, 114.0, 115.0],
                [116.0, 117.0, 118.0, 119.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

        self.assertEqual(
            shard_state,
            VividVRSequenceShardState(
                enabled=True,
                original_seq_len=5,
                local_seq_len=3,
                seq_pad=1,
            ),
        )
        self.assertTrue(torch.equal(local_hidden_states, expected_hidden_states))
        self.assertIsNotNone(local_rope)
        assert local_rope is not None
        self.assertTrue(torch.equal(local_rope[0], expected_cos))
        self.assertTrue(torch.equal(local_rope[1], expected_sin))

    def test_gather_video_tokens_trims_padding_after_all_gather(self):
        local_hidden_states = torch.tensor([[[6.0, 7.0], [8.0, 9.0], [0.0, 0.0]]])
        gathered_hidden_states = torch.tensor(
            [[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0], [0.0, 0.0]]]
        )

        with patch(
            "sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common.sequence_model_parallel_all_gather",
            return_value=gathered_hidden_states,
        ) as gather_mock:
            hidden_states = gather_vividvr_video_tokens(
                local_hidden_states,
                VividVRSequenceShardState(
                    enabled=True,
                    original_seq_len=5,
                    local_seq_len=3,
                    seq_pad=1,
                ),
            )

        gather_mock.assert_called_once()
        self.assertTrue(
            torch.equal(
                hidden_states,
                torch.tensor(
                    [[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]]
                ),
            )
        )

    def test_restore_connector_global_control_state_trims_sequence_padding(self):
        local_state = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]])
        gathered_state = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [0.0, 0.0], [0.0, 0.0]]]
        )

        with patch(
            "sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common.sequence_model_parallel_all_gather",
            return_value=gathered_state,
        ) as gather_mock:
            restored = restore_vividvr_connector_global_control_state(
                local_state,
                VividVRSequenceShardState(
                    enabled=True,
                    original_seq_len=5,
                    local_seq_len=3,
                    seq_pad=1,
                ),
            )

        gather_mock.assert_called_once()
        self.assertTrue(
            torch.equal(
                restored,
                torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [0.0, 0.0]]]),
            )
        )

    def test_build_connector_control_states_restores_global_states_by_default(
        self,
    ):
        local_states = (
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]),
        )
        gathered_states = torch.tensor(
            [
                [[[0.5, 1.0], [1.5, 2.0], [9.0, 9.5], [10.0, 10.5]]],
                [[[2.5, 3.0], [3.5, 4.0], [11.0, 11.5], [12.0, 12.5]]],
            ]
        )

        with (
            patch(
                "sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common.sequence_model_parallel_all_gather",
                return_value=gathered_states,
            ) as gather_mock,
            patch.dict(environ, {}, clear=True),
        ):
            connector_states = build_vividvr_connector_control_states(
                local_states,
                VividVRSequenceShardState(
                    enabled=True,
                    original_seq_len=4,
                    local_seq_len=2,
                    seq_pad=0,
                ),
                conditioning_scale=0.5,
            )

        self.assertEqual(len(connector_states), 2)
        self.assertEqual(len(connector_states[0]), 2)
        gather_mock.assert_called_once()
        self.assertTrue(
            torch.equal(
                connector_states[0][0],
                torch.tensor([[[0.5, 1.0], [1.5, 2.0]]]),
            )
        )
        self.assertTrue(
            torch.equal(
                connector_states[0][1],
                gathered_states[0],
            )
        )
        self.assertTrue(
            torch.equal(
                connector_states[1][0],
                torch.tensor([[[2.5, 3.0], [3.5, 4.0]]]),
            )
        )
        self.assertTrue(torch.equal(connector_states[1][1], gathered_states[1]))

    def test_build_connector_control_states_restores_global_states_in_eager_mode(self):
        local_states = (
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]),
        )
        gathered_states = torch.tensor(
            [
                [[[0.5, 1.0], [1.5, 2.0], [9.0, 9.5], [10.0, 10.5]]],
                [[[2.5, 3.0], [3.5, 4.0], [11.0, 11.5], [12.0, 12.5]]],
            ]
        )

        with (
            patch(
                "sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common.sequence_model_parallel_all_gather",
                return_value=gathered_states,
            ) as gather_mock,
            patch.dict(
                environ,
                {"SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE": "eager_global"},
                clear=False,
            ),
        ):
            connector_states = build_vividvr_connector_control_states(
                local_states,
                VividVRSequenceShardState(
                    enabled=True,
                    original_seq_len=4,
                    local_seq_len=2,
                    seq_pad=0,
                ),
                conditioning_scale=0.5,
            )

        gather_mock.assert_called_once()
        self.assertEqual(len(connector_states), 2)
        self.assertEqual(len(connector_states[0]), 2)
        self.assertTrue(
            torch.equal(
                connector_states[0][0],
                torch.tensor([[[0.5, 1.0], [1.5, 2.0]]]),
            )
        )
        self.assertTrue(torch.equal(connector_states[0][1], gathered_states[0]))
        self.assertTrue(
            torch.equal(
                connector_states[1][0],
                torch.tensor([[[2.5, 3.0], [3.5, 4.0]]]),
            )
        )
        self.assertTrue(torch.equal(connector_states[1][1], gathered_states[1]))

    def test_restore_connector_global_control_states_trims_padding_after_packed_gather(self):
        local_states = (
            torch.tensor([[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]]),
            torch.tensor([[[5.0, 6.0], [7.0, 8.0], [0.0, 0.0]]]),
        )
        gathered_states = torch.tensor(
            [
                [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [0.0, 0.0], [0.0, 0.0]]],
                [[[5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [0.0, 0.0], [0.0, 0.0]]],
            ]
        )

        with patch(
            "sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common.sequence_model_parallel_all_gather",
            return_value=gathered_states,
        ) as gather_mock:
            restored_states = restore_vividvr_connector_global_control_states(
                local_states,
                VividVRSequenceShardState(
                    enabled=True,
                    original_seq_len=5,
                    local_seq_len=3,
                    seq_pad=1,
                ),
            )

        gather_mock.assert_called_once()
        self.assertEqual(len(restored_states), 2)
        self.assertEqual(restored_states[0].shape, (1, 5, 2))
        self.assertEqual(restored_states[1].shape, (1, 5, 2))
        self.assertTrue(
            torch.equal(
                restored_states[0],
                torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [0.0, 0.0]]]),
            )
        )
        self.assertTrue(
            torch.equal(
                restored_states[1],
                torch.tensor([[[5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [0.0, 0.0]]]),
            )
        )

    def test_unpack_connector_context_supports_legacy_and_global_restored_shapes(self):
        local_only = torch.tensor([[[1.0, 2.0]]])
        global_restored = torch.tensor([[[3.0, 4.0], [5.0, 6.0]]])

        local_ctx, global_ctx = unpack_vividvr_connector_context((local_only,))
        self.assertTrue(torch.equal(local_ctx, local_only))
        self.assertTrue(torch.equal(global_ctx, local_only))

        local_ctx, global_ctx = unpack_vividvr_connector_context(
            (local_only, global_restored)
        )
        self.assertTrue(torch.equal(local_ctx, local_only))
        self.assertTrue(torch.equal(global_ctx, global_restored))

    def test_connector_uses_global_context_for_attention_and_local_context_for_mlp(self):
        connector = Connector(hidden_size=2, num_attention_heads=1)
        connector.to_q = nn.Identity()
        connector.to_k = nn.Identity()
        connector.norm_q = nn.Identity()
        connector.norm_k = nn.Identity()
        connector.out_layer = nn.Identity()
        connector.c_mlp = nn.Identity()

        local_control = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        global_control = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [5.0, 0.0], [0.0, 5.0]]]
        )
        hidden_states = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        captured: dict[str, torch.Tensor] = {}

        def _fake_sdpa(q, k, v):
            captured["q"] = q.detach().clone()
            captured["k"] = k.detach().clone()
            captured["v"] = v.detach().clone()
            return torch.zeros_like(q)

        with patch(
            "sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common.F.scaled_dot_product_attention",
            side_effect=_fake_sdpa,
        ):
            output = connector((local_control, global_control), hidden_states)

        self.assertEqual(captured["k"].shape[2], 4)
        self.assertEqual(captured["v"].shape[2], 4)
        self.assertTrue(torch.equal(output, hidden_states + local_control))

    def test_connector_uses_distributed_attention_for_local_sharded_context(self):
        connector = Connector(hidden_size=2, num_attention_heads=1)
        connector.to_q = nn.Identity()
        connector.to_k = nn.Identity()
        connector.norm_q = nn.Identity()
        connector.norm_k = nn.Identity()
        connector.out_layer = nn.Identity()
        connector.c_mlp = nn.Identity()

        local_control = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float16)
        hidden_states = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float16)
        captured: dict[str, torch.Tensor] = {}

        def _fake_sp_attn(q, k, v):
            captured["q"] = q.detach().clone()
            captured["k"] = k.detach().clone()
            captured["v"] = v.detach().clone()
            return torch.zeros_like(q)

        with (
            patch(
                "sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common.vividvr_sequence_shard_enabled",
                return_value=True,
            ),
            patch.dict(
                environ,
                {"SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE": "distributed_local"},
                clear=False,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common.run_vividvr_connector_sequence_parallel_attention",
                side_effect=_fake_sp_attn,
            ) as sp_attn_mock,
        ):
            output = connector((local_control,), hidden_states)

        sp_attn_mock.assert_called_once()
        self.assertEqual(captured["q"].shape, (1, 2, 1, 2))
        self.assertTrue(torch.equal(captured["k"], local_control.view(1, 2, 1, 2)))
        self.assertTrue(torch.equal(captured["v"], local_control.view(1, 2, 1, 2)))
        self.assertTrue(torch.equal(output, hidden_states + local_control))

    def test_connector_uses_local_flash_attention_for_deferred_global_local_steps(self):
        connector = Connector(hidden_size=2, num_attention_heads=1)
        connector.to_q = nn.Identity()
        connector.to_k = nn.Identity()
        connector.norm_q = nn.Identity()
        connector.norm_k = nn.Identity()
        connector.out_layer = nn.Identity()
        connector.c_mlp = nn.Identity()

        local_control = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float16)
        hidden_states = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float16)
        captured: dict[str, torch.Tensor] = {}

        def _fake_local_exact(q, k, v):
            captured["q"] = q.detach().clone()
            captured["k"] = k.detach().clone()
            captured["v"] = v.detach().clone()
            return torch.zeros_like(q)

        with (
            patch(
                "sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common.vividvr_sequence_shard_enabled",
                return_value=True,
            ),
            patch.dict(
                environ,
                {"SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE": "deferred_global"},
                clear=False,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common.run_vividvr_connector_sequence_parallel_attention",
                side_effect=AssertionError(
                    "distributed sequence-parallel attention should not run in deferred_global local-only steps"
                ),
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common.run_vividvr_connector_attention",
                side_effect=_fake_local_exact,
            ) as local_exact_mock,
        ):
            output = connector((local_control,), hidden_states)

        local_exact_mock.assert_called_once()
        self.assertEqual(captured["q"].shape, (1, 2, 1, 2))
        self.assertTrue(torch.equal(captured["k"], local_control.view(1, 2, 1, 2)))
        self.assertTrue(torch.equal(captured["v"], local_control.view(1, 2, 1, 2)))
        self.assertTrue(torch.equal(output, hidden_states + local_control))

    def test_connector_sequence_shard_attention_uses_usp_attention(self):
        query = torch.arange(8, dtype=torch.float16).view(1, 2, 1, 4)
        key = query + 10
        value = query + 20
        context = SimpleNamespace(
            forward_batch=SimpleNamespace(enable_sequence_shard=True)
        )
        captured: dict[str, object] = {}

        class _FakeSequenceParallelAttention:
            def __call__(self, q, k, v):
                captured["q"] = q.detach().clone()
                captured["k"] = k.detach().clone()
                captured["v"] = v.detach().clone()
                return v + 100

        with (
            patch(
                "sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common.get_forward_context",
                return_value=context,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common.get_sp_world_size",
                return_value=2,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common._get_vividvr_connector_sequence_parallel_attention",
                return_value=_FakeSequenceParallelAttention(),
            ) as get_sp_attn_mock,
        ):
            out = run_vividvr_connector_sequence_parallel_attention(query, key, value)

        get_sp_attn_mock.assert_called_once_with(num_heads=1, head_size=4)
        self.assertTrue(torch.equal(captured["q"], query))
        self.assertTrue(torch.equal(captured["k"], key))
        self.assertTrue(torch.equal(captured["v"], value))
        self.assertTrue(torch.equal(out, value + 100))

    def test_connector_sequence_shard_attention_uses_local_flash_attention_without_collectives(
        self,
    ):
        query = torch.arange(8, dtype=torch.float16).view(1, 2, 1, 4)
        key = query + 10
        value = query + 20
        context = SimpleNamespace(
            forward_batch=SimpleNamespace(enable_sequence_shard=True)
        )

        with (
            patch(
                "sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common.get_forward_context",
                return_value=context,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common.get_sp_world_size",
                return_value=2,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common.F.scaled_dot_product_attention",
                side_effect=AssertionError(
                    "scaled_dot_product_attention should not run on the flash fast path"
                ),
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common.flash_attn_func",
                side_effect=lambda **kwargs: kwargs["v"],
            ) as flash_mock,
        ):
            out = run_vividvr_connector_attention(query, key, value)

        flash_mock.assert_called_once()
        self.assertTrue(torch.equal(out, value))


if __name__ == "__main__":
    unittest.main()
