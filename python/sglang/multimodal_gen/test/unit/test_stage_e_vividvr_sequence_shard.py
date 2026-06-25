import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common import (
    VividVRSequenceShardState,
    gather_vividvr_video_tokens,
    shard_vividvr_video_tokens,
    vividvr_sequence_shard_enabled,
)


class TestVividVRSequenceShardHelpers(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
