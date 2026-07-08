import unittest
from argparse import Namespace
from unittest.mock import patch

from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.tools.run_vividvr_inference import (
    build_runtime_config_snapshot,
)


class TestVividVRRuntimeSnapshot(unittest.TestCase):
    def test_runtime_snapshot_includes_sp4_fields(self):
        args = Namespace(
            attention_backend="fa",
            use_runai_model_streamer=None,
            use_vividvr_vae_decode_tiling=False,
        )
        server_args = ServerArgs(model_path="/tmp/model", disable_autocast=False)
        server_args.pipeline_config.vae_tiling = False

        snapshot = build_runtime_config_snapshot(
            args=args,
            server_args=server_args,
            debug={
                "enable_sequence_shard": True,
                "sp_world_size": 4,
                "sp_rank": 0,
                "sp_sequence_tokens_global": 17552,
                "sp_sequence_tokens_local": 4388,
                "sp_sequence_tokens_pad": 0,
                "connector_context_mode": "sp_exact_distributed_control_attention",
                "denoise_loop_local_compute_ms": 18234.5,
                "denoise_loop_sp_comm_ms": 3112.0,
            },
        )

        self.assertTrue(snapshot["enable_sequence_shard"])
        self.assertEqual(snapshot["sp_world_size"], 4)
        self.assertEqual(snapshot["sp_rank"], 0)
        self.assertEqual(snapshot["sp_sequence_tokens_global"], 17552)
        self.assertEqual(snapshot["sp_sequence_tokens_local"], 4388)
        self.assertEqual(snapshot["sp_sequence_tokens_pad"], 0)
        self.assertEqual(
            snapshot["connector_context_mode"],
            "sp_exact_distributed_control_attention",
        )
        self.assertEqual(snapshot["denoise_loop_local_compute_ms"], 18234.5)
        self.assertEqual(snapshot["denoise_loop_sp_comm_ms"], 3112.0)

    def test_runtime_snapshot_includes_e31_e32_helper_fields(self):
        args = Namespace(
            attention_backend="fa",
            use_runai_model_streamer=True,
            use_vividvr_vae_decode_tiling=False,
        )
        server_args = ServerArgs(model_path="/tmp/model", disable_autocast=False)
        server_args.pipeline_config.vae_tiling = False

        with patch.dict(
            "os.environ",
            {"SGLANG_USE_RUNAI_MODEL_STREAMER": "1"},
            clear=False,
        ):
            snapshot = build_runtime_config_snapshot(
                args=args,
                server_args=server_args,
                debug={
                    "denoising_autocast_enabled": True,
                    "denoising_target_dtype": "bfloat16",
                    "denoising_device_type": "cuda",
                    "device_placement_helper": "DenoisingStage._manage_device_placement",
                    "denoising_step_profile_helper": "DenoisingStage.step_profile",
                    "attn_metadata_enabled": True,
                    "attn_metadata_backend": "fa",
                    "attn_metadata_builder": "FlashAttentionMetadataBuilder",
                    "enable_sequence_shard": True,
                    "sp_sequence_shard_strategy": "model_native_video_token_shard",
                    "sp_sequence_tokens_global": 17550,
                    "sp_sequence_tokens_local": 8775,
                    "sp_sequence_tokens_pad": 0,
                    "sp_video_token_layout": "contiguous_flat_video_token_sequence",
                    "runtime_num_timesteps": 20,
                    "connector_context_mode": "sp_exact_local_attention",
                    "control_context_shape_local": (2, 8775, 3072),
                    "control_context_shape_global": None,
                    "vae_tiling_enabled": True,
                },
            )

        self.assertTrue(snapshot["denoising_autocast_enabled"])
        self.assertEqual(snapshot["denoising_target_dtype"], "bfloat16")
        self.assertEqual(snapshot["denoising_device_type"], "cuda")
        self.assertTrue(snapshot["runai_model_streamer_enabled"])
        self.assertTrue(snapshot["runai_model_streamer_requested"])
        self.assertFalse(snapshot["vividvr_vae_decode_tiling_requested"])
        self.assertFalse(snapshot["vividvr_vae_decode_tiling_config"])
        self.assertEqual(
            snapshot["device_placement_helper"],
            "DenoisingStage._manage_device_placement",
        )
        self.assertEqual(
            snapshot["denoising_step_profile_helper"],
            "DenoisingStage.step_profile",
        )
        self.assertTrue(snapshot["attn_metadata_enabled"])
        self.assertEqual(snapshot["attn_metadata_backend"], "fa")
        self.assertEqual(
            snapshot["attn_metadata_builder"],
            "FlashAttentionMetadataBuilder",
        )
        self.assertTrue(snapshot["enable_sequence_shard"])
        self.assertEqual(
            snapshot["sp_sequence_shard_strategy"],
            "model_native_video_token_shard",
        )
        self.assertEqual(snapshot["sp_sequence_tokens_global"], 17550)
        self.assertEqual(snapshot["sp_sequence_tokens_local"], 8775)
        self.assertEqual(snapshot["sp_sequence_tokens_pad"], 0)
        self.assertEqual(
            snapshot["sp_video_token_layout"],
            "contiguous_flat_video_token_sequence",
        )
        self.assertEqual(snapshot["runtime_num_timesteps"], 20)
        self.assertEqual(
            snapshot["connector_context_mode"],
            "sp_exact_local_attention",
        )
        self.assertEqual(
            snapshot["control_context_shape_local"],
            [2, 8775, 3072],
        )
        self.assertIsNone(snapshot["control_context_shape_global"])
        self.assertTrue(snapshot["vae_tiling_enabled"])


if __name__ == "__main__":
    unittest.main()
