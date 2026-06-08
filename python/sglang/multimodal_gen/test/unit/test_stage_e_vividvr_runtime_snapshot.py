import unittest
from argparse import Namespace

from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.tools.run_vividvr_inference import (
    build_runtime_config_snapshot,
)


class TestVividVRRuntimeSnapshot(unittest.TestCase):
    def test_runtime_snapshot_includes_e31_e32_helper_fields(self):
        args = Namespace(attention_backend="fa")
        server_args = ServerArgs(model_path="/tmp/model", disable_autocast=False)

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
                "vae_tiling_enabled": True,
            },
        )

        self.assertTrue(snapshot["denoising_autocast_enabled"])
        self.assertEqual(snapshot["denoising_target_dtype"], "bfloat16")
        self.assertEqual(snapshot["denoising_device_type"], "cuda")
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
        self.assertTrue(snapshot["vae_tiling_enabled"])


if __name__ == "__main__":
    unittest.main()
