import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.videoedit.compare import compare_videos
from sglang.multimodal_gen.runtime.videoedit.frame_cache import (
    cache_video_frames,
    clear_cached_video_frames,
)
from sglang.multimodal_gen.runtime.videoedit.preprocess import load_video_frames
from sglang.multimodal_gen.tools.run_vividvr_inference import (
    build_runtime_config_snapshot,
    build_server_args,
    parse_args,
)


class TestVividVRInferenceTool(unittest.TestCase):
    def tearDown(self):
        clear_cached_video_frames()

    def test_parse_args_supports_runtime_control_flags(self):
        argv = [
            "run_vividvr_inference.py",
            "--input-video",
            "/tmp/input.mp4",
            "--no-use-runai-model-streamer",
            "--no-use-vividvr-vae-decode-tiling",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertFalse(args.use_runai_model_streamer)
        self.assertFalse(args.use_vividvr_vae_decode_tiling)

    def test_parse_args_supports_qk_norm_rope_fusion_flags(self):
        argv = [
            "run_vividvr_inference.py",
            "--input-video",
            "/tmp/input.mp4",
            "--enable-cogvideox-qk-norm-rope-fusion",
            "--cogvideox-qk-norm-rope-fusion-targets",
            "transformer,controlnet",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertTrue(args.enable_cogvideox_qk_norm_rope_fusion)
        self.assertEqual(
            args.cogvideox_qk_norm_rope_fusion_targets,
            "transformer,controlnet",
        )

    def test_build_server_args_forwards_qk_norm_rope_fusion(self):
        args = Namespace(
            cogvideox_ckpt_path=Path("/home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B"),
            vividvr_ckpt_path=Path("/home/zhiheng/Vivid-VR/ckpts/Vivid-VR"),
            attention_backend="fa",
            attention_backend_config=None,
            use_runai_model_streamer=None,
            use_vividvr_vae_decode_tiling=False,
            dit_cpu_offload=False,
            text_encoder_cpu_offload=False,
            vae_cpu_offload=False,
            enable_torch_compile=False,
            enable_cogvideox_modulation_fusion=False,
            cogvideox_modulation_fusion_targets="transformer",
            enable_cogvideox_qkv_fusion=False,
            cogvideox_qkv_fusion_targets="transformer",
            enable_cogvideox_qk_norm_rope_fusion=True,
            cogvideox_qk_norm_rope_fusion_targets="transformer,controlnet",
            warmup=False,
            warmup_steps=1,
            disable_autocast=None,
            output_dir=Path("/tmp"),
        )

        server_args = build_server_args(args)

        self.assertIsInstance(server_args, ServerArgs)
        self.assertTrue(server_args.enable_cogvideox_qk_norm_rope_fusion)
        self.assertEqual(
            server_args.cogvideox_qk_norm_rope_fusion_targets,
            "transformer,controlnet",
        )
        self.assertFalse(server_args.pipeline_config.vae_tiling)

    def test_runtime_snapshot_includes_qk_norm_rope_fusion_fields(self):
        args = Namespace(
            attention_backend="fa",
            use_runai_model_streamer=False,
            use_vividvr_vae_decode_tiling=False,
        )
        server_args = ServerArgs(
            model_path="/tmp/model",
            enable_cogvideox_qk_norm_rope_fusion=True,
            cogvideox_qk_norm_rope_fusion_targets="transformer,controlnet",
        )
        server_args.pipeline_config.vae_tiling = False

        with patch.dict(
            "os.environ",
            {"SGLANG_USE_RUNAI_MODEL_STREAMER": "0"},
            clear=False,
        ):
            snapshot = build_runtime_config_snapshot(
                args=args,
                server_args=server_args,
                debug={
                    "qk_norm_rope_fusion_targets": ["transformer", "controlnet"],
                    "qk_norm_rope_fusion_transformer": "sglang_layernorm+rope_accel",
                    "qk_norm_rope_fusion_controlnet": "sglang_layernorm+rope_accel",
                },
            )

        self.assertTrue(snapshot["enable_cogvideox_qk_norm_rope_fusion"])
        self.assertEqual(
            snapshot["cogvideox_qk_norm_rope_fusion_targets"],
            ["transformer", "controlnet"],
        )
        self.assertEqual(
            snapshot["qk_norm_rope_fusion_transformer"],
            "sglang_layernorm+rope_accel",
        )
        self.assertEqual(
            snapshot["qk_norm_rope_fusion_controlnet"],
            "sglang_layernorm+rope_accel",
        )
        self.assertFalse(snapshot["runai_model_streamer_enabled"])
        self.assertFalse(snapshot["vividvr_vae_decode_tiling_config"])

    def test_runtime_snapshot_includes_e31_helper_fields(self):
        args = Namespace(
            attention_backend="fa",
            use_runai_model_streamer=None,
            use_vividvr_vae_decode_tiling=None,
        )
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

    def test_load_video_frames_reuses_compare_cache(self):
        cache_video_frames(
            "/tmp/cached.mp4",
            [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(2)],
            12.5,
        )

        with patch(
            "sglang.multimodal_gen.runtime.videoedit.preprocess.cv2.VideoCapture",
            side_effect=AssertionError("VideoCapture should not be called when cache is warm"),
        ):
            frames, fps = load_video_frames("/tmp/cached.mp4")

        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0].size, (3, 2))
        self.assertEqual(fps, 12.5)

    def test_compare_videos_reports_original_frame_counts(self):
        frame = np.zeros((2, 3, 3), dtype=np.uint8)

        with patch(
            "sglang.multimodal_gen.runtime.videoedit.compare._read_video",
            side_effect=[
                ([frame, frame], 24.0),
                ([frame, frame, frame], 24.0),
            ],
        ):
            report = compare_videos(
                "/tmp/reference.mp4",
                "/tmp/candidate.mp4",
                allow_frame_count_delta=1,
            )

        summary = report["summary"]
        self.assertEqual(summary["reference_frame_count"], 2)
        self.assertEqual(summary["candidate_frame_count"], 3)
        self.assertEqual(summary["frame_count_delta"], 1)


if __name__ == "__main__":
    unittest.main()
