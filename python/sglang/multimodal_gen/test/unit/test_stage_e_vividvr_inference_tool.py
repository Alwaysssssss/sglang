import sys
import tempfile
import unittest
from argparse import Namespace
import os
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
    _ensure_python_dev_headers_for_torch_compile,
    _synchronize_ranks_before_cleanup,
    build_dry_run_payload,
    build_runtime_config_snapshot,
    build_server_args,
    parse_args,
    validate_args,
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

    def test_parse_args_supports_qk_norm_fusion_flags(self):
        argv = [
            "run_vividvr_inference.py",
            "--input-video",
            "/tmp/input.mp4",
            "--enable-cogvideox-qk-norm-fusion",
            "--cogvideox-qk-norm-fusion-targets",
            "transformer,controlnet",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertTrue(args.enable_cogvideox_qk_norm_fusion)
        self.assertEqual(
            args.cogvideox_qk_norm_fusion_targets,
            "transformer,controlnet",
        )

    def test_parse_args_supports_distributed_flags(self):
        argv = [
            "run_vividvr_inference.py",
            "--input-video",
            "/tmp/input.mp4",
            "--num-gpus",
            "2",
            "--tp-size",
            "1",
            "--sp-degree",
            "2",
            "--ulysses-degree",
            "2",
            "--ring-degree",
            "1",
            "--master-port",
            "30123",
            "--dist-timeout",
            "1800",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.num_gpus, 2)
        self.assertEqual(args.tp_size, 1)
        self.assertEqual(args.sp_degree, 2)
        self.assertEqual(args.ulysses_degree, 2)
        self.assertEqual(args.ring_degree, 1)
        self.assertEqual(args.master_port, 30123)
        self.assertEqual(args.dist_timeout, 1800)

    def test_build_server_args_forwards_qk_norm_rope_fusion(self):
        args = Namespace(
            cogvideox_ckpt_path=Path("/home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B"),
            vividvr_ckpt_path=Path("/home/zhiheng/Vivid-VR/ckpts/Vivid-VR"),
            num_gpus=2,
            tp_size=1,
            sp_degree=2,
            ulysses_degree=2,
            ring_degree=1,
            dp_size=1,
            dp_degree=1,
            enable_cfg_parallel=False,
            master_port=30123,
            dist_timeout=1800,
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
            enable_cogvideox_qk_norm_fusion=True,
            cogvideox_qk_norm_fusion_targets="transformer,controlnet",
            enable_cogvideox_qk_norm_rope_fusion=True,
            cogvideox_qk_norm_rope_fusion_targets="transformer,controlnet",
            warmup=False,
            warmup_steps=1,
            disable_autocast=None,
            output_dir=Path("/tmp"),
        )

        server_args = build_server_args(args)

        self.assertIsInstance(server_args, ServerArgs)
        self.assertTrue(server_args.enable_cogvideox_qk_norm_fusion)
        self.assertEqual(
            server_args.cogvideox_qk_norm_fusion_targets,
            "transformer,controlnet",
        )
        self.assertTrue(server_args.enable_cogvideox_qk_norm_rope_fusion)
        self.assertEqual(
            server_args.cogvideox_qk_norm_rope_fusion_targets,
            "transformer,controlnet",
        )
        self.assertEqual(server_args.num_gpus, 2)
        self.assertEqual(server_args.sp_degree, 2)
        self.assertEqual(server_args.ulysses_degree, 2)
        self.assertEqual(server_args.ring_degree, 1)
        self.assertEqual(server_args.master_port, 30123)
        self.assertEqual(server_args.dist_timeout, 1800)
        self.assertFalse(server_args.pipeline_config.vae_tiling)

    def test_runtime_snapshot_includes_qk_norm_fusion_fields(self):
        args = Namespace(
            attention_backend="fa",
            use_runai_model_streamer=False,
            use_vividvr_vae_decode_tiling=False,
        )
        server_args = ServerArgs(
            model_path="/tmp/model",
            enable_cogvideox_qk_norm_fusion=True,
            cogvideox_qk_norm_fusion_targets="transformer,controlnet",
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
                    "qk_norm_fusion_targets": ["transformer", "controlnet"],
                    "qk_norm_fusion_transformer": "sglang_layernorm",
                    "qk_norm_fusion_controlnet": "sglang_layernorm",
                },
            )

        self.assertTrue(snapshot["enable_cogvideox_qk_norm_fusion"])
        self.assertEqual(
            snapshot["cogvideox_qk_norm_fusion_targets"],
            ["transformer", "controlnet"],
        )
        self.assertEqual(
            snapshot["qk_norm_fusion_transformer"],
            "sglang_layernorm",
        )
        self.assertEqual(
            snapshot["qk_norm_fusion_controlnet"],
            "sglang_layernorm",
        )
        self.assertFalse(snapshot["runai_model_streamer_enabled"])
        self.assertFalse(snapshot["vividvr_vae_decode_tiling_config"])

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

    def test_build_dry_run_payload_includes_distributed_fields(self):
        args = Namespace(
            input_video=Path("/tmp/input.mp4"),
            prompt_file=Path("/tmp/prompt.txt"),
            caption_file=None,
            reference_video=None,
            phase_label="phase_e41",
            mode_label="acceptance",
            seed=42,
            num_inference_steps=20,
            guidance_scale=7.5,
            restoration_guidance_scale=1.0,
            num_temporal_process_frames=49,
            dtype="bfloat16",
            enable_spatial_tiling=True,
            enable_temporal_tiling=False,
            tile_size=128,
            tile_stride=64,
            attention_backend="fa",
            attention_backend_config=None,
            use_runai_model_streamer=None,
            use_vividvr_vae_decode_tiling=None,
            dit_cpu_offload=False,
            text_encoder_cpu_offload=False,
            vae_cpu_offload=False,
            enable_torch_compile=True,
            enable_cogvideox_modulation_fusion=False,
            cogvideox_modulation_fusion_targets="transformer",
            enable_cogvideox_qkv_fusion=False,
            cogvideox_qkv_fusion_targets="transformer",
            enable_cogvideox_qk_norm_fusion=False,
            cogvideox_qk_norm_fusion_targets="transformer",
            enable_cogvideox_qk_norm_rope_fusion=False,
            cogvideox_qk_norm_rope_fusion_targets="transformer",
            num_gpus=2,
            tp_size=1,
            dp_size=1,
            dp_degree=1,
            sp_degree=2,
            ulysses_degree=2,
            ring_degree=1,
            enable_cfg_parallel=False,
            master_port=30123,
            dist_timeout=1800,
            warmup=False,
            warmup_steps=1,
            disable_autocast=None,
            write_report=True,
        )

        with patch.dict(
            "os.environ",
            {"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"},
            clear=False,
        ):
            payload = build_dry_run_payload(
                args,
                candidate_path=Path("/tmp/out.mp4"),
                report_path=Path("/tmp/report.json"),
                run_id="dryrun",
            )

        self.assertEqual(payload["num_gpus"], 2)
        self.assertEqual(payload["sp_degree"], 2)
        self.assertEqual(payload["ulysses_degree"], 2)
        self.assertEqual(payload["ring_degree"], 1)
        self.assertEqual(payload["master_port"], 30123)
        self.assertEqual(payload["dist_timeout"], 1800)
        self.assertEqual(payload["distributed_env"]["world_size"], 2)
        self.assertEqual(payload["distributed_env"]["rank"], 0)
        self.assertEqual(payload["distributed_env"]["local_rank"], 0)

    def test_validate_args_rejects_num_gpus_world_size_mismatch(self):
        args = Namespace(
            input_video=Path("/tmp/input.mp4"),
            cogvideox_ckpt_path=Path("/tmp/cogvideox"),
            vividvr_ckpt_path=Path("/tmp/vividvr"),
            caption_file=None,
            prompt_file=Path("/tmp/prompt.txt"),
            reference_video=None,
            num_inference_steps=20,
            num_temporal_process_frames=49,
            allow_frame_count_delta=1,
            num_gpus=1,
            dp_size=1,
            dp_degree=1,
            tp_size=1,
            sp_degree=1,
            ulysses_degree=1,
            ring_degree=1,
            dist_timeout=3600,
        )

        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "exists", return_value=True),
            patch.dict(
                "os.environ",
                {"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1"},
                clear=False,
            ),
        ):
            with self.assertRaises(SystemExit) as exc_info:
                validate_args(args)

        self.assertIn("WORLD_SIZE", str(exc_info.exception))

    def test_ensure_python_dev_headers_uses_fallback_include_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            include_dir = (
                Path(tmpdir)
                / "tmp_py310dev"
                / "extracted"
                / "usr"
                / "include"
                / "python3.10"
            )
            include_dir.mkdir(parents=True)
            (include_dir / "Python.h").write_text("/* test header */\n")
            multiarch_dir = (
                Path(tmpdir)
                / "tmp_py310dev"
                / "extracted"
                / "usr"
                / "include"
                / "x86_64-linux-gnu"
                / "python3.10"
            )
            multiarch_dir.mkdir(parents=True)
            (multiarch_dir / "pyconfig.h").write_text("/* test pyconfig */\n")

            def fake_get_config_var(name):
                if name == "INCLUDEPY":
                    return "/missing/python3.10"
                if name == "MULTIARCH":
                    return "x86_64-linux-gnu"
                return None

            def fake_get_path(name):
                return None

            with (
                patch(
                    "sglang.multimodal_gen.tools.run_vividvr_inference.sysconfig.get_config_var",
                    side_effect=fake_get_config_var,
                ),
                patch(
                    "sglang.multimodal_gen.tools.run_vividvr_inference.sysconfig.get_path",
                    side_effect=fake_get_path,
                ),
                patch(
                    "sglang.multimodal_gen.tools.run_vividvr_inference.Path.home",
                    return_value=Path(tmpdir),
                ),
                patch.dict("os.environ", {}, clear=True),
            ):
                resolved = _ensure_python_dev_headers_for_torch_compile()
                self.assertEqual(
                    os.environ["CPATH"],
                    f"{include_dir.parent}{os.pathsep}{include_dir}",
                )
                self.assertEqual(
                    os.environ["C_INCLUDE_PATH"],
                    f"{include_dir.parent}{os.pathsep}{include_dir}",
                )

            self.assertEqual(resolved, include_dir)

    def test_synchronize_ranks_before_cleanup_barriers_for_multi_rank(self):
        with (
            patch(
                "sglang.multimodal_gen.tools.run_vividvr_inference.torch.distributed.is_available",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.tools.run_vividvr_inference.torch.distributed.is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.tools.run_vividvr_inference.torch.distributed.get_world_size",
                return_value=2,
            ),
            patch(
                "sglang.multimodal_gen.tools.run_vividvr_inference.torch.distributed.barrier"
            ) as barrier_mock,
        ):
            _synchronize_ranks_before_cleanup()

        barrier_mock.assert_called_once_with()

    def test_synchronize_ranks_before_cleanup_skips_single_rank(self):
        with (
            patch(
                "sglang.multimodal_gen.tools.run_vividvr_inference.torch.distributed.is_available",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.tools.run_vividvr_inference.torch.distributed.is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.tools.run_vividvr_inference.torch.distributed.get_world_size",
                return_value=1,
            ),
            patch(
                "sglang.multimodal_gen.tools.run_vividvr_inference.torch.distributed.barrier"
            ) as barrier_mock,
        ):
            _synchronize_ranks_before_cleanup()

        barrier_mock.assert_not_called()

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
