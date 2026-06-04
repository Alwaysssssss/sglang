import json
import os
import unittest
from datetime import datetime, timezone
from pathlib import Path

import torch

from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.pipeline_configs.vividvr import (
    VividVRPipelineConfig,
    vividvr_t5_postprocess_text,
)
from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.configs.sample.vividvr import VividVRSamplingParams
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    post_process_sample,
    prepare_request,
)
from sglang.multimodal_gen.runtime.pipelines_core import build_pipeline
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
from sglang.multimodal_gen.runtime.videoedit.compare import compare_videos
from sglang.multimodal_gen.runtime.videoedit.preprocess import load_video_frames
from sglang.multimodal_gen.runtime.vividvr.preprocess import load_control_video

VIVIDVR_ROOT = Path("/home/zhiheng/Vivid-VR")
COGVIDEOX_ROOT = VIVIDVR_ROOT / "ckpts" / "CogVideoX1.5-5B"
VIVIDVR_CKPT_ROOT = VIVIDVR_ROOT / "ckpts" / "Vivid-VR"
INPUT_VIDEO = VIVIDVR_ROOT / "input" / "720p" / "test_video_960x720.mp4"
PROMPT_FILE = VIVIDVR_ROOT / "input" / "720p" / "prompt.txt"
REFERENCE_VIDEO = (
    VIVIDVR_ROOT
    / "result"
    / "720p_up1_result_vivid_ori"
    / "videos"
    / "test_video_960x720.mp4"
)
ACCEPTANCE_ROOT = Path("/home/zhiheng/sglang/Vivid_Acceptance")
INDICATOR_DIR = ACCEPTANCE_ROOT / "indicator"
RESULT_VIDEOS_DIR = ACCEPTANCE_ROOT / "result_videos"
ACCEPTANCE_COMMAND = (
    "SGLANG_RUN_VIVIDVR_ACCEPTANCE=1 PYTHONPATH=python uv run "
    "--with pytest --with diffusers==0.37.0 --with imageio==2.36.0 "
    "--with imageio-ffmpeg==0.5.1 --with addict==2.4.0 --with PyYAML==6.0.1 "
    "--with av==16.1.0 --with scikit-image==0.25.2 --with cache-dit==1.3.0 "
    "--with opencv-python-headless==4.10.0.84 --with trimesh "
    "python -m pytest python/sglang/multimodal_gen/test/unit/test_stage_c_vividvr_single_clip.py -q"
)


class TestStageCVividVRContracts(unittest.TestCase):
    def test_text_encoder_contract_matches_vividvr_reference(self):
        config = VividVRPipelineConfig()
        self.assertEqual(config.text_encoder_configs[0].arch_config.text_len, 226)
        self.assertIn("Cinematic, High Contrast", config.default_positive_prompt_suffix)

    def test_text_postprocess_preserves_vividvr_sequence_length(self):
        hidden_state = torch.randn(1, 226, 32)
        outputs = BaseEncoderOutput(
            last_hidden_state=hidden_state,
            attention_mask=torch.ones(1, 226, dtype=torch.long),
        )
        text_inputs = {"input_ids": torch.ones(1, 226, dtype=torch.long)}

        prompt_embeds = vividvr_t5_postprocess_text(outputs, text_inputs)

        self.assertEqual(tuple(prompt_embeds.shape), (1, 226, 32))
        self.assertTrue(torch.equal(prompt_embeds, hidden_state))

    def test_control_video_padding_contract_matches_reference_wrapper(self):
        info = load_control_video(str(INPUT_VIDEO))

        self.assertEqual(info["original_num_frames"], int(info["reference_video"].shape[0]))
        self.assertEqual(
            int(info["video"].shape[0]),
            int(info["reference_video"].shape[0]) + int(info["num_padding_frames"]),
        )
        self.assertEqual(int(info["num_padding_frames"]), 3)


@unittest.skipUnless(
    os.environ.get("SGLANG_RUN_VIVIDVR_ACCEPTANCE") == "1",
    "Set SGLANG_RUN_VIVIDVR_ACCEPTANCE=1 to run the heavy Phase C acceptance test",
)
@unittest.skipUnless(torch.cuda.is_available(), "Phase C acceptance requires CUDA")
class TestStageCVividVRSingleClip(unittest.TestCase):
    def _build_server_args(self, output_path: str) -> ServerArgs:
        server_args = ServerArgs(
            model_path=str(COGVIDEOX_ROOT),
            pipeline_class_name="CogVideoXVividVRControlNetPipeline",
            pipeline_config=VividVRPipelineConfig(),
            component_paths={"vividvr": str(VIVIDVR_CKPT_ROOT)},
            num_gpus=1,
            tp_size=1,
            dp_size=1,
            dp_degree=1,
            sp_degree=1,
            dit_cpu_offload=False,
            text_encoder_cpu_offload=False,
            vae_cpu_offload=False,
            nunchaku_config=None,
            output_path=output_path,
        )
        server_args._adjust_parameters()
        set_global_server_args(server_args)
        return server_args

    def _make_request(
        self,
        *,
        server_args: ServerArgs,
        output_path: str,
        output_file_name: str,
        seed: int,
    ):
        params = VividVRSamplingParams.from_user_kwargs(
            server_args,
            prompt=" ",
            video_input_path=str(INPUT_VIDEO),
            prompt_file_path=str(PROMPT_FILE),
            output_path=output_path,
            output_file_name=output_file_name,
            save_output=False,
            return_file_paths_only=False,
            seed=seed,
        )
        return prepare_request(server_args, params)

    def test_single_clip_reference_alignment(self):
        INDICATOR_DIR.mkdir(parents=True, exist_ok=True)
        RESULT_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate_path = RESULT_VIDEOS_DIR / f"phase_c_candidate_seed42_{run_id}.mp4"
        report_path = INDICATOR_DIR / f"phase_c_metrics_seed42_{run_id}.json"

        server_args = self._build_server_args(str(RESULT_VIDEOS_DIR))
        pipeline = build_pipeline(server_args)

        first_req = self._make_request(
            server_args=server_args,
            output_path=str(RESULT_VIDEOS_DIR),
            output_file_name=candidate_path.name,
            seed=42,
        )
        first_result = pipeline.forward(first_req, server_args)

        second_req = self._make_request(
            server_args=server_args,
            output_path=str(RESULT_VIDEOS_DIR),
            output_file_name=f"phase_c_candidate_repeat_seed42_{run_id}.mp4",
            seed=42,
        )
        second_result = pipeline.forward(second_req, server_args)

        torch.testing.assert_close(first_result.output, second_result.output)

        post_process_sample(
            first_result.output,
            DataType.VIDEO,
            int(first_result.fps),
            save_output=True,
            save_file_path=str(candidate_path),
            video_reference_path=str(REFERENCE_VIDEO),
        )

        report = compare_videos(
            str(REFERENCE_VIDEO),
            str(candidate_path),
            min_ssim=0.90,
            max_mse=150.0,
            max_mae=8.0,
            allow_frame_count_delta=1,
            max_failed_frame_ratio=0.05,
        )
        summary = report["summary"]
        debug = first_result.extra["vividvr_debug"]

        ref_frames, _ = load_video_frames(str(REFERENCE_VIDEO))
        cand_frames, _ = load_video_frames(str(candidate_path))
        failed_frame_ratio = (
            len(summary["failed_frames"]) / summary["compared_frames"]
            if summary["compared_frames"] > 0
            else 1.0
        )
        metrics_record = {
            "phase": "C",
            "run_id": run_id,
            "run_datetime_utc": datetime.now(timezone.utc).isoformat(),
            "command": ACCEPTANCE_COMMAND,
            "seed": 42,
            "prompt_path": str(PROMPT_FILE),
            "input_video_path": str(INPUT_VIDEO),
            "reference_video_path": str(REFERENCE_VIDEO),
            "candidate_video_path": str(candidate_path),
            "reference_frame_count": len(ref_frames),
            "candidate_frame_count": len(cand_frames),
            "frame_count_delta": abs(len(ref_frames) - len(cand_frames)),
            "failed_frame_ratio": failed_frame_ratio,
            "summary": summary,
            "frames": report["frames"],
            "debug": debug,
        }
        report_path.write_text(
            json.dumps(metrics_record, indent=2),
            encoding="utf-8",
        )

        for key in (
            "prompt_embed_shape",
            "control_latent_shape",
            "latents_shape",
            "timestep_count",
            "tile_count",
        ):
            self.assertIn(key, debug)

        self.assertTrue(candidate_path.exists())
        self.assertTrue(report_path.exists())
        self.assertGreaterEqual(summary["ssim_min"], 0.90)
        self.assertLessEqual(summary["mse_max"], 150.0)
        self.assertLessEqual(summary["mae_max"], 8.0)
        self.assertLessEqual(failed_frame_ratio, 0.05)
        self.assertTrue(summary["pass_compare"])
