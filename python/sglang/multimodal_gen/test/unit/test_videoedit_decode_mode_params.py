import json
import unittest
from datetime import datetime
from unittest.mock import patch

import sglang.multimodal_gen.runtime.entrypoints.openai.video_api as video_api_mod
from sglang.multimodal_gen.configs.sample.videoedit_wan import (
    WanVideoEditSamplingParams,
)
from sglang.multimodal_gen.configs.sample.wan_teacache import _wan_14b_coefficients
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoResponse,
    VideoRepairRequest,
    default_video_repair_output_object_key,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import (
    _build_video_repair_callback_payload,
    _failed_video_repair_submission_job,
    _job_reason,
    _normalize_video_repair_payload,
    _split_output_path,
    _store_failed_video_repair_submission,
    _task_id_from_video_repair_body,
    _video_repair_submit_response,
    _with_video_extension,
)
from sglang.multimodal_gen.runtime.videoedit.cli import build_parser


class TestVideoEditDecodeModeParams(unittest.TestCase):
    def test_sampling_params_accept_default_stream_decode_mode(self):
        params = WanVideoEditSamplingParams()
        self.assertEqual(params.decode_mode, "stream")

    def test_sampling_params_default_num_inference_steps_is_40(self):
        params = WanVideoEditSamplingParams()
        self.assertEqual(params.num_inference_steps, 40)
        self.assertEqual(params.overlap, 9)
        self.assertEqual(params.dilate_px, 15)
        self.assertEqual(params.mask_scale, 1.2)
        self.assertEqual(params.feather_px, 15)
        self.assertEqual(params.adain_boundary_dilate, 0)
        self.assertEqual(params.bbox_expand_scale, 2.5)
        self.assertTrue(params.use_clip)
        self.assertFalse(params.use_repaired_context)
        self.assertEqual(params.init_latent_mode, "noise")
        self.assertEqual(params.mask_downsample_mode, "nearest")
        self.assertEqual(params.overlap_commit_mode, "weighted")
        self.assertEqual(params.tail_padding_mode, "reflect")
        self.assertTrue(params.enable_teacache)

    def test_sampling_params_default_teacache_matches_wan_i2v_14b_720p(self):
        params = WanVideoEditSamplingParams()
        teacache_params = params.teacache_params

        self.assertEqual(teacache_params.teacache_thresh, 0.3)
        self.assertTrue(teacache_params.use_ret_steps)
        self.assertEqual(teacache_params.start_skipping, 5)
        self.assertEqual(teacache_params.end_skipping, 1.0)
        self.assertEqual(
            teacache_params.get_coefficients(),
            _wan_14b_coefficients(teacache_params),
        )

    def test_sampling_params_accept_stream_decode_mode(self):
        params = WanVideoEditSamplingParams(decode_mode="stream")
        self.assertEqual(params.decode_mode, "stream")

    def test_sampling_params_reject_unknown_decode_mode(self):
        with self.assertRaisesRegex(ValueError, "decode_mode must be one of"):
            WanVideoEditSamplingParams(decode_mode="invalid")

    def test_sampling_params_reject_unknown_native_alignment_modes(self):
        with self.assertRaisesRegex(ValueError, "init_latent_mode must be one of"):
            WanVideoEditSamplingParams(init_latent_mode="bad")
        with self.assertRaisesRegex(ValueError, "mask_downsample_mode must be one of"):
            WanVideoEditSamplingParams(mask_downsample_mode="linear")
        with self.assertRaisesRegex(ValueError, "overlap_commit_mode must be one of"):
            WanVideoEditSamplingParams(overlap_commit_mode="blend")
        with self.assertRaisesRegex(ValueError, "tail_padding_mode must be one of"):
            WanVideoEditSamplingParams(tail_padding_mode="pad_last")

    def test_video_repair_request_defaults_to_stream(self):
        request = VideoRepairRequest(
            task_id="task-1",
            callback_url="http://127.0.0.1/callback",
            prompt="repair video",
            video_input_path="/tmp/video.mp4",
            mask_input_path="/tmp/mask.mp4",
        )
        self.assertEqual(request.decode_mode, "stream")
        self.assertEqual(request.overlap, 9)
        self.assertEqual(request.dilate_px, 15)
        self.assertEqual(request.mask_scale, 1.2)
        self.assertEqual(request.feather_px, 15)
        self.assertEqual(request.adain_boundary_dilate, 0)
        self.assertEqual(request.bbox_expand_scale, 2.5)
        self.assertTrue(request.use_clip)
        self.assertFalse(request.use_repaired_context)
        self.assertEqual(request.init_latent_mode, "noise")
        self.assertEqual(request.mask_downsample_mode, "nearest")
        self.assertEqual(request.overlap_commit_mode, "weighted")
        self.assertEqual(request.tail_padding_mode, "reflect")
        self.assertTrue(request.enable_teacache)
        self.assertEqual(request.teacache_thresh, 0.3)
        self.assertEqual(request.teacache_start_skipping, 5)
        self.assertEqual(request.teacache_end_skipping, 1.0)

    def test_video_repair_request_accepts_teacache_overrides(self):
        request = VideoRepairRequest(
            task_id="task-1",
            prompt="repair video",
            video_input_path="/tmp/video.mp4",
            mask_input_path="/tmp/mask.mp4",
            teacache_thresh=0.2,
            teacache_start_skipping=8,
            teacache_end_skipping=0.9,
        )

        self.assertEqual(request.teacache_thresh, 0.2)
        self.assertEqual(request.teacache_start_skipping, 8)
        self.assertEqual(request.teacache_end_skipping, 0.9)

    def test_video_repair_request_normalizes_teacache_camel_case(self):
        payload = _normalize_video_repair_payload(
            {
                "taskId": "task-1",
                "prompt": "repair video",
                "video_input_path": "/tmp/video.mp4",
                "mask_input_path": "/tmp/mask.mp4",
                "teacacheThresh": 0.25,
                "teacacheStartSkipping": 6,
                "teacacheEndSkipping": 0.8,
            }
        )

        request = VideoRepairRequest(**payload)

        self.assertEqual(request.teacache_thresh, 0.25)
        self.assertEqual(request.teacache_start_skipping, 6)
        self.assertEqual(request.teacache_end_skipping, 0.8)

    def test_video_repair_request_defaults_timeout_to_no_limit(self):
        request = VideoRepairRequest(
            task_id="task-1",
            callback_url="http://127.0.0.1/callback",
            prompt="repair video",
            video_input_path="/tmp/video.mp4",
            mask_input_path="/tmp/mask.mp4",
        )
        self.assertEqual(request.timeout, -1)

    def test_video_repair_request_defaults_num_inference_steps_to_40(self):
        request = VideoRepairRequest(
            task_id="task-1",
            callback_url="http://127.0.0.1/callback",
            prompt="repair video",
            video_input_path="/tmp/video.mp4",
            mask_input_path="/tmp/mask.mp4",
        )
        self.assertEqual(request.num_inference_steps, 40)

    def test_video_repair_request_accepts_missing_callback_url(self):
        request = VideoRepairRequest(
            task_id="task-1",
            prompt="repair video",
            video_input_path="/tmp/video.mp4",
            mask_input_path="/tmp/mask.mp4",
        )
        self.assertIsNone(request.callback_url)

    def test_default_output_object_key_uses_date_name(self):
        self.assertEqual(
            default_video_repair_output_object_key(
                "task-1", datetime(2026, 6, 8, 9, 10, 11)
            ),
            "2026/06/08/091011_task-1.mp4",
        )

    def test_default_output_object_key_accepts_mov_extension(self):
        self.assertEqual(
            default_video_repair_output_object_key(
                "task-1", datetime(2026, 6, 8, 9, 10, 11), extension=".mov"
            ),
            "2026/06/08/091011_task-1.mov",
        )

    def test_output_object_key_overrides_to_output_video_extension(self):
        self.assertEqual(
            _with_video_extension("jobs/result.mp4", ".mov"),
            "jobs/result.mov",
        )

    def test_output_object_key_appends_output_video_extension(self):
        self.assertEqual(
            _with_video_extension("jobs/result", ".mov"),
            "jobs/result.mov",
        )

    def test_video_repair_request_accepts_stream(self):
        request = VideoRepairRequest(
            task_id="task-1",
            callback_url="http://127.0.0.1/callback",
            prompt="repair video",
            video_input_path="/tmp/video.mp4",
            mask_input_path="/tmp/mask.mp4",
            decode_mode="stream",
        )
        self.assertEqual(request.decode_mode, "stream")

    def test_sampling_params_output_file_name_follows_mov_source(self):
        params = WanVideoEditSamplingParams(
            prompt="repair video",
            video_input_path="/tmp/source.MOV",
            mask_input_path="/tmp/mask.mp4",
        )

        params._set_output_file_name()

        self.assertTrue(params.output_file_name.endswith(".mov"))

    def test_sampling_params_output_file_name_overrides_to_source_extension(self):
        params = WanVideoEditSamplingParams(
            prompt="repair video",
            video_input_path="/tmp/source.mov",
            mask_input_path="/tmp/mask.mp4",
            output_file_name="result.mp4",
        )

        params._set_output_file_name()

        self.assertEqual(params.output_file_name, "result.mov")

    def test_sampling_params_output_file_name_overrides_mov_to_mp4_source(self):
        params = WanVideoEditSamplingParams(
            prompt="repair video",
            video_input_path="/tmp/source.mp4",
            mask_input_path="/tmp/mask.mp4",
            output_file_name="result.mov",
        )

        params._set_output_file_name()

        self.assertEqual(params.output_file_name, "result.mp4")

    def test_split_output_path_accepts_mov_file_path(self):
        output_dir, output_file_name = _split_output_path(
            "/tmp/result.mov",
            "job-1",
            "/srv/output",
        )

        self.assertEqual(output_dir, "/tmp")
        self.assertEqual(output_file_name, "result.mov")

    def test_split_output_path_forces_reference_extension(self):
        output_dir, output_file_name = _split_output_path(
            "/tmp/result.mp4",
            "job-1",
            "/srv/output",
            reference_path="/tmp/source.mov",
        )

        self.assertEqual(output_dir, "/tmp")
        self.assertEqual(output_file_name, "result.mov")

    def test_split_output_path_forces_mp4_reference_extension(self):
        output_dir, output_file_name = _split_output_path(
            "/tmp/result.mov",
            "job-1",
            "/srv/output",
            reference_path="/tmp/source.mp4",
        )

        self.assertEqual(output_dir, "/tmp")
        self.assertEqual(output_file_name, "result.mp4")

    def test_split_output_path_defaults_to_reference_extension(self):
        output_dir, output_file_name = _split_output_path(
            None,
            "job-1",
            "/srv/output",
            reference_path="/tmp/source.mov",
        )

        self.assertEqual(output_dir, "/srv/output")
        self.assertEqual(output_file_name, "job-1.mov")

    def test_cli_parser_accepts_decode_mode(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "repair",
                "--model-path",
                "/tmp/model",
                "--prompt",
                "repair video",
                "--video-input-path",
                "/tmp/video.mp4",
                "--mask-input-path",
                "/tmp/mask.mp4",
                "--output-path",
                "/tmp/output.mp4",
                "--decode-mode",
                "stream",
            ]
        )
        self.assertEqual(args.decode_mode, "stream")
        self.assertEqual(args.num_inference_steps, 40)
        self.assertEqual(args.overlap, 9)
        self.assertEqual(args.dilate_px, 15)
        self.assertEqual(args.mask_scale, 1.2)
        self.assertEqual(args.feather_px, 15)
        self.assertEqual(args.adain_boundary_dilate, 0)
        self.assertEqual(args.bbox_expand_scale, 2.5)
        self.assertTrue(args.use_clip)
        self.assertFalse(args.use_repaired_context)
        self.assertEqual(args.init_latent_mode, "noise")
        self.assertEqual(args.mask_downsample_mode, "nearest")
        self.assertEqual(args.overlap_commit_mode, "weighted")
        self.assertEqual(args.tail_padding_mode, "reflect")
        self.assertTrue(args.enable_teacache)
        self.assertEqual(args.teacache_thresh, 0.3)
        self.assertEqual(args.teacache_start_skipping, 5)
        self.assertEqual(args.teacache_end_skipping, 1.0)

    def test_cli_parser_accepts_teacache_overrides(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "repair",
                "--model-path",
                "/tmp/model",
                "--prompt",
                "repair video",
                "--video-input-path",
                "/tmp/video.mp4",
                "--mask-input-path",
                "/tmp/mask.mp4",
                "--output-path",
                "/tmp/output.mp4",
                "--teacache-thresh",
                "0.2",
                "--teacache-start-skipping",
                "8",
                "--teacache-end-skipping",
                "0.9",
            ]
        )

        self.assertEqual(args.teacache_thresh, 0.2)
        self.assertEqual(args.teacache_start_skipping, 8)
        self.assertEqual(args.teacache_end_skipping, 0.9)

    def test_failed_job_reason_uses_error_message(self):
        job = {"status": "failed", "error": {"message": "mask file not found"}}
        self.assertEqual(_job_reason(job), "mask file not found")

    def test_failed_job_reason_prefers_reason_field(self):
        job = {
            "status": "failed",
            "error": {"message": "low-level error"},
            "reason": "user-facing reason",
        }
        self.assertEqual(_job_reason(job), "user-facing reason")

    def test_running_job_reason_is_none(self):
        job = {"status": "running", "error": {"message": "not final"}}
        self.assertIsNone(_job_reason(job))

    def test_video_repair_failed_callback_includes_reason(self):
        payload = _build_video_repair_callback_payload(
            "task-1",
            {"status": "failed", "error": {"message": "task timeout"}},
        )
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["reason"], "task timeout")
        self.assertEqual(payload["output"], "")

    def test_video_repair_running_callback_includes_current_progress(self):
        payload = _build_video_repair_callback_payload(
            "task-1",
            {"status": "running", "progress": 37},
        )
        self.assertEqual(payload["status"], "running")
        self.assertEqual(payload["progress"], 37)
        self.assertEqual(payload["reason"], "")
        self.assertEqual(payload["output"], "")

    def test_video_repair_success_callback_uses_downstream_output_format(self):
        payload = _build_video_repair_callback_payload(
            "task-1",
            {
                "status": "completed",
                "url": "https://minio.example.com/outputs/result.mp4",
                "output_object_key": "2026/06/09/060635_task-1.mp4",
                "created_at": 10,
                "completed_at": 55,
            },
        )
        self.assertEqual(payload["status"], "succeeded")
        self.assertEqual(payload["progress"], 100)
        self.assertEqual(payload["reason"], "")
        self.assertEqual(
            json.loads(payload["output"]),
            {
                "gen_video_url": "2026/06/09/060635_task-1.mp4",
                "duration": 45,
            },
        )

    def test_video_repair_submit_failure_includes_reason(self):
        payload = _video_repair_submit_response(1, "videoUrl is required")
        self.assertEqual(payload["message"], "videoUrl is required")
        self.assertEqual(payload["reason"], "videoUrl is required")

    def test_video_repair_submit_failure_task_id_from_camel_case_body(self):
        self.assertEqual(
            _task_id_from_video_repair_body({"taskId": "task-1"}),
            "task-1",
        )

    def test_failed_video_repair_submission_job_exposes_reason(self):
        job = _failed_video_repair_submission_job(
            "task-1",
            "Invalid request body: timeout must be positive or -1",
            body={
                "taskId": "task-1",
                "callbackUrl": "http://127.0.0.1/callback",
                "timeout": 0,
            },
        )
        self.assertEqual(job["status"], "failed")
        self.assertEqual(
            job["reason"], "Invalid request body: timeout must be positive or -1"
        )
        self.assertEqual(
            job["error"]["message"],
            "Invalid request body: timeout must be positive or -1",
        )
        self.assertEqual(job["callback_url"], "http://127.0.0.1/callback")
        self.assertEqual(job["timeout"], 0)

    def test_failed_video_repair_submission_callbacks_reason(self):
        calls = []

        async def fake_post_callback(job_id, callback_url, payload, **kwargs):
            calls.append((job_id, callback_url, payload))

        async def run_test():
            with patch.object(
                video_api_mod, "_post_video_callback", fake_post_callback
            ):
                await _store_failed_video_repair_submission(
                    "task-1",
                    "An error occurred (404) when calling the HeadObject operation: Not Found",
                    body={
                        "taskId": "task-1",
                        "callbackUrl": "http://127.0.0.1/callback",
                    },
                )
                await asyncio.sleep(0)

        import asyncio

        asyncio.run(run_test())

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "task-1")
        self.assertEqual(calls[0][1], "http://127.0.0.1/callback")
        self.assertEqual(calls[0][2]["status"], "failed")
        self.assertEqual(
            calls[0][2]["reason"],
            "An error occurred (404) when calling the HeadObject operation: Not Found",
        )

    def test_video_response_accepts_reason(self):
        response = VideoResponse(
            id="task-1",
            status="failed",
            error={"message": "mask file not found"},
            reason="mask file not found",
        )
        self.assertEqual(response.reason, "mask file not found")


if __name__ == "__main__":
    unittest.main()
