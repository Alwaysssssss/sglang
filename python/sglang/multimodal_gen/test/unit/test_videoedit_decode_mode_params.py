import json
import unittest
from datetime import datetime

from sglang.multimodal_gen.configs.sample.videoedit_wan import (
    WanVideoEditSamplingParams,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoResponse,
    VideoRepairRequest,
    default_video_repair_output_object_key,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import (
    _build_video_repair_callback_payload,
    _failed_video_repair_submission_job,
    _job_reason,
    _task_id_from_video_repair_body,
    _video_repair_submit_response,
)
from sglang.multimodal_gen.runtime.videoedit.cli import build_parser


class TestVideoEditDecodeModeParams(unittest.TestCase):
    def test_sampling_params_accept_default_stream_decode_mode(self):
        params = WanVideoEditSamplingParams()
        self.assertEqual(params.decode_mode, "stream")

    def test_sampling_params_accept_stream_decode_mode(self):
        params = WanVideoEditSamplingParams(decode_mode="stream")
        self.assertEqual(params.decode_mode, "stream")

    def test_sampling_params_reject_unknown_decode_mode(self):
        with self.assertRaisesRegex(ValueError, "decode_mode must be one of"):
            WanVideoEditSamplingParams(decode_mode="invalid")

    def test_video_repair_request_defaults_to_stream(self):
        request = VideoRepairRequest(
            task_id="task-1",
            callback_url="http://127.0.0.1/callback",
            prompt="repair video",
            video_input_path="/tmp/video.mp4",
            mask_input_path="/tmp/mask.mp4",
        )
        self.assertEqual(request.decode_mode, "stream")

    def test_video_repair_request_defaults_timeout_to_no_limit(self):
        request = VideoRepairRequest(
            task_id="task-1",
            callback_url="http://127.0.0.1/callback",
            prompt="repair video",
            video_input_path="/tmp/video.mp4",
            mask_input_path="/tmp/mask.mp4",
        )
        self.assertEqual(request.timeout, -1)

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
                "result_url": "https://minio.example.com/outputs/result.mp4",
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
