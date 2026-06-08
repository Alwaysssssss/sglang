import unittest
from datetime import datetime

from sglang.multimodal_gen.configs.sample.videoedit_wan import (
    WanVideoEditSamplingParams,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoRepairRequest,
    default_video_repair_output_object_key,
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
            "20260608/091011_task-1.mp4",
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


if __name__ == "__main__":
    unittest.main()
