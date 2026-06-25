import sys
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.tools.run_vividvr_phase_d_long_video import (
    make_request,
    parse_args,
)


class TestPhaseDLongVideoTool(unittest.TestCase):
    def test_parse_args_supports_original_upscale_flag(self):
        argv = [
            "run_vividvr_phase_d_long_video.py",
            "--upscale",
            "2.0",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.upscale, 2.0)

    def test_make_request_forwards_original_upscale_contract(self):
        server_args = ServerArgs(
            model_path="/tmp/model",
            output_path="/tmp/out",
        )

        fake_request = SimpleNamespace(kind="prepared")
        with (
            patch(
                "sglang.multimodal_gen.tools.run_vividvr_phase_d_long_video.VividVRSamplingParams.from_user_kwargs"
            ) as from_user_kwargs_mock,
            patch(
                "sglang.multimodal_gen.tools.run_vividvr_phase_d_long_video.prepare_request",
                return_value=fake_request,
            ) as prepare_request_mock,
        ):
            params = SimpleNamespace(upscale=0.0)
            from_user_kwargs_mock.return_value = params

            request = make_request(
                server_args=server_args,
                input_video_path=Path("/tmp/input.mp4"),
                output_file_name="candidate.mp4",
                seed=42,
                num_inference_steps=20,
                caption_file_path=Path("/tmp/captions.txt"),
                upscale=0.0,
            )

        self.assertIs(request, fake_request)
        self.assertEqual(from_user_kwargs_mock.call_args.kwargs["upscale"], 0.0)
        prepare_request_mock.assert_called_once_with(server_args, params)


if __name__ == "__main__":
    unittest.main()
