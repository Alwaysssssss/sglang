import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.multimodal_gen.configs.pipeline_configs.vividvr import (
    VividVRPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.entrypoints import utils


class TestSaveOutputsNormalization(unittest.TestCase):
    def test_save_outputs_treats_single_video_tensor_as_one_sample(self):
        sample = torch.zeros(3, 70, 8, 8)
        observed_shapes = []

        def fake_post_process_sample(output, *args, **kwargs):
            observed_shapes.append(tuple(output.shape))
            return []

        with patch.object(
            utils,
            "post_process_sample",
            side_effect=fake_post_process_sample,
        ):
            output_paths = utils.save_outputs(
                sample,
                DataType.VIDEO,
                fps=8,
                save_output=False,
                build_output_path=lambda idx: f"/tmp/out_{idx}.mp4",
            )

        self.assertEqual(output_paths, ["/tmp/out_0.mp4"])
        self.assertEqual(observed_shapes, [(3, 70, 8, 8)])

    def test_save_outputs_splits_batched_video_tensor_by_batch_dim(self):
        sample = torch.zeros(2, 3, 16, 8, 8)
        observed_shapes = []

        def fake_post_process_sample(output, *args, **kwargs):
            observed_shapes.append(tuple(output.shape))
            return []

        with patch.object(
            utils,
            "post_process_sample",
            side_effect=fake_post_process_sample,
        ):
            output_paths = utils.save_outputs(
                sample,
                DataType.VIDEO,
                fps=8,
                save_output=False,
                build_output_path=lambda idx: f"/tmp/out_{idx}.mp4",
            )

        self.assertEqual(output_paths, ["/tmp/out_0.mp4", "/tmp/out_1.mp4"])
        self.assertEqual(observed_shapes, [(3, 16, 8, 8), (3, 16, 8, 8)])


class TestResolveVideoReferencePath(unittest.TestCase):
    def test_explicit_reference_override_wins_for_vividvr(self):
        pipeline_config = VividVRPipelineConfig()
        server_args = SimpleNamespace(pipeline_config=pipeline_config)
        request = SimpleNamespace(video_input_path="/tmp/input.mp4")

        resolved = utils.resolve_video_reference_path(
            request_like=request,
            server_args=server_args,
            explicit_path="/tmp/explicit.mp4",
        )

        self.assertEqual(resolved, "/tmp/explicit.mp4")

    def test_vividvr_falls_back_to_pipeline_reference_video(self):
        pipeline_config = VividVRPipelineConfig()
        server_args = SimpleNamespace(pipeline_config=pipeline_config)
        request = SimpleNamespace(video_input_path="/tmp/input.mp4")

        resolved = utils.resolve_video_reference_path(
            request_like=request,
            server_args=server_args,
            explicit_path=None,
        )

        self.assertEqual(resolved, pipeline_config.reference_video_path)

    def test_non_vividvr_uses_explicit_path(self):
        server_args = SimpleNamespace(pipeline_config=object())

        resolved = utils.resolve_video_reference_path(
            request_like=SimpleNamespace(video_input_path="/tmp/input.mp4"),
            server_args=server_args,
            explicit_path="/tmp/explicit.mp4",
        )

        self.assertEqual(resolved, "/tmp/explicit.mp4")

    def test_request_level_reference_path_beats_video_input_path(self):
        request = SimpleNamespace(
            video_input_path="/tmp/input.mp4",
            reference_video_path="/tmp/request_reference.mp4",
        )

        resolved = utils.resolve_video_reference_path(
            request_like=request,
            explicit_path=None,
        )

        self.assertEqual(resolved, "/tmp/request_reference.mp4")

    def test_falls_back_to_request_video_input_path(self):
        request = SimpleNamespace(video_input_path="/tmp/input.mp4")

        resolved = utils.resolve_video_reference_path(
            request_like=request,
            explicit_path=None,
        )

        self.assertEqual(resolved, "/tmp/input.mp4")


class TestVideoEncodingPolicy(unittest.TestCase):
    def test_vividvr_uses_reference_profile_encoding_mode_and_quality(self):
        server_args = SimpleNamespace(pipeline_config=VividVRPipelineConfig())

        self.assertEqual(
            utils.resolve_video_encoding_mode(server_args),
            utils.VIDEO_ENCODING_MODE_REFERENCE_PROFILE,
        )
        self.assertEqual(
            utils.resolve_video_encoding_quality(
                server_args=server_args,
                output_compression=None,
            ),
            8,
        )

    def test_non_vividvr_uses_reference_profile_defaults(self):
        server_args = SimpleNamespace(pipeline_config=object())

        self.assertEqual(
            utils.resolve_video_encoding_mode(server_args),
            utils.VIDEO_ENCODING_MODE_REFERENCE_PROFILE,
        )
        self.assertEqual(
            utils.resolve_video_encoding_quality(
                server_args=server_args,
                output_compression=None,
            ),
            5,
        )

    def test_explicit_compression_overrides_default_quality(self):
        server_args = SimpleNamespace(pipeline_config=VividVRPipelineConfig())

        self.assertEqual(
            utils.resolve_video_encoding_quality(
                server_args=server_args,
                output_compression=90,
            ),
            9,
        )

    def test_post_process_sample_uses_reference_writer_for_vividvr_default(self):
        sample = np.zeros((3, 2, 8, 8), dtype=np.float32)

        with patch(
            "sglang.multimodal_gen.runtime.videoedit.io.save_video_frames"
        ) as mock_save_video_frames:
            with patch(
                "sglang.multimodal_gen.runtime.videoedit.ffmpeg_io.save_video_frames_like_reference"
            ) as mock_save_like_reference:
                utils.post_process_sample(
                    sample,
                    DataType.VIDEO,
                    fps=8,
                    save_output=True,
                    save_file_path="/tmp/vividvr_reference_writer.mp4",
                    video_reference_path="/tmp/reference.mp4",
                    video_encoding_mode=utils.VIDEO_ENCODING_MODE_REFERENCE_PROFILE,
                    default_video_quality=8,
                )

        mock_save_video_frames.assert_not_called()
        mock_save_like_reference.assert_called_once()
        _, kwargs = mock_save_like_reference.call_args
        self.assertEqual(kwargs["fps"], 8)
        self.assertIsNone(kwargs["quality"])
        self.assertEqual(kwargs["refer_file"], "/tmp/reference.mp4")

    def test_post_process_sample_uses_vividvr_original_writer_when_explicit(self):
        sample = np.zeros((3, 2, 8, 8), dtype=np.float32)

        with patch(
            "sglang.multimodal_gen.runtime.videoedit.io.save_video_frames"
        ) as mock_save_video_frames:
            with patch(
                "sglang.multimodal_gen.runtime.videoedit.ffmpeg_io.save_video_frames_like_reference"
            ) as mock_save_like_reference:
                utils.post_process_sample(
                    sample,
                    DataType.VIDEO,
                    fps=8,
                    save_output=True,
                    save_file_path="/tmp/vividvr_original_writer.mp4",
                    video_reference_path="/tmp/reference.mp4",
                    video_encoding_mode=utils.VIDEO_ENCODING_MODE_VIVIDVR_ORIGINAL,
                    default_video_quality=8,
                )

        mock_save_video_frames.assert_called_once()
        _, kwargs = mock_save_video_frames.call_args
        self.assertEqual(kwargs["fps"], 8)
        self.assertEqual(kwargs["quality"], 8)
        mock_save_like_reference.assert_not_called()


if __name__ == "__main__":
    unittest.main()
