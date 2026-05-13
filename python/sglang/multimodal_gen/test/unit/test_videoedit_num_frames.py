import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sglang.multimodal_gen.configs.sample.videoedit_wan import (
    WanVideoEditSamplingParams,
)
from sglang.multimodal_gen.runtime.videoedit.preprocess import (
    resolve_videoedit_num_frames,
)


def _write_test_video(path: Path, frame_count: int) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (8, 8),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not open test video writer")
    try:
        for i in range(frame_count):
            frame = np.full((8, 8, 3), i % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


class TestVideoEditNumFrames(unittest.TestCase):
    def test_positive_num_frames_is_returned_without_probing(self):
        self.assertEqual(
            resolve_videoedit_num_frames(81, "missing_video.avi", "missing_mask.avi"),
            81,
        )

    def test_minus_one_resolves_to_all_available_frames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video = Path(temp_dir) / "video.avi"
            mask = Path(temp_dir) / "mask.avi"
            _write_test_video(video, 12)
            _write_test_video(mask, 12)

            self.assertEqual(resolve_videoedit_num_frames(-1, str(video), str(mask)), 12)

    def test_minus_one_uses_shorter_video_or_mask(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video = Path(temp_dir) / "video.avi"
            mask = Path(temp_dir) / "mask.avi"
            _write_test_video(video, 12)
            _write_test_video(mask, 7)

            self.assertEqual(resolve_videoedit_num_frames(-1, str(video), str(mask)), 7)

    def test_minus_one_missing_video_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            resolve_videoedit_num_frames(-1, "missing_video.avi", "missing_mask.avi")

    def test_sampling_params_rejects_unresolved_minus_one(self):
        with self.assertRaisesRegex(ValueError, "API/CLI entrypoints"):
            WanVideoEditSamplingParams(num_frames=-1)


if __name__ == "__main__":
    unittest.main()
