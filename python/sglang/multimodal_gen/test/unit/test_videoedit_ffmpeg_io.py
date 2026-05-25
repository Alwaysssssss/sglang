import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from sglang.multimodal_gen.runtime.videoedit.ffmpeg_io import (
    probe_video_profile,
    save_video_frames_like_reference,
)


def _write_reference_video(path: Path, frame_count: int = 3) -> bool:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (8, 8),
    )
    if not writer.isOpened():
        return False
    try:
        for i in range(frame_count):
            frame = np.full((8, 8, 3), i * 40, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return True


class TestVideoEditFfmpegIO(unittest.TestCase):
    def setUp(self):
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg/ffprobe is not available")

    def test_save_video_frames_like_reference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ref_path = Path(temp_dir) / "reference.mp4"
            out_path = Path(temp_dir) / "output.mp4"
            if not _write_reference_video(ref_path):
                self.skipTest("OpenCV could not create reference mp4")

            ref_profile = probe_video_profile(str(ref_path))
            frames = [
                Image.fromarray(np.full((8, 8, 3), i * 30, dtype=np.uint8))
                for i in range(3)
            ]
            save_video_frames_like_reference(
                frames,
                str(out_path),
                refer_file=str(ref_path),
                fps=ref_profile["fps"],
            )
            out_profile = probe_video_profile(str(out_path))

            self.assertEqual(out_profile["codec_name"], ref_profile["codec_name"])
            self.assertEqual(round(out_profile["fps"]), round(ref_profile["fps"]))
            self.assertEqual(out_profile["width"], 8)
            self.assertEqual(out_profile["height"], 8)


if __name__ == "__main__":
    unittest.main()
