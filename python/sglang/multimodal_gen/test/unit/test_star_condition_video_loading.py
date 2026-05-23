import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from sglang.multimodal_gen.configs.sample.star_cogvideox_sr import (
    StarCogVideoXSRSamplingParams,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.video_condition_loading import (
    STARConditionVideoLoadingStage,
)

_GLOBAL_ARGS_PATCH = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args"
)


class TestSTARConditionVideoLoadingStage(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.video_path = str(Path(self.tempdir.name) / "condition.gif")
        self._write_test_gif(self.video_path)

        with patch(_GLOBAL_ARGS_PATCH, return_value=MagicMock()):
            self.stage = STARConditionVideoLoadingStage()

    def tearDown(self):
        self.tempdir.cleanup()

    @staticmethod
    def _write_test_gif(path: str) -> None:
        frames = [
            Image.fromarray(
                np.full((48, 64, 3), fill_value=index * 40, dtype=np.uint8),
                mode="RGB",
            )
            for index in range(5)
        ]
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )

    @staticmethod
    def _make_server_args(
        width=None, height=None, num_frames=None, condition_video_num_frames=None
    ):
        return SimpleNamespace(
            pipeline_config=SimpleNamespace(
                width=width,
                height=height,
                num_frames=num_frames,
                condition_video_num_frames=condition_video_num_frames,
            )
        )

    @staticmethod
    def _make_batch(
        *,
        condition_video_path: str | None,
        explicit_fields: set[str] | None = None,
        **kwargs,
    ) -> Req:
        sampling_params = StarCogVideoXSRSamplingParams(
            prompt="test",
            condition_video_path=condition_video_path,
            **kwargs,
        )
        if explicit_fields is not None:
            sampling_params._explicit_fields = set(explicit_fields)
        batch = Req(sampling_params=sampling_params)
        sampling_params.apply_request_extra(batch)
        return batch

    def test_stage_uses_pipeline_defaults_when_size_and_frames_are_implicit(self):
        batch = self._make_batch(condition_video_path=self.video_path)
        server_args = self._make_server_args(
            width=40,
            height=24,
            num_frames=4,
            condition_video_num_frames=4,
        )

        result = self.stage.forward(batch, server_args)

        self.assertEqual(result.condition_video.shape, (1, 4, 3, 24, 40))
        self.assertEqual(result.original_condition_video_size, (64, 48))
        self.assertAlmostEqual(result.original_condition_video_fps, 10.0)
        self.assertEqual(result.condition_video_indices, [0, 1, 2, 3])
        self.assertEqual(result.condition_video_num_frames, 4)
        self.assertEqual((result.width, result.height), (40, 24))

    def test_stage_prefers_explicit_dimensions_and_num_frames(self):
        batch = self._make_batch(
            condition_video_path=self.video_path,
            explicit_fields={"width", "height", "num_frames", "condition_video_path"},
            width=32,
            height=32,
            num_frames=3,
            condition_video_num_frames=3,
        )
        server_args = self._make_server_args(width=40, height=24, num_frames=4)

        result = self.stage.forward(batch, server_args)

        self.assertEqual(result.condition_video.shape, (1, 3, 3, 32, 32))
        self.assertEqual(result.condition_video_indices, [0, 1, 2])
        self.assertEqual((result.width, result.height), (32, 32))

    def test_stage_uses_condition_video_default_not_output_num_frames(self):
        batch = self._make_batch(condition_video_path=self.video_path, num_frames=3)
        server_args = self._make_server_args(
            width=64,
            height=48,
            num_frames=3,
            condition_video_num_frames=5,
        )

        result = self.stage.forward(batch, server_args)

        self.assertEqual(result.condition_video.shape, (1, 5, 3, 48, 64))
        self.assertEqual(result.condition_video_indices, [0, 1, 2, 3, 4])
        self.assertEqual(result.condition_video_num_frames, 5)

    def test_stage_supports_start_frame_and_stride(self):
        batch = self._make_batch(
            condition_video_path=self.video_path,
            condition_video_start_frame=1,
            condition_video_frame_stride=2,
            condition_video_num_frames=2,
        )
        server_args = self._make_server_args()

        result = self.stage.forward(batch, server_args)

        self.assertEqual(result.condition_video.shape, (1, 2, 3, 48, 64))
        self.assertEqual(result.condition_video_indices, [1, 3])
        self.assertEqual(result.condition_video_num_frames, 2)

    def test_missing_condition_video_path_raises_clear_error(self):
        batch = self._make_batch(condition_video_path=None)
        server_args = self._make_server_args()

        with self.assertRaisesRegex(
            ValueError, "condition_video_path is required"
        ):
            self.stage.forward(batch, server_args)


if __name__ == "__main__":
    unittest.main()
