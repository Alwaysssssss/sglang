import unittest
from unittest.mock import patch

import torch

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


if __name__ == "__main__":
    unittest.main()
