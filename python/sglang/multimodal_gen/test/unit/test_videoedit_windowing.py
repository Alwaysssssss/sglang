import types
import unittest

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.runtime.pipelines.wan_videoedit_pipeline import (
    WanVideoEditPipeline,
)
from sglang.multimodal_gen.runtime.videoedit.preprocess import prepare_window_inputs
from sglang.multimodal_gen.runtime.videoedit.windowing import (
    build_videoedit_window_specs,
)


def _native_window_starts(num_frames: int, infer_len: int, overlap: int) -> list[int]:
    stride = infer_len - overlap
    if num_frames <= infer_len:
        return [0]
    starts = [0]
    next_start = stride
    while next_start + overlap < num_frames:
        starts.append(next_start)
        next_start += stride
    return starts


def _rgb_frame(color: tuple[int, int, int], size: tuple[int, int] = (4, 4)) -> Image.Image:
    return Image.new("RGB", size, color)


def _mask(value: int, size: tuple[int, int] = (4, 4)) -> Image.Image:
    return Image.new("L", size, value)


class TestVideoEditWindowing(unittest.TestCase):
    def test_window_starts_match_native_infer(self):
        cases = [
            (81, 81, 0),
            (156, 81, 0),
            (156, 81, 5),
            (162, 81, 5),
            (162, 81, 10),
        ]
        for num_frames, infer_len, overlap in cases:
            with self.subTest(num_frames=num_frames, overlap=overlap):
                specs = build_videoedit_window_specs(
                    num_frames=num_frames,
                    infer_len=infer_len,
                    overlap=overlap,
                )
                self.assertEqual(
                    [spec.start_index for spec in specs],
                    _native_window_starts(num_frames, infer_len, overlap),
                )

    def test_non_first_window_records_native_reference_contract(self):
        specs = build_videoedit_window_specs(num_frames=156, infer_len=81, overlap=5)

        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].stride, 76)
        self.assertIsNone(specs[0].reference_prev_local_idx)
        self.assertIsNone(specs[0].reference_global_index)
        self.assertEqual(specs[0].overlap_mask_zero_count, 0)
        self.assertEqual(specs[0].commit_start_local_idx, 0)

        self.assertEqual(specs[1].start_index, 76)
        self.assertEqual(specs[1].stride, 76)
        self.assertEqual(specs[1].reference_prev_local_idx, 76)
        self.assertEqual(specs[1].reference_global_index, 76)
        self.assertEqual(specs[1].overlap_mask_zero_count, 5)
        self.assertEqual(specs[1].commit_start_local_idx, 5)

    def test_overlap_zero_keeps_commit_start_zero(self):
        specs = build_videoedit_window_specs(num_frames=156, infer_len=81, overlap=0)

        self.assertEqual([spec.start_index for spec in specs], [0, 81])
        self.assertIsNone(specs[1].reference_prev_local_idx)
        self.assertIsNone(specs[1].reference_global_index)
        self.assertEqual(specs[1].overlap_mask_zero_count, 0)
        self.assertEqual(specs[1].commit_start_local_idx, 0)

    def test_weighted_window_uses_one_reference_frame_before_edit_span(self):
        specs = build_videoedit_window_specs(
            num_frames=156,
            infer_len=81,
            overlap=10,
            overlap_commit_mode="weighted",
        )

        self.assertEqual([spec.start_index for spec in specs], [0, 71, 141])
        self.assertEqual(specs[1].input_indices[:12], [70] + list(range(71, 82)))
        self.assertEqual(specs[1].reference_prev_local_idx, 70)
        self.assertEqual(specs[1].reference_global_index, 70)
        self.assertEqual(specs[1].overlap_mask_zero_count, 1)
        self.assertEqual(specs[1].commit_start_local_idx, 1)
        self.assertEqual(specs[1].commit_local_to_global[1], 71)
        self.assertEqual(specs[1].commit_local_to_global[80], 150)

    def test_tail_padding_modes_are_preserved(self):
        native_specs = build_videoedit_window_specs(
            num_frames=162,
            infer_len=81,
            overlap=5,
            tail_padding_mode="native_reverse_mirror",
        )
        reflect_specs = build_videoedit_window_specs(
            num_frames=162,
            infer_len=81,
            overlap=5,
            tail_padding_mode="reflect",
        )

        self.assertEqual(native_specs[-1].start_index, 152)
        self.assertEqual(reflect_specs[-1].start_index, 152)
        self.assertEqual(
            native_specs[-1].input_indices[:12],
            list(range(152, 162)) + [161, 160],
        )
        self.assertEqual(
            reflect_specs[-1].input_indices[:12],
            list(range(152, 162)) + [160, 159],
        )

    def test_tail_padding_default_uses_standard_reflect(self):
        default_specs = build_videoedit_window_specs(
            num_frames=162,
            infer_len=81,
            overlap=5,
        )

        self.assertEqual(default_specs[-1].start_index, 152)
        self.assertEqual(
            default_specs[-1].input_indices[:12],
            list(range(152, 162)) + [160, 159],
        )

    def test_materialize_window_uses_previous_stride_frame_as_reference(self):
        pipeline = object.__new__(WanVideoEditPipeline)
        params = types.SimpleNamespace(
            runtime_frame_provider=None,
            runtime_resized_frames=[
                _rgb_frame((idx % 256, 1, 2)) for idx in range(156)
            ],
            runtime_resized_masks=[_mask(255) for _ in range(156)],
            use_repaired_context=False,
            runtime_accum_frames=[
                np.zeros((4, 4, 3), dtype=np.float32) for _ in range(156)
            ],
            runtime_accum_weights=np.zeros((156,), dtype=np.float32),
            runtime_num_input_frames=156,
            runtime_prev_window_index=0,
            runtime_prev_window_output_frames=[
                _rgb_frame((200, local_idx, 0)) for local_idx in range(81)
            ],
            runtime_window_materialize_metadata=[],
            overlap_commit_mode="native_skip",
            overlap=5,
            infer_len=81,
        )
        window_spec = build_videoedit_window_specs(
            num_frames=156,
            infer_len=81,
            overlap=5,
        )[1]

        pipeline._materialize_window_inputs(params, window_spec)

        self.assertEqual(
            np.asarray(params.runtime_window_frames[0])[0, 0].tolist(),
            [200, 76, 0],
        )
        self.assertEqual(
            np.asarray(params.runtime_window_frames[1])[0, 0].tolist(),
            [77, 1, 2],
        )
        for local_idx in range(5):
            self.assertEqual(int(np.asarray(params.runtime_window_masks[local_idx]).sum()), 0)
        self.assertGreater(int(np.asarray(params.runtime_window_masks[5]).sum()), 0)

        metadata = params.runtime_window_materialize_metadata[0]
        self.assertTrue(metadata["reference_from_previous_window"])
        self.assertEqual(metadata["reference_prev_local_idx"], 76)
        self.assertEqual(metadata["reference_global_index"], 76)
        self.assertEqual(metadata["zeroed_overlap_mask_count"], 5)
        self.assertEqual(metadata["commit_start_local_idx"], 5)

    def test_weighted_materialize_uses_previous_frame_as_single_reference(self):
        pipeline = object.__new__(WanVideoEditPipeline)
        params = types.SimpleNamespace(
            runtime_frame_provider=None,
            runtime_resized_frames=[
                _rgb_frame((idx % 256, 1, 2)) for idx in range(156)
            ],
            runtime_resized_masks=[_mask(255) for _ in range(156)],
            use_repaired_context=False,
            runtime_accum_frames=[
                np.zeros((4, 4, 3), dtype=np.float32) for _ in range(156)
            ],
            runtime_accum_weights=np.zeros((156,), dtype=np.float32),
            runtime_num_input_frames=156,
            runtime_prev_window_index=0,
            runtime_prev_window_output_frames=[
                _rgb_frame((200, local_idx, 0)) for local_idx in range(81)
            ],
            runtime_window_materialize_metadata=[],
            overlap_commit_mode="weighted",
            overlap=10,
            infer_len=81,
        )
        window_spec = build_videoedit_window_specs(
            num_frames=156,
            infer_len=81,
            overlap=10,
            overlap_commit_mode="weighted",
        )[1]

        pipeline._materialize_window_inputs(params, window_spec)

        self.assertEqual(
            np.asarray(params.runtime_window_frames[0])[0, 0].tolist(),
            [200, 70, 0],
        )
        self.assertEqual(
            np.asarray(params.runtime_window_frames[1])[0, 0].tolist(),
            [71, 1, 2],
        )
        self.assertEqual(int(np.asarray(params.runtime_window_masks[0]).sum()), 0)
        self.assertGreater(int(np.asarray(params.runtime_window_masks[1]).sum()), 0)

        metadata = params.runtime_window_materialize_metadata[0]
        self.assertTrue(metadata["reference_from_previous_window"])
        self.assertEqual(metadata["reference_prev_local_idx"], 70)
        self.assertEqual(metadata["reference_global_index"], 70)
        self.assertEqual(metadata["zeroed_overlap_mask_count"], 1)
        self.assertEqual(metadata["commit_start_local_idx"], 1)

    def test_commit_window_output_skips_native_overlap_prefix(self):
        pipeline = object.__new__(WanVideoEditPipeline)
        window_spec = build_videoedit_window_specs(
            num_frames=156,
            infer_len=81,
            overlap=5,
        )[1]
        params = types.SimpleNamespace(
            runtime_window_output_frames=[
                _rgb_frame((local_idx, 0, 0)) for local_idx in range(81)
            ],
            runtime_num_input_frames=156,
            runtime_accum_frames=[
                np.zeros((4, 4, 3), dtype=np.float32) for _ in range(156)
            ],
            runtime_accum_weights=np.zeros((156,), dtype=np.float32),
            runtime_prev_window_output_frames=None,
            runtime_prev_window_index=None,
            overlap_commit_mode="native_skip",
            overlap=5,
        )

        pipeline._commit_window_output(params, window_spec)

        self.assertTrue(np.all(params.runtime_accum_weights[76:81] == 0))
        self.assertEqual(params.runtime_accum_weights[81], 1.0)
        self.assertEqual(params.runtime_accum_weights[155], 1.0)
        self.assertEqual(np.asarray(params.runtime_prev_window_output_frames[0])[0, 0, 0], 0)
        self.assertEqual(params.runtime_prev_window_index, 1)

    def test_commit_window_output_weighted_skips_reference_and_blends_overlap(self):
        pipeline = object.__new__(WanVideoEditPipeline)
        window_spec = build_videoedit_window_specs(
            num_frames=156,
            infer_len=81,
            overlap=10,
            overlap_commit_mode="weighted",
        )[1]
        params = types.SimpleNamespace(
            runtime_window_output_frames=[
                _rgb_frame((local_idx, 0, 0)) for local_idx in range(81)
            ],
            runtime_num_input_frames=156,
            runtime_window_specs=build_videoedit_window_specs(
                num_frames=156,
                infer_len=81,
                overlap=10,
                overlap_commit_mode="weighted",
            ),
            runtime_accum_frames=[
                np.zeros((4, 4, 3), dtype=np.float32) for _ in range(156)
            ],
            runtime_accum_weights=np.zeros((156,), dtype=np.float32),
            runtime_prev_window_output_frames=None,
            runtime_prev_window_index=None,
            overlap_commit_mode="weighted",
            overlap=10,
            infer_len=81,
        )

        pipeline._commit_window_output(params, window_spec)

        self.assertEqual(params.runtime_accum_weights[70], 0.0)
        self.assertAlmostEqual(params.runtime_accum_weights[71], 1.0 / 11.0)
        self.assertAlmostEqual(params.runtime_accum_weights[80], 10.0 / 11.0)
        self.assertEqual(params.runtime_accum_weights[81], 1.0)
        self.assertAlmostEqual(params.runtime_accum_weights[141], 10.0 / 11.0)
        self.assertAlmostEqual(params.runtime_accum_weights[150], 1.0 / 11.0)
        self.assertEqual(params.runtime_accum_weights[151], 0.0)
        self.assertEqual(np.asarray(params.runtime_prev_window_output_frames[0])[0, 0, 0], 0)
        self.assertEqual(params.runtime_prev_window_index, 1)

    def test_non_first_window_tensor_preserves_reference_when_overlap_mask_is_black(self):
        reference = _rgb_frame((100, 50, 25), size=(16, 16))
        window_video = [reference] + [
            _rgb_frame((idx, 10, 20), size=(16, 16)) for idx in range(1, 5)
        ]
        window_masks = [_mask(0, size=(16, 16)), _mask(0, size=(16, 16))] + [
            _mask(255, size=(16, 16)) for _ in range(3)
        ]

        prepared = prepare_window_inputs(
            window_video,
            window_masks,
            device="cpu",
            dtype=torch.float32,
            preserve_first_frame=False,
        )

        expected = torch.from_numpy(np.asarray(reference).astype(np.float32))
        expected = expected.permute(2, 0, 1) / 127.5 - 1.0
        self.assertTrue(torch.allclose(prepared["masked_video_tensor"][0], expected))
        self.assertEqual(float(prepared["mask_video_tensor"][0].sum()), 0.0)
        self.assertTrue(torch.all(prepared["cond_masks"][:, :, 0] == 1.0))


if __name__ == "__main__":
    unittest.main()
