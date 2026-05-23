import unittest

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.star_cogvideox_sr_decoding import (
    STARCogVideoXSRDecodingStage,
)


class TestStarDecodingWindows(unittest.TestCase):
    def test_reference_window_layout_for_common_lengths(self):
        self.assertEqual(
            STARCogVideoXSRDecodingStage.build_decode_windows(7),
            [(0, 3, False), (3, 5, False), (5, 7, True)],
        )
        self.assertEqual(
            STARCogVideoXSRDecodingStage.build_decode_windows(9),
            [(0, 3, False), (3, 5, False), (5, 7, False), (7, 9, True)],
        )
        self.assertEqual(
            STARCogVideoXSRDecodingStage.build_decode_windows(11),
            [
                (0, 3, False),
                (3, 5, False),
                (5, 7, False),
                (7, 9, False),
                (9, 11, True),
            ],
        )
        self.assertEqual(
            STARCogVideoXSRDecodingStage.build_decode_windows(13),
            [
                (0, 3, False),
                (3, 5, False),
                (5, 7, False),
                (7, 9, False),
                (9, 11, False),
                (11, 13, True),
            ],
        )

    def test_noncanonical_lengths_still_cover_full_sequence(self):
        self.assertEqual(
            STARCogVideoXSRDecodingStage.build_decode_windows(1),
            [(0, 1, True)],
        )
        self.assertEqual(
            STARCogVideoXSRDecodingStage.build_decode_windows(2),
            [(0, 2, True)],
        )
        self.assertEqual(
            STARCogVideoXSRDecodingStage.build_decode_windows(4),
            [(0, 3, False), (3, 4, True)],
        )
        self.assertEqual(
            STARCogVideoXSRDecodingStage.build_decode_windows(6),
            [(0, 3, False), (3, 5, False), (5, 6, True)],
        )

    def test_invalid_length_raises(self):
        with self.assertRaises(ValueError):
            STARCogVideoXSRDecodingStage.build_decode_windows(0)


if __name__ == "__main__":
    unittest.main()
