# SPDX-License-Identifier: Apache-2.0
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from sglang.multimodal_gen.runtime.vsr.verify.compare_tensors import compare


class TensorGateTest(unittest.TestCase):
    def test_empty_dump_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(compare(root / "ref", root / "cand")["structural_pass"])

    def test_nonfinite_tensor_fails_without_numeric_gate(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for side in ["ref", "cand"]:
                path = root / side / "latent"
                path.mkdir(parents=True)
                torch.save(torch.tensor([float("inf")]), path / "tile_00000.pt")
            self.assertFalse(compare(root / "ref", root / "cand")["structural_pass"])


if __name__ == "__main__":
    unittest.main()
