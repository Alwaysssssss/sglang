# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from sglang.multimodal_gen.runtime.layers.linear import LinearBase
from sglang.multimodal_gen.runtime.utils.quantization_audit import (
    build_quantization_audit,
)


class Fp8LinearMethod:
    def runtime_kernel_name(self, layer):
        return "dynamic_per_token_fp8+sgl_kernel.fp8_scaled_mm"


class UnquantizedLinearMethod:
    pass


class FakeLinear(LinearBase):
    def __init__(
        self,
        *,
        method,
        weight_dtype: torch.dtype,
        fp8_layout: bool,
    ):
        torch.nn.Module.__init__(self)
        self.quant_method = method
        self.prefix = "fake"
        self.output_size = 32
        self.output_size_per_partition = 32
        shape = (16, 32) if fp8_layout else (32, 16)
        self.weight = torch.nn.Parameter(
            torch.empty(shape, dtype=weight_dtype),
            requires_grad=False,
        )
        if fp8_layout:
            self.weight_scale = torch.nn.Parameter(
                torch.ones(32, dtype=torch.float32),
                requires_grad=False,
            )


class TestDiffusionQuantizationAudit(unittest.TestCase):
    def test_reports_fp8_and_unquantized_linears(self):
        model = torch.nn.Module()
        model.fp8 = FakeLinear(
            method=Fp8LinearMethod(),
            weight_dtype=torch.float8_e4m3fn,
            fp8_layout=True,
        )
        model.bf16 = FakeLinear(
            method=UnquantizedLinearMethod(),
            weight_dtype=torch.bfloat16,
            fp8_layout=False,
        )

        audit = build_quantization_audit(
            model,
            component="transformer",
            rank=0,
        )

        self.assertEqual(audit["linear_total"], 2)
        self.assertEqual(audit["fp8_method_count"], 1)
        self.assertEqual(audit["fp8_weight_count"], 1)
        self.assertEqual(audit["predicted_true_w8a8_count"], 1)
        self.assertEqual(
            audit["predicted_kernel_route_counts"],
            {
                "dynamic_per_token_fp8+sgl_kernel.fp8_scaled_mm": 1,
                "torch.nn.functional.linear": 1,
            },
        )
        self.assertEqual(audit["unquantized_linear_names"], ["bf16"])
        self.assertEqual(audit["fp8_weight_dtype_mismatch_names"], [])


if __name__ == "__main__":
    unittest.main()
