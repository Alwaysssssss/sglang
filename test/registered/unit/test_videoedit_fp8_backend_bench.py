import importlib.util
import json
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "videoedit_l20_fp8_backend_bench.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("videoedit_fp8_backend_bench", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_shapes_filters_deduplicates_and_sorts(tmp_path):
    module = load_module()
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            [
                {
                    "method": "Fp8LinearMethod",
                    "kernel": "dynamic_per_token_fp8+sgl_kernel.fp8_scaled_mm",
                    "m": 100,
                    "k": 64,
                    "n": 128,
                },
                {
                    "method": "Fp8LinearMethod",
                    "kernel": "dynamic_per_token_fp8+sgl_kernel.fp8_scaled_mm",
                    "m": 100,
                    "k": 64,
                    "n": 128,
                },
                {
                    "method": "Fp8LinearMethod",
                    "kernel": "dynamic_per_token_fp8+sgl_kernel.fp8_scaled_mm",
                    "m": 200,
                    "k": 128,
                    "n": 256,
                },
                {
                    "method": "UnquantizedLinearMethod",
                    "kernel": "torch.nn.functional.linear",
                    "m": 1000,
                    "k": 512,
                    "n": 512,
                },
            ]
        ),
        encoding="utf-8",
    )

    shapes = module.load_shapes(audit, max_m=150, max_shapes=2)

    assert shapes == [
        {"original_m": 200, "m": 150, "k": 128, "n": 256},
        {"original_m": 100, "m": 100, "k": 64, "n": 128},
    ]
