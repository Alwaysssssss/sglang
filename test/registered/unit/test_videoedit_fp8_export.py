# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "videoedit_export_fp8_checkpoint.py"
)


def load_export_module():
    spec = importlib.util.spec_from_file_location("videoedit_fp8_export", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_expected_videoedit_linear_weight_count():
    module = load_export_module()

    names = module.expected_linear_weight_names(40)

    assert len(names) == 488
    assert "blocks.0.ffn.net.0.proj.weight" in names
    assert "blocks.39.attn2.add_v_proj.weight" in names
    assert "condition_embedder.time_proj.weight" in names
    assert "proj_out.weight" in names
    assert "patch_embedding.weight" not in names


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_export_quantization_matches_runtime_kernel():
    module = load_export_module()
    from sglang.srt.layers.quantization.fp8_kernel import (
        per_token_group_quant_fp8,
    )

    torch.manual_seed(7)
    weight = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)

    actual_weight, actual_scale = module.quantize_weight_per_channel(weight)
    expected_weight, expected_scale = per_token_group_quant_fp8(
        weight, weight.shape[-1]
    )

    assert torch.equal(actual_weight, expected_weight)
    assert torch.equal(actual_scale, expected_scale.reshape(-1))


def test_videoedit_export_writes_serialized_channel_checkpoint(tmp_path):
    module = load_export_module()
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    config = {
        "_class_name": "WanVideoEditTransformer3DModel",
        "num_layers": 1,
    }
    (input_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    expected_names = module.expected_linear_weight_names(1)
    tensors = {
        name: torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.bfloat16)
        for name in expected_names
    }
    tensors["blocks.0.attn1.norm_q.weight"] = torch.ones(2, dtype=torch.bfloat16)
    shard_name = "diffusion_pytorch_model-00001-of-00001.safetensors"
    save_file(tensors, input_dir / shard_name, metadata={"format": "pt"})
    index = {
        "metadata": {
            "total_size": sum(
                tensor.numel() * tensor.element_size() for tensor in tensors.values()
            )
        },
        "weight_map": {name: shard_name for name in tensors},
    }
    (input_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )

    manifest = module.convert_checkpoint(
        input_dir,
        output_dir,
        device="cpu",
        compute_checksums=False,
        skip_space_check=True,
    )

    assert manifest["quantized_linear_weight_count"] == 20
    output_config = json.loads((output_dir / "config.json").read_text())
    assert output_config["quantization_config"] == {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_scale_granularity": "channel",
    }
    with safe_open(output_dir / shard_name, framework="pt", device="cpu") as output:
        weight = output.get_tensor("blocks.0.ffn.net.0.proj.weight")
        scale = output.get_tensor("blocks.0.ffn.net.0.proj.weight_scale")
        norm = output.get_tensor("blocks.0.attn1.norm_q.weight")
        assert weight.dtype == torch.float8_e4m3fn
        assert scale.shape == (2,)
        assert norm.dtype == torch.bfloat16
        reconstructed = weight.float() * scale[:, None]
        torch.testing.assert_close(
            reconstructed,
            tensors["blocks.0.ffn.net.0.proj.weight"].float(),
            rtol=0.05,
            atol=0.05,
        )
