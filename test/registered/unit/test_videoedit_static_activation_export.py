from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from sglang.multimodal_gen.runtime.layers.linear import MergedColumnParallelLinear
from sglang.multimodal_gen.runtime.models.parameter import PerTensorScaleParameter
from sglang.multimodal_gen.runtime.layers.layernorm import _ensure_contiguous
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config
from safetensors import safe_open
from sglang.multimodal_gen.runtime.loader.utils import (
    get_param_names_mapping,
    hf_to_custom_state_dict,
)
from sglang.multimodal_gen.runtime.models.dits.wanvideo import (
    build_wan_fused_projection_mapping,
)
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.utils.activation_calibration import (
    FP8_E4M3_MAX,
    checkpoint_aliases_for_runtime_linear,
)


def load_exporter_module():
    script_path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "videoedit_export_static_activation_checkpoint.py"
    )
    spec = importlib.util.spec_from_file_location(
        "videoedit_export_static_activation_checkpoint", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def runtime_module_names() -> list[str]:
    return [
        "condition_embedder.time_embedder.mlp.fc_in",
        "condition_embedder.time_embedder.mlp.fc_out",
        "condition_embedder.time_modulation.linear",
        "condition_embedder.text_embedder.fc_in",
        "condition_embedder.text_embedder.fc_out",
        "condition_embedder.image_embedder.ff.fc_in",
        "condition_embedder.image_embedder.ff.fc_out",
        "blocks.0.to_qkv",
        "blocks.0.to_out",
        "blocks.0.attn2.to_q",
        "blocks.0.attn2.to_kv",
        "blocks.0.attn2.to_out",
        "blocks.0.attn2.to_added_kv",
        "blocks.0.ffn.fc_in",
        "blocks.0.ffn.fc_out",
        "proj_out",
    ]


def write_source_checkpoint(root: Path, exporter) -> None:
    root.mkdir()
    quantization = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_scale_granularity": "channel",
    }
    (root / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "WanVideoEditTransformer3DModel",
                "num_layers": 1,
                "quantization_config": quantization,
            }
        ),
        encoding="utf-8",
    )
    tensors: dict[str, torch.Tensor] = {}
    for weight_name in exporter.expected_linear_weight_names(1):
        prefix = weight_name.removesuffix(".weight")
        tensors[weight_name] = torch.ones((2, 3), dtype=torch.float8_e4m3fn)
        tensors[f"{prefix}.weight_scale"] = torch.ones(2, dtype=torch.float32)
    shard_name = "diffusion_pytorch_model-00001-of-00001.safetensors"
    save_file(tensors, root / shard_name)
    (root / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": sum(
                        tensor.numel() * tensor.element_size()
                        for tensor in tensors.values()
                    )
                },
                "weight_map": {name: shard_name for name in tensors},
            }
        ),
        encoding="utf-8",
    )


def write_calibration(path: Path) -> None:
    modules = []
    for name in runtime_module_names():
        modules.append(
            {
                "name": name,
                "checkpoint_aliases": checkpoint_aliases_for_runtime_linear(name),
                "input_features": 3,
                "observation_count": 20,
                "token_count": 100,
                "nonfinite_token_count": 0,
                "absmax": 4.0,
                "candidate_thresholds": {"max": 4.0},
                "candidate_scales": {"max": 4.0 / FP8_E4M3_MAX},
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "completed_requests": ["calib_case"],
                "quantization": {
                    "dtype": "fp8_e4m3fn",
                    "symmetric": True,
                    "zero_point": None,
                    "fp8_max": FP8_E4M3_MAX,
                    "granularity": "per_runtime_linear",
                },
                "modules": modules,
            }
        ),
        encoding="utf-8",
    )


def test_export_adds_static_scales_without_rewriting_weights(tmp_path):
    exporter = load_exporter_module()
    input_dir = tmp_path / "dynamic"
    output_dir = tmp_path / "static"
    calibration_path = tmp_path / "activation_calibration.json"
    write_source_checkpoint(input_dir, exporter)
    write_calibration(calibration_path)

    manifest = exporter.export_static_checkpoint(
        input_dir,
        calibration_path,
        output_dir,
        candidate="max",
        safety_factor=1.05,
        copy_mode="copy",
        skip_space_check=True,
    )

    assert manifest["runtime_linear_count"] == 16
    assert manifest["checkpoint_input_scale_count"] == 20
    output_config = json.loads((output_dir / "config.json").read_text())
    assert output_config["quantization_config"]["activation_scheme"] == "static"
    output_index = json.loads(
        (output_dir / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )
    assert output_index["weight_map"]["blocks.0.attn1.to_q.input_scale"] == (
        exporter.SCALE_SHARD_NAME
    )
    with safe_open(
        output_dir / exporter.SCALE_SHARD_NAME, framework="pt", device="cpu"
    ) as scales:
        assert len(list(scales.keys())) == 20
        value = scales.get_tensor("blocks.0.attn1.to_q.input_scale")
        assert value.shape == torch.Size([1])
        assert value.item() == pytest.approx(4.0 / FP8_E4M3_MAX * 1.05)
    source_shard = input_dir / "diffusion_pytorch_model-00001-of-00001.safetensors"
    output_shard = output_dir / source_shard.name
    assert source_shard.read_bytes() == output_shard.read_bytes()


def test_fused_qkv_input_scales_are_concatenated_by_fsdp_loader():
    mapping = get_param_names_mapping(build_wan_fused_projection_mapping({}))
    state_dict, _ = hf_to_custom_state_dict(
        {
            "blocks.0.attn1.to_q.input_scale": torch.tensor([1.0]),
            "blocks.0.attn1.to_k.input_scale": torch.tensor([2.0]),
            "blocks.0.attn1.to_v.input_scale": torch.tensor([3.0]),
        },
        mapping,
    )

    assert state_dict["blocks.0.to_qkv.input_scale"].shape == torch.Size([3])
    assert state_dict["blocks.0.to_qkv.input_scale"].tolist() == [
        1.0,
        2.0,
        3.0,
    ]

    fused_scale = state_dict["blocks.0.to_qkv.input_scale"]
    parameter = PerTensorScaleParameter(
        data=torch.empty(3, dtype=torch.float32),
        weight_loader=lambda *_args, **_kwargs: None,
    )
    MergedColumnParallelLinear.weight_loader_v2(object(), parameter, fused_scale)
    assert parameter.tolist() == [1.0, 2.0, 3.0]


def test_merged_static_fp8_linear_allocates_one_scale_per_logical_matrix():
    quantization = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="static",
        weight_scale_granularity="channel",
        gemm_backend="triton",
    )
    with (
        patch(
            "sglang.multimodal_gen.runtime.layers.linear.get_tp_group",
            return_value=object(),
        ),
        patch(
            "sglang.multimodal_gen.runtime.layers.linear.get_group_size",
            return_value=1,
        ),
        patch(
            "sglang.multimodal_gen.runtime.layers.linear.get_group_rank",
            return_value=0,
        ),
        patch(
            "sglang.multimodal_gen.runtime.layers.quantization.fp8.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
    ):
        layer = MergedColumnParallelLinear(
            4, [4, 4], bias=False, quant_config=quantization
        )

    assert layer.output_partition_sizes == [4, 4]
    assert layer.input_scale.shape == torch.Size([2])


def test_fused_norm_inputs_are_reallocated_when_data_pointer_is_misaligned():
    storage = torch.arange(33, dtype=torch.float32)
    misaligned = storage[1:]
    assert misaligned.is_contiguous()
    assert misaligned.data_ptr() % 32 != 0

    aligned = _ensure_contiguous(misaligned)

    assert aligned.data_ptr() % 32 == 0
    torch.testing.assert_close(aligned, misaligned)
