# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import sglang.multimodal_gen.runtime.layers.quantization.fp8 as fp8_module
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
)
from sglang.multimodal_gen.runtime.models.parameter import (
    ChannelQuantScaleParameter,
)
from sglang.srt.layers.quantization.fp8_utils import (
    normalize_fp8_weight_scale_for_triton,
    resolve_fp8_linear_gemm_backend,
)


@pytest.mark.parametrize(
    ("requested_backend", "tokens", "expected"),
    [
        ("auto", None, "sgl_cutlass"),
        ("sgl_cutlass", 41958, "sgl_cutlass"),
        ("triton", 41958, "triton"),
        ("hybrid", None, "hybrid_shape_dependent"),
        ("hybrid", 512, "sgl_cutlass"),
        ("hybrid", 513, "triton"),
        ("hybrid", 41958, "triton"),
    ],
)
def test_resolve_fp8_linear_gemm_backend(requested_backend, tokens, expected):
    assert (
        resolve_fp8_linear_gemm_backend(
            requested_backend=requested_backend,
            cutlass_fp8_supported=True,
            weight_shape=(5120, 13824),
            weight_scale_numel=13824,
            input_num_tokens=tokens,
            is_cuda_platform=True,
        )
        == expected
    )


def test_auto_preserves_non_cutlass_fallback():
    assert (
        resolve_fp8_linear_gemm_backend(
            requested_backend="auto",
            cutlass_fp8_supported=False,
            weight_shape=(5120, 13824),
            weight_scale_numel=1,
            input_num_tokens=41958,
            is_cuda_platform=True,
        )
        == "non_cutlass"
    )


def test_explicit_cutlass_rejects_incompatible_shape():
    with pytest.raises(ValueError, match="multiples of 16"):
        resolve_fp8_linear_gemm_backend(
            requested_backend="sgl_cutlass",
            cutlass_fp8_supported=True,
            weight_shape=(5119, 13824),
            weight_scale_numel=13824,
            input_num_tokens=32,
            is_cuda_platform=True,
        )


def test_normalize_triton_weight_scale_transposes_logical_layout():
    source = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    normalized = normalize_fp8_weight_scale_for_triton(source, output_size=4)

    assert normalized.shape == (4, 1)
    assert normalized.is_contiguous()
    torch.testing.assert_close(normalized.flatten(), source.flatten())


def test_fp8_config_validates_gemm_backend():
    config = Fp8Config(gemm_backend="hybrid")
    assert config.gemm_backend == "hybrid"

    with pytest.raises(ValueError, match="Unsupported FP8 GEMM backend"):
        Fp8Config(gemm_backend="unknown")


def test_fp8_config_parses_serialized_channel_scales():
    config = Fp8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_scale_granularity": "channel",
        }
    )

    assert config.is_checkpoint_fp8_serialized
    assert config.weight_scale_granularity == "channel"

    with pytest.raises(ValueError, match="weight scale granularity"):
        Fp8Config(weight_scale_granularity="group")


def test_serialized_channel_fp8_preserves_weight_and_scale(monkeypatch):
    monkeypatch.setattr(fp8_module, "get_tensor_model_parallel_world_size", lambda: 1)
    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        gemm_backend="triton",
        weight_scale_granularity="channel",
    )
    method = Fp8LinearMethod(config)
    layer = torch.nn.Module()
    method.create_weights(
        layer=layer,
        input_size_per_partition=4,
        output_partition_sizes=[3],
        input_size=4,
        output_size=3,
        params_dtype=torch.bfloat16,
        weight_loader=lambda param, loaded: param.data.copy_(loaded),
    )

    assert layer.weight.dtype == torch.float8_e4m3fn
    assert isinstance(layer.weight_scale, ChannelQuantScaleParameter)
    assert layer.weight_scale.shape == (3,)

    checkpoint_weight = torch.tensor(
        [[1.0, -2.0, 3.0, -4.0], [2.0, 1.0, 0.0, -1.0], [4.0, 3.0, 2.0, 1.0]],
        dtype=torch.float8_e4m3fn,
    )
    checkpoint_scale = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float32)
    layer.weight.data.copy_(checkpoint_weight)
    layer.weight_scale.data.copy_(checkpoint_scale)

    method.process_weights_after_loading(layer)

    assert layer.weight.shape == (4, 3)
    assert layer.weight_scale.shape == (1, 3)
    torch.testing.assert_close(layer.weight.float(), checkpoint_weight.t().float())
    torch.testing.assert_close(layer.weight_scale.flatten(), checkpoint_scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_apply_fp8_linear_executes_triton_and_hybrid_cuda_paths():
    from sglang.srt.layers.quantization.fp8_kernel import (
        per_token_group_quant_fp8,
    )
    from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

    torch.manual_seed(7)
    device = torch.device("cuda:0")
    m, k, n = 513, 512, 512
    input_tensor = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    source_weight = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    qweight, weight_scale = per_token_group_quant_fp8(source_weight, k)
    weight = qweight.t()
    weight_scale = weight_scale.t().contiguous()

    cutlass_output = apply_fp8_linear(
        input_tensor,
        weight,
        weight_scale,
        cutlass_fp8_supported=True,
        gemm_backend="sgl_cutlass",
    )
    triton_output = apply_fp8_linear(
        input_tensor,
        weight,
        weight_scale,
        cutlass_fp8_supported=True,
        gemm_backend="triton",
    )
    hybrid_output = apply_fp8_linear(
        input_tensor,
        weight,
        weight_scale,
        cutlass_fp8_supported=True,
        gemm_backend="hybrid",
    )

    cosine = torch.nn.functional.cosine_similarity(
        cutlass_output.float().flatten(),
        triton_output.float().flatten(),
        dim=0,
    )
    assert cosine.item() > 0.999
    torch.testing.assert_close(hybrid_output, triton_output)
