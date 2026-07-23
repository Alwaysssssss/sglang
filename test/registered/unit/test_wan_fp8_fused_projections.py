# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.dits.wanvideo import WanVideoConfig
from sglang.multimodal_gen.runtime.loader.utils import (
    get_param_names_mapping,
    hf_to_custom_state_dict,
)
from sglang.multimodal_gen.runtime.models.dits import wanvideo
from sglang.multimodal_gen.runtime.models.dits.wan_videoedit import (
    resolve_videoedit_attention_backend,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


@pytest.mark.parametrize(
    ("source_name", "target_name", "merge_index", "merge_count"),
    [
        (
            "blocks.3.attn1.to_q.weight",
            "blocks.3.to_qkv.weight",
            0,
            3,
        ),
        (
            "blocks.3.attn1.to_k.bias",
            "blocks.3.to_qkv.bias",
            1,
            3,
        ),
        (
            "blocks.3.attn1.to_v.weight",
            "blocks.3.to_qkv.weight",
            2,
            3,
        ),
        (
            "blocks.3.attn2.to_k.weight",
            "blocks.3.attn2.to_kv.weight",
            0,
            2,
        ),
        (
            "blocks.3.attn2.to_v.weight",
            "blocks.3.attn2.to_kv.weight",
            1,
            2,
        ),
        (
            "blocks.3.attn2.add_k_proj.weight",
            "blocks.3.attn2.to_added_kv.weight",
            0,
            2,
        ),
        (
            "blocks.3.attn2.add_v_proj.weight",
            "blocks.3.attn2.to_added_kv.weight",
            1,
            2,
        ),
    ],
)
def test_wan_fused_projection_mapping(
    source_name, target_name, merge_index, merge_count
):
    base_mapping = WanVideoConfig().param_names_mapping
    fused_mapping = wanvideo.build_wan_fused_projection_mapping(base_mapping)
    mapping_fn = get_param_names_mapping(fused_mapping)

    assert mapping_fn(source_name) == (target_name, merge_index, merge_count)
    assert fused_mapping is not base_mapping


def test_wan_fused_qkv_checkpoint_merge_matches_separate_linears():
    torch.manual_seed(7)
    dim = 4
    x = torch.randn(2, 3, dim)
    weights = [torch.randn(dim, dim) for _ in range(3)]
    biases = [torch.randn(dim) for _ in range(3)]
    checkpoint = []
    for index, name in [(2, "v"), (0, "q"), (1, "k")]:
        checkpoint.extend(
            [
                (f"blocks.0.attn1.to_{name}.weight", weights[index]),
                (f"blocks.0.attn1.to_{name}.bias", biases[index]),
            ]
        )

    mapping = wanvideo.build_wan_fused_projection_mapping(
        WanVideoConfig().param_names_mapping
    )
    state_dict, _ = hf_to_custom_state_dict(
        checkpoint, get_param_names_mapping(mapping)
    )
    fused = F.linear(
        x,
        state_dict["blocks.0.to_qkv.weight"],
        state_dict["blocks.0.to_qkv.bias"],
    ).chunk(3, dim=-1)
    separate = tuple(F.linear(x, weight, bias) for weight, bias in zip(weights, biases))

    for fused_part, separate_part in zip(fused, separate):
        torch.testing.assert_close(fused_part, separate_part)


def test_wan_fused_qkv_checkpoint_merges_channel_scales():
    scales = [
        torch.tensor([0.01, 0.02]),
        torch.tensor([0.03, 0.04]),
        torch.tensor([0.05, 0.06]),
    ]
    checkpoint = [
        (f"blocks.0.attn1.to_{name}.weight_scale", scales[index])
        for index, name in [(2, "v"), (0, "q"), (1, "k")]
    ]
    mapping = wanvideo.build_wan_fused_projection_mapping(
        WanVideoConfig().param_names_mapping
    )

    state_dict, _ = hf_to_custom_state_dict(
        checkpoint, get_param_names_mapping(mapping)
    )

    torch.testing.assert_close(
        state_dict["blocks.0.to_qkv.weight_scale"], torch.cat(scales)
    )


class _DummyFp8Config:
    @classmethod
    def get_name(cls):
        return "fp8"


def test_wan_fp8_fusion_flag_and_lora_guard(monkeypatch):
    args = SimpleNamespace(
        transformer_fp8_fused_projections=True,
        lora_path=None,
    )
    monkeypatch.setattr(wanvideo, "get_global_server_args", lambda: args)
    assert wanvideo.use_fp8_fused_projections(_DummyFp8Config())

    args.lora_path = "/tmp/adapter"
    assert not wanvideo.use_fp8_fused_projections(_DummyFp8Config())

    args.lora_path = None
    args.transformer_fp8_fused_projections = False
    assert not wanvideo.use_fp8_fused_projections(_DummyFp8Config())
    assert not wanvideo.use_fp8_fused_projections(None)


class _DummyModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs


class _DummyMergedLinear(_DummyModule):
    pass


def test_wan_block_builds_fused_qkv_kv_and_added_kv(monkeypatch):
    monkeypatch.setattr(
        wanvideo,
        "use_fp8_fused_projections",
        lambda quant_config: quant_config is not None,
    )
    monkeypatch.setattr(wanvideo, "get_tp_world_size", lambda: 1)
    monkeypatch.setattr(wanvideo, "MergedColumnParallelLinear", _DummyMergedLinear)
    for name in (
        "ColumnParallelLinear",
        "RowParallelLinear",
        "LayerNormScaleShift",
        "RMSNorm",
        "ScaleResidualLayerNormScaleShift",
        "USPAttention",
        "MLP",
        "MulAdd",
    ):
        monkeypatch.setattr(wanvideo, name, _DummyModule)

    block = wanvideo.WanTransformerBlock(
        dim=4,
        ffn_dim=8,
        num_heads=2,
        qk_norm="rms_norm_across_heads",
        cross_attn_norm=True,
        added_kv_proj_dim=4,
        supported_attention_backends=set(),
        quant_config=_DummyFp8Config(),
    )

    assert isinstance(block.to_qkv, _DummyMergedLinear)
    assert not hasattr(block, "to_q")
    assert not hasattr(block, "to_k")
    assert not hasattr(block, "to_v")
    assert isinstance(block.attn2.to_kv, _DummyMergedLinear)
    assert not hasattr(block.attn2, "to_k")
    assert not hasattr(block.attn2, "to_v")
    assert isinstance(block.attn2.to_added_kv, _DummyMergedLinear)
    assert not hasattr(block.attn2, "add_k_proj")


def test_videoedit_attention_backend_resolution():
    assert resolve_videoedit_attention_backend(None) is None
    assert (
        resolve_videoedit_attention_backend("SAGE_ATTN")
        == AttentionBackendEnum.SAGE_ATTN
    )
    assert resolve_videoedit_attention_backend("fa") == AttentionBackendEnum.FA

    with pytest.raises(ValueError, match="Unsupported VideoEdit Attention backend"):
        resolve_videoedit_attention_backend("unknown")


def test_wan_block_routes_attention_backend_by_role(monkeypatch):
    monkeypatch.setattr(wanvideo, "use_fp8_fused_projections", lambda _: False)
    monkeypatch.setattr(wanvideo, "get_tp_world_size", lambda: 1)
    for name in (
        "ColumnParallelLinear",
        "RowParallelLinear",
        "LayerNormScaleShift",
        "RMSNorm",
        "ScaleResidualLayerNormScaleShift",
        "USPAttention",
        "MLP",
        "MulAdd",
    ):
        monkeypatch.setattr(wanvideo, name, _DummyModule)

    block = wanvideo.WanTransformerBlock(
        dim=4,
        ffn_dim=8,
        num_heads=2,
        qk_norm="rms_norm_across_heads",
        cross_attn_norm=True,
        added_kv_proj_dim=4,
        supported_attention_backends={
            AttentionBackendEnum.FA,
            AttentionBackendEnum.SAGE_ATTN,
        },
        self_attention_backend=AttentionBackendEnum.SAGE_ATTN,
        cross_attention_backend=AttentionBackendEnum.FA,
    )

    assert block.attn1.kwargs["backend_override"] == AttentionBackendEnum.SAGE_ATTN
    assert block.attn1.kwargs["sage_attention_kernel"] == "qk_int8_pv_fp8_cuda"
    assert block.attn2.attn.kwargs["backend_override"] == AttentionBackendEnum.FA
    assert block.attn2.attn.kwargs["sage_attention_kernel"] == "auto"
