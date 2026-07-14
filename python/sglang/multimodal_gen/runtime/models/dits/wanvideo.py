# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.dits import WanVideoConfig
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams
from sglang.multimodal_gen.configs.sample.wan_teacache import (
    _wan_1_3b_coefficients,
    _wan_14b_coefficients,
)
from sglang.multimodal_gen.runtime.distributed import (
    divide,
    get_sp_group,
    get_sp_world_size,
    get_tp_world_size,
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.layers.attention import (
    MinimalA2AAttnOp,
    UlyssesAttention_VSA,
    USPAttention,
)
from sglang.multimodal_gen.runtime.layers.elementwise import MulAdd
from sglang.multimodal_gen.runtime.layers.layernorm import (
    FP32LayerNorm,
    LayerNormScaleShift,
    RMSNorm,
    ScaleResidualLayerNormScaleShift,
    tensor_parallel_rms_norm,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.mlp import MLP
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    NDRotaryEmbedding,
    _apply_rotary_emb,
    apply_flashinfer_rope_qk_inplace,
)
from sglang.multimodal_gen.runtime.layers.visual_embedding import (
    ModulateProjection,
    PatchEmbed,
    TimestepEmbedder,
)
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    get_is_main_process,
    init_logger,
)
from sglang.srt.utils import add_prefix

logger = init_logger(__name__)
_is_cuda = current_platform.is_cuda()


def _poly_eval(coefficients: list[float], x: float) -> float:
    result = 0.0
    for coefficient in coefficients:
        result = result * x + float(coefficient)
    return result


def _metric_float(value: torch.Tensor) -> float | None:
    result = float(value.detach().cpu().item())
    return result if math.isfinite(result) else None


def _tensor_pair_stats(
    current: torch.Tensor | None,
    previous: torch.Tensor | None,
) -> dict[str, Any]:
    if current is None:
        return {"available": False}

    stats: dict[str, Any] = {
        "available": True,
        "numel": int(current.numel()),
        "shape": list(current.shape),
    }
    if previous is None:
        stats["has_previous"] = False
        return stats

    stats["has_previous"] = True
    stats["previous_shape"] = list(previous.shape)
    if current.shape != previous.shape:
        stats["shape_changed"] = True
        return stats

    current_f = current.detach().float()
    previous_f = previous.detach().to(device=current.device).float()
    diff = current_f - previous_f
    eps = torch.tensor(1.0e-12, device=current.device, dtype=torch.float32)

    current_abs_mean = current_f.abs().mean()
    previous_abs_mean = previous_f.abs().mean()
    diff_abs_mean = diff.abs().mean()
    current_rms = current_f.square().mean().sqrt()
    previous_rms = previous_f.square().mean().sqrt()
    rmse = diff.square().mean().sqrt()
    current_norm = torch.linalg.vector_norm(current_f)
    previous_norm = torch.linalg.vector_norm(previous_f)
    diff_norm = torch.linalg.vector_norm(diff)
    norm_product = current_norm * previous_norm

    cosine_similarity = None
    norm_product_value = _metric_float(norm_product)
    if norm_product_value is not None and norm_product_value > 0.0:
        cosine = (current_f * previous_f).sum() / torch.maximum(norm_product, eps)
        cosine_similarity = _metric_float(cosine)

    stats.update(
        {
            "shape_changed": False,
            "current_abs_mean": _metric_float(current_abs_mean),
            "previous_abs_mean": _metric_float(previous_abs_mean),
            "mean_abs_delta": _metric_float(diff_abs_mean),
            "relative_l1": _metric_float(
                diff_abs_mean / torch.maximum(previous_abs_mean, eps)
            ),
            "current_rms": _metric_float(current_rms),
            "previous_rms": _metric_float(previous_rms),
            "rmse": _metric_float(rmse),
            "relative_l2": _metric_float(rmse / torch.maximum(previous_rms, eps)),
            "current_norm": _metric_float(current_norm),
            "previous_norm": _metric_float(previous_norm),
            "delta_norm": _metric_float(diff_norm),
            "relative_norm": _metric_float(
                diff_norm / torch.maximum(previous_norm, eps)
            ),
            "cosine_similarity": cosine_similarity,
            "cosine_distance": None
            if cosine_similarity is None
            else 1.0 - cosine_similarity,
        }
    )
    return stats


def _relative_l1_stats(
    current: torch.Tensor | None,
    previous: torch.Tensor | None,
) -> dict[str, Any]:
    if current is None:
        return {"available": False}
    stats: dict[str, Any] = {
        "available": True,
        "numel": int(current.numel()),
        "shape": list(current.shape),
    }
    if previous is None:
        stats["has_previous"] = False
        return stats
    stats["has_previous"] = True
    stats["previous_shape"] = list(previous.shape)
    if current.shape != previous.shape:
        stats["shape_changed"] = True
        return stats

    current_f = current.detach().float()
    previous_f = previous.detach().to(device=current.device).float()
    previous_abs_mean = previous_f.abs().mean()
    eps = torch.tensor(1.0e-12, device=current.device, dtype=torch.float32)
    rel_l1 = (current_f - previous_f).abs().mean() / torch.maximum(
        previous_abs_mean, eps
    )
    stats["relative_l1"] = _metric_float(rel_l1)
    stats["previous_abs_mean"] = _metric_float(previous_abs_mean)
    stats["current_abs_mean"] = _metric_float(current_f.abs().mean())
    return stats


def _normalize_encoder_hidden_states_image(
    encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None,
) -> torch.Tensor | None:
    if isinstance(encoder_hidden_states_image, list):
        if len(encoder_hidden_states_image) == 0:
            return None
        return encoder_hidden_states_image[0]
    return encoder_hidden_states_image


class WanImageEmbedding(torch.nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = MLP(
            in_features,
            in_features,
            out_features,
            act_type="gelu",
            prefix=add_prefix("ff.net", prefix),
            quant_config=quant_config,
        )
        self.norm2 = FP32LayerNorm(out_features)

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        dtype = encoder_hidden_states_image.dtype
        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states).to(dtype)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):

    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        text_embed_dim: int,
        image_embed_dim: int | None = None,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()

        self.time_embedder = TimestepEmbedder(
            dim,
            frequency_embedding_size=time_freq_dim,
            act_layer="silu",
            prefix=add_prefix("time_embedder", prefix),
            quant_config=quant_config,
        )
        self.time_modulation = ModulateProjection(
            dim,
            factor=6,
            act_layer="silu",
            prefix=add_prefix("time_proj", prefix),
            quant_config=quant_config,
        )
        self.text_embedder = MLP(
            text_embed_dim,
            dim,
            dim,
            bias=True,
            act_type="gelu_pytorch_tanh",
            quant_config=quant_config,
            fc_in_quant_prefix=add_prefix("text_embedder.linear_1", prefix),
            fc_out_quant_prefix=add_prefix("text_embedder.linear_2", prefix),
        )

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(
                image_embed_dim,
                dim,
                prefix=add_prefix("image_embedder", prefix),
                quant_config=quant_config,
            )

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        timestep_seq_len: int | None = None,
    ):
        temb = self.time_embedder(timestep, timestep_seq_len)
        timestep_proj = self.time_modulation(temb)

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            assert self.image_embedder is not None
            encoder_hidden_states_image = self.image_embedder(
                encoder_hidden_states_image
            )

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanSelfAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        parallel_attention=False,
        prefix: str = "",
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        is_cross_attention: bool = False,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.parallel_attention = parallel_attention
        tp_size = get_tp_world_size()

        # layers
        self.to_q = ColumnParallelLinear(
            dim,
            dim,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("to_q", prefix),
        )
        self.to_k = ColumnParallelLinear(
            dim,
            dim,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("to_k", prefix),
        )
        self.to_v = ColumnParallelLinear(
            dim,
            dim,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("to_v", prefix),
        )
        self.to_out = RowParallelLinear(
            dim,
            dim,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=add_prefix("to_out.0", prefix),
        )
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.tp_rmsnorm = tp_size > 1 and qk_norm
        self.local_num_heads = divide(num_heads, tp_size)

        # Scaled dot product attention
        self.attn = USPAttention(
            num_heads=self.local_num_heads,
            head_size=self.head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            skip_sequence_parallel=is_cross_attention,
            quant_config=quant_config,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor, context_lens: int):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
        """
        pass


class WanT2VCrossAttention(WanSelfAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, is_cross_attention=True)

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        q, _ = self.to_q(x)
        if self.tp_rmsnorm:
            q = tensor_parallel_rms_norm(q, self.norm_q)
        else:
            q = self.norm_q(q)
        q = q.unflatten(2, (self.local_num_heads, self.head_dim))

        k, _ = self.to_k(context)
        if self.tp_rmsnorm:
            k = tensor_parallel_rms_norm(k, self.norm_k)
        else:
            k = self.norm_k(k)
        k = k.unflatten(2, (self.local_num_heads, self.head_dim))

        v, _ = self.to_v(context)
        v = v.unflatten(2, (self.local_num_heads, self.head_dim))

        # compute attention
        x = self.attn(q, k, v)

        # output
        x = x.flatten(2)
        x, _ = self.to_out(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        prefix: str = "",
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(
            dim,
            num_heads,
            window_size,
            qk_norm,
            eps,
            supported_attention_backends=supported_attention_backends,
            is_cross_attention=True,
            quant_config=quant_config,
        )

        self.add_k_proj = ColumnParallelLinear(
            dim,
            dim,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("add_k_proj", prefix),
        )
        self.add_v_proj = ColumnParallelLinear(
            dim,
            dim,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("add_v_proj", prefix),
        )
        self.norm_added_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]

        q, _ = self.to_q(x)
        if self.tp_rmsnorm:
            q = tensor_parallel_rms_norm(q, self.norm_q)
        else:
            q = self.norm_q(q)
        q = q.unflatten(2, (self.local_num_heads, self.head_dim))

        k, _ = self.to_k(context)
        if self.tp_rmsnorm:
            k = tensor_parallel_rms_norm(k, self.norm_k)
        else:
            k = self.norm_k(k)
        k = k.unflatten(2, (self.local_num_heads, self.head_dim))

        v, _ = self.to_v(context)
        v = v.unflatten(2, (self.local_num_heads, self.head_dim))

        k_img, _ = self.add_k_proj(context_img)
        if self.tp_rmsnorm:
            k_img = tensor_parallel_rms_norm(k_img, self.norm_added_k)
        else:
            k_img = self.norm_added_k(k_img)
        k_img = k_img.unflatten(2, (self.local_num_heads, self.head_dim))

        v_img, _ = self.add_v_proj(context_img)
        v_img = v_img.unflatten(2, (self.local_num_heads, self.head_dim))

        img_x = self.attn(q, k_img, v_img)
        x = self.attn(q, k, v)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x, _ = self.to_out(x)
        return x


class WanTransformerBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
        attention_type: str = "original",
        sla_topk: float = 0.1,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = LayerNormScaleShift(
            dim,
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )
        self.to_q = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("attn1.to_q", prefix),
        )
        self.to_k = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("attn1.to_k", prefix),
        )
        self.to_v = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("attn1.to_v", prefix),
        )

        self.to_out = RowParallelLinear(
            dim,
            dim,
            bias=True,
            reduce_results=True,
            quant_config=quant_config,
            prefix=add_prefix("attn1.to_out.0", prefix),
        )
        tp_size = get_tp_world_size()
        self.local_num_heads = divide(num_heads, tp_size)
        self_attn_backends = supported_attention_backends

        if attention_type in ("sla", "sagesla"):
            self.attn1 = MinimalA2AAttnOp(
                num_heads=self.local_num_heads,
                head_size=dim // num_heads,
                attention_type=attention_type,
                topk=sla_topk,
                supported_attention_backends={
                    AttentionBackendEnum.SLA_ATTN,
                    AttentionBackendEnum.SAGE_SLA_ATTN,
                },
                prefix=add_prefix("attn1", prefix),
            )
        else:
            self.attn1 = USPAttention(
                num_heads=self.local_num_heads,
                head_size=dim // num_heads,
                causal=False,
                supported_attention_backends=self_attn_backends,
                prefix=add_prefix("attn1", prefix),
                quant_config=quant_config,
                is_cross_attention=False,
            )

        self.hidden_dim = dim
        self.num_attention_heads = num_heads
        self.dim_head = dim // num_heads
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(self.dim_head, eps=eps)
            self.norm_k = RMSNorm(self.dim_head, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            logger.error("QK Norm type not supported")
            raise Exception
        assert cross_attn_norm is True
        self.qk_norm = qk_norm
        self.tp_rmsnorm = qk_norm == "rms_norm_across_heads" and tp_size > 1
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            eps=eps,
            elementwise_affine=True,
            dtype=torch.float32,
        )

        # 2. Cross-attention
        cross_attn_backends = {
            b for b in supported_attention_backends if not b.is_sparse
        }
        if added_kv_proj_dim is not None:
            # I2V
            self.attn2 = WanI2VCrossAttention(
                dim,
                num_heads,
                qk_norm=qk_norm,
                eps=eps,
                prefix=add_prefix("attn2", prefix),
                supported_attention_backends=cross_attn_backends,
                quant_config=quant_config,
            )
        else:
            # T2V
            self.attn2 = WanT2VCrossAttention(
                dim,
                num_heads,
                qk_norm=qk_norm,
                eps=eps,
                prefix=add_prefix("attn2", prefix),
                supported_attention_backends=cross_attn_backends,
                quant_config=quant_config,
            )
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )

        # 3. Feed-forward
        self.ffn = MLP(
            dim,
            ffn_dim,
            act_type="gelu_pytorch_tanh",
            prefix=add_prefix("ffn.net", prefix),
            quant_config=quant_config,
        )
        self.mlp_residual = MulAdd()

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        bs, seq_length, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype
        if temb.dim() == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            e = self.scale_shift_table + temb.float()
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                e.chunk(6, dim=1)
            )

        assert shift_msa.dtype == torch.float32

        # 1. Self-attention
        norm_hidden_states = self.norm1(hidden_states, shift_msa, scale_msa)
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)

        if self.norm_q is not None:
            if self.tp_rmsnorm:
                query = tensor_parallel_rms_norm(query, self.norm_q)
            else:
                query = self.norm_q(query)
        if self.norm_k is not None:
            if self.tp_rmsnorm:
                key = tensor_parallel_rms_norm(key, self.norm_k)
            else:
                key = self.norm_k(key)
        query = query.squeeze(1).unflatten(2, (self.local_num_heads, self.dim_head))
        key = key.squeeze(1).unflatten(2, (self.local_num_heads, self.dim_head))
        value = value.squeeze(1).unflatten(2, (self.local_num_heads, self.dim_head))

        # Apply rotary embeddings
        cos, sin = freqs_cis
        if _is_cuda and query.shape == key.shape:
            cos_sin_cache = torch.cat(
                [
                    cos.to(dtype=torch.float32).contiguous(),
                    sin.to(dtype=torch.float32).contiguous(),
                ],
                dim=-1,
            )
            query, key = apply_flashinfer_rope_qk_inplace(
                query, key, cos_sin_cache, is_neox=False
            )
        else:
            query, key = _apply_rotary_emb(
                query, cos, sin, is_neox_style=False
            ), _apply_rotary_emb(key, cos, sin, is_neox_style=False)
        attn_output = self.attn1(query, key, value)
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        null_shift = null_scale = torch.zeros(
            (1,), device=hidden_states.device, dtype=hidden_states.dtype
        )
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale
        )
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype
        ), hidden_states.to(orig_dtype)

        # 2. Cross-attention
        attn_output = self.attn2(
            norm_hidden_states, context=encoder_hidden_states, context_lens=None
        )
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa, c_scale_msa
        )
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype
        ), hidden_states.to(orig_dtype)

        # 3. Feed-forward
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = self.mlp_residual(ff_output, c_gate_msa, hidden_states)
        hidden_states = hidden_states.to(orig_dtype)

        return hidden_states


class WanTransformerBlock_VSA(nn.Module):

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = LayerNormScaleShift(
            dim,
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )
        self.to_q = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=True,
            quant_config=quant_config,
            prefix=add_prefix("attn1.to_q", prefix),
        )
        self.to_k = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=True,
            quant_config=quant_config,
            prefix=add_prefix("attn1.to_k", prefix),
        )
        self.to_v = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=True,
            quant_config=quant_config,
            prefix=add_prefix("attn1.to_v", prefix),
        )
        self.to_gate_compress = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=True,
            quant_config=quant_config,
            prefix=add_prefix("attn1.to_gate_compress", prefix),
        )

        self.to_out = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=True,
            quant_config=quant_config,
            prefix=add_prefix("attn1.to_out.0", prefix),
        )
        self.attn1 = UlyssesAttention_VSA(
            num_heads=num_heads,
            head_size=dim // num_heads,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            prefix=add_prefix("attn1", prefix),
            quant_config=quant_config,
        )
        self.hidden_dim = dim
        self.num_attention_heads = num_heads
        dim_head = dim // num_heads
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            logger.error("QK Norm type not supported")
            raise Exception
        assert cross_attn_norm is True
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            eps=eps,
            elementwise_affine=True,
            dtype=torch.float32,
        )

        # 2. Cross-attention
        cross_attn_backends = {
            b for b in supported_attention_backends if not b.is_sparse
        }
        if added_kv_proj_dim is not None:
            # I2V
            self.attn2 = WanI2VCrossAttention(
                dim,
                num_heads,
                qk_norm=qk_norm,
                eps=eps,
                prefix=add_prefix("attn2", prefix),
                supported_attention_backends=cross_attn_backends,
                quant_config=quant_config,
            )
        else:
            # T2V
            self.attn2 = WanT2VCrossAttention(
                dim,
                num_heads,
                qk_norm=qk_norm,
                eps=eps,
                prefix=add_prefix("attn2", prefix),
                supported_attention_backends=cross_attn_backends,
                quant_config=quant_config,
            )
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )

        # 3. Feed-forward
        self.ffn = MLP(
            dim,
            ffn_dim,
            act_type="gelu_pytorch_tanh",
            prefix=add_prefix("ffn.net", prefix),
            quant_config=quant_config,
        )
        self.mlp_residual = MulAdd()

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        bs, seq_length, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype
        # assert orig_dtype != torch.float32
        e = self.scale_shift_table + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(
            6, dim=1
        )
        assert shift_msa.dtype == torch.float32

        # 1. Self-attention
        norm_hidden_states = self.norm1(hidden_states, shift_msa, scale_msa)
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)
        gate_compress, _ = self.to_gate_compress(norm_hidden_states)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        gate_compress = gate_compress.squeeze(1).unflatten(
            2, (self.num_attention_heads, -1)
        )

        # Apply rotary embeddings
        cos, sin = freqs_cis
        if _is_cuda and query.shape == key.shape:
            cos_sin_cache = torch.cat(
                [
                    cos.to(dtype=torch.float32).contiguous(),
                    sin.to(dtype=torch.float32).contiguous(),
                ],
                dim=-1,
            )
            query, key = apply_flashinfer_rope_qk_inplace(
                query, key, cos_sin_cache, is_neox=False
            )
        else:
            query, key = _apply_rotary_emb(
                query, cos, sin, is_neox_style=False
            ), _apply_rotary_emb(key, cos, sin, is_neox_style=False)

        attn_output = self.attn1(query, key, value, gate_compress=gate_compress)
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        null_shift = null_scale = torch.zeros((1,), device=hidden_states.device)
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale
        )
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype
        ), hidden_states.to(orig_dtype)

        # 2. Cross-attention
        attn_output = self.attn2(
            norm_hidden_states, context=encoder_hidden_states, context_lens=None
        )
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa, c_scale_msa
        )
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype
        ), hidden_states.to(orig_dtype)

        # 3. Feed-forward
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = self.mlp_residual(ff_output, c_gate_msa, hidden_states)
        hidden_states = hidden_states.to(orig_dtype)

        return hidden_states


class WanTransformer3DModel(CachableDiT, OffloadableDiTMixin):
    _fsdp_shard_conditions = WanVideoConfig()._fsdp_shard_conditions
    _compile_conditions = WanVideoConfig()._compile_conditions
    _supported_attention_backends = WanVideoConfig()._supported_attention_backends
    param_names_mapping = WanVideoConfig().param_names_mapping
    reverse_param_names_mapping = WanVideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = WanVideoConfig().lora_param_names_mapping

    def __init__(
        self,
        config: WanVideoConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)

        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_channels_latents = config.num_channels_latents
        self.patch_size = config.patch_size
        self.text_len = config.text_len

        # 1. Patch & position embedding
        self.patch_embedding = PatchEmbed(
            in_chans=config.in_channels,
            embed_dim=inner_dim,
            patch_size=config.patch_size,
            flatten=False,
        )

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            text_embed_dim=config.text_dim,
            image_embed_dim=config.image_dim,
            prefix="condition_embedder",
            quant_config=quant_config,
        )

        # 3. Transformer blocks
        attn_backend = get_global_server_args().attention_backend
        transformer_block = (
            WanTransformerBlock_VSA
            if (attn_backend and attn_backend.lower() == "video_sparse_attn")
            else WanTransformerBlock
        )
        self.blocks = nn.ModuleList(
            [
                transformer_block(
                    inner_dim,
                    config.ffn_dim,
                    config.num_attention_heads,
                    config.qk_norm,
                    config.cross_attn_norm,
                    config.eps,
                    config.added_kv_proj_dim,
                    self._supported_attention_backends
                    | {AttentionBackendEnum.VIDEO_SPARSE_ATTN},
                    prefix=f"blocks.{i}",
                    attention_type=config.attention_type,
                    sla_topk=config.sla_topk,
                    quant_config=quant_config,
                )
                for i in range(config.num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = LayerNormScaleShift(
            inner_dim,
            eps=config.eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )
        self.proj_out = ColumnParallelLinear(
            inner_dim,
            config.out_channels * math.prod(config.patch_size),
            bias=True,
            gather_output=True,
            prefix=f"proj_out",
            quant_config=quant_config,
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        # For type checking

        self.cnt = 0
        self._teacache_proxy_validation_states: dict[
            tuple[str, str], dict[str, Any]
        ] = {}
        self.__post_init__()

        # misc
        self.sp_size = get_sp_world_size()

        # Get rotary embeddings
        d = self.hidden_size // self.num_attention_heads
        self.rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]

        self.rotary_emb = NDRotaryEmbedding(
            rope_dim_list=self.rope_dim_list,
            rope_theta=10000,
            dtype=(
                torch.float32
                if current_platform.is_mps() or current_platform.is_musa()
                else torch.float64
            ),
        )

        self.layer_names = ["blocks"]

    @lru_cache(maxsize=1)
    def _compute_rope_for_sequence_shard(
        self,
        local_len: int,
        rank: int,
        frame_stride_local: int,
        width_local: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_start = rank * local_len
        token_indices = torch.arange(
            token_start,
            token_start + local_len,
            device=device,
            dtype=torch.long,
        )
        t_idx = token_indices // frame_stride_local
        rem = token_indices % frame_stride_local
        h_idx = rem // width_local
        w_idx = rem % width_local
        positions = torch.stack((t_idx, h_idx, w_idx), dim=1)
        return self.rotary_emb.forward_uncached(positions)

    def _get_teacache_residual_trace_path(self, forward_batch: Any) -> str | None:
        sampling_params = getattr(forward_batch, "sampling_params", None)
        if sampling_params is not None:
            path = getattr(sampling_params, "teacache_residual_trace_path", None)
            if path:
                return path
        path = getattr(forward_batch, "teacache_residual_trace_path", None)
        return path or None

    def _reset_teacache_proxy_validation_state(self) -> None:
        self._teacache_proxy_validation_states = {}

    def _wan_teacache_coefficients(
        self,
        *,
        model_size: str,
        use_ret_steps: bool,
    ) -> list[float]:
        params = TeaCacheParams(
            teacache_thresh=0.0,
            use_ret_steps=use_ret_steps,
        )
        if model_size == "1.3b":
            return _wan_1_3b_coefficients(params)
        if model_size == "14b":
            return _wan_14b_coefficients(params)
        raise ValueError(f"Unsupported Wan TeaCache model size: {model_size}")

    def _first_block_modulated_hidden_proxy(
        self,
        hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.blocks:
            return None
        block = self.blocks[0]
        if not hasattr(block, "norm1") or not hasattr(block, "scale_shift_table"):
            return None
        if timestep_proj.dim() == 4:
            e = block.scale_shift_table.unsqueeze(0) + timestep_proj.float()
            shift_msa, scale_msa, *_ = e.chunk(6, dim=2)
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
        else:
            e = block.scale_shift_table + timestep_proj.float()
            shift_msa, scale_msa, *_ = e.chunk(6, dim=1)
        return block.norm1(hidden_states, shift_msa, scale_msa)

    def _build_teacache_proxy_candidates(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
        temb: torch.Tensor,
        threshold: float,
    ) -> list[dict[str, Any]]:
        wan14_ret = self._wan_teacache_coefficients(
            model_size="14b", use_ret_steps=True
        )
        wan14_temb = self._wan_teacache_coefficients(
            model_size="14b", use_ret_steps=False
        )
        wan13_ret = self._wan_teacache_coefficients(
            model_size="1.3b", use_ret_steps=True
        )
        wan13_temb = self._wan_teacache_coefficients(
            model_size="1.3b", use_ret_steps=False
        )

        candidates: list[dict[str, Any]] = [
            {
                "name": "wan14_i2v_timestep_proj",
                "description": "Current VideoEdit/Wan I2V 14B TeaCache proxy.",
                "tensor": timestep_proj,
                "coefficients": wan14_ret,
                "coefficients_source": "wan_14b_use_ret_steps_true",
                "decision_threshold": threshold,
            },
            {
                "name": "wan14_t2v_temb",
                "description": "Wan 14B non-ret/t2v-style time embedding proxy.",
                "tensor": temb,
                "coefficients": wan14_temb,
                "coefficients_source": "wan_14b_use_ret_steps_false",
                "decision_threshold": threshold,
            },
            {
                "name": "wan13_timestep_proj",
                "description": "Wan 1.3B ret-step proxy evaluated on VideoEdit tensors.",
                "tensor": timestep_proj,
                "coefficients": wan13_ret,
                "coefficients_source": "wan_1p3b_use_ret_steps_true",
                "decision_threshold": threshold,
            },
            {
                "name": "wan13_temb",
                "description": "Wan 1.3B non-ret time embedding proxy.",
                "tensor": temb,
                "coefficients": wan13_temb,
                "coefficients_source": "wan_1p3b_use_ret_steps_false",
                "decision_threshold": threshold,
            },
            {
                "name": "patch_hidden_channel_mean",
                "description": "Content-aware patched hidden-state channel mean.",
                "tensor": hidden_states.detach().float().mean(dim=1),
                "coefficients": [1.0, 0.0],
                "coefficients_source": "identity",
                "decision_threshold": threshold,
            },
        ]

        first_block_modulated = self._first_block_modulated_hidden_proxy(
            hidden_states,
            timestep_proj,
        )
        if first_block_modulated is not None:
            if os.environ.get("SGLANG_TEACACHE_PROXY_INCLUDE_FULL_HIDDEN") == "1":
                candidates.append(
                    {
                        "name": "first_block_modulated_hidden",
                        "description": (
                            "Exact paper-style first-block modulated hidden proxy: "
                            "norm(x) * (1 + scale) + shift."
                        ),
                        "tensor": first_block_modulated,
                        "coefficients": [1.0, 0.0],
                        "coefficients_source": "identity",
                        "decision_threshold": threshold,
                    }
                )
            candidates.append(
                {
                    "name": "first_block_modulated_channel_mean",
                    "description": (
                        "Pooled paper-style first-block modulated hidden proxy: "
                        "mean over tokens after norm(x) * (1 + scale) + shift."
                    ),
                    "tensor": first_block_modulated.detach().float().mean(dim=1),
                    "coefficients": [1.0, 0.0],
                    "coefficients_source": "identity",
                    "decision_threshold": threshold,
                }
            )

        return candidates

    def _compute_teacache_proxy_candidate_metrics(
        self,
        *,
        name: str,
        branch: str,
        tensor: torch.Tensor,
        coefficients: list[float],
        coefficients_source: str,
        decision_threshold: float,
        is_boundary_step: bool,
    ) -> dict[str, Any]:
        state_key = (name, branch)
        state = self._teacache_proxy_validation_states.setdefault(
            state_key,
            {
                "previous_tensor": None,
                "accumulated": 0.0,
            },
        )

        previous_tensor = state["previous_tensor"]
        rel_stats = _relative_l1_stats(tensor, previous_tensor)
        rel_l1 = rel_stats.get("relative_l1")
        accumulated_before = float(state["accumulated"])
        rescaled_l1 = None
        candidate_accumulated = None

        if is_boundary_step or rel_l1 is None:
            action = "compute"
            accumulated_after = 0.0
        else:
            rescaled_l1 = _poly_eval(coefficients, float(rel_l1))
            candidate_accumulated = accumulated_before + float(rescaled_l1)
            if candidate_accumulated >= decision_threshold:
                action = "compute"
                accumulated_after = 0.0
            else:
                action = "skip"
                accumulated_after = candidate_accumulated

        state["previous_tensor"] = tensor.detach().clone()
        state["accumulated"] = float(accumulated_after)

        return {
            "name": name,
            "branch": branch,
            "shape": list(tensor.shape),
            "numel": int(tensor.numel()),
            "coefficients_source": coefficients_source,
            "coefficients": [float(value) for value in coefficients],
            "decision_threshold": float(decision_threshold),
            "relative_l1": rel_l1,
            "relative_l1_stats": rel_stats,
            "rescaled_l1": rescaled_l1,
            "accumulated_before": accumulated_before,
            "candidate_accumulated": candidate_accumulated,
            "accumulated_after": float(accumulated_after),
            "would_action": action,
            "would_skip": action == "skip",
        }

    def _prepare_teacache_proxy_candidates(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
        temb: torch.Tensor,
        is_boundary_step: bool,
        threshold: float,
        branch: str,
    ) -> dict[str, Any]:
        candidates = self._build_teacache_proxy_candidates(
            hidden_states=hidden_states,
            timestep_proj=timestep_proj,
            temb=temb,
            threshold=threshold,
        )
        proxy_metrics = {}
        for candidate in candidates:
            proxy_metrics[candidate["name"]] = (
                self._compute_teacache_proxy_candidate_metrics(
                    name=candidate["name"],
                    branch=branch,
                    tensor=candidate["tensor"],
                    coefficients=candidate["coefficients"],
                    coefficients_source=candidate["coefficients_source"],
                    decision_threshold=candidate["decision_threshold"],
                    is_boundary_step=is_boundary_step,
                )
            )
        return proxy_metrics

    def _prepare_teacache_residual_validation(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
        temb: torch.Tensor,
    ) -> dict[str, Any] | None:
        """Simulate TeaCache decisions during a full-forward run.

        This is intentionally enabled only when real TeaCache is disabled. It validates
        whether the current proxy would skip/compute correctly on the clean trajectory.
        """
        forward_context = get_forward_context()
        forward_batch = forward_context.forward_batch
        if forward_batch is None or getattr(forward_batch, "enable_teacache", False):
            return None
        trace_path = self._get_teacache_residual_trace_path(forward_batch)
        if not trace_path or getattr(forward_batch, "teacache_params", None) is None:
            return None

        current_timestep = int(forward_context.current_timestep)
        is_cfg_negative = bool(getattr(forward_batch, "is_cfg_negative", False))
        if current_timestep == 0 and not is_cfg_negative:
            self.reset_teacache_state()
            self._reset_teacache_proxy_validation_state()

        teacache_params = forward_batch.teacache_params
        num_inference_steps = int(forward_batch.num_inference_steps)
        do_cfg = bool(getattr(forward_batch, "do_classifier_free_guidance", False))
        use_ret_steps = teacache_params.use_ret_steps
        start_skipping, end_skipping = teacache_params.get_skip_boundaries(
            num_inference_steps,
            do_cfg,
        )
        is_boundary_step = self.cnt < start_skipping or self.cnt >= end_skipping
        modulated_inp = timestep_proj if use_ret_steps else temb
        branch = "uncond" if is_cfg_negative else "cond"

        self.is_cfg_negative = is_cfg_negative
        should_calc = self._compute_teacache_decision(
            modulated_inp=modulated_inp,
            is_boundary_step=is_boundary_step,
            coefficients=teacache_params.get_coefficients(),
            teacache_thresh=teacache_params.teacache_thresh,
            force_enabled=True,
        )
        cached_residual = (
            self.previous_residual_negative
            if is_cfg_negative and self._supports_cfg_cache
            else self.previous_residual
        )

        sampling_params = getattr(forward_batch, "sampling_params", None)
        window_spec = getattr(sampling_params, "runtime_window_spec", None)
        return {
            "trace_path": trace_path,
            "request_id": getattr(forward_batch, "request_id", None),
            "window_index": getattr(sampling_params, "runtime_window_index", None),
            "window_start_index": getattr(window_spec, "start_index", None),
            "window_end_index": getattr(window_spec, "end_index", None),
            "denoise_step": current_timestep,
            "num_inference_steps": num_inference_steps,
            "branch": branch,
            "is_cfg_negative": is_cfg_negative,
            "do_cfg": do_cfg,
            "raw_forward_index": int(self.cnt),
            "would_skip": not should_calc,
            "would_action": "compute" if should_calc else "skip",
            "is_boundary_step": bool(is_boundary_step),
            "start_skipping": int(start_skipping),
            "end_skipping": int(end_skipping),
            "threshold": float(teacache_params.teacache_thresh),
            "use_ret_steps": use_ret_steps,
            "rel_l1": self._teacache_last_rel_l1,
            "rescaled_l1": self._teacache_last_rescaled_l1,
            "accumulated_before": self._teacache_last_accumulated_before,
            "candidate_accumulated": self._teacache_last_candidate_accumulated,
            "accumulated_after": self._current_teacache_accumulated(),
            "proxy_candidates": self._prepare_teacache_proxy_candidates(
                hidden_states=hidden_states,
                timestep_proj=timestep_proj,
                temb=temb,
                is_boundary_step=bool(is_boundary_step),
                threshold=float(teacache_params.teacache_thresh),
                branch=branch,
            ),
            "cached_residual": cached_residual,
            "should_calc": should_calc,
        }

    def _write_teacache_residual_validation_trace(
        self,
        validation: dict[str, Any],
        *,
        residual_error: dict[str, Any],
    ) -> None:
        if not get_is_main_process():
            return
        record = {
            "event": "teacache_residual_validation",
            "request_id": validation["request_id"],
            "window_index": validation["window_index"],
            "window_start_index": validation["window_start_index"],
            "window_end_index": validation["window_end_index"],
            "denoise_step": validation["denoise_step"],
            "num_inference_steps": validation["num_inference_steps"],
            "branch": validation["branch"],
            "is_cfg_negative": validation["is_cfg_negative"],
            "do_cfg": validation["do_cfg"],
            "raw_forward_index": validation["raw_forward_index"],
            "would_skip": validation["would_skip"],
            "would_action": validation["would_action"],
            "is_boundary_step": validation["is_boundary_step"],
            "start_skipping": validation["start_skipping"],
            "end_skipping": validation["end_skipping"],
            "threshold": validation["threshold"],
            "use_ret_steps": validation["use_ret_steps"],
            "rel_l1": validation["rel_l1"],
            "rescaled_l1": validation["rescaled_l1"],
            "accumulated_before": validation["accumulated_before"],
            "candidate_accumulated": validation["candidate_accumulated"],
            "accumulated_after": validation["accumulated_after"],
            "proxy_candidates": validation.get("proxy_candidates"),
            "cached_residual_available": bool(
                validation["cached_residual"] is not None
            ),
            "residual_error": residual_error,
        }
        path = Path(validation["trace_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def _finalize_teacache_residual_validation(
        self,
        validation: dict[str, Any] | None,
        *,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
    ) -> None:
        if validation is None:
            return
        current_residual = hidden_states.squeeze(0) - original_hidden_states
        residual_error = _tensor_pair_stats(
            current_residual,
            validation["cached_residual"],
        )
        self._write_teacache_residual_validation_trace(
            validation,
            residual_error=residual_error,
        )
        if validation["should_calc"]:
            if validation["is_cfg_negative"] and self._supports_cfg_cache:
                self.previous_residual_negative = current_residual
            else:
                self.previous_residual = current_residual

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance=None,
        **kwargs,
    ) -> torch.Tensor:
        forward_batch = get_forward_context().forward_batch
        if forward_batch is not None:
            sequence_shard_enabled = (
                forward_batch.enable_sequence_shard and self.sp_size > 1
            )
        else:
            sequence_shard_enabled = False
        self.enable_teacache = (
            forward_batch is not None and forward_batch.enable_teacache
        )

        orig_dtype = hidden_states.dtype
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        encoder_hidden_states_image = _normalize_encoder_hidden_states_image(
            encoder_hidden_states_image
        )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        if not sequence_shard_enabled:
            # The rotary embedding layer correctly handles SP offsets internally.
            freqs_cos, freqs_sin = self.rotary_emb.forward_from_grid(
                (
                    post_patch_num_frames * self.sp_size,
                    post_patch_height,
                    post_patch_width,
                ),
                shard_dim=0,
                start_frame=0,
                device=hidden_states.device,
            )
            assert freqs_cos.dtype == torch.float32
            assert freqs_cos.device == hidden_states.device
            freqs_cis = (
                (freqs_cos.float(), freqs_sin.float())
                if freqs_cos is not None
                else None
            )

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # shape is [B, T' * H' * W', C]
        seq_len_orig = hidden_states.shape[1]
        seq_shard_pad = 0
        if sequence_shard_enabled:
            if seq_len_orig % self.sp_size != 0:
                seq_shard_pad = self.sp_size - (seq_len_orig % self.sp_size)
                pad = torch.zeros(
                    (batch_size, seq_shard_pad, hidden_states.shape[2]),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                hidden_states = torch.cat([hidden_states, pad], dim=1)
            sp_rank = get_sp_group().rank_in_group
            local_seq_len = hidden_states.shape[1] // self.sp_size
            hidden_states = hidden_states.view(
                batch_size, self.sp_size, local_seq_len, hidden_states.shape[2]
            )
            hidden_states = hidden_states[:, sp_rank, :, :]

            frame_stride = post_patch_height * post_patch_width
            freqs_cos, freqs_sin = self._compute_rope_for_sequence_shard(
                local_seq_len,
                sp_rank,
                frame_stride,
                post_patch_width,
                hidden_states.device,
            )
            freqs_cis = (freqs_cos.float(), freqs_sin.float())

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.dim() == 2:
            # ti2v
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image,
                timestep_seq_len=ts_seq_len,
            )
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if sequence_shard_enabled and ts_seq_len is not None:
            if seq_shard_pad > 0:
                pad = torch.zeros(
                    (
                        batch_size,
                        seq_shard_pad,
                        timestep_proj.shape[2],
                        timestep_proj.shape[3],
                    ),
                    dtype=timestep_proj.dtype,
                    device=timestep_proj.device,
                )
                timestep_proj = torch.cat([timestep_proj, pad], dim=1)
            timestep_proj = timestep_proj.view(
                batch_size,
                self.sp_size,
                local_seq_len,
                timestep_proj.shape[2],
                timestep_proj.shape[3],
            )
            timestep_proj = timestep_proj[:, sp_rank, :, :, :]

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )

        encoder_hidden_states = (
            encoder_hidden_states.to(orig_dtype)
            if not current_platform.is_amp_supported()
            else encoder_hidden_states
        )  # cast to orig_dtype for MPS

        assert encoder_hidden_states.dtype == orig_dtype

        # 4. Transformer blocks
        # if caching is enabled, we might be able to skip the forward pass
        should_skip_forward = self.should_skip_forward_for_cached_states(
            timestep_proj=timestep_proj, temb=temb
        )
        residual_validation = self._prepare_teacache_residual_validation(
            hidden_states=hidden_states,
            timestep_proj=timestep_proj,
            temb=temb,
        )

        if should_skip_forward:
            hidden_states = self.retrieve_cached_states(hidden_states)
        else:
            # if teacache is enabled, we need to cache the original hidden states
            if self.enable_teacache or residual_validation is not None:
                original_hidden_states = hidden_states.clone()

            for block in self.blocks:
                hidden_states = block(
                    hidden_states, encoder_hidden_states, timestep_proj, freqs_cis
                )
            if residual_validation is not None:
                self._finalize_teacache_residual_validation(
                    residual_validation,
                    hidden_states=hidden_states,
                    original_hidden_states=original_hidden_states,
                )
            # if teacache is enabled, we need to cache the original hidden states
            if self.enable_teacache:
                self.maybe_cache_states(hidden_states, original_hidden_states)
        self.cnt += 1

        if sequence_shard_enabled:
            hidden_states = hidden_states.contiguous()
            hidden_states = sequence_model_parallel_all_gather(hidden_states, dim=1)
            if seq_shard_pad > 0:
                hidden_states = hidden_states[:, :seq_len_orig, :]

        # 5. Output norm, projection & unpatchify
        if temb.dim() == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (
                self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)
            ).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states, _ = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return output

    def maybe_cache_states(
        self, hidden_states: torch.Tensor, original_hidden_states: torch.Tensor
    ) -> None:
        """Cache residual with CFG positive/negative separation."""
        residual = hidden_states.squeeze(0) - original_hidden_states
        if not self.is_cfg_negative:
            self.previous_residual = residual
        else:
            self.previous_residual_negative = residual

    def should_skip_forward_for_cached_states(self, **kwargs) -> bool:
        if not self.enable_teacache:
            return False
        ctx = self._get_teacache_context()
        if ctx is None:
            return False

        # Initialize Wan-specific parameters
        teacache_params = ctx.teacache_params
        use_ret_steps = teacache_params.use_ret_steps
        start_skipping, end_skipping = teacache_params.get_skip_boundaries(
            ctx.num_inference_steps, ctx.do_cfg
        )

        # Determine boundary step
        is_boundary_step = self.cnt < start_skipping or self.cnt >= end_skipping

        timestep_proj = kwargs["timestep_proj"]
        temb = kwargs["temb"]
        modulated_inp = timestep_proj if use_ret_steps else temb

        self.is_cfg_negative = ctx.is_cfg_negative

        # Use shared helper to compute cache decision
        should_calc = self._compute_teacache_decision(
            modulated_inp=modulated_inp,
            is_boundary_step=is_boundary_step,
            coefficients=ctx.coefficients,
            teacache_thresh=ctx.teacache_thresh,
        )
        self._record_teacache_decision_trace(
            ctx=ctx,
            should_calc=should_calc,
            is_boundary_step=is_boundary_step,
            start_skipping=start_skipping,
            end_skipping=end_skipping,
            use_ret_steps=use_ret_steps,
        )

        return not should_calc

    def retrieve_cached_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Retrieve cached residual with CFG positive/negative separation."""
        residual = (
            self.previous_residual_negative
            if self.is_cfg_negative
            else self.previous_residual
        )
        if residual is None:
            return hidden_states
        return hidden_states + residual.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )


EntryClass = WanTransformer3DModel
