# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0


import sageattention
import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (  # FlashAttentionMetadata,
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class SageAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SAGE_ATTN

    @staticmethod
    def get_impl_cls() -> type["SageAttentionImpl"]:
        return SageAttentionImpl


class SageAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = extra_impl_args.get("dropout_p", 0.0)
        self.kernel_name = extra_impl_args.get("sage_attention_kernel", "auto")
        valid_kernels = {
            "auto",
            "qk_int8_pv_fp16_cuda",
            "qk_int8_pv_fp8_cuda",
        }
        if self.kernel_name not in valid_kernels:
            raise ValueError(
                f"Unsupported SageAttention kernel {self.kernel_name!r}; "
                f"expected one of {sorted(valid_kernels)}"
            )
        self.qk_quant_gran = extra_impl_args.get("sage_qk_quant_gran", "per_thread")
        self.smooth_k = bool(extra_impl_args.get("sage_smooth_k", True))
        default_accum_dtype = (
            "fp32+fp16" if self.kernel_name == "qk_int8_pv_fp8_cuda" else "fp16+fp32"
        )
        self.pv_accum_dtype = extra_impl_args.get(
            "sage_pv_accum_dtype", default_accum_dtype
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
        *,
        return_softmax_lse: bool = False,
    ) -> torch.Tensor:
        common_kwargs = {
            "tensor_layout": "NHD",
            "is_causal": self.causal,
            "sm_scale": self.softmax_scale,
            "return_lse": return_softmax_lse,
        }
        if self.kernel_name == "auto":
            output = sageattention.sageattn(
                query,
                key,
                value,
                **common_kwargs,
            )
        else:
            kernel = getattr(sageattention, f"sageattn_{self.kernel_name}")
            output = kernel(
                query,
                key,
                value,
                qk_quant_gran=self.qk_quant_gran,
                pv_accum_dtype=self.pv_accum_dtype,
                smooth_k=self.smooth_k,
                **common_kwargs,
            )
        if return_softmax_lse:
            output, softmax_lse = output
            return output, softmax_lse
        return output
