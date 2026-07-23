# SPDX-License-Identifier: Apache-2.0
import torch

from sglang.multimodal_gen.configs.models.dits.wan_videoedit import (
    WanVideoEditConfig,
)
from sglang.multimodal_gen.runtime.models.dits.wanvideo import (
    WanI2VCrossAttention,
    WanTransformer3DModel,
    tensor_parallel_rms_norm,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import get_global_server_args

_VIDEOEDIT_ATTENTION_BACKENDS = {
    "fa": AttentionBackendEnum.FA,
    "sage_attn": AttentionBackendEnum.SAGE_ATTN,
}


def resolve_videoedit_attention_backend(
    name: str | None,
) -> AttentionBackendEnum | None:
    if name is None:
        return None
    normalized = name.lower()
    try:
        return _VIDEOEDIT_ATTENTION_BACKENDS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported VideoEdit Attention backend {name!r}; expected one of "
            f"{sorted(_VIDEOEDIT_ATTENTION_BACKENDS)}"
        ) from exc


class WanVideoEditCrossAttention(WanI2VCrossAttention):
    """I2V-style cross-attention with dynamic image-token split.

    VideoEdit sends only 512 text tokens, while Wan I2V sends image tokens followed
    by text tokens. The inherited Wan I2V attention hard-codes 257 image tokens;
    this adapter derives image token count from the context length instead.
    """

    text_context_len: int = 512

    def forward(self, x, context, context_lens):
        image_context_length = max(context.shape[1] - self.text_context_len, 0)
        context_img = (
            context[:, :image_context_length] if image_context_length > 0 else None
        )
        context = context[:, image_context_length:]

        q, _ = self.to_q(x)
        if self.tp_rmsnorm:
            q = tensor_parallel_rms_norm(q, self.norm_q)
        else:
            q = self.norm_q(q)
        q = q.unflatten(2, (self.local_num_heads, self.head_dim))

        if self.use_fused_kv:
            kv, _ = self.to_kv(context)
            k, v = tuple(part.contiguous() for part in kv.chunk(2, dim=-1))
        else:
            k, _ = self.to_k(context)
            v, _ = self.to_v(context)
        if self.tp_rmsnorm:
            k = tensor_parallel_rms_norm(k, self.norm_k)
        else:
            k = self.norm_k(k)
        k = k.unflatten(2, (self.local_num_heads, self.head_dim))
        v = v.unflatten(2, (self.local_num_heads, self.head_dim))

        x = self.attn(q, k, v, profile_kind="text_cross").flatten(2)

        if context_img is not None and context_img.shape[1] > 0:
            if self.use_fused_added_kv:
                added_kv, _ = self.to_added_kv(context_img)
                k_img, v_img = tuple(
                    part.contiguous() for part in added_kv.chunk(2, dim=-1)
                )
            else:
                k_img, _ = self.add_k_proj(context_img)
                v_img, _ = self.add_v_proj(context_img)
            if self.tp_rmsnorm:
                k_img = tensor_parallel_rms_norm(k_img, self.norm_added_k)
            else:
                k_img = self.norm_added_k(k_img)
            k_img = k_img.unflatten(2, (self.local_num_heads, self.head_dim))
            v_img = v_img.unflatten(2, (self.local_num_heads, self.head_dim))
            x = x + self.attn(q, k_img, v_img, profile_kind="image_cross").flatten(2)

        x, _ = self.to_out(x)
        return x


class WanVideoEditTransformer3DModel(WanTransformer3DModel):
    _fsdp_shard_conditions = WanVideoEditConfig()._fsdp_shard_conditions
    _compile_conditions = WanVideoEditConfig()._compile_conditions
    _supported_attention_backends = WanVideoEditConfig()._supported_attention_backends
    param_names_mapping = WanVideoEditConfig().param_names_mapping
    reverse_param_names_mapping = WanVideoEditConfig().reverse_param_names_mapping
    lora_param_names_mapping = WanVideoEditConfig().lora_param_names_mapping

    def _get_attention_backend_overrides(
        self,
    ) -> tuple[AttentionBackendEnum | None, AttentionBackendEnum | None]:
        server_args = get_global_server_args()
        self_backend_name = getattr(
            server_args, "videoedit_self_attention_backend", None
        )
        cross_backend_name = getattr(
            server_args, "videoedit_cross_attention_backend", None
        )
        if self_backend_name == "sage_attn" and cross_backend_name is None:
            cross_backend_name = "fa"
        return (
            resolve_videoedit_attention_backend(self_backend_name),
            resolve_videoedit_attention_backend(cross_backend_name),
        )

    def __init__(self, config: WanVideoEditConfig, hf_config, quant_config=None):
        super().__init__(config=config, hf_config=hf_config, quant_config=quant_config)
        for i, block in enumerate(self.blocks):
            block.attn2 = WanVideoEditCrossAttention(
                config.hidden_size,
                config.num_attention_heads,
                qk_norm=config.qk_norm,
                eps=config.eps,
                prefix=f"blocks.{i}.attn2",
                supported_attention_backends={
                    b for b in self._supported_attention_backends if not b.is_sparse
                },
                attention_backend_override=self.cross_attention_backend_override,
                quant_config=quant_config,
            )


EntryClass = WanVideoEditTransformer3DModel
