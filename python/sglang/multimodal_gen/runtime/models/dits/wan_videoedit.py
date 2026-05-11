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

        k, _ = self.to_k(context)
        if self.tp_rmsnorm:
            k = tensor_parallel_rms_norm(k, self.norm_k)
        else:
            k = self.norm_k(k)
        k = k.unflatten(2, (self.local_num_heads, self.head_dim))

        v, _ = self.to_v(context)
        v = v.unflatten(2, (self.local_num_heads, self.head_dim))

        x = self.attn(q, k, v).flatten(2)

        if context_img is not None and context_img.shape[1] > 0:
            k_img, _ = self.add_k_proj(context_img)
            if self.tp_rmsnorm:
                k_img = tensor_parallel_rms_norm(k_img, self.norm_added_k)
            else:
                k_img = self.norm_added_k(k_img)
            k_img = k_img.unflatten(2, (self.local_num_heads, self.head_dim))

            v_img, _ = self.add_v_proj(context_img)
            v_img = v_img.unflatten(2, (self.local_num_heads, self.head_dim))
            x = x + self.attn(q, k_img, v_img).flatten(2)

        x, _ = self.to_out(x)
        return x


class WanVideoEditTransformer3DModel(WanTransformer3DModel):
    _fsdp_shard_conditions = WanVideoEditConfig()._fsdp_shard_conditions
    _compile_conditions = WanVideoEditConfig()._compile_conditions
    _supported_attention_backends = WanVideoEditConfig()._supported_attention_backends
    param_names_mapping = WanVideoEditConfig().param_names_mapping
    reverse_param_names_mapping = WanVideoEditConfig().reverse_param_names_mapping
    lora_param_names_mapping = WanVideoEditConfig().lora_param_names_mapping

    def __init__(self, config: WanVideoEditConfig, hf_config, quant_config=None):
        super().__init__(config=config, hf_config=hf_config, quant_config=quant_config)
        for block in self.blocks:
            block.attn2 = WanVideoEditCrossAttention(
                config.hidden_size,
                config.num_attention_heads,
                qk_norm=config.qk_norm,
                eps=config.eps,
                supported_attention_backends={
                    b for b in self._supported_attention_backends if not b.is_sparse
                },
                quant_config=quant_config,
            )


EntryClass = WanVideoEditTransformer3DModel
