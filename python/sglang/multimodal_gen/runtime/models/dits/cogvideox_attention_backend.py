from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0
from diffusers.models.embeddings import apply_rotary_emb
from torch import nn

from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    flash_attn_func,
)


def normalize_cogvideox_attention_backend(backend: str | None) -> str | None:
    if backend is None:
        return None

    normalized = backend.strip().lower()
    alias_map = {
        "fa": "fa",
        "fa2": "fa",
        "fa3": "fa",
        "fa4": "fa",
        "flash": "fa",
        "flash_attn": "fa",
        "flash_attention": "fa",
        "native": "native",
        "torch_native": "native",
        "torch_sdpa": "native",
        "sdpa": "native",
        "sage": "sage_attn",
        "sage_attn": "sage_attn",
        "sage_attn_3": "sage_attn_3",
        "xformers": "xformers",
        "xformers_memory_efficient": "xformers",
    }
    return alias_map.get(normalized, normalized)


def _prepare_cogvideox_qkv(
    attn: Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None,
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    if encoder_hidden_states is None:
        raise ValueError("CogVideoX attention expects encoder_hidden_states.")

    text_seq_length = encoder_hidden_states.size(1)
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    batch_size = hidden_states.shape[0]
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads
    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    if image_rotary_emb is not None:
        query[:, :, text_seq_length:] = apply_rotary_emb(
            query[:, :, text_seq_length:],
            image_rotary_emb,
        )
        if not attn.is_cross_attention:
            key[:, :, text_seq_length:] = apply_rotary_emb(
                key[:, :, text_seq_length:],
                image_rotary_emb,
            )

    return text_seq_length, query, key, value


class CogVideoXNativeAttnProcessor:
    _attention_backend = "native"

    def __init__(self):
        self._processor = CogVideoXAttnProcessor2_0()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._processor(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )


class CogVideoXFlashAttnProcessor:
    _attention_backend = "fa"

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The sglang flash kernel path does not support arbitrary masks on this
        # processor, so keep native SDPA as the exact fallback if masks show up.
        if attention_mask is not None:
            return CogVideoXNativeAttnProcessor()(
                attn=attn,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
            )

        text_seq_length, query, key, value = _prepare_cogvideox_qkv(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        if query.dtype not in (torch.float16, torch.bfloat16):
            return CogVideoXNativeAttnProcessor()(
                attn=attn,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
            )

        batch_size = query.shape[0]
        head_dim = query.shape[-1]
        sequence_length = query.shape[2]

        hidden_states = flash_attn_func(
            q=query.transpose(1, 2).contiguous(),
            k=key.transpose(1, 2).contiguous(),
            v=value.transpose(1, 2).contiguous(),
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=sequence_length,
            max_seqlen_k=sequence_length,
            softmax_scale=getattr(attn, "scale", None),
            causal=False,
        )
        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length],
            dim=1,
        )
        return hidden_states, encoder_hidden_states


def build_cogvideox_attention_processor(backend: str) -> object:
    normalized_backend = normalize_cogvideox_attention_backend(backend)
    if normalized_backend == "native":
        return CogVideoXNativeAttnProcessor()
    if normalized_backend == "fa":
        return CogVideoXFlashAttnProcessor()

    raise ValueError(
        "CogVideoX/VividVR attention backend "
        f"{backend!r} is not supported yet. Supported backends: fa, torch_sdpa."
    )


def set_cogvideox_attention_backend(module: nn.Module, backend: str) -> str:
    normalized_backend = normalize_cogvideox_attention_backend(backend)
    processor = build_cogvideox_attention_processor(backend)
    applied = 0
    for child in module.modules():
        if isinstance(child, Attention):
            child.set_processor(type(processor)())
            applied += 1

    if applied == 0:
        raise ValueError(
            f"No diffusers Attention modules were found while applying backend {backend!r}."
        )

    return normalized_backend


def inspect_cogvideox_attention_backend(module: nn.Module) -> str | None:
    for child in module.modules():
        if not isinstance(child, Attention):
            continue
        processor = getattr(child, "processor", None)
        backend = getattr(processor, "_attention_backend", None)
        if backend is not None:
            return str(backend)
    return None
