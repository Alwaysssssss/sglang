from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace

import torch
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0
from diffusers.models.embeddings import apply_rotary_emb
from torch import nn

from sglang.jit_kernel.diffusion.triton.norm import norm_infer
from sglang.multimodal_gen.runtime.layers.linear import MergedColumnParallelLinear
from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    flash_attn_func,
)
from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    apply_flashinfer_rope_qk_inplace,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum
from sglang.multimodal_gen.runtime.distributed import model_parallel_is_initialized
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_world_size,
)

_COGVIDEOX_FLASHINFER_ROPE_CACHE: dict[tuple[object, ...], torch.Tensor | None] = {}
_COGVIDEOX_LAYERNORM_KERNEL_CACHE: dict[tuple[int, torch.dtype, str], bool] = {}


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
        "fa_sp": "fa_sp",
        "sp_fa": "fa_sp",
        "usp": "fa_sp",
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
    if getattr(attn, "use_fused_qkv", False) and hasattr(attn, "to_qkv"):
        qkv = attn.to_qkv(hidden_states)
        if isinstance(qkv, tuple):
            qkv = qkv[0]
        output_sizes = getattr(attn, "_sglang_qkv_output_sizes", None)
        if output_sizes is None:
            output_sizes = (qkv.shape[-1] // 3,) * 3
        query, key, value = [
            tensor.contiguous()
            for tensor in torch.split(qkv, output_sizes, dim=-1)
        ]
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads
    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    use_qk_norm_fusion = bool(getattr(attn, "_sglang_enable_qk_norm_fusion", False))
    use_qk_norm_rope_fusion = bool(
        getattr(attn, "_sglang_enable_qk_norm_rope_fusion", False)
    )

    if not use_qk_norm_fusion and not use_qk_norm_rope_fusion:
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

    if attn.norm_q is not None:
        query = _apply_cogvideox_qk_norm(
            query,
            attn.norm_q,
            prefer_sglang_kernel=True,
        )
    if attn.norm_k is not None:
        key = _apply_cogvideox_qk_norm(
            key,
            attn.norm_k,
            prefer_sglang_kernel=True,
        )

    if image_rotary_emb is not None and use_qk_norm_rope_fusion:
        query, key = _apply_cogvideox_image_rope(
            attn=attn,
            query=query,
            key=key,
            text_seq_length=text_seq_length,
            image_rotary_emb=image_rotary_emb,
            prefer_sglang_kernel=True,
        )
    elif image_rotary_emb is not None:
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


def _apply_cogvideox_qk_norm(
    x: torch.Tensor,
    norm: nn.Module,
    *,
    prefer_sglang_kernel: bool,
) -> torch.Tensor:
    if not isinstance(norm, nn.LayerNorm):
        return norm(x)

    hidden_size = int(norm.normalized_shape[-1])
    if (
        not prefer_sglang_kernel
        or not current_platform.is_cuda()
        or x.shape[-1] != hidden_size
        or norm.weight is None
        or not _can_use_cogvideox_layernorm_kernel(
            hidden_size=hidden_size,
            dtype=x.dtype,
            device_type=x.device.type,
        )
    ):
        return norm(x)

    try:
        return norm_infer(
            x.reshape(-1, hidden_size),
            norm.weight,
            norm.bias,
            eps=norm.eps,
            is_rms_norm=False,
        ).view_as(x)
    except Exception:
        return norm(x)


def _can_use_cogvideox_layernorm_kernel(
    *,
    hidden_size: int,
    dtype: torch.dtype,
    device_type: str,
) -> bool:
    cache_key = (hidden_size, dtype, device_type)
    cached = _COGVIDEOX_LAYERNORM_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if device_type != "cuda":
        return False

    probe_device = torch.device(device_type)
    probe_input = torch.zeros((1, hidden_size), device=probe_device, dtype=dtype)
    probe_weight = torch.ones(hidden_size, device=probe_device, dtype=dtype)
    probe_bias = torch.zeros(hidden_size, device=probe_device, dtype=dtype)
    try:
        norm_infer(
            probe_input,
            probe_weight,
            probe_bias,
            eps=1e-6,
            is_rms_norm=False,
        )
        torch.cuda.synchronize(probe_device)
        _COGVIDEOX_LAYERNORM_KERNEL_CACHE[cache_key] = True
        return True
    except Exception:
        _COGVIDEOX_LAYERNORM_KERNEL_CACHE[cache_key] = False
        return False


def _can_use_cogvideox_flashinfer_rope(
    attn: Attention,
    query: torch.Tensor,
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    prefer_sglang_kernel: bool,
) -> bool:
    if (
        not prefer_sglang_kernel
        or not current_platform.is_cuda()
        or getattr(attn, "is_cross_attention", False)
        or query.dtype not in (torch.float16, torch.bfloat16)
        or query.dim() != 4
    ):
        return False

    if not (
        isinstance(image_rotary_emb, tuple)
        and len(image_rotary_emb) == 2
        and isinstance(image_rotary_emb[0], torch.Tensor)
        and isinstance(image_rotary_emb[1], torch.Tensor)
    ):
        return False

    cos_sin_cache = _build_cogvideox_flashinfer_cos_sin_cache(image_rotary_emb)
    return (
        cos_sin_cache is not None
        and cos_sin_cache.dim() == 2
        and cos_sin_cache.shape[-1] <= query.shape[-1]
    )


def _build_cogvideox_flashinfer_cos_sin_cache(
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor | None:
    cos, sin = image_rotary_emb
    cache_key = (
        int(cos.data_ptr()),
        int(cos.storage_offset()),
        tuple(cos.shape),
        tuple(cos.stride()),
        str(cos.device),
        str(cos.dtype),
        int(sin.data_ptr()),
        int(sin.storage_offset()),
        tuple(sin.shape),
        tuple(sin.stride()),
        str(sin.device),
        str(sin.dtype),
    )
    if cache_key in _COGVIDEOX_FLASHINFER_ROPE_CACHE:
        return _COGVIDEOX_FLASHINFER_ROPE_CACHE[cache_key]

    if (
        cos.dim() != 2
        or sin.dim() != 2
        or cos.shape != sin.shape
        or cos.shape[-1] % 2 != 0
    ):
        _COGVIDEOX_FLASHINFER_ROPE_CACHE[cache_key] = None
        return None

    # CogVideoX uses diffusers' repeat_interleave_real=True RoPE layout where
    # each rotary frequency appears twice: [c0, c0, c1, c1, ...]. FlashInfer
    # and SGLang rotary kernels expect [cos_half, sin_half].
    cos_pairs = cos.reshape(cos.shape[0], -1, 2)
    sin_pairs = sin.reshape(sin.shape[0], -1, 2)
    if not torch.equal(cos_pairs[..., 0], cos_pairs[..., 1]):
        _COGVIDEOX_FLASHINFER_ROPE_CACHE[cache_key] = None
        return None
    if not torch.equal(sin_pairs[..., 0], sin_pairs[..., 1]):
        _COGVIDEOX_FLASHINFER_ROPE_CACHE[cache_key] = None
        return None

    cache = torch.cat(
        [
            cos_pairs[..., 0].to(dtype=torch.float32).contiguous(),
            sin_pairs[..., 0].to(dtype=torch.float32).contiguous(),
        ],
        dim=-1,
    )
    _COGVIDEOX_FLASHINFER_ROPE_CACHE[cache_key] = cache
    return cache


def _apply_cogvideox_image_rope(
    attn: Attention,
    query: torch.Tensor,
    key: torch.Tensor,
    text_seq_length: int,
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    *,
    prefer_sglang_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query.size(2) <= text_seq_length:
        return query, key

    q_image = query[:, :, text_seq_length:]
    k_image = key[:, :, text_seq_length:]
    if _can_use_cogvideox_flashinfer_rope(
        attn,
        q_image.transpose(1, 2),
        image_rotary_emb,
        prefer_sglang_kernel=prefer_sglang_kernel,
    ):
        cos_sin_cache = _build_cogvideox_flashinfer_cos_sin_cache(image_rotary_emb)
        if cos_sin_cache is None:
            raise RuntimeError(
                "FlashInfer RoPE fast path was selected without a compatible CogVideoX cos/sin cache."
            )
        q_image, k_image = apply_flashinfer_rope_qk_inplace(
            q_image.transpose(1, 2),
            k_image.transpose(1, 2),
            cos_sin_cache,
            is_neox=False,
        )
        query[:, :, text_seq_length:] = q_image.transpose(1, 2)
        key[:, :, text_seq_length:] = k_image.transpose(1, 2)
        return query, key

    query[:, :, text_seq_length:] = apply_rotary_emb(
        q_image,
        image_rotary_emb,
    )
    if not attn.is_cross_attention:
        key[:, :, text_seq_length:] = apply_rotary_emb(
            k_image,
            image_rotary_emb,
        )
    return query, key


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


class CogVideoXSPFlashAttnProcessor:
    """SP-aware CogVideoX joint-attention processor using Ulysses all-to-all.

    When sequence parallelism (SP) is enabled (sp_world_size > 1), text tokens
    are treated as a *replicated prefix* and video tokens as an *SP-sharded
    suffix*.  Joint attention is dispatched through
    ``USPAttention._forward_with_replicated_prefix`` which performs the
    all-to-all shuffle on the sharded suffix only, yielding mathematically
    equivalent results to single-GPU joint attention.

    When SP is not active the processor transparently delegates to the standard
    flash-attention path so single-GPU and non-SP deployments are unaffected.
    """

    _attention_backend = "fa_sp"

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ---- SP not active → delegate to standard flash path ---------------
        sp_size = (
            get_sp_world_size() if model_parallel_is_initialized() else 1
        )
        if sp_size <= 1:
            return CogVideoXFlashAttnProcessor()(
                attn=attn,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
            )

        # ---- SP path: text = replicated prefix, video = sharded suffix ----
        text_seq_length = encoder_hidden_states.size(1)
        batch_size = hidden_states.shape[0]
        num_heads = attn.heads

        # Reuse the existing QKV preparation pipeline (projection + QK norm +
        # RoPE).  Returns [B, H, S_total, D] layout.
        _, query, key, value = _prepare_cogvideox_qkv(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        # query/key/value:  [B, H, S_total, D]   S_total = text + video
        head_dim = query.shape[-1]

        # Get or create a cached USPAttention instance configured for CogVideoX
        # joint attention.  Built lazily so single-GPU runs never pay the cost.
        usp_attn = _get_cogvideox_sp_usp_attention(
            num_heads=num_heads,
            head_size=head_dim,
        )

        # USPAttention expects [B, S_local, H, D] input.
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()

        # num_replicated_prefix informs USPAttention to skip the all-to-all on
        # the first *text_seq_length* tokens (identical across ranks) and only
        # shuffle the video suffix — exactly what CogVideoX joint attention needs.
        out = usp_attn(
            q,
            k,
            v,
            num_replicated_prefix=text_seq_length,
        )
        # out:  [B, S_local, H, D]

        # Output projection + split back to text / video
        out = out.reshape(batch_size, -1, num_heads * head_dim)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        encoder_hidden_states, hidden_states = out.split(
            [text_seq_length, out.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


@lru_cache(maxsize=4)
def _get_cogvideox_sp_usp_attention(
    *,
    num_heads: int,
    head_size: int,
) -> USPAttention:
    """Create a cached USPAttention instance for CogVideoX SP joint attention.

    Uses ``skip_sequence_parallel=False`` (the default) so that the full
    Ulysses all-to-all pipeline runs when SP is active.  When SP is not active
    (world_size == 1), USPAttention internally degrades to local attention.
    """
    return USPAttention(
        num_heads=num_heads,
        head_size=head_size,
        softmax_scale=None,
        causal=False,
        supported_attention_backends={
            AttentionBackendEnum.FA,
            AttentionBackendEnum.FA2,
        },
        prefix=f"cogvideox_sp_attn_{num_heads}_{head_size}",
    )


def build_cogvideox_attention_processor(backend: str) -> object:
    normalized_backend = normalize_cogvideox_attention_backend(backend)
    if normalized_backend == "native":
        return CogVideoXNativeAttnProcessor()
    if normalized_backend == "fa":
        return CogVideoXFlashAttnProcessor()
    if normalized_backend == "fa_sp":
        return CogVideoXSPFlashAttnProcessor()

    raise ValueError(
        "CogVideoX/VividVR attention backend "
        f"{backend!r} is not supported yet. Supported backends: native, fa, fa_sp."
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


def _can_fuse_attention_qkv(attn: Attention) -> bool:
    projections = (
        getattr(attn, "to_q", None),
        getattr(attn, "to_k", None),
        getattr(attn, "to_v", None),
    )
    return all(isinstance(proj, nn.Linear) for proj in projections)


def _resolve_single_process_tp_group() -> object | None:
    if model_parallel_is_initialized():
        return None
    return SimpleNamespace(world_size=1, rank_in_group=0)


def _enable_attention_qkv_fusion(attn: Attention) -> bool:
    if getattr(attn, "use_fused_qkv", False) and hasattr(attn, "to_qkv"):
        return True

    if not _can_fuse_attention_qkv(attn):
        return False

    output_sizes = (
        int(attn.to_q.out_features),
        int(attn.to_k.out_features),
        int(attn.to_v.out_features),
    )
    if (
        attn.to_q.in_features != attn.to_k.in_features
        or attn.to_q.in_features != attn.to_v.in_features
    ):
        return False

    has_bias = attn.to_q.bias is not None
    if (
        (attn.to_k.bias is not None) != has_bias
        or (attn.to_v.bias is not None) != has_bias
    ):
        return False

    if getattr(attn, "is_cross_attention", False):
        return False

    fused_qkv = MergedColumnParallelLinear(
        input_size=int(attn.to_q.in_features),
        output_sizes=list(output_sizes),
        bias=has_bias,
        gather_output=False,
        params_dtype=attn.to_q.weight.dtype,
        prefix="cogvideox_qkv_fusion",
        tp_group=_resolve_single_process_tp_group(),
    ).to(device=attn.to_q.weight.device, dtype=attn.to_q.weight.dtype)

    with torch.no_grad():
        fused_qkv.weight_loader(fused_qkv.weight, attn.to_q.weight.detach(), 0)
        fused_qkv.weight_loader(fused_qkv.weight, attn.to_k.weight.detach(), 1)
        fused_qkv.weight_loader(fused_qkv.weight, attn.to_v.weight.detach(), 2)
        if has_bias and fused_qkv.bias is not None:
            fused_qkv.weight_loader(fused_qkv.bias, attn.to_q.bias.detach(), 0)
            fused_qkv.weight_loader(fused_qkv.bias, attn.to_k.bias.detach(), 1)
            fused_qkv.weight_loader(fused_qkv.bias, attn.to_v.bias.detach(), 2)

    attn.to_qkv = fused_qkv.eval()
    attn.use_fused_qkv = True
    attn._sglang_qkv_output_sizes = output_sizes
    attn._sglang_qkv_fusion_impl = "sglang_merged_column_linear"
    return True


def _enable_attention_qk_norm_fusion(attn: Attention) -> bool:
    if getattr(attn, "_sglang_enable_qk_norm_fusion", False):
        return True

    norms = (getattr(attn, "norm_q", None), getattr(attn, "norm_k", None))
    if all(norm is None for norm in norms):
        return False
    if not all(norm is None or isinstance(norm, nn.LayerNorm) for norm in norms):
        return False

    attn._sglang_enable_qk_norm_fusion = True
    attn._sglang_qk_norm_fusion_impl = "sglang_layernorm"
    return True


def _enable_attention_qk_norm_rope_fusion(attn: Attention) -> bool:
    if getattr(attn, "_sglang_enable_qk_norm_rope_fusion", False):
        return True

    norms = (getattr(attn, "norm_q", None), getattr(attn, "norm_k", None))
    if not all(norm is None or isinstance(norm, nn.LayerNorm) for norm in norms):
        return False

    attn._sglang_enable_qk_norm_rope_fusion = True
    attn._sglang_qk_norm_rope_fusion_impl = "sglang_layernorm+rope_accel"
    return True


def enable_cogvideox_qkv_fusion(module: nn.Module) -> int:
    applied = 0
    for child in module.modules():
        if isinstance(child, Attention) and _enable_attention_qkv_fusion(child):
            applied += 1

    if applied == 0:
        raise ValueError(
            "No fuseable diffusers Attention modules were found while enabling CogVideoX QKV fusion."
        )

    return applied


def enable_cogvideox_qk_norm_fusion(module: nn.Module) -> int:
    applied = 0
    for child in module.modules():
        if isinstance(child, Attention) and _enable_attention_qk_norm_fusion(child):
            applied += 1

    if applied == 0:
        raise ValueError(
            "No compatible diffusers Attention modules were found while enabling CogVideoX QK-norm fusion."
        )

    return applied


def enable_cogvideox_qk_norm_rope_fusion(module: nn.Module) -> int:
    applied = 0
    for child in module.modules():
        if isinstance(child, Attention) and _enable_attention_qk_norm_rope_fusion(child):
            applied += 1

    if applied == 0:
        raise ValueError(
            "No compatible diffusers Attention modules were found while enabling CogVideoX QK-norm/RoPE fusion."
        )

    return applied


def inspect_cogvideox_qkv_fusion(module: nn.Module) -> str | None:
    for child in module.modules():
        if not isinstance(child, Attention):
            continue
        if getattr(child, "use_fused_qkv", False) and hasattr(child, "to_qkv"):
            return str(
                getattr(
                    child,
                    "_sglang_qkv_fusion_impl",
                    "sglang_merged_column_linear",
                )
            )
    return None


def inspect_cogvideox_qk_norm_fusion(module: nn.Module) -> str | None:
    for child in module.modules():
        if not isinstance(child, Attention):
            continue
        impl = getattr(child, "_sglang_qk_norm_fusion_impl", None)
        if impl is not None:
            return str(impl)
    return None


def inspect_cogvideox_qk_norm_rope_fusion(module: nn.Module) -> str | None:
    for child in module.modules():
        if not isinstance(child, Attention):
            continue
        impl = getattr(child, "_sglang_qk_norm_rope_fusion_impl", None)
        if impl is not None:
            return str(impl)
    return None


def inspect_cogvideox_attention_backend(module: nn.Module) -> str | None:
    for child in module.modules():
        if not isinstance(child, Attention):
            continue
        processor = getattr(child, "processor", None)
        backend = getattr(processor, "_attention_backend", None)
        if backend is not None:
            return str(backend)
    return None
