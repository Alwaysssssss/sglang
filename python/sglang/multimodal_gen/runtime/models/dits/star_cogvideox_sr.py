# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.configs.models.dits.star_cogvideox_sr import (
    StarCogVideoXSRDiTConfig,
)
from sglang.multimodal_gen.runtime.distributed import (
    divide,
    get_tp_world_size,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_sp_world_size,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention, USPAttention
from sglang.multimodal_gen.runtime.layers.layernorm import LayerNorm
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    apply_flashinfer_rope_qk_inplace,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin

try:
    from cache_dit import ForwardPattern
    from cache_dit.caching.block_adapters import BlockAdapter, BlockAdapterRegister
except Exception:
    ForwardPattern = None
    BlockAdapter = None
    BlockAdapterRegister = None


def _ensure_contiguous(tensor: torch.Tensor | None) -> torch.Tensor | None:
    return tensor.contiguous() if tensor is not None else None


def _get_star_pipeline_option(option_name: str, default: Any) -> Any:
    try:
        pipeline_config = getattr(get_global_server_args(), "pipeline_config", None)
    except Exception:
        pipeline_config = None
    if pipeline_config is None:
        return default
    return getattr(pipeline_config, option_name, default)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.view(*x.shape[:-1], -1, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _build_rotary_cache(
    *,
    compressed_num_frames: int,
    height: int,
    width: int,
    head_dim: int,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    dim_t = head_dim // 4
    dim_h = head_dim // 8 * 3
    dim_w = head_dim // 8 * 3

    def _axis_freqs(size: int, dim: int) -> torch.Tensor:
        base = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        coords = torch.arange(size, dtype=torch.float32)
        freqs = torch.einsum("n,d->nd", coords, base)
        return freqs.repeat_interleave(2, dim=-1)

    freqs_t = _axis_freqs(compressed_num_frames, dim_t)[:, None, None, :]
    freqs_h = _axis_freqs(height, dim_h)[None, :, None, :]
    freqs_w = _axis_freqs(width, dim_w)[None, None, :, :]

    freqs = torch.cat(
        [
            freqs_t.expand(compressed_num_frames, height, width, -1),
            freqs_h.expand(compressed_num_frames, height, width, -1),
            freqs_w.expand(compressed_num_frames, height, width, -1),
        ],
        dim=-1,
    ).reshape(compressed_num_frames * height * width, head_dim)
    return freqs.sin().contiguous(), freqs.cos().contiguous()


def _build_flashinfer_rotary_cache(
    freqs_sin: torch.Tensor,
    freqs_cos: torch.Tensor,
) -> torch.Tensor:
    # STAR's RoPE uses adjacent-pair rotation, so FlashInfer expects the
    # de-duplicated half-dim cos/sin cache concatenated along the last dim.
    cos_half = freqs_cos[:, 0::2].to(dtype=torch.float32).contiguous()
    sin_half = freqs_sin[:, 0::2].to(dtype=torch.float32).contiguous()
    return torch.cat([cos_half, sin_half], dim=-1)


def _reset_linear_parameters(module: nn.Module, fan_in: int) -> None:
    weight = getattr(module, "weight", None)
    if weight is not None:
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    bias = getattr(module, "bias", None)
    if bias is not None:
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(bias, -bound, bound)


class _TensorOnlyReplicatedLinear(ReplicatedLinear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        _reset_linear_parameters(self, self.input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = super().forward(x)
        return output


class _TensorOnlyColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        _reset_linear_parameters(self, self.input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = super().forward(x)
        return output


class _TensorOnlyMergedColumnParallelLinear(MergedColumnParallelLinear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        _reset_linear_parameters(self, self.input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = super().forward(x)
        return output


class _TensorOnlyRowParallelLinear(RowParallelLinear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        _reset_linear_parameters(self, self.input_size_per_partition)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = super().forward(x)
        return output


class _WrappedLinear(nn.Module):
    def __init__(self, original: nn.Module) -> None:
        super().__init__()
        self.original = original

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x)


class _StarLayerNorm(LayerNorm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.shape[-1] == self.hidden_size:
            return super().forward(x.contiguous() if not x.is_contiguous() else x)
        else:
            weight = self.weight.to(device=x.device, dtype=torch.float32)
            bias = (
                self.bias.to(device=x.device, dtype=torch.float32)
                if self.bias is not None
                else None
            )
            return F.layer_norm(
                x.to(torch.float32),
                (self.hidden_size,),
                weight,
                bias,
                self.eps,
            ).to(dtype=x.dtype)
        return super().forward(x)

    def forward_scale_shift(
        self,
        x: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        if x.numel() == 0:
            return x
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
        if x.is_cuda and x.shape[-1] % 256 == 0:
            from sglang.jit_kernel.diffusion.cutedsl.scale_residual_norm_scale_shift import (
                fused_norm_scale_shift,
            )

            out = fused_norm_scale_shift(
                x.contiguous(),
                _ensure_contiguous(self.weight),
                _ensure_contiguous(self.bias),
                scale.contiguous(),
                shift.contiguous(),
                "layer",
                self.eps,
            )
            return out.to(dtype=x.dtype)
        return _modulate(self.forward(x), shift[:, 0, :], scale[:, 0, :]).to(dtype=x.dtype)

    def forward_residual_scale_shift(
        self,
        residual: torch.Tensor,
        x: torch.Tensor,
        gate: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual.numel() == 0:
            return self.forward_scale_shift(residual, shift, scale), residual
        gate = gate.unsqueeze(1)
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
        if residual.is_cuda and residual.shape[-1] % 256 == 0:
            from sglang.jit_kernel.diffusion.cutedsl.scale_residual_norm_scale_shift import (
                fused_scale_residual_norm_scale_shift,
            )

            modulated, residual_output = fused_scale_residual_norm_scale_shift(
                residual.contiguous(),
                x.contiguous(),
                gate.contiguous(),
                _ensure_contiguous(self.weight),
                _ensure_contiguous(self.bias),
                scale.contiguous(),
                shift.contiguous(),
                "layer",
                self.eps,
            )
            return modulated.to(dtype=residual.dtype), residual_output.to(
                dtype=residual.dtype
            )
        residual_output = residual + x * gate
        modulated = self.forward_scale_shift(
            residual_output,
            shift[:, 0, :],
            scale[:, 0, :],
        )
        return modulated.to(dtype=residual.dtype), residual_output.to(dtype=residual.dtype)


class _SpatialLocalEnhancer(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            pooled = torch.cat(
                [
                    x.amax(dim=1, keepdim=True),
                    x.mean(dim=1, keepdim=True),
                ],
                dim=1,
            )
            weight = self.conv1.weight.unsqueeze(2)
            gate = torch.sigmoid(
                F.conv3d(
                    pooled,
                    weight,
                    bias=None,
                    padding=(0, self.conv1.padding[0], self.conv1.padding[1]),
                )
            )
            return x * gate
        pooled = torch.cat(
            [
                x.amax(dim=1, keepdim=True),
                x.mean(dim=1, keepdim=True),
            ],
            dim=1,
        )
        gate = torch.sigmoid(self.conv1(pooled))
        return x * gate


class _TemporalLocalEnhancer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Linear(2, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            pooled = torch.cat(
                [
                    x.amax(dim=1, keepdim=True),
                    x.mean(dim=1, keepdim=True),
                ],
                dim=1,
            )
            weight = self.conv1.weight.view(1, 2, 1, 1, 1)
            gate = torch.sigmoid(F.conv3d(pooled, weight, bias=None))
            return x * gate
        pooled = torch.stack([x.amax(dim=-1), x.mean(dim=-1)], dim=-1)
        gate = torch.sigmoid(self.conv1(pooled))
        return x * gate


class _StarPatchEmbedMixin(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        hidden_size: int,
        patch_size: int,
        text_hidden_size: int,
        quant_config: QuantizationConfig | None,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj_sr = nn.Conv2d(
            in_channels * 2,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self.text_proj = _TensorOnlyReplicatedLinear(
            text_hidden_size,
            hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix="mixins.patch_embed.text_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor, int, int, int]:
        batch_size, channels, num_frames, height, width = hidden_states.shape
        frames = hidden_states.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames,
            channels,
            height,
            width,
        )
        video_tokens = self.proj_sr(frames)
        grid_h, grid_w = video_tokens.shape[-2:]
        video_tokens = video_tokens.reshape(
            batch_size,
            num_frames,
            video_tokens.shape[1],
            grid_h * grid_w,
        ).permute(0, 1, 3, 2)
        video_tokens = video_tokens.reshape(batch_size, num_frames * grid_h * grid_w, -1)

        if encoder_hidden_states is None:
            return video_tokens, num_frames, grid_h, grid_w

        if encoder_hidden_states.shape[-1] == self.text_proj.input_size:
            text_tokens = self.text_proj(encoder_hidden_states)
        elif encoder_hidden_states.shape[-1] == self.text_proj.output_size:
            text_tokens = encoder_hidden_states
        else:
            raise ValueError(
                "encoder_hidden_states last dimension must match either "
                f"text_hidden_size={self.text_proj.input_size} or hidden_size={self.text_proj.output_size}, "
                f"got {encoder_hidden_states.shape[-1]}"
            )
        return torch.cat([text_tokens, video_tokens], dim=1), num_frames, grid_h, grid_w


class _StarPosEmbedMixin(nn.Module):
    def __init__(
        self,
        *,
        compressed_num_frames: int,
        height: int,
        width: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        freqs_sin, freqs_cos = _build_rotary_cache(
            compressed_num_frames=compressed_num_frames,
            height=height,
            width=width,
            head_dim=head_dim,
        )
        self.register_buffer("freqs_sin", freqs_sin, persistent=True)
        self.register_buffer("freqs_cos", freqs_cos, persistent=True)

    def get_rotary_cache(
        self,
        *,
        image_token_count: int,
        head_dim: int,
        num_frames: int,
        grid_h: int,
        grid_w: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.freqs_sin.shape[0] >= image_token_count:
            freqs_sin = self.freqs_sin[:image_token_count]
            freqs_cos = self.freqs_cos[:image_token_count]
            return (
                freqs_sin.to(device=device),
                freqs_cos.to(device=device),
                _build_flashinfer_rotary_cache(freqs_sin, freqs_cos).to(device=device),
            )
        freqs_sin, freqs_cos = _build_rotary_cache(
            compressed_num_frames=num_frames,
            height=grid_h,
            width=grid_w,
            head_dim=head_dim,
        )
        freqs_cos_sin_cache = _build_flashinfer_rotary_cache(freqs_sin, freqs_cos)
        return (
            freqs_sin.to(device=device),
            freqs_cos.to(device=device),
            freqs_cos_sin_cache.to(device=device),
        )


class _StarAdaLNMixin(nn.Module):
    def __init__(
        self,
        *,
        num_layers: int,
        time_embed_dim: int,
        hidden_size: int,
        head_dim: int,
        qk_ln: bool,
        elementwise_affine: bool,
        quant_config: QuantizationConfig | None,
    ) -> None:
        super().__init__()
        self.qk_ln = qk_ln
        self.adaLN_modulations = nn.ModuleList(
            [
                nn.Sequential(
                    nn.SiLU(),
                    _TensorOnlyReplicatedLinear(
                        time_embed_dim,
                        12 * hidden_size,
                        bias=True,
                        quant_config=quant_config,
                        prefix=f"mixins.adaln_layer.adaLN_modulations.{layer_idx}.1",
                    ),
                )
                for layer_idx in range(num_layers)
            ]
        )
        if qk_ln:
            self.query_layernorm_list = nn.ModuleList(
                [
                    _StarLayerNorm(
                        head_dim,
                        eps=1e-6,
                        elementwise_affine=elementwise_affine,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.key_layernorm_list = nn.ModuleList(
                [
                    _StarLayerNorm(
                        head_dim,
                        eps=1e-6,
                        elementwise_affine=elementwise_affine,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.query_layernorm_list = nn.ModuleList()
            self.key_layernorm_list = nn.ModuleList()


class _StarFinalLayerMixin(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        time_embed_dim: int,
        patch_size: int,
        out_channels: int,
        elementwise_affine: bool,
        quant_config: QuantizationConfig | None,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.norm_final = _StarLayerNorm(
            hidden_size,
            eps=1e-6,
            elementwise_affine=elementwise_affine,
        )
        self.linear = _TensorOnlyReplicatedLinear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True,
            quant_config=quant_config,
            prefix="mixins.final_layer.linear",
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            _TensorOnlyReplicatedLinear(
                time_embed_dim,
                2 * hidden_size,
                bias=True,
                quant_config=quant_config,
                prefix="mixins.final_layer.adaLN_modulation.1",
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor,
        *,
        text_length: int,
        num_frames: int,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        img_hidden_states = hidden_states[:, text_length:, :]
        shift, scale = self.adaLN_modulation(emb).chunk(2, dim=1)
        img_hidden_states = self.norm_final.forward_scale_shift(
            img_hidden_states,
            shift,
            scale,
        ).to(orig_dtype)
        img_hidden_states = self.linear(img_hidden_states)

        batch_size = img_hidden_states.shape[0]
        patch = self.patch_size
        out_channels = self.out_channels
        img_hidden_states = img_hidden_states.view(
            batch_size,
            num_frames,
            grid_h,
            grid_w,
            out_channels,
            patch,
            patch,
        )
        img_hidden_states = img_hidden_states.permute(0, 4, 1, 2, 5, 3, 6)
        return img_hidden_states.reshape(
            batch_size,
            out_channels,
            num_frames,
            grid_h * patch,
            grid_w * patch,
        )


class _StarAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        quant_config: QuantizationConfig | None,
        supported_attention_backends: set[AttentionBackendEnum],
        prefix: str,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.runtime_parallel_initialized = model_parallel_is_initialized()
        self.tp_size = (
            get_tp_world_size() if self.runtime_parallel_initialized else 1
        )
        self.sp_size = (
            get_sp_world_size() if self.runtime_parallel_initialized else 1
        )
        self.ulysses_size = (
            get_ulysses_parallel_world_size()
            if self.runtime_parallel_initialized
            else 1
        )
        self.ring_size = (
            get_ring_parallel_world_size() if self.runtime_parallel_initialized else 1
        )
        self.parallelized = self.runtime_parallel_initialized and max(
            self.tp_size,
            self.sp_size,
            self.ulysses_size,
            self.ring_size,
        ) > 1
        self.local_num_heads = divide(num_attention_heads, self.tp_size)
        self.head_dim = hidden_size // num_attention_heads
        try:
            self.use_flashinfer_rope = bool(
                _get_star_pipeline_option("use_flashinfer_rope", False)
            )
        except Exception:
            self.use_flashinfer_rope = False
        if self.parallelized:
            query_key_value = _TensorOnlyMergedColumnParallelLinear(
                hidden_size,
                [hidden_size, hidden_size, hidden_size],
                bias=True,
                gather_output=False,
                quant_config=quant_config,
                prefix=f"{prefix}.query_key_value.original",
            )
            dense = _TensorOnlyRowParallelLinear(
                hidden_size,
                hidden_size,
                bias=True,
                input_is_parallel=True,
                quant_config=quant_config,
                prefix=f"{prefix}.dense.original",
            )
            self.attn = USPAttention(
                num_heads=self.local_num_heads,
                head_size=self.head_dim,
                causal=False,
                dropout_rate=0.0,
                supported_attention_backends=supported_attention_backends,
                prefix=f"{prefix}.usp_attn",
            )
        else:
            query_key_value = _TensorOnlyReplicatedLinear(
                hidden_size,
                hidden_size * 3,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.query_key_value.original",
            )
            dense = _TensorOnlyReplicatedLinear(
                hidden_size,
                hidden_size,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.dense.original",
            )
            # Even on a single GPU, use the SGLang attention abstraction so the
            # runtime can honor backend selection such as FlashAttention.
            self.attn = LocalAttention(
                num_heads=self.local_num_heads,
                head_size=self.head_dim,
                causal=False,
                supported_attention_backends=supported_attention_backends,
                prefix=f"{prefix}.local_attn",
            )
        self.query_key_value = _WrappedLinear(query_key_value)
        self.dense = _WrappedLinear(dense)

    def _apply_rotary(
        self,
        tensor: torch.Tensor,
        *,
        text_length: int,
        freqs_sin: torch.Tensor | None,
        freqs_cos: torch.Tensor | None,
    ) -> torch.Tensor:
        if freqs_sin is None or freqs_cos is None or tensor.shape[1] <= text_length:
            return tensor
        image_tokens = tensor[:, text_length:, :, :]
        sin = freqs_sin[: image_tokens.shape[1]].to(
            device=image_tokens.device,
            dtype=image_tokens.dtype,
        )[None, :, None, :]
        cos = freqs_cos[: image_tokens.shape[1]].to(
            device=image_tokens.device,
            dtype=image_tokens.dtype,
        )[None, :, None, :]
        rotated = image_tokens * cos + _rotate_half(image_tokens) * sin
        return torch.cat([tensor[:, :text_length, :, :], rotated], dim=1)

    def _apply_rotary_pair(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        text_length: int,
        freqs_sin: torch.Tensor | None,
        freqs_cos: torch.Tensor | None,
        freqs_cos_sin_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if freqs_sin is None or freqs_cos is None or query.shape[1] <= text_length:
            return query, key

        image_length = query.shape[1] - text_length
        if (
            self.use_flashinfer_rope
            and freqs_cos_sin_cache is not None
            and query.is_cuda
            and key.is_cuda
            and freqs_cos_sin_cache.shape[0] >= image_length
        ):
            q_img, k_img = apply_flashinfer_rope_qk_inplace(
                query[:, text_length:, :, :].contiguous(),
                key[:, text_length:, :, :].contiguous(),
                freqs_cos_sin_cache[:image_length].to(device=query.device),
                is_neox=False,
            )
            if text_length == 0:
                return q_img, k_img
            return (
                torch.cat([query[:, :text_length, :, :], q_img], dim=1),
                torch.cat([key[:, :text_length, :, :], k_img], dim=1),
            )

        return (
            self._apply_rotary(
                query,
                text_length=text_length,
                freqs_sin=freqs_sin,
                freqs_cos=freqs_cos,
            ),
            self._apply_rotary(
                key,
                text_length=text_length,
                freqs_sin=freqs_sin,
                freqs_cos=freqs_cos,
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        text_length: int,
        query_layernorm: nn.Module | None,
        key_layernorm: nn.Module | None,
        freqs_sin: torch.Tensor | None,
        freqs_cos: torch.Tensor | None,
        freqs_cos_sin_cache: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.query_key_value(hidden_states).view(
            batch_size,
            seq_len,
            3,
            self.local_num_heads,
            self.head_dim,
        )
        query = qkv[:, :, 0]
        key = qkv[:, :, 1]
        value = qkv[:, :, 2]

        if query_layernorm is not None:
            query = query_layernorm(query)
        if key_layernorm is not None:
            key = key_layernorm(key)

        query, key = self._apply_rotary_pair(
            query,
            key,
            text_length=text_length,
            freqs_sin=freqs_sin,
            freqs_cos=freqs_cos,
            freqs_cos_sin_cache=freqs_cos_sin_cache,
        )
        target_dtype = hidden_states.dtype
        query = query.to(dtype=target_dtype)
        key = key.to(dtype=target_dtype)
        value = value.to(dtype=target_dtype)

        attn_output = self.attn(query, key, value)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        return self.dense(attn_output)


class _StarMLP(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        mlp_ratio: float,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        inner_dim = int(hidden_size * mlp_ratio)
        if model_parallel_is_initialized():
            self.dense_h_to_4h = _TensorOnlyColumnParallelLinear(
                hidden_size,
                inner_dim,
                bias=True,
                gather_output=False,
                quant_config=quant_config,
                prefix=f"{prefix}.dense_h_to_4h",
            )
            self.dense_4h_to_h = _TensorOnlyRowParallelLinear(
                inner_dim,
                hidden_size,
                bias=True,
                input_is_parallel=True,
                quant_config=quant_config,
                prefix=f"{prefix}.dense_4h_to_h",
            )
        else:
            self.dense_h_to_4h = _TensorOnlyReplicatedLinear(
                hidden_size,
                inner_dim,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.dense_h_to_4h",
            )
            self.dense_4h_to_h = _TensorOnlyReplicatedLinear(
                inner_dim,
                hidden_size,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.dense_4h_to_h",
            )
        self.activation = nn.GELU(approximate="tanh")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.dense_4h_to_h(hidden_states)


class _StarTransformerLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        mlp_ratio: float,
        elementwise_affine: bool,
        local_spatial_kernel_size: int,
        quant_config: QuantizationConfig | None,
        supported_attention_backends: set[AttentionBackendEnum],
        prefix: str,
    ) -> None:
        super().__init__()
        self.input_layernorm = _StarLayerNorm(
            hidden_size,
            eps=1e-6,
            elementwise_affine=elementwise_affine,
        )
        self.attention = _StarAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            quant_config=quant_config,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attention",
        )
        self.post_attention_layernorm = _StarLayerNorm(
            hidden_size,
            eps=1e-6,
            elementwise_affine=elementwise_affine,
        )
        self.mlp = _StarMLP(
            hidden_size=hidden_size,
            mlp_ratio=mlp_ratio,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.spa_local = _SpatialLocalEnhancer(local_spatial_kernel_size)
        self.temp_local = _TemporalLocalEnhancer()
        self.local_enhancer_mode = str(
            _get_star_pipeline_option("local_enhancer_mode", "legacy")
        )

    @staticmethod
    def _apply_local_enhancers(
        img_hidden_states: torch.Tensor,
        layer: "_StarTransformerLayer",
        *,
        num_frames: int,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        batch_size, token_count, hidden_size = img_hidden_states.shape
        if token_count != num_frames * grid_h * grid_w:
            return img_hidden_states
        if layer.local_enhancer_mode == "fused_5d":
            video_hidden = img_hidden_states.view(
                batch_size,
                num_frames,
                grid_h,
                grid_w,
                hidden_size,
            ).permute(0, 4, 1, 2, 3).contiguous()
            video_hidden = layer.temp_local(layer.spa_local(video_hidden))
            return video_hidden.permute(0, 2, 3, 4, 1).reshape(
                batch_size,
                token_count,
                hidden_size,
            )
        spatial_hidden = img_hidden_states.view(
            batch_size, num_frames, grid_h, grid_w, hidden_size
        ).permute(0, 1, 4, 2, 3).reshape(
            batch_size * num_frames, hidden_size, grid_h, grid_w
        )
        spatial_hidden = layer.spa_local(spatial_hidden)
        temporal_hidden = spatial_hidden.view(
            batch_size, num_frames, hidden_size, grid_h, grid_w
        ).permute(0, 3, 4, 1, 2).reshape(
            batch_size * grid_h * grid_w, num_frames, hidden_size
        )
        temporal_hidden = layer.temp_local(temporal_hidden)
        return temporal_hidden.view(
            batch_size, grid_h, grid_w, num_frames, hidden_size
        ).permute(0, 3, 1, 2, 4).reshape(batch_size, token_count, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        emb: torch.Tensor,
        text_length: int,
        modulation: torch.Tensor,
        query_layernorm: nn.Module | None,
        key_layernorm: nn.Module | None,
        freqs_sin: torch.Tensor | None,
        freqs_cos: torch.Tensor | None,
        freqs_cos_sin_cache: torch.Tensor | None,
        num_frames: int,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        text_hidden = hidden_states[:, :text_length, :]
        img_hidden = hidden_states[:, text_length:, :]

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            text_shift_msa,
            text_scale_msa,
            text_gate_msa,
            text_shift_mlp,
            text_scale_mlp,
            text_gate_mlp,
        ) = modulation.chunk(12, dim=1)

        img_attn_input = self.input_layernorm.forward_scale_shift(
            img_hidden,
            shift_msa,
            scale_msa,
        ).to(orig_dtype)
        text_attn_input = self.input_layernorm.forward_scale_shift(
            text_hidden,
            text_shift_msa,
            text_scale_msa,
        ).to(orig_dtype)
        img_attn_input = self._apply_local_enhancers(
            img_attn_input,
            self,
            num_frames=num_frames,
            grid_h=grid_h,
            grid_w=grid_w,
        ).to(orig_dtype)

        attn_input = torch.cat([text_attn_input, img_attn_input], dim=1)
        attn_output = self.attention(
            attn_input,
            text_length=text_length,
            query_layernorm=query_layernorm,
            key_layernorm=key_layernorm,
            freqs_sin=freqs_sin,
            freqs_cos=freqs_cos,
            freqs_cos_sin_cache=freqs_cos_sin_cache,
        ).to(orig_dtype)

        text_attn_output = attn_output[:, :text_length, :].to(orig_dtype)
        img_attn_output = attn_output[:, text_length:, :].to(orig_dtype)

        text_mlp_input, text_hidden = self.post_attention_layernorm.forward_residual_scale_shift(
            text_hidden,
            text_attn_output,
            text_gate_msa,
            text_shift_mlp,
            text_scale_mlp,
        )
        img_mlp_input, img_hidden = self.post_attention_layernorm.forward_residual_scale_shift(
            img_hidden,
            img_attn_output,
            gate_msa,
            shift_mlp,
            scale_mlp,
        )
        mlp_output = self.mlp(torch.cat([text_mlp_input, img_mlp_input], dim=1)).to(
            orig_dtype
        )
        text_hidden = text_hidden + text_gate_mlp.unsqueeze(1) * mlp_output[:, :text_length, :]
        img_hidden = img_hidden + gate_mlp.unsqueeze(1) * mlp_output[:, text_length:, :]
        return torch.cat([text_hidden, img_hidden], dim=1)


class _StarTransformerStack(nn.Module):
    def __init__(
        self,
        layers: nn.ModuleList,
        *,
        hidden_size: int,
        elementwise_affine: bool,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.final_layernorm = _StarLayerNorm(
            hidden_size,
            eps=1e-6,
            elementwise_affine=elementwise_affine,
        )


class _StarMixins(nn.Module):
    def __init__(
        self,
        *,
        patch_embed: _StarPatchEmbedMixin,
        pos_embed: _StarPosEmbedMixin,
        adaln_layer: _StarAdaLNMixin,
        final_layer: _StarFinalLayerMixin,
    ) -> None:
        super().__init__()
        self.patch_embed = patch_embed
        self.pos_embed = pos_embed
        self.adaln_layer = adaln_layer
        self.final_layer = final_layer


class StarCogVideoXSRTransformer3DModel(CachableDiT, OffloadableDiTMixin):
    _aliases = ["StarCogVideoXSRTransformer3DModel"]
    _fsdp_shard_conditions = StarCogVideoXSRDiTConfig().arch_config._fsdp_shard_conditions
    _compile_conditions = StarCogVideoXSRDiTConfig().arch_config._compile_conditions
    param_names_mapping: dict[str, str] = {}
    reverse_param_names_mapping: dict[str, str] = {}
    lora_param_names_mapping: dict[str, str] = {}
    layer_names = ["layers"]
    _CFG_SUPPORTED_PREFIXES = set(CachableDiT._CFG_SUPPORTED_PREFIXES) | {
        "star_cogvideox_sr"
    }
    _supported_attention_backends: set[AttentionBackendEnum] = {
        AttentionBackendEnum.TORCH_SDPA,
        AttentionBackendEnum.FA,
        AttentionBackendEnum.AITER,
        AttentionBackendEnum.SAGE_ATTN,
        AttentionBackendEnum.SAGE_ATTN_3,
    }

    @classmethod
    def get_fp8_keep_bf16_patterns(cls) -> list[str]:
        return [
            r"time_embed\.(0|2)",
            r"mixins\.patch_embed\.text_proj",
            r"mixins\.adaln_layer\.adaLN_modulations\.\d+\.1",
            r"mixins\.adaln_layer\.(query_layernorm_list|key_layernorm_list)\.\d+",
            r"mixins\.final_layer\.adaLN_modulation\.1",
            r"mixins\.final_layer\.linear",
            r"transformer\.final_layernorm",
            r"transformer\.layers\.\d+\.(input_layernorm|post_attention_layernorm)",
            r"transformer\.layers\.\d+\.(spa_local|temp_local)\.conv1",
        ]

    @classmethod
    def get_phase7_quantization_presets(cls) -> dict[str, dict[str, Any]]:
        return {
            "fp8_attn_mlp_only": {
                "keep_bf16_patterns": cls.get_fp8_keep_bf16_patterns(),
                "notes": (
                    "Preserves timestep/modulation/local-enhancer/final-layer paths in "
                    "BF16 while allowing attention and MLP linears to stay quantized."
                ),
            },
            "nunchaku_svdquant": cls.get_nunchaku_quant_rules(),
        }

    @classmethod
    def get_nunchaku_quant_rules(cls) -> dict[str, list[str]]:
        return {
            "skip": [
                "norm",
                "pos_embed",
                "proj_sr",
                "spa_local",
                "temp_local",
            ],
            "svdq_w4a4": [
                "attention.query_key_value.original",
                "attention.dense.original",
                "mlp.dense_h_to_4h",
                "mlp.dense_4h_to_h",
            ],
            "awq_w4a16": [
                "time_embed",
                "adaLN_modulations",
                "text_proj",
                "final_layer.linear",
            ],
        }

    def __init__(
        self,
        config: StarCogVideoXSRDiTConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)
        arch = config.arch_config

        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.out_channels
        self.total_in_channels = arch.in_channels * 2
        self.out_channels = arch.out_channels
        self.patch_size = arch.patch_size
        self.time_embed_dim = arch.time_embed_dim
        self.text_length = arch.text_length
        self.head_dim = arch.hidden_size // arch.num_attention_heads
        self.quant_config = quant_config

        self.time_embed = nn.Sequential(
            _TensorOnlyReplicatedLinear(
                arch.hidden_size,
                arch.time_embed_dim,
                bias=True,
                quant_config=quant_config,
                prefix="time_embed.0",
            ),
            nn.SiLU(),
            _TensorOnlyReplicatedLinear(
                arch.time_embed_dim,
                arch.time_embed_dim,
                bias=True,
                quant_config=quant_config,
                prefix="time_embed.2",
            ),
        )

        patch_embed = _StarPatchEmbedMixin(
            in_channels=arch.in_channels,
            hidden_size=arch.hidden_size,
            patch_size=arch.patch_size,
            text_hidden_size=arch.text_hidden_size,
            quant_config=quant_config,
        )
        pos_embed = _StarPosEmbedMixin(
            compressed_num_frames=((arch.num_frames - 1) // arch.time_compressed_rate + 1),
            height=arch.latent_height // arch.patch_size,
            width=arch.latent_width // arch.patch_size,
            head_dim=self.head_dim,
        )
        adaln_layer = _StarAdaLNMixin(
            num_layers=arch.num_layers,
            time_embed_dim=arch.time_embed_dim,
            hidden_size=arch.hidden_size,
            head_dim=self.head_dim,
            qk_ln=arch.qk_ln,
            elementwise_affine=arch.elementwise_affine,
            quant_config=quant_config,
        )
        final_layer = _StarFinalLayerMixin(
            hidden_size=arch.hidden_size,
            time_embed_dim=arch.time_embed_dim,
            patch_size=arch.patch_size,
            out_channels=arch.out_channels,
            elementwise_affine=arch.elementwise_affine,
            quant_config=quant_config,
        )
        self.mixins = _StarMixins(
            patch_embed=patch_embed,
            pos_embed=pos_embed,
            adaln_layer=adaln_layer,
            final_layer=final_layer,
        )
        self.transformer = _StarTransformerStack(
            nn.ModuleList(
                [
                    _StarTransformerLayer(
                        hidden_size=arch.hidden_size,
                        num_attention_heads=arch.num_attention_heads,
                        mlp_ratio=arch.mlp_ratio,
                        elementwise_affine=arch.elementwise_affine,
                        local_spatial_kernel_size=arch.local_spatial_kernel_size,
                        quant_config=quant_config,
                        supported_attention_backends=self._supported_attention_backends,
                        prefix=f"transformer.layers.{layer_idx}",
                    )
                    for layer_idx in range(arch.num_layers)
                ]
            ),
            hidden_size=arch.hidden_size,
            elementwise_affine=arch.elementwise_affine,
        )
        self.__post_init__()

    @property
    def layers(self) -> nn.ModuleList:
        return self.transformer.layers

    def _timestep_embedding(
        self,
        timestep: torch.Tensor,
        dim: int,
        max_period: int = 10000,
    ) -> torch.Tensor:
        half = dim // 2
        exponent = -math.log(max_period) * torch.arange(
            half,
            device=timestep.device,
            dtype=torch.float32,
        ) / max(half, 1)
        freqs = torch.exp(exponent)
        args = timestep.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def _coerce_text_tensor(
        self,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor] | None,
    ) -> torch.Tensor | None:
        if encoder_hidden_states is None:
            return None
        if isinstance(encoder_hidden_states, list):
            if len(encoder_hidden_states) == 0:
                return None
            encoder_hidden_states = encoder_hidden_states[0]
        return encoder_hidden_states

    def _get_model_dtype(self) -> torch.dtype:
        return self.time_embed[0].weight.dtype

    def should_skip_forward_for_cached_states(self, *, emb: torch.Tensor) -> bool:
        try:
            ctx = self._get_teacache_context()
        except AssertionError:
            return False
        if ctx is None:
            return False
        self.is_cfg_negative = ctx.is_cfg_negative
        start_skipping, end_skipping = ctx.teacache_params.get_skip_boundaries(
            ctx.num_inference_steps,
            ctx.do_cfg,
        )
        is_boundary_step = (
            ctx.current_timestep < start_skipping or ctx.current_timestep >= end_skipping
        )
        should_calc = self._compute_teacache_decision(
            modulated_inp=emb,
            is_boundary_step=is_boundary_step,
            coefficients=ctx.coefficients,
            teacache_thresh=ctx.teacache_thresh,
        )
        return not should_calc

    def maybe_cache_states(
        self, hidden_states: torch.Tensor, original_hidden_states: torch.Tensor
    ) -> None:
        del original_hidden_states
        if not self.is_cfg_negative:
            self.previous_residual = hidden_states
        elif self._supports_cfg_cache:
            self.previous_residual_negative = hidden_states

    def retrieve_cached_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        del hidden_states
        if self.is_cfg_negative and self._supports_cfg_cache:
            assert self.previous_residual_negative is not None
            return self.previous_residual_negative
        assert self.previous_residual is not None
        return self.previous_residual

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor] | None = None,
        timestep: torch.LongTensor | torch.Tensor | None = None,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | list[torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del encoder_hidden_states_image, guidance, encoder_attention_mask, kwargs
        if hidden_states.ndim != 5:
            raise ValueError(
                "STAR transformer expects hidden_states with shape [B, C, T, H, W], "
                f"got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[1] != self.total_in_channels:
            raise ValueError(
                f"Expected {self.total_in_channels} latent channels after condition concatenation, "
                f"got {hidden_states.shape[1]}"
            )
        if timestep is None:
            raise ValueError("timestep is required for STAR transformer forward")

        original_hidden_states = hidden_states
        text_hidden_states = self._coerce_text_tensor(encoder_hidden_states)
        model_dtype = self._get_model_dtype()
        if hidden_states.dtype != model_dtype:
            hidden_states = hidden_states.to(dtype=model_dtype)
        if text_hidden_states is not None and text_hidden_states.dtype != model_dtype:
            text_hidden_states = text_hidden_states.to(dtype=model_dtype)
        batch_size = hidden_states.shape[0]
        timestep = timestep.reshape(-1)
        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        time_emb = self._timestep_embedding(timestep, self.hidden_size)
        emb = self.time_embed(time_emb.to(hidden_states.device, dtype=hidden_states.dtype))

        if self.should_skip_forward_for_cached_states(emb=emb):
            return self.retrieve_cached_states(original_hidden_states)

        hidden_states, num_frames, grid_h, grid_w = self.mixins.patch_embed(
            hidden_states,
            text_hidden_states,
        )
        text_length = 0 if text_hidden_states is None else text_hidden_states.shape[1]
        image_token_count = num_frames * grid_h * grid_w
        freqs_sin, freqs_cos, freqs_cos_sin_cache = self.mixins.pos_embed.get_rotary_cache(
            image_token_count=image_token_count,
            head_dim=self.head_dim,
            num_frames=num_frames,
            grid_h=grid_h,
            grid_w=grid_w,
            device=hidden_states.device,
        )

        for layer_id, layer in enumerate(self.transformer.layers):
            hidden_states = layer(
                hidden_states,
                emb=emb,
                text_length=text_length,
                modulation=self.mixins.adaln_layer.adaLN_modulations[layer_id](emb),
                query_layernorm=(
                    self.mixins.adaln_layer.query_layernorm_list[layer_id]
                    if self.mixins.adaln_layer.qk_ln
                    else None
                ),
                key_layernorm=(
                    self.mixins.adaln_layer.key_layernorm_list[layer_id]
                    if self.mixins.adaln_layer.qk_ln
                    else None
                ),
                freqs_sin=freqs_sin,
                freqs_cos=freqs_cos,
                freqs_cos_sin_cache=freqs_cos_sin_cache,
                num_frames=num_frames,
                grid_h=grid_h,
                grid_w=grid_w,
            )

        hidden_states = self.transformer.final_layernorm(hidden_states).to(model_dtype)
        output = self.mixins.final_layer(
            hidden_states,
            emb,
            text_length=text_length,
            num_frames=num_frames,
            grid_h=grid_h,
            grid_w=grid_w,
        )
        self.maybe_cache_states(output, original_hidden_states)
        return output


if (
    BlockAdapterRegister is not None
    and BlockAdapter is not None
    and ForwardPattern is not None
):

    @BlockAdapterRegister.register("StarCogVideoXSR")
    def star_cogvideox_sr_adapter(pipe, **kwargs) -> BlockAdapter:
        transformer = pipe.transformer
        return BlockAdapter(
            pipe=pipe,
            transformer=transformer,
            blocks=transformer.transformer.layers,
            forward_pattern=ForwardPattern.Pattern_2,
            check_forward_pattern=False,
            has_separate_cfg=True,
            **kwargs,
        )


EntryClass = StarCogVideoXSRTransformer3DModel
