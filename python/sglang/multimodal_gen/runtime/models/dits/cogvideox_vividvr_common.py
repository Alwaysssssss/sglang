# SPDX-License-Identifier: Apache-2.0
import os
from functools import lru_cache
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.activations import get_activation
from diffusers.models.resnet import ResnetBlock2D

from sglang.multimodal_gen.runtime.distributed import (
    get_sp_parallel_rank,
    get_sp_world_size,
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    flash_attn_func,
)
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE_ENV = "SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE"
_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE_DEFERRED_GLOBAL = "deferred_global"
_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE_EAGER_GLOBAL = "eager_global"
_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE_DISTRIBUTED_LOCAL = "distributed_local"
_VIVIDVR_CONNECTOR_SEQUENCE_PARALLEL_ATTENTION_BACKENDS = frozenset(
    {AttentionBackendEnum.FA, AttentionBackendEnum.FA2}
)

# Spatial pooling of control states BEFORE all-gather (Path B)
_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE_ENV = "SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE"
_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE_DEFAULT = 1  # default to no pooling for quality-safe SP
_COGVIDEOX_SPATIAL_H = 30   # patch grid height after patch_embed
_COGVIDEOX_SPATIAL_W = 45   # patch grid width after patch_embed
_COGVIDEOX_PER_FRAME_TOKENS = _COGVIDEOX_SPATIAL_H * _COGVIDEOX_SPATIAL_W  # 1350


def get_vividvr_connector_control_pool_size() -> int:
    """Return the spatial pool size for control state compression before all-gather.

    0 or 1 = no compression. Default 1 keeps the quality-safe path unless pooling is
    explicitly requested via environment override.
    """
    val = os.environ.get(_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE_ENV, "").strip()
    if not val:
        return _VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE_DEFAULT
    try:
        return int(val)
    except ValueError:
        return _VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE_DEFAULT


def _pool_control_state_2d(
    x: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Spatially pool a control state tensor using 2D patch layout.

    Args:
        x: (B, seq_len, hidden_dim) where seq_len = num_frames * 30 * 45
        pool_size: spatial downsampling factor (e.g. 2 → 30×45 → 15×22 per frame)

    Returns:
        (B, pooled_seq_len, hidden_dim)
    """
    if pool_size <= 1 or x.shape[1] <= 0:
        return x

    B = x.shape[0]
    seq_len = x.shape[1]
    C = x.shape[2]
    h, w = _COGVIDEOX_SPATIAL_H, _COGVIDEOX_SPATIAL_W
    per_frame = h * w  # 1350

    total_frames = seq_len // per_frame
    if total_frames <= 0 or seq_len % per_frame != 0:
        return x  # not divisible by per-frame tokens, skip

    hp = max(1, h // pool_size)
    wp = max(1, w // pool_size)

    # (B, F*H*W, C) → (B, F, H, W, C) → (B*F, C, H, W)
    x = x.reshape(B, total_frames, h, w, C)
    x = x.permute(0, 1, 4, 2, 3).reshape(B * total_frames, C, h, w)

    x = F.adaptive_avg_pool2d(x, (hp, wp))

    # (B*F, C, hp, wp) → (B, F, C, hp*wp) → (B, F*hp*wp, C)
    x = x.reshape(B, total_frames, C, hp * wp)
    x = x.permute(0, 1, 3, 2).reshape(B, total_frames * hp * wp, C)

    return x.contiguous()


def zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


def resolve_num_groups(preferred: int, *channels: int) -> int:
    max_groups = min(preferred, *channels)
    for groups in range(max_groups, 0, -1):
        if all(channel % groups == 0 for channel in channels):
            return groups
    return 1


class Connector(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int) -> None:
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads for VividVR connectors"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = hidden_size // num_attention_heads
        self.to_q = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.SiLU(),
            nn.Linear(512, hidden_size),
        )
        self.to_k = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.SiLU(),
            nn.Linear(512, hidden_size),
        )
        self.norm_q = nn.LayerNorm(self.attention_head_dim, eps=1e-6)
        self.norm_k = nn.LayerNorm(self.attention_head_dim, eps=1e-6)
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.SiLU(),
            zero_module(nn.Linear(512, hidden_size)),
        )
        self.c_mlp = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.SiLU(),
            zero_module(nn.Linear(512, hidden_size)),
        )

    def forward(self, c, h: torch.Tensor) -> torch.Tensor:
        local_control, global_control = unpack_vividvr_connector_context(c)
        batch_size, seq_len, hidden_size = h.shape

        control_for_attention = global_control
        control_seq_len = control_for_attention.shape[1]
        connector_sp_context_mode = get_vividvr_connector_sp_context_mode()

        q = self.to_q(h).view(
            batch_size, seq_len, self.num_attention_heads, self.attention_head_dim
        )
        k = self.to_k(control_for_attention).view(
            batch_size,
            control_seq_len,
            self.num_attention_heads,
            self.attention_head_dim,
        )
        v = control_for_attention.view(
            batch_size,
            control_seq_len,
            self.num_attention_heads,
            self.attention_head_dim,
        )

        q = self.norm_q(q)
        k = self.norm_k(k)

        if (
            connector_sp_context_mode
            == _VIVIDVR_CONNECTOR_SP_CONTEXT_MODE_DISTRIBUTED_LOCAL
            and _vividvr_connector_can_use_sequence_parallel_attention(
                query_dtype=q.dtype
            )
            and control_for_attention.shape[1] == local_control.shape[1]
        ):
            out = run_vividvr_connector_sequence_parallel_attention(q, k, v)
        else:
            out = run_vividvr_connector_attention(q, k, v)
        out = out.reshape(batch_size, seq_len, hidden_size)
        if local_control.shape[1] != seq_len:
            raise ValueError(
                "VividVR connector local control sequence length must match local "
                f"hidden states: {local_control.shape[1]} != {seq_len}"
            )
        return h + self.out_layer(out) + self.c_mlp(local_control)


@dataclass(frozen=True)
class VividVRSequenceShardState:
    enabled: bool
    original_seq_len: int
    local_seq_len: int
    seq_pad: int


def get_vividvr_connector_sp_context_mode() -> str:
    mode = (
        os.environ.get(
            _VIVIDVR_CONNECTOR_SP_CONTEXT_MODE_ENV,
            _VIVIDVR_CONNECTOR_SP_CONTEXT_MODE_DEFERRED_GLOBAL,
        )
        .strip()
        .lower()
    )
    if mode not in {
        _VIVIDVR_CONNECTOR_SP_CONTEXT_MODE_DEFERRED_GLOBAL,
        _VIVIDVR_CONNECTOR_SP_CONTEXT_MODE_EAGER_GLOBAL,
        _VIVIDVR_CONNECTOR_SP_CONTEXT_MODE_DISTRIBUTED_LOCAL,
    }:
        raise ValueError(
            "Unsupported VividVR connector SP context mode "
            f"{mode!r}. Expected one of: deferred_global, eager_global, distributed_local."
        )
    return mode


def unpack_vividvr_connector_context(
    control_context: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(control_context, torch.Tensor):
        return control_context, control_context

    if isinstance(control_context, (tuple, list)):
        if len(control_context) == 0:
            raise ValueError("VividVR connector context must not be empty")
        if len(control_context) == 1:
            local_control = control_context[0]
            return local_control, local_control
        if len(control_context) == 2:
            return control_context[0], control_context[1]

    raise TypeError(
        "VividVR connector context must be a tensor or a sequence of one/two tensors"
    )


def vividvr_sequence_shard_enabled() -> bool:
    try:
        forward_batch = get_forward_context().forward_batch
        sp_world_size = get_sp_world_size()
    except AssertionError:
        return False

    return bool(
        forward_batch is not None
        and getattr(forward_batch, "enable_sequence_shard", False)
        and sp_world_size > 1
    )


def _vividvr_connector_can_use_sequence_parallel_attention(
    *,
    query_dtype: torch.dtype,
) -> bool:
    return vividvr_sequence_shard_enabled() and query_dtype in (
        torch.float16,
        torch.bfloat16,
    )


def run_vividvr_connector_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    if not vividvr_sequence_shard_enabled() or query.dtype not in (
        torch.float16,
        torch.bfloat16,
    ):
        out = F.scaled_dot_product_attention(
            query.permute(0, 2, 1, 3),
            key.permute(0, 2, 1, 3),
            value.permute(0, 2, 1, 3),
        )
        return out.permute(0, 2, 1, 3).contiguous()

    return flash_attn_func(
        q=query.contiguous(),
        k=key.contiguous(),
        v=value.contiguous(),
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=query.shape[1],
        max_seqlen_k=key.shape[1],
        softmax_scale=None,
        causal=False,
    ).contiguous()


def run_vividvr_connector_sequence_parallel_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    if not _vividvr_connector_can_use_sequence_parallel_attention(
        query_dtype=query.dtype
    ):
        return run_vividvr_connector_attention(query, key, value)

    sp_attn = _get_vividvr_connector_sequence_parallel_attention(
        num_heads=query.shape[2],
        head_size=query.shape[3],
    )
    return sp_attn(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
    ).contiguous()


@lru_cache(maxsize=8)
def _get_vividvr_connector_sequence_parallel_attention(
    *,
    num_heads: int,
    head_size: int,
) -> USPAttention:
    return USPAttention(
        num_heads=num_heads,
        head_size=head_size,
        softmax_scale=None,
        causal=False,
        supported_attention_backends=_VIVIDVR_CONNECTOR_SEQUENCE_PARALLEL_ATTENTION_BACKENDS,
        prefix=f"vividvr_connector_sp_{num_heads}_{head_size}",
    )


def shard_vividvr_video_tokens(
    hidden_states: torch.Tensor,
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor] | None,
    VividVRSequenceShardState,
]:
    original_seq_len = hidden_states.shape[1]
    if not vividvr_sequence_shard_enabled():
        return (
            hidden_states,
            image_rotary_emb,
            VividVRSequenceShardState(
                enabled=False,
                original_seq_len=original_seq_len,
                local_seq_len=original_seq_len,
                seq_pad=0,
            ),
        )

    sp_world_size = get_sp_world_size()
    sp_rank = get_sp_parallel_rank()
    seq_pad = (-original_seq_len) % sp_world_size
    if seq_pad:
        pad = hidden_states.new_zeros(
            (hidden_states.shape[0], seq_pad, hidden_states.shape[2])
        )
        hidden_states = torch.cat([hidden_states, pad], dim=1)

    local_seq_len = hidden_states.shape[1] // sp_world_size
    hidden_states = (
        hidden_states.view(
            hidden_states.shape[0],
            sp_world_size,
            local_seq_len,
            hidden_states.shape[2],
        )[:, sp_rank]
        .contiguous()
    )

    local_image_rotary_emb = image_rotary_emb
    if image_rotary_emb is not None:
        cos, sin = image_rotary_emb
        if seq_pad:
            rope_pad_shape = (seq_pad, *cos.shape[1:])
            cos = torch.cat([cos, cos.new_zeros(rope_pad_shape)], dim=0)
            sin = torch.cat([sin, sin.new_zeros(rope_pad_shape)], dim=0)
        cos = cos.view(sp_world_size, local_seq_len, *cos.shape[1:])[sp_rank].contiguous()
        sin = sin.view(sp_world_size, local_seq_len, *sin.shape[1:])[sp_rank].contiguous()
        local_image_rotary_emb = (cos, sin)

    return (
        hidden_states,
        local_image_rotary_emb,
        VividVRSequenceShardState(
            enabled=True,
            original_seq_len=original_seq_len,
            local_seq_len=local_seq_len,
            seq_pad=seq_pad,
        ),
    )


def gather_vividvr_video_tokens(
    hidden_states: torch.Tensor,
    shard_state: VividVRSequenceShardState,
) -> torch.Tensor:
    if not shard_state.enabled:
        return hidden_states

    hidden_states = sequence_model_parallel_all_gather(hidden_states.contiguous(), dim=1)
    if shard_state.seq_pad:
        hidden_states = hidden_states[:, : shard_state.original_seq_len, :].contiguous()
    return hidden_states


def restore_vividvr_connector_global_control_state(
    local_control_state: torch.Tensor,
    shard_state: VividVRSequenceShardState,
) -> torch.Tensor:
    if not shard_state.enabled:
        return local_control_state.contiguous()

    global_control_state = sequence_model_parallel_all_gather(
        local_control_state.contiguous(),
        dim=1,
    )
    if shard_state.seq_pad:
        global_control_state = global_control_state[
            :, : shard_state.original_seq_len, :
        ].contiguous()
    return global_control_state


def restore_vividvr_connector_global_control_states(
    local_control_states: tuple[torch.Tensor, ...],
    shard_state: VividVRSequenceShardState,
) -> tuple[torch.Tensor, ...]:
    if len(local_control_states) == 0:
        return ()

    if not shard_state.enabled:
        return tuple(state.contiguous() for state in local_control_states)

    stacked_local_states = torch.stack(
        tuple(state.contiguous() for state in local_control_states),
        dim=0,
    )
    gathered_states = sequence_model_parallel_all_gather(
        stacked_local_states,
        dim=2,
    )
    if shard_state.seq_pad:
        gathered_states = gathered_states[
            :, :, : shard_state.original_seq_len, :
        ].contiguous()
    return tuple(gathered_states[index].contiguous() for index in range(gathered_states.shape[0]))


def build_vividvr_connector_control_states(
    control_states: tuple[torch.Tensor, ...],
    shard_state: VividVRSequenceShardState,
    *,
    conditioning_scale: float = 1.0,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    if len(control_states) == 0:
        return ()

    scaled_local_states = tuple(state * conditioning_scale for state in control_states)
    if not shard_state.enabled:
        return tuple((state,) for state in scaled_local_states)

    context_mode = get_vividvr_connector_sp_context_mode()
    if context_mode == _VIVIDVR_CONNECTOR_SP_CONTEXT_MODE_DISTRIBUTED_LOCAL:
        return tuple((state.contiguous(),) for state in scaled_local_states)

    if context_mode == _VIVIDVR_CONNECTOR_SP_CONTEXT_MODE_DEFERRED_GLOBAL:
        return tuple((state.contiguous(),) for state in scaled_local_states)

    # eager_global: spatially pool control states BEFORE all-gather (Path B)
    pool_size = get_vividvr_connector_control_pool_size()
    if pool_size > 1:
        pooled_states = tuple(
            _pool_control_state_2d(s, pool_size) for s in scaled_local_states
        )
        global_control_states = restore_vividvr_connector_global_control_states(
            pooled_states,
            shard_state,
        )
        # one-shot diagnostic (module-level, once per step total)
        _tag = "_ctrl_pool_reported"
        if not getattr(build_vividvr_connector_control_states, _tag, False):
            setattr(build_vividvr_connector_control_states, _tag, True)
            sp_rank = get_sp_parallel_rank()
            orig_len = scaled_local_states[0].shape[1]
            pooled_len = global_control_states[0].shape[1]
            ratio = orig_len / pooled_len if pooled_len > 0 else 0
            print(
                f"[CTRL POOL rank={sp_rank}] pool_size={pool_size} "
                f"local={orig_len} → pooled_local={pooled_states[0].shape[1]} "
                f"→ all_gather → global={pooled_len} tokens "
                f"({ratio:.1f}× compression vs unpooled all-gather)"
            )
    else:
        global_control_states = restore_vividvr_connector_global_control_states(
            scaled_local_states,
            shard_state,
        )
    return tuple(
        (
            scaled_local_states[index].contiguous(),
            global_control_states[index],
        )
        for index in range(len(scaled_local_states))
    )


class TemporalResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: int = 512,
        eps: float = 1e-6,
        groups: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        kernel_size = (3, 1, 1)
        padding = [kernel // 2 for kernel in kernel_size]

        self.norm1 = nn.GroupNorm(
            num_groups=groups,
            num_channels=in_channels,
            eps=eps,
            affine=True,
        )
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.time_emb_proj = (
            nn.Linear(temb_channels, out_channels) if temb_channels is not None else None
        )
        self.norm2 = nn.GroupNorm(
            num_groups=groups,
            num_channels=out_channels,
            eps=eps,
            affine=True,
        )
        self.dropout = nn.Dropout(0.0)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.nonlinearity = get_activation("silu")
        self.use_in_shortcut = self.in_channels != out_channels
        self.conv_shortcut = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if self.use_in_shortcut
            else None
        )

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm1(input_tensor)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, :, None, None]
            temb = temb.permute(0, 2, 1, 3, 4)
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        return input_tensor + hidden_states


class AlphaBlender(nn.Module):
    strategies = ["learned", "fixed", "learned_with_images"]

    def __init__(
        self,
        alpha: float,
        merge_strategy: str = "learned_with_images",
        switch_spatial_to_temporal_mix: bool = False,
    ) -> None:
        super().__init__()
        self.merge_strategy = merge_strategy
        self.switch_spatial_to_temporal_mix = switch_spatial_to_temporal_mix

        if merge_strategy not in self.strategies:
            raise ValueError(f"merge_strategy needs to be in {self.strategies}")

        if merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.tensor([alpha]))
        else:
            self.register_parameter("mix_factor", nn.Parameter(torch.tensor([alpha])))

    def get_alpha(self, image_only_indicator: torch.Tensor, ndims: int) -> torch.Tensor:
        if self.merge_strategy == "fixed":
            alpha = self.mix_factor
        elif self.merge_strategy == "learned":
            alpha = torch.sigmoid(self.mix_factor)
        elif self.merge_strategy == "learned_with_images":
            if image_only_indicator is None:
                raise ValueError(
                    "Please provide image_only_indicator to use learned_with_images merge strategy"
                )
            alpha = torch.where(
                image_only_indicator.bool(),
                torch.ones(1, 1, device=image_only_indicator.device),
                torch.sigmoid(self.mix_factor)[..., None],
            )
            if ndims == 5:
                alpha = alpha[:, None, :, None, None]
            elif ndims == 3:
                alpha = alpha.reshape(-1)[:, None, None]
            else:
                raise ValueError(
                    f"Unexpected ndims {ndims}. Dimensions should be 3 or 5"
                )
        else:
            raise NotImplementedError

        return alpha

    def forward(
        self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        alpha = self.get_alpha(image_only_indicator, x_spatial.ndim)
        alpha = alpha.to(x_spatial.dtype)
        if self.switch_spatial_to_temporal_mix:
            alpha = 1.0 - alpha
        return alpha * x_spatial + (1.0 - alpha) * x_temporal


class SpatioTemporalResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: int = 512,
        eps: float = 1e-6,
        temporal_eps: Optional[float] = None,
        merge_factor: float = 0.5,
        merge_strategy: str = "learned_with_images",
        switch_spatial_to_temporal_mix: bool = False,
        groups: int = 32,
    ) -> None:
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.spatial_res_block = ResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            groups=groups,
            eps=eps,
        )
        self.temporal_res_block = TemporalResnetBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            groups=groups,
            eps=temporal_eps if temporal_eps is not None else eps,
        )
        self.time_mixer = AlphaBlender(
            alpha=merge_factor,
            merge_strategy=merge_strategy,
            switch_spatial_to_temporal_mix=switch_spatial_to_temporal_mix,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_frames = image_only_indicator.shape[-1]
        hidden_states = self.spatial_res_block(hidden_states, temb)

        batch_frames, channels, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames

        hidden_states_mix = hidden_states.reshape(
            batch_size, num_frames, channels, height, width
        ).permute(0, 2, 1, 3, 4)
        hidden_states = hidden_states.reshape(
            batch_size, num_frames, channels, height, width
        ).permute(0, 2, 1, 3, 4)

        if temb is not None:
            temb = temb.reshape(batch_size, num_frames, -1)

        hidden_states = self.temporal_res_block(hidden_states, temb)
        hidden_states = self.time_mixer(
            x_spatial=hidden_states_mix,
            x_temporal=hidden_states,
            image_only_indicator=image_only_indicator,
        )
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(
            batch_frames, channels, height, width
        )
        return hidden_states


def build_control_feat_proj(in_channels: int, time_embed_dim: int) -> nn.ModuleList:
    return nn.ModuleList(
        [
            SpatioTemporalResBlock(
                in_channels,
                320,
                time_embed_dim,
                merge_strategy="learned",
                groups=resolve_num_groups(16, in_channels, 320),
            ),
            SpatioTemporalResBlock(
                320,
                320,
                time_embed_dim,
                merge_strategy="learned",
                groups=resolve_num_groups(32, 320, 320),
            ),
            SpatioTemporalResBlock(
                320,
                in_channels,
                time_embed_dim,
                merge_strategy="learned",
                groups=resolve_num_groups(16, 320, in_channels),
            ),
        ]
    )
