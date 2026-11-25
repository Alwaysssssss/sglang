# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import logging
from typing import TYPE_CHECKING, Callable

import torch
import torch.distributed._functional_collectives as ft_c
from torch.distributed.tensor.experimental._attention import _cp_options

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_group,
    get_ulysses_parallel_world_size,
)
from sglang.srt.utils.common import torch_release

_cp_options.enable_load_balance = False

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
        AttentionImpl,
    )

logger = logging.getLogger(__name__)


def _maybe_wait(tensor: torch.Tensor) -> torch.Tensor:
    """
    When tracing the code, the result tensor is not an AsyncCollectiveTensor,
    so we cannot call ``wait()``.
    """
    if isinstance(tensor, ft_c.AsyncCollectiveTensor):
        return tensor.wait()
    return tensor


def _usp_all_to_all_single(x: torch.Tensor) -> torch.Tensor:
    return _maybe_wait(_usp_all_to_all_single_async(x)).reshape(x.shape)


def _usp_all_to_all_single_async(x: torch.Tensor) -> torch.Tensor:
    """Launch without consuming the result on the compute stream."""
    ulysses_pg = get_sp_group().ulysses_group
    assert ulysses_pg is not None, "Ulysses process group is not initialized."
    return ft_c.all_to_all_single(
        x.flatten(), output_split_sizes=None, input_split_sizes=None, group=ulysses_pg
    )


def _usp_input_all_to_all(x: torch.Tensor, head_dim: int = 1) -> torch.Tensor:
    """
    Perform Ulysses-style input all-to-all over the head dimension.

    Default layout expects heads at dim=1 and sequence at dim=2:
        [b, h, s_local, d] -> [b, h_local, s_global, d]

    If heads are at dim=2 (input is [b, s_local, h, d]), set head_dim=2, and the
    function returns [b, s_global, h_local, d], preserving the original
    head/sequence dim ordering.

    Args:
        x: A 4D tensor with layout [b, *, *, d] where '*' are sequence and heads
        head_dim: Which dimension index corresponds to heads (1 or 2)

    Returns:
        Tensor with the same dim order as input, with heads sharded and sequence gathered.
    """
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, f"x must have 4 dimensions, got {x.ndim}"
    assert head_dim in (1, 2), f"head_dim must be 1 or 2, got {head_dim}"

    # Move the dimension to be split (h_global) to dim 0 for all_to_all_single
    if head_dim == 1:
        b, h_global, s_local, d = x.shape
        # Shape transition: [b, h_global, s_local, d] -> [h_global, b, s_local, d]
        permute_order = (1, 0, 2, 3)
    else:  # head_dim == 2
        b, s_local, h_global, d = x.shape
        # Shape transition: [b, s_local, h_global, d] -> [h_global, b, s_local, d]
        permute_order = (2, 0, 1, 3)

    assert (
        h_global % world_size == 0
    ), f"h_global ({h_global}) must be divisible by world_size ({world_size})"

    h_local, s_global = h_global // world_size, s_local * world_size

    x = x.permute(permute_order).contiguous()
    x = _usp_all_to_all_single(x)
    x = x.reshape(world_size, h_local, b, s_local, d)

    # Reorder dims to place 'world_size' adjacent to 's_local' to merge them into 's_global'
    if head_dim == 1:
        # Shape transition: [world_size, h_local, b, s_local, d] -> [b, h_local, world_size, s_local, d]
        x = x.permute(2, 1, 0, 3, 4).contiguous().reshape(b, h_local, s_global, d)
    else:  # head_dim == 2
        # Shape transition: [world_size, h_local, b, s_local, d] -> [b, world_size, s_local, h_local, d]
        x = x.permute(2, 0, 3, 1, 4).contiguous().reshape(b, s_global, h_local, d)

    return x


def _usp_qkv_input_all_to_all(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, head_dim: int = 2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack the batch axis to exchange equal-layout Q/K/V in one collective.

    GQA and mixed dtypes retain the existing separate exchanges.
    """
    if get_ulysses_parallel_world_size() <= 1:
        return q, k, v
    if not (
        q.shape == k.shape == v.shape
        and q.dtype == k.dtype == v.dtype
        and q.device == k.device == v.device
    ):
        return tuple(_usp_input_all_to_all(x, head_dim=head_dim) for x in (q, k, v))
    packed = _usp_input_all_to_all(torch.cat((q, k, v), dim=0), head_dim=head_dim)
    return packed.chunk(3, dim=0)


def _usp_output_all_to_all(x: torch.Tensor, head_dim: int = 1) -> torch.Tensor:
    """
    Perform Ulysses-style output all-to-all over the head dimension (inverse of input).

    Default layout expects heads at dim=1 and sequence at dim=2:
        [b, h_local, s, d] -> [b, h, s_local, d]

    If heads are at dim=2 (input is [b, s_global, h // world_size, d]), set head_dim=2,
    and the function returns [b, s_local, h, d], preserving the original head/sequence
    dim ordering.

    Args:
        x: A 4D tensor with layout [b, *, *, d] where '*' are sequence and heads
        head_dim: Which dimension index corresponds to heads (1 or 2)

    Returns:
        Tensor with the same dim order as input, with heads gathered and sequence sharded.
    """
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, f"x must have 4 dimensions, got {x.ndim}"
    assert head_dim in (1, 2), f"head_dim must be 1 or 2, got {head_dim}"

    # Move the dimension to be split (s_global) to dim 0 for all_to_all_single
    if head_dim == 1:
        b, h_local, s_global, d = x.shape
        # Shape transition: [b, h_local, s_global, d] -> [s_global, b, h_local, d]
        permute_order = (2, 0, 1, 3)
    else:  # head_dim == 2
        b, s_global, h_local, d = x.shape
        # Shape transition: [b, s_global, h_local, d] -> [s_global, b, h_local, d]
        permute_order = (1, 0, 2, 3)

    assert (
        s_global % world_size == 0
    ), f"s_global ({s_global}) must be divisible by world_size ({world_size})"

    s_local, h_global = s_global // world_size, h_local * world_size

    x = x.permute(permute_order).contiguous()
    x = _usp_all_to_all_single(x)
    x = x.reshape(world_size, s_local, b, h_local, d)

    # Reorder dims to place 'world_size' adjacent to 'h_local' to merge them into 'h_global'
    if head_dim == 1:
        # Shape transition: [world_size, s_local, b, h_local, d] -> [b, world_size, h_local, s_local, d]
        x = x.permute(2, 0, 3, 1, 4).contiguous().reshape(b, h_global, s_local, d)
    else:  # head_dim == 2
        # Shape transition: [world_size, s_local, b, h_local, d] -> [b, s_local, world_size, h_local, d]
        x = x.permute(2, 1, 0, 3, 4).contiguous().reshape(b, s_local, h_global, d)

    return x


@torch.compiler.disable
def ulysses_attention_pipelined(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention: Callable,
    chunks: int,
) -> torch.Tensor:
    """Overlap head-chunk all-to-all with independent attention computation.

    Inference-only BSHD, equal Q/K/V layouts, no Ring or replicated tokens.
    Every chunk still attends to the entire global sequence. The caller must
    use identical chunk counts on all ranks and a head-independent backend.
    Functional collectives retain their send buffers until NCCL completes.
    """
    world = get_ulysses_parallel_world_size()
    assert q.ndim == 4 and q.shape == k.shape == v.shape
    assert q.dtype == k.dtype == v.dtype and q.device == k.device == v.device
    b, s, heads, d = q.shape
    assert chunks > 1 and heads % (world * chunks) == 0
    chunk_heads = heads // chunks
    local_heads = chunk_heads // world

    def start_input(index):
        start = index * chunk_heads
        # Pack directly in destination-head order, avoiding an intermediate
        # full QKV concatenation followed by a second contiguous copy.
        packed = torch.cat(
            [
                x[:, :, start : start + chunk_heads].permute(2, 0, 1, 3)
                for x in (q, k, v)
            ],
            dim=1,
        )
        return _usp_all_to_all_single_async(packed)

    def finish_input(pending):
        packed = _maybe_wait(pending).reshape(world, local_heads, 3 * b, s, d)
        packed = packed.permute(2, 0, 3, 1, 4).contiguous()
        return packed.reshape(3 * b, world * s, local_heads, d).chunk(3, dim=0)

    output = torch.empty_like(q, memory_format=torch.contiguous_format)

    def finish_output(pending, index):
        received = _maybe_wait(pending).reshape(world, s, b, local_heads, d)
        received = received.permute(2, 1, 0, 3, 4).reshape(b, s, chunk_heads, d)
        output[:, :, index * chunk_heads : (index + 1) * chunk_heads].copy_(received)

    next_input = start_input(0)
    previous_output = None
    for index in range(chunks):
        # Enqueue the next transfer before waiting for the current one, so it
        # can run while the current chunk's attention kernels execute.
        current_input = next_input
        next_input = start_input(index + 1) if index + 1 < chunks else None
        cq, ck, cv = finish_input(current_input)
        attended = attention(cq, ck, cv)
        pending_output = _usp_all_to_all_single_async(
            attended.permute(1, 0, 2, 3).contiguous()
        )
        # Drain the preceding output only after enqueueing this chunk's compute.
        if previous_output is not None:
            finish_output(previous_output, index - 1)
        previous_output = pending_output
        del cq, ck, cv, attended, current_input
    finish_output(previous_output, chunks - 1)
    return output


def ring_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_impl: "AttentionImpl",
    is_causal: bool = False,
    dropout_p: float = 0.0,
):
    """
    Ring Attention implementation.

    This function implements Ring Attention, a strategy for distributed attention
    computation that reduces peak memory usage. It accepts a generic attention
    implementation (`attn_impl`) which is called by the underlying PyTorch
    distributed attention primitive.

    Args:
        query, key, value: The input tensors for attention.
        attn_impl: An instance of an attention implementation backend
                   (e.g., FlashAttentionImpl) whose `forward` method will be
                   used as the computational kernel.
        is_causal: Whether to apply causal masking.
        dropout_p: Dropout probability.
    """
    # torch.distributed.tensor.experimental._attention is not a public API,
    from torch.distributed.tensor.experimental._attention import (
        _templated_ring_attention,
    )

    ring_pg = get_sp_group().ring_group
    assert ring_pg is not None, "Ring process group is not initialized."

    # Ring attention primitives expect tensors in [B, H, S, D] layout.
    # We permute the inputs here.
    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()

    # Create an adapter function that matches the signature expected by
    # _templated_ring_attention. The `attn_impl` already has dropout and
    # causal settings configured during its initialization.

    # Note: Please be aware that Attention Backend and Ring Attention may require different QKV tensor shapes.
    # For example, FlashAttention expects the format to be BSHD.
    def attn_callable_adapter(q, k, v, *args, **kwargs):
        # We ignore the dropout_p and is_causal passed by _templated_ring_attention
        # and rely on the pre-configured attn_impl.
        # The `attn_metadata` is not available here, so we pass None.
        # This is a limitation we must accept when using this experimental API.
        q = torch.permute(q, [0, 2, 1, 3])
        k = torch.permute(k, [0, 2, 1, 3])
        v = torch.permute(v, [0, 2, 1, 3])
        # logger.warning(f"Warning: return_softmax_lse is only supported for FlashAttentionImpl")
        output, softmax_lse, *rest = attn_impl.forward(
            q,
            k,
            v,
            attn_metadata=None,
            return_softmax_lse=True,
        )
        output = torch.permute(output, [0, 2, 1, 3])
        return output, softmax_lse, *rest

    # Starting from torch 2.6.0, _templated_ring_attention expects an integer
    # segment_id for the attention function.
    use_segment_id = torch_release >= (2, 6)

    attn_kwargs = dict(
        op=attn_callable_adapter,
        dropout_p=dropout_p,
        is_causal=is_causal,
        query=query,
        key=key,
        value=value,
        group=ring_pg,  # https://github.com/pytorch/pytorch/blob/c907c778f42ba2fdaf25b733dd25baf9779c6a12/torch/distributed/tensor/experimental/_context_parallel/_attention.py#L309
    )

    if use_segment_id:
        # For torch >= 2.6, segment_id is required. The value '1' is a placeholder
        # as we are not using complex segmentation features.
        out, *_ = _templated_ring_attention(
            seq_dim=1,  # segment_id
            **attn_kwargs,
        )
    else:
        out, *_ = _templated_ring_attention(
            **attn_kwargs,
        )

    # Permute the output back to [B, S, H, D] layout.
    output = torch.permute(out, [0, 2, 1, 3])
    return output
