from __future__ import annotations

import torch
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock
from torch import nn

from sglang.multimodal_gen.runtime.layers.elementwise import MulAdd
from sglang.multimodal_gen.runtime.layers.layernorm import (
    LayerNormScaleShift,
    ScaleResidualLayerNormScaleShift,
)

_COGVIDEOX_MODULATION_FUSION_IMPL = "sglang_modulation_fused_ops"


def _resolve_module_device(module: nn.Module) -> torch.device:
    for parameter in module.parameters(recurse=False):
        return parameter.device
    for parameter in module.parameters():
        return parameter.device
    return torch.device("cpu")


def _build_layernorm_scale_shift(
    source_norm: nn.LayerNorm,
    *,
    device: torch.device,
) -> LayerNormScaleShift:
    hidden_size = int(source_norm.normalized_shape[0])
    fused_norm = LayerNormScaleShift(
        hidden_size=hidden_size,
        eps=source_norm.eps,
        elementwise_affine=source_norm.weight is not None,
        dtype=torch.float32,
    ).to(device=device)

    with torch.no_grad():
        if source_norm.weight is not None and fused_norm.norm.weight is not None:
            fused_norm.norm.weight.copy_(
                source_norm.weight.detach().to(
                    device=fused_norm.norm.weight.device,
                    dtype=fused_norm.norm.weight.dtype,
                )
            )
        if source_norm.bias is not None and fused_norm.norm.bias is not None:
            fused_norm.norm.bias.copy_(
                source_norm.bias.detach().to(
                    device=fused_norm.norm.bias.device,
                    dtype=fused_norm.norm.bias.dtype,
                )
            )
    return fused_norm


def _build_scale_residual_layernorm_scale_shift(
    source_norm: nn.LayerNorm,
    *,
    device: torch.device,
) -> ScaleResidualLayerNormScaleShift:
    hidden_size = int(source_norm.normalized_shape[0])
    fused_norm = ScaleResidualLayerNormScaleShift(
        hidden_size=hidden_size,
        eps=source_norm.eps,
        elementwise_affine=source_norm.weight is not None,
        dtype=torch.float32,
    ).to(device=device)

    with torch.no_grad():
        if source_norm.weight is not None and fused_norm.norm.weight is not None:
            fused_norm.norm.weight.copy_(
                source_norm.weight.detach().to(
                    device=fused_norm.norm.weight.device,
                    dtype=fused_norm.norm.weight.dtype,
                )
            )
        if source_norm.bias is not None and fused_norm.norm.bias is not None:
            fused_norm.norm.bias.copy_(
                source_norm.bias.detach().to(
                    device=fused_norm.norm.bias.device,
                    dtype=fused_norm.norm.bias.dtype,
                )
            )
    return fused_norm


def _apply_layernorm_scale_shift(
    module: LayerNormScaleShift,
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    if x.is_cuda:
        return module(x, shift, scale)

    normalized = module.norm(x)
    return (normalized * (1 + scale)[:, None, :] + shift[:, None, :]).to(x.dtype)


def _apply_scale_residual_layernorm_scale_shift(
    module: ScaleResidualLayerNormScaleShift,
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor | int,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if residual.is_cuda and x.is_cuda:
        return module(residual, x, gate, shift, scale)

    if isinstance(gate, int):
        residual_output = residual + gate * x
    elif gate.dim() == 4:
        num_frames = gate.shape[1]
        frame_seqlen = x.shape[1] // num_frames
        residual_output = residual + (
            x.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * gate
        ).flatten(1, 2)
    else:
        residual_output = residual + x * gate

    normalized = module.norm(residual_output)
    modulated = (normalized * (1 + scale)[:, None, :] + shift[:, None, :]).to(
        residual.dtype
    )
    return modulated, residual_output


def _apply_mul_add(
    module: MulAdd,
    x: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    if x.is_cuda and gate.is_cuda and residual.is_cuda:
        if not x.is_contiguous():
            x = x.contiguous()
        if not gate.is_contiguous():
            gate = gate.contiguous()
        if not residual.is_contiguous():
            residual = residual.contiguous()
        return module(x, gate, residual)
    return residual + x * gate


class CogVideoXModulationFusedBlock(nn.Module):
    """Reuses sglang fused kernels for CogVideoX block modulation/residual paths."""

    def __init__(self, source_block: CogVideoXBlock):
        super().__init__()
        device = _resolve_module_device(source_block)

        self.norm1_silu = source_block.norm1.silu
        self.norm1_linear = source_block.norm1.linear
        self.norm1_modulation = _build_layernorm_scale_shift(
            source_block.norm1.norm,
            device=device,
        )

        self.attn1 = source_block.attn1

        self.norm2_silu = source_block.norm2.silu
        self.norm2_linear = source_block.norm2.linear
        self.norm2_residual_modulation = _build_scale_residual_layernorm_scale_shift(
            source_block.norm2.norm,
            device=device,
        )

        self.ff = source_block.ff
        self.ff_residual = MulAdd()
        self._sglang_cogvideox_modulation_fusion_impl = (
            _COGVIDEOX_MODULATION_FUSION_IMPL
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_kwargs: dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_seq_length = encoder_hidden_states.size(1)
        attention_kwargs = attention_kwargs or {}

        (
            shift_msa,
            scale_msa,
            gate_msa,
            enc_shift_msa,
            enc_scale_msa,
            enc_gate_msa,
        ) = self.norm1_linear(self.norm1_silu(temb)).chunk(6, dim=1)

        norm_hidden_states = _apply_layernorm_scale_shift(
            self.norm1_modulation,
            hidden_states,
            shift_msa,
            scale_msa,
        )
        norm_encoder_hidden_states = _apply_layernorm_scale_shift(
            self.norm1_modulation,
            encoder_hidden_states,
            enc_shift_msa,
            enc_scale_msa,
        )

        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **attention_kwargs,
        )

        (
            shift_ff,
            scale_ff,
            gate_ff,
            enc_shift_ff,
            enc_scale_ff,
            enc_gate_ff,
        ) = self.norm2_linear(self.norm2_silu(temb)).chunk(6, dim=1)

        norm_hidden_states, hidden_states = (
            _apply_scale_residual_layernorm_scale_shift(
                self.norm2_residual_modulation,
                hidden_states,
                attn_hidden_states,
                gate_msa[:, None, :],
                shift_ff,
                scale_ff,
            )
        )
        norm_encoder_hidden_states, encoder_hidden_states = (
            _apply_scale_residual_layernorm_scale_shift(
                self.norm2_residual_modulation,
                encoder_hidden_states,
                attn_encoder_hidden_states,
                enc_gate_msa[:, None, :],
                enc_shift_ff,
                enc_scale_ff,
            )
        )

        ff_input = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(ff_input)

        hidden_states = _apply_mul_add(
            self.ff_residual,
            ff_output[:, text_seq_length:],
            gate_ff[:, None, :],
            hidden_states,
        )
        encoder_hidden_states = _apply_mul_add(
            self.ff_residual,
            ff_output[:, :text_seq_length],
            enc_gate_ff[:, None, :],
            encoder_hidden_states,
        )
        return hidden_states, encoder_hidden_states


def enable_cogvideox_modulation_fusion(module: nn.Module) -> int:
    blocks = getattr(module, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise ValueError(
            "CogVideoX modulation fusion expects a module with transformer_blocks."
        )

    new_blocks: list[nn.Module] = []
    total_fused = 0
    replaced = 0

    for block in blocks:
        if isinstance(block, CogVideoXModulationFusedBlock):
            new_blocks.append(block)
            total_fused += 1
            continue
        if isinstance(block, CogVideoXBlock):
            new_blocks.append(CogVideoXModulationFusedBlock(block))
            total_fused += 1
            replaced += 1
            continue
        new_blocks.append(block)

    if total_fused == 0:
        raise ValueError(
            "No CogVideoXBlock modules were found while enabling CogVideoX modulation fusion."
        )

    if replaced > 0:
        module.transformer_blocks = nn.ModuleList(new_blocks)

    setattr(module, "_sglang_cogvideox_modulation_fusion_enabled", True)
    setattr(
        module,
        "_sglang_cogvideox_modulation_fusion_impl",
        _COGVIDEOX_MODULATION_FUSION_IMPL,
    )
    return total_fused


def inspect_cogvideox_modulation_fusion(module: nn.Module) -> str | None:
    impl = getattr(module, "_sglang_cogvideox_modulation_fusion_impl", None)
    if impl is not None:
        return str(impl)

    blocks = getattr(module, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        return None

    for block in blocks:
        impl = getattr(block, "_sglang_cogvideox_modulation_fusion_impl", None)
        if impl is not None:
            return str(impl)
    return None
