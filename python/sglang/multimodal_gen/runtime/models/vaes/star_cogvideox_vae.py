# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from torch import nn

from sglang.multimodal_gen.configs.models.vaes.star_cogvideox_vae import (
    StarCogVideoXSRVAEConfig,
)


def _group_norm(num_channels: int) -> nn.GroupNorm:
    groups = min(32, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels, eps=1e-6, affine=True)


class _Conv3dWrapper(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _PerFrameConv2d(nn.Module):
    def __init__(self, channels: int, transpose: bool = False) -> None:
        super().__init__()
        if transpose:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        else:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.transpose = transpose

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, num_frames, height, width = x.shape
        x2d = x.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
        if self.transpose:
            x2d = F.interpolate(x2d, scale_factor=2.0, mode="nearest")
        else:
            x2d = F.avg_pool2d(x2d, kernel_size=2, stride=2)
        x2d = self.conv(x2d)
        out_height, out_width = x2d.shape[-2:]
        return x2d.reshape(batch_size, num_frames, channels, out_height, out_width).permute(0, 2, 1, 3, 4)


class _LatentConditionedNorm3D(nn.Module):
    def __init__(self, channels: int, cond_channels: int) -> None:
        super().__init__()
        self.norm_layer = _group_norm(channels)
        self.conv_y = _Conv3dWrapper(cond_channels, channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = _Conv3dWrapper(cond_channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cond = F.interpolate(
            cond,
            size=x.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        normed = self.norm_layer(x)
        scale = self.conv_y(cond)
        bias = self.conv_b(cond)
        return normed * (1.0 + scale) + bias


class _EncoderResnetBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.conv1 = _Conv3dWrapper(in_channels, out_channels)
        self.norm2 = _group_norm(out_channels)
        self.conv2 = _Conv3dWrapper(out_channels, out_channels)
        self.nin_shortcut = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.nin_shortcut is None else self.nin_shortcut(x)
        x = F.silu(self.norm1(x))
        x = self.conv1(x)
        x = F.silu(self.norm2(x))
        x = self.conv2(x)
        return x + residual


class _DecoderResnetBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_channels: int) -> None:
        super().__init__()
        self.norm1 = _LatentConditionedNorm3D(in_channels, cond_channels)
        self.conv1 = _Conv3dWrapper(in_channels, out_channels)
        self.norm2 = _LatentConditionedNorm3D(out_channels, cond_channels)
        self.conv2 = _Conv3dWrapper(out_channels, out_channels)
        self.nin_shortcut = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = x if self.nin_shortcut is None else self.nin_shortcut(x)
        x = F.silu(self.norm1(x, cond))
        x = self.conv1(x)
        x = F.silu(self.norm2(x, cond))
        x = self.conv2(x)
        return x + residual


class _EncoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_res_blocks: int,
        with_downsample: bool,
    ) -> None:
        super().__init__()
        blocks = []
        block_in_channels = in_channels
        for _ in range(num_res_blocks):
            blocks.append(_EncoderResnetBlock3D(block_in_channels, out_channels))
            block_in_channels = out_channels
        self.block = nn.ModuleList(blocks)
        self.downsample = _PerFrameConv2d(out_channels) if with_downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.block:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class _EncoderMid(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block_1 = _EncoderResnetBlock3D(channels, channels)
        self.block_2 = _EncoderResnetBlock3D(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        x = self.block_2(x)
        return x


class _DecoderStage(nn.Module):
    def __init__(
        self,
        channels_sequence: list[int],
        *,
        cond_channels: int,
        with_upsample: bool,
    ) -> None:
        super().__init__()
        self.block = nn.ModuleList(
            [
                _DecoderResnetBlock3D(
                    channels_sequence[index],
                    channels_sequence[index + 1],
                    cond_channels,
                )
                for index in range(len(channels_sequence) - 1)
            ]
        )
        self.upsample = _PerFrameConv2d(channels_sequence[-1], transpose=True) if with_upsample else None

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for block in self.block:
            x = block(x, cond)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class _DecoderMid(nn.Module):
    def __init__(self, channels: int, cond_channels: int) -> None:
        super().__init__()
        self.block_1 = _DecoderResnetBlock3D(channels, channels, cond_channels)
        self.block_2 = _DecoderResnetBlock3D(channels, channels, cond_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x, cond)
        x = self.block_2(x, cond)
        return x


class _StarVideoEncoder(nn.Module):
    def __init__(self, config: StarCogVideoXSRVAEConfig) -> None:
        super().__init__()
        arch = config.arch_config
        channels = [arch.ch * mult for mult in arch.ch_mult]
        self.conv_in = _Conv3dWrapper(arch.in_channels, channels[0])
        self.down = nn.ModuleList(
            [
                _EncoderStage(
                    channels[index - 1] if index > 0 else channels[0],
                    channels[index],
                    num_res_blocks=arch.num_res_blocks,
                    with_downsample=index < len(channels) - 1,
                )
                for index in range(len(channels))
            ]
        )
        self.mid = _EncoderMid(channels[-1])
        self.norm_out = _group_norm(channels[-1])
        self.conv_out = _Conv3dWrapper(channels[-1], arch.z_channels * 2)
        self.temporal_compression_ratio = arch.temporal_compression_ratio

    def _compress_time(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal_compression_ratio <= 1 or x.shape[2] <= 1:
            return x
        indices = torch.arange(
            0,
            x.shape[2],
            self.temporal_compression_ratio,
            device=x.device,
        )
        if indices[-1].item() != x.shape[2] - 1:
            indices = torch.cat(
                [indices, indices.new_tensor([x.shape[2] - 1])],
                dim=0,
            ).unique(sorted=True)
        return x.index_select(2, indices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for stage in self.down:
            x = stage(x)
        x = self.mid(x)
        x = F.silu(self.norm_out(x))
        x = self.conv_out(x)
        return self._compress_time(x)


class _StarVideoDecoder(nn.Module):
    def __init__(self, config: StarCogVideoXSRVAEConfig) -> None:
        super().__init__()
        arch = config.arch_config
        channels = [arch.ch * mult for mult in arch.ch_mult]
        self.temporal_compression_ratio = arch.temporal_compression_ratio
        self.conv_in = _Conv3dWrapper(arch.z_channels, channels[-1])
        self.mid = _DecoderMid(channels[-1], arch.z_channels)
        up_sequences = [
            [channels[-2], channels[0], channels[0], channels[0], channels[0]],
            [channels[-2], channels[-2], channels[-2], channels[-2], channels[-2]],
            [channels[-1], channels[-2], channels[-2], channels[-2], channels[-2]],
            [channels[-1], channels[-1], channels[-1], channels[-1], channels[-1]],
        ]
        self.up = nn.ModuleList(
            [
                _DecoderStage(
                    sequence,
                    cond_channels=arch.z_channels,
                    with_upsample=index > 0,
                )
                for index, sequence in enumerate(up_sequences)
            ]
        )
        self.norm_out = _LatentConditionedNorm3D(channels[0], arch.z_channels)
        self.conv_out = _Conv3dWrapper(channels[0], arch.out_channels)

    def _expand_time(
        self,
        latents: torch.Tensor,
        *,
        target_num_frames: int | None,
    ) -> torch.Tensor:
        if target_num_frames is None:
            target_num_frames = max(
                1,
                (latents.shape[2] - 1) * self.temporal_compression_ratio + 1,
            )
        if latents.shape[2] == target_num_frames:
            return latents
        return F.interpolate(
            latents,
            size=(target_num_frames, latents.shape[3], latents.shape[4]),
            mode="trilinear",
            align_corners=False,
        )

    def forward(
        self,
        latents: torch.Tensor,
        *,
        target_num_frames: int | None = None,
    ) -> torch.Tensor:
        cond = latents
        x = self.conv_in(latents)
        x = self.mid(x, cond)
        for stage in reversed(self.up):
            x = stage(x, cond)
        x = self._expand_time(x, target_num_frames=target_num_frames)
        x = F.silu(self.norm_out(x, cond))
        return self.conv_out(x)


class StarCogVideoXSRVAE(nn.Module):
    """STAR CogVideoX-SR 3D VAE adapter."""

    def __init__(self, config: StarCogVideoXSRVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.scaling_factor = config.arch_config.scaling_factor
        self.shift_factor = getattr(config.arch_config, "shift_factor", None)
        self.encoder = _StarVideoEncoder(config)
        self.decoder = _StarVideoDecoder(config)
        self.use_tiling = False

    def enable_tiling(self, use_tiling: bool = True) -> None:
        self.use_tiling = use_tiling

    def disable_tiling(self) -> None:
        self.enable_tiling(False)

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        if x.ndim != 5:
            raise ValueError(
                "STAR VAE encode expects [B, C, T, H, W], "
                f"got {tuple(x.shape)}"
            )
        moments = self.encoder(x)
        latent_dist = DiagonalGaussianDistribution(moments)
        if not return_dict:
            return (latent_dist,)
        return AutoencoderKLOutput(latent_dist=latent_dist)

    def decode(
        self,
        latents: torch.Tensor,
        return_dict: bool = True,
        target_num_frames: int | None = None,
        **kwargs,
    ):
        del kwargs
        if latents.ndim != 5:
            raise ValueError(
                "STAR VAE decode expects [B, C, T, H, W], "
                f"got {tuple(latents.shape)}"
            )
        sample = self.decoder(latents, target_num_frames=target_num_frames)
        if not return_dict:
            return (sample,)
        return DecoderOutput(sample=sample)

    def forward(self, x: torch.Tensor, sample_posterior: bool = False):
        posterior = self.encode(x).latent_dist
        latents = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(latents)


EntryClass = StarCogVideoXSRVAE
