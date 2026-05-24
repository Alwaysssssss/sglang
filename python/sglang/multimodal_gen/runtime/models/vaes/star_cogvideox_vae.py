# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import os
from pathlib import Path
import sys
import tempfile
import warnings

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from torch import nn

from sglang.multimodal_gen.configs.models.vaes.star_cogvideox_vae import (
    StarCogVideoXSRVAEConfig,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_dp_group,
    get_sp_group,
    get_tp_group,
    get_world_group,
)
from sglang.multimodal_gen.runtime.utils.startup_debug import (
    write_startup_debug_event,
)

_STAR_LOCAL_DIST_INIT_PATH: str | None = None


def _resolve_star_sat_root() -> Path | None:
    candidates: list[Path] = []
    for env_name in ("SGLANG_STAR_SAT_ROOT", "STAR_COGVIDEOX_SAT_ROOT"):
        raw = os.environ.get(env_name)
        if raw:
            candidates.append(Path(raw).expanduser().resolve())

    current = Path(__file__).resolve()
    for parent in current.parents:
        candidates.append(parent / "STAR_mg" / "cogvideox-based" / "sat")
        candidates.append(parent.parent / "STAR_mg" / "cogvideox-based" / "sat")

    for candidate in candidates:
        if (candidate / "vae_modules" / "autoencoder.py").is_file():
            return candidate
    return None


def _load_original_star_modules():
    sat_root = _resolve_star_sat_root()
    if sat_root is None:
        raise FileNotFoundError(
            "Unable to locate STAR SAT source root. Set `SGLANG_STAR_SAT_ROOT` "
            "or place `STAR_mg/cogvideox-based/sat` alongside the `sglang` repo."
        )

    sat_root_str = str(sat_root)
    if sat_root_str not in sys.path:
        sys.path.insert(0, sat_root_str)

    autoencoder_mod = importlib.import_module("vae_modules.autoencoder")
    util_mod = importlib.import_module("sgm.util")
    return autoencoder_mod.VideoAutoencoderInferenceWrapper, util_mod


def _ensure_star_context_parallel(
    util_mod,
    *,
    context_parallel_size: int,
) -> None:
    global _STAR_LOCAL_DIST_INIT_PATH

    # The original STAR VAE implementation still dereferences a context-parallel
    # group inside fake-CP convolution helpers even when the wrapper is used in
    # cp_size=0 mode. We therefore need a valid singleton process group, but we
    # must not ask SAT to create new groups on top of SGLang's existing
    # distributed topology because that can deadlock multi-rank startup.
    if context_parallel_size <= 1:
        if util_mod.is_context_parallel_initialized():
            return

        singleton_group = None
        for candidate in (get_tp_group(), get_sp_group(), get_dp_group()):
            if candidate.world_size == 1:
                singleton_group = candidate.device_group
                break
        if singleton_group is None:
            world_group = get_world_group()
            if world_group.world_size == 1:
                singleton_group = world_group.device_group

        if singleton_group is None:
            raise RuntimeError(
                "Unable to find an existing singleton process group for STAR VAE "
                "fake context-parallel operations."
            )

        write_startup_debug_event("STAR VAE set_context_parallel_group singleton")
        util_mod.set_context_parallel_group(1, singleton_group)
        return

    if not dist.is_initialized():
        write_startup_debug_event("STAR VAE local dist.init_process_group start")
        if _STAR_LOCAL_DIST_INIT_PATH is None:
            fd, init_path = tempfile.mkstemp(
                prefix="sglang-star-cp-", suffix=".dist"
            )
            os.close(fd)
            _STAR_LOCAL_DIST_INIT_PATH = init_path
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{_STAR_LOCAL_DIST_INIT_PATH}",
            rank=0,
            world_size=1,
        )
        write_startup_debug_event("STAR VAE local dist.init_process_group done")

    if not util_mod.is_context_parallel_initialized():
        write_startup_debug_event("STAR VAE initialize_context_parallel start")
        util_mod.initialize_context_parallel(1)
        write_startup_debug_event("STAR VAE initialize_context_parallel done")


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
            if self.temporal_compression_ratio <= 1 or latents.shape[2] <= 1:
                return latents

            ratio = int(self.temporal_compression_ratio)
            if ratio > 0 and ratio & (ratio - 1) == 0:
                expanded = latents
                num_temporal_upsamples = int(math.log2(ratio))
                for _ in range(num_temporal_upsamples):
                    if expanded.shape[2] <= 1:
                        break
                    if expanded.shape[2] % 2 == 1:
                        first_frame = expanded[:, :, :1]
                        rest_frames = expanded[:, :, 1:].repeat_interleave(2, dim=2)
                        expanded = torch.cat([first_frame, rest_frames], dim=2)
                    else:
                        expanded = expanded.repeat_interleave(2, dim=2)
                return expanded

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
        self.use_tiling = False
        self._star_util_mod = None
        self._use_original_impl = False

        try:
            write_startup_debug_event("STAR VAE original_impl load_modules start")
            wrapper_cls, util_mod = _load_original_star_modules()
            write_startup_debug_event("STAR VAE original_impl load_modules done")
            _ensure_star_context_parallel(util_mod, context_parallel_size=0)
            self._star_util_mod = util_mod
            write_startup_debug_event("STAR VAE original_impl wrapper init start")
            self.impl = wrapper_cls(
                cp_size=0,
                loss_config={"target": "torch.nn.Identity"},
                regularizer_config={
                    "target": "vae_modules.regularizers.DiagonalGaussianRegularizer"
                },
                encoder_config={
                    "target": "vae_modules.cp_enc_dec.ContextParallelEncoder3D",
                    "params": {
                        "double_z": True,
                        "z_channels": config.arch_config.z_channels,
                        "resolution": config.arch_config.resolution,
                        "in_channels": config.arch_config.in_channels,
                        "out_ch": config.arch_config.out_channels,
                        "ch": config.arch_config.ch,
                        "ch_mult": config.arch_config.ch_mult,
                        "attn_resolutions": [],
                        "num_res_blocks": config.arch_config.num_res_blocks,
                        "dropout": config.arch_config.dropout,
                        "gather_norm": True,
                    },
                },
                decoder_config={
                    "target": "vae_modules.cp_enc_dec.ContextParallelDecoder3D",
                    "params": {
                        "double_z": True,
                        "z_channels": config.arch_config.z_channels,
                        "resolution": config.arch_config.resolution,
                        "in_channels": config.arch_config.in_channels,
                        "out_ch": config.arch_config.out_channels,
                        "ch": config.arch_config.ch,
                        "ch_mult": config.arch_config.ch_mult,
                        "attn_resolutions": [],
                        "num_res_blocks": config.arch_config.num_res_blocks,
                        "dropout": config.arch_config.dropout,
                        "gather_norm": False,
                    },
                },
            )
            write_startup_debug_event("STAR VAE original_impl wrapper init done")
            self._use_original_impl = True
        except Exception as exc:
            warnings.warn(
                "Falling back to the approximate STAR VAE implementation because "
                f"the original SAT VAE could not be initialized: {exc}",
                stacklevel=2,
            )
            self.encoder = _StarVideoEncoder(config)
            self.decoder = _StarVideoDecoder(config)

    def _ensure_original_runtime(self) -> None:
        if self._use_original_impl and self._star_util_mod is not None:
            _ensure_star_context_parallel(self._star_util_mod, context_parallel_size=0)

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
        if self._use_original_impl:
            self._ensure_original_runtime()
            moments = self.impl.encoder(x)
        else:
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
        clear_fake_cp_cache: bool = False,
        **kwargs,
    ):
        del kwargs
        if latents.ndim != 5:
            raise ValueError(
                "STAR VAE decode expects [B, C, T, H, W], "
                f"got {tuple(latents.shape)}"
            )
        if self._use_original_impl:
            self._ensure_original_runtime()
            if target_num_frames is not None:
                expected_frames = latents.shape[2] * 4 - 3
                if target_num_frames != expected_frames:
                    warnings.warn(
                        "STAR original VAE decode ignores `target_num_frames`; "
                        f"requested {target_num_frames}, implied {expected_frames}.",
                        stacklevel=2,
                    )
            sample = self.impl.decode(
                latents,
                clear_fake_cp_cache=clear_fake_cp_cache,
            )
        else:
            del clear_fake_cp_cache
            sample = self.decoder(latents, target_num_frames=target_num_frames)
        if not return_dict:
            return (sample,)
        return DecoderOutput(sample=sample)

    def forward(self, x: torch.Tensor, sample_posterior: bool = False):
        posterior = self.encode(x).latent_dist
        latents = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(latents)

    def state_dict(self, *args, **kwargs):
        if self._use_original_impl:
            return self.impl.state_dict(*args, **kwargs)
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        if self._use_original_impl:
            return self.impl.load_state_dict(state_dict, strict=strict)
        return super().load_state_dict(state_dict, strict=strict)


EntryClass = StarCogVideoXSRVAE
