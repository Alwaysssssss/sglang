# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
    DecodingStage,
    _ensure_tensor_decode_output,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE


class STARCogVideoXSRDecodingStage(DecodingStage):
    """STAR-specific decoding stage with temporal windowed VAE decode."""

    @staticmethod
    def _tensor_summary(tensor: torch.Tensor) -> dict[str, object]:
        tensor_f = tensor.detach().cpu().float()
        return {
            "shape": list(tensor_f.shape),
            "mean": float(tensor_f.mean()),
            "std": float(tensor_f.std()),
            "min": float(tensor_f.min()),
            "max": float(tensor_f.max()),
        }

    @staticmethod
    def build_decode_windows(num_frames: int) -> list[tuple[int, int, bool]]:
        """Build STAR temporal decode windows.

        The reference STAR path decodes the first three latent frames together,
        then continues in two-frame chunks. This keeps the same layout while
        still covering non-canonical frame counts with a final short tail.
        """
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")

        if num_frames <= 3:
            return [(0, num_frames, True)]

        windows: list[tuple[int, int, bool]] = [(0, min(3, num_frames), False)]
        start_frame = windows[0][1]
        while start_frame < num_frames:
            end_frame = min(start_frame + 2, num_frames)
            windows.append((start_frame, end_frame, False))
            start_frame = end_frame

        last_start, last_end, _ = windows[-1]
        windows[-1] = (last_start, last_end, True)
        return windows

    def prepare_latents_for_decode(
        self,
        latents: torch.Tensor,
        server_args: ServerArgs,
    ) -> torch.Tensor:
        latents = latents.to(get_local_torch_device())
        latents = self.scale_and_shift(latents, server_args)
        return server_args.pipeline_config.preprocess_decoding(
            latents,
            server_args,
            vae=self.vae,
        )

    def decode_in_windows(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        decoded_windows: list[torch.Tensor] = []
        for start_frame, end_frame, clear_cache in self.build_decode_windows(
            latents.shape[2]
        ):
            window_latents = latents[:, :, start_frame:end_frame].contiguous()
            decode_output = self.vae.decode(
                window_latents,
                clear_fake_cp_cache=clear_cache,
            )
            window_frames = _ensure_tensor_decode_output(decode_output)
            if window_frames.shape[2] <= 0:
                raise ValueError(
                    "STAR decode window returned an invalid frame count: "
                    f"window=({start_frame}, {end_frame}), "
                    f"decoded={window_frames.shape[2]}, "
                )
            decoded_windows.append(window_frames)

        if not decoded_windows:
            raise ValueError("STAR decode produced no temporal windows")

        return torch.cat(decoded_windows, dim=2)

    def apply_color_fix(
        self,
        frames: torch.Tensor,
        batch: Req | None,
        server_args: ServerArgs,
    ) -> torch.Tensor:
        enable_color_fix = bool(
            getattr(server_args.pipeline_config, "enable_color_fix", False)
        )
        if batch is not None:
            enable_color_fix = bool(
                getattr(batch, "enable_color_fix", enable_color_fix)
            )

        if not enable_color_fix:
            return frames

        color_fix_mode = getattr(server_args.pipeline_config, "color_fix_mode", None)
        if batch is not None:
            color_fix_mode = getattr(batch, "color_fix_mode", color_fix_mode)

        self.log_warning(
            "STAR color fix requested (mode=%s), but phase5 parity mode keeps "
            "reference-free decode and returns frames unchanged.",
            color_fix_mode or "none",
        )
        return frames

    def postprocess_decoded_video(
        self,
        frames: torch.Tensor,
        batch: Req | None,
        server_args: ServerArgs,
    ) -> torch.Tensor:
        frames = (frames / 2 + 0.5).clamp(0, 1)
        return self.apply_color_fix(frames, batch, server_args)

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        server_args: ServerArgs,
        *,
        batch: Req | None = None,
    ) -> torch.Tensor:
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        self.vae = self.vae.to(device=get_local_torch_device(), dtype=vae_dtype)
        if batch is not None and batch.return_trajectory_latents and batch.metrics is not None:
            batch.metrics.record_annotation(
                "final_latents_before_decode_summary",
                self._tensor_summary(latents),
            )
        latents = self.prepare_latents_for_decode(latents, server_args)
        if batch is not None and batch.return_trajectory_latents and batch.metrics is not None:
            batch.metrics.record_annotation(
                "decode_input_latents_summary",
                self._tensor_summary(latents),
            )
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not server_args.disable_autocast

        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=vae_dtype,
            enabled=vae_autocast_enabled,
        ):
            try:
                if server_args.pipeline_config.vae_tiling:
                    self.vae.enable_tiling()
            except Exception:
                pass

            if not vae_autocast_enabled:
                latents = latents.to(vae_dtype)
            frames = self.decode_in_windows(latents)

        return self.postprocess_decoded_video(frames, batch, server_args)

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:
        self.load_model()

        frames = self.decode(batch.latents, server_args, batch=batch)

        if batch.return_trajectory_decoded:
            assert (
                batch.trajectory_latents is not None
            ), "batch should have trajectory latents"

            batch_size, num_steps, channels, num_frames, height, width = (
                batch.trajectory_latents.shape
            )
            flat_latents = batch.trajectory_latents.view(
                batch_size * num_steps,
                channels,
                num_frames,
                height,
                width,
            )
            all_decoded = self.decode(flat_latents, server_args, batch=None)
            decoded_tensor = all_decoded.view(
                batch_size, num_steps, *all_decoded.shape[1:]
            )
            trajectory_decoded = [
                decoded_tensor[:, index] for index in range(num_steps)
            ]
        else:
            trajectory_decoded = None

        frames = server_args.pipeline_config.post_decoding(frames, server_args)
        output_batch = OutputBatch(
            output=frames,
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            rollout_trajectory_data=batch.rollout_trajectory_data,
            trajectory_decoded=trajectory_decoded,
            metrics=batch.metrics,
            noise_pred=None,
        )

        if not getattr(batch, "is_warmup", False):
            self.offload_model()

        return output_batch
