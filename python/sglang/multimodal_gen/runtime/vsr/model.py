# SPDX-License-Identifier: Apache-2.0
"""VSR model components: the one-step restoration pipeline.

Port of ``infer/models/stage3.py``.

This is **not** a sampling diffusion pipeline. Each window gets exactly one DiT
forward at a fixed timestep with a zero text condition, and the latent is
updated by subtracting the predicted velocity::

    z_lq = (VAE.encode(x).mode() - mu) / sigma
    v    = DiT(z_lq, t=1000, encoder_hidden_states=zeros)
    x_hat = VAE.decode(z_lq - v)

No scheduler, no CFG, no noise, no text encoder, and no per-window randomness --
which is why the reference is reproducible run to run and why caching schemes
such as TeaCache or Cache-DiT have nothing to cache here
(``requirements.md`` §5.2).

Phase 1 deliberately uses the diffusers classes rather than SGLang's own Wan
implementations, so the numerics stay comparable to the reference; swapping them
in is phase 2 (``requirements.md`` §1).
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: The rectified-flow endpoint the reference evaluates at. It is a *timestep
#: condition*, not an iteration count and not a frame index.
FM_TIMESTEP = 1000.0


class VSRAutoencoder(nn.Module):
    """Base Wan2.2 VAE encoder plus an optional fine-tuned decoder.

    Stage-3 training only writes ``vae_decoder.pt``; the encoder stays frozen at
    the base Wan2.2 weights. With ``--use_ema`` the run also writes
    ``vae_decoder_ema.pt``, and inference prefers it so the EMA'd DiT is paired
    with the matching EMA'd decoder.

    The per-channel latent normalisation mirrors the training code so the DiT
    sees the same latent distribution. Encoding uses ``.mode()`` rather than
    ``.sample()``: deterministic, and it is what the reference measures against.
    """

    def __init__(self, vae, decoder_state_path: str = None):
        super().__init__()
        self.vae = vae
        if decoder_state_path is not None:
            state = torch.load(str(decoder_state_path), map_location="cpu")
            self.vae.decoder.load_state_dict(state)
        mean = torch.as_tensor(vae.config.latents_mean, dtype=torch.float32)
        std = torch.as_tensor(vae.config.latents_std, dtype=torch.float32)
        self.register_buffer("latents_mean", mean.view(1, -1, 1, 1, 1))
        self.register_buffer("latents_std", std.view(1, -1, 1, 1, 1))

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """``[B, C, T, H, W]`` in ``[-1, 1]`` -> normalised latents (deterministic)."""
        vae_dtype = next(self.vae.parameters()).dtype
        latent = self.vae.encode(frames.to(vae_dtype)).latent_dist.mode()
        return (latent - self.latents_mean.to(latent)) / self.latents_std.to(latent)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Normalised latents -> ``[B, C, T, H, W]`` in ``[-1, 1]``.

        Note ``.sample`` here is the output field of the decoder result object,
        not the ``DiagonalGaussianDistribution.sample()`` method.
        """
        vae_dtype = next(self.vae.parameters()).dtype
        latent = latent.to(vae_dtype)
        latent = latent * self.latents_std.to(latent) + self.latents_mean.to(latent)
        return self.vae.decode(latent).sample


def _resolve_weights(checkpoint_dir, wan_root):
    """Pick the transformer and decoder paths, applying the reference's rules.

    EMA DiT must be paired with ``vae_decoder_ema.pt``; mixing an EMA DiT with
    the online decoder would silently combine weights from two different
    training runs. When the matching decoder is missing the reference keeps the
    *base* decoder -- it does not fall back to the other variant.
    """
    from pathlib import Path

    ckpt = Path(checkpoint_dir)
    ema_path = ckpt / "transformer_ema"
    online_path = ckpt / "transformer"
    decoder_ema = ckpt / "vae_decoder_ema.pt"
    decoder_online = ckpt / "vae_decoder.pt"

    if ema_path.exists():
        transformer_path = ema_path
        decoder_path = str(decoder_ema) if decoder_ema.exists() else None
        variant = "ema"
    elif online_path.exists():
        transformer_path = online_path
        decoder_path = str(decoder_online) if decoder_online.exists() else None
        variant = "online"
    else:
        raise FileNotFoundError(
            f"No transformer weights in checkpoint dir: {ckpt} "
            f"(looked for transformer_ema/ and transformer/)"
        )
    return transformer_path, decoder_path, variant


class VSRRestorer(nn.Module):
    """One-step restoration of a single pixel window.

    The window is ``[B, C, tile_t, tile_h, tile_w]`` in ``[-1, 1]`` and the
    result has the same shape. Tiling, blending and streaming live in
    ``blending`` / ``stream``; this class only owns the model call.
    """

    def __init__(
        self,
        vae: VSRAutoencoder,
        dit,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        tile_t: int = 33,
        tile_h: int = 320,
        tile_w: int = 640,
        t_overlap: int = 5,
        s_overlap: int = 32,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.tile_t = tile_t
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.t_overlap = t_overlap
        self.s_overlap = s_overlap

        self.vae = vae.to(device, dtype=dtype).eval()
        self.dit = dit.to(device, dtype=dtype).eval()

        # Zero text conditioning. Deliberately a zero tensor and not the
        # embedding of an empty string: the reference loads no text encoder at
        # all. See requirements.md §7-3.
        text_dim = self.dit.config.text_dim
        self.register_buffer(
            "empty_prompt", torch.zeros(1, 1, text_dim, dtype=dtype, device=self.device)
        )

    @torch.no_grad()
    def restore_window(self, window: torch.Tensor) -> torch.Tensor:
        """FM one-step restore of ``[B, C, tile_t, H, W]``.

        ``z_hq = z_lq - v(z_lq, t=1000)`` -- the rectified-flow sigma=1 endpoint.
        """
        window = window.to(self.device, dtype=self.dtype)
        B = window.shape[0]

        z_lq = self.vae.encode(window)
        t = torch.full((B,), FM_TIMESTEP, device=self.device, dtype=torch.float32)
        prompt = self.empty_prompt.expand(B, -1, -1)
        velocity = self.dit(
            hidden_states=z_lq,
            timestep=t,
            encoder_hidden_states=prompt,
            return_dict=False,
        )[0]
        z_hq = z_lq - velocity
        return self.vae.decode(z_hq)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str,
        wan_root: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        tile_t: int = 33,
        tile_h: int = 320,
        tile_w: int = 640,
        t_overlap: int = 5,
        s_overlap: int = 32,
        verbose: bool = True,
    ) -> "VSRRestorer":
        """Load a Stage-3 checkpoint.

        Args:
            checkpoint_dir: holds ``transformer_ema/`` + ``vae_decoder_ema.pt``,
                or the non-EMA ``transformer/`` + ``vae_decoder.pt``.
            wan_root: base Wan2.2-TI2V-5B-Diffusers directory. Only ``vae/`` is
                read from it -- the DiT always comes from ``checkpoint_dir``.
        """
        from diffusers import AutoencoderKLWan, WanTransformer3DModel

        transformer_path, decoder_path, variant = _resolve_weights(checkpoint_dir, wan_root)
        if verbose:
            print(f"[vsr] transformer: {transformer_path} ({variant})")
            if decoder_path:
                print(f"[vsr] fine-tuned VAE decoder: {decoder_path}")
            else:
                print("[vsr] no fine-tuned VAE decoder found; using the base decoder")

        base_vae = AutoencoderKLWan.from_pretrained(wan_root, subfolder="vae")
        vae = VSRAutoencoder(base_vae, decoder_state_path=decoder_path)
        dit = WanTransformer3DModel.from_pretrained(str(transformer_path))

        return cls(
            vae=vae,
            dit=dit,
            device=device,
            dtype=dtype,
            tile_t=tile_t,
            tile_h=tile_h,
            tile_w=tile_w,
            t_overlap=t_overlap,
            s_overlap=s_overlap,
        )
