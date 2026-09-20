# SPDX-License-Identifier: Apache-2.0
"""Pipeline config for VSR (video super-resolution / restoration)."""

from __future__ import annotations

from dataclasses import dataclass

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.wan import Wan2_2_TI2V_5B_Config


@dataclass
class WanVSRPipelineConfig(Wan2_2_TI2V_5B_Config):
    """Configuration for the VSR restore pipeline.

    It inherits from the Wan2.2 TI2V-5B config because the checkpoint *is* a
    fine-tuned TI2V-5B transformer plus a fine-tuned VAE decoder -- the DiT
    shape (48 in / 48 out, patch [1, 2, 2], text_dim 4096) is stock, so reusing
    that config keeps the declared shapes honest instead of inventing a
    near-duplicate.

    What differs is everything around the model, declared here so the stage
    carries no magic numbers:

    * no text encoder and no tokenizer are loaded -- the conditioning is a zero
      tensor, not the embedding of an empty string (``requirements.md`` §7-3);
    * there is no scheduler and no denoising loop: each tile gets a single DiT
      forward at a fixed timestep (see ``runtime/vsr/model.py``);
    * the unit of work is a 3D pixel-space tile, not a whole latent volume, so
      the inherited latent-shape helpers go unused.
    """

    task_type: ModelTaskType = ModelTaskType.VSR

    # --- tiling: must match training (the reference's configs/default.yaml)
    tile_t: int = 33
    tile_h: int = 320
    tile_w: int = 640
    temporal_overlap: int = 5
    spatial_overlap: int = 32

    # --- colour
    color_ref: str = "global"
    color_ref_samples: int = 64

    # --- streaming / encoding
    read_queue: int = 2
    write_queue: int = 4
    crf: int = 5

    # --- geometry; None means "fall back to long_edge"
    # Spelled H x W, matching the reference implementation. Reading it as W x H
    # is the single most likely way to get a silently wrong output.
    target_resolution: str | None = None
    long_edge: int = 3840
