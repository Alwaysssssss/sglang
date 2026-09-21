# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for VSR.

Business parameters live here rather than as ad-hoc CLI flags, so they cannot be
mistaken for component paths by ``ServerArgs._extract_component_paths()`` -- the
failure mode the VideoEdit docs call out. Only genuine weight locations belong
in ``ServerArgs`` (``-model-path`` and ``-wan-root``).

Every field here that affects the output mirrors the reference CLI one-for-one,
so a reference run can be reproduced flag for flag (``requirements.md`` §5.1).
"""

from __future__ import annotations

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class WanVRSamplingParams(SamplingParams):
    # --- text conditioning: there is none ------------------------------------
    # The framework's request model insists on a string prompt, and VSR has no
    # text encoder at all -- the DiT is conditioned on a zero tensor, not on the
    # embedding of an empty string (requirements.md §7-3). Declaring the empty
    # default here keeps every caller from having to supply a meaningless field,
    # and keeps the fact that it is meaningless in one place.
    prompt: str = ""

    # --- input ---------------------------------------------------------------
    # The reference's `--input`. Single file only in phase 1; directory batching
    # is deferred (requirements.md §5.1).
    video_input_path: str | None = None

    # --- geometry ------------------------------------------------------------
    # Exact target size as "HxW" (height first), overriding long_edge. Spelled
    # the same way as the reference CLI, deliberately.
    target_resolution: str | None = None
    long_edge: int | None = None

    # --- tiling (must match training) ---------------------------------------
    tile_t: int | None = None
    tile_h: int | None = None
    tile_w: int | None = None
    temporal_overlap: int | None = None
    spatial_overlap: int | None = None

    # --- colour --------------------------------------------------------------
    # "global" | "chunk" | "none". `global` runs a cheap pre-pass over sampled
    # source frames; that pre-pass is part of the reference's semantics, not an
    # optional optimisation (requirements.md §7-6).
    color_ref: str | None = None
    color_ref_samples: int | None = None

    # --- precision / encoding ------------------------------------------------
    dtype: str | None = None
    crf: int | None = None

    # --- streaming -----------------------------------------------------------
    # The memory knobs. They do not change the output, but they must be settable
    # for a performance comparison to be comparable (requirements.md §5.1).
    read_queue: int | None = None
    write_queue: int | None = None
    gpu_postprocess: bool | None = None

    # --- debug ---------------------------------------------------------------
    save_tiles_dir: str | None = None

    # --- filled in by the stage ---------------------------------------------
    # Scratch fields written by the pipeline, never sent back in a response.
    runtime_frames_written: int | None = None
    runtime_target_h: int | None = None
    runtime_target_w: int | None = None

    @classmethod
    def from_user_kwargs(cls, server_args, *args, **kwargs) -> WanVRSamplingParams:
        """Build params from caller kwargs, then let the base class normalise.

        Each model family defines its own — this mirrors the VideoEdit one. The
        two things it drops are a ``diffusers_kwargs`` passthrough and a
        ``negative_prompt`` of None; VSR has no text conditioning, so
        ``negative_prompt`` would be meaningless even if set, but the base
        class's ``_adjust`` still validates the field set, so it is filtered
        the same way rather than special-cased.
        """
        user_kwargs = dict(kwargs)
        user_kwargs.pop("diffusers_kwargs", None)
        if user_kwargs.get("negative_prompt") is None:
            user_kwargs.pop("negative_prompt", None)
        params = cls(*args, **user_kwargs)
        params._adjust(server_args)
        params._validate_with_pipeline_config(server_args.pipeline_config)
        return params
