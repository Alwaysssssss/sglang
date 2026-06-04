# SPDX-License-Identifier: Apache-2.0
import math
from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.vividvr_defaults import (
    DEFAULT_VIVIDVR_NEGATIVE_PROMPT,
    DEFAULT_VIVIDVR_PROMPT_FILE_PATH,
)


@dataclass
class VividVRSamplingParams(SamplingParams):
    """Sampling contract for the Stage A/C VividVR integration."""

    video_input_path: str | None = None
    prompt_file_path: str | None = DEFAULT_VIVIDVR_PROMPT_FILE_PATH
    caption_source: str = "prompt_file"
    use_live_cogvlm2_caption: bool = False
    cogvlm2_model_path: str | None = None

    dtype: str = "bf16"
    enable_spatial_tiling: bool = True
    enable_temporal_tiling: bool = False
    tile_size: int = 128
    tile_stride: int = 64
    num_temporal_process_frames: int = 121
    restoration_guidance_scale: float = -1.0

    height: int = 720
    width: int = 960
    num_frames: int = 121
    fps: int = 25
    guidance_scale: float = 6.0
    num_inference_steps: int = 50
    negative_prompt: str | None = DEFAULT_VIVIDVR_NEGATIVE_PROMPT
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [(960, 720)]
    )

    # Runtime request-derived fields
    runtime_prompt_file_path: str | None = field(default=None, init=False, repr=False)
    runtime_raw_prompt_text: str | None = field(default=None, init=False, repr=False)
    runtime_model_prompt_text: str | None = field(default=None, init=False, repr=False)
    runtime_negative_prompt_text: str | None = field(default=None, init=False, repr=False)
    runtime_do_cfg: bool = field(default=False, init=False, repr=False)

    # Runtime input / condition fields
    runtime_control_video: Any | None = field(default=None, init=False, repr=False)
    runtime_reference_video: Any | None = field(default=None, init=False, repr=False)
    runtime_original_height: int | None = field(default=None, init=False, repr=False)
    runtime_original_width: int | None = field(default=None, init=False, repr=False)
    runtime_original_num_frames: int | None = field(default=None, init=False, repr=False)
    runtime_num_padding_frames: int | None = field(default=None, init=False, repr=False)
    runtime_padded_input_frames: int | None = field(default=None, init=False, repr=False)
    runtime_fps: int | None = field(default=None, init=False, repr=False)

    # Runtime tensor fields
    runtime_prompt_embeds: Any | None = field(default=None, init=False, repr=False)
    runtime_negative_prompt_embeds: Any | None = field(default=None, init=False, repr=False)
    runtime_tiled_prompt_embeds: Any | None = field(default=None, init=False, repr=False)
    runtime_tiled_negative_prompt_embeds: Any | None = field(
        default=None, init=False, repr=False
    )
    runtime_control_latents: Any | None = field(default=None, init=False, repr=False)
    runtime_generator: Any | None = field(default=None, init=False, repr=False)
    runtime_latents: Any | None = field(default=None, init=False, repr=False)
    runtime_num_latent_padding_frames: int | None = field(
        default=None, init=False, repr=False
    )
    runtime_tiling_infos: list[Any] | None = field(default=None, init=False, repr=False)
    runtime_tile_count: int | None = field(default=None, init=False, repr=False)
    runtime_timesteps: Any | None = field(default=None, init=False, repr=False)
    runtime_timestep_count: int | None = field(default=None, init=False, repr=False)
    runtime_progress: float | None = field(default=None, init=False, repr=False)
    runtime_decoded_video: Any | None = field(default=None, init=False, repr=False)
    runtime_output_video: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.prompt_path = self._normalize_prompt_file_aliases()
        self.prompt_file_path = self.prompt_path
        super().__post_init__()
        self._validate_vividvr()

    def _normalize_prompt_file_aliases(self) -> str | None:
        if self.prompt_path and self.prompt_file_path:
            if self.prompt_path != self.prompt_file_path:
                raise ValueError(
                    "VividVR prompt_path and prompt_file_path must match when both are provided"
                )
            return self.prompt_path
        return self.prompt_path or self.prompt_file_path

    def _validate_vividvr(self) -> None:
        if self.caption_source != "prompt_file":
            raise ValueError(
                "VividVR integration currently only supports caption_source='prompt_file'"
            )
        if self.use_live_cogvlm2_caption:
            raise ValueError(
                "VividVR integration does not allow live CogVLM2 captioning during Stage A"
            )
        if self.cogvlm2_model_path not in (None, ""):
            raise ValueError(
                "VividVR integration reserves CogVLM2 model loading for a later stage; "
                "leave cogvlm2_model_path empty for now"
            )
        if self.num_outputs_per_prompt != 1:
            raise ValueError("VividVR only supports num_outputs_per_prompt=1")
        if self.dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError(f"dtype must be one of bf16/fp16/fp32, got {self.dtype!r}")
        if not isinstance(self.tile_size, int) or self.tile_size <= 0:
            raise ValueError(f"tile_size must be a positive int, got {self.tile_size!r}")
        if not isinstance(self.tile_stride, int) or self.tile_stride <= 0:
            raise ValueError(
                f"tile_stride must be a positive int, got {self.tile_stride!r}"
            )
        if self.tile_stride > self.tile_size:
            raise ValueError(
                f"tile_stride must be <= tile_size, got tile_stride={self.tile_stride!r}, "
                f"tile_size={self.tile_size!r}"
            )
        if (
            not isinstance(self.num_temporal_process_frames, int)
            or self.num_temporal_process_frames <= 0
        ):
            raise ValueError(
                "num_temporal_process_frames must be a positive int, "
                f"got {self.num_temporal_process_frames!r}"
            )
        if (self.num_temporal_process_frames - 1) % 8 != 0:
            raise ValueError(
                "num_temporal_process_frames must satisfy (num_temporal_process_frames - 1) % 8 == 0"
            )
        if (
            isinstance(self.restoration_guidance_scale, bool)
            or not isinstance(self.restoration_guidance_scale, (int, float))
            or not math.isfinite(float(self.restoration_guidance_scale))
        ):
            raise ValueError(
                "restoration_guidance_scale must be a finite number, "
                f"got {self.restoration_guidance_scale!r}"
            )

    def _validate_with_pipeline_config(self, pipeline_config):
        super()._validate_with_pipeline_config(pipeline_config)
        if self.video_input_path is None:
            raise ValueError("VividVR requires video_input_path")
        if self.prompt_path is None:
            raise ValueError("VividVR requires prompt_file_path/prompt_path")

    @classmethod
    def from_user_kwargs(cls, server_args, *args, **kwargs) -> "VividVRSamplingParams":
        user_kwargs = dict(kwargs)
        user_kwargs.pop("diffusers_kwargs", None)
        if (
            "prompt_path" not in user_kwargs
            and "prompt_file_path" not in user_kwargs
            and getattr(server_args, "prompt_file_path", None) is not None
        ):
            user_kwargs["prompt_file_path"] = server_args.prompt_file_path
        params = cls(*args, **user_kwargs)
        params._adjust(server_args)
        params._validate_with_pipeline_config(server_args.pipeline_config)
        return params

    def reset_runtime(self) -> None:
        self.runtime_prompt_file_path = None
        self.runtime_raw_prompt_text = None
        self.runtime_model_prompt_text = None
        self.runtime_negative_prompt_text = None
        self.runtime_do_cfg = False

        self.runtime_control_video = None
        self.runtime_reference_video = None
        self.runtime_original_height = None
        self.runtime_original_width = None
        self.runtime_original_num_frames = None
        self.runtime_num_padding_frames = None
        self.runtime_padded_input_frames = None
        self.runtime_fps = None

        self.runtime_prompt_embeds = None
        self.runtime_negative_prompt_embeds = None
        self.runtime_tiled_prompt_embeds = None
        self.runtime_tiled_negative_prompt_embeds = None
        self.runtime_control_latents = None
        self.runtime_generator = None
        self.runtime_latents = None
        self.runtime_num_latent_padding_frames = None
        self.runtime_tiling_infos = None
        self.runtime_tile_count = None
        self.runtime_timesteps = None
        self.runtime_timestep_count = None
        self.runtime_progress = None
        self.runtime_decoded_video = None
        self.runtime_output_video = None
