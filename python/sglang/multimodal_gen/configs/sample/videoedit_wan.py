# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams
from sglang.multimodal_gen.configs.sample.wan_teacache import _wan_1_3b_coefficients


DEFAULT_VIDEOEDIT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


@dataclass
class WanVideoEditSamplingParams(SamplingParams):
    # Request fields
    video_input_path: str | None = None
    mask_input_path: str | None = None
    reference_image_path: str | None = None
    infer_len: int = 81
    overlap: int = 0
    strength: float = 1.0
    dtype: str = "bf16"

    dynamic_cfg: bool = True
    dynamic_cfg_max_step: int = 15
    dynamic_cfg_min: float = 1.0

    bbox_padding: int = 0
    dilate_px: int = 15
    mask_scale: float = 1.2
    feather_px: int = 12
    adain_boundary_dilate: int = 15

    enable_paste_back: bool = True
    save_crop_only: bool = False
    drop_reference_frame: bool = True
    keep_intermediate_windows: bool = False
    use_repaired_context: bool = True
    vary_seed_by_window: bool = False

    # VideoEdit defaults
    num_frames: int = 81
    fps: int = 16
    guidance_scale: float = 5.0
    num_inference_steps: int = 20
    negative_prompt: str | None = DEFAULT_VIDEOEDIT_NEGATIVE_PROMPT
    teacache_params: TeaCacheParams = field(
        default_factory=lambda: TeaCacheParams(
            teacache_thresh=0.08,
            use_ret_steps=True,
            coefficients_callback=_wan_1_3b_coefficients,
            start_skipping=5,
            end_skipping=1.0,
        )
    )

    # Global runtime fields
    runtime_original_frames: list[Any] | None = field(default=None, init=False, repr=False)
    runtime_original_masks: list[Any] | None = field(default=None, init=False, repr=False)
    runtime_resized_frames: list[Any] | None = field(default=None, init=False, repr=False)
    runtime_resized_masks: list[Any] | None = field(default=None, init=False, repr=False)
    runtime_dilated_cropped_masks: list[Any] | None = field(default=None, init=False, repr=False)
    runtime_window_specs: list[Any] | None = field(default=None, init=False, repr=False)
    runtime_accum_frames: Any | None = field(default=None, init=False, repr=False)
    runtime_accum_weights: Any | None = field(default=None, init=False, repr=False)
    runtime_bbox: tuple[int, int, int, int] | None = field(default=None, init=False, repr=False)
    runtime_crop_h: int | None = field(default=None, init=False, repr=False)
    runtime_crop_w: int | None = field(default=None, init=False, repr=False)
    runtime_aligned_h: int | None = field(default=None, init=False, repr=False)
    runtime_aligned_w: int | None = field(default=None, init=False, repr=False)
    runtime_fps: float | None = field(default=None, init=False, repr=False)
    runtime_num_input_frames: int | None = field(default=None, init=False, repr=False)

    # Per-window runtime fields
    runtime_window_spec: Any | None = field(default=None, init=False, repr=False)
    runtime_window_frames: list[Any] | None = field(default=None, init=False, repr=False)
    runtime_window_masks: list[Any] | None = field(default=None, init=False, repr=False)
    runtime_window_index: int | None = field(default=None, init=False, repr=False)
    runtime_window_validated: bool = field(default=False, init=False, repr=False)
    runtime_height: int | None = field(default=None, init=False, repr=False)
    runtime_width: int | None = field(default=None, init=False, repr=False)
    runtime_num_frames: int | None = field(default=None, init=False, repr=False)

    # Stage runtime tensors
    runtime_prompt_embeds: Any | None = field(default=None, init=False, repr=False)
    runtime_negative_prompt_embeds: Any | None = field(default=None, init=False, repr=False)
    runtime_do_cfg: bool = field(default=False, init=False, repr=False)
    runtime_masked_video_tensor: Any | None = field(default=None, init=False, repr=False)
    runtime_raw_video_tensor: Any | None = field(default=None, init=False, repr=False)
    runtime_mask_video_tensor: Any | None = field(default=None, init=False, repr=False)
    runtime_cond_masks: Any | None = field(default=None, init=False, repr=False)
    runtime_cond_latents: Any | None = field(default=None, init=False, repr=False)
    runtime_video_latents: Any | None = field(default=None, init=False, repr=False)
    runtime_condition_latent: Any | None = field(default=None, init=False, repr=False)
    runtime_generator: Any | None = field(default=None, init=False, repr=False)
    runtime_noise: Any | None = field(default=None, init=False, repr=False)
    runtime_latents: Any | None = field(default=None, init=False, repr=False)
    runtime_timesteps: Any | None = field(default=None, init=False, repr=False)
    runtime_effective_num_inference_steps: int | None = field(default=None, init=False, repr=False)
    runtime_num_warmup_steps: int | None = field(default=None, init=False, repr=False)
    runtime_initial_timestep: Any | None = field(default=None, init=False, repr=False)
    runtime_current_step: int | None = field(default=None, init=False, repr=False)
    runtime_current_timestep: Any | None = field(default=None, init=False, repr=False)
    runtime_current_cfg: float | None = field(default=None, init=False, repr=False)
    runtime_progress: float | None = field(default=None, init=False, repr=False)
    runtime_decoded_video_tensor: Any | None = field(default=None, init=False, repr=False)
    runtime_window_output_frames: list[Any] | None = field(default=None, init=False, repr=False)
    runtime_window_metadata: dict[str, Any] | None = field(default=None, init=False, repr=False)
    runtime_output_video_path: str | None = field(default=None, init=False, repr=False)
    runtime_crop_video_path: str | None = field(default=None, init=False, repr=False)
    runtime_metadata_path: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_frames is not None and self.num_frames <= 0:
            raise ValueError(
                "VideoEdit num_frames must be positive after request normalization. "
                "Use num_frames=-1 only at API/CLI entrypoints to mean all frames."
            )
        super().__post_init__()
        self._validate_videoedit()

    def _validate_videoedit(self) -> None:
        if self.infer_len != 81:
            raise ValueError(f"VideoEdit infer_len must be 81, got {self.infer_len}")
        if (self.infer_len - 1) % 4 != 0:
            raise ValueError("VideoEdit infer_len must satisfy (infer_len - 1) % 4 == 0")
        if self.num_frames is not None and self.num_frames <= 0:
            raise ValueError(
                "VideoEdit num_frames must be positive after request normalization. "
                "Use num_frames=-1 only at API/CLI entrypoints to mean all frames."
            )
        if not (0 <= self.overlap < self.infer_len):
            raise ValueError(
                f"overlap must be in [0, {self.infer_len}), got {self.overlap}"
            )
        if not (0.0 < float(self.strength) <= 1.0):
            raise ValueError(f"strength must be in (0, 1], got {self.strength!r}")
        if self.num_outputs_per_prompt != 1:
            raise ValueError("VideoEdit only supports num_outputs_per_prompt=1")
        if self.dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError(f"dtype must be one of bf16/fp16/fp32, got {self.dtype!r}")

    def _validate_with_pipeline_config(self, pipeline_config):
        super()._validate_with_pipeline_config(pipeline_config)
        if self.video_input_path is None:
            raise ValueError("VideoEdit requires video_input_path")
        if self.mask_input_path is None:
            raise ValueError("VideoEdit requires mask_input_path")

    @classmethod
    def from_user_kwargs(cls, server_args, *args, **kwargs) -> "WanVideoEditSamplingParams":
        user_kwargs = dict(kwargs)
        user_kwargs.pop("diffusers_kwargs", None)
        if user_kwargs.get("negative_prompt") is None:
            user_kwargs.pop("negative_prompt", None)
        params = cls(*args, **user_kwargs)
        params._adjust(server_args)
        params._validate_with_pipeline_config(server_args.pipeline_config)
        return params

    def reset_window_runtime(self, window_spec: Any) -> None:
        self.runtime_window_spec = window_spec
        self.runtime_window_index = window_spec.window_index
        self.runtime_window_frames = None
        self.runtime_window_masks = None
        self.runtime_window_validated = False
        self.runtime_height = None
        self.runtime_width = None
        self.runtime_num_frames = None
        self.runtime_prompt_embeds = None
        self.runtime_negative_prompt_embeds = None
        self.runtime_do_cfg = False
        self.runtime_masked_video_tensor = None
        self.runtime_raw_video_tensor = None
        self.runtime_mask_video_tensor = None
        self.runtime_cond_masks = None
        self.runtime_cond_latents = None
        self.runtime_video_latents = None
        self.runtime_condition_latent = None
        self.runtime_generator = None
        self.runtime_noise = None
        self.runtime_latents = None
        self.runtime_timesteps = None
        self.runtime_effective_num_inference_steps = None
        self.runtime_num_warmup_steps = None
        self.runtime_initial_timestep = None
        self.runtime_current_step = None
        self.runtime_current_timestep = None
        self.runtime_current_cfg = None
        self.runtime_progress = None
        self.runtime_decoded_video_tensor = None
        self.runtime_window_output_frames = None
        self.runtime_window_metadata = None
