# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models import EncoderConfig
from sglang.multimodal_gen.configs.models.dits.cogvideox import CogVideoXConfig
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.t5 import T5Config
from sglang.multimodal_gen.configs.models.vaes.cogvideox import CogVideoXVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.wan import t5_postprocess_text
from sglang.multimodal_gen.configs.vividvr_defaults import (
    DEFAULT_VIVIDVR_NEGATIVE_PROMPT,
    DEFAULT_VIVIDVR_PROMPT_FILE_PATH,
    DEFAULT_VIVIDVR_REFERENCE_VIDEO_PATH,
)


@dataclass
class VividVRPipelineConfig(PipelineConfig):
    """Stage A contract for the VividVR integration.

    This config deliberately freezes the current integration policy:
    captioning is file-driven during integration, and live CogVLM2 inference is
    not part of the active runtime path yet.
    """

    task_type: ModelTaskType = ModelTaskType.VIDEO_EDIT
    dit_config: CogVideoXConfig = field(default_factory=CogVideoXConfig)
    vae_config: CogVideoXVAEConfig = field(default_factory=CogVideoXVAEConfig)

    precision: str = "bf16"
    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    vae_tiling: bool = True
    vae_sp: bool = False

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = (
        field(default_factory=lambda: (t5_postprocess_text,))
    )

    base_model_family: str = "CogVideoX1.5-5B"
    caption_source: str = "prompt_file"
    default_prompt_file_path: str = DEFAULT_VIVIDVR_PROMPT_FILE_PATH
    reference_video_path: str = DEFAULT_VIVIDVR_REFERENCE_VIDEO_PATH
    allow_live_cogvlm2_caption: bool = False
    cogvlm2_model_path: str | None = None

    tile_size: int = 128
    tile_stride: int = 64
    num_temporal_process_frames: int = 121
    restoration_guidance_scale: float = -1.0
    default_negative_prompt: str = DEFAULT_VIVIDVR_NEGATIVE_PROMPT

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True

    def adjust_num_frames(self, num_frames):
        return num_frames
