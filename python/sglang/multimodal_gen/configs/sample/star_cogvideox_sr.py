# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class StarCogVideoXSRSamplingParams(SamplingParams):
    """Sampling parameters for STAR CogVideoX video super-resolution."""

    negative_prompt: str = ""
    condition_video_path: str | None = None
    condition_video_start_frame: int | None = None
    condition_video_num_frames: int | None = None
    condition_video_sample_fps: int | None = None
    condition_video_frame_stride: int | None = None

    enable_color_fix: bool = False
    color_fix_mode: str | None = None

    guidance_scale: float = 6.0
    num_inference_steps: int = 50
