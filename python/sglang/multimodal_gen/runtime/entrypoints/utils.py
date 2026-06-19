# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
DiffGenerator module for sglang-diffusion.

This module provides a consolidated interface for generating videos using
diffusion models.
"""

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Union

import imageio
import numpy as np
import torch

try:
    import scipy.io.wavfile as scipy_wavfile
except ImportError:  # pragma: no cover
    scipy_wavfile = None

try:
    import imageio_ffmpeg as _imageio_ffmpeg
except ImportError:  # pragma: no cover
    _imageio_ffmpeg = None

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)
from sglang.multimodal_gen.configs.pipeline_configs.vividvr import (
    VividVRPipelineConfig,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import CYAN, RESET, init_logger

logger = init_logger(__name__)

VIDEO_ENCODING_MODE_REFERENCE_PROFILE = "reference_profile"
VIDEO_ENCODING_MODE_VIVIDVR_ORIGINAL = "vividvr_original"


@dataclass
class SetLoraReq:
    lora_nickname: Union[str, List[str]]
    lora_path: Optional[Union[str, List[Optional[str]]]] = None
    target: Union[str, List[str]] = "all"
    strength: Union[float, List[float]] = 1.0


@dataclass
class MergeLoraWeightsReq:
    target: str = "all"
    strength: float = 1.0


@dataclass
class UnmergeLoraWeightsReq:
    target: str = "all"


@dataclass
class ListLorasReq:
    pass


@dataclass
class ShutdownReq:
    pass


def resolve_video_reference_path(
    *,
    request_like: Any = None,
    server_args: Optional[ServerArgs] = None,
    explicit_path: Optional[str] = None,
) -> Optional[str]:
    """Resolve the reference video profile used when saving generated videos.

    Prefer a request-level reference override when one is available. This keeps
    long-video VividVR service requests aligned with the correct reference
    profile instead of always reusing the single-clip default.
    """
    if explicit_path:
        return explicit_path

    request_reference_path = getattr(request_like, "reference_video_path", None)
    if request_reference_path:
        return request_reference_path

    if server_args is not None and isinstance(
        server_args.pipeline_config, VividVRPipelineConfig
    ):
        reference_video_path = getattr(
            server_args.pipeline_config, "reference_video_path", None
        )
        if reference_video_path:
            return reference_video_path

    return getattr(request_like, "video_input_path", None)


def is_vividvr_pipeline(server_args: Optional[ServerArgs]) -> bool:
    return server_args is not None and isinstance(
        getattr(server_args, "pipeline_config", None), VividVRPipelineConfig
    )


def resolve_video_encoding_mode(server_args: Optional[ServerArgs]) -> str:
    return VIDEO_ENCODING_MODE_REFERENCE_PROFILE


def resolve_video_encoding_quality(
    *,
    server_args: Optional[ServerArgs],
    output_compression: Optional[int],
) -> float:
    if output_compression is not None:
        return output_compression / 10
    if is_vividvr_pipeline(server_args):
        return 8
    return 5


def format_lora_message(
    lora_nickname: Union[str, List[str]],
    target: Union[str, List[str]],
    strength: Union[float, List[float]],
) -> tuple[str, str, str]:
    """Format success message for single or multiple LoRAs."""
    if isinstance(lora_nickname, list):
        nickname_str = ", ".join(lora_nickname)
        target_str = ", ".join(target) if isinstance(target, list) else target
        strength_str = (
            ", ".join(f"{s:.2f}" for s in strength)
            if isinstance(strength, list)
            else f"{strength:.2f}"
        )
    else:
        nickname_str = lora_nickname
        target_str = target if isinstance(target, str) else ", ".join(target)
        strength_str = (
            f"{strength:.2f}"
            if isinstance(strength, (int, float))
            else ", ".join(f"{s:.2f}" for s in strength)
        )
    return nickname_str, target_str, strength_str


@dataclass
class GenerationResult:
    """Result of a single generation request from DiffGenerator."""

    samples: Any = None
    frames: Any = None
    audio: Any = None
    prompt: str | None = None
    size: tuple | None = None  # (height, width, num_frames)
    generation_time: float = 0.0
    peak_memory_mb: float = 0.0
    metrics: dict = field(default_factory=dict)
    trajectory_latents: Any = None
    trajectory_timesteps: Any = None
    trajectory_decoded: Any = None
    prompt_index: int = 0
    output_file_path: str | None = None


def _normalize_audio_to_numpy(audio: Any) -> np.ndarray | None:
    """Convert audio (torch / numpy) into a float32 numpy array in [-1, 1], best-effort."""
    if audio is None:
        return None
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().float().clamp(-1.0, 1.0).cpu().numpy()
    elif isinstance(audio, np.ndarray):
        audio_np = audio.astype(np.float32, copy=False)
        audio_np = np.clip(audio_np, -1.0, 1.0)
    else:
        return None

    # 1. Squeeze leading singleton dimensions (Batch, etc.)
    while audio_np.ndim > 1 and audio_np.shape[0] == 1:
        audio_np = audio_np.squeeze(0)

    # 2. Handle (C, L) -> (L, C)
    if audio_np.ndim == 2 and audio_np.shape[0] < audio_np.shape[1]:
        audio_np = audio_np.transpose(1, 0)

    # 3. Final safety check: if still 2D and channels (dim 1) is huge, something is wrong
    if audio_np.ndim == 2 and audio_np.shape[1] > 256 and audio_np.shape[0] == 1:
        audio_np = audio_np.flatten()

    return audio_np


def _pick_audio_sample_rate(
    *,
    audio_np: np.ndarray,
    audio_sample_rate: Optional[int],
    fps: int,
    num_frames: int,
) -> int:
    """Pick a plausible sample rate, falling back to inferring from video duration."""
    selected_sr = int(audio_sample_rate) if audio_sample_rate is not None else None
    if selected_sr is None or not (8000 <= selected_sr <= 192000):
        selected_sr = 24000
        try:
            duration_s = float(num_frames) / float(fps) if fps else 0.0
            if duration_s > 0:
                audio_len = (
                    int(audio_np.shape[0])
                    if audio_np.ndim == 2
                    else int(audio_np.shape[-1])
                )
                inferred_sr = int(round(float(audio_len) / duration_s))
                if 8000 <= inferred_sr <= 192000:
                    selected_sr = inferred_sr
        except Exception:
            pass
    return selected_sr


def _resolve_ffmpeg_exe() -> str:
    ffmpeg_exe = "ffmpeg"
    ffmpeg_on_path = shutil.which("ffmpeg")
    if ffmpeg_on_path:
        ffmpeg_exe = ffmpeg_on_path
    try:
        if _imageio_ffmpeg is not None:
            ffmpeg_exe = _imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    ffmpeg_ok = False
    if ffmpeg_exe:
        if os.path.isabs(ffmpeg_exe):
            ffmpeg_ok = os.path.exists(ffmpeg_exe)
        else:
            ffmpeg_ok = shutil.which(ffmpeg_exe) is not None
    if not ffmpeg_ok:
        raise RuntimeError("ffmpeg not found")
    return ffmpeg_exe


def _mux_audio_np_into_mp4(
    *,
    save_file_path: str,
    audio_np: np.ndarray,
    sample_rate: int,
    ffmpeg_exe: str,
) -> None:
    merged_path = save_file_path.rsplit(".", 1)[0] + ".tmp_mux.mp4"
    tmp_wav_path = None
    try:
        if scipy_wavfile is None:
            raise RuntimeError(
                "scipy is required to mux audio into mp4 (pip install scipy)"
            )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav_path = f.name
        scipy_wavfile.write(tmp_wav_path, sample_rate, audio_np)
        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-i",
                save_file_path,
                "-i",
                tmp_wav_path,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-strict",
                "experimental",
                merged_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.replace(merged_path, save_file_path)
    finally:
        if tmp_wav_path:
            try:
                os.remove(tmp_wav_path)
            except OSError:
                pass
        if os.path.exists(merged_path):
            try:
                os.remove(merged_path)
            except OSError:
                pass


def _maybe_mux_audio_into_mp4(
    *,
    save_file_path: str,
    audio: Any,
    frames: list,
    fps: int,
    audio_sample_rate: Optional[int],
) -> None:
    """Best-effort mux audio into an already-written mp4 at save_file_path.

    Any failure should keep the silent video and only log a warning.
    """
    audio_np = _normalize_audio_to_numpy(audio)
    if audio_np is None:
        return
    selected_sr = _pick_audio_sample_rate(
        audio_np=audio_np,
        audio_sample_rate=audio_sample_rate,
        fps=fps,
        num_frames=len(frames),
    )

    try:
        ffmpeg_exe = _resolve_ffmpeg_exe()
        _mux_audio_np_into_mp4(
            save_file_path=save_file_path,
            audio_np=audio_np,
            sample_rate=selected_sr,
            ffmpeg_exe=ffmpeg_exe,
        )
    except Exception as e:
        logger.warning(
            "Failed to mux audio into mp4 (saved silent video): %s",
            str(e),
        )


def prepare_request(
    server_args: ServerArgs,
    sampling_params: SamplingParams,
) -> Req:
    """
    Create a Req object with sampling_params as a parameter.
    """
    req = Req(
        sampling_params=sampling_params,
        VSA_sparsity=server_args.attention_backend_config.VSA_sparsity,
    )
    try:
        diffusers_kwargs = sampling_params.diffusers_kwargs
    except AttributeError:
        diffusers_kwargs = None
    if diffusers_kwargs:
        req.extra["diffusers_kwargs"] = diffusers_kwargs

    req.adjust_size(server_args)

    if not isinstance(req.prompt, str):
        raise TypeError(f"`prompt` must be a string, but got {type(req.prompt)}")

    if (req.width is not None and req.width <= 0) or (
        req.height is not None and req.height <= 0
    ):
        raise ValueError(
            f"Height and width must be positive, got height={req.height}, width={req.width}"
        )

    return req


def attach_audio_to_video_sample(
    sample: Any,
    audio: Any,
    output_idx: int,
) -> Any:
    """Attach per-sample audio for video outputs when available."""
    if audio is None:
        return sample
    if isinstance(audio, torch.Tensor) and audio.ndim >= 2:
        audio = audio[output_idx] if audio.shape[0] > output_idx else None
    elif isinstance(audio, np.ndarray) and audio.ndim >= 2:
        audio = audio[output_idx] if audio.shape[0] > output_idx else None

    if audio is not None and not (
        isinstance(sample, (tuple, list)) and len(sample) == 2
    ):
        return (sample, audio)
    return sample


def _split_outputs_into_samples(outputs: Any, data_type: DataType) -> list[Any]:
    """Normalize generator outputs into a per-sample list."""
    if isinstance(outputs, torch.Tensor):
        if data_type == DataType.VIDEO:
            if outputs.dim() == 4:
                return [outputs]
            if outputs.dim() == 5:
                return [outputs[idx] for idx in range(outputs.shape[0])]
        elif data_type == DataType.IMAGE:
            if outputs.dim() == 3:
                return [outputs]
            if outputs.dim() == 4:
                return [outputs[idx] for idx in range(outputs.shape[0])]
        return [outputs]

    if isinstance(outputs, np.ndarray):
        if data_type == DataType.VIDEO:
            if outputs.ndim == 4:
                return [outputs]
            if outputs.ndim == 5:
                return [outputs[idx] for idx in range(outputs.shape[0])]
        elif data_type == DataType.IMAGE:
            if outputs.ndim == 3:
                return [outputs]
            if outputs.ndim == 4 and outputs.shape[-1] in (1, 3, 4):
                return [outputs]
            if outputs.ndim == 4:
                return [outputs[idx] for idx in range(outputs.shape[0])]
            if outputs.ndim == 5:
                return [outputs[idx] for idx in range(outputs.shape[0])]
        return [outputs]

    if isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes)):
        return list(outputs)

    return [outputs]


def save_outputs(
    outputs: Sequence[Any],
    data_type: DataType,
    fps: int,
    save_output: bool,
    build_output_path: Callable[[int], str],
    *,
    audio: Any = None,
    audio_sample_rate: Optional[int] = None,
    samples_out: Optional[list[Any]] = None,
    audios_out: Optional[list[Any]] = None,
    frames_out: Optional[list[Any]] = None,
    output_compression: Optional[int] = None,
    enable_frame_interpolation: bool = False,
    frame_interpolation_exp: int = 1,
    frame_interpolation_scale: float = 1.0,
    frame_interpolation_model_path: Optional[str] = None,
    enable_upscaling: bool = False,
    upscaling_model_path: Optional[str] = None,
    upscaling_scale: int = 4,
    video_reference_path: Optional[str] = None,
    video_encoding_mode: str = VIDEO_ENCODING_MODE_REFERENCE_PROFILE,
    default_video_quality: Optional[float] = None,
) -> list[str]:
    """Save outputs to files and return the list of file paths."""
    output_paths: list[str] = []
    normalized_outputs = _split_outputs_into_samples(outputs, data_type)
    for idx, output in enumerate(normalized_outputs):
        save_file_path = build_output_path(idx)
        sample = output
        if data_type == DataType.VIDEO:
            sample = attach_audio_to_video_sample(sample, audio, idx)

        frames = post_process_sample(
            sample,
            data_type,
            fps,
            save_output,
            save_file_path,
            audio_sample_rate=audio_sample_rate,
            output_compression=output_compression,
            enable_frame_interpolation=enable_frame_interpolation,
            frame_interpolation_exp=frame_interpolation_exp,
            frame_interpolation_scale=frame_interpolation_scale,
            frame_interpolation_model_path=frame_interpolation_model_path,
            enable_upscaling=enable_upscaling,
            upscaling_model_path=upscaling_model_path,
            upscaling_scale=upscaling_scale,
            video_reference_path=video_reference_path,
            video_encoding_mode=video_encoding_mode,
            default_video_quality=default_video_quality,
        )

        if samples_out is not None:
            samples_out.append(sample)
        if audios_out is not None:
            if data_type == DataType.VIDEO:
                audio_item = audio
                if isinstance(audio, torch.Tensor) and audio.ndim >= 2:
                    audio_item = audio[idx] if audio.shape[0] > idx else None
                elif isinstance(audio, np.ndarray) and audio.ndim >= 2:
                    audio_item = audio[idx] if audio.shape[0] > idx else None
                audios_out.append(audio_item)
            else:
                audios_out.append(audio)
        if frames_out is not None:
            frames_out.append(frames)
        output_paths.append(save_file_path)
    return output_paths


def post_process_sample(
    sample: Any,
    data_type: DataType,
    fps: int,
    save_output: bool = True,
    save_file_path: Optional[str] = None,
    audio_sample_rate: Optional[int] = None,
    output_compression: Optional[int] = None,
    enable_frame_interpolation: bool = False,
    frame_interpolation_exp: int = 1,
    frame_interpolation_scale: float = 1.0,
    frame_interpolation_model_path: Optional[str] = None,
    enable_upscaling: bool = False,
    upscaling_model_path: Optional[str] = None,
    upscaling_scale: int = 4,
    video_reference_path: Optional[str] = None,
    video_encoding_mode: str = VIDEO_ENCODING_MODE_REFERENCE_PROFILE,
    default_video_quality: Optional[float] = None,
):
    """
    Process sample output, optionally interpolate video frames, and save.
    """
    audio = None
    if isinstance(sample, (tuple, list)) and len(sample) == 2:
        sample, audio = sample

    # 1. Convert tensor / array to list of uint8 HWC frames
    frames = None
    if isinstance(sample, torch.Tensor):
        if sample.dim() == 3:
            sample = sample.unsqueeze(1)
        sample = (sample * 255).clamp(0, 255).to(torch.uint8)
        videos = sample.permute(1, 2, 3, 0).cpu().numpy()
        frames = list(videos)
    else:
        if not isinstance(sample, np.ndarray):
            raise TypeError(f"Unsupported sample type: {type(sample)}")

        arr = sample
        if arr.ndim == 3:
            if arr.shape[-1] in (1, 3, 4):
                arr = arr[None, ...]
            else:
                arr = arr[..., None]
        if arr.ndim != 4:
            raise ValueError(f"Unexpected numpy sample shape: {tuple(arr.shape)}")

        if arr.shape[-1] not in (1, 3, 4) and arr.shape[0] in (1, 3, 4):
            t = torch.from_numpy(arr)
            if t.dim() == 3:
                t = t.unsqueeze(1)
            t = (t * 255).clamp(0, 255).to(torch.uint8)
            videos = t.permute(1, 2, 3, 0).cpu().numpy()
            frames = list(videos)
        else:
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            frames = list(arr)

    # 2. Frame interpolation (video only)
    if enable_frame_interpolation and data_type == DataType.VIDEO and len(frames) > 1:
        from sglang.multimodal_gen.runtime.postprocess import (
            interpolate_video_frames,
        )

        frames, multiplier = interpolate_video_frames(
            frames,
            exp=frame_interpolation_exp,
            scale=frame_interpolation_scale,
            model_path=frame_interpolation_model_path,
        )
        fps = fps * multiplier

    # 3. Upscaling (images and videos)
    if enable_upscaling and frames:
        from sglang.multimodal_gen.runtime.postprocess import upscale_frames

        frames = upscale_frames(
            frames,
            model_path=upscaling_model_path,
            scale=upscaling_scale,
        )

    # 4. Save outputs if requested
    if save_output:
        if save_file_path:
            os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
            if data_type == DataType.VIDEO:
                imageio_quality = (
                    default_video_quality
                    if default_video_quality is not None
                    else (
                        output_compression / 10
                        if output_compression is not None
                        else 5
                    )
                )
                saved_with_reference = False
                if (
                    video_encoding_mode == VIDEO_ENCODING_MODE_REFERENCE_PROFILE
                    and video_reference_path
                ):
                    try:
                        from sglang.multimodal_gen.runtime.videoedit.ffmpeg_io import (
                            save_video_frames_like_reference,
                        )

                        save_video_frames_like_reference(
                            frames,
                            save_file_path,
                            refer_file=video_reference_path,
                            fps=fps,
                            quality=(
                                imageio_quality
                                if output_compression is not None
                                else None
                            ),
                        )
                        saved_with_reference = True
                    except Exception as e:
                        logger.warning(
                            "Failed to save video with reference profile %s: %s. "
                            "Falling back to default imageio writer.",
                            video_reference_path,
                            e,
                        )

                if not saved_with_reference:
                    if video_encoding_mode == VIDEO_ENCODING_MODE_VIVIDVR_ORIGINAL:
                        from sglang.multimodal_gen.runtime.videoedit.io import (
                            save_video_frames,
                        )

                        save_video_frames(
                            frames,
                            save_file_path,
                            fps=fps,
                            quality=imageio_quality,
                        )
                    else:
                        imageio.mimsave(
                            save_file_path,
                            frames,
                            fps=fps,
                            format=data_type.get_default_extension(),
                            codec="libx264",
                            quality=imageio_quality,
                        )

                _maybe_mux_audio_into_mp4(
                    save_file_path=save_file_path,
                    audio=audio,
                    frames=frames,
                    fps=fps,
                    audio_sample_rate=audio_sample_rate,
                )

            else:
                quality = output_compression if output_compression is not None else 75
                if len(frames) > 1:
                    for i, image in enumerate(frames):
                        parts = save_file_path.rsplit(".", 1)
                        if len(parts) == 2:
                            indexed_path = f"{parts[0]}_{i}.{parts[1]}"
                        else:
                            indexed_path = f"{save_file_path}_{i}"
                        imageio.imwrite(indexed_path, image, quality=quality)
                else:
                    imageio.imwrite(save_file_path, frames[0], quality=quality)
            logger.info(f"Output saved to {CYAN}{save_file_path}{RESET}")
        else:
            logger.info(f"No output path provided, output not saved")

    return frames
