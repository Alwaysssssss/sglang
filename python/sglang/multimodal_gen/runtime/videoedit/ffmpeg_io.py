# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import re
import subprocess
from fractions import Fraction
from typing import Any

import numpy as np
from PIL import Image


_CODEC_ENCODER_MAP = {
    "h264": "libx264",
    "hevc": "libx265",
    "h265": "libx265",
    "mpeg4": "mpeg4",
    "mjpeg": "mjpeg",
    "prores": "prores_ks",
    "vp8": "libvpx",
    "vp9": "libvpx-vp9",
}


def _parse_fps(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    try:
        return float(Fraction(value))
    except Exception:
        return None


def probe_video_profile(video_path: str) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-show_format",
        "-print_format",
        "json",
        video_path,
    ]
    result = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)
    video_streams = [
        stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"
    ]
    if not video_streams:
        raise ValueError(f"No video stream found in reference file: {video_path}")

    stream = video_streams[0]
    fmt = payload.get("format", {})
    fps = _parse_fps(stream.get("avg_frame_rate")) or _parse_fps(
        stream.get("r_frame_rate")
    )
    bit_rate = stream.get("bit_rate") or fmt.get("bit_rate")

    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "codec_name": stream.get("codec_name"),
        "pix_fmt": stream.get("pix_fmt"),
        "bit_rate": int(bit_rate) if bit_rate else None,
        "color_space": stream.get("color_space"),
        "color_transfer": stream.get("color_transfer"),
        "color_primaries": stream.get("color_primaries"),
        "field_order": stream.get("field_order"),
        "profile": stream.get("profile"),
        "level": stream.get("level"),
    }


def _encoder_for_codec(codec_name: str | None) -> str:
    if not codec_name:
        return "libx264"
    return _CODEC_ENCODER_MAP.get(codec_name.lower(), "libx264")


def _quality_to_crf(quality: int | float) -> int:
    quality = max(0.0, min(10.0, float(quality)))
    return int(round(51 - quality * 5.1))


def _normalize_profile_name(profile: str | None) -> str | None:
    if not profile:
        return None
    normalized = re.sub(r"[^a-z0-9]", "", profile.lower())
    return normalized or None


def _normalize_level(level: Any) -> str | None:
    if level is None:
        return None
    try:
        level_int = int(level)
    except (TypeError, ValueError):
        level_text = str(level).strip()
        return level_text or None

    if level_int >= 10:
        major = level_int // 10
        minor = level_int % 10
        return f"{major}.{minor}"
    return str(level_int)


def _as_rgb24_array(frame: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(frame, Image.Image):
        arr = np.asarray(frame.convert("RGB"))
    else:
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Unsupported video frame shape: {arr.shape}")

    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(arr)


def _build_ffmpeg_cmd(
    output_path: str,
    width: int,
    height: int,
    fps: float,
    profile: dict[str, Any],
    quality: int | float | None,
    loglevel: str,
) -> list[str]:
    encoder = _encoder_for_codec(profile.get("codec_name"))
    pix_fmt = profile.get("pix_fmt") or "yuv420p"
    normalized_profile = _normalize_profile_name(profile.get("profile"))
    normalized_level = _normalize_level(profile.get("level"))
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        loglevel,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        encoder,
        "-pix_fmt",
        pix_fmt,
        "-r",
        str(fps),
    ]

    if quality is not None and encoder in {"libx264", "libx265"}:
        cmd.extend(["-crf", str(_quality_to_crf(quality))])
    elif profile.get("bit_rate"):
        bit_rate = str(profile["bit_rate"])
        cmd.extend(["-b:v", bit_rate])
        if encoder in {"libx264", "libx265"}:
            cmd.extend(["-minrate", bit_rate, "-maxrate", bit_rate, "-bufsize", bit_rate])
            if encoder == "libx264":
                cmd.extend(["-x264-params", "nal-hrd=cbr:force-cfr=1"])

    if profile.get("color_space") and profile["color_space"] != "unknown":
        cmd.extend(["-colorspace", profile["color_space"]])
    if profile.get("color_transfer") and profile["color_transfer"] != "unknown":
        cmd.extend(["-color_trc", profile["color_transfer"]])
    if profile.get("color_primaries") and profile["color_primaries"] != "unknown":
        cmd.extend(["-color_primaries", profile["color_primaries"]])

    field_order = profile.get("field_order")
    if field_order in {"progressive", "tt", "bb", "tb", "bt"}:
        cmd.extend(["-field_order", field_order])

    if encoder in {"libx264", "libx265"} and normalized_profile:
        cmd.extend(["-profile:v", normalized_profile])
    if encoder in {"libx264", "libx265"} and normalized_level:
        cmd.extend(["-level:v", normalized_level])

    cmd.append(output_path)
    return cmd


def save_video_frames_like_reference(
    frames: list[Image.Image] | list[np.ndarray],
    output_path: str,
    refer_file: str,
    fps: float | None = None,
    quality: int | float | None = None,
    loglevel: str = "warning",
) -> str:
    if not frames:
        raise ValueError("No video frames to save")

    profile = probe_video_profile(refer_file)
    first = _as_rgb24_array(frames[0])
    height, width = first.shape[:2]
    output_fps = float(fps or profile.get("fps") or 24.0)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cmd = _build_ffmpeg_cmd(
        output_path=output_path,
        width=width,
        height=height,
        fps=output_fps,
        profile=profile,
        quality=quality,
        loglevel=loglevel,
    )

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        process.stdin.write(first.tobytes())
        for frame in frames[1:]:
            arr = _as_rgb24_array(frame)
            if arr.shape[:2] != (height, width):
                raise ValueError(
                    "All frames must have the same size, got "
                    f"{arr.shape[:2]} and expected {(height, width)}"
                )
            process.stdin.write(arr.tobytes())
        process.stdin.close()
        stderr_bytes = process.stderr.read() if process.stderr is not None else b""
        if process.stderr is not None:
            process.stderr.close()
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        return_code = process.wait()
    except Exception:
        process.kill()
        process.wait()
        if process.stderr is not None:
            process.stderr.close()
        raise

    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed while saving video: {stderr}")
    return output_path
