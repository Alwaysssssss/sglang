from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio
import imageio.v3 as iio
import numpy as np


def parse_component_overrides(entries: list[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(
                f"Invalid component override '{entry}'. Expected component=path."
            )
        component, path = entry.split("=", 1)
        component = component.strip().replace("-", "_")
        path = path.strip()
        if not component or not path:
            raise ValueError(
                f"Invalid component override '{entry}'. Expected component=path."
            )
        overrides[component] = path
    return overrides


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a STAR CogVideoX-SR SGLang smoke generation and emit a summary JSON."
    )
    parser.add_argument("--model-path", required=True, help="Converted STAR model dir.")
    parser.add_argument(
        "--condition-video-path",
        required=True,
        help="Low-quality condition video used for STAR SR.",
    )
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Optional negative prompt override.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for outputs.")
    parser.add_argument(
        "--output-file-name",
        default="star_sglang_candidate.mp4",
        help="Saved candidate file name.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional summary JSON path. Defaults to <output-dir>/star_smoke_summary.json.",
    )
    parser.add_argument(
        "--reference-video",
        default=None,
        help="Optional STAR_mg reference mp4 path for manifest bookkeeping.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=7)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument(
        "--condition-video-start-frame",
        type=int,
        default=None,
        help="Optional condition-video start frame.",
    )
    parser.add_argument(
        "--condition-video-num-frames",
        type=int,
        default=None,
        help="Optional number of condition-video frames to sample.",
    )
    parser.add_argument(
        "--condition-video-sample-fps",
        type=int,
        default=None,
        help="Optional sampled FPS for the condition video.",
    )
    parser.add_argument(
        "--condition-video-frame-stride",
        type=int,
        default=None,
        help="Optional frame stride for the condition video.",
    )
    parser.add_argument(
        "--pipeline-class-name",
        default="StarCogVideoXSRPipeline",
        help="Pipeline class to force during launch.",
    )
    parser.add_argument(
        "--backend",
        default="sglang",
        choices=["sglang", "diffusers", "auto"],
        help="Runtime backend.",
    )
    parser.add_argument(
        "--attention-backend",
        default=None,
        help="Optional attention backend override.",
    )
    parser.add_argument(
        "--component-path",
        action="append",
        default=[],
        help="Optional component override in component=path form.",
    )
    parser.add_argument(
        "--disable-autocast",
        action="store_true",
        help="Disable autocast for the generator run.",
    )
    parser.add_argument(
        "--dit-cpu-offload",
        action="store_true",
        help="Enable DiT CPU offload.",
    )
    parser.add_argument(
        "--text-encoder-cpu-offload",
        action="store_true",
        help="Enable text-encoder CPU offload.",
    )
    parser.add_argument(
        "--vae-cpu-offload",
        action="store_true",
        help="Enable VAE CPU offload.",
    )
    parser.add_argument(
        "--save-frame-pngs",
        action="store_true",
        help="Also write candidate frames as PNGs.",
    )
    parser.add_argument(
        "--enable-color-fix",
        action="store_true",
        help="Request color-fix mode. Phase5 parity should normally leave this off.",
    )
    parser.add_argument(
        "--color-fix-mode",
        default=None,
        help="Optional color-fix mode label for bookkeeping.",
    )
    return parser.parse_args()


def _read_video_metadata(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return {
            "path": str(resolved),
            "exists": False,
        }
    frames = [np.asarray(frame) for frame in iio.imiter(resolved)]
    fps = None
    try:
        with imageio.get_reader(resolved) as reader:
            fps = reader.get_meta_data().get("fps")
    except Exception:
        fps = None
    first_shape = list(frames[0].shape) if frames else None
    return {
        "path": str(resolved),
        "exists": True,
        "num_frames": len(frames),
        "fps": fps,
        "first_frame_shape": first_shape,
    }


def _summarize_frames(frames: list[np.ndarray]) -> dict[str, Any]:
    frame_stack = np.stack([np.asarray(frame) for frame in frames], axis=0)
    return {
        "num_frames": int(frame_stack.shape[0]),
        "frame_shape": list(frame_stack.shape[1:]),
        "pixel_mean": float(frame_stack.mean()),
        "pixel_std": float(frame_stack.std()),
        "pixel_min": int(frame_stack.min()),
        "pixel_max": int(frame_stack.max()),
        "first_frame_mean": float(frame_stack[0].mean()),
        "last_frame_mean": float(frame_stack[-1].mean()),
    }


def _save_frame_pngs(frames: list[np.ndarray], output_dir: Path) -> list[str]:
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    for index, frame in enumerate(frames):
        path = frame_dir / f"frame_{index:04d}.png"
        iio.imwrite(path, np.asarray(frame))
        saved_paths.append(str(path))
    return saved_paths


def main() -> int:
    args = _parse_args()
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = (
        Path(args.summary_json).expanduser().resolve()
        if args.summary_json
        else output_dir / "star_smoke_summary.json"
    )

    server_kwargs = {
        "model_path": args.model_path,
        "pipeline_class_name": args.pipeline_class_name,
        "backend": args.backend,
        "output_path": str(output_dir),
        "component_paths": parse_component_overrides(args.component_path),
        "disable_autocast": args.disable_autocast,
        "dit_cpu_offload": args.dit_cpu_offload,
        "text_encoder_cpu_offload": args.text_encoder_cpu_offload,
        "vae_cpu_offload": args.vae_cpu_offload,
    }
    if args.attention_backend:
        server_kwargs["attention_backend"] = args.attention_backend

    sampling_params_kwargs = {
        "prompt": args.prompt,
        "condition_video_path": args.condition_video_path,
        "seed": args.seed,
        "num_frames": args.num_frames,
        "height": args.height,
        "width": args.width,
        "fps": args.fps,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "output_path": str(output_dir),
        "output_file_name": args.output_file_name,
        "save_output": True,
        "return_file_paths_only": False,
        "enable_color_fix": args.enable_color_fix,
        "color_fix_mode": args.color_fix_mode,
    }
    if args.negative_prompt is not None:
        sampling_params_kwargs["negative_prompt"] = args.negative_prompt
    if args.condition_video_start_frame is not None:
        sampling_params_kwargs["condition_video_start_frame"] = (
            args.condition_video_start_frame
        )
    if args.condition_video_num_frames is not None:
        sampling_params_kwargs["condition_video_num_frames"] = (
            args.condition_video_num_frames
        )
    if args.condition_video_sample_fps is not None:
        sampling_params_kwargs["condition_video_sample_fps"] = (
            args.condition_video_sample_fps
        )
    if args.condition_video_frame_stride is not None:
        sampling_params_kwargs["condition_video_frame_stride"] = (
            args.condition_video_frame_stride
        )

    with DiffGenerator.from_pretrained(local_mode=True, **server_kwargs) as generator:
        result = generator.generate(sampling_params_kwargs=sampling_params_kwargs)

    if result is None:
        raise RuntimeError("Smoke generation returned no result.")
    if isinstance(result, list):
        if len(result) != 1:
            raise RuntimeError(
                f"Expected a single GenerationResult, received {len(result)} results."
            )
        result = result[0]

    if result.frames is not None:
        frames = [np.asarray(frame) for frame in result.frames]
    elif result.output_file_path:
        frames = [np.asarray(frame) for frame in iio.imiter(result.output_file_path)]
    else:
        raise RuntimeError(
            "Smoke generation did not return frames or an output file path."
        )

    summary = {
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "pipeline_class_name": args.pipeline_class_name,
        "backend": args.backend,
        "request": {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "condition_video_path": str(
                Path(args.condition_video_path).expanduser().resolve()
            ),
            "seed": args.seed,
            "num_frames": args.num_frames,
            "height": args.height,
            "width": args.width,
            "fps": args.fps,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "condition_video_start_frame": args.condition_video_start_frame,
            "condition_video_num_frames": args.condition_video_num_frames,
            "condition_video_sample_fps": args.condition_video_sample_fps,
            "condition_video_frame_stride": args.condition_video_frame_stride,
            "enable_color_fix": args.enable_color_fix,
            "color_fix_mode": args.color_fix_mode,
        },
        "server": {
            "attention_backend": args.attention_backend,
            "disable_autocast": args.disable_autocast,
            "dit_cpu_offload": args.dit_cpu_offload,
            "text_encoder_cpu_offload": args.text_encoder_cpu_offload,
            "vae_cpu_offload": args.vae_cpu_offload,
            "component_paths": server_kwargs["component_paths"],
        },
        "result": {
            "output_file_path": result.output_file_path,
            "generation_time": result.generation_time,
            "peak_memory_mb": result.peak_memory_mb,
            "size": list(result.size) if result.size is not None else None,
            "metrics": result.metrics,
            "frame_summary": _summarize_frames(frames),
        },
        "condition_video": _read_video_metadata(args.condition_video_path),
        "reference_video": _read_video_metadata(args.reference_video),
    }

    if args.save_frame_pngs:
        summary["result"]["saved_frame_paths"] = _save_frame_pngs(frames, output_dir)

    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"candidate: {result.output_file_path}")
    print(f"summary:   {summary_json_path}")
    print(f"frames:    {summary['result']['frame_summary']['num_frames']}")
    print(f"shape:     {summary['result']['frame_summary']['frame_shape']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
