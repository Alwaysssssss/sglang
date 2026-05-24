from __future__ import annotations

import argparse
import json
import os
import time
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
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to launch with.",
    )
    parser.add_argument(
        "--enable-cfg-parallel",
        action="store_true",
        help="Enable classifier-free guidance parallelism across two GPU ranks.",
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
        "--enable-torch-compile",
        action="store_true",
        help="Enable torch.compile for the DiT runtime.",
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
        "--enable-stage-logging",
        action="store_true",
        help="Enable internal stage timing collection through SGLANG_DIFFUSION_STAGE_LOGGING.",
    )
    parser.add_argument(
        "--sync-stage-profiling",
        action="store_true",
        help="Synchronize step timing around stage profiling for more stable denoise timings.",
    )
    parser.add_argument(
        "--enable-batched-cfg",
        dest="enable_batched_cfg",
        action="store_true",
        help="Force STAR batched CFG on for this request.",
    )
    parser.add_argument(
        "--disable-batched-cfg",
        dest="enable_batched_cfg",
        action="store_false",
        help="Force STAR batched CFG off for this request.",
    )
    parser.set_defaults(enable_batched_cfg=None)
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


def _summarize_internal_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    stages = metrics.get("stages") or {}
    step_durations = metrics.get("steps") or []
    internal_total_ms = float(metrics.get("total_duration_ms") or 0.0)
    denoise_total_ms = float(sum(step_durations))
    return {
        "internal_total_duration_ms": internal_total_ms,
        "internal_total_duration_s": internal_total_ms / 1000.0,
        "denoise_total_duration_ms": denoise_total_ms,
        "denoise_total_duration_s": denoise_total_ms / 1000.0,
        "denoise_step_count": len(step_durations),
        "denoise_step_avg_ms": (
            denoise_total_ms / len(step_durations) if step_durations else None
        ),
        "stage_durations_ms": stages,
    }


def main() -> int:
    args = _parse_args()
    if args.enable_stage_logging:
        os.environ["SGLANG_DIFFUSION_STAGE_LOGGING"] = "1"
    if args.sync_stage_profiling:
        os.environ["SGLANG_DIFFUSION_SYNC_STAGE_PROFILING"] = "1"
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
        "num_gpus": args.num_gpus,
        "enable_cfg_parallel": args.enable_cfg_parallel,
        "component_paths": parse_component_overrides(args.component_path),
        "disable_autocast": args.disable_autocast,
        "enable_torch_compile": args.enable_torch_compile,
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
    if args.enable_batched_cfg is not None:
        sampling_params_kwargs["enable_batched_cfg"] = args.enable_batched_cfg
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

    total_start = time.perf_counter()
    load_start = time.perf_counter()
    with DiffGenerator.from_pretrained(local_mode=True, **server_kwargs) as generator:
        load_duration_s = time.perf_counter() - load_start
        generate_start = time.perf_counter()
        result = generator.generate(sampling_params_kwargs=sampling_params_kwargs)
        generate_duration_s = time.perf_counter() - generate_start
    total_duration_s = time.perf_counter() - total_start

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

    metrics_summary = _summarize_internal_metrics(result.metrics or {})
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
            "enable_batched_cfg": args.enable_batched_cfg,
            "enable_color_fix": args.enable_color_fix,
            "color_fix_mode": args.color_fix_mode,
        },
        "server": {
            "attention_backend": args.attention_backend,
            "num_gpus": args.num_gpus,
            "enable_cfg_parallel": args.enable_cfg_parallel,
            "disable_autocast": args.disable_autocast,
            "enable_torch_compile": args.enable_torch_compile,
            "dit_cpu_offload": args.dit_cpu_offload,
            "text_encoder_cpu_offload": args.text_encoder_cpu_offload,
            "vae_cpu_offload": args.vae_cpu_offload,
            "component_paths": server_kwargs["component_paths"],
        },
        "timing": {
            "load_duration_s": load_duration_s,
            "generate_wall_clock_s": generate_duration_s,
            "total_wall_clock_s": total_duration_s,
            **metrics_summary,
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
