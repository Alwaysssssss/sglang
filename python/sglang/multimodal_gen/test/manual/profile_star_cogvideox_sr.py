from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.test.manual.run_star_cogvideox_sr_smoke import (
    _read_video_metadata,
    _save_frame_pngs,
    _summarize_frames,
    _summarize_internal_metrics,
    parse_component_overrides,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile a STAR CogVideoX-SR run with warmup/measured iterations and parity reports."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--condition-video-path", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-path", default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--reference-video", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-frames", type=int, default=7)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--condition-video-num-frames", type=int, default=25)
    parser.add_argument("--pipeline-class-name", default="StarCogVideoXSRPipeline")
    parser.add_argument("--backend", default="sglang", choices=["sglang", "diffusers", "auto"])
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--enable-cfg-parallel", action="store_true")
    parser.add_argument("--component-path", action="append", default=[])
    parser.add_argument("--enable-torch-compile", action="store_true")
    parser.add_argument("--disable-autocast", action="store_true")
    parser.add_argument("--dit-cpu-offload", action="store_true")
    parser.add_argument("--text-encoder-cpu-offload", action="store_true")
    parser.add_argument("--vae-cpu-offload", action="store_true")
    parser.add_argument(
        "--enable-batched-cfg",
        dest="enable_batched_cfg",
        action="store_true",
    )
    parser.add_argument(
        "--disable-batched-cfg",
        dest="enable_batched_cfg",
        action="store_false",
    )
    parser.set_defaults(enable_batched_cfg=None)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--measured-runs", type=int, default=1)
    parser.add_argument("--save-frame-pngs", action="store_true")
    parser.add_argument("--enable-stage-logging", action="store_true")
    parser.add_argument("--sync-stage-profiling", action="store_true")
    parser.add_argument("--original-star-cold-e2e-s", type=float, default=None)
    parser.add_argument("--original-star-warm-e2e-s", type=float, default=None)
    parser.add_argument("--original-star-denoise-s", type=float, default=None)
    return parser.parse_args()


def _load_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    if args.prompt_path is not None:
        return Path(args.prompt_path).read_text(encoding="utf-8").strip()
    raise ValueError("Either --prompt or --prompt-path is required.")


def _run_compare(reference: str, candidate: str, mode: str, output_json: Path) -> dict[str, Any]:
    command = [
        "python",
        "-m",
        "sglang.multimodal_gen.test.manual.compare_star_sglang_outputs",
        "--reference",
        reference,
        "--candidate",
        candidate,
        "--mode",
        mode,
        "--output-json",
        str(output_json),
    ]
    result = subprocess.run(command, check=False)
    if not output_json.exists():
        raise RuntimeError(
            f"Compare command for mode={mode} exited with code {result.returncode} "
            f"without producing {output_json}."
        )
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    payload["compare_exit_code"] = result.returncode
    return payload


def _build_server_kwargs(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
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
        kwargs["attention_backend"] = args.attention_backend
    return kwargs


def _build_capability_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "attention_backend": {
            "runtime_request_backend": args.attention_backend or "auto",
            "star_transformer_backend": "torch_sdpa_only",
            "non_torch_sdpa_supported": False,
        },
        "parallel": {
            "cfg_parallel": "supported" if args.enable_cfg_parallel else "available_but_disabled",
            "tp": "not_integrated",
            "sp_ulysses_ring": "not_integrated",
        },
        "cache": {
            "teacache": "not_integrated_for_star",
            "cache_dit": "not_integrated_for_star",
        },
        "quantization": {
            "status": "not_integrated_for_star",
        },
    }


def _build_sampling_kwargs(
    args: argparse.Namespace,
    prompt: str,
    run_output_dir: Path,
    output_file_name: str,
    *,
    save_output: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "condition_video_path": args.condition_video_path,
        "negative_prompt": args.negative_prompt if args.negative_prompt is not None else "",
        "seed": args.seed,
        "num_frames": args.num_frames,
        "height": args.height,
        "width": args.width,
        "fps": args.fps,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "condition_video_num_frames": args.condition_video_num_frames,
        "output_path": str(run_output_dir),
        "output_file_name": output_file_name,
        "save_output": save_output,
        "return_file_paths_only": False,
    }
    if args.enable_batched_cfg is not None:
        kwargs["enable_batched_cfg"] = args.enable_batched_cfg
    return kwargs


def main() -> int:
    args = _parse_args()
    if args.enable_stage_logging:
        os.environ["SGLANG_DIFFUSION_STAGE_LOGGING"] = "1"
    if args.sync_stage_profiling:
        os.environ["SGLANG_DIFFUSION_SYNC_STAGE_PROFILING"] = "1"

    prompt = _load_prompt(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    server_kwargs = _build_server_kwargs(args, output_dir)
    run_records: list[dict[str, Any]] = []
    measured_records: list[dict[str, Any]] = []

    total_start = time.perf_counter()
    load_start = time.perf_counter()
    with DiffGenerator.from_pretrained(local_mode=True, **server_kwargs) as generator:
        load_duration_s = time.perf_counter() - load_start
        total_runs = args.warmup_runs + args.measured_runs
        last_result = None
        last_frames: list[np.ndarray] | None = None
        for run_idx in range(total_runs):
            is_warmup = run_idx < args.warmup_runs
            phase = "warmup" if is_warmup else "measured"
            phase_index = run_idx if is_warmup else run_idx - args.warmup_runs
            run_dir = runs_dir / f"{run_idx:02d}_{phase}"
            run_dir.mkdir(parents=True, exist_ok=True)
            run_output_name = "candidate.mp4"
            sampling_kwargs = _build_sampling_kwargs(
                args,
                prompt,
                run_dir,
                run_output_name,
                save_output=not is_warmup,
            )
            run_start = time.perf_counter()
            result = generator.generate(sampling_params_kwargs=sampling_kwargs)
            wall_clock_s = time.perf_counter() - run_start
            if isinstance(result, list):
                result = result[0]
            if result is None:
                raise RuntimeError(f"Generation failed for run {run_idx}.")
            if result.frames is not None:
                frames = [np.asarray(frame) for frame in result.frames]
            elif result.output_file_path:
                frames = [np.asarray(frame) for frame in iio.imiter(result.output_file_path)]
            else:
                raise RuntimeError(f"No frames or output path returned for run {run_idx}.")

            metrics_summary = _summarize_internal_metrics(result.metrics or {})
            record = {
                "run_index": run_idx,
                "phase": phase,
                "phase_index": phase_index,
                "output_file_path": result.output_file_path,
                "generation_time_s": float(result.generation_time),
                "wall_clock_s": wall_clock_s,
                "peak_memory_mb": float(result.peak_memory_mb or 0.0),
                "timing": metrics_summary,
                "frame_summary": _summarize_frames(frames),
            }
            run_records.append(record)
            if not is_warmup:
                measured_records.append(record)
                last_result = result
                last_frames = frames

    total_duration_s = time.perf_counter() - total_start
    if not measured_records:
        raise ValueError("At least one measured run is required.")
    assert last_result is not None
    assert last_frames is not None

    avg_generation_time_s = sum(r["generation_time_s"] for r in measured_records) / len(
        measured_records
    )
    avg_wall_clock_s = sum(r["wall_clock_s"] for r in measured_records) / len(
        measured_records
    )
    avg_internal_total_s = sum(
        float(r["timing"]["internal_total_duration_s"]) for r in measured_records
    ) / len(measured_records)
    avg_denoise_s = sum(
        float(r["timing"]["denoise_total_duration_s"]) for r in measured_records
    ) / len(measured_records)

    baseline_report = None
    strict_report = None
    if args.reference_video and last_result.output_file_path:
        baseline_report = _run_compare(
            args.reference_video,
            last_result.output_file_path,
            "baseline",
            output_dir / "compare_baseline.json",
        )
        strict_report = _run_compare(
            args.reference_video,
            last_result.output_file_path,
            "strict",
            output_dir / "compare_strict.json",
        )

    speedup = {}
    if args.original_star_cold_e2e_s:
        speedup["cold_e2e_speedup"] = (
            float(args.original_star_cold_e2e_s) / float(total_duration_s)
        )
    if args.original_star_warm_e2e_s:
        speedup["warm_e2e_speedup"] = (
            float(args.original_star_warm_e2e_s) / float(avg_wall_clock_s)
        )
    if args.original_star_denoise_s:
        speedup["denoise_speedup"] = (
            float(args.original_star_denoise_s) / float(avg_denoise_s)
        )

    summary = {
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "reference_video": _read_video_metadata(args.reference_video),
        "condition_video": _read_video_metadata(args.condition_video_path),
        "request": {
            "prompt": prompt,
            "negative_prompt": args.negative_prompt,
            "seed": args.seed,
            "num_frames": args.num_frames,
            "height": args.height,
            "width": args.width,
            "fps": args.fps,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "condition_video_num_frames": args.condition_video_num_frames,
            "enable_batched_cfg": args.enable_batched_cfg,
        },
        "server": {
            "backend": args.backend,
            "attention_backend": args.attention_backend,
            "num_gpus": args.num_gpus,
            "enable_cfg_parallel": args.enable_cfg_parallel,
            "enable_torch_compile": args.enable_torch_compile,
            "disable_autocast": args.disable_autocast,
            "dit_cpu_offload": args.dit_cpu_offload,
            "text_encoder_cpu_offload": args.text_encoder_cpu_offload,
            "vae_cpu_offload": args.vae_cpu_offload,
            "component_paths": server_kwargs["component_paths"],
        },
        "capabilities": _build_capability_summary(args),
        "profile": {
            "load_duration_s": load_duration_s,
            "total_wall_clock_s": total_duration_s,
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.measured_runs,
            "avg_generation_time_s": avg_generation_time_s,
            "avg_wall_clock_s": avg_wall_clock_s,
            "avg_internal_total_s": avg_internal_total_s,
            "avg_denoise_s": avg_denoise_s,
            "last_candidate_path": last_result.output_file_path,
            "speedup": speedup,
        },
        "runs": run_records,
        "parity": {
            "baseline": baseline_report,
            "strict": strict_report,
        },
    }

    if args.save_frame_pngs:
        summary["saved_frame_paths"] = _save_frame_pngs(last_frames, output_dir)

    summary_path = output_dir / "summary.json"
    profile_path = output_dir / "profile.json"
    payload = json.dumps(summary, indent=2, ensure_ascii=False)
    summary_path.write_text(payload, encoding="utf-8")
    profile_path.write_text(payload, encoding="utf-8")

    print(f"summary:   {summary_path}")
    print(f"profile:   {profile_path}")
    print(f"candidate: {last_result.output_file_path}")
    if "warm_e2e_speedup" in speedup:
        print(f"warm_e2e_speedup: {speedup['warm_e2e_speedup']:.4f}")
    if "denoise_speedup" in speedup:
        print(f"denoise_speedup: {speedup['denoise_speedup']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
