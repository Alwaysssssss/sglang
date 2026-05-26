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

from sglang.multimodal_gen.configs.quantization.nunchaku import (
    NunchakuSVDQuantArgs,
)
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.test.manual.run_star_cogvideox_sr_smoke import (
    _build_teacache_params,
    _configure_cache_dit_env,
    _read_video_metadata,
    _resolve_request_fps,
    _save_frame_pngs,
    _summarize_frames,
    _summarize_internal_metrics,
    _write_trace_artifacts,
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
    parser.add_argument("--reference-frame-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-frames", type=int, default=7)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--condition-video-num-frames", type=int, default=25)
    parser.add_argument(
        "--output-quality",
        default="default",
        choices=["default", "maximum", "high", "medium", "low"],
    )
    parser.add_argument("--output-compression", type=int, default=None)
    parser.add_argument("--pipeline-class-name", default="StarCogVideoXSRPipeline")
    parser.add_argument("--backend", default="sglang", choices=["sglang", "diffusers", "auto"])
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--transformer-weights-path", default=None)
    parser.add_argument("--enable-svdquant", action="store_true")
    parser.add_argument("--quantization-precision", default=None)
    parser.add_argument("--quantization-rank", type=int, default=None)
    parser.add_argument("--quantization-act-unsigned", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument("--sp-degree", type=int, default=None)
    parser.add_argument("--ulysses-degree", type=int, default=None)
    parser.add_argument("--ring-degree", type=int, default=None)
    parser.add_argument("--enable-cfg-parallel", action="store_true")
    parser.add_argument("--component-path", action="append", default=[])
    parser.add_argument("--enable-torch-compile", action="store_true")
    parser.add_argument("--disable-autocast", action="store_true")
    parser.add_argument("--dit-cpu-offload", action="store_true")
    parser.add_argument("--text-encoder-cpu-offload", action="store_true")
    parser.add_argument("--vae-cpu-offload", action="store_true")
    parser.add_argument(
        "--vae-tiling",
        dest="vae_tiling",
        action="store_true",
    )
    parser.add_argument(
        "--no-vae-tiling",
        dest="vae_tiling",
        action="store_false",
    )
    parser.set_defaults(vae_tiling=None)
    parser.add_argument(
        "--use-flashinfer-rope",
        dest="use_flashinfer_rope",
        action="store_true",
    )
    parser.add_argument(
        "--disable-flashinfer-rope",
        dest="use_flashinfer_rope",
        action="store_false",
    )
    parser.set_defaults(use_flashinfer_rope=None)
    parser.add_argument(
        "--local-enhancer-mode",
        choices=["legacy", "fused_5d"],
        default=None,
    )
    parser.add_argument(
        "--condition-video-vae-peak-memory-mode",
        choices=[
            "legacy",
            "off",
            "text_encoder_only",
            "transformer_only",
            "text_encoder_and_transformer",
            "auto",
        ],
        default=None,
    )
    parser.add_argument(
        "--condition-video-vae-target-headroom-gb",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--condition-video-vae-rng-mode",
        choices=["generator", "global_seed"],
        default=None,
    )
    parser.add_argument(
        "--release-text-encoder-after-prompt-encode",
        dest="release_text_encoder_after_prompt_encode",
        action="store_true",
    )
    parser.add_argument(
        "--keep-text-encoder-after-prompt-encode",
        dest="release_text_encoder_after_prompt_encode",
        action="store_false",
    )
    parser.set_defaults(release_text_encoder_after_prompt_encode=None)
    parser.add_argument(
        "--keep-transformer-gpu-resident-between-requests",
        dest="keep_transformer_gpu_resident_between_requests",
        action="store_true",
    )
    parser.add_argument(
        "--disable-keep-transformer-gpu-resident-between-requests",
        dest="keep_transformer_gpu_resident_between_requests",
        action="store_false",
    )
    parser.set_defaults(keep_transformer_gpu_resident_between_requests=None)
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
    parser.add_argument("--enable-teacache", action="store_true")
    parser.add_argument("--teacache-thresh", type=float, default=0.0)
    parser.add_argument("--teacache-start-skipping", type=float, default=5.0)
    parser.add_argument("--teacache-end-skipping", type=float, default=-1.0)
    parser.add_argument("--teacache-coefficients", default="1,0")
    parser.add_argument("--enable-cache-dit", action="store_true")
    parser.add_argument("--cache-dit-fn", type=int, default=1)
    parser.add_argument("--cache-dit-bn", type=int, default=0)
    parser.add_argument("--cache-dit-warmup", type=int, default=4)
    parser.add_argument("--cache-dit-rdt", type=float, default=0.24)
    parser.add_argument("--cache-dit-mc", type=int, default=3)
    parser.add_argument("--cache-dit-scm-preset", default="none")
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--measured-runs", type=int, default=1)
    parser.add_argument("--save-frame-pngs", action="store_true")
    parser.add_argument("--save-trace", action="store_true")
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
    pipeline_config_overrides: dict[str, Any] = {}
    if args.vae_tiling is not None:
        pipeline_config_overrides["vae_tiling"] = args.vae_tiling
    if args.use_flashinfer_rope is not None:
        pipeline_config_overrides["use_flashinfer_rope"] = args.use_flashinfer_rope
    if args.local_enhancer_mode is not None:
        pipeline_config_overrides["local_enhancer_mode"] = args.local_enhancer_mode
    if args.condition_video_vae_peak_memory_mode is not None:
        pipeline_config_overrides["condition_video_vae_peak_memory_mode"] = (
            args.condition_video_vae_peak_memory_mode
        )
    if args.condition_video_vae_target_headroom_gb is not None:
        pipeline_config_overrides["condition_video_vae_target_headroom_gb"] = (
            args.condition_video_vae_target_headroom_gb
        )
    if args.condition_video_vae_rng_mode is not None:
        pipeline_config_overrides["condition_video_vae_sample_rng_mode"] = (
            args.condition_video_vae_rng_mode
        )
    if args.release_text_encoder_after_prompt_encode is not None:
        pipeline_config_overrides["release_text_encoder_after_prompt_encode"] = (
            args.release_text_encoder_after_prompt_encode
        )
    if args.keep_transformer_gpu_resident_between_requests is not None:
        pipeline_config_overrides["keep_transformer_gpu_resident_between_requests"] = (
            args.keep_transformer_gpu_resident_between_requests
        )

    kwargs: dict[str, Any] = {
        "model_path": args.model_path,
        "pipeline_class_name": args.pipeline_class_name,
        "backend": args.backend,
        "output_path": str(output_dir),
        "num_gpus": args.num_gpus,
        "tp_size": args.tp_size,
        "sp_degree": args.sp_degree,
        "ulysses_degree": args.ulysses_degree,
        "ring_degree": args.ring_degree,
        "enable_cfg_parallel": args.enable_cfg_parallel,
        "component_paths": parse_component_overrides(args.component_path),
        "disable_autocast": args.disable_autocast,
        "enable_torch_compile": args.enable_torch_compile,
        "dit_cpu_offload": args.dit_cpu_offload,
        "text_encoder_cpu_offload": args.text_encoder_cpu_offload,
        "vae_cpu_offload": args.vae_cpu_offload,
        "transformer_weights_path": args.transformer_weights_path,
    }
    if pipeline_config_overrides:
        kwargs["pipeline_config"] = pipeline_config_overrides
    if (
        args.enable_svdquant
        or args.quantization_precision is not None
        or args.quantization_rank is not None
        or args.quantization_act_unsigned
    ):
        kwargs["nunchaku_config"] = NunchakuSVDQuantArgs.from_dict(
            {
                "enable_svdquant": args.enable_svdquant,
                "transformer_weights_path": args.transformer_weights_path,
                "quantization_precision": args.quantization_precision,
                "quantization_rank": args.quantization_rank,
                "quantization_act_unsigned": args.quantization_act_unsigned,
            }
        )
    if args.attention_backend:
        kwargs["attention_backend"] = args.attention_backend
    return kwargs


def _build_capability_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "attention_backend": {
            "runtime_request_backend": args.attention_backend or "auto",
            "star_supported_backends": [
                "torch_sdpa",
                "fa",
                "aiter",
                "sage_attn",
                "sage_attn_3",
            ],
            "flashinfer_rope": (
                args.use_flashinfer_rope
                if args.use_flashinfer_rope is not None
                else "integrated_default_off"
            ),
        },
        "parallel": {
            "cfg_parallel": "supported" if args.enable_cfg_parallel else "available_but_disabled",
            "tp": "enabled" if args.tp_size not in (None, 1) else "integrated",
            "sp_ulysses_ring": (
                "enabled"
                if (args.sp_degree not in (None, 1) or args.ulysses_degree not in (None, 1) or args.ring_degree not in (None, 1))
                else "integrated"
            ),
        },
        "cache": {
            "teacache": "enabled" if args.enable_teacache else "integrated",
            "cache_dit": "enabled" if args.enable_cache_dit else "integrated",
        },
        "local_enhancer": {
            "mode": args.local_enhancer_mode or "legacy",
        },
        "resident_strategy": {
            "condition_video_vae_peak_memory_mode": (
                args.condition_video_vae_peak_memory_mode or "legacy"
            ),
            "condition_video_vae_target_headroom_gb": (
                args.condition_video_vae_target_headroom_gb
            ),
            "release_text_encoder_after_prompt_encode": (
                args.release_text_encoder_after_prompt_encode
            ),
            "keep_transformer_gpu_resident_between_requests": (
                args.keep_transformer_gpu_resident_between_requests
            ),
        },
        "quantization": {
            "status": (
                "enabled"
                if (args.transformer_weights_path or args.enable_svdquant)
                else "runtime_integrated_requires_quantized_weights"
            ),
            "transformer_weights_path": args.transformer_weights_path,
            "enable_svdquant": args.enable_svdquant,
            "quantization_precision": args.quantization_precision,
            "quantization_rank": args.quantization_rank,
            "quantization_act_unsigned": args.quantization_act_unsigned,
        },
    }


def _build_sampling_kwargs(
    args: argparse.Namespace,
    prompt: str,
    run_output_dir: Path,
    output_file_name: str,
    resolved_fps: int,
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
        "fps": resolved_fps,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "condition_video_num_frames": args.condition_video_num_frames,
        "output_path": str(run_output_dir),
        "output_file_name": output_file_name,
        "output_quality": args.output_quality,
        "output_compression": args.output_compression,
        "save_output": save_output,
        "return_file_paths_only": False,
        "return_trajectory_latents": args.save_trace,
    }
    if args.enable_batched_cfg is not None:
        kwargs["enable_batched_cfg"] = args.enable_batched_cfg
    teacache_params = _build_teacache_params(args)
    if teacache_params is not None:
        kwargs["enable_teacache"] = True
        kwargs["teacache_params"] = teacache_params
    return kwargs


def main() -> int:
    args = _parse_args()
    if args.enable_stage_logging:
        os.environ["SGLANG_DIFFUSION_STAGE_LOGGING"] = "1"
    if args.sync_stage_profiling:
        os.environ["SGLANG_DIFFUSION_SYNC_STAGE_PROFILING"] = "1"
    _configure_cache_dit_env(args)

    prompt = _load_prompt(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    resolved_fps, fps_source = _resolve_request_fps(
        args.fps,
        reference_video=args.reference_video,
        condition_video_path=args.condition_video_path,
    )

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
                resolved_fps,
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

    saved_frame_paths = None
    if args.save_frame_pngs:
        saved_frame_paths = _save_frame_pngs(last_frames, output_dir)

    raw_baseline_report = None
    raw_strict_report = None
    if args.reference_frame_dir and saved_frame_paths:
        candidate_frame_dir = output_dir / "frames"
        raw_baseline_report = _run_compare(
            args.reference_frame_dir,
            str(candidate_frame_dir),
            "baseline",
            output_dir / "compare_raw_baseline.json",
        )
        raw_strict_report = _run_compare(
            args.reference_frame_dir,
            str(candidate_frame_dir),
            "strict",
            output_dir / "compare_raw_strict.json",
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
            "fps": resolved_fps,
            "fps_source": fps_source,
            "requested_fps": args.fps,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "condition_video_num_frames": args.condition_video_num_frames,
            "enable_batched_cfg": args.enable_batched_cfg,
            "output_quality": args.output_quality,
            "output_compression": args.output_compression,
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
            "pipeline_config_overrides": server_kwargs.get("pipeline_config"),
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
            "raw_baseline": raw_baseline_report,
            "raw_strict": raw_strict_report,
        },
    }

    if saved_frame_paths is not None:
        summary["saved_frame_paths"] = saved_frame_paths
    if args.save_trace:
        summary["trace_artifacts"] = _write_trace_artifacts(
            output_dir,
            request=summary["request"],
            condition_video=summary["condition_video"],
            reference_video=summary["reference_video"],
            metrics=last_result.metrics or {},
            trajectory_latents=last_result.trajectory_latents,
            trajectory_timesteps=last_result.trajectory_timesteps,
            frame_summary=measured_records[-1]["frame_summary"],
            output_file_path=last_result.output_file_path,
        )

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
