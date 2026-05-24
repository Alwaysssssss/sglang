from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


DEFAULT_MATRIX = (
    {
        "name": "baseline_offload_sdpa",
        "dit_cpu_offload": True,
        "text_encoder_cpu_offload": True,
        "vae_cpu_offload": False,
        "enable_torch_compile": False,
        "enable_batched_cfg": False,
        "enable_cfg_parallel": False,
    },
    {
        "name": "offload_fa",
        "dit_cpu_offload": True,
        "text_encoder_cpu_offload": True,
        "vae_cpu_offload": False,
        "enable_torch_compile": False,
        "enable_batched_cfg": False,
        "enable_cfg_parallel": False,
        "attention_backend": "fa",
    },
    {
        "name": "offload_fa_batched_cfg",
        "dit_cpu_offload": True,
        "text_encoder_cpu_offload": True,
        "vae_cpu_offload": False,
        "enable_torch_compile": False,
        "enable_batched_cfg": True,
        "enable_cfg_parallel": False,
        "attention_backend": "fa",
    },
    {
        "name": "offload_fa_compile_batched_cfg",
        "dit_cpu_offload": True,
        "text_encoder_cpu_offload": True,
        "vae_cpu_offload": False,
        "enable_torch_compile": True,
        "enable_batched_cfg": True,
        "enable_cfg_parallel": False,
        "attention_backend": "fa",
    },
    {
        "name": "offload_fa_teacache_batched_cfg",
        "dit_cpu_offload": True,
        "text_encoder_cpu_offload": True,
        "vae_cpu_offload": False,
        "enable_torch_compile": False,
        "enable_batched_cfg": True,
        "enable_cfg_parallel": False,
        "attention_backend": "fa",
        "enable_teacache": True,
        "teacache_thresh": 0.15,
        "teacache_start_skipping": 5.0,
        "teacache_end_skipping": -1.0,
        "teacache_coefficients": "1,0",
    },
    {
        "name": "offload_fa_cache_dit_batched_cfg",
        "dit_cpu_offload": True,
        "text_encoder_cpu_offload": True,
        "vae_cpu_offload": False,
        "enable_torch_compile": False,
        "enable_batched_cfg": True,
        "enable_cfg_parallel": False,
        "attention_backend": "fa",
        "enable_cache_dit": True,
        "cache_dit_fn": 1,
        "cache_dit_bn": 0,
        "cache_dit_warmup": 4,
        "cache_dit_rdt": 0.24,
        "cache_dit_mc": 3,
        "cache_dit_scm_preset": "none",
    },
    {
        "name": "offload_fa_teacache_cache_dit_compile_batched_cfg",
        "dit_cpu_offload": True,
        "text_encoder_cpu_offload": True,
        "vae_cpu_offload": False,
        "enable_torch_compile": True,
        "enable_batched_cfg": True,
        "enable_cfg_parallel": False,
        "attention_backend": "fa",
        "enable_teacache": True,
        "teacache_thresh": 0.15,
        "teacache_start_skipping": 5.0,
        "teacache_end_skipping": -1.0,
        "teacache_coefficients": "1,0",
        "enable_cache_dit": True,
        "cache_dit_fn": 1,
        "cache_dit_bn": 0,
        "cache_dit_warmup": 4,
        "cache_dit_rdt": 0.24,
        "cache_dit_mc": 3,
        "cache_dit_scm_preset": "none",
    },
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a fixed STAR CogVideoX-SR config matrix and collect parity/speedup summaries."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--condition-video-path", required=True)
    parser.add_argument("--prompt-path", required=True)
    parser.add_argument("--reference-video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=7)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--condition-video-num-frames", type=int, default=25)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--measured-runs", type=int, default=1)
    parser.add_argument("--original-star-cold-e2e-s", type=float, default=None)
    parser.add_argument("--original-star-warm-e2e-s", type=float, default=None)
    parser.add_argument("--original-star-denoise-s", type=float, default=None)
    parser.add_argument("--pipeline-class-name", default="StarCogVideoXSRPipeline")
    return parser.parse_args()


def _run_profile(args: argparse.Namespace, config: dict, run_dir: Path) -> dict:
    command = [
        "python",
        "-m",
        "sglang.multimodal_gen.test.manual.profile_star_cogvideox_sr",
        "--model-path",
        args.model_path,
        "--condition-video-path",
        args.condition_video_path,
        "--prompt-path",
        args.prompt_path,
        "--reference-video",
        args.reference_video,
        "--output-dir",
        str(run_dir),
        "--seed",
        str(args.seed),
        "--num-frames",
        str(args.num_frames),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--fps",
        str(args.fps),
        "--num-inference-steps",
        str(args.num_inference_steps),
        "--guidance-scale",
        str(args.guidance_scale),
        "--condition-video-num-frames",
        str(args.condition_video_num_frames),
        "--warmup-runs",
        str(args.warmup_runs),
        "--measured-runs",
        str(args.measured_runs),
        "--enable-stage-logging",
        "--sync-stage-profiling",
        "--pipeline-class-name",
        args.pipeline_class_name,
        "--num-gpus",
        str(config.get("num_gpus", args.num_gpus)),
    ]
    attention_backend = config.get("attention_backend")
    if attention_backend:
        command.extend(["--attention-backend", str(attention_backend)])
    if args.original_star_cold_e2e_s is not None:
        command.extend(
            ["--original-star-cold-e2e-s", str(args.original_star_cold_e2e_s)]
        )
    if args.original_star_warm_e2e_s is not None:
        command.extend(
            ["--original-star-warm-e2e-s", str(args.original_star_warm_e2e_s)]
        )
    if args.original_star_denoise_s is not None:
        command.extend(
            ["--original-star-denoise-s", str(args.original_star_denoise_s)]
        )
    if config["dit_cpu_offload"]:
        command.append("--dit-cpu-offload")
    if config["text_encoder_cpu_offload"]:
        command.append("--text-encoder-cpu-offload")
    if config["vae_cpu_offload"]:
        command.append("--vae-cpu-offload")
    if config["enable_torch_compile"]:
        command.append("--enable-torch-compile")
    if config["enable_cfg_parallel"]:
        command.append("--enable-cfg-parallel")
    if config["enable_batched_cfg"]:
        command.append("--enable-batched-cfg")
    else:
        command.append("--disable-batched-cfg")
    if config.get("enable_teacache"):
        command.append("--enable-teacache")
        command.extend(["--teacache-thresh", str(config.get("teacache_thresh", 0.0))])
        command.extend(
            [
                "--teacache-start-skipping",
                str(config.get("teacache_start_skipping", 5.0)),
            ]
        )
        command.extend(
            [
                "--teacache-end-skipping",
                str(config.get("teacache_end_skipping", -1.0)),
            ]
        )
        command.extend(
            [
                "--teacache-coefficients",
                str(config.get("teacache_coefficients", "1,0")),
            ]
        )
    if config.get("enable_cache_dit"):
        command.append("--enable-cache-dit")
        command.extend(["--cache-dit-fn", str(config.get("cache_dit_fn", 1))])
        command.extend(["--cache-dit-bn", str(config.get("cache_dit_bn", 0))])
        command.extend(
            ["--cache-dit-warmup", str(config.get("cache_dit_warmup", 4))]
        )
        command.extend(["--cache-dit-rdt", str(config.get("cache_dit_rdt", 0.24))])
        command.extend(["--cache-dit-mc", str(config.get("cache_dit_mc", 3))])
        command.extend(
            [
                "--cache-dit-scm-preset",
                str(config.get("cache_dit_scm_preset", "none")),
            ]
        )
    if config.get("transformer_weights_path"):
        command.extend(
            [
                "--transformer-weights-path",
                str(config["transformer_weights_path"]),
            ]
        )
    if config.get("enable_svdquant"):
        command.append("--enable-svdquant")
    if config.get("quantization_precision"):
        command.extend(
            [
                "--quantization-precision",
                str(config["quantization_precision"]),
            ]
        )
    if config.get("quantization_rank") is not None:
        command.extend(
            ["--quantization-rank", str(config["quantization_rank"])]
        )
    if config.get("quantization_act_unsigned"):
        command.append("--quantization-act-unsigned")

    subprocess.run(command, check=True)
    summary_path = run_dir / "summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for config in DEFAULT_MATRIX:
        run_dir = output_dir / config["name"]
        summary = _run_profile(args, config, run_dir)
        parity = summary.get("parity", {}).get("baseline") or {}
        speedup = summary.get("profile", {}).get("speedup", {})
        results.append(
            {
                "name": config["name"],
                "config": config,
                "num_gpus": config.get("num_gpus", args.num_gpus),
                "summary_path": str((run_dir / "summary.json").resolve()),
                "candidate_path": summary.get("profile", {}).get("last_candidate_path"),
                "avg_generation_time_s": summary.get("profile", {}).get(
                    "avg_generation_time_s"
                ),
                "avg_denoise_s": summary.get("profile", {}).get("avg_denoise_s"),
                "warm_e2e_speedup": speedup.get("warm_e2e_speedup"),
                "denoise_speedup": speedup.get("denoise_speedup"),
                "baseline_passed": parity.get("passed"),
                "baseline_ssim_mean": parity.get("summary", {}).get("ssim_mean"),
                "baseline_failed_frames": parity.get("summary", {}).get(
                    "num_failed_frames"
                ),
            }
        )

    best_result = None
    ranked = [r for r in results if r["baseline_passed"]]
    if ranked:
        ranked.sort(
            key=lambda item: (
                item["warm_e2e_speedup"] is None,
                -(item["warm_e2e_speedup"] or 0.0),
            )
        )
        best_result = ranked[0]

    matrix_summary = {
        "comparison_policy": {
            "same_gpu_count_required": True,
            "target_num_gpus": args.num_gpus,
        },
        "results": results,
        "best_result": best_result,
    }
    summary_path = output_dir / "matrix_summary.json"
    summary_path.write_text(
        json.dumps(matrix_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"matrix summary: {summary_path}")
    if best_result is not None:
        print(f"best config: {best_result['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
