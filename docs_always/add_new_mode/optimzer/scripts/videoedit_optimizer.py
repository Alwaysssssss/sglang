#!/usr/bin/env python3
"""Run and inspect the VideoEdit 50-step optimizer stages.

The stage definitions mirror docs_always/add_new_mode/optimzer/optimizer.md.
Use --dry-run first; the real CLI/serve commands can take a long time.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]

DEFAULTS = {
    "MODEL_PATH": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model",
    "TRANSFORMER_PATH": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/step-55000-diffusers-lh/transformer",
    "INPUT_VIDEO": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4",
    "INPUT_MASK": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4",
    "OUT_DIR": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs",
    "PROMPT": "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video.",
    "VIDEO_BASENAME": "15108907_3840_2160_50fps",
    "SERVER_URL": "http://127.0.0.1:30000",
    "QUANT_TRANSFORMER_PATH": "/path/to/quantized/videoedit/transformer",
}

OFFLOAD_FIELDS = (
    "dit_cpu",
    "dit_layerwise",
    "text_encoder",
    "image_encoder",
    "vae",
)

CLI_OFFLOAD_FLAGS = {
    "dit_cpu": "--dit-cpu-offload",
    "dit_layerwise": "--dit-layerwise-offload",
    "text_encoder": "--text-encoder-cpu-offload",
    "image_encoder": "--image-encoder-cpu-offload",
    "vae": "--vae-cpu-offload",
}

SERVE_OFFLOAD_FLAGS = dict(CLI_OFFLOAD_FLAGS)

BACKENDS = ("torch_sdpa", "fa", "sage_attn", "sage_attn_3")


@dataclass(frozen=True)
class StageSpec:
    name: str
    description: str
    num_gpus: int = 1
    sp_degree: int = 1
    ulysses_degree: int = 1
    ring_degree: int = 1
    tp_size: int | None = None
    cfg_parallel: bool = False
    offload: dict[str, bool] = field(default_factory=dict)
    attention_backend: str | None = None
    torch_compile: bool = False
    teacache: bool = False
    teacache_thresh: str = "0.3"
    teacache_start_skipping: str = "5"
    teacache_end_skipping: str = "1.0"
    cache_dit_env: dict[str, str] = field(default_factory=dict)
    transformer_quantization: str | None = None
    transformer_path_env: str | None = None
    dit_offload_prefetch_size: str | None = None

    @property
    def expected_backend(self) -> str:
        return self.attention_backend or "default"


def _offload(**values: bool) -> dict[str, bool]:
    return {field_name: values.get(field_name, False) for field_name in OFFLOAD_FIELDS}


def _cache_env(fn: int, bn: int, warmup: int, rdt: str, mc: int, preset: str) -> dict[str, str]:
    return {
        "SGLANG_CACHE_DIT_FN": str(fn),
        "SGLANG_CACHE_DIT_BN": str(bn),
        "SGLANG_CACHE_DIT_WARMUP": str(warmup),
        "SGLANG_CACHE_DIT_RDT": rdt,
        "SGLANG_CACHE_DIT_MC": str(mc),
        "SGLANG_CACHE_DIT_SCM_PRESET": preset,
    }


def _base_stages() -> dict[str, StageSpec]:
    stages: dict[str, StageSpec] = {}
    all_offload = _offload(
        dit_cpu=True,
        dit_layerwise=True,
        text_encoder=True,
        image_encoder=True,
        vae=True,
    )
    no_offload = _offload()

    stages["sp1_offload"] = StageSpec(
        name="sp1_offload",
        description="Single-GPU offload baseline.",
        offload=all_offload,
    )
    stages["sp1_no_offload"] = StageSpec(
        name="sp1_no_offload",
        description="Single-GPU no-offload baseline.",
        offload=no_offload,
    )
    stages["sp1_no_offload_compile"] = StageSpec(
        name="sp1_no_offload_compile",
        description="Single-GPU no-offload with torch.compile.",
        offload=no_offload,
        torch_compile=True,
    )

    for backend in BACKENDS:
        stages[f"sp1_no_offload_compile_{backend}"] = StageSpec(
            name=f"sp1_no_offload_compile_{backend}",
            description=f"Single-GPU compile with {backend} attention.",
            offload=no_offload,
            torch_compile=True,
            attention_backend=backend,
        )
        stages[f"sp2_no_offload_{backend}"] = StageSpec(
            name=f"sp2_no_offload_{backend}",
            description=f"SP2 Ulysses no-offload with {backend} attention.",
            num_gpus=2,
            sp_degree=2,
            ulysses_degree=2,
            ring_degree=1,
            offload=no_offload,
            attention_backend=backend,
        )

    stages["sp2_ring_no_offload_fa"] = StageSpec(
        name="sp2_ring_no_offload_fa",
        description="SP2 Ring no-offload with FlashAttention.",
        num_gpus=2,
        sp_degree=2,
        ulysses_degree=1,
        ring_degree=2,
        offload=no_offload,
        attention_backend="fa",
    )
    stages["tp2_no_offload_fa"] = StageSpec(
        name="tp2_no_offload_fa",
        description="TP2 no-offload with FlashAttention.",
        num_gpus=2,
        sp_degree=1,
        ulysses_degree=1,
        ring_degree=1,
        tp_size=2,
        offload=no_offload,
        attention_backend="fa",
    )
    stages["sp2_no_offload_compile_fa"] = StageSpec(
        name="sp2_no_offload_compile_fa",
        description="SP2 Ulysses no-offload with compile and FlashAttention.",
        num_gpus=2,
        sp_degree=2,
        ulysses_degree=2,
        ring_degree=1,
        offload=no_offload,
        attention_backend="fa",
        torch_compile=True,
    )
    stages["sp2_no_offload_compile_fa_teacache"] = StageSpec(
        name="sp2_no_offload_compile_fa_teacache",
        description="SP2 compile FlashAttention with request-level TeaCache.",
        num_gpus=2,
        sp_degree=2,
        ulysses_degree=2,
        ring_degree=1,
        offload=no_offload,
        attention_backend="fa",
        torch_compile=True,
        teacache=True,
    )

    stages["cfg2_offload"] = StageSpec(
        name="cfg2_offload",
        description="CFG2 offload baseline.",
        num_gpus=2,
        sp_degree=1,
        ulysses_degree=1,
        ring_degree=1,
        cfg_parallel=True,
        offload=all_offload,
    )
    stages["cfg2_offload_fa"] = StageSpec(
        name="cfg2_offload_fa",
        description="CFG2 offload with FlashAttention.",
        num_gpus=2,
        sp_degree=1,
        ulysses_degree=1,
        ring_degree=1,
        cfg_parallel=True,
        offload=all_offload,
        attention_backend="fa",
    )

    for suffix, cache_env in {
        "rdt010": _cache_env(1, 1, 4, "0.10", 2, "medium"),
        "rdt012": _cache_env(1, 1, 4, "0.12", 2, "medium"),
        "rdt018": _cache_env(1, 1, 4, "0.18", 2, "medium"),
        "fast": _cache_env(1, 0, 2, "0.24", 3, "fast"),
    }.items():
        name = f"sp2_no_offload_compile_fa_cache_{suffix}"
        stages[name] = StageSpec(
            name=name,
            description=f"SP2 compile FlashAttention with Cache-DiT {suffix}.",
            num_gpus=2,
            sp_degree=2,
            ulysses_degree=2,
            ring_degree=1,
            offload=no_offload,
            attention_backend="fa",
            torch_compile=True,
            cache_dit_env=cache_env,
        )

    stages["quant_branch_fp8_dynamic"] = StageSpec(
        name="quant_branch_fp8_dynamic",
        description="Quantized transformer branch using fp8_dynamic.",
        num_gpus=2,
        sp_degree=2,
        ulysses_degree=2,
        ring_degree=1,
        offload=no_offload,
        attention_backend="fa",
        transformer_quantization="fp8_dynamic",
        transformer_path_env="QUANT_TRANSFORMER_PATH",
    )
    stages["offload_branch"] = StageSpec(
        name="offload_branch",
        description="Memory-constrained branch with text/VAE/layerwise offload.",
        offload=_offload(text_encoder=True, vae=True, dit_layerwise=True),
        dit_offload_prefetch_size="1",
    )
    return stages


STAGES = _base_stages()


def default_env() -> dict[str, str]:
    return {key: os.environ.get(key, value) for key, value in DEFAULTS.items()}


def get_stage(name: str) -> StageSpec:
    try:
        return STAGES[name]
    except KeyError as exc:
        known = ", ".join(sorted(STAGES))
        raise SystemExit(f"unknown stage {name!r}; known stages: {known}") from exc


def _stage_runtime_env(spec: StageSpec) -> dict[str, str]:
    env = {"SGLANG_CACHE_DIT_ENABLED": "true" if spec.cache_dit_env else "false"}
    if spec.torch_compile or spec.cache_dit_env:
        env["SGLANG_TORCH_COMPILE_MODE"] = "max-autotune-no-cudagraphs"
    env.update(spec.cache_dit_env)
    return env


def _transformer_path(spec: StageSpec, env: dict[str, str]) -> str:
    if spec.transformer_path_env:
        return env.get(spec.transformer_path_env, DEFAULTS[spec.transformer_path_env])
    return env["TRANSFORMER_PATH"]


def _add_parallel_args(argv: list[str], spec: StageSpec) -> None:
    argv.extend(["--num-gpus", str(spec.num_gpus)])
    if spec.tp_size is not None:
        argv.extend(["--tp-size", str(spec.tp_size)])
    if spec.cfg_parallel:
        argv.append("--enable-cfg-parallel")
    argv.extend(
        [
            "--sp-degree",
            str(spec.sp_degree),
            "--ulysses-degree",
            str(spec.ulysses_degree),
            "--ring-degree",
            str(spec.ring_degree),
        ]
    )


def _add_cli_offload_args(argv: list[str], spec: StageSpec) -> None:
    for field_name in OFFLOAD_FIELDS:
        flag = CLI_OFFLOAD_FLAGS[field_name]
        argv.append(flag if spec.offload.get(field_name, False) else f"--no-{flag[2:]}")
    if spec.dit_offload_prefetch_size is not None:
        argv.extend(["--dit-offload-prefetch-size", spec.dit_offload_prefetch_size])


def _add_serve_offload_args(argv: list[str], spec: StageSpec) -> None:
    for field_name in OFFLOAD_FIELDS:
        argv.extend(
            [
                SERVE_OFFLOAD_FLAGS[field_name],
                "true" if spec.offload.get(field_name, False) else "false",
            ]
        )
    if spec.dit_offload_prefetch_size is not None:
        argv.extend(["--dit-offload-prefetch-size", spec.dit_offload_prefetch_size])


def _add_runtime_args(argv: list[str], spec: StageSpec) -> None:
    if spec.torch_compile:
        argv.append("--enable-torch-compile")
    if spec.attention_backend:
        argv.extend(["--attention-backend", spec.attention_backend])
    if spec.transformer_quantization:
        argv.extend(["--transformer-quantization", spec.transformer_quantization])


def build_cli_command(stage: str, env: dict[str, str] | None = None) -> tuple[dict[str, str], list[str]]:
    env = env or default_env()
    spec = get_stage(stage)
    out_dir = env["OUT_DIR"]
    base = env["VIDEO_BASENAME"]
    argv = [
        "python",
        "-m",
        "sglang.multimodal_gen.runtime.videoedit.cli",
        "repair",
        "--model-path",
        env["MODEL_PATH"],
        "--transformer-path",
        _transformer_path(spec, env),
        "--prompt",
        env["PROMPT"],
        "--video-input-path",
        env["INPUT_VIDEO"],
        "--mask-input-path",
        env["INPUT_MASK"],
        "--output-path",
        out_dir,
        "--output-file-name",
        f"{base}_{stage}.mp4",
        "--num-frames",
        "81",
        "--infer-len",
        "81",
        "--overlap",
        "0",
        "--num-inference-steps",
        "50",
        "--guidance-scale",
        "5.0",
        "--dynamic-cfg",
        "--dynamic-cfg-max-step",
        "15",
        "--seed",
        "42",
        "--dtype",
        "bf16",
        "--enable-paste-back",
        "--drop-reference-frame",
    ]
    if spec.teacache:
        argv.extend(
            [
                "--enable-teacache",
                "--teacache-thresh",
                spec.teacache_thresh,
                "--teacache-start-skipping",
                spec.teacache_start_skipping,
                "--teacache-end-skipping",
                spec.teacache_end_skipping,
            ]
        )
    else:
        argv.append("--no-enable-teacache")
    argv.extend(
        [
            "--warmup",
            "--warmup-steps",
            "1",
            "--perf-dump-path",
            f"{out_dir}/videoedit_perf_{stage}.json",
        ]
    )
    _add_parallel_args(argv, spec)
    _add_cli_offload_args(argv, spec)
    _add_runtime_args(argv, spec)
    return _stage_runtime_env(spec), argv


def build_serve_command(stage: str, env: dict[str, str] | None = None) -> tuple[dict[str, str], list[str]]:
    env = env or default_env()
    spec = get_stage(stage)
    argv = [
        "sglang",
        "serve",
        "--model-type",
        "diffusion",
        "--model-path",
        env["MODEL_PATH"],
        "--host",
        "0.0.0.0",
        "--port",
        "30000",
        "--warmup",
        "true",
        "--warmup-steps",
        "1",
        "--output-path",
        env["OUT_DIR"],
        "--input-save-path",
        "/tmp/sglang-videoedit-inputs",
        "--transformer-path",
        _transformer_path(spec, env),
    ]
    _add_parallel_args(argv, spec)
    _add_serve_offload_args(argv, spec)
    if spec.torch_compile:
        argv.extend(["--enable-torch-compile", "true"])
    if spec.attention_backend:
        argv.extend(["--attention-backend", spec.attention_backend])
    if spec.transformer_quantization:
        argv.extend(["--transformer-quantization", spec.transformer_quantization])
    serve_env = {"VIDEOEDIT_QUEUE_CAPACITY": "1"}
    serve_env.update(_stage_runtime_env(spec))
    return serve_env, argv


def _to_json_number(value: str) -> int | float:
    return float(value) if any(marker in value for marker in (".", "e", "E")) else int(value)


def build_submit_payload(
    stage: str,
    env: dict[str, str] | None = None,
    *,
    task_id: str | None = None,
    api_output: bool = True,
) -> dict[str, object]:
    env = env or default_env()
    spec = get_stage(stage)
    out_dir = env["OUT_DIR"]
    base = env["VIDEO_BASENAME"]
    output_prefix = f"{base}_api_{stage}" if api_output else f"{base}_{stage}"
    return {
        "task_id": task_id or stage,
        "prompt": env["PROMPT"],
        "video_input_path": env["INPUT_VIDEO"],
        "mask_input_path": env["INPUT_MASK"],
        "output_storage": "local",
        "output_path": f"{out_dir}/{output_prefix}.mp4",
        "num_frames": 81,
        "infer_len": 81,
        "overlap": 0,
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "dynamic_cfg": True,
        "dynamic_cfg_max_step": 15,
        "seed": 42,
        "dtype": "bf16",
        "enable_paste_back": True,
        "drop_reference_frame": True,
        "enable_teacache": spec.teacache,
        "teacache_thresh": float(spec.teacache_thresh),
        "teacache_start_skipping": _to_json_number(spec.teacache_start_skipping),
        "teacache_end_skipping": _to_json_number(spec.teacache_end_skipping),
        "perf_dump_path": f"{out_dir}/videoedit_perf_api_{stage}.json",
    }


def build_compare_command(
    stage: str,
    env: dict[str, str] | None = None,
    *,
    candidate: str | None = None,
    api: bool = False,
    drop_candidate_first_frame: bool = False,
) -> tuple[dict[str, str], list[str]]:
    env = env or default_env()
    base = env["VIDEO_BASENAME"]
    prefix = f"{base}_api_{stage}" if api else f"{base}_{stage}"
    candidate = candidate or f"{env['OUT_DIR']}/{prefix}.mp4"
    report_stage = f"api_{stage}" if api else stage
    argv = [
        "python",
        "python/sglang/multimodal_gen/runtime/videoedit/compare.py",
        "--reference",
        f"{env['OUT_DIR']}/reference/15108907_3840_2160_50fps.mp4",
        "--candidate",
        candidate,
        "--report-json",
        f"{env['OUT_DIR']}/videoedit_compare_{report_stage}.json",
        "--min-ssim",
        "0.90",
        "--max-mse",
        "150.0",
        "--max-mae",
        "8.0",
        "--allow-frame-count-delta",
        "1",
        "--max-failed-frame-ratio",
        "0.05",
    ]
    if drop_candidate_first_frame:
        argv.append("--drop-candidate-first-frame")
    return {}, argv


def render_shell_command(argv: Iterable[str], env: dict[str, str] | None = None) -> str:
    env = env or {}
    env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    command = " ".join(shlex.quote(str(arg)) for arg in argv)
    return f"{env_prefix} {command}".strip()


def _stream_process(argv: list[str], extra_env: dict[str, str], log_path: str | None) -> int:
    run_env = os.environ.copy()
    run_env.update(extra_env)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        argv,
        cwd=str(REPO_ROOT),
        env=run_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    log_file = open(log_path, "a", encoding="utf-8") if log_path else None
    try:
        for line in process.stdout:
            sys.stdout.write(line)
            if log_file:
                log_file.write(line)
    finally:
        if log_file:
            log_file.close()
    return process.wait()


def run_cli_stage(stage: str, dry_run: bool) -> int:
    env = default_env()
    extra_env, argv = build_cli_command(stage, env)
    log_path = f"{env['OUT_DIR']}/videoedit_bench_{stage}.log"
    if dry_run:
        print(render_shell_command(argv, extra_env) + f" 2>&1 | tee {shlex.quote(log_path)}")
        return 0
    Path(env["OUT_DIR"]).mkdir(parents=True, exist_ok=True)
    return _stream_process(argv, extra_env, log_path)


def run_serve_stage(stage: str, dry_run: bool) -> int:
    env = default_env()
    extra_env, argv = build_serve_command(stage, env)
    log_path = f"{env['OUT_DIR']}/videoedit_serve_{stage}.log"
    if dry_run:
        print(render_shell_command(argv, extra_env) + f" 2>&1 | tee {shlex.quote(log_path)}")
        return 0
    Path(env["OUT_DIR"]).mkdir(parents=True, exist_ok=True)
    return _stream_process(argv, extra_env, log_path)


def submit_stage(stage: str, dry_run: bool, task_id: str | None) -> int:
    env = default_env()
    payload = build_submit_payload(stage, env, task_id=task_id)
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    if dry_run:
        print(data)
        return 0
    req = urllib.request.Request(
        f"{env['SERVER_URL'].rstrip('/')}/v1/videos/repairs",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        print(response.read().decode())
    return 0


def poll_job(job_id: str, interval: float) -> int:
    env = default_env()
    url = f"{env['SERVER_URL'].rstrip('/')}/v1/videos/{job_id}"
    while True:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode())
        print(
            data.get("status"),
            data.get("progress"),
            data.get("file_path") or data.get("url"),
            data.get("inference_time_s"),
        )
        if data.get("status") == "completed":
            return 0
        if data.get("status") == "failed":
            print(json.dumps(data.get("error"), ensure_ascii=False), file=sys.stderr)
            return 1
        time.sleep(interval)


def run_compare(stage: str, dry_run: bool, api: bool, candidate: str | None, drop_first: bool) -> int:
    extra_env, argv = build_compare_command(
        stage,
        default_env(),
        candidate=candidate,
        api=api,
        drop_candidate_first_frame=drop_first,
    )
    if dry_run:
        print(render_shell_command(argv, extra_env))
        return 0
    return subprocess.call(argv, cwd=str(REPO_ROOT), env=os.environ.copy())


def check_reference() -> int:
    env = default_env()
    path = Path(env["OUT_DIR"]) / "reference" / "15108907_3840_2160_50fps.mp4"
    if not path.exists():
        print(f"missing reference: {path}", file=sys.stderr)
        return 1
    try:
        import cv2  # type: ignore
    except Exception as exc:
        print(f"cv2 is required to inspect reference metadata: {exc}", file=sys.stderr)
        return 1
    cap = cv2.VideoCapture(str(path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(json.dumps({"path": str(path), "frames": frames, "width": width, "height": height, "fps": fps}, indent=2))
    if frames not in (80, 81) or width <= 0 or height <= 0:
        return 1
    return 0


def parse_backend(stage: str, api: bool) -> int:
    env = default_env()
    log_name = f"videoedit_serve_{stage}.log" if api else f"videoedit_bench_{stage}.log"
    path = Path(env["OUT_DIR"]) / log_name
    if not path.exists():
        print(f"missing log: {path}", file=sys.stderr)
        return 1
    pattern = re.compile(
        r"(attention backend|Using .*Attention|Sage|fallback|No module named|Selected attention)",
        re.IGNORECASE,
    )
    hits = [line.rstrip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if pattern.search(line)]
    print("\n".join(hits))
    return 0 if hits else 1


def describe_stage(stage: str) -> int:
    spec = get_stage(stage)
    print(
        json.dumps(
            {
                "name": spec.name,
                "description": spec.description,
                "expected_backend": spec.expected_backend,
                "num_gpus": spec.num_gpus,
                "sp_degree": spec.sp_degree,
                "ulysses_degree": spec.ulysses_degree,
                "ring_degree": spec.ring_degree,
                "tp_size": spec.tp_size,
                "cfg_parallel": spec.cfg_parallel,
                "offload": spec.offload,
                "torch_compile": spec.torch_compile,
                "teacache": spec.teacache,
                "cache_dit_env": spec.cache_dit_env,
                "transformer_quantization": spec.transformer_quantization,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-stages", help="List registered optimizer stages.")

    describe = sub.add_parser("describe", help="Print a stage spec.")
    describe.add_argument("stage")

    cli = sub.add_parser("cli", help="Run or print the local CLI command for a stage.")
    cli.add_argument("stage")
    cli.add_argument("--dry-run", action="store_true")

    serve = sub.add_parser("serve", help="Run or print the serve command for a stage.")
    serve.add_argument("stage")
    serve.add_argument("--dry-run", action="store_true")

    submit = sub.add_parser("submit", help="Submit a serve request for a stage.")
    submit.add_argument("stage")
    submit.add_argument("--task-id")
    submit.add_argument("--dry-run", action="store_true")

    poll = sub.add_parser("poll", help="Poll a submitted job.")
    poll.add_argument("job_id")
    poll.add_argument("--interval", type=float, default=5.0)

    compare = sub.add_parser("compare", help="Run or print compare.py for a stage.")
    compare.add_argument("stage")
    compare.add_argument("--api", action="store_true", help="Compare the api_<stage> output.")
    compare.add_argument("--candidate")
    compare.add_argument("--drop-candidate-first-frame", action="store_true")
    compare.add_argument("--dry-run", action="store_true")

    sub.add_parser("check-reference", help="Check that the fixed reference video exists.")

    parse = sub.add_parser("parse-backend", help="Extract backend/fallback lines from a log.")
    parse.add_argument("stage")
    parse.add_argument("--api", action="store_true", help="Read serve log instead of CLI log.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list-stages":
        for name in sorted(STAGES):
            spec = STAGES[name]
            print(f"{name}\t{spec.description}")
        return 0
    if args.command == "describe":
        return describe_stage(args.stage)
    if args.command == "cli":
        return run_cli_stage(args.stage, args.dry_run)
    if args.command == "serve":
        return run_serve_stage(args.stage, args.dry_run)
    if args.command == "submit":
        return submit_stage(args.stage, args.dry_run, args.task_id)
    if args.command == "poll":
        return poll_job(args.job_id, args.interval)
    if args.command == "compare":
        return run_compare(
            args.stage,
            args.dry_run,
            args.api,
            args.candidate,
            args.drop_candidate_first_frame,
        )
    if args.command == "check-reference":
        return check_reference()
    if args.command == "parse-backend":
        return parse_backend(args.stage, args.api)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
