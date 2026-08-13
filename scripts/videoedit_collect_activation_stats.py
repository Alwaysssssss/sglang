#!/usr/bin/env python3
"""Collect per-Linear VideoEdit activation statistics from erase_data_case."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from sglang.multimodal_gen.runtime.utils.activation_calibration import (  # noqa: E402
    CALIBRATION_DIR_ENV,
    EXPECTED_LINEAR_COUNT_ENV,
    HISTOGRAM_BINS_ENV,
    HISTOGRAM_LOG2_MAX_ENV,
    HISTOGRAM_LOG2_MIN_ENV,
    REQUEST_PREFIX_ENV,
    merge_rank_calibration,
)

DEFAULT_DATA_ROOT = REPO_ROOT / "erase_data_case"
DEFAULT_MODEL_PATH = Path(
    "/mnt/nas/models/DifusserEdit/pretrain_models/VideoEdit-diffusers-model"
)
DEFAULT_TRANSFORMER_PATH = REPO_ROOT / "videoedit_fp8_offline" / "transformer"
DEFAULT_OUT_ROOT = REPO_ROOT / "videoedit_activation_calibration"
REQUEST_PREFIX = "calib_"
NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@dataclass(frozen=True)
class CalibrationCase:
    case_id: str
    caption: str
    video: Path
    mask: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 60,
) -> dict[str, Any]:
    body = None
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with NO_PROXY_OPENER.open(request, timeout=timeout) as response:
            text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {text}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Non-JSON response from {url}: {text[:1000]}") from exc


def load_cases(args: argparse.Namespace) -> list[CalibrationCase]:
    captions_path = args.captions.resolve()
    cases: list[CalibrationCase] = []
    seen: set[str] = set()
    with captions_path.open("r", encoding="utf-8") as captions_file:
        for line_number, raw_line in enumerate(captions_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {captions_path}:{line_number}: {exc}"
                ) from exc
            case_id = str(record.get("id", "")).strip()
            caption = str(record.get("caption", "")).strip()
            if not case_id or not caption:
                raise ValueError(
                    f"Missing id or caption at {captions_path}:{line_number}"
                )
            if case_id in seen:
                raise ValueError(f"Duplicate case id in captions: {case_id}")
            seen.add(case_id)
            declared_video = Path(str(record.get("video", "")))
            expected_video = args.videos_dir / f"{case_id}.mp4"
            if declared_video.name and declared_video.name != expected_video.name:
                raise ValueError(
                    f"Caption video does not match id {case_id}: {declared_video}"
                )
            cases.append(
                CalibrationCase(
                    case_id=case_id,
                    caption=caption,
                    video=expected_video.resolve(),
                    mask=(args.masks_dir / f"{case_id}_mask.mp4").resolve(),
                )
            )

    if args.sample_id:
        requested = set(args.sample_id)
        unknown = sorted(requested - seen)
        if unknown:
            raise ValueError(f"Unknown --sample-id values: {unknown}")
        cases = [case for case in cases if case.case_id in requested]
    if args.max_samples is not None:
        cases = cases[: args.max_samples]
    if not cases:
        raise ValueError("No calibration cases were selected")
    return cases


def probe_video(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise ValueError(f"No video stream found in {path}")
    stream = streams[0]
    frame_rate = stream.get("avg_frame_rate", "0/1")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": float(Fraction(frame_rate)),
        "frame_rate_raw": frame_rate,
        "frame_count": (
            int(stream["nb_frames"])
            if str(stream.get("nb_frames", "")).isdigit()
            else None
        ),
        "duration": (
            float(stream["duration"])
            if stream.get("duration") not in (None, "N/A")
            else None
        ),
    }


def validate_cases(cases: list[CalibrationCase]) -> list[dict[str, Any]]:
    if shutil.which("ffprobe") is None:
        raise FileNotFoundError("ffprobe is required for media validation")
    records: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        for label, path in (("video", case.video), ("mask", case.mask)):
            if not path.is_file():
                raise FileNotFoundError(f"{label} does not exist: {path}")
            if not os.access(path, os.R_OK):
                raise PermissionError(f"{label} is not readable: {path}")
        video_info = probe_video(case.video)
        mask_info = probe_video(case.mask)
        for key in ("width", "height", "frame_count"):
            if (
                video_info[key] is not None
                and mask_info[key] is not None
                and video_info[key] != mask_info[key]
            ):
                raise ValueError(
                    f"Video/mask {key} mismatch for {case.case_id}: "
                    f"{video_info[key]} != {mask_info[key]}"
                )
        if abs(video_info["frame_rate"] - mask_info["frame_rate"]) > 1e-3:
            raise ValueError(
                f"Video/mask frame rate mismatch for {case.case_id}: "
                f"{video_info['frame_rate_raw']} != {mask_info['frame_rate_raw']}"
            )
        records.append(
            {
                "index": index,
                "id": case.case_id,
                "caption": case.caption,
                "video": str(case.video),
                "mask": str(case.mask),
                "video_info": video_info,
                "mask_info": mask_info,
            }
        )
        print(
            f"[validate] {index:02d}/{len(cases):02d} id={case.case_id} "
            f"shape={video_info['width']}x{video_info['height']} "
            f"frames={video_info['frame_count']}",
            flush=True,
        )
    return records


def dataset_fingerprint(records: list[dict[str, Any]]) -> str:
    stable_records = [
        {
            "id": record["id"],
            "caption": record["caption"],
            "video": record["video"],
            "mask": record["mask"],
            "video_info": record["video_info"],
            "mask_info": record["mask_info"],
        }
        for record in records
    ]
    encoded = json.dumps(
        stable_records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def extract_first_frame(video: Path, output: Path) -> None:
    if output.is_file():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.png")
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(temporary),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to extract first frame from {video}: {result.stderr.strip()}"
        )
    os.replace(temporary, output)


def bool_arg(value: bool) -> str:
    return "true" if value else "false"


def build_service_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    command = [
        args.serve_executable,
        "serve",
        "--model-type",
        "diffusion",
        "--model-path",
        str(args.model_path),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--num-gpus",
        str(args.num_gpus),
        "--sp-degree",
        str(args.sp_degree),
        "--ulysses-degree",
        str(args.ulysses_degree),
        "--ring-degree",
        str(args.ring_degree),
        "--dit-cpu-offload",
        "false",
        "--dit-layerwise-offload",
        bool_arg(args.dit_layerwise_offload),
        "--text-encoder-cpu-offload",
        "true",
        "--image-encoder-cpu-offload",
        "true",
        "--vae-cpu-offload",
        "true",
        "--warmup",
        "true",
        "--warmup-steps",
        "1",
        "--output-path",
        str(run_dir / "server_outputs"),
        "--input-save-path",
        str(run_dir / "server_inputs"),
        "--transformer-path",
        str(args.transformer_path),
        "--transformer-fp8-gemm-backend",
        args.fp8_gemm_backend,
        "--transformer-fp8-fused-projections",
        "true",
        "--videoedit-self-attention-backend",
        args.self_attention_backend,
        "--videoedit-cross-attention-backend",
        args.cross_attention_backend,
    ]
    command.extend(args.server_extra_arg)
    return command


def build_service_env(args: argparse.Namespace, collector_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["SGLANG_DIFFUSION_QUANT_AUDIT"] = "1"
    env["SGLANG_DIFFUSION_LINEAR_RUNTIME_AUDIT"] = "1"
    env[CALIBRATION_DIR_ENV] = str(collector_dir)
    env[REQUEST_PREFIX_ENV] = REQUEST_PREFIX
    env[EXPECTED_LINEAR_COUNT_ENV] = str(args.expected_linear_count)
    env[HISTOGRAM_BINS_ENV] = str(args.histogram_bins)
    env[HISTOGRAM_LOG2_MIN_ENV] = str(args.histogram_log2_min)
    env[HISTOGRAM_LOG2_MAX_ENV] = str(args.histogram_log2_max)
    no_proxy = env.get("NO_PROXY") or env.get("no_proxy") or ""
    entries = [entry for entry in no_proxy.split(",") if entry]
    for entry in ("127.0.0.1", "localhost"):
        if entry not in entries:
            entries.append(entry)
    env["NO_PROXY"] = ",".join(entries)
    env["no_proxy"] = env["NO_PROXY"]
    return env


def port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def endpoint_ready(url: str, timeout: float = 3) -> bool:
    request = urllib.request.Request(url, method="GET")
    try:
        with NO_PROXY_OPENER.open(request, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def tail_text(path: Path, lines: int = 80) -> str:
    if not path.is_file():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace")
    return "".join(content.splitlines(True)[-lines:])


def wait_for_service(
    process: subprocess.Popen[str],
    base_url: str,
    timeout: float,
    log_path: Path,
) -> float:
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"Service exited before readiness with code {returncode}\n"
                f"{tail_text(log_path)}"
            )
        if endpoint_ready(f"{base_url}/health") and endpoint_ready(
            f"{base_url}/v1/models", timeout=5
        ):
            return time.monotonic() - started
        time.sleep(2)
    raise TimeoutError(
        f"Service did not become ready within {timeout:.0f}s\n{tail_text(log_path)}"
    )


def process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def stop_process_group(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    process_group = process.pid
    for sig, timeout in (
        (signal.SIGINT, 120.0),
        (signal.SIGTERM, 30.0),
        (signal.SIGKILL, 10.0),
    ):
        if not process_group_exists(process_group):
            return
        try:
            os.killpg(process_group, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not process_group_exists(process_group):
                return
            time.sleep(0.5)


def build_payload(
    args: argparse.Namespace,
    case: CalibrationCase,
    index: int,
    reference: Path,
    output: Path,
) -> dict[str, Any]:
    return {
        "task_id": f"{REQUEST_PREFIX}{case.case_id}",
        "timeout": -1,
        "prompt": case.caption,
        "video_input_path": str(case.video),
        "mask_input_path": str(case.mask),
        "reference_image_url": str(reference),
        "output_storage": "local",
        "output_path": str(output),
        "num_frames": args.num_frames,
        "infer_len": args.infer_len,
        "overlap": args.overlap,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed + index,
        "dtype": "bf16",
        "dynamic_cfg": True,
        "dynamic_cfg_max_step": 15,
        "dynamic_cfg_min": 1.0,
        "bbox_expand_scale": args.bbox_expand_scale,
        "decode_mode": "stream",
        "enable_teacache": False,
        "drop_reference_frame": True,
        "enable_paste_back": False,
    }


def submit_task(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    url = f"{args.base_url}/v1/videos/repairs"
    started = time.monotonic()
    while True:
        response = request_json("POST", url, payload, timeout=args.http_timeout)
        if response.get("code") == 0:
            return
        if response.get("code") == 2:
            if time.monotonic() - started >= args.busy_retry_timeout:
                raise RuntimeError(f"Service remained busy: {response}")
            print("[busy] another task is active; retrying", flush=True)
            time.sleep(args.busy_retry_interval)
            continue
        raise RuntimeError(f"Task submission failed: {response}")


def wait_for_task(args: argparse.Namespace, task_id: str) -> dict[str, Any]:
    url = f"{args.base_url}/v1/videos/{task_id}/progress"
    while True:
        progress = request_json("GET", url, timeout=args.http_timeout)
        print(
            f"[progress] {task_id} status={progress.get('status')} "
            f"progress={progress.get('progress')} reason={progress.get('reason')}",
            flush=True,
        )
        if progress.get("status") == "completed":
            return progress
        if progress.get("status") == "failed":
            raise RuntimeError(
                f"Task failed: {json.dumps(progress, ensure_ascii=False)}"
            )
        time.sleep(args.poll_interval)


def completed_rank_requests(
    collector_dir: Path,
    expected_rank_count: int,
) -> set[str]:
    expected_rank_names = {f"rank{rank}" for rank in range(expected_rank_count)}
    existing_rank_names = {
        path.name
        for path in collector_dir.glob("rank*")
        if path.is_dir() and (path / "manifest.json").is_file()
    }
    if existing_rank_names and existing_rank_names != expected_rank_names:
        raise ValueError(
            "Calibration rank directories do not match --num-gpus: "
            f"{sorted(existing_rank_names)} != {sorted(expected_rank_names)}"
        )
    request_sets: list[set[str]] = []
    for rank in range(expected_rank_count):
        manifest_path = collector_dir / f"rank{rank}" / "manifest.json"
        if not manifest_path.is_file():
            return set()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        request_sets.append(set(manifest.get("completed_requests", [])))
    if any(request_set != request_sets[0] for request_set in request_sets[1:]):
        raise ValueError("Rank completed request sets do not match")
    return request_sets[0]


def validate_checkpoint(args: argparse.Namespace) -> None:
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")
    if not args.transformer_path.is_dir():
        raise FileNotFoundError(
            f"Transformer path does not exist: {args.transformer_path}"
        )
    config_path = args.transformer_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quantization = config.get("quantization_config") or {}
    expected = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_scale_granularity": "channel",
    }
    mismatches = {
        key: {"expected": value, "actual": quantization.get(key)}
        for key, value in expected.items()
        if quantization.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Activation collection requires the serialized FP8 checkpoint with "
            f"dynamic activation quantization; mismatches: {mismatches}"
        )


def run_configuration(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset_fingerprint": dataset_fingerprint(records),
        "selected_case_ids": [record["id"] for record in records],
        "model_path": str(args.model_path.resolve()),
        "transformer_path": str(args.transformer_path.resolve()),
        "num_frames": args.num_frames,
        "infer_len": args.infer_len,
        "overlap": args.overlap,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "bbox_expand_scale": args.bbox_expand_scale,
        "seed": args.seed,
        "num_gpus": args.num_gpus,
        "sp_degree": args.sp_degree,
        "ulysses_degree": args.ulysses_degree,
        "ring_degree": args.ring_degree,
        "dit_layerwise_offload": args.dit_layerwise_offload,
        "fp8_gemm_backend": args.fp8_gemm_backend,
        "self_attention_backend": args.self_attention_backend,
        "cross_attention_backend": args.cross_attention_backend,
        "expected_linear_count": args.expected_linear_count,
        "histogram_bins": args.histogram_bins,
        "histogram_log2_min": args.histogram_log2_min,
        "histogram_log2_max": args.histogram_log2_max,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--captions", type=Path, default=None)
    parser.add_argument("--videos-dir", type=Path, default=None)
    parser.add_argument("--masks-dir", type=Path, default=None)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--transformer-path", type=Path, default=DEFAULT_TRANSFORMER_PATH
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--resume-run", type=Path, default=None)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-output-videos", action="store_true")
    parser.add_argument("--num-frames", type=int, default=80)
    parser.add_argument("--infer-len", type=int, default=81)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--bbox-expand-scale", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--sp-degree", type=int, default=2)
    parser.add_argument("--ulysses-degree", type=int, default=2)
    parser.add_argument("--ring-degree", type=int, default=1)
    parser.add_argument(
        "--dit-layerwise-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fp8-gemm-backend", default="triton")
    parser.add_argument("--self-attention-backend", default="sage_attn")
    parser.add_argument("--cross-attention-backend", default="fa")
    parser.add_argument("--serve-executable", default="sglang")
    parser.add_argument("--server-extra-arg", action="append", default=[])
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--http-timeout", type=float, default=120)
    parser.add_argument("--poll-interval", type=float, default=15)
    parser.add_argument("--busy-retry-interval", type=float, default=30)
    parser.add_argument("--busy-retry-timeout", type=float, default=3600)
    parser.add_argument("--expected-linear-count", type=int, default=328)
    parser.add_argument("--histogram-bins", type=int, default=2048)
    parser.add_argument("--histogram-log2-min", type=float, default=-24.0)
    parser.add_argument("--histogram-log2-max", type=float, default=16.0)
    args = parser.parse_args()

    args.data_root = args.data_root.expanduser().resolve()
    args.captions = (
        (args.captions or args.data_root / "caption_frames" / "captions.jsonl")
        .expanduser()
        .resolve()
    )
    args.videos_dir = (
        (args.videos_dir or args.data_root / "videos").expanduser().resolve()
    )
    args.masks_dir = (
        (args.masks_dir or args.data_root / "video_masks").expanduser().resolve()
    )
    args.model_path = args.model_path.expanduser().resolve()
    args.transformer_path = args.transformer_path.expanduser().resolve()
    args.out_root = args.out_root.expanduser().resolve()
    if args.resume_run is not None:
        args.resume_run = args.resume_run.expanduser().resolve()
    args.base_url = f"http://{args.host}:{args.port}"

    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.num_inference_steps <= 0:
        parser.error("--num-inference-steps must be positive")
    if args.num_frames == 0 or args.num_frames < -1:
        parser.error("--num-frames must be positive or -1")
    if args.infer_len <= 0:
        parser.error("--infer-len must be positive")
    if args.histogram_bins <= 0:
        parser.error("--histogram-bins must be positive")
    if args.histogram_log2_min >= args.histogram_log2_max:
        parser.error("--histogram-log2-min must be smaller than --histogram-log2-max")
    if args.merge_only and args.resume_run is None:
        parser.error("--merge-only requires --resume-run")
    return args


def main() -> int:
    args = parse_args()
    if args.merge_only:
        collector_dir = args.resume_run / "collector"
        manifest = merge_rank_calibration(collector_dir, output_dir=args.resume_run)
        print(
            f"[merged] {args.resume_run / 'activation_calibration.json'} "
            f"requests={manifest['completed_request_count']}",
            flush=True,
        )
        return 0

    if not args.captions.is_file():
        raise FileNotFoundError(f"Captions file does not exist: {args.captions}")
    cases = load_cases(args)
    records = validate_cases(cases)
    configuration = run_configuration(args, records)

    if args.dry_run:
        dry_run_dir = args.out_root / "dry_run"
        print(f"[dry-run] cases={len(cases)}")
        print(f"[dry-run] fingerprint={configuration['dataset_fingerprint']}")
        print(f"[dry-run] serve={shlex.join(build_service_command(args, dry_run_dir))}")
        print(
            f"[dry-run] first_task={REQUEST_PREFIX}{cases[0].case_id} "
            f"steps={args.num_inference_steps} frames={args.num_frames}",
            flush=True,
        )
        return 0

    validate_checkpoint(args)
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg is required to extract reference frames")
    if shutil.which(args.serve_executable) is None:
        raise FileNotFoundError(
            f"Service executable is not available: {args.serve_executable}"
        )
    if port_is_open(args.host, args.port):
        raise RuntimeError(
            f"Port {args.host}:{args.port} is already in use; stop that service first"
        )

    run_dir = args.resume_run or args.out_root / f"activation_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    collector_dir = run_dir / "collector"
    references_dir = run_dir / "references"
    outputs_dir = run_dir / "generated_videos"
    service_log = run_dir / "service.log"
    run_manifest_path = run_dir / "run_manifest.json"
    results_path = run_dir / "case_results.json"
    for directory in (
        collector_dir,
        references_dir,
        outputs_dir,
        run_dir / "server_outputs",
        run_dir / "server_inputs",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    if run_manifest_path.is_file():
        previous = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if previous.get("configuration") != configuration:
            raise ValueError(
                "Resume configuration does not match the existing run_manifest.json"
            )
        run_manifest = previous
    else:
        run_manifest = {
            "schema_version": 1,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "status": "starting",
            "run_dir": str(run_dir),
            "configuration": configuration,
            "cases": records,
        }
        atomic_write_json(run_manifest_path, run_manifest)

    results = {}
    if results_path.is_file():
        results = json.loads(results_path.read_text(encoding="utf-8"))

    for case in cases:
        extract_first_frame(case.video, references_dir / f"{case.case_id}.png")

    command = build_service_command(args, run_dir)
    env = build_service_env(args, collector_dir)
    run_manifest.pop("error", None)
    run_manifest["service_command"] = command
    run_manifest["status"] = "starting_service"
    run_manifest["updated_at"] = utc_now()
    atomic_write_json(run_manifest_path, run_manifest)
    print(f"[serve] {shlex.join(command)}", flush=True)

    process: subprocess.Popen[str] | None = None
    log_file = service_log.open("a", encoding="utf-8")
    interrupted = False
    try:
        log_file.write(f"\n===== service started {utc_now()} =====\n")
        log_file.write(f"$ {shlex.join(command)}\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        ready_seconds = wait_for_service(
            process,
            args.base_url,
            args.startup_timeout,
            service_log,
        )
        print(f"[ready] service initialized in {ready_seconds:.1f}s", flush=True)
        run_manifest["status"] = "running"
        run_manifest["service_ready_seconds"] = ready_seconds
        run_manifest["updated_at"] = utc_now()
        atomic_write_json(run_manifest_path, run_manifest)

        already_completed = completed_rank_requests(collector_dir, args.num_gpus)
        for index, case in enumerate(cases):
            task_id = f"{REQUEST_PREFIX}{case.case_id}"
            if task_id in already_completed:
                print(f"[skip] already calibrated: {task_id}", flush=True)
                continue
            reference = references_dir / f"{case.case_id}.png"
            output = outputs_dir / f"{task_id}.mp4"
            output.parent.mkdir(parents=True, exist_ok=True)
            payload = build_payload(args, case, index, reference, output)
            print(
                f"[submit] {index + 1:02d}/{len(cases):02d} {task_id} "
                f"caption={case.caption}",
                flush=True,
            )
            started = time.monotonic()
            submit_task(args, payload)
            progress = wait_for_task(args, task_id)
            elapsed = time.monotonic() - started
            results[case.case_id] = {
                "task_id": task_id,
                "status": "completed",
                "elapsed_seconds": elapsed,
                "completed_at": utc_now(),
                "progress": progress,
                "output_path": str(output) if args.keep_output_videos else None,
            }
            atomic_write_json(results_path, results)
            if not args.keep_output_videos:
                output.unlink(missing_ok=True)
            print(f"[completed] {task_id} elapsed={elapsed:.1f}s", flush=True)

        run_manifest["status"] = "stopping_service"
        run_manifest["updated_at"] = utc_now()
        atomic_write_json(run_manifest_path, run_manifest)
    except KeyboardInterrupt:
        interrupted = True
        run_manifest["status"] = "interrupted"
        run_manifest["updated_at"] = utc_now()
        atomic_write_json(run_manifest_path, run_manifest)
        print("[interrupted] state is resumable with --resume-run", flush=True)
    except BaseException as exc:
        run_manifest["status"] = "failed"
        run_manifest["error"] = repr(exc)
        run_manifest["updated_at"] = utc_now()
        atomic_write_json(run_manifest_path, run_manifest)
        raise
    finally:
        stop_process_group(process)
        log_file.write(f"===== service stopped {utc_now()} =====\n")
        log_file.close()

    if interrupted:
        return 130

    completed = completed_rank_requests(collector_dir, args.num_gpus)
    expected = {f"{REQUEST_PREFIX}{case.case_id}" for case in cases}
    if completed != expected:
        raise RuntimeError(
            f"Collector completed request mismatch: missing={sorted(expected - completed)} "
            f"unexpected={sorted(completed - expected)}"
        )
    merge_manifest = merge_rank_calibration(collector_dir, output_dir=run_dir)
    run_manifest["status"] = "completed"
    run_manifest["updated_at"] = utc_now()
    run_manifest["calibration_manifest"] = merge_manifest
    atomic_write_json(run_manifest_path, run_manifest)
    print(
        f"[summary] {run_dir / 'activation_calibration.json'} "
        f"cases={len(cases)} ranks={merge_manifest['rank_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
