#!/usr/bin/env python3
"""Run VideoEdit Phase 0 BF16 baseline and capture logs automatically.

This script submits /v1/videos/repairs requests sequentially, polls progress,
stores perf JSONs, and captures docker logs + nvidia-smi into the output dir.

Example:
  python scripts/videoedit_phase0_baseline.py smoke profile81 single81 full
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


DEFAULT_DATA_ROOT = Path("/mnt/nas/models/DifusserEdit/pexel_test_data_0410")
DEFAULT_MODEL_PATH = Path(
    "/mnt/nas/models/DifusserEdit/pretrain_models/VideoEdit-diffusers-model"
)
DEFAULT_TRANSFORMER_PATH = DEFAULT_MODEL_PATH / "transformer"
DEFAULT_VIDEO = Path("/sgl-workspace/sglang/demo/1080.mp4")
DEFAULT_MASK = Path("/sgl-workspace/sglang/demo/mask_1080_acc.mp4")
DEFAULT_REFERENCE = Path("/sgl-workspace/sglang/demo/local.png")
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "videoedit_phase0_outputs"
DEFAULT_PROMPT = (
    "A squirrel moves across a textured pavement, its bushy tail swaying as it walks."
)


SCENARIOS: dict[str, dict[str, Any]] = {
    # Minimal end-to-end service, loading, denoising, and video-output check.
    "smoke1": {
        "num_frames": 16,
        "infer_len": 81,
        "overlap": 0,
        "num_inference_steps": 1,
        "warmups": 0,
        "runs": 1,
    },
    # Fast endpoint/path/mask smoke test; reference image is optional.
    "smoke": {
        "num_frames": 16,
        "infer_len": 81,
        "overlap": 0,
        "num_inference_steps": 4,
        "warmups": 0,
        "runs": 1,
    },
    # Representative 81-frame shapes with fewer denoising steps for profiling.
    "profile81": {
        "num_frames": 80,
        "infer_len": 81,
        "overlap": 0,
        "num_inference_steps": 4,
        "warmups": 1,
        "runs": 3,
    },
    # Single 81-frame internal window. With reference, use 80 raw frames + 1 reference frame.
    "single81": {
        "num_frames": 80,
        "infer_len": 81,
        "overlap": 0,
        "num_inference_steps": 40,
        "warmups": 2,
        "runs": 5,
    },
    # Full input. With a reference image, effective internal frame count is raw+1.
    "full": {
        "num_frames": -1,
        "infer_len": 81,
        "overlap": 10,
        "num_inference_steps": 40,
        "warmups": 2,
        "runs": 5,
    },
}


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def docker_since_timestamp() -> str:
    # Docker accepts RFC3339 timestamps for --since.
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def p95(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = max(0, math.ceil(0.95 * len(values)) - 1)
    return values[idx]


def request_json(
    method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = 60
) -> dict[str, Any]:
    body = None
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with NO_PROXY_OPENER.open(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {text}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Non-JSON response from {url}: {text[:1000]}") from exc


class BackgroundProcess:
    def __init__(self, name: str, cmd: list[str], log_path: Path):
        self.name = name
        self.cmd = cmd
        self.log_path = log_path
        self.proc: subprocess.Popen[bytes] | None = None
        self.file = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.log_path, "ab")
        header = f"\n===== {self.name} started at {datetime.now(timezone.utc).isoformat()} =====\n"
        self.file.write(header.encode("utf-8"))
        self.file.flush()
        try:
            self.proc = subprocess.Popen(
                self.cmd,
                stdout=self.file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            msg = f"{self.name} not started; command not found: {self.cmd[0]}\n"
            self.file.write(msg.encode("utf-8"))
            self.file.flush()
            print(f"[warn] {msg.strip()}")
            self.proc = None
        except Exception as exc:
            msg = f"{self.name} not started: {exc}\n"
            self.file.write(msg.encode("utf-8"))
            self.file.flush()
            print(f"[warn] {msg.strip()}")
            self.proc = None

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except Exception:
                    pass
        if self.file is not None:
            footer = f"\n===== {self.name} stopped at {datetime.now(timezone.utc).isoformat()} =====\n"
            self.file.write(footer.encode("utf-8"))
            self.file.close()


class LocalInputServer:
    def __init__(self, root: Path, bind: str, port: int):
        self.root = root.resolve()
        self.bind = bind
        self.port = port
        self.httpd: http.server.ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> int:
        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler,
            directory=str(self.root),
        )
        self.httpd = http.server.ThreadingHTTPServer((self.bind, self.port), handler)
        self.httpd.daemon_threads = True
        actual_port = int(self.httpd.server_address[1])
        self.thread = threading.Thread(
            target=self.httpd.serve_forever,
            name="phase0-input-http-server",
            daemon=True,
        )
        self.thread.start()
        return actual_port

    def stop(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)


def run_text(cmd: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except FileNotFoundError as exc:
        return {"cmd": cmd, "error": f"command not found: {exc}"}
    except Exception as exc:
        return {"cmd": cmd, "error": str(exc)}


def path_access(path: str | None) -> dict[str, Any]:
    if not path:
        return {"configured": False}
    raw_path = Path(path).expanduser()
    try:
        absolute_path = str(raw_path.resolve(strict=False))
        resolve_error = None
    except Exception as exc:
        absolute_path = str(raw_path.absolute())
        resolve_error = repr(exc)
    info: dict[str, Any] = {
        "configured": True,
        "path": str(raw_path),
        "absolute_path": absolute_path,
        "exists": False,
        "readable": os.access(raw_path, os.R_OK),
        "writable": os.access(raw_path, os.W_OK),
        "executable": os.access(raw_path, os.X_OK),
    }
    if resolve_error:
        info["resolve_error"] = resolve_error
    try:
        stat = raw_path.stat()
    except FileNotFoundError:
        parent = raw_path.parent
        info["parent"] = str(parent)
        info["parent_exists"] = parent.exists()
        info["parent_writable"] = os.access(parent, os.W_OK)
    except PermissionError as exc:
        info["error"] = f"PermissionError: {exc}"
    except Exception as exc:
        info["error"] = repr(exc)
    else:
        info.update(
            {
                "exists": True,
                "is_file": raw_path.is_file(),
                "is_dir": raw_path.is_dir(),
                "mode": oct(stat.st_mode & 0o777),
                "uid": stat.st_uid,
                "gid": stat.st_gid,
                "size_bytes": stat.st_size if raw_path.is_file() else None,
            }
        )
    return info


def configured_path_checks(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_path": path_access(args.model_path),
        "transformer_path": path_access(args.transformer_path),
        "video": path_access(args.video),
        "mask": path_access(args.mask),
        "reference": path_access(args.reference),
        "out_dir": path_access(args.out_dir),
    }


def collect_env_manifest(
    container: str | None, out_path: Path, path_checks: dict[str, Any]
) -> None:
    py = r"""
import importlib.metadata as md
import json
import platform

def version(name):
    try:
        return md.version(name)
    except Exception as e:
        return None

data = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "packages": {
        "torch": version("torch"),
        "triton": version("triton"),
        "sglang": version("sglang"),
        "sglang-kernel": version("sglang-kernel"),
    },
}

try:
    import torch
    data["torch"] = {
        "version": torch.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "float8_e4m3fn_available": hasattr(torch, "float8_e4m3fn"),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            data["torch"]["devices"].append({
                "index": i,
                "name": p.name,
                "total_memory_bytes": p.total_memory,
                "compute_capability": [p.major, p.minor],
            })
except Exception as e:
    data["torch_error"] = repr(e)

try:
    from sglang.srt.layers.quantization.fp8_utils import cutlass_fp8_supported
    data["cutlass_fp8_supported"] = cutlass_fp8_supported()
except Exception as e:
    data["cutlass_fp8_supported_error"] = repr(e)

try:
    import sgl_kernel
    data["sgl_kernel"] = {
        "imported": True,
        "fp8_scaled_mm": hasattr(sgl_kernel, "fp8_scaled_mm"),
        "int8_scaled_mm": hasattr(sgl_kernel, "int8_scaled_mm"),
    }
except Exception as e:
    data["sgl_kernel"] = {"imported": False, "error": repr(e)}

print(json.dumps(data, ensure_ascii=False, indent=2))
"""
    manifest: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "path_checks": path_checks,
        "nvidia_smi": run_text(["nvidia-smi"], timeout=30),
    }
    if container:
        manifest["docker_exec_python"] = run_text(
            ["docker", "exec", container, "python", "-c", py],
            timeout=60,
        )
    manifest["local_python"] = run_text([sys.executable, "-c", py], timeout=60)
    out_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_payload(
    args: argparse.Namespace, scenario: str, task_id: str
) -> dict[str, Any]:
    cfg = SCENARIOS[scenario]
    reference = args.reference or None
    payload: dict[str, Any] = {
        "task_id": task_id,
        "timeout": args.task_timeout,
        "prompt": args.prompt,
        "output_storage": "local",
        "output_path": str(Path(args.out_dir) / f"{task_id}.mp4"),
        "perf_dump_path": str(Path(args.out_dir) / f"{task_id}.perf.json"),
        "num_frames": cfg["num_frames"],
        "infer_len": cfg["infer_len"],
        "overlap": cfg["overlap"],
        "num_inference_steps": cfg["num_inference_steps"],
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "dtype": "bf16",
        "dynamic_cfg": True,
        "dynamic_cfg_max_step": 15,
        "dynamic_cfg_min": 1.0,
        "bbox_expand_scale": args.bbox_expand_scale,
        "decode_mode": "stream",
        "enable_teacache": False,
        "drop_reference_frame": bool(reference) and args.drop_reference_frame,
        "enable_paste_back": True,
    }
    input_url_base = getattr(args, "input_url_base_effective", None)
    if input_url_base:
        payload["video_url"] = make_input_url(args, args.video)
        payload["mask_url"] = make_input_url(args, args.mask)
        if reference:
            payload["reference_image_url"] = make_input_url(args, reference)
    else:
        payload["video_input_path"] = args.video
        payload["mask_input_path"] = args.mask
        if reference:
            payload["reference_image_url"] = reference
    if args.negative_prompt is not None:
        payload["negative_prompt"] = args.negative_prompt
    if args.profile:
        payload["profile"] = True
        payload["num_profiled_timesteps"] = args.num_profiled_timesteps
        payload["profile_all_stages"] = args.profile_all_stages
    return payload


def make_input_url(args: argparse.Namespace, path: str) -> str:
    root = Path(args.input_root_effective).resolve()
    source = Path(path).resolve()
    rel = source.relative_to(root).as_posix()
    return f"{args.input_url_base_effective.rstrip('/')}/{urllib.parse.quote(rel)}"


def infer_input_root(args: argparse.Namespace) -> Path:
    if args.input_root:
        return Path(args.input_root).resolve()
    paths = [Path(args.video).resolve(), Path(args.mask).resolve()]
    if args.reference:
        paths.append(Path(args.reference).resolve())
    return Path(os.path.commonpath([str(p.parent) for p in paths])).resolve()


def validate_local_inputs(args: argparse.Namespace) -> None:
    for label, value in (
        ("video", args.video),
        ("mask", args.mask),
        ("reference", args.reference),
    ):
        if not value:
            continue
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"{label} path does not exist: {value}")
        if not path.is_file():
            raise ValueError(f"{label} path is not a file: {value}")
        if not os.access(path, os.R_OK):
            raise PermissionError(f"{label} path is not readable: {value}")


def submit(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    task_id = str(payload["task_id"])
    submit_url = args.base_url.rstrip("/") + "/v1/videos/repairs"
    started = time.monotonic()
    attempt = 0
    while True:
        resp = request_json(
            "POST",
            submit_url,
            payload,
            timeout=args.submit_timeout,
        )
        if resp.get("code") == 0:
            return resp
        if resp.get("code") == 2 and args.busy_retry_timeout != 0:
            elapsed = time.monotonic() - started
            if args.busy_retry_timeout > 0 and elapsed >= args.busy_retry_timeout:
                raise RuntimeError(f"submit busy timeout for {task_id}: {resp}")
            attempt += 1
            wait_s = max(1, args.busy_retry_interval)
            if args.busy_retry_timeout > 0:
                remaining = max(1, int(args.busy_retry_timeout - elapsed))
                wait_s = min(wait_s, remaining)
            print(
                f"[busy] service already has a running task; retrying {task_id} "
                f"in {wait_s}s (attempt {attempt})",
                flush=True,
            )
            time.sleep(wait_s)
            continue
        raise RuntimeError(f"submit failed for {task_id}: {resp}")


def wait_done(args: argparse.Namespace, task_id: str) -> dict[str, Any]:
    progress_url = f"{args.base_url.rstrip('/')}/v1/videos/{task_id}/progress"
    while True:
        progress = request_json("GET", progress_url, timeout=args.submit_timeout)
        print(
            f"[progress] {task_id} status={progress.get('status')} "
            f"progress={progress.get('progress')} reason={progress.get('reason')}",
            flush=True,
        )
        if progress.get("status") == "completed":
            return progress
        if progress.get("status") == "failed":
            raise RuntimeError(
                f"task failed: {json.dumps(progress, ensure_ascii=False)}"
            )
        time.sleep(args.poll_interval)


def read_perf(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"missing": True, "path": path}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc), "path": path}


def parse_start_at(value: str | None) -> tuple[str, str, int] | None:
    if not value:
        return None
    try:
        scenario, marker = value.split(":", 1)
    except ValueError as exc:
        raise ValueError(
            "--start-at must look like SCENARIO:warmupNN or SCENARIO:runNN"
        ) from exc
    for kind in ("warmup", "run"):
        if marker.startswith(kind):
            suffix = marker[len(kind) :]
            if suffix.isdigit():
                return scenario, kind, int(suffix)
    raise ValueError("--start-at must look like SCENARIO:warmupNN or SCENARIO:runNN")


def should_skip_task(
    args: argparse.Namespace, scenario: str, kind: str, index: int
) -> bool:
    start = getattr(args, "start_at_parsed", None)
    if start is None:
        return False
    start_scenario, start_kind, start_index = start
    scenario_order = list(SCENARIOS)
    scenario_pos = scenario_order.index(scenario)
    start_scenario_pos = scenario_order.index(start_scenario)
    if scenario_pos < start_scenario_pos:
        return True
    if scenario_pos > start_scenario_pos:
        return False
    kind_order = {"warmup": 0, "run": 1}
    if kind_order[kind] < kind_order[start_kind]:
        return True
    if kind_order[kind] > kind_order[start_kind]:
        return False
    return index < start_index


def run_scenario(args: argparse.Namespace, scenario: str, tag: str) -> dict[str, Any]:
    cfg = dict(SCENARIOS[scenario])
    if args.warmups is not None:
        cfg["warmups"] = args.warmups
    if args.runs is not None:
        cfg["runs"] = args.runs
    records: list[dict[str, Any]] = []
    print(f"[scenario] {scenario}: {cfg}", flush=True)

    for kind, count in (("warmup", int(cfg["warmups"])), ("run", int(cfg["runs"]))):
        for i in range(count):
            task_id = f"{args.task_prefix}_{scenario}_{tag}_{kind}{i:02d}"
            if should_skip_task(args, scenario, kind, i):
                print(f"[skip] {task_id} before --start-at {args.start_at}", flush=True)
                continue
            payload = build_payload(args, scenario, task_id)
            payload_path = Path(args.out_dir) / f"{task_id}.payload.json"
            payload_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            print(f"[submit] {task_id}", flush=True)
            submit_resp = submit(args, payload)
            started = time.perf_counter()
            progress = wait_done(args, task_id)
            elapsed_s = time.perf_counter() - started
            perf = read_perf(payload["perf_dump_path"])

            record = {
                "task_id": task_id,
                "scenario": scenario,
                "kind": kind,
                "submit_response": submit_resp,
                "progress": progress,
                "payload_path": str(payload_path),
                "output_path": payload["output_path"],
                "perf_path": payload["perf_dump_path"],
                "wall_wait_s": elapsed_s,
                "total_duration_ms": perf.get("total_duration_ms"),
                "perf_missing": perf.get("missing", False),
                "perf_error": perf.get("error"),
            }
            records.append(record)

    formal = [
        float(r["total_duration_ms"])
        for r in records
        if r["kind"] == "run" and isinstance(r.get("total_duration_ms"), (int, float))
    ]
    return {
        "scenario": scenario,
        "config": cfg,
        "records": records,
        "stats": {
            "formal_count": len(formal),
            "median_ms": statistics.median(formal) if formal else None,
            "p95_ms": p95(formal),
            "min_ms": min(formal) if formal else None,
            "max_ms": max(formal) if formal else None,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenarios",
        nargs="*",
        default=["smoke"],
        help=(
            "Scenarios to run: smoke1, smoke, profile81, single81, full, or all. "
            "Default: smoke. 'all' retains the Phase 0 smoke/single81/full set."
        ),
    )
    parser.add_argument(
        "--base-url", default=os.getenv("VIDEOEDIT_BASE_URL", "http://127.0.0.1:30000")
    )
    parser.add_argument(
        "--video", default=os.getenv("VIDEOEDIT_INPUT_VIDEO", str(DEFAULT_VIDEO))
    )
    parser.add_argument(
        "--mask", default=os.getenv("VIDEOEDIT_INPUT_MASK", str(DEFAULT_MASK))
    )
    parser.add_argument(
        "--reference",
        default=os.getenv("VIDEOEDIT_REFERENCE_IMAGE", str(DEFAULT_REFERENCE)),
        help="Optional local reference image path.",
    )
    parser.add_argument(
        "--out-dir", default=os.getenv("VIDEOEDIT_OUT_DIR", str(DEFAULT_OUT_DIR))
    )
    parser.add_argument(
        "--task-prefix",
        default=os.getenv("VIDEOEDIT_TASK_PREFIX", "phase0_bf16_1080"),
        help="Prefix used for task IDs and output filenames.",
    )
    parser.add_argument("--container", default=os.getenv("VIDEOEDIT_CONTAINER", ""))
    parser.add_argument(
        "--model-path",
        default=os.getenv("VIDEOEDIT_MODEL_PATH", str(DEFAULT_MODEL_PATH)),
        help="Model path to record in the Phase 0 manifest. The running service must already be started with this model.",
    )
    parser.add_argument(
        "--transformer-path",
        default=os.getenv("VIDEOEDIT_TRANSFORMER_PATH", str(DEFAULT_TRANSFORMER_PATH)),
        help="Transformer path to record/check in the Phase 0 manifest. The request API does not load it dynamically.",
    )
    parser.add_argument(
        "--drop-reference-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set drop_reference_frame in repair payload when --reference is provided.",
    )
    parser.add_argument(
        "--serve-inputs",
        action="store_true",
        help=(
            "Start a local HTTP server for video/mask/reference and send URL fields "
            "instead of server-local paths. Use this when the VideoEdit service cannot "
            "see the configured local input paths directly."
        ),
    )
    parser.add_argument(
        "--input-root",
        default=None,
        help="Directory to serve when --serve-inputs is set. Default: common parent of video/mask/reference.",
    )
    parser.add_argument("--input-server-bind", default="0.0.0.0")
    parser.add_argument("--input-server-port", type=int, default=18081)
    parser.add_argument(
        "--input-url-host",
        default="127.0.0.1",
        help=(
            "Host/IP that the VideoEdit service should use to reach the local input server. "
            "If the service runs in another container, set this to an address reachable from that container."
        ),
    )
    parser.add_argument(
        "--input-url-base",
        default=None,
        help="Explicit base URL for served inputs, e.g. http://172.17.0.1:18081.",
    )
    parser.add_argument(
        "--prompt", default=os.getenv("VIDEOEDIT_PROMPT", DEFAULT_PROMPT)
    )
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--bbox-expand-scale", type=float, default=0.3)
    parser.add_argument("--poll-interval", type=int, default=15)
    parser.add_argument("--submit-timeout", type=int, default=60)
    parser.add_argument("--busy-retry-interval", type=int, default=30)
    parser.add_argument("--busy-retry-timeout", type=int, default=3600)
    parser.add_argument("--task-timeout", type=int, default=-1)
    parser.add_argument(
        "--warmups",
        type=int,
        default=None,
        help="Override the selected scenario's warmup count.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Override the selected scenario's formal run count.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable the diffusion torch profiler for each submitted request.",
    )
    parser.add_argument(
        "--num-profiled-timesteps",
        type=int,
        default=1,
        help="Denoising timesteps to capture after one profiler warmup step.",
    )
    parser.add_argument(
        "--profile-all-stages",
        action="store_true",
        help="Profile every pipeline stage instead of denoising timesteps only.",
    )
    parser.add_argument(
        "--start-at",
        default=None,
        help="Skip tasks before SCENARIO:warmupNN or SCENARIO:runNN, e.g. full:warmup01.",
    )
    parser.add_argument("--no-docker-logs", action="store_true")
    parser.add_argument("--no-gpu-log", action="store_true")
    parser.add_argument("--no-env-manifest", action="store_true")
    args = parser.parse_args()

    if args.num_profiled_timesteps == 0 or args.num_profiled_timesteps < -1:
        parser.error("--num-profiled-timesteps must be positive or -1")
    if args.warmups is not None and args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.runs is not None and args.runs < 1:
        parser.error("--runs must be positive")

    expanded: list[str] = []
    for scenario in args.scenarios:
        if scenario == "all":
            expanded.extend(["smoke", "single81", "full"])
        else:
            expanded.append(scenario)
    unknown = [s for s in expanded if s not in SCENARIOS]
    if unknown:
        parser.error(f"unknown scenarios: {unknown}; valid: {sorted(SCENARIOS)} or all")
    args.scenarios = expanded
    try:
        args.start_at_parsed = parse_start_at(args.start_at)
    except ValueError as exc:
        parser.error(str(exc))
    if args.start_at_parsed is not None and args.start_at_parsed[0] not in SCENARIOS:
        parser.error(
            f"unknown --start-at scenario: {args.start_at_parsed[0]}; valid: {sorted(SCENARIOS)}"
        )
    return args


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    validate_local_inputs(args)
    path_checks = configured_path_checks(args)
    tag = utc_tag()

    collectors: list[BackgroundProcess] = []
    if not args.no_docker_logs and args.container:
        collectors.append(
            BackgroundProcess(
                "docker_logs",
                [
                    "docker",
                    "logs",
                    "-f",
                    "--since",
                    docker_since_timestamp(),
                    args.container,
                ],
                out_dir / f"phase0_{tag}.docker.log",
            )
        )
    if not args.no_gpu_log:
        collectors.append(
            BackgroundProcess(
                "nvidia_smi_loop",
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,index,name,memory.used,utilization.gpu",
                    "--format=csv",
                    "-l",
                    "5",
                ],
                out_dir / f"phase0_{tag}.gpu_mem.csv",
            )
        )

    summary: dict[str, Any] = {
        "tag": tag,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "path_checks": path_checks,
        "scenarios": [],
        "log_paths": {
            "docker": str(out_dir / f"phase0_{tag}.docker.log"),
            "gpu_mem": str(out_dir / f"phase0_{tag}.gpu_mem.csv"),
            "env_manifest": str(out_dir / f"phase0_{tag}.env_manifest.json"),
        },
    }

    input_server: LocalInputServer | None = None
    try:
        if args.serve_inputs:
            input_root = infer_input_root(args)
            args.input_root_effective = str(input_root)
            input_server = LocalInputServer(
                root=input_root,
                bind=args.input_server_bind,
                port=args.input_server_port,
            )
            actual_port = input_server.start()
            if args.input_url_base:
                args.input_url_base_effective = args.input_url_base
            else:
                args.input_url_base_effective = (
                    f"http://{args.input_url_host}:{actual_port}"
                )
            summary["input_server"] = {
                "root": str(input_root),
                "bind": args.input_server_bind,
                "port": actual_port,
                "url_base": args.input_url_base_effective,
            }
            print(
                f"[input-server] serving {input_root} at {args.input_url_base_effective}",
                flush=True,
            )
        else:
            args.input_root_effective = None
            args.input_url_base_effective = None

        for collector in collectors:
            collector.start()

        if not args.no_env_manifest:
            collect_env_manifest(
                args.container, out_dir / f"phase0_{tag}.env_manifest.json", path_checks
            )

        for scenario in args.scenarios:
            summary["scenarios"].append(run_scenario(args, scenario, tag))

        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        summary["status"] = "completed"
        return 0
    except KeyboardInterrupt:
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        summary["status"] = "interrupted"
        raise
    except Exception as exc:
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        summary["status"] = "failed"
        summary["error"] = repr(exc)
        print(f"[error] {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if input_server is not None:
            input_server.stop()
        for collector in collectors:
            collector.stop()
        summary_path = out_dir / f"phase0_{tag}.summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[summary] {summary_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
