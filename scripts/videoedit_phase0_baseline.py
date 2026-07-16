#!/usr/bin/env python3
"""Run VideoEdit Phase 0 BF16 baseline and capture logs automatically.

This script submits /v1/videos/repairs requests sequentially, polls progress,
stores perf JSONs, and captures docker logs + nvidia-smi into the output dir.

Example:
  python scripts/videoedit_phase0_baseline.py smoke single81 full
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


SCENARIOS: dict[str, dict[str, Any]] = {
    # Fast endpoint/path/mask/reference smoke test.
    "smoke": {
        "num_frames": 16,
        "infer_len": 81,
        "overlap": 0,
        "num_inference_steps": 4,
        "warmups": 0,
        "runs": 1,
    },
    # Single 81-frame internal window: 80 raw frames + 1 reference frame.
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


def request_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = 60) -> dict[str, Any]:
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
        with urllib.request.urlopen(req, timeout=timeout) as resp:
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


def collect_env_manifest(container: str | None, out_path: Path) -> None:
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
        "nvidia_smi": run_text(["nvidia-smi"], timeout=30),
    }
    if container:
        manifest["docker_exec_python"] = run_text(
            ["docker", "exec", container, "python", "-c", py],
            timeout=60,
        )
    manifest["local_python"] = run_text([sys.executable, "-c", py], timeout=60)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def build_payload(args: argparse.Namespace, scenario: str, task_id: str) -> dict[str, Any]:
    cfg = SCENARIOS[scenario]
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
        "drop_reference_frame": True,
        "enable_paste_back": True,
    }
    input_url_base = getattr(args, "input_url_base_effective", None)
    if input_url_base:
        payload["video_url"] = make_input_url(args, args.video)
        payload["mask_url"] = make_input_url(args, args.mask)
        payload["reference_image_url"] = make_input_url(args, args.reference)
    else:
        payload["video_input_path"] = args.video
        payload["mask_input_path"] = args.mask
        payload["reference_image_url"] = args.reference
    if args.negative_prompt is not None:
        payload["negative_prompt"] = args.negative_prompt
    return payload


def make_input_url(args: argparse.Namespace, path: str) -> str:
    root = Path(args.input_root_effective).resolve()
    source = Path(path).resolve()
    rel = source.relative_to(root).as_posix()
    return f"{args.input_url_base_effective.rstrip('/')}/{urllib.parse.quote(rel)}"


def infer_input_root(args: argparse.Namespace) -> Path:
    if args.input_root:
        return Path(args.input_root).resolve()
    paths = [Path(args.video).resolve(), Path(args.mask).resolve(), Path(args.reference).resolve()]
    return Path(os.path.commonpath([str(p.parent) for p in paths])).resolve()


def submit(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    resp = request_json(
        "POST",
        f"{args.base_url.rstrip('/')}/v1/videos/repairs",
        payload,
        timeout=args.submit_timeout,
    )
    if resp.get("code") != 0:
        raise RuntimeError(f"submit failed for {payload['task_id']}: {resp}")
    return resp


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
            raise RuntimeError(f"task failed: {json.dumps(progress, ensure_ascii=False)}")
        time.sleep(args.poll_interval)


def read_perf(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"missing": True, "path": path}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc), "path": path}


def run_scenario(args: argparse.Namespace, scenario: str, tag: str) -> dict[str, Any]:
    cfg = SCENARIOS[scenario]
    records: list[dict[str, Any]] = []
    print(f"[scenario] {scenario}: {cfg}", flush=True)

    for kind, count in (("warmup", int(cfg["warmups"])), ("run", int(cfg["runs"]))):
        for i in range(count):
            task_id = f"phase0_bf16_1080_{scenario}_{tag}_{kind}{i:02d}"
            payload = build_payload(args, scenario, task_id)
            payload_path = Path(args.out_dir) / f"{task_id}.payload.json"
            payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

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
        help="Scenarios to run: smoke, single81, full, or all. Default: smoke.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--video", default="/root/VideoEdit/test/1080.mp4")
    parser.add_argument("--mask", default="/root/VideoEdit/test/mask_1080_merged.mp4")
    parser.add_argument("--reference", default="/root/VideoEdit/test/local.png")
    parser.add_argument("--out-dir", default="/root/VideoEdit/test/phase0_bf16")
    parser.add_argument("--container", default="videoedit_reset")
    parser.add_argument(
        "--serve-inputs",
        action="store_true",
        help=(
            "Start a local HTTP server for video/mask/reference and send URL fields "
            "instead of server-local paths. Use this when the VideoEdit service cannot "
            "see /root/VideoEdit/test directly."
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
    parser.add_argument("--prompt", default="一个男人在舞台演讲，背后有两排文字。")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--bbox-expand-scale", type=float, default=0.3)
    parser.add_argument("--poll-interval", type=int, default=15)
    parser.add_argument("--submit-timeout", type=int, default=60)
    parser.add_argument("--task-timeout", type=int, default=-1)
    parser.add_argument("--no-docker-logs", action="store_true")
    parser.add_argument("--no-gpu-log", action="store_true")
    parser.add_argument("--no-env-manifest", action="store_true")
    args = parser.parse_args()

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
    return args


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = utc_tag()

    collectors: list[BackgroundProcess] = []
    if not args.no_docker_logs and args.container:
        collectors.append(
            BackgroundProcess(
                "docker_logs",
                ["docker", "logs", "-f", "--since", docker_since_timestamp(), args.container],
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
                args.input_url_base_effective = f"http://{args.input_url_host}:{actual_port}"
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
            collect_env_manifest(args.container, out_dir / f"phase0_{tag}.env_manifest.json")

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
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[summary] {summary_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
