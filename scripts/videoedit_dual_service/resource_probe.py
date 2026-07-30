#!/usr/bin/env python3
"""Resource snapshots and hard gates for VideoEdit dual-service startup."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

GIB = 1024**3


def _read_int_or_max(path: str) -> int | None:
    try:
        value = Path(path).read_text().strip()
    except OSError:
        return None
    return None if value == "max" else int(value)


def _mem_available() -> int:
    with open("/proc/meminfo", encoding="utf-8") as meminfo:
        for line in meminfo:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable was not found in /proc/meminfo")


def _gpu_rows() -> list[dict[str, int | str]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in output.splitlines():
        index, name, total, used, free = [part.strip() for part in line.split(",", 4)]
        rows.append(
            {
                "index": int(index),
                "name": name,
                "total_mib": int(total),
                "used_mib": int(used),
                "free_mib": int(free),
            }
        )
    return rows


def snapshot() -> dict[str, Any]:
    return {
        "timestamp": time.time(),
        "gpus": _gpu_rows(),
        "mem_available_bytes": _mem_available(),
        "cgroup_memory_current": _read_int_or_max("/sys/fs/cgroup/memory.current"),
        "cgroup_memory_max": _read_int_or_max("/sys/fs/cgroup/memory.max"),
        "cgroup_memory_events": _memory_events(),
    }


def _memory_events() -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        lines = Path("/sys/fs/cgroup/memory.events").read_text().splitlines()
    except OSError:
        return result
    for line in lines:
        key, value = line.split()
        result[key] = int(value)
    return result


def _write_json(path: str | None, value: dict[str, Any]) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)


def _healthy(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def monitor_startup(args: argparse.Namespace) -> int:
    baseline = snapshot()
    peak = baseline
    samples = [baseline]
    deadline = time.monotonic() + args.timeout
    healthy_at = None
    last_used = [gpu["used_mib"] for gpu in baseline["gpus"]]
    stable_since = None

    while time.monotonic() < deadline:
        current = snapshot()
        samples.append(current)
        peak = _merge_peak(peak, current)
        used = [gpu["used_mib"] for gpu in current["gpus"]]
        if _healthy(args.health_url):
            healthy_at = healthy_at or time.time()
            if used == last_used:
                stable_since = stable_since or time.monotonic()
            else:
                stable_since = None
            if (
                stable_since is not None
                and time.monotonic() - stable_since >= args.stable_seconds
            ):
                result = _startup_result(baseline, peak, current, healthy_at, samples)
                _write_json(args.output, result)
                return 0
        last_used = used
        time.sleep(1)

    result = _startup_result(baseline, peak, snapshot(), healthy_at, samples)
    result["error"] = f"Health did not stabilize within {args.timeout}s"
    _write_json(args.output, result)
    return 1


def _merge_peak(peak: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    merged = dict(peak)
    merged["gpus"] = []
    for old, new in zip(peak["gpus"], current["gpus"], strict=True):
        row = dict(new)
        row["used_mib"] = max(int(old["used_mib"]), int(new["used_mib"]))
        merged["gpus"].append(row)
    for key in ("cgroup_memory_current",):
        values = [
            value for value in (peak.get(key), current.get(key)) if value is not None
        ]
        merged[key] = max(values) if values else None
    return merged


def _startup_result(baseline, peak, final, healthy_at, samples):
    return {
        "baseline": baseline,
        "peak": peak,
        "final": final,
        "healthy_at": healthy_at,
        "sample_count": len(samples),
        "gpu_peak_delta_mib": [
            int(high["used_mib"]) - int(low["used_mib"])
            for low, high in zip(baseline["gpus"], peak["gpus"], strict=True)
        ],
        "gpu_idle_delta_mib": [
            int(end["used_mib"]) - int(low["used_mib"])
            for low, end in zip(baseline["gpus"], final["gpus"], strict=True)
        ],
        "cgroup_peak_delta_bytes": max(
            0,
            int(peak.get("cgroup_memory_current") or 0)
            - int(baseline.get("cgroup_memory_current") or 0),
        ),
    }


def gate_second(args: argparse.Namespace) -> int:
    metrics = json.loads(Path(args.metrics).read_text())
    current = snapshot()
    reasons = []
    reserve_mib = int(args.gpu_headroom_gib * 1024)
    for gpu, peak_delta in zip(
        current["gpus"], metrics["gpu_peak_delta_mib"], strict=True
    ):
        required = int(peak_delta) + reserve_mib
        if int(gpu["free_mib"]) < required:
            reasons.append(
                f"GPU {gpu['index']} free {gpu['free_mib']} MiB < required {required} MiB"
            )

    cpu_required = int(
        1.15 * metrics["cgroup_peak_delta_bytes"] + args.host_headroom_gib * GIB
    )
    if current["mem_available_bytes"] < cpu_required:
        reasons.append("MemAvailable is below first-service delta plus host reserve")
    if current["cgroup_memory_max"] is not None:
        cgroup_free = current["cgroup_memory_max"] - int(
            current["cgroup_memory_current"] or 0
        )
        required = int(
            1.15 * metrics["cgroup_peak_delta_bytes"] + args.cgroup_headroom_gib * GIB
        )
        if cgroup_free < required:
            reasons.append("cgroup memory headroom is below the configured reserve")

    result = {"ok": not reasons, "reasons": reasons, "current": current}
    _write_json(args.output, result)
    return 0 if not reasons else 1


def gate_idle(args: argparse.Namespace) -> int:
    current = snapshot()
    reasons = []
    minimum_free = int(args.gpu_headroom_gib * 1024)
    for gpu in current["gpus"]:
        if int(gpu["free_mib"]) < minimum_free:
            reasons.append(f"GPU {gpu['index']} has insufficient idle headroom")
    if current["mem_available_bytes"] < args.host_headroom_gib * GIB:
        reasons.append("MemAvailable is below idle reserve")
    if current["cgroup_memory_max"] is not None:
        free = current["cgroup_memory_max"] - int(current["cgroup_memory_current"] or 0)
        if free < args.cgroup_headroom_gib * GIB:
            reasons.append("cgroup memory headroom is below idle reserve")
    result = {"ok": not reasons, "reasons": reasons, "current": current}
    _write_json(args.output, result)
    return 0 if not reasons else 1


def check_ports(ports: list[int]) -> int:
    unavailable = []
    sockets = []
    try:
        for port in ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(("127.0.0.1", port))
                sockets.append(sock)
            except OSError as error:
                unavailable.append({"port": port, "error": str(error)})
                sock.close()
    finally:
        for sock in sockets:
            sock.close()
    print(json.dumps({"ok": not unavailable, "unavailable": unavailable}, indent=2))
    return 0 if not unavailable else 1


def validate_transformer(path_value: str) -> int:
    path = Path(path_value)
    errors = []
    config_path = path / "config.json"
    if not config_path.is_file():
        errors.append("config.json is missing")
        config = {}
    else:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as error:
            errors.append(f"config.json is invalid: {error}")
            config = {}
    if config.get("_class_name") != "WanVideoEditTransformer3DModel":
        errors.append(
            "_class_name must be WanVideoEditTransformer3DModel, got "
            f"{config.get('_class_name')!r}"
        )
    for key, expected in (("in_channels", 36), ("out_channels", 16)):
        if config.get(key) != expected:
            errors.append(f"{key} must be {expected}, got {config.get(key)!r}")

    weights = sorted(path.glob("*.safetensors"))
    if not weights:
        errors.append("no safetensors weights were found")
    for weight in weights:
        if weight.stat().st_size == 0:
            errors.append(f"empty weight file: {weight.name}")

    index_path = path / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            referenced = set(index.get("weight_map", {}).values())
            for filename in sorted(referenced):
                target = path / filename
                if not target.is_file() or target.stat().st_size == 0:
                    errors.append(f"missing or empty indexed shard: {filename}")
        except Exception as error:
            errors.append(f"weight index is invalid: {error}")

    result = {"ok": not errors, "path": str(path), "errors": errors}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    snap = subparsers.add_parser("snapshot")
    snap.add_argument("--output")

    monitor = subparsers.add_parser("monitor-startup")
    monitor.add_argument("--health-url", required=True)
    monitor.add_argument("--output", required=True)
    monitor.add_argument("--timeout", type=int, default=900)
    monitor.add_argument("--stable-seconds", type=int, default=10)

    second = subparsers.add_parser("gate-second")
    second.add_argument("--metrics", required=True)
    second.add_argument("--output")
    second.add_argument("--gpu-headroom-gib", type=float, default=2)
    second.add_argument("--host-headroom-gib", type=float, default=40)
    second.add_argument("--cgroup-headroom-gib", type=float, default=40)

    idle = subparsers.add_parser("gate-idle")
    idle.add_argument("--output")
    idle.add_argument("--gpu-headroom-gib", type=float, default=4)
    idle.add_argument("--host-headroom-gib", type=float, default=40)
    idle.add_argument("--cgroup-headroom-gib", type=float, default=40)

    ports = subparsers.add_parser("check-ports")
    ports.add_argument("ports", nargs="+", type=int)

    transformer = subparsers.add_parser("validate-transformer")
    transformer.add_argument("path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "snapshot":
        _write_json(args.output, snapshot())
        return 0
    if args.command == "monitor-startup":
        return monitor_startup(args)
    if args.command == "gate-second":
        return gate_second(args)
    if args.command == "gate-idle":
        return gate_idle(args)
    if args.command == "check-ports":
        return check_ports(args.ports)
    return validate_transformer(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
