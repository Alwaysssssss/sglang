#!/usr/bin/env python3
"""Add calibrated static FP8 activation scales to a VideoEdit FP8 checkpoint."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.utils.activation_calibration import (
    FP8_E4M3_MAX,
    checkpoint_aliases_for_runtime_linear,
)

FORMAT_NAME = "sglang-videoedit-fp8-static-activation"
FORMAT_VERSION = 1
SCALE_SHARD_NAME = "activation_input_scales.safetensors"
SCALE_CANDIDATES = ("max", "p99", "p99_9", "p99_99", "p99_999", "p99_9999")

BLOCK_LINEAR_SUFFIXES = (
    "attn1.to_q.weight",
    "attn1.to_k.weight",
    "attn1.to_v.weight",
    "attn1.to_out.0.weight",
    "attn2.to_q.weight",
    "attn2.to_k.weight",
    "attn2.to_v.weight",
    "attn2.to_out.0.weight",
    "attn2.add_k_proj.weight",
    "attn2.add_v_proj.weight",
    "ffn.net.0.proj.weight",
    "ffn.net.2.weight",
)

CONDITION_LINEAR_WEIGHTS = {
    "condition_embedder.image_embedder.ff.net.0.proj.weight",
    "condition_embedder.image_embedder.ff.net.2.weight",
    "condition_embedder.text_embedder.linear_1.weight",
    "condition_embedder.text_embedder.linear_2.weight",
    "condition_embedder.time_embedder.linear_1.weight",
    "condition_embedder.time_embedder.linear_2.weight",
    "condition_embedder.time_proj.weight",
    "proj_out.weight",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def expected_linear_weight_names(num_layers: int) -> set[str]:
    names = set(CONDITION_LINEAR_WEIGHTS)
    for layer_index in range(num_layers):
        names.update(
            f"blocks.{layer_index}.{suffix}" for suffix in BLOCK_LINEAR_SUFFIXES
        )
    return names


def discover_checkpoint(
    input_dir: Path,
) -> tuple[list[Path], dict[str, str], str, dict[str, Any]]:
    index_paths = sorted(input_dir.glob("*.safetensors.index.json"))
    if len(index_paths) > 1:
        raise ValueError(f"Multiple safetensors indexes found in {input_dir}")

    if index_paths:
        index_path = index_paths[0]
        index = load_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid weight_map in {index_path}")
        shard_paths = [input_dir / name for name in sorted(set(weight_map.values()))]
        index_name = index_path.name
    else:
        shard_paths = sorted(input_dir.glob("*.safetensors"))
        weight_map: dict[str, str] = {}
        for shard_path in shard_paths:
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in weight_map:
                        raise ValueError(f"Duplicate tensor {key!r}")
                    weight_map[key] = shard_path.name
        index_name = "diffusion_pytorch_model.safetensors.index.json"
        index = {"metadata": {}, "weight_map": weight_map}

    if not shard_paths:
        raise FileNotFoundError(f"No safetensors checkpoint found in {input_dir}")
    missing = [str(path) for path in shard_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint shards: {missing}")
    return shard_paths, weight_map, index_name, index


def resolve_calibration_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "activation_calibration.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing activation calibration: {path}")
    return path


def validate_source_config(config: dict[str, Any]) -> int:
    if config.get("_class_name") != "WanVideoEditTransformer3DModel":
        raise ValueError(
            "Static activation export only supports " "WanVideoEditTransformer3DModel"
        )
    num_layers = config.get("num_layers")
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError(f"Invalid num_layers: {num_layers!r}")
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise ValueError("Input checkpoint is not a serialized FP8 checkpoint")
    if quantization.get("quant_method") != "fp8":
        raise ValueError(f"Expected quant_method='fp8', got {quantization!r}")
    if quantization.get("activation_scheme") != "dynamic":
        raise ValueError(
            "Input checkpoint must use dynamic activations before static export"
        )
    if quantization.get("weight_scale_granularity") != "channel":
        raise ValueError("Input checkpoint must use per-channel FP8 weight scales")
    return num_layers


def validate_source_weights(
    weight_map: dict[str, str], expected_weights: set[str]
) -> None:
    available = set(weight_map)
    missing_weights = sorted(expected_weights - available)
    if missing_weights:
        raise ValueError(
            f"Missing {len(missing_weights)} expected Linear weights: "
            f"{missing_weights[:8]}"
        )
    missing_scales = sorted(
        weight.removesuffix(".weight") + ".weight_scale"
        for weight in expected_weights
        if weight.removesuffix(".weight") + ".weight_scale" not in available
    )
    if missing_scales:
        raise ValueError(
            f"Missing {len(missing_scales)} serialized weight scales: "
            f"{missing_scales[:8]}"
        )
    input_scales = sorted(key for key in available if key.endswith(".input_scale"))
    if input_scales:
        raise ValueError(
            f"Input checkpoint already contains static activation scales: "
            f"{input_scales[:8]}"
        )


def read_weight_shapes(
    shard_paths: list[Path], expected_weights: set[str]
) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key in set(handle.keys()) & expected_weights:
                shapes[key] = tuple(handle.get_slice(key).get_shape())
    missing = expected_weights - set(shapes)
    if missing:
        raise ValueError(f"Could not read shapes for weights: {sorted(missing)[:8]}")
    return shapes


def build_scale_plan(
    calibration: dict[str, Any],
    *,
    candidate: str,
    safety_factor: float,
    expected_prefixes: set[str],
    weight_shapes: dict[str, tuple[int, ...]],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    quantization = calibration.get("quantization")
    if not isinstance(quantization, dict):
        raise ValueError("Calibration is missing quantization metadata")
    expected_quantization = {
        "dtype": "fp8_e4m3fn",
        "symmetric": True,
        "zero_point": None,
        "granularity": "per_runtime_linear",
    }
    for key, expected in expected_quantization.items():
        if quantization.get(key) != expected:
            raise ValueError(
                f"Calibration quantization {key!r} must be {expected!r}, "
                f"got {quantization.get(key)!r}"
            )
    if float(quantization.get("fp8_max", 0.0)) != FP8_E4M3_MAX:
        raise ValueError("Calibration FP8 maximum does not match E4M3FN")

    completed_requests = calibration.get("completed_requests")
    if not isinstance(completed_requests, list) or not completed_requests:
        raise ValueError("Calibration has no completed requests")
    modules = calibration.get("modules")
    if not isinstance(modules, list) or not modules:
        raise ValueError("Calibration has no module records")

    scale_tensors: dict[str, torch.Tensor] = {}
    module_plan: list[dict[str, Any]] = []
    alias_owners: dict[str, str] = {}
    for record in modules:
        name = record.get("name")
        if not isinstance(name, str):
            raise ValueError(f"Invalid calibration module record: {record!r}")
        if int(record.get("nonfinite_token_count", 0)) != 0:
            raise ValueError(f"Calibration module {name!r} contains non-finite tokens")
        if int(record.get("observation_count", 0)) <= 0:
            raise ValueError(f"Calibration module {name!r} has no observations")

        aliases = checkpoint_aliases_for_runtime_linear(name)
        if record.get("checkpoint_aliases") != aliases:
            raise ValueError(
                f"Stale checkpoint aliases for {name!r}; rerun calibration merge"
            )
        candidate_scales = record.get("candidate_scales")
        candidate_thresholds = record.get("candidate_thresholds")
        if not isinstance(candidate_scales, dict) or candidate not in candidate_scales:
            raise ValueError(f"Module {name!r} has no scale candidate {candidate!r}")
        if (
            not isinstance(candidate_thresholds, dict)
            or candidate not in candidate_thresholds
        ):
            raise ValueError(
                f"Module {name!r} has no threshold candidate {candidate!r}"
            )
        source_scale = float(candidate_scales[candidate])
        threshold = float(candidate_thresholds[candidate])
        exported_scale = source_scale * safety_factor
        if not math.isfinite(exported_scale) or exported_scale <= 0:
            raise ValueError(f"Invalid exported scale for {name!r}: {exported_scale}")

        input_features = int(record.get("input_features", 0))
        for alias in aliases:
            previous_owner = alias_owners.get(alias)
            if previous_owner is not None:
                raise ValueError(
                    f"Checkpoint projection {alias!r} is mapped by both "
                    f"{previous_owner!r} and {name!r}"
                )
            alias_owners[alias] = name
            weight_name = f"{alias}.weight"
            shape = weight_shapes.get(weight_name)
            if shape is None or len(shape) != 2:
                raise ValueError(f"Missing two-dimensional weight {weight_name!r}")
            if shape[-1] != input_features:
                raise ValueError(
                    f"Input feature mismatch for {name!r} -> {weight_name!r}: "
                    f"calibration={input_features}, checkpoint={shape[-1]}"
                )
            scale_tensors[f"{alias}.input_scale"] = torch.tensor(
                [exported_scale], dtype=torch.float32
            )

        module_plan.append(
            {
                "runtime_name": name,
                "checkpoint_aliases": aliases,
                "input_features": input_features,
                "observation_count": int(record["observation_count"]),
                "token_count": int(record.get("token_count", 0)),
                "absmax": float(record.get("absmax", 0.0)),
                "candidate": candidate,
                "candidate_threshold": threshold,
                "candidate_scale": source_scale,
                "safety_factor": safety_factor,
                "exported_scale": exported_scale,
                "exported_fp8_range": exported_scale * FP8_E4M3_MAX,
            }
        )

    aliases = set(alias_owners)
    missing_aliases = sorted(expected_prefixes - aliases)
    unexpected_aliases = sorted(aliases - expected_prefixes)
    if missing_aliases or unexpected_aliases:
        raise ValueError(
            "Calibration/checkpoint projection coverage mismatch: "
            f"missing={missing_aliases[:8]}, unexpected={unexpected_aliases[:8]}"
        )
    return scale_tensors, module_plan


def materialize_shard(source: Path, destination: Path, mode: str) -> str:
    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError as error:
        if mode == "hardlink" or error.errno not in {
            errno.EXDEV,
            errno.EPERM,
            errno.EACCES,
            errno.ENOTSUP,
        }:
            raise
    shutil.copy2(source, destination)
    return "copy"


def copy_auxiliary_files(
    input_dir: Path, output_dir: Path, shard_names: set[str]
) -> None:
    for source in sorted(input_dir.iterdir()):
        if not source.is_file() or source.name in shard_names:
            continue
        if source.name == "config.json":
            continue
        if source.name.endswith(".safetensors.index.json"):
            continue
        if source.name == "videoedit_static_activation_manifest.json":
            continue
        shutil.copy2(source, output_dir / source.name)


def check_disk_space(
    output_parent: Path,
    shard_paths: list[Path],
    *,
    copy_mode: str,
    skip_space_check: bool,
) -> None:
    if skip_space_check:
        return
    same_device = all(
        path.stat().st_dev == output_parent.stat().st_dev for path in shard_paths
    )
    bytes_needed = 1024**3
    if copy_mode == "copy" or (copy_mode == "auto" and not same_device):
        bytes_needed += sum(path.stat().st_size for path in shard_paths)
    free_bytes = shutil.disk_usage(output_parent).free
    if free_bytes < bytes_needed:
        raise OSError(
            f"Insufficient output space: need about {bytes_needed / 1024**3:.1f} "
            f"GiB, have {free_bytes / 1024**3:.1f} GiB"
        )


def export_static_checkpoint(
    input_dir: Path,
    calibration_path: Path,
    output_dir: Path,
    *,
    candidate: str,
    safety_factor: float,
    copy_mode: str,
    skip_space_check: bool,
    dry_run: bool = False,
) -> dict[str, Any]:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    calibration_path = resolve_calibration_path(calibration_path)
    if not math.isfinite(safety_factor) or safety_factor <= 0:
        raise ValueError("safety_factor must be finite and positive")
    if input_dir == output_dir:
        raise ValueError("Input and output transformer directories must differ")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    config_path = input_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing transformer config: {config_path}")
    source_config = load_json(config_path)
    num_layers = validate_source_config(source_config)
    expected_weights = expected_linear_weight_names(num_layers)
    expected_prefixes = {name.removesuffix(".weight") for name in expected_weights}
    shard_paths, weight_map, index_name, source_index = discover_checkpoint(input_dir)
    validate_source_weights(weight_map, expected_weights)
    weight_shapes = read_weight_shapes(shard_paths, expected_weights)
    calibration = load_json(calibration_path)
    scale_tensors, module_plan = build_scale_plan(
        calibration,
        candidate=candidate,
        safety_factor=safety_factor,
        expected_prefixes=expected_prefixes,
        weight_shapes=weight_shapes,
    )

    input_scale_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in scale_tensors.values()
    )
    scale_values = torch.stack(list(scale_tensors.values())).float()
    manifest: dict[str, Any] = {
        "created_at": utc_now(),
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "source_transformer": str(input_dir),
        "output_transformer": str(output_dir),
        "calibration_file": str(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "completed_request_count": len(calibration["completed_requests"]),
        "num_layers": num_layers,
        "scale_candidate": candidate,
        "safety_factor": safety_factor,
        "runtime_linear_count": len(module_plan),
        "checkpoint_input_scale_count": len(scale_tensors),
        "input_scale_bytes": input_scale_bytes,
        "input_scale_min": float(scale_values.min().item()),
        "input_scale_max": float(scale_values.max().item()),
        "modules": module_plan,
    }
    if dry_run:
        manifest["dry_run"] = True
        return manifest

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    check_disk_space(
        output_dir.parent,
        shard_paths,
        copy_mode=copy_mode,
        skip_space_check=skip_space_check,
    )
    staging_dir = output_dir.parent / f".{output_dir.name}.export-{os.getpid()}"
    if staging_dir.exists():
        raise FileExistsError(f"Staging directory already exists: {staging_dir}")
    staging_dir.mkdir()
    try:
        shard_modes: dict[str, str] = {}
        for index, source in enumerate(shard_paths, start=1):
            print(
                f"[{index}/{len(shard_paths)}] materializing {source.name}", flush=True
            )
            shard_modes[source.name] = materialize_shard(
                source, staging_dir / source.name, copy_mode
            )

        static_quantization = dict(source_config["quantization_config"])
        static_quantization["activation_scheme"] = "static"
        metadata = {
            "format": "pt",
            "_quantization_metadata": json.dumps(
                static_quantization, separators=(",", ":"), sort_keys=True
            ),
            "_activation_calibration_metadata": json.dumps(
                {
                    "candidate": candidate,
                    "safety_factor": safety_factor,
                    "source_sha256": manifest["calibration_sha256"],
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
        }
        save_file(
            dict(sorted(scale_tensors.items())),
            staging_dir / SCALE_SHARD_NAME,
            metadata=metadata,
        )

        output_config = dict(source_config)
        output_config["quantization_config"] = static_quantization
        write_json(staging_dir / "config.json", output_config)
        copy_auxiliary_files(
            input_dir, staging_dir, {path.name for path in shard_paths}
        )

        output_weight_map = dict(weight_map)
        output_weight_map.update({name: SCALE_SHARD_NAME for name in scale_tensors})
        output_index_metadata = dict(source_index.get("metadata") or {})
        source_total_size = int(output_index_metadata.get("total_size", 0))
        output_index_metadata["total_size"] = source_total_size + input_scale_bytes
        write_json(
            staging_dir / index_name,
            {
                "metadata": output_index_metadata,
                "weight_map": dict(sorted(output_weight_map.items())),
            },
        )

        manifest["quantization_config"] = static_quantization
        manifest["scale_shard"] = SCALE_SHARD_NAME
        manifest["scale_shard_sha256"] = sha256_file(staging_dir / SCALE_SHARD_NAME)
        manifest["source_shard_materialization"] = shard_modes
        manifest["checkpoint_bytes"] = sum(
            path.stat().st_size for path in staging_dir.iterdir() if path.is_file()
        )
        write_json(staging_dir / "videoedit_static_activation_manifest.json", manifest)
        os.replace(staging_dir, output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export calibrated per-Linear static FP8 activation scales into an "
            "existing serialized VideoEdit FP8 weight checkpoint."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale-candidate", choices=SCALE_CANDIDATES, default="max")
    parser.add_argument("--safety-factor", type=float, default=1.05)
    parser.add_argument(
        "--copy-mode",
        choices=("auto", "hardlink", "copy"),
        default="auto",
        help="Use hardlinks for existing weight shards when possible.",
    )
    parser.add_argument("--skip-space-check", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate calibration coverage and tensor shapes without writing output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = export_static_checkpoint(
            args.input_dir,
            args.calibration,
            args.output_dir,
            candidate=args.scale_candidate,
            safety_factor=args.safety_factor,
            copy_mode=args.copy_mode,
            skip_space_check=args.skip_space_check,
            dry_run=args.dry_run,
        )
    except Exception as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1

    action = "validated" if args.dry_run else "completed"
    print(f"[{action}] static VideoEdit FP8 activation checkpoint")
    print(f"  calibration cases: {manifest['completed_request_count']}")
    print(f"  runtime Linears: {manifest['runtime_linear_count']}")
    print(f"  checkpoint input scales: {manifest['checkpoint_input_scale_count']}")
    print(
        "  exported scale range: "
        f"{manifest['input_scale_min']:.8g} .. {manifest['input_scale_max']:.8g}"
    )
    if not args.dry_run:
        print(f"  output: {manifest['output_transformer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
