#!/usr/bin/env python3
"""Export a Wan VideoEdit transformer as serialized per-channel FP8 weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_DTYPE = torch.float8_e4m3fn
FP8_INFO = torch.finfo(FP8_DTYPE)
FP8_MIN = FP8_INFO.min
FP8_MAX = FP8_INFO.max
FORMAT_NAME = "sglang-videoedit-fp8-channel"
FORMAT_VERSION = 1

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


def expected_linear_weight_names(num_layers: int) -> set[str]:
    names = set(CONDITION_LINEAR_WEIGHTS)
    for layer_idx in range(num_layers):
        names.update(f"blocks.{layer_idx}.{suffix}" for suffix in BLOCK_LINEAR_SUFFIXES)
    return names


def quantize_weight_per_channel(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError(
            f"FP8 Linear weight must be two-dimensional, got {tuple(weight.shape)}"
        )
    if weight.is_cuda:
        # Match Fp8LinearMethod.process_weights_after_loading bit-for-bit.
        from sglang.srt.layers.quantization.fp8_kernel import (
            per_token_group_quant_fp8,
        )

        qweight, scale = per_token_group_quant_fp8(weight, weight.shape[-1])
        return qweight.contiguous(), scale.reshape(-1).contiguous()

    weight_fp32 = weight.float()
    scale = weight_fp32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / FP8_MAX
    qweight = torch.clamp(weight_fp32 / scale, FP8_MIN, FP8_MAX).to(FP8_DTYPE)
    return qweight.contiguous(), scale.squeeze(1).contiguous()


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def discover_checkpoint(
    input_dir: Path,
) -> tuple[list[Path], dict[str, str], str]:
    index_paths = sorted(input_dir.glob("*.safetensors.index.json"))
    if len(index_paths) > 1:
        raise ValueError(
            f"Expected at most one safetensors index in {input_dir}, "
            f"found {len(index_paths)}"
        )

    if index_paths:
        index_path = index_paths[0]
        index = load_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid weight_map in {index_path}")
        filenames = sorted(set(weight_map.values()))
        shard_paths = [input_dir / filename for filename in filenames]
        index_name = index_path.name
    else:
        shard_paths = sorted(input_dir.glob("*.safetensors"))
        weight_map: dict[str, str] = {}
        for shard_path in shard_paths:
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in weight_map:
                        raise ValueError(f"Duplicate tensor {key} in input checkpoint")
                    weight_map[key] = shard_path.name
        index_name = "diffusion_pytorch_model.safetensors.index.json"

    if not shard_paths:
        raise ValueError(f"No safetensors shards found in {input_dir}")
    missing_shards = [str(path) for path in shard_paths if not path.is_file()]
    if missing_shards:
        raise FileNotFoundError(f"Missing checkpoint shards: {missing_shards}")
    return shard_paths, weight_map, index_name


def validate_source_config(config: dict[str, Any], input_dir: Path) -> int:
    class_name = config.get("_class_name")
    if class_name != "WanVideoEditTransformer3DModel":
        raise ValueError(
            "This exporter only supports WanVideoEditTransformer3DModel, got "
            f"{class_name!r} from {input_dir / 'config.json'}"
        )
    if config.get("quantization_config") is not None:
        raise ValueError("Input transformer is already quantized")
    num_layers = config.get("num_layers")
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError(f"Invalid num_layers in transformer config: {num_layers!r}")
    return num_layers


def validate_source_weights(
    weight_map: dict[str, str], expected_weights: set[str]
) -> None:
    available = set(weight_map)
    missing = sorted(expected_weights - available)
    if missing:
        examples = ", ".join(missing[:8])
        raise ValueError(
            f"Checkpoint is missing {len(missing)} expected VideoEdit Linear "
            f"weights (e.g. {examples})"
        )
    preexisting_scales = sorted(
        key for key in available if key.endswith(".weight_scale")
    )
    if preexisting_scales:
        raise ValueError(
            "Input checkpoint already contains FP8 weight scales: "
            f"{preexisting_scales[:8]}"
        )


def ensure_output_directory(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.iterdir())
    if existing:
        examples = ", ".join(path.name for path in sorted(existing)[:8])
        raise FileExistsError(
            f"Output directory must be empty: {output_dir} contains {examples}"
        )


def check_disk_space(
    output_dir: Path, source_bytes: int, *, skip_space_check: bool
) -> None:
    if skip_space_check:
        return
    free_bytes = shutil.disk_usage(output_dir).free
    estimated_bytes = int(source_bytes * 0.60) + 1024**3
    if free_bytes < estimated_bytes:
        raise OSError(
            "Insufficient free space for the FP8 checkpoint: "
            f"need approximately {estimated_bytes / 1024**3:.1f} GiB, "
            f"have {free_bytes / 1024**3:.1f} GiB. Use a different output "
            "filesystem or --skip-space-check after verifying capacity."
        )


def copy_auxiliary_files(input_dir: Path, output_dir: Path) -> None:
    for source in sorted(input_dir.iterdir()):
        if not source.is_file():
            continue
        if source.name == "config.json":
            continue
        if source.name.endswith(".safetensors"):
            continue
        if source.name.endswith(".safetensors.index.json"):
            continue
        shutil.copy2(source, output_dir / source.name)


def convert_checkpoint(
    input_dir: Path,
    output_dir: Path,
    *,
    device: str,
    compute_checksums: bool,
    skip_space_check: bool,
) -> dict[str, Any]:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if input_dir == output_dir:
        raise ValueError("Input and output directories must be different")

    config_path = input_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing transformer config: {config_path}")
    source_config = load_json(config_path)
    num_layers = validate_source_config(source_config, input_dir)
    expected_weights = expected_linear_weight_names(num_layers)

    shard_paths, source_weight_map, index_name = discover_checkpoint(input_dir)
    validate_source_weights(source_weight_map, expected_weights)
    ensure_output_directory(output_dir)
    source_shard_bytes = sum(path.stat().st_size for path in shard_paths)
    check_disk_space(output_dir, source_shard_bytes, skip_space_check=skip_space_check)

    target_device = torch.device(device)
    if target_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is false"
            )
        torch.cuda.set_device(target_device)

    quantization_config = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_scale_granularity": "channel",
    }
    safetensors_quant_metadata = json.dumps(
        quantization_config, separators=(",", ":"), sort_keys=True
    )

    output_weight_map: dict[str, str] = {}
    shard_records: list[dict[str, Any]] = []
    total_output_tensor_bytes = 0
    source_linear_bytes = 0
    fp8_linear_bytes = 0
    scale_bytes = 0
    quantized_names: set[str] = set()

    for shard_idx, source_path in enumerate(shard_paths, start=1):
        print(
            f"[{shard_idx}/{len(shard_paths)}] converting {source_path.name}",
            flush=True,
        )
        output_tensors: dict[str, torch.Tensor] = {}
        with safe_open(
            source_path, framework="pt", device=str(target_device)
        ) as source:
            metadata = dict(source.metadata() or {})
            for key in source.keys():
                tensor = source.get_tensor(key)
                if key in expected_weights:
                    qweight, scale = quantize_weight_per_channel(tensor)
                    scale_name = key.removesuffix(".weight") + ".weight_scale"
                    output_tensors[key] = qweight
                    output_tensors[scale_name] = scale
                    source_linear_bytes += tensor_bytes(tensor)
                    fp8_linear_bytes += tensor_bytes(qweight)
                    scale_bytes += tensor_bytes(scale)
                    quantized_names.add(key)
                else:
                    output_tensors[key] = tensor.contiguous()

        metadata["format"] = metadata.get("format", "pt")
        metadata["_quantization_metadata"] = safetensors_quant_metadata
        temporary_path = output_dir / f".{source_path.name}.tmp"
        output_path = output_dir / source_path.name
        save_file(output_tensors, temporary_path, metadata=metadata)
        os.replace(temporary_path, output_path)

        shard_tensor_bytes = sum(tensor_bytes(t) for t in output_tensors.values())
        total_output_tensor_bytes += shard_tensor_bytes
        for key in output_tensors:
            output_weight_map[key] = output_path.name
        shard_record: dict[str, Any] = {
            "file": output_path.name,
            "size_bytes": output_path.stat().st_size,
            "tensor_bytes": shard_tensor_bytes,
            "tensor_count": len(output_tensors),
        }
        if compute_checksums:
            print(f"[{shard_idx}/{len(shard_paths)}] checksumming", flush=True)
            shard_record["sha256"] = sha256_file(output_path)
        shard_records.append(shard_record)
        del output_tensors
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    missing_converted = expected_weights - quantized_names
    unexpected_converted = quantized_names - expected_weights
    if missing_converted or unexpected_converted:
        raise RuntimeError(
            "Converted weight set mismatch: "
            f"missing={sorted(missing_converted)[:8]}, "
            f"unexpected={sorted(unexpected_converted)[:8]}"
        )

    output_config = dict(source_config)
    output_config["quantization_config"] = quantization_config
    write_json_atomic(output_dir / "config.json", output_config)
    copy_auxiliary_files(input_dir, output_dir)

    index = {
        "metadata": {"total_size": total_output_tensor_bytes},
        "weight_map": dict(sorted(output_weight_map.items())),
    }
    write_json_atomic(output_dir / index_name, index)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "source_transformer": str(input_dir),
        "output_transformer": str(output_dir),
        "model_class": source_config["_class_name"],
        "num_layers": num_layers,
        "quantization_config": quantization_config,
        "quantization_backend": (
            "sglang.per_token_group_quant_fp8"
            if target_device.type == "cuda"
            else "torch.reference_absmax"
        ),
        "quantized_linear_weight_count": len(quantized_names),
        "source_checkpoint_bytes": source_shard_bytes,
        "source_linear_weight_bytes": source_linear_bytes,
        "fp8_linear_weight_bytes": fp8_linear_bytes,
        "weight_scale_bytes": scale_bytes,
        "output_tensor_bytes": total_output_tensor_bytes,
        "output_checkpoint_bytes": sum(
            record["size_bytes"] for record in shard_records
        ),
        "shards": shard_records,
    }
    write_json_atomic(output_dir / "videoedit_fp8_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(
        description=(
            "Export Wan VideoEdit BF16 Linear weights as serialized FP8 E4M3FN "
            "with per-output-channel scales."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=default_device)
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Do not calculate SHA256 for output shards.",
    )
    parser.add_argument(
        "--skip-space-check",
        action="store_true",
        help="Skip the conservative output filesystem capacity check.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = convert_checkpoint(
            args.input_dir,
            args.output_dir,
            device=args.device,
            compute_checksums=not args.skip_checksums,
            skip_space_check=args.skip_space_check,
        )
    except Exception as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1

    gib = 1024**3
    print("[completed] serialized VideoEdit FP8 checkpoint")
    print(f"  output: {manifest['output_transformer']}")
    print(f"  quantized Linear weights: {manifest['quantized_linear_weight_count']}")
    print(
        "  checkpoint size: "
        f"{manifest['source_checkpoint_bytes'] / gib:.2f} GiB -> "
        f"{manifest['output_checkpoint_bytes'] / gib:.2f} GiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
