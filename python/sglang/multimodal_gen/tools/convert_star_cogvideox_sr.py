"""Convert STAR CogVideoX-SR assets into an SGLang-native model directory."""

from __future__ import annotations

import argparse
import os
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import save_file as safetensors_save_file

from sglang.multimodal_gen.tools.star_cogvideox_keymap import (
    extract_transformer_state_dict,
    extract_vae_state_dict,
)
from sglang.multimodal_gen.tools.star_cogvideox_manifest import (
    ComponentExportRecord,
    ConversionReport,
    KeyMappingSummary,
    SourceAssetsManifest,
    describe_directory_files,
    describe_file,
    write_dataclass_json,
    write_json,
)

PIPELINE_CLASS_NAME = "StarCogVideoXSRPipeline"
TRANSFORMER_CLASS_NAME = "StarCogVideoXSRTransformer3DModel"
VAE_CLASS_NAME = "StarCogVideoXSRVAE"
SCHEDULER_CLASS_NAME = "StarVPSDEDPMPP2MScheduler"
INTEGRATION_CONFIG_NAME = "star_integration_config.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert STAR CogVideoX-SR checkpoints into an SGLang-native model "
            "directory layout."
        )
    )
    parser.add_argument("--src-transformer", type=str, required=True)
    parser.add_argument("--src-vae", type=str, default=None)
    parser.add_argument("--src-text-encoder", type=str, required=True)
    parser.add_argument("--src-tokenizer", type=str, default=None)
    parser.add_argument("--src-config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help="Skip SHA256 hashing in generated manifests for faster dry-runs.",
    )
    return parser.parse_args()


def _resolve_checkpoint_path(path: str, *, component_name: str) -> str:
    candidate = Path(path)
    if candidate.is_file():
        return str(candidate)
    if not candidate.is_dir():
        raise FileNotFoundError(f"{component_name} path does not exist: {path}")

    patterns = ("*.pt", "*.ckpt", "*.safetensors")
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(candidate.glob(pattern)))
    if len(matches) == 1:
        return str(matches[0])

    preferred = [
        candidate / "mp_rank_00_model_states.pt",
        candidate / "3d-vae.pt",
        candidate / "model.safetensors",
    ]
    for item in preferred:
        if item.exists():
            return str(item)

    raise FileNotFoundError(
        f"Could not resolve a unique checkpoint file for {component_name} under {path}"
    )


def _load_torch_checkpoint(path: str) -> object:
    return torch.load(path, map_location="cpu", weights_only=False)


def _ensure_output_dir(path: str, *, overwrite: bool) -> None:
    output_dir = Path(path)
    if output_dir.exists():
        if not overwrite and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory already exists and is not empty: {path}. "
                "Use --overwrite to replace it."
            )
        if overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _link_or_copy_file(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src_path, dst_path)
    except OSError:
        shutil.copy2(src_path, dst_path)


def _copy_dir_subset(src_dir: str, dst_dir: str, include_names: set[str] | None = None) -> list[str]:
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for entry in sorted(src.iterdir()):
        if include_names is not None and entry.name not in include_names:
            continue
        if entry.is_file():
            _link_or_copy_file(entry, dst / entry.name)
            copied.append(entry.name)
    return copied


def _read_yaml_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path, encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML root dict in {path}")
    return payload


def _build_model_index(tokenizer_class_name: str) -> dict[str, Any]:
    return {
        "_class_name": PIPELINE_CLASS_NAME,
        "_diffusers_version": "0.0.0",
        "transformer": ["diffusers", TRANSFORMER_CLASS_NAME],
        "vae": ["diffusers", VAE_CLASS_NAME],
        "text_encoder": ["transformers", "T5EncoderModel"],
        "tokenizer": ["transformers", tokenizer_class_name],
        "scheduler": ["diffusers", SCHEDULER_CLASS_NAME],
    }


def _infer_tokenizer_class_name(tokenizer_dir: str) -> str:
    tokenizer_config = Path(tokenizer_dir) / "tokenizer_config.json"
    if tokenizer_config.exists():
        import json

        with open(tokenizer_config, encoding="utf-8") as f:
            payload = json.load(f)
        class_name = payload.get("tokenizer_class")
        if isinstance(class_name, str) and class_name:
            return class_name
    return "T5Tokenizer"


def _build_transformer_component_config(source_cfg: dict[str, Any]) -> dict[str, Any]:
    model_cfg = source_cfg.get("model", {})
    network = model_cfg.get("network_config", {})
    params = dict(network.get("params", {}) or {})
    params["_class_name"] = TRANSFORMER_CLASS_NAME
    params["_source_format"] = "star-cogvideox-sat"
    params["_source_target"] = network.get("target")
    params["source_key_prefix"] = "model.diffusion_model."
    return params


def _get_transformer_lora_alpha(source_cfg: dict[str, Any]) -> float | None:
    params = (
        source_cfg.get("model", {})
        .get("network_config", {})
        .get("params", {})
        .get("modules", {})
        .get("lora_config", {})
        .get("params", {})
    )
    value = params.get("lora_alpha", params.get("alpha"))
    if value is None:
        return None
    return float(value)


def _build_vae_component_config(source_cfg: dict[str, Any]) -> dict[str, Any]:
    model_cfg = source_cfg.get("model", {})
    first_stage = model_cfg.get("first_stage_config", {})
    params = dict(first_stage.get("params", {}) or {})
    params["_class_name"] = VAE_CLASS_NAME
    params["_source_format"] = "star-cogvideox-sat"
    params["_source_target"] = first_stage.get("target")
    params["source_key_prefix"] = "first_stage_model."
    params["latent_scale_factor"] = model_cfg.get("scale_factor")
    if "ckpt_path" in params:
        params.pop("ckpt_path")
    return params


def _build_scheduler_component_config(source_cfg: dict[str, Any]) -> dict[str, Any]:
    sampler_cfg = source_cfg.get("model", {}).get("sampler_config", {})
    params = dict(sampler_cfg.get("params", {}) or {})
    params["_class_name"] = SCHEDULER_CLASS_NAME
    params["_source_format"] = "star-cogvideox-sat"
    params["_source_target"] = sampler_cfg.get("target")
    return params


def _build_integration_config(source_cfg: dict[str, Any]) -> dict[str, Any]:
    args_cfg = source_cfg.get("args", {})
    model_cfg = source_cfg.get("model", {})
    network_params = model_cfg.get("network_config", {}).get("params", {}) or {}
    guider_params = (
        model_cfg.get("sampler_config", {})
        .get("params", {})
        .get("guider_config", {})
        .get("params", {})
        or {}
    )
    return {
        "pipeline_class_name": PIPELINE_CLASS_NAME,
        "pipeline_config_class_name": "StarCogVideoXSRPipelineConfig",
        "sampling_params_class_name": "StarCogVideoXSRSamplingParams",
        "transformer_class_name": TRANSFORMER_CLASS_NAME,
        "vae_class_name": VAE_CLASS_NAME,
        "scheduler_class_name": SCHEDULER_CLASS_NAME,
        "source_format": "star-cogvideox-sat",
        "latent_scale_factor": model_cfg.get("scale_factor"),
        "default_sampling_num_frames": args_cfg.get("sampling_num_frames"),
        "default_num_inference_steps": model_cfg.get("sampler_config", {})
        .get("params", {})
        .get("num_steps"),
        "dynamic_cfg_exp": guider_params.get("exp"),
        "latent_channels": args_cfg.get("latent_channels"),
        "transformer_summary": {
            "hidden_size": network_params.get("hidden_size"),
            "num_layers": network_params.get("num_layers"),
            "num_attention_heads": network_params.get("num_attention_heads"),
            "patch_size": network_params.get("patch_size"),
            "latent_width": network_params.get("latent_width"),
            "latent_height": network_params.get("latent_height"),
            "time_compressed_rate": network_params.get("time_compressed_rate"),
        },
    }


def _save_safetensors(path: str, state_dict: "OrderedDict[str, torch.Tensor]") -> tuple[int, int]:
    tensors = {key: tensor for key, tensor in state_dict.items()}
    safetensors_save_file(tensors, path)
    parameter_count = int(sum(t.numel() for t in tensors.values()))
    return len(tensors), parameter_count


def _inspect_source_assets(args: argparse.Namespace) -> tuple[dict[str, str], SourceAssetsManifest]:
    transformer_path = _resolve_checkpoint_path(
        args.src_transformer, component_name="transformer"
    )
    vae_path = (
        _resolve_checkpoint_path(args.src_vae, component_name="vae")
        if args.src_vae is not None
        else None
    )
    text_encoder_dir = str(Path(args.src_text_encoder))
    tokenizer_dir = str(Path(args.src_tokenizer or args.src_text_encoder))
    config_path = str(Path(args.src_config)) if args.src_config is not None else None

    for directory, name in (
        (text_encoder_dir, "text_encoder"),
        (tokenizer_dir, "tokenizer"),
    ):
        if not Path(directory).is_dir():
            raise FileNotFoundError(f"{name} directory does not exist: {directory}")

    include_hash = not args.skip_hashes
    manifest = SourceAssetsManifest(
        transformer_checkpoint=describe_file(transformer_path, include_hash=include_hash),
        vae_checkpoint=(
            describe_file(vae_path, include_hash=include_hash) if vae_path else None
        ),
        text_encoder_dir=text_encoder_dir,
        tokenizer_dir=tokenizer_dir,
        text_encoder_files=describe_directory_files(
            text_encoder_dir, include_hash=include_hash
        ),
        tokenizer_files=describe_directory_files(
            tokenizer_dir, include_hash=include_hash
        ),
        config_path=describe_file(config_path, include_hash=include_hash)
        if config_path
        else None,
    )
    return {
        "transformer_path": transformer_path,
        "vae_path": vae_path,
        "text_encoder_dir": text_encoder_dir,
        "tokenizer_dir": tokenizer_dir,
        "config_path": config_path,
    }, manifest


def run_conversion(args: argparse.Namespace) -> ConversionReport:
    source_paths, source_manifest = _inspect_source_assets(args)
    output_dir = str(Path(args.output_dir))
    report = ConversionReport(
        source_format="star-cogvideox-sat",
        output_dir=output_dir,
        pipeline_class_name=PIPELINE_CLASS_NAME,
        notes=[],
    )

    source_cfg = _read_yaml_config(source_paths["config_path"])
    tokenizer_class_name = _infer_tokenizer_class_name(source_paths["tokenizer_dir"])

    if args.dry_run:
        report.notes.append("Dry run only; no files were written.")
        report.notes.append(
            "Transformer/VAE configs were inferred from source YAML when provided."
        )
        report.key_mapping.append(
            KeyMappingSummary(
                component_name="transformer",
                source_key_count=0,
                exported_key_count=0,
            )
        )
        return report

    _ensure_output_dir(output_dir, overwrite=args.overwrite)
    output_root = Path(output_dir)
    manifests_dir = output_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    transformer_ckpt = _load_torch_checkpoint(source_paths["transformer_path"])
    transformer_extracted = extract_transformer_state_dict(
        transformer_ckpt,
        lora_alpha=_get_transformer_lora_alpha(source_cfg),
    )
    transformer_dir = output_root / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    transformer_tensors, transformer_params = _save_safetensors(
        str(transformer_dir / "model.safetensors"), transformer_extracted.state_dict
    )
    write_json(
        transformer_dir / "config.json",
        _build_transformer_component_config(source_cfg),
    )
    report.components.append(
        ComponentExportRecord(
            component_name="transformer",
            output_dir=str(transformer_dir),
            output_files=["config.json", "model.safetensors"],
            tensor_count=transformer_tensors,
            parameter_count=transformer_params,
        )
    )
    report.key_mapping.append(
        KeyMappingSummary(
            component_name="transformer",
            source_key_count=transformer_extracted.source_key_count,
            exported_key_count=len(transformer_extracted.state_dict),
            stripped_prefixes=transformer_extracted.stripped_prefixes,
            dropped_key_prefixes=transformer_extracted.dropped_key_prefixes,
            dropped_key_count=transformer_extracted.dropped_key_count,
        )
    )

    vae_source_path = source_paths["vae_path"] or source_paths["transformer_path"]
    vae_ckpt = _load_torch_checkpoint(vae_source_path)
    vae_extracted = extract_vae_state_dict(
        vae_ckpt, allow_embedded_prefix=source_paths["vae_path"] is None
    )
    vae_dir = output_root / "vae"
    vae_dir.mkdir(parents=True, exist_ok=True)
    vae_tensors, vae_params = _save_safetensors(
        str(vae_dir / "model.safetensors"), vae_extracted.state_dict
    )
    write_json(
        vae_dir / "config.json",
        _build_vae_component_config(source_cfg),
    )
    report.components.append(
        ComponentExportRecord(
            component_name="vae",
            output_dir=str(vae_dir),
            output_files=["config.json", "model.safetensors"],
            tensor_count=vae_tensors,
            parameter_count=vae_params,
            notes=(
                ["VAE exported from dedicated VAE checkpoint."]
                if source_paths["vae_path"]
                else ["VAE exported from embedded first_stage_model weights."]
            ),
        )
    )
    report.key_mapping.append(
        KeyMappingSummary(
            component_name="vae",
            source_key_count=vae_extracted.source_key_count,
            exported_key_count=len(vae_extracted.state_dict),
            stripped_prefixes=vae_extracted.stripped_prefixes,
            dropped_key_prefixes=vae_extracted.dropped_key_prefixes,
            dropped_key_count=vae_extracted.dropped_key_count,
        )
    )

    text_encoder_dir = output_root / "text_encoder"
    text_encoder_files = _copy_dir_subset(
        source_paths["text_encoder_dir"],
        str(text_encoder_dir),
    )
    report.components.append(
        ComponentExportRecord(
            component_name="text_encoder",
            output_dir=str(text_encoder_dir),
            output_files=text_encoder_files,
            notes=["Copied from source HuggingFace-style text encoder directory."],
        )
    )

    tokenizer_dir = output_root / "tokenizer"
    tokenizer_include = {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "spiece.model",
        "added_tokens.json",
    }
    tokenizer_files = _copy_dir_subset(
        source_paths["tokenizer_dir"],
        str(tokenizer_dir),
        include_names=tokenizer_include,
    )
    report.components.append(
        ComponentExportRecord(
            component_name="tokenizer",
            output_dir=str(tokenizer_dir),
            output_files=tokenizer_files,
        )
    )

    scheduler_dir = output_root / "scheduler"
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        scheduler_dir / "scheduler_config.json",
        _build_scheduler_component_config(source_cfg),
    )
    report.components.append(
        ComponentExportRecord(
            component_name="scheduler",
            output_dir=str(scheduler_dir),
            output_files=["scheduler_config.json"],
        )
    )

    write_json(output_root / "model_index.json", _build_model_index(tokenizer_class_name))
    write_json(
        output_root / INTEGRATION_CONFIG_NAME,
        _build_integration_config(source_cfg),
    )

    write_dataclass_json(manifests_dir / "source_assets.json", source_manifest)
    write_dataclass_json(manifests_dir / "conversion_report.json", report)
    write_json(
        manifests_dir / "key_mapping_report.json",
        {
            "items": [
                {
                    "component_name": item.component_name,
                    "source_key_count": item.source_key_count,
                    "exported_key_count": item.exported_key_count,
                    "stripped_prefixes": item.stripped_prefixes,
                    "dropped_key_prefixes": item.dropped_key_prefixes,
                    "dropped_key_count": item.dropped_key_count,
                }
                for item in report.key_mapping
            ]
        },
    )
    return report


def main() -> None:
    args = _parse_args()
    report = run_conversion(args)
    print(
        f"STAR conversion prepared for pipeline {report.pipeline_class_name} "
        f"at {report.output_dir}"
    )
    for note in report.notes:
        print(f"- {note}")


if __name__ == "__main__":
    main()
