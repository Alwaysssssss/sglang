"""Key extraction and normalization helpers for STAR CogVideoX-SR conversion."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable

import torch

TRANSFORMER_PREFIX = "model.diffusion_model."
EMBEDDED_VAE_PREFIX = "first_stage_model."
VAE_IGNORED_PREFIXES = ("loss.",)
TRANSFORMER_DROPPED_PREFIXES = ("first_stage_model.", "conditioner.")


@dataclass
class ExtractedStateDict:
    state_dict: "OrderedDict[str, torch.Tensor]"
    source_key_count: int
    stripped_prefixes: list[str] = field(default_factory=list)
    dropped_key_prefixes: list[str] = field(default_factory=list)
    dropped_key_count: int = 0


def _merge_lora_weights(
    state_dict: "OrderedDict[str, torch.Tensor]",
    *,
    lora_alpha: float | None = None,
) -> "OrderedDict[str, torch.Tensor]":
    merged = OrderedDict(state_dict)
    grouped: dict[str, dict[int, dict[str, torch.Tensor]]] = {}

    for key, tensor in state_dict.items():
        if ".matrix_A." in key:
            base, index_str = key.split(".matrix_A.", 1)
            grouped.setdefault(base, {}).setdefault(int(index_str), {})["A"] = tensor
        elif ".matrix_B." in key:
            base, index_str = key.split(".matrix_B.", 1)
            grouped.setdefault(base, {}).setdefault(int(index_str), {})["B"] = tensor

    for base, partition_map in grouped.items():
        original_key = f"{base}.original.weight"
        original_weight = state_dict.get(original_key)
        if original_weight is None:
            raise ValueError(
                f"STAR LoRA merge expected base weight {original_key}, but it was missing."
            )

        deltas: list[torch.Tensor] = []
        consumed_keys: list[str] = []
        for index in sorted(partition_map):
            tensors = partition_map[index]
            matrix_a = tensors.get("A")
            matrix_b = tensors.get("B")
            if matrix_a is None or matrix_b is None:
                raise ValueError(
                    f"STAR LoRA merge found incomplete pair for {base} partition {index}."
                )
            rank = int(matrix_a.shape[0])
            scaling = (float(lora_alpha) / rank) if lora_alpha is not None else (1.0 / rank)
            delta = (matrix_b.float() @ matrix_a.float()) * scaling
            deltas.append(delta)
            consumed_keys.extend(
                [f"{base}.matrix_A.{index}", f"{base}.matrix_B.{index}"]
            )

        merged_weight = original_weight.float() + torch.cat(deltas, dim=0)
        merged[original_key] = merged_weight.to(original_weight.dtype).contiguous()
        for key in consumed_keys:
            merged.pop(key, None)

    return merged


def _to_plain_state_dict(obj: object) -> dict[str, torch.Tensor]:
    """Normalize common checkpoint payloads to a plain state dict."""
    if isinstance(obj, dict):
        if "module" in obj and isinstance(obj["module"], (dict, OrderedDict)):
            return dict(obj["module"])
        if "state_dict" in obj and isinstance(obj["state_dict"], (dict, OrderedDict)):
            return dict(obj["state_dict"])
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return dict(obj)
    raise ValueError(
        "Unsupported checkpoint format. Expected a state_dict, a Lightning "
        "'state_dict' payload, or a DeepSpeed-style 'module' payload."
    )


def _strip_prefix_if_present(key: str, prefix: str) -> tuple[str, bool]:
    if key.startswith(prefix):
        return key[len(prefix) :], True
    return key, False


def _filter_tensor_items(items: Iterable[tuple[str, object]]) -> list[tuple[str, torch.Tensor]]:
    filtered: list[tuple[str, torch.Tensor]] = []
    for key, value in items:
        if isinstance(value, torch.Tensor):
            filtered.append((key, value.detach().cpu().contiguous()))
    return filtered


def extract_transformer_state_dict(
    checkpoint_obj: object, *, lora_alpha: float | None = None
) -> ExtractedStateDict:
    """Extract and normalize STAR transformer weights from a full checkpoint."""
    raw_state = _to_plain_state_dict(checkpoint_obj)
    tensor_items = _filter_tensor_items(raw_state.items())
    exported = OrderedDict()
    dropped_key_count = 0
    stripped = set()

    for key, value in tensor_items:
        if key.startswith(TRANSFORMER_PREFIX):
            new_key, did_strip = _strip_prefix_if_present(key, TRANSFORMER_PREFIX)
            if did_strip:
                stripped.add(TRANSFORMER_PREFIX)
            exported[new_key] = value
        else:
            dropped_key_count += 1

    exported = _merge_lora_weights(exported, lora_alpha=lora_alpha)

    return ExtractedStateDict(
        state_dict=exported,
        source_key_count=len(tensor_items),
        stripped_prefixes=sorted(stripped),
        dropped_key_prefixes=list(TRANSFORMER_DROPPED_PREFIXES),
        dropped_key_count=dropped_key_count,
    )


def extract_vae_state_dict(
    checkpoint_obj: object, *, allow_embedded_prefix: bool = True
) -> ExtractedStateDict:
    """Extract and normalize STAR VAE weights from a VAE-only or full checkpoint."""
    raw_state = _to_plain_state_dict(checkpoint_obj)
    tensor_items = _filter_tensor_items(raw_state.items())
    exported = OrderedDict()
    dropped_key_count = 0
    stripped = set()

    for key, value in tensor_items:
        if any(key.startswith(prefix) for prefix in VAE_IGNORED_PREFIXES):
            dropped_key_count += 1
            continue

        if allow_embedded_prefix and key.startswith(EMBEDDED_VAE_PREFIX):
            new_key, did_strip = _strip_prefix_if_present(key, EMBEDDED_VAE_PREFIX)
            if did_strip:
                stripped.add(EMBEDDED_VAE_PREFIX)
            key = new_key

        if any(key.startswith(prefix) for prefix in VAE_IGNORED_PREFIXES):
            dropped_key_count += 1
            continue

        exported[key] = value

    return ExtractedStateDict(
        state_dict=exported,
        source_key_count=len(tensor_items),
        stripped_prefixes=sorted(stripped),
        dropped_key_prefixes=list(VAE_IGNORED_PREFIXES),
        dropped_key_count=dropped_key_count,
    )
