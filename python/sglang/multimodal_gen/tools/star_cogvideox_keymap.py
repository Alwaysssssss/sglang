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


def extract_transformer_state_dict(checkpoint_obj: object) -> ExtractedStateDict:
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
