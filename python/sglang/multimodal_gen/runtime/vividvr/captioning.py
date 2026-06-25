# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Any

from sglang.multimodal_gen.configs.pipeline_configs.vividvr import VividVRPipelineConfig
from sglang.multimodal_gen.configs.sample.vividvr import VividVRSamplingParams
from sglang.multimodal_gen.runtime.vividvr.preprocess import (
    compose_positive_prompt,
    read_prompt_file,
    resolve_negative_prompt,
    resolve_prompt_file_path,
)


def resolve_caption_file_path(params: VividVRSamplingParams) -> str:
    if params.caption_file_path in (None, ""):
        raise ValueError("VividVR caption_file_path is not configured")
    return str(Path(params.caption_file_path).expanduser())


def read_caption_file(caption_file_path: str) -> list[str]:
    path = Path(caption_file_path).expanduser()
    caption_texts = [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not caption_texts:
        raise ValueError(f"Caption file is empty: {path}")
    return caption_texts


def _direct_prompt_file_caption_context(
    params: VividVRSamplingParams,
    pipeline_config: VividVRPipelineConfig,
) -> dict[str, str]:
    prompt_file_path = resolve_prompt_file_path(params, pipeline_config)
    prompt_text = read_prompt_file(prompt_file_path)
    return {
        "prompt_file_path": prompt_file_path,
        "prompt_text": prompt_text,
        "model_prompt_text": compose_positive_prompt(prompt_text, pipeline_config),
        "negative_prompt_text": resolve_negative_prompt(params, pipeline_config),
        "caption_backend": "prompt_file",
    }


def _caption_file_context(
    params: VividVRSamplingParams,
    pipeline_config: VividVRPipelineConfig,
) -> dict[str, Any]:
    caption_file_path = resolve_caption_file_path(params)
    caption_texts = read_caption_file(caption_file_path)
    prompt_file_path = None
    if (
        params.prompt_file_path is not None
        or params.prompt_path is not None
        or pipeline_config.default_prompt_file_path is not None
    ):
        prompt_file_path = resolve_prompt_file_path(params, pipeline_config)

    return {
        "prompt_file_path": prompt_file_path,
        "caption_file_path": caption_file_path,
        "caption_texts": caption_texts,
        "prompt_text": caption_texts[0],
        "model_prompt_text": compose_positive_prompt(caption_texts[0], pipeline_config),
        "negative_prompt_text": resolve_negative_prompt(params, pipeline_config),
        "caption_backend": "caption_file",
    }


def prepare_vividvr_prompt_context(
    params: VividVRSamplingParams,
    pipeline_config: VividVRPipelineConfig,
    *,
    debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if params.caption_source == "caption_file":
        return _caption_file_context(params, pipeline_config)
    if not params.enable_optional_caption_module:
        return _direct_prompt_file_caption_context(params, pipeline_config)

    try:
        return _direct_prompt_file_caption_context(params, pipeline_config)
    except Exception as exc:
        if not params.allow_optional_module_fallback:
            raise

        if debug is not None:
            warnings = debug.setdefault("optional_module_warnings", [])
            warnings.append(f"caption_module_fallback: {exc}")
        return _direct_prompt_file_caption_context(params, pipeline_config)


def build_vividvr_caption_prompt_lists(
    *,
    caption_texts: list[str],
    start_index: int,
    tile_count: int,
    negative_prompt_text: str | None,
    pipeline_config: VividVRPipelineConfig,
) -> dict[str, Any]:
    if tile_count <= 0:
        raise ValueError(f"tile_count must be positive, got {tile_count}")
    if start_index < 0:
        raise ValueError(f"start_index must be non-negative, got {start_index}")
    end_index = start_index + tile_count
    if end_index > len(caption_texts):
        raise ValueError(
            "caption file does not contain enough entries for the requested tiles: "
            f"need {tile_count} entries starting at {start_index}, "
            f"but only {len(caption_texts)} captions are available"
        )

    raw_caption_texts = caption_texts[start_index:end_index]
    prompt_list = [
        compose_positive_prompt(caption_text, pipeline_config)
        for caption_text in raw_caption_texts
    ]
    negative_prompt_list = None
    if negative_prompt_text is not None:
        negative_prompt_list = [negative_prompt_text for _ in range(tile_count)]
    return {
        "caption_texts": raw_caption_texts,
        "prompt_list": prompt_list,
        "negative_prompt_list": negative_prompt_list,
        "next_index": end_index,
    }


def build_vividvr_tiled_prompt_lists(
    *,
    model_prompt_text: str,
    negative_prompt_text: str | None,
    tile_count: int,
) -> dict[str, list[str] | None]:
    if tile_count <= 0:
        raise ValueError(f"tile_count must be positive, got {tile_count}")
    if not model_prompt_text:
        raise ValueError("model_prompt_text must not be empty")

    prompt_list = [model_prompt_text for _ in range(tile_count)]
    negative_prompt_list = None
    if negative_prompt_text is not None:
        negative_prompt_list = [negative_prompt_text for _ in range(tile_count)]
    return {
        "prompt_list": prompt_list,
        "negative_prompt_list": negative_prompt_list,
    }
