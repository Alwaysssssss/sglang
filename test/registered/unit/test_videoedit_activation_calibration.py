from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from sglang.multimodal_gen.runtime.utils.activation_calibration import (
    FP8_E4M3_MAX,
    HistogramConfig,
    checkpoint_aliases_for_runtime_linear,
    histogram_percentile,
    merge_rank_calibration,
    summarize_activation_tensor,
)


def load_script_module():
    script_path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "videoedit_collect_activation_stats.py"
    )
    spec = importlib.util.spec_from_file_location(
        "videoedit_collect_activation_stats", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_checkpoint_aliases_expand_fused_projections():
    assert checkpoint_aliases_for_runtime_linear("blocks.0.to_qkv") == [
        "blocks.0.attn1.to_q",
        "blocks.0.attn1.to_k",
        "blocks.0.attn1.to_v",
    ]
    assert checkpoint_aliases_for_runtime_linear("blocks.0.attn2.to_kv") == [
        "blocks.0.attn2.to_k",
        "blocks.0.attn2.to_v",
    ]
    assert checkpoint_aliases_for_runtime_linear("blocks.0.attn2.to_added_kv") == [
        "blocks.0.attn2.add_k_proj",
        "blocks.0.attn2.add_v_proj",
    ]
    assert checkpoint_aliases_for_runtime_linear("blocks.0.to_out") == [
        "blocks.0.attn1.to_out.0"
    ]
    assert checkpoint_aliases_for_runtime_linear("blocks.0.ffn.fc_in") == [
        "blocks.0.ffn.net.0.proj"
    ]
    assert checkpoint_aliases_for_runtime_linear(
        "condition_embedder.time_embedder.mlp.fc_in"
    ) == ["condition_embedder.time_embedder.linear_1"]
    assert checkpoint_aliases_for_runtime_linear("proj_out") == ["proj_out"]
    with pytest.raises(ValueError, match="Unsupported VideoEdit runtime Linear"):
        checkpoint_aliases_for_runtime_linear("blocks.0.unknown")


def test_summarize_activation_tracks_token_amax_distribution():
    config = HistogramConfig(bins=8, log2_min=-2.0, log2_max=6.0)
    tensor = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, -2.0],
            [8.0, 4.0],
            [float("nan"), 1.0],
        ],
        dtype=torch.float32,
    )

    summary = summarize_activation_tensor(tensor, config)

    assert summary["token_count"] == 4
    assert summary["element_count"] == 8
    assert summary["input_features"] == 2
    assert int(summary["nonfinite_token_count"].item()) == 1
    assert int(summary["zero_token_count"].item()) == 1
    assert int(summary["histogram"].sum().item()) == 2
    assert float(summary["absmax"].item()) == 8.0


def test_histogram_percentile_uses_upper_bin_edge():
    threshold = histogram_percentile(
        torch.tensor([1, 2, 3, 4], dtype=torch.int64),
        zero_count=0,
        underflow_count=0,
        overflow_count=0,
        log2_min=0.0,
        log2_max=4.0,
        percentile=0.5,
        absmax=16.0,
    )
    assert threshold == 8.0


def write_rank_state(root: Path, rank: int) -> None:
    rank_dir = root / f"rank{rank}"
    rank_dir.mkdir(parents=True)
    stats_file = "stats.safetensors"
    save_file(
        {
            "histogram": torch.tensor(
                [[0, 2 + rank, 0, 0], [0, 0, 1 + rank, 0]],
                dtype=torch.int64,
            ),
            "absmax": torch.tensor([4.0 + rank, 8.0 + rank]),
            "nonfinite_token_count": torch.zeros(2, dtype=torch.int64),
            "zero_token_count": torch.tensor([1, 0], dtype=torch.int64),
            "underflow_token_count": torch.zeros(2, dtype=torch.int64),
            "overflow_token_count": torch.zeros(2, dtype=torch.int64),
            "token_count": torch.tensor([3 + rank, 1 + rank], dtype=torch.int64),
            "element_count": torch.tensor(
                [(3 + rank) * 4, (1 + rank) * 8], dtype=torch.int64
            ),
            "observation_count": torch.ones(2, dtype=torch.int64),
            "min_tokens_per_observation": torch.tensor(
                [3 + rank, 1 + rank], dtype=torch.int64
            ),
            "max_tokens_per_observation": torch.tensor(
                [3 + rank, 1 + rank], dtype=torch.int64
            ),
            "input_features": torch.tensor([4, 8], dtype=torch.int64),
        },
        rank_dir / stats_file,
    )
    (rank_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rank": rank,
                "module_names": ["blocks.0.to_qkv", "proj_out"],
                "histogram": {
                    "kind": "token_amax_log2",
                    "bins": 4,
                    "log2_min": 0.0,
                    "log2_max": 4.0,
                },
                "completed_requests": ["calib_case"],
                "failed_requests": [],
                "stats_file": stats_file,
            }
        ),
        encoding="utf-8",
    )


def test_merge_rank_calibration_writes_static_scale_candidates(tmp_path):
    collector_dir = tmp_path / "collector"
    write_rank_state(collector_dir, 0)
    write_rank_state(collector_dir, 1)
    output_dir = tmp_path / "merged"

    manifest = merge_rank_calibration(collector_dir, output_dir=output_dir)

    assert manifest["rank_count"] == 2
    assert manifest["module_count"] == 2
    assert manifest["completed_requests"] == ["calib_case"]
    tensors = load_file(output_dir / "activation_stats.safetensors")
    assert tensors["histogram"].tolist() == [[0, 5, 0, 0], [0, 0, 3, 0]]
    assert tensors["input_scale_max"][0].item() == pytest.approx(5.0 / FP8_E4M3_MAX)
    calibration = json.loads(
        (output_dir / "activation_calibration.json").read_text(encoding="utf-8")
    )
    assert calibration["quantization"]["symmetric"] is True
    assert calibration["quantization"]["zero_point"] is None
    assert calibration["modules"][0]["checkpoint_aliases"] == [
        "blocks.0.attn1.to_q",
        "blocks.0.attn1.to_k",
        "blocks.0.attn1.to_v",
    ]


def test_case_loader_uses_video_masks_and_caption_order(tmp_path):
    module = load_script_module()
    data_root = tmp_path / "erase_data_case"
    captions_dir = data_root / "caption_frames"
    videos_dir = data_root / "videos"
    masks_dir = data_root / "video_masks"
    captions_dir.mkdir(parents=True)
    videos_dir.mkdir()
    masks_dir.mkdir()
    (captions_dir / "captions.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"id": "2", "video": "videos/2.mp4", "caption": "second"}),
                json.dumps({"id": "1", "video": "videos/1.mp4", "caption": "first"}),
            ]
        ),
        encoding="utf-8",
    )
    for case_id in ("1", "2"):
        (videos_dir / f"{case_id}.mp4").touch()
        (masks_dir / f"{case_id}_mask.mp4").touch()
    args = argparse.Namespace(
        captions=captions_dir / "captions.jsonl",
        videos_dir=videos_dir,
        masks_dir=masks_dir,
        sample_id=[],
        max_samples=None,
    )

    cases = module.load_cases(args)

    assert [case.case_id for case in cases] == ["2", "1"]
    assert cases[0].mask == (masks_dir / "2_mask.mp4").resolve()
