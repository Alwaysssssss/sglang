# SPDX-License-Identifier: Apache-2.0
"""诊断 Connector cross-attention 中本地 vs 远端 control 的注意力分布。

在 v2 (eager_global) 模式下，每个 Connector 的 K/V 是 all-gather 后的全局 control states。
本脚本采样分析：Q 对本地 control (前 13500 token) vs 远端 control (后 13500 token) 的
注意力比例，以判断 v2 的全局 K/V 到底带来了多少信息增益。

用法:
    cd /home/zhiheng/sglang && export PYTHONPATH=python && \
    export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global && \
    /home/zhiheng/sglang/.venv/bin/torchrun --nproc_per_node=2 --master_port=30101 \
    python/sglang/multimodal_gen/tools/diag_connector_attention.py

输出: Vivid_Acceptance/indicator/diag_connector_attention_<run_id>.json
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.configs.pipeline_configs.vividvr import VividVRPipelineConfig
from sglang.multimodal_gen.configs.sample.vividvr import VividVRSamplingParams
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_common import (
    Connector,
    get_vividvr_connector_sp_context_mode,
    unpack_vividvr_connector_context,
)
from sglang.multimodal_gen.runtime.pipelines_core import build_pipeline
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args

VIVIDVR_ROOT = Path("/home/zhiheng/Vivid-VR")
COGVIDEOX_ROOT = VIVIDVR_ROOT / "ckpts" / "CogVideoX1.5-5B"
VIVIDVR_CKPT_ROOT = VIVIDVR_ROOT / "ckpts" / "Vivid-VR"
INPUT_VIDEO = VIVIDVR_ROOT / "input" / "720p_long" / "test_video_long_960x720_130f.mp4"
PROMPT_FILE = VIVIDVR_ROOT / "input" / "720p" / "prompt.txt"
ACCEPTANCE_ROOT = Path("/home/zhiheng/sglang/Vivid_Acceptance")
INDICATOR_DIR = ACCEPTANCE_ROOT / "indicator"

# 每 connector 采样参数
SAMPLE_Q = 128  # 随机采样的 query position 数
SAMPLE_H = 4    # 随机采样的 head 数

# 全局统计收集
_stats: list[dict[str, Any]] = []


def _capture_attention_stats(
    connector_idx: int,
    mod: Connector,
    q: torch.Tensor,
    k: torch.Tensor,
    local_len: int,
) -> None:
    """从 Q, K 采样计算本地 vs 远端 attention 分布。"""
    if q.shape[0] != 1:
        q = q[:1]
        k = k[:1]

    seq_len = q.shape[1]
    global_len = k.shape[1]
    num_heads = q.shape[2]

    # 采样 query positions
    rng_q = random.Random(42 + connector_idx * 100)
    if seq_len <= SAMPLE_Q:
        q_indices = list(range(seq_len))
    else:
        q_indices = sorted(rng_q.sample(range(seq_len), SAMPLE_Q))

    # 采样 heads
    rng_h = random.Random(200 + connector_idx * 100)
    head_indices = sorted(
        rng_h.sample(range(num_heads), min(SAMPLE_H, num_heads))
    )

    q_sample = q[:, q_indices, :, :][:, :, head_indices, :]  # [1, Qs, Hs, D]
    k_sample = k[:, :, head_indices, :]                        # [1, Kg, Hs, D]

    # attention logits: [Hs, Qs, Kg]
    logits = torch.einsum("bqhd,bkhd->hbqk", q_sample.float(), k_sample.float())
    logits = logits.squeeze(1) / (q.shape[-1] ** 0.5)
    attn = F.softmax(logits, dim=-1)  # [Hs, Qs, Kg]

    # 分割 local (0:local_len) vs remote (local_len:)
    attn_local = attn[:, :, :local_len].sum(dim=-1)    # [Hs, Qs]
    attn_remote = attn[:, :, local_len:].sum(dim=-1)   # [Hs, Qs]

    remote_ratio = attn_remote / (attn_local + attn_remote + 1e-8)

    # 逐层平均（先对 heads 平均，再对 positions）
    mean_across_heads = remote_ratio.mean(dim=0)  # [Qs]

    # 计算每 head 的统计
    for hi, hidx in enumerate(head_indices):
        _stats.append({
            "connector_index": connector_idx,
            "head_index": hidx,
            "local_seq_len": local_len,
            "global_seq_len": global_len,
            "query_seq_len": seq_len,
            "attn_to_local": float(attn_local[hi].mean()),
            "attn_to_remote": float(attn_remote[hi].mean()),
            "remote_ratio": float(remote_ratio[hi].mean()),
        })

    # 逐 position 统计（仅存关键信息）
    sorted_ratios, _ = mean_across_heads.sort()
    n = len(sorted_ratios)
    _stats.append({
        "connector_index": connector_idx,
        "head_index": -1,  # aggregated
        "local_seq_len": local_len,
        "global_seq_len": global_len,
        "query_seq_len": seq_len,
        "remote_ratio_p10": float(sorted_ratios[int(n * 0.1)]),
        "remote_ratio_p50": float(sorted_ratios[int(n * 0.5)]),
        "remote_ratio_p90": float(sorted_ratios[int(n * 0.9)]),
        "remote_ratio_mean": float(mean_across_heads.mean()),
        "remote_ratio_max": float(mean_across_heads.max()),
        "remote_ratio_min": float(mean_across_heads.min()),
    })


# ---- Hook Connector.forward ----
_original_connector_forward = Connector.forward


def _hooked_connector_forward(self: Connector, c, h: torch.Tensor) -> torch.Tensor:
    import torch.distributed as dist

    local_control, global_control = unpack_vividvr_connector_context(c)

    # ---- 正常计算 Q, K ----
    batch_size, seq_len, hidden_size = h.shape
    q = self.to_q(h).view(
        batch_size, seq_len, self.num_attention_heads, self.attention_head_dim
    )
    k = self.to_k(global_control).view(
        batch_size, global_control.shape[1],
        self.num_attention_heads, self.attention_head_dim,
    )
    q = self.norm_q(q)
    k = self.norm_k(k)

    # ---- 诊断：采样分析 attention 分布 ----
    if not getattr(torch.compiler, "is_compiling", lambda: False)():
        idx = getattr(self, "_diag_connector_idx", -1)
        if idx >= 0 and local_control.shape[1] < global_control.shape[1]:
            _capture_attention_stats(idx, self, q, k, local_control.shape[1])

    # ---- 调用原始 forward ----
    return _original_connector_forward(self, c, h)


def _install_diag_hooks(pipeline) -> int:
    """在所有 Connector 模块上安装诊断 hook。返回找到的 Connector 数量。"""
    count = 0
    transformer = pipeline.get_module("transformer")
    if transformer is None:
        return 0
    for module in transformer.modules():
        if isinstance(module, Connector):
            module._diag_connector_idx = count
            module.forward = _hooked_connector_forward.__get__(module, Connector)
            count += 1
    return count


def _restore_connector_forwards(pipeline) -> None:
    transformer = pipeline.get_module("transformer")
    if transformer is None:
        return
    for module in transformer.modules():
        if isinstance(module, Connector):
            module.forward = _original_connector_forward.__get__(module, Connector)


def main() -> int:
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    INDICATOR_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = INDICATOR_DIR / f"diag_connector_attention_{run_id}.json"

    server_args = build_server_args()
    pipeline = build_pipeline(server_args)

    n_connectors = _install_diag_hooks(pipeline)
    if dist.get_rank() == 0:
        print(f"[Diag] Installed hooks on {n_connectors} Connector modules")

    params = VividVRSamplingParams.from_user_kwargs(
        server_args,
        prompt=" ",
        video_input_path=str(INPUT_VIDEO),
        prompt_file_path=str(PROMPT_FILE),
        output_path=str(ACCEPTANCE_ROOT / "result_videos"),
        output_file_name=f"diag_connector_attn_{run_id}.mp4",
        save_output=False,
        return_file_paths_only=False,
        seed=42,
        num_inference_steps=2,  # 诊断只需要少量步骤
    )
    request = prepare_request(server_args, params)

    print(f"[Diag rank={dist.get_rank()}] Running forward pass...")
    result = pipeline.forward(request, server_args)
    print(f"[Diag rank={dist.get_rank()}] Forward pass done")

    # 收集统计
    local_stats = list(_stats)
    gathered = [None] * dist.get_world_size()
    dist.gather_object(local_stats, gathered if dist.get_rank() == 0 else None, dst=0)

    if dist.get_rank() == 0:
        all_stats = []
        for g in gathered:
            if g:
                all_stats.extend(g)

        # 汇总
        connectors_summary = {}
        for s in all_stats:
            if s.get("head_index") == -1:
                ci = s["connector_index"]
                connectors_summary[ci] = s

        summary = {
            "run_id": run_id,
            "num_connectors": n_connectors,
            "sp_degree": 2,
            "connector_mode": get_vividvr_connector_sp_context_mode(),
            "connectors": sorted(connectors_summary.values(), key=lambda x: x["connector_index"]),
            "raw_per_head_stats": [s for s in all_stats if s.get("head_index", -1) >= 0],
        }

        # 整体指标
        if connectors_summary:
            mean_across = [s["remote_ratio_mean"] for s in connectors_summary.values()]
            summary["overall_remote_ratio_mean"] = sum(mean_across) / len(mean_across)
            summary["overall_remote_ratio_range"] = [min(mean_across), max(mean_across)]

        report_path.write_text(json.dumps(summary, indent=2))
        print(f"\n[Diag] Report: {report_path}")
        print(f"[Diag] Overall remote attention ratio: {summary['overall_remote_ratio_mean']:.4f}")
        for ci, s in sorted(connectors_summary.items()):
            print(
                f"  Connector {ci}: remote_ratio={s['remote_ratio_mean']:.4f} "
                f"(p10={s['remote_ratio_p10']:.4f} p50={s['remote_ratio_p50']:.4f} "
                f"p90={s['remote_ratio_p90']:.4f})"
            )

    dist.barrier()
    return 0


def build_server_args() -> ServerArgs:
    import os as _os
    mp = int(_os.environ.get("MASTER_PORT", "30101"))
    server_args = ServerArgs(
        model_path=str(COGVIDEOX_ROOT),
        pipeline_class_name="CogVideoXVividVRControlNetPipeline",
        pipeline_config=VividVRPipelineConfig(),
        component_paths={"vividvr": str(VIVIDVR_CKPT_ROOT)},
        num_gpus=2,
        tp_size=1,
        dp_size=1,
        dp_degree=1,
        sp_degree=2,
        ulysses_degree=2,
        ring_degree=1,
        dist_timeout=3600,
        master_port=mp,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        enable_torch_compile=False,
        warmup=False,
        output_path=str(ACCEPTANCE_ROOT / "result_videos"),
    )
    server_args._adjust_parameters()
    set_global_server_args(server_args)
    return server_args


if __name__ == "__main__":
    raise SystemExit(main())
