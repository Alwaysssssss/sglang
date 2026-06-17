# VividVR Phase E4.1 SP 质量回归深度排查交接文档

更新时间：`2026-06-17 UTC`

## 1. 文档目的

本 handover 面向下一位接手 `Phase E4.1` **SP 双卡质量回归根因排查**的 Codex，目标是：

1. 完整呈现当前项目状态、已验证的事实、已排除的假设
2. 提供可直接查阅的视频文件、指标 JSON、日志路径
3. 明确当前仍未解答的核心问题及下一步排查方向

## 2. 项目状态概览

### 2.1 代码锚点

| 项 | 值 |
|----|-----|
| **分支** | `sglang_Vivid` |
| **HEAD commit** | `c0008cd89` — "Restore vividvr native SP v1/v2 connector semantics" |
| **前一关键 commit** | `9e43f5d31` — "sp2 双卡并行加速"（首次引入 SP shard/gather） |
| **工作区脏文件** | `cogvideox_vividvr_common.py`（仅 SP all-gather 诊断代码，不影响功能） |
| **脏文件内容** | `Connector.forward()` 中添加了一次性 NCCL all-gather 校验日志 |

### 2.2 环境信息

| 项 | 值 |
|----|-----|
| **Python** | 3.10（`/usr/bin/python3`） |
| **虚拟环境** | `/home/zhiheng/sglang/.venv` |
| **torch** | 2.9.1 |
| **triton** | 3.5.1 |
| **flash-attn** | 4.0.0b16 |
| **GPU** | 2× NVIDIA A100-SXM4-80GB |
| **所有包安装日期** | 2026-06-04 |

## 3. 核心问题

### 问题描述

**同样是 `eager_global`（v2）语义路径，2026-06-11 的推理产出 SSIM=0.9846（与单卡一致），但 2026-06-12 起同一台机器、同一份代码 SSIM 回退至 ~0.9643。**

### 问题特征

所有双卡 SP 坏结果（无论如何变化 compile/FA/SDPA）SSIM 都收敛到 **0.962-0.965** 区间，与 `native_sp_only` 基线（0.9628）一致。这意味着：

> 在双卡 SP 模式下，Connector 的 cross-attention 全局优化路径**完全没有提供任何质量增益**，效果等同于纯局部控制。

## 4. 已完成的全量实验时间线

以下按时间顺序列出所有双卡 SP 相关 20-step formal 推理：

### 4.1 好结果簇（2026-06-11）

| 时间 UTC | SSIM | PSNR | 语义路径 | 备注 |
|----------|------|------|----------|------|
| 06/11 03:33 | 0.9847 | 37.68 | unknown | SP-only runtime 双卡对照 |
| 06/11 04:10 | 0.9846 | 37.65 | unknown | SP-only runtime 双卡对照 |
| 06/11 05:29 | 0.9628 | 33.71 | **native_sp_only** | 最激进 fast path，质量不佳但速度快 |
| 06/11 07:36 | 0.9847 | 37.70 | **v7** (local) | 早前 Connector 优化版本 |
| 06/11 08:15 | 0.9785 | 35.21 | **v1** (deferred_global) | v1 质量恢复版 |
| **06/11 08:42** | **0.9846** | **37.31** | **v2 (eager_global)** | ✅ **第一个 GOOD v2** |
| **06/11 13:35** | **0.9786** | **35.20** | **v1** (deferred_global) | v1 recheck |
| **06/11 13:49** | **0.9846** | **37.31** | **v2 (eager_global)** | ✅ **冻结的正式 v2 基线** |

### 4.2 坏结果簇（2026-06-12 起）

| 时间 UTC | SSIM | PSNR | compile | attn | 备注 |
|----------|------|------|---------|------|------|
| 06/12 03:45 | 0.9642 | 33.92 | True | fa | ❌ 第一个 v2 坏结果 |
| 06/12 04:24 | 0.9644 | 33.91 | True | fa | ❌ identity_scale |
| 06/12 04:47 | 0.9643 | 33.92 | True | fa | ❌ identity_only |
| 06/12 05:29 | 0.9643 | 33.92 | True | fa | ❌ packed_local_fix |
| 06/12 06:03 | 0.9644 | 33.92 | True | fa | ❌ packed_eager |
| 06/12 07:40 | 0.9645 | 33.92 | True | fa | ❌ qkv |
| 06/12 08:05 | 0.9652 | 33.97 | True | fa | ❌ modfusion |
| 06/12 08:29 | 0.9650 | 33.95 | True | fa | ❌ modfusion_tfonly |
| 06/12 09:10 | 0.9647 | 33.93 | True | fa | ❌ dyn0 |
| 06/15 04:01 | 0.9651 | 33.98 | **False** | **None** | ❌ 无 FA 无 compile |
| 06/15 04:55 | 0.9651 | 33.98 | **False** | **torch_sdpa** | ❌ SDPA |
| 06/15 05:49 | 0.9651 | 33.98 | **False** | **torch_sdpa** | ❌ pure sp2 |
| 06/15 06:52 | 0.9645 | 33.91 | **True** | **fa** | ❌ FA+compile recheck |
| 06/16 03:13 | 0.9643 | 33.90 | True | fa | ❌ orig_restore_perlayer |
| 06/16 03:35 | 0.9645 | 33.92 | True | fa | ❌ clean_exact_sdpa |
| 06/16 05:02 | 0.9625 | 33.87 | True | fa | ❌ ControlNet global restore |
| 06/16 05:23 | 0.9644 | 33.94 | True | fa | ❌ clean c0008cd89 rerun |
| 06/16 05:45 | 0.9626 | 33.88 | True | fa | ❌ Plan C fullseq CN |
| 06/16 06:36 | 0.9625 | 33.90 | True | fa | ❌ Plan C fullseq CN v2 |
| 06/16 09:10 | 0.9643 | 33.93 | True | fa | ❌ clean verify |
| 06/16 09:32 | 0.9653 | 33.99 | **False** | fa | ❌ no compile |
| 06/16 13:42 | 0.9644 | 33.92 | True | fa | ❌ SDPA verify |
| 06/16 14:15 | 0.9625 | 33.90 | True | fa | ❌ Plan C full seq |
| 06/16 14:44 | 0.9642 | 33.92 | True | fa | ❌ clean cache rerun |

### 4.3 单卡对照（2026-06-16）

| 时间 UTC | SSIM | PSNR | compile | 备注 |
|----------|------|------|---------|------|
| 06/16 15:05 | **0.9845** | 37.29 | True | ✅ **单卡正常，代码逻辑无误** |

### 4.4 v1 / native_sp_only 对照（2026-06-16）

| 时间 UTC | SSIM | 语义路径 | 备注 |
|----------|------|----------|------|
| 06/16 04:34 | 0.9629 | native_sp_only | ❌ native 基线回归 |
| 06/16 13:29 | 0.9627 | v1 (deferred_global) | ❌ v1 验证 |

## 5. 已排除的假设

| # | 假设 | 验证方式 | 结论 |
|---|------|----------|------|
| 1 | 代码逻辑错误 | 单卡推理 SSIM=0.9845 | **排除**：代码在单卡下正确 |
| 2 | torch.compile 导致 | compile=False 双卡仍 ~0.965 | **排除**：与 compile 无关 |
| 3 | Transformer FA vs SDPA | 切换 SDPA/None 双卡仍 ~0.965 | **排除**：与 Transformer attention backend 无关 |
| 4 | inductor/triton cache 污染 | 清理 cache 后重跑 SSIM=0.9642 | **排除**：不是 cache 问题 |
| 5 | ControlNet SP sharding 边界 | Plan C 绕过 ControlNet SP 仍 0.9625 | **排除**：ControlNet SP 不是根因 |
| 6 | NCCL all-gather 通信 | 诊断代码确认 local≠global 且形状正确 | **排除**：NCCL 通信正常 |
| 7 | v1 vs v2 语义模式 | v1/v2/native_sp_only 全部 ~0.963 | **排除**：与 Connector context mode 无关 |
| 8 | Python 包版本变更 | 所有包安装于 6/4，未变更 | **排除**：依赖未变 |
| 9 | runtime_config 差异 | Good/Bad 的 runtime_config 逐字段对比完全一致 | **排除**：配置未变 |
| 10 | git commit 差异 | Good 和 Bad 都使用 c0008cd89 | **排除**：代码未变 |

## 6. 关键代码路径说明

### 6.1 三个 Connector 语义路径

| 语义路径 | 环境变量 | debug 标签 | Connector 行为 |
|----------|----------|------------|----------------|
| **v1 (deferred_global)** | `SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=deferred_global` | `sp_exact_local_attention` | K/V=local control (13500 tokens)，FA3 attention |
| **v2 (eager_global)** | `SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global` | `sp_exact_global_control_attention` | K/V=**global** control (27000 tokens 经 NCCL all-gather)，FA3 attention |
| **native_sp_only** | N/A（不设 env var，默认 deferred_global 但...） | 取决于代码路径 | 最基础的 SP shard/gather，无 Connector 全局优化 |

### 6.2 关键数据流

```
ControlNet.forward()
  → shard_vividvr_video_tokens()     # 每个 GPU 拿 13500 tokens
  → 6层 CogVideoXBlock
  → build_vividvr_connector_control_states()  # v2: NCCL all-gather → 27000 global
  → 返回 tuple of (local_control, global_control)

Transformer.forward()
  → shard_vividvr_video_tokens()
  → 30层 CogVideoXBlock, 每5层插入一次 Connector
  → Connector.forward(local_control, global_control)
      → Q = hidden_states (13500 local)
      → K/V = global_control (27000, FA3 cross-attention)
      → residual = c_mlp(local_control)
      → output = h + attention_out + residual
  → gather_vividvr_video_tokens()
```

### 6.3 核心文件

| 文件 | 行数 | 作用 |
|------|------|------|
| `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py` | 673 | Connector、SP shard/gather、all-gather、attention |
| `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_controlnet.py` | 336 | ControlNet (6层) forward |
| `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr.py` | 257 | Transformer (30层) forward |
| `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py` | 646 | Transformer attention processor 安装 |
| `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py` | 1227 | Denoising 阶段编排、set_forward_context |

## 7. 可用于视觉判断的视频文件

所有视频均位于 `/home/zhiheng/sglang/Vivid_Acceptance/result_videos/`：

| 标签 | 视频文件 | SSIM |
|------|----------|------|
| **Reference（真值）** | `/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4` | 1.000 |
| **v2 GOOD (6/11 冻结基线)** | `phase_e41_native_sp_quality_opt_v2_130f_20step_compile_seed42_20260611T134903Z.mp4` | 0.9846 |
| **v2 BAD (6/16)** | `phase_e41_native_sp_v2_clean_verify_seed42_20260616T091021Z.mp4` | 0.9643 |
| **v1 (deferred_global)** | `phase_e41_native_sp_v1_verify_130f_20step_compile_seed42_20260616T132918Z.mp4` | 0.9627 |
| **native_sp_only** | `phase_e41_native_sp_only_130f_20step_compile_seed42_20260611T052918Z.mp4` | 0.9628 |
| **单卡 v2** | `phase_e41_single_gpu_v2_130f_20step_compile_seed42_20260616T150500Z.mp4` | 0.9845 |
| **Plan C (全序列 ControlNet)** | `phase_e41_sp_plan_c_full_seq_controlnet_130f_20step_compile_seed42_20260616T141520Z.mp4` | 0.9625 |

## 8. 可用于分析的关键指标 JSON

| 标签 | JSON 路径 |
|------|-----------|
| **冻结 v2 真值** | `Vivid_Acceptance/indicator/phase_e41_native_sp_quality_opt_v2_130f_20step_compile_metrics_seed42_20260611T134903Z.json` |
| **单卡验证** | `Vivid_Acceptance/indicator/phase_e41_single_gpu_v2_130f_20step_compile_metrics_seed42_20260616T150500Z.json` |
| **E3.2 单卡基线** | `Vivid_Acceptance/indicator/phase_e32_runtime_e2_align_130f_20step_compile_metrics_seed42_20260609T025514Z.json` |
| **v1 验证** | `Vivid_Acceptance/indicator/phase_e41_native_sp_v1_verify_130f_20step_compile_metrics_seed42_20260616T132918Z.json` |
| **native_sp_only** | `Vivid_Acceptance/indicator/phase_e41_native_sp_only_130f_20step_compile_metrics_seed42_20260611T052918Z.json` |

每个 JSON 中值得关注的字段：
- `summary.ssim_mean` / `summary.psnr_mean` — 总体质量
- `summary.frames[].ssim` — 逐帧 SSIM，可看哪些帧退化最严重
- `runtime_config.*` — 完整的推理配置快照
- `debug.connector_context_mode` — 实际进入的语义路径
- `debug.control_context_shape_local/global` — 控制张量形状

## 9. 未解答的核心问题

### 9.1 为什么 6/11 和 6/12 结果不同？

这是当前最大的未解之谜：
- 代码相同（`c0008cd89`）
- 配置相同（`SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global`）
- 硬件相同（同一台机器，2×A100）
- 所有 Python 包安装日期为 6/4，无后续更新

**候选解释（均未验证）**：
1. **CUDA kernel 选择的非确定性**：FA3 和 torch.compile inductor 在不同 session 中 autotune 出不同的 kernel 实现，产生微小数值差异，在 130 帧 20 步扩散过程中被放大
2. **GPU 时钟/温度影响**：不同时间的 GPU Boost 频率不同，影响某些 CUDA kernel 的数值精度
3. **系统级变更**：CUDA runtime、GPU 驱动、内核模块等在 6/11-6/12 之间发生了未被记录的更新

### 9.2 为什么单卡正常而双卡异常？

单卡 SSIM=0.9845 证明 v2 代码路径在逻辑上完全正确。但双卡下 v2 的结果与 native_sp_only（纯局部控制）无差异。这意味着在双卡 SP 模式下，Connector 的全局 cross-attention 路径**虽然执行了计算，但计算结果与局部控制等价**——即 NCCL all-gather 传来的 global_control 在 FA3 attention 后并未提供比 local_control 更多的信息。

## 10. 建议的下一步排查方向

### 10.1 逐帧 SSIM 对比（低开销）

对比 Good v2 和 Bad v2 的各帧 SSIM 曲线：
- 如果所有帧均匀退化 → 全局性的数值精度问题
- 如果特定帧区间退化严重 → SP boundary / clip 边界问题
- 如果前期帧正常后期退化 → 扩散过程累积误差

### 10.2 第一 timestep 张量对比（中开销）

在同一 session 内先跑单卡再跑双卡，在第一层 Connector forward 处 dump：
- Q, K, V 张量的数值（norm、分布）
- attention output 的数值
- c_mlp(local_control) 的数值
- 对比单卡 vs 双卡 rank0 的差异

### 10.3 FA3 与 SDPA 在 Connector 中的数值对比（中开销）

当前 Transformer 的 FA/SDPA 切换不影响 Connector——Connector 在 SP 模式下始终使用 FA3（`flash_attn_func`）。可以临时修改 `run_vividvr_connector_attention()` 强制走 SDPA 路径，验证 Connector 的 FA3 是否为根因。

### 10.4 复现 6/11 好结果（高开销但最关键）

如果能找到 6/11 时的完整系统状态快照（GPU 驱动版本、CUDA runtime 版本、NCCL 版本、内核模块），尝试在那个状态下重新运行当前代码，验证是否能复现 0.9846。

## 11. 工作纪律（继承自上一轮 handover）

1. ✅ 不要再把 "pass_compare=true" 当作质量验收通过
2. ✅ 任何质量判断必须对照冻结 v2 视频或指标
3. ✅ 任何策略如果再次落到 `ssim_mean ≈ 0.964` 坏簇，可快速判为失败
4. ✅ 在没有恢复 v2 真语义前，不要继续叠加新的提速技巧
5. ✅ 优先做语义收敛，而不是继续追求更快的 `s/it`

## 12. 相关文档索引

| 文档 | 路径 |
|------|------|
| v1/v2 语义与早期提速 | `docs_xzh/hand_over/phase_e_e41_native_sp_v2_quality_control_and_next_speedup_handover.md` |
| 质量闭环计划 | `docs_xzh/add_strategy/13_phase_e_sp_quality_closure_plan.md` |
| 上轮 status handover | `docs_xzh/hand_over/phase_e_e41_sp_quality_regression_status_and_semantic_fix_handover.md` |
| Benchmark 命令 | `docs_xzh/run_vivid_benchmark.md` |
| E3.2 交接 | `docs_xzh/hand_over/phase_e_e32_completion_and_e33_next_steps_handover.md` |

## 13. debug_tensors 目录

`/home/zhiheng/sglang/Vivid_Acceptance/debug_tensors/` 下有三次 dump：

| 子目录 | 说明 |
|--------|------|
| `single_20260616/` | 单卡推理 dump |
| `single_nocmp_20260616/` | 单卡无 compile dump |
| `sp2_v2_nocmp_20260616/` | 双卡 v2 无 compile dump |

可用于张量级 cross-reference 分析。

---

**本 handover 的核心信息**：当前代码逻辑正确（单卡验证通过），配置正确（runtime_config 一致），环境未变（包安装日期为 6/4）。唯一未解释的现象是 6/11 好结果和 6/12+ 坏结果之间的差异——这个差异不是代码/配置层面，而是运行时层面的。下一步应聚焦于**数值精度级别的对比分析**（逐帧 SSIM、第一 timestep 张量 dump、FA3 vs SDPA 在 Connector 中的影响）。
