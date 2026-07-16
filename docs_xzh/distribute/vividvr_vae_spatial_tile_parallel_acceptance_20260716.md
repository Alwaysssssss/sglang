# VividVR CogVideoX VAE 空间 Tile 并行验收记录

日期：2026-07-16
结论：实现与验收通过；`vae_sp` 继续作为默认关闭的实验性 opt-in 开关。

## 1. 验收范围与结论

本轮在 Diffusers 0.37.0 的 CogVideoX VAE tiled decode 语义上实现了 SP subgroup 内的空间 tile 并行，并完成以下验证：

- CPU 单元测试覆盖 tile plan、round-robin 分配、单 tile temporal cache、descriptor/payload transport、row-major merge、fallback、统计与 VividVR stage 接线。
- 真实 NCCL 固定 latent 验证覆盖 `SP=2`、`SP=4`、`CFG=2 × SP=2`，包括 rank 间故意不同且非 contiguous 的输入。
- 两条正式 `130f / 20 step` FlowCut 服务 treatment 均完成 warmup、formal、对象存储、callback、下载、逐帧 compare 和清理。
- R99、R100 的 `Decode/Trim`、模型推理和端到端总耗时均相对各自历史 control 获得正收益。
- 抽帧人工检查未发现 tile 接缝、闪烁、颜色漂移或 trim/stitch 边界变化。

正式 compare JSON 仍保留原始 `quality_failed` 状态。R99 的 SSIM mean 比 control 低 `0.000206`、SSIM min 低 `0.000900`，failed-frame ratio 为 `2/130`；用户明确确认该差异可忽略并批准通过。R100 的 mean 差异更小、failed-frame ratio 与 control 相同，本验收沿用同一人工豁免口径。原始 JSON 未被修改。

## 2. 实现结果

核心实现提交：

- `1ca30dd7b`：锁定 CogVideoX VAE tile plan 与分配合同。
- `14a1610c3`：保持 tiled decode 的 temporal cache、blend、crop 和 merge 语义。
- `cf8be47ae`：实现 SP subgroup 内 descriptor 与 padded tensor payload gather。
- `2e4cff479`：接入 VAE 空间 tile 并行 decode、开关与 fail-fast/fallback。
- `da687d21d`：接入单 clip/多 clip VAE 并行统计。
- `6e6438b90`：增加 R99/R100 VAE SP benchmark treatment 和历史控制派生指标。
- `22dc14157`：补齐真实 NCCL subgroup 验证。
- `8f3e2ff9b`：在 VAE decode 前按 SP subgroup canonicalize latent。
- `8701b6197`：将 canonical latent 显式复制为 contiguous memory，兼容 NCCL broadcast。

实现遵守以下边界：

- collective 只通过 `get_sp_group()`，没有 WORLD group 或 `all_gather_object`。
- `CFG=2 × SP=2` 的 `[0, 1]` 与 `[2, 3]` 两个 subgroup 独立通信。
- `vae_sp=False` 默认值、现有 Phase E 正式配置和 FlowCut 请求契约均未改变。
- stage 只传播开关与统计；tile plan、decode、transport、merge 均位于 CogVideoX VAE 内。

### 2.1 实施中确认的必要偏差

正式 warmup 暴露出两个仅靠合成 replicated latent 无法覆盖的运行时事实：VividVR 进入 VAE decode 时各 SP rank 的 latent 可能存在差异，而且实际 latent 可能是非 contiguous view。为保证各 rank 构造相同 tile payload，进入并行 decode 前增加 SP subgroup root latent broadcast；broadcast buffer 使用 `clone(memory_format=torch.contiguous_format)`。

这是对设计文档中“若 decode 输入未复制，先在同一 SP group 内显式恢复完整 latent”的落实，不改变串行路径、tile 数值语义或 subgroup 边界。对应的 TDD 回归测试和真实非 contiguous NCCL 验证均已通过。

## 3. 固定 Latent 正确性

| 拓扑 | SP subgroup | tiles / 本地分配 | 输入条件 | 串行与并行误差 | 结果 |
| --- | --- | --- | --- | --- | --- |
| SP=2 | `[0, 1]` | `9 / [5, 4]` | rank-divergent、non-contiguous | max `0`，mean `0`，bitwise equal | 通过 |
| SP=4 | `[0, 1, 2, 3]` | `9 / [3, 2, 2, 2]` | rank-divergent、non-contiguous | max `0`，mean `0`，bitwise equal | 通过 |
| CFG=2 × SP=2 | `[0, 1]`、`[2, 3]` | `9 / [5, 4]`（每组） | 两组独立标记、rank-divergent、non-contiguous | max `0`，mean `0`，bitwise equal | 通过 |

指标与日志：

- `Vivid_Acceptance/indicator/vividvr_vae_sp_fixed_latent_sp2_noncontig_20260716.json`
- `Vivid_Acceptance/indicator/vividvr_vae_sp_fixed_latent_sp4_noncontig_20260716.json`
- `Vivid_Acceptance/indicator/vividvr_vae_sp_fixed_latent_cfg2_sp2_noncontig_20260716.json`
- 对应日志位于 `Vivid_Acceptance/logs/`，文件名与指标同 stem。

三份指标均为 `overall_pass=true`、`noncontiguous_inputs_exercised=true`；rank-divergent 输入被检测到并在各自 subgroup 内恢复为相同 root latent。

## 4. 正式性能验收

历史控制只读取下列目录中的既有 JSON，没有重跑或改写：

```text
Vivid_Acceptance/acceleration_benchmark/vividvr_accel_full_warmup1_20260716
```

| 指标 | R99 control | R100 control |
| --- | ---: | ---: |
| 总耗时（s） | 551.119174 | 370.881417 |
| 模型推理（s） | 544.320553 | 365.066904 |
| Denoise（s） | 380.175751 | 195.651912 |
| Decode/Trim（s） | 100.274310 | 101.785656 |
| SSIM mean | 0.984667384 | 0.984619328 |
| SSIM min | 0.980502773 | 0.978691849 |
| failed-frame ratio | 0/130 | 2/130 |

控制记录在验收前后的 mtime 保持不变：R99 `1784176039`，R100 `1784176727`。

### 4.1 Treatment 性能

| 指标 | R99 + VAE SP（SP=2） | R100 + VAE SP（CFG=2 × SP=2） |
| --- | ---: | ---: |
| VAE SP requested / effective | true / true | true / true |
| tiles / 本地分配 | `32 / [16, 16]` | `32 / [16, 16]`（每 SP 组） |
| tile decode（s） | 58.512990 | 59.758195 |
| gather（s） | 0.012420 | 0.012690 |
| merge（s） | 0.139183 | 0.140075 |
| VAE decode（s） | 58.665689 | 59.918715 |
| Decode/Trim（s） | 58.937737 | 60.178877 |
| Decode/Trim speedup | 1.701360× | 1.691385× |
| 模型推理（s） | 502.578220 | 334.402225 |
| 模型推理 speedup | 1.083056× | 1.091700× |
| 总耗时（s） | 510.930752 | 341.004324 |
| 端到端 speedup | 1.078657× | 1.087615× |
| treatment GPU·秒 | 1021.861505 | 1364.017296 |
| control GPU·秒 | 1102.238348 | 1483.525670 |
| rank 峰值显存（MiB） | `[54354.375, 45492.375]` | `[55016.375, 45314.375, 55014.375, 54860.375]` |

正式记录：

- `Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp_r99_canonicalized_v2_20260716/records/R99_VAE_SP_formal.json`
- `Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp_r100_canonicalized_20260716/records/R100_VAE_SP_formal.json`

R99 另完成一次 formal 重复验证：`vividvr_vae_sp_r99_canonicalized_v3_20260716`。其 Decode/Trim 为 `58.198362 s`（1.722975×），模型推理为 `502.869282 s`（1.082430×），总耗时为 `510.964968 s`（1.078585×），证明主要收益可复现。

### 4.2 质量与人工豁免

| 指标 | R99 treatment | 相对 R99 control | R100 treatment | 相对 R100 control |
| --- | ---: | ---: | ---: | ---: |
| SSIM mean | 0.984461093 | -0.000206290 | 0.984603124 | -0.000016203 |
| SSIM min | 0.979602732 | -0.000900040 | 0.977483506 | -0.001208343 |
| failed-frame ratio | 2/130 | +2/130 | 2/130 | 0 |
| 原始 record status | `quality_failed` | — | `quality_failed` | — |
| 最终验收 | 用户批准人工豁免，通过 | — | 同口径人工豁免，通过 | — |

R99 重复运行 v3 的 SSIM mean 为 `0.984651467`，只比 control 低 `0.000015917`，failed-frame ratio 为 `1/130`，进一步说明硬阈值附近存在微小运行间波动。验收不篡改硬门禁结果，而是在原始证据之上记录人工决策。

人工检查分别对 R99、R100 的 control/treatment 抽取帧 `0, 30, 60, 90, 112–116, 129` 制作 contact sheet；未观察到空间 tile 接缝、闪烁迹象、颜色漂移或长视频 trim/stitch 边界异常。

## 5. 服务合同证据

R99、R100 均在 tmux 管理的完整服务生命周期中运行，并保留：

- Moto S3 日志、callback JSONL、主服务日志；
- warmup/formal 下载视频与 perf JSON；
- formal compare JSON 和标准 record；
- caption bridge 生成的一行一个 temporal clip 的 caption 文件；
- `accepted → input_ready → caption_ready → denoising → uploading_result → succeeded` callback 序列。

R99 batch：

```text
Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp_r99_canonicalized_v2_20260716
```

R100 batch：

```text
Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp_r100_canonicalized_20260716
```

mock caption helper 不向 stdout 输出，因此两批 `caption.log` 为空；但服务日志明确记录 `mode=mock`、`fallback=false`、caption count 为 2，且 warmup/formal caption 文件均存在并为 1143 bytes。这不影响服务合同有效性。

## 6. 回归与静态验证

最终轻量回归命令：

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_parallel.py \
  python/sglang/multimodal_gen/test/unit/test_stage_b_vividvr_components.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_offload.py \
  python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py -q
```

结果：`154 passed, 5 warnings, 1145 subtests passed`。

额外检查：

- 相关实现与工具脚本 `py_compile` 通过；
- 源码检索确认没有 `all_gather_object`、`dist.group.WORLD` 或 `group=WORLD`；
- 所有 VAE transport 与 canonical latent broadcast 均绑定 `get_sp_group()`；
- 当前 Diffusers 版本为 `0.37.0`。

## 7. 最终决策与已知风险

本阶段目标已完成并通过验收。VAE tile 并行在 R99/R100 两个正式拓扑上均获得约 `1.69–1.70×` Decode/Trim 加速和约 `1.08–1.09×` 端到端加速。

`vae_sp` 暂不自动替换现有正式默认配置，原因如下：

- 当前实现会在每个 SP rank 恢复完整 decoded tile 集，gather staging 与 replicated merge 仍带来额外显存；
- 视频质量硬门禁位于极严的逐帧阈值附近，存在很小的运行间数值波动，正式启用策略应另行评估门禁容差；
- 后续 leader merge/broadcast、分批 gather 或通信计算重叠属于独立优化，不在本阶段范围内。

因此本轮不修改 `AGENTS.md`、Phase E 默认配置或服务请求契约。
