# VividVR 加速实验统计（纯表格版）

## 1. 统计口径

| 项目 | 固定口径 |
| --- | --- |
| 耗时单位 | 秒，表格至少保留 2 位小数，原始 JSON 保留完整精度 |
| 显存单位 | GiB；记录每张 GPU 峰值，主表填写最大单卡峰值 |
| 总耗时 | `total_runtime_seconds` |
| 模型推理耗时 | `model_inference_runtime_seconds` |
| Denoise 耗时 | `VividVRMultiClipDenoisingStage` 同步耗时 |
| Stage 耗时 | 同步后的 pipeline stage profiling；多 clip 填整次请求累计值 |
| 正式计时 | 完整 warmup 一次，第二次完整请求记为正式结果 |
| 单卡公平性 | 同一时刻只运行一个单卡推理进程 |
| 累计加速比 | R0 模型推理耗时 / 当前方案模型推理耗时 |
| 模块增量加速比 | 指定对照方案模型推理耗时 / 当前方案模型推理耗时 |
| GPU·秒 | GPU 数量 × 当前方案模型推理耗时 |
| 相对 R0 资源效率 | R0 GPU·秒 / 当前方案 GPU·秒 |
| 质量口径 | `pass_compare`、`SSIM mean`、`SSIM min`、`failed frame ratio` |
| N/A | 方案不涉及或 profiler 无法可靠拆分 |

## 2. 固定实验口径

| 项目 | 固定值 |
| --- | --- |
| 输入视频 | `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4` |
| Caption | `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt` |
| Prompt | `/home/zhiheng/Vivid-VR/input/720p/prompt.txt` |
| Reference | `/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4` |
| 输入帧数 | 130 |
| Temporal process frames | 121 |
| Temporal clip 数 | 2 |
| 推理步数 | 20 |
| Seed | 42 |
| Guidance scale | 6 |
| Restoration guidance scale | -1.0 |
| Upscale | 1.0 |
| Dtype | `bfloat16` |
| Prompt embedding 长度 | 226 |
| VAE tiling | tile sample min height/width：240/360 |
| SP connector | `SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global` |
| Compile 正式口径 | 固定形状完整 warmup 后记录第二次完整请求 |
| 质量最低要求 | `pass_compare=true`，且质量指标无明显漂移 |

## 3. 正式实验方案

| 编号 | 关键加速方案 | 增益对照 | 统计目标 | 状态 |
| --- | --- | --- | --- | --- |
| R0 | 单卡 SDPA eager | — | 原始性能基线 | 已实现，待统一测试 |
| R1 | 单卡 FA eager | R0 | Attention backend | 已实现，待统一测试 |
| R2 | 单卡 FA + `torch.compile` | R1 | `torch.compile` | 当前单卡正式配置 |
| R3 | 双卡 SP=2 + FA-SP + `torch.compile` | R2 | 双卡 SP | 当前双卡正式配置 |
| R4 | 四卡 SP=4 + FA-SP + `torch.compile` | R3 | SP 2→4 卡扩展 | 需统一复测 |
| R5 | 四卡 CFG=2 × SP=2 + FA-SP + `torch.compile` | R3、R4 | CFG parallel、四卡拓扑 | 已实现并验收 |
| R6 | R2 + 算子融合 | R2 | 算子融合 | 待实现和测试 |
| R7 | R2 + Cache-DiT | R2 | Cache-DiT | 待实现和测试 |
| R8 | R2 + TeaCache | R2 | TeaCache | 待实现和测试 |
| R9 | R2 + 通用量化 | R2 | 量化、显存、质量 | 量化方案待定义 |
| R99 | 双卡 SP=2 + 全部兼容且有正收益的加速 | R3 | 双卡综合最快方案 | 待前置模块验证 |
| R100 | 最快四卡拓扑 + 全部兼容且有正收益的加速 | R4、R5 中更快者 | 四卡综合最快方案 | 待前置模块验证 |

## 4. 总体结果

| 方案 | 总耗时 | 模型推理耗时 | Denoise 耗时 | 相对 R0 加速比 | 模块增量加速比 | GPU·秒 | 相对 R0 资源效率 | 最大单卡峰值显存 | `pass_compare` | SSIM mean | SSIM min | Failed frame ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| R0 |  |  |  | 1.0000× | — |  | 1.0000× |  |  |  |  |  |
| R1 |  |  |  |  | R0 / R1 |  |  |  |  |  |  |  |
| R2 |  |  |  |  | R1 / R2 |  |  |  |  |  |  |  |
| R3 |  |  |  |  | R2 / R3 |  |  |  |  |  |  |  |
| R4 |  |  |  |  | R3 / R4 |  |  |  |  |  |  |  |
| R5 |  |  |  |  | R3 / R5 |  |  |  |  |  |  |  |
| R6 |  |  |  |  | R2 / R6 |  |  |  |  |  |  |  |
| R7 |  |  |  |  | R2 / R7 |  |  |  |  |  |  |  |
| R8 |  |  |  |  | R2 / R8 |  |  |  |  |  |  |  |
| R9 |  |  |  |  | R2 / R9 |  |  |  |  |  |  |  |
| R99 |  |  |  |  | R3 / R99 |  |  |  |  |  |  |  |
| R100 |  |  |  |  | 最快(R4, R5) / R100 |  |  |  |  |  |  |  |

## 5. Stage 耗时明细

| Stage | R0 | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R99 | R100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `VividVRInputValidationStage` |  |  |  |  |  |  |  |  |  |  |  |  |
| `VividVRPromptPreparationStage` |  |  |  |  |  |  |  |  |  |  |  |  |
| `VividVRTemporalWindowPlanningStage` |  |  |  |  |  |  |  |  |  |  |  |  |
| `VividVRLongClipPreparationStage` |  |  |  |  |  |  |  |  |  |  |  |  |
| `VividVRTimestepPreparationStage` |  |  |  |  |  |  |  |  |  |  |  |  |
| `VividVRMultiClipDenoisingStage` |  |  |  |  |  |  |  |  |  |  |  |  |
| `VividVRMultiClipDecodeTrimStage` |  |  |  |  |  |  |  |  |  |  |  |  |
| `VividVRTemporalStitchPostprocessStage` |  |  |  |  |  |  |  |  |  |  |  |  |
| 未归类开销 |  |  |  |  |  |  |  |  |  |  |  |  |
| 模型推理总计 |  |  |  |  |  |  |  |  |  |  |  |  |

## 6. Denoising 核心耗时

| 方案 | Temporal clip 数 | 推理步数 | Denoise 总耗时 | 占模型推理比例 | 平均每 step | Steady step 中位数 | SP 通信耗时 | CFG 通信耗时 | Cache 执行/跳过 | 最大单卡峰值显存 | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| R0 | 2 | 20 |  |  |  |  | N/A | N/A | N/A |  |  |
| R1 | 2 | 20 |  |  |  |  | N/A | N/A | N/A |  |  |
| R2 | 2 | 20 |  |  |  |  | N/A | N/A | N/A |  |  |
| R3 | 2 | 20 |  |  |  |  |  | N/A | N/A |  |  |
| R4 | 2 | 20 |  |  |  |  |  | N/A | N/A |  |  |
| R5 | 2 | 20 |  |  |  |  |  |  | N/A |  |  |
| R6 | 2 | 20 |  |  |  |  | N/A | N/A | N/A |  |  |
| R7 | 2 | 20 |  |  |  |  | N/A | N/A |  |  |  |
| R8 | 2 | 20 |  |  |  |  | N/A | N/A |  |  |  |
| R9 | 2 | 20 |  |  |  |  | N/A | N/A | N/A |  |  |
| R99 | 2 | 20 |  |  |  |  |  | N/A |  |  |  |
| R100 | 2 | 20 |  |  |  |  |  |  |  |  |  |

## 7. 实验环境与结论

### 7.1 实验环境

| 项目 | 值 |
| --- | --- |
| 机器型号 |  |
| GPU 型号与可用数量 |  |
| CUDA 版本 |  |
| Driver 版本 |  |
| PyTorch 版本 |  |
| FlashAttention 版本 |  |
| sglang commit |  |
| Python 路径 | `/home/zhiheng/sglang/.venv/bin/python` |
| 模型/checkpoint 路径 |  |
| Dtype | `bfloat16` |
| Compile mode |  |
| 计时方式 |  |
| 显存统计方式 |  |
| Stage profiling 是否同步 |  |
| Perf/report 目录 | `/home/zhiheng/sglang/Vivid_Acceptance/indicator` |
| 结果视频目录 | `/home/zhiheng/sglang/Vivid_Acceptance/result_videos` |

### 7.2 方案运行时快照

| 方案 | Requested backend | Effective backend | Compile 生效 | 并行拓扑 | Fusion 配置 | Cache 配置 | 量化配置 | Warmup/Formal 产物 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | `sdpa` |  | 否 | 单卡 | 无 | 无 | 无 |  |
| R1 | `fa` |  | 否 | 单卡 | 无 | 无 | 无 |  |
| R2 | `fa` |  |  | 单卡 | 无 | 无 | 无 |  |
| R3 | `fa` |  |  | SP=2 | 无 | 无 | 无 |  |
| R4 | `fa` |  |  | SP=4 | 无 | 无 | 无 |  |
| R5 | `fa` |  |  | CFG=2 × SP=2 | 无 | 无 | 无 |  |
| R6 | `fa` |  |  | 单卡 |  | 无 | 无 |  |
| R7 | `fa` |  |  | 单卡 | 无 |  | 无 |  |
| R8 | `fa` |  |  | 单卡 | 无 |  | 无 |  |
| R9 | `fa` |  |  | 单卡 | 无 | 无 |  |  |
| R99 | `fa` |  | 是 | SP=2 | 最优有效配置 | Cache-DiT / TeaCache / 关闭（三选一） | 最优有效配置 |  |
| R100 | `fa` |  | 是 | SP=4 / CFG=2 × SP=2（择优） | 最优有效配置 | Cache-DiT / TeaCache / 关闭（三选一） | 最优有效配置 |  |

### 7.3 模块收益结论

| 加速模块 | Treatment | Control | 延迟增量加速比 | GPU·秒变化 | 显存变化 | 质量变化 | 正式结论 |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| Attention backend | R1 | R0 |  |  |  |  |  |
| `torch.compile` | R2 | R1 |  |  |  |  |  |
| 双卡 SP | R3 | R2 |  |  |  |  |  |
| SP 2→4 卡扩展 | R4 | R3 |  |  |  |  |  |
| CFG parallel | R5 | R3 |  |  |  |  |  |
| 四卡并行拓扑 | R5 | R4 |  |  |  |  |  |
| 算子融合 | R6 | R2 |  |  |  |  |  |
| Cache-DiT | R7 | R2 |  |  |  |  |  |
| TeaCache | R8 | R2 |  |  |  |  |  |
| 通用量化 | R9 | R2 |  |  |  |  |  |
| 双卡综合最快方案 | R99 | R3 |  |  |  |  |  |
| 四卡综合最快方案 | R100 | R4/R5 最快者 |  |  |  |  |  |
| 综合方案 2→4 卡扩展 | R100 | R99 |  |  |  |  |  |
