# VividVR 加速实验统计（纯表格版）

## 1. 统计口径

| 项目 | 固定口径 |
| --- | --- |
| 耗时单位 | 秒；JSON 保留完整精度 |
| 显存单位 | GiB；记录逐卡峰值，主表使用最大单卡峰值 |
| 总耗时 | 从请求提交前到服务终态的 `total_runtime_seconds` |
| 模型推理耗时 | perf 中 `total_duration_ms / 1000` |
| Denoise 耗时 | `VividVRMultiClipDenoisingStage` |
| Stage 耗时 | 八个 VividVR pipeline stage 的整次请求累计值 |
| Warmup | 仅 compile 方案完整 warmup 一次；eager 方案不 warmup |
| Formal | warmup 成功后下一次完整请求；eager 方案的第一次完整请求 |
| 累计加速比 | R0 模型推理耗时 / 当前方案模型推理耗时 |
| 模块增量加速比 | 指定质量通过的对照方案耗时 / 当前方案耗时 |
| GPU·秒 | GPU 数量 × 模型推理耗时 |
| 相对 R0 资源效率 | R0 GPU·秒 / 当前方案 GPU·秒 |
| 质量口径 | `pass_compare`、SSIM mean、SSIM min、failed frame ratio |
| 不可观测值 | JSON 使用 `null` 并提供 `reason`，不猜测通信或 cache 数据 |

## 2. 固定实验口径

| 项目 | 固定值 |
| --- | --- |
| 输入视频 | `/home/zhiheng/input/test_video_long_960x720_130f.mp4` |
| Caption | `/home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars/quad-test-video-long-960x720-130f-run2-20260708T060202Z.txt` |
| Prompt 来源 | 仅由固定 caption sidecar mock 返回的 `caption_file_path`；不传固定 `prompt.txt` |
| Reference | `/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark/downloads/quad-test-video-long-960x720-130f-run2-20260708T060202Z.bridge-downloaded.mp4` |
| 输入帧数 / 推理步数 | 130 / 20 |
| Temporal process frames / clip 数 | 121 / 2 |
| Seed / Guidance / Restoration guidance | 42 / 6 / -1.0 |
| Upscale / Dtype | 1.0 / `bfloat16` |
| Prompt embedding 长度 | 226 |
| VAE tiling | tile sample min height/width：240/360 |
| SP connector | `SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global` |
| 单卡公平性 | 同一时刻只运行一个单卡推理进程 |
| 质量最低要求 | `pass_compare=true`，SSIM frame threshold 为 0.98，failed frame ratio 为 0 |

## 3. 正式实验方案

| 编号 | 关键加速方案 | 增益对照 | 统计目标 | 执行状态 |
| --- | --- | --- | --- | --- |
| R0 | 单卡 SDPA eager | — | 原始性能基线 | 可执行 |
| R1 | 单卡 FA eager | R0 | Attention backend | 可执行 |
| R2 | 单卡 FA + `torch.compile` | R1 | `torch.compile` | 可执行 |
| R3 | 双卡 SP=2 + FA-SP + `torch.compile` | R2 | 双卡 SP 综合收益 | 可执行 |
| R4 | 四卡 SP=4 + FA-SP + `torch.compile` | R2 | 四卡纯 SP 综合收益 | 可执行 |
| R5 | 四卡 CFG=2 × SP=2 + FA-SP + `torch.compile` | R4 | 四卡 CFG×SP 拓扑 | 可执行 |
| R6 | R2 + CogVideoX modulation/residual fusion（transformer、controlnet） | R2 | 已实现算子融合 | 可执行 |
| R7 | R2 + Cache-DiT | R2 | Cache-DiT | 不可执行：原生 VividVR denoise 未集成 |
| R8 | R2 + TeaCache | R2 | TeaCache | 不可执行：原生 VividVR denoise 未实现 |
| R9 | R2 + 通用量化 | R2 | 量化、显存、质量 | 不可执行：无已验证 VividVR 量化权重/加载路径 |
| R99 | 双卡 SP=2 + FA-SP + compile + modulation/residual fusion | R3 | 双卡全部已实现加速 | 可执行 |
| R100 | 四卡 CFG=2 × SP=2 + FA-SP + compile + modulation/residual fusion | R4、R5 中质量通过且更快者 | 四卡全部已实现加速 | 可执行 |

## 4. 总体结果

| 方案 | 总耗时 | 模型推理耗时 | Denoise 耗时 | 相对 R0 加速比 | 模块增量加速比 | GPU·秒 | 相对 R0 资源效率 | 最大单卡峰值显存 | `pass_compare` | SSIM mean | SSIM min | Failed frame ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| R0 |  |  |  | 1.0000× | — |  | 1.0000× |  |  |  |  |  |
| R1 |  |  |  |  | R0 / R1 |  |  |  |  |  |  |  |
| R2 |  |  |  |  | R1 / R2 |  |  |  |  |  |  |  |
| R3 |  |  |  |  | R2 / R3 |  |  |  |  |  |  |  |
| R4 |  |  |  |  | R2 / R4 |  |  |  |  |  |  |  |
| R5 |  |  |  |  | R4 / R5 |  |  |  |  |  |  |  |
| R6 |  |  |  |  | R2 / R6 |  |  |  |  |  |  |  |
| R7 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| R8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| R9 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| R99 |  |  |  |  | R3 / R99 |  |  |  |  |  |  |  |
| R100 |  |  |  |  | 最快有效(R4, R5) / R100 |  |  |  |  |  |  |  |

## 5. Stage 耗时明细

| Stage | R0 | R1 | R2 | R3 | R4 | R5 | R6 | R99 | R100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `VividVRInputValidationStage` |  |  |  |  |  |  |  |  |  |
| `VividVRPromptPreparationStage` |  |  |  |  |  |  |  |  |  |
| `VividVRTemporalWindowPlanningStage` |  |  |  |  |  |  |  |  |  |
| `VividVRLongClipPreparationStage` |  |  |  |  |  |  |  |  |  |
| `VividVRTimestepPreparationStage` |  |  |  |  |  |  |  |  |  |
| `VividVRMultiClipDenoisingStage` |  |  |  |  |  |  |  |  |  |
| `VividVRMultiClipDecodeTrimStage` |  |  |  |  |  |  |  |  |  |
| `VividVRTemporalStitchPostprocessStage` |  |  |  |  |  |  |  |  |  |
| 未归类开销 |  |  |  |  |  |  |  |  |  |
| 模型推理总计 |  |  |  |  |  |  |  |  |  |

## 6. Denoising 核心耗时

| 方案 | Clip 数 | 步数 | Denoise 总耗时 | 推理占比 | 平均 step | Steady step 中位数 | SP 通信 | CFG 通信 | Cache 执行/跳过 | 峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: |
| R0 |  |  |  |  |  |  | N/A | N/A | N/A |  |
| R1 |  |  |  |  |  |  | N/A | N/A | N/A |  |
| R2 |  |  |  |  |  |  | N/A | N/A | N/A |  |
| R3 |  |  |  |  |  |  | 未单独 profiling | N/A | N/A |  |
| R4 |  |  |  |  |  |  | 未单独 profiling | N/A | N/A |  |
| R5 |  |  |  |  |  |  | 未单独 profiling | 未单独 profiling | N/A |  |
| R6 |  |  |  |  |  |  | N/A | N/A | N/A |  |
| R99 |  |  |  |  |  |  | 未单独 profiling | N/A | N/A |  |
| R100 |  |  |  |  |  |  | 未单独 profiling | 未单独 profiling | N/A |  |

## 7. 运行时快照与模块结论

| 方案 | Requested backend | Effective backend | Compile | 并行拓扑 | Fusion | Cache | 量化 | Warmup / Formal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | `sdpa` | `sdpa` | 否 | 单卡 | 无 | 无 | 无 | formal |
| R1 | `fa` | `fa` | 否 | 单卡 | 无 | 无 | 无 | formal |
| R2 | `fa` | `fa` | 是 | 单卡 | 无 | 无 | 无 | warmup + formal |
| R3 | `fa` | `fa_sp` | 是 | SP=2 | 无 | 无 | 无 | warmup + formal |
| R4 | `fa` | `fa_sp` | 是 | SP=4 | 无 | 无 | 无 | warmup + formal |
| R5 | `fa` | `fa_sp` | 是 | CFG=2 × SP=2 | 无 | 无 | 无 | warmup + formal |
| R6 | `fa` | `fa` | 是 | 单卡 | modulation/residual | 无 | 无 | warmup + formal |
| R7 | N/A | N/A | N/A | N/A | N/A | 不支持 | 无 | unsupported JSON |
| R8 | N/A | N/A | N/A | N/A | N/A | 不支持 | 无 | unsupported JSON |
| R9 | N/A | N/A | N/A | N/A | N/A | 无 | 不支持 | unsupported JSON |
| R99 | `fa` | `fa_sp` | 是 | SP=2 | modulation/residual | 无 | 无 | warmup + formal |
| R100 | `fa` | `fa_sp` | 是 | CFG=2 × SP=2 | modulation/residual | 无 | 无 | warmup + formal |

| 加速模块 | Treatment | Control | 延迟增量加速比 | GPU·秒变化 | 显存变化 | 质量变化 | 正式结论 |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| Attention backend | R1 | R0 |  |  |  |  |  |
| `torch.compile` | R2 | R1 |  |  |  |  |  |
| 双卡 SP | R3 | R2 |  |  |  |  |  |
| 四卡 SP | R4 | R2 |  |  |  |  |  |
| 四卡 CFG×SP 拓扑 | R5 | R4 |  |  |  |  |  |
| Modulation/residual fusion | R6 | R2 |  |  |  |  |  |
| Cache-DiT | R7 | R2 | N/A | N/A | N/A | N/A | 当前不支持 |
| TeaCache | R8 | R2 | N/A | N/A | N/A | N/A | 当前不支持 |
| 通用量化 | R9 | R2 | N/A | N/A | N/A | N/A | 当前不支持 |
| 双卡全部已实现加速 | R99 | R3 |  |  |  |  |  |
| 四卡全部已实现加速 | R100 | R4/R5 最快有效者 |  |  |  |  |  |
