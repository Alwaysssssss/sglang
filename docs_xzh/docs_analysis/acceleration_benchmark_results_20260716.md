# VividVR 加速测试耗时总结

## 测试口径

| 项目 | 口径 |
| --- | --- |
| 输入帧数 | 130 帧 |
| 正式推理 | 20 step |
| Warmup | 仅 `torch.compile` 方案执行 1 step；不计入正式结果 |
| 总耗时 | 从请求提交前到服务终态的 `total_runtime_seconds` |
| 模型推理耗时 | `pipeline.forward(...)` 对应的完整模型阶段耗时 |
| Denoise 耗时 | `VividVRMultiClipDenoisingStage` 耗时 |
| 相对 R0 加速比 | R0 模型推理耗时 / 当前方案模型推理耗时 |

## 总体耗时

| 方案 | 关键配置 | 总耗时（s） | 模型推理耗时（s） | Denoise（s） | 平均 Step（s） | 相对 R0 模型加速比 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| R0 | 单卡 SDPA eager | 1111.828 | 1102.828 | 928.872 | 46.348 | 1.00× |
| R1 | 单卡 FA eager | 1041.804 | 1030.589 | 870.188 | 43.424 | 1.07× |
| R2 | 单卡 FA + `torch.compile` | 941.516 | 936.101 | 772.493 | 38.621 | 1.18× |
| R3 | 双卡 SP=2 + FA-SP + `torch.compile` | 551.136 | 547.222 | 383.512 | 19.171 | 2.02× |
| R4 | 四卡 SP=4 + FA-SP + `torch.compile` | 380.823 | 374.242 | 201.709 | 10.081 | 2.95× |
| R5 | 四卡 CFG=2 × SP=2 + FA-SP + `torch.compile` | 380.814 | 369.751 | 194.807 | 9.733 | 2.98× |
| R6 | 单卡 FA + `torch.compile` + modulation fusion | 941.453 | 936.864 | 775.704 | 38.781 | 1.18× |
| R99 | 双卡全部已实现加速 | 551.119 | 544.321 | 380.176 | 19.003 | 2.03× |
| R100 | 四卡全部已实现加速 | 370.881 | 365.067 | 195.652 | 9.779 | 3.02× |

## 完整 Stage 耗时

Stage 横向总表的单位均为秒。列名与代码中的完整 Stage 类名对应如下：

| 表格列名 | Stage 类名 |
| --- | --- |
| Input Validation | `VividVRInputValidationStage` |
| Prompt Preparation | `VividVRPromptPreparationStage` |
| Window Planning | `VividVRTemporalWindowPlanningStage` |
| Long Clip Preparation | `VividVRLongClipPreparationStage` |
| Timestep Preparation | `VividVRTimestepPreparationStage` |
| Multi-Clip Denoising | `VividVRMultiClipDenoisingStage` |
| Decode/Trim | `VividVRMultiClipDecodeTrimStage` |
| Stitch/Postprocess | `VividVRTemporalStitchPostprocessStage` |

| 方案 | Input Validation | Prompt Preparation | Window Planning | Long Clip Preparation | Timestep Preparation | Multi-Clip Denoising | Decode/Trim | Stitch/Postprocess |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R0 | 0.000 | 0.000 | 0.000 | 60.293 | 0.000 | 928.872 | 111.985 | 0.184 |
| R1 | 0.000 | 0.000 | 0.000 | 59.664 | 0.000 | 870.188 | 99.267 | 0.188 |
| R2 | 0.000 | 0.001 | 0.000 | 61.335 | 0.001 | 772.493 | 100.704 | 0.172 |
| R3 | 0.000 | 0.000 | 0.000 | 61.081 | 0.001 | 383.512 | 101.167 | 0.144 |
| R4 | 0.001 | 0.002 | 0.000 | 63.102 | 0.001 | 201.709 | 107.193 | 0.245 |
| R5 | 0.000 | 0.001 | 0.001 | 62.421 | 0.001 | 194.807 | 110.560 | 0.239 |
| R6 | 0.000 | 0.000 | 0.000 | 60.596 | 0.001 | 775.704 | 98.781 | 0.231 |
| R99 | 0.000 | 0.000 | 0.000 | 61.824 | 0.001 | 380.176 | 100.274 | 0.190 |
| R100 | 0.000 | 0.000 | 0.000 | 65.857 | 0.001 | 195.652 | 101.786 | 0.132 |

## 实验环境记录

| 项目 | 值 |
| --- | --- |
| Batch | `vividvr_accel_full_warmup1_20260716` |
| 机器型号 | `6U GPU Server` |
| GPU 型号与可用数量 | 8 × NVIDIA A100-SXM4-80GB；本轮单个方案最多使用 4 张 |
| CUDA 版本 | PyTorch CUDA `12.8`；系统无 `nvcc`，CUDA Toolkit 版本未单独确认 |
| Driver 版本 | `550.90.07` |
| PyTorch 版本 | `2.9.1+cu128` |
| FlashAttention 版本 | FlashAttention 4 `4.0.0b19` |
| sglang commit | N/A；本批次正式 JSON 未记录 commit，不根据文件时间戳推测 |
| Python 路径 | `/home/zhiheng/sglang/.venv/bin/python`（Python 3.10.12） |
| 模型/checkpoint 路径 | 模型：`/home/zhiheng/ckpts/CogVideoX1.5-5B`；Vivid-VR：`/home/zhiheng/ckpts/Vivid-VR` |
| Dtype | `bfloat16` |
| Compile mode | R0/R1：eager；R2–R6、R99、R100：`torch.compile`；R7–R9：未实现 |
| 计时方式 | 总耗时使用 `total_runtime_seconds`；模型耗时使用 `model_inference_runtime_seconds`；Stage 使用同步 profiling 累计值 |
| 显存统计方式 | NVML 采样；模块收益表使用 `max_single_gpu_peak_gib` |
| Stage profiling 是否同步 | 是；GPU 同步后记录 Stage 耗时 |
| Perf/report 目录 | `/home/zhiheng/sglang/Vivid_Acceptance/acceleration_benchmark/vividvr_accel_full_warmup1_20260716` |
| 结果视频目录 | 上述批次目录下 `requests/*/downloaded.mp4` |

## 模块收益结论

延迟增量加速比使用 Control 与 Treatment 的模型推理耗时计算。GPU·秒变化、最大单卡峰值显存变化和质量变化均为 Treatment 相对 Control 的差值。

| 加速模块 | Treatment | Control | 延迟增量加速比 | GPU·秒变化 | 最大单卡峰值显存变化 | 质量变化 | 正式结论 |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| Attention backend | R1 | R0 | 1.0701× | -72.238 s（-6.55%） | -0.215 GiB | 人工验收 PASS；ΔSSIM mean=-0.000047 | FA 相对 SDPA 具有明确端到端收益，延迟和 GPU·秒均下降。 |
| `torch.compile` | R2 | R1 | 1.1009× | -94.488 s（-9.17%） | +0.328 GiB | 人工验收 PASS；ΔSSIM mean=+0.000633 | Compile 提供明确端到端收益，代价是最大单卡峰值显存小幅增加。 |
| 双卡 SP | R3 | R2 | 1.7106× | +158.343 s（+16.92%） | +0.770 GiB | 人工验收 PASS；ΔSSIM mean=-0.000278 | 双卡 SP 显著降低延迟，但 GPU·秒和最大单卡峰值显存增加。 |
| SP 2→4 卡扩展 | R4 | R3 | 1.4622× | +402.524 s（+36.78%） | +0.594 GiB | 人工验收 PASS；ΔSSIM mean=+0.000178 | SP 扩展到四卡继续降低延迟，但资源成本明显上升。 |
| CFG parallel | R5 | R3 | 1.4800× | +384.560 s（+35.14%） | +0.588 GiB | 人工验收 PASS；ΔSSIM mean=-0.000087 | 四卡 CFG×SP 显著降低延迟，但 GPU·秒明显增加。 |
| 四卡并行拓扑 | R5 | R4 | 1.0121× | -17.964 s（-1.20%） | -0.006 GiB | 人工验收 PASS；ΔSSIM mean=-0.000266 | 同为四卡时，CFG=2 × SP=2 略快于 SP=4，资源成本也略低。 |
| 算子融合 | R6 | R2 | 0.9992× | +0.763 s（+0.08%） | -0.625 GiB | 人工验收 PASS；ΔSSIM mean=-0.000198 | Modulation fusion 本轮降低了峰值显存，但未形成端到端加速。 |
| Cache-DiT | R7 | R2 | N/A | N/A | N/A | N/A | 当前未实现，无正式收益结论。 |
| TeaCache | R8 | R2 | N/A | N/A | N/A | N/A | 当前未实现，无正式收益结论。 |
| 通用量化 | R9 | R2 | N/A | N/A | N/A | N/A | 当前未实现，无正式收益结论。 |
| 双卡综合最快方案 | R99 | R3 | 1.0053× | -5.803 s（-0.53%） | -0.373 GiB | 人工验收 PASS；ΔSSIM mean=-0.000129 | 叠加已实现模块后取得小幅正收益，并降低峰值显存。 |
| 四卡综合最快方案 | R100 | R5 | 1.0128× | -18.737 s（-1.27%） | -0.275 GiB | 人工验收 PASS；ΔSSIM mean=-0.000090 | 四卡综合方案相对最快基础拓扑取得小幅正收益。 |
| 综合方案 2→4 卡扩展 | R100 | R99 | 1.4910× | +371.627 s（+34.14%） | +0.686 GiB | 人工验收 PASS；ΔSSIM mean=-0.000048 | 综合方案扩展到四卡显著降低延迟，但 GPU·秒和峰值显存增加。 |

质量栏采用本轮人工验收结论：所有已执行视频结果均视为通过。该结论与正式 JSON 中更严格的 `pass_compare` 阈值判定不是同一口径；SSIM 差值保留用于量化比较。

## VAE tiled encode SP Treatment（2026-07-17）

以下三条是额外的 Treatment-only 实验，不改写上面的历史总体排行。每条 Control 都是对应拓扑的历史 **decode-only** 正式 record：R99_ENCODE_SP 对 R99_VAE_SP，R100_ENCODE_SP 对 R100_VAE_SP，R101_ENCODE_SP4 对 R101_VAE_SP4。Control 未重跑，前后 SHA-256 与 `mtime_ns` 一致。

Model/Total 列写作 “treatment 秒数 / 相对 Control speedup”，Denoise 与 Decode/Trim 写作 “treatment 秒数 / 相对 Control 回归”。Bitwise gate 要求完整 moments 和等价 generator sampled latents 均为 `torch.equal`。

| Treatment | GPU / topology | Long Clip Preparation（s） | Stage speedup | Model（s / speedup） | Total（s / speedup） | Denoise（s / 回归） | Decode/Trim（s / 回归） | GPU·秒 | SSIM mean / min / failed ratio | Bitwise gate | Performance gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| R99_ENCODE_SP | 2 / SP2 | 42.176 | 1.4441× | 486.321 / 1.0334× | 491.181 / 1.0402× | 380.798 / -0.10% | 61.428 / +4.23% | 982.362 | 0.984562 / 0.976291 / 1.5385% | PASS | **FAIL** |
| R100_ENCODE_SP | 4 / CFG2×SP2 | 47.257 | 1.5464× | 309.705 / 1.0797× | 310.903 / 1.0968× | 195.837 / -1.47% | 64.287 / +6.83% | 1243.610 | 0.984504 / 0.978161 / 1.5385% | PASS | **FAIL** |
| R101_ENCODE_SP4 | 4 / SP4 | 28.748 | 2.2414× | 265.806 / 1.1248× | 270.743 / 1.1479× | 203.575 / +0.30% | 30.963 / +3.22% | 1082.971 | 0.984169 / 0.977092 / 1.5385% | PASS | **FAIL** |

正式性能门槛由四项组成：Long Clip Preparation speedup（SP2/CFG2×SP2 至少 1.5×，SP4 至少 2.5×）、模型推理耗时改善、Denoise 回归不超过 3%、Decode/Trim 回归不超过 3%。R99 的 Long Clip 与 Decode/Trim 失败，R100 的 Decode/Trim 失败，R101 的 Long Clip 与 Decode/Trim 失败，所以本轮 encode SP 正式性能验收未通过，能力继续保持实验性和默认关闭。

## 最快方案

| GPU 数量 | 最快方案 | 总耗时（s） | 模型推理耗时（s） | Denoise（s） | 相对 R0 模型加速比 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | R2：FA + `torch.compile` | 941.516 | 936.101 | 772.493 | 1.18× |
| 2 | R99：SP=2 + FA-SP + `torch.compile` + modulation fusion | 551.119 | 544.321 | 380.176 | 2.03× |
| 4 | R100：CFG=2 × SP=2 + FA-SP + `torch.compile` + modulation fusion | 370.881 | 365.067 | 195.652 | 3.02× |

## VAE 空间 Tile 并行专项验收补充（2026-07-16 至 2026-07-17）

本节补充 VAE tiled decode、VAE tiled encode 和纯净 SP 扩展的专项验收结果。它们使用独立的 Control/Treatment 配对或专项服务口径，不能回填或改写上文 R0–R100 主表的历史总体排行。除非另有说明，正式请求均为相同的 `130f / 20 step / seed 42` 长视频服务请求。

### VAE tiled decode SP：通过专项验收，但保持 opt-in

CogVideoX VAE tiled decode 已实现 SP subgroup 内的空间 tile 分配、transport 和原序 merge。固定 latent 的 SP2、SP4、CFG2×SP2 验证均与串行结果 bitwise equal；collective 仅使用对应 SP subgroup。以下是与同拓扑 decode-only Control 的正式 A/B 结果：

| Treatment | Topology | Decode/Trim（s / speedup） | 模型推理（s / speedup） | 总耗时（s / speedup） | GPU·秒（treatment / control） |
| --- | --- | ---: | ---: | ---: | ---: |
| R99 + VAE SP | SP2 | 58.938 / 1.7014× | 502.578 / 1.0831× | 510.931 / 1.0787× | 1021.862 / 1102.238 |
| R100 + VAE SP | CFG2×SP2 | 60.179 / 1.6914× | 334.402 / 1.0917× | 341.004 / 1.0876× | 1364.017 / 1483.526 |

R99 的一次独立 formal 重复运行得到 `58.198 s` Decode/Trim（`1.7230×`）和 `510.965 s` 总耗时（`1.0786×`），证明主要收益可复现。端到端人工检查未见 tile 接缝、闪烁、颜色漂移或 trim/stitch 边界异常。严格逐帧 compare 仍会将原始 record 标为 `quality_failed`：R99/R100 均按已记录的人工豁免口径通过，原始机器门禁状态未被修改。

该能力当前仍为 `vae_sp=False` 默认关闭的实验性 opt-in 开关：各 SP rank 仍需恢复完整 decoded tile 集，gather staging 与 replicated merge 会带来额外显存；因此不替换 `single_gpu_fa_compile`、`dual_gpu_fa_eager_compile` 或其他正式服务默认配置。完整证据见 [VAE spatial tile parallel 验收记录](../distribute/vividvr_vae_spatial_tile_parallel_acceptance_20260716.md)。

### VAE tiled encode SP：bitwise 正确，但未通过性能门槛

上文的 `VAE tiled encode SP Treatment` 表是 encode 专项的完整性能结果。三种拓扑的 posterior moments 与等价 sampled latents 均为 bitwise equal，且模型总耗时均改善；但 Long Clip Preparation 或 Decode/Trim 未满足既定门槛，三条 treatment 的正式性能验收均为 FAIL。因此 `--vae-encode-sp` 与 `--vae-sp` 相互独立，且继续默认关闭，不进入任何正式默认配置。完整门禁和失败分析见 [VAE spatial tiled encode 并行验收记录](../distribute/vividvr_vae_spatial_tiled_encode_parallel_acceptance_20260717.md)。

### R0 纯净基线下的 SP2/SP4 扩展

为隔离多卡 SP 拓扑和 VAE tiled encode/decode 空间并行的端到端收益，以下测试相对历史 R0 关闭了 `torch.compile`、modulation fusion、CFG parallel、cache 和 quantization；两组 treatment 均为 eager 且没有 warmup。R0 使用单卡 `SDPA eager`，SP2/SP4 在多卡运行时有效 backend 分别为 `sdpa_sp`。

| 方案 | 准备阶段（s / speedup） | Decode/Trim（s / speedup） | Denoise（s / speedup） | 模型推理（s / speedup） | 端到端（s / speedup） | 相对 R0 资源效率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R0，单卡 | 60.293 / 1.0000× | 111.985 / 1.0000× | 928.872 / 1.0000× | 1102.828 / 1.0000× | 1111.828 / 1.0000× | 1.0000× |
| SP2 | 36.663 / 1.6445× | 57.720 / 1.9401× | 471.769 / 1.9689× | 567.547 / 1.9431× | 571.113 / 1.9468× | 0.9734× |
| SP4 | 22.738 / 2.6516× | 29.041 / 3.8560× | 258.643 / 3.5913× | 311.990 / 3.5348× | 320.751 / 3.4663× | 0.8666× |

SP2 是延迟与资源效率更均衡的选择；SP4 将端到端延迟再降低到 `320.751 s`，但以更多 GPU·秒为代价。两组请求的服务生命周期（推理、上传、进度查询和 callback）均成功完成。严格逐帧门禁在 R0、SP2 和 SP4 上均存在极少量失败帧，属于质量 comparator 的严格状态，而非服务或性能采集失败。完整测试口径见 [R0 基线 VAE SP2/SP4 纯净服务测试验收](../distribute/vividvr_r0_vae_sp_clean_benchmark_20260717.md)。

### 四卡 VAE-SP 拓扑选择：SP4 优于 CFG2×SP2

在同为四卡、均开启 `FA-SP`、`torch.compile`、modulation/residual fusion 和 VAE spatial tile parallel 的条件下，纯 SP4 优于 CFG2×SP2：

| 指标 | 纯 SP4 | CFG2×SP2 | SP4 相对 CFG2×SP2 |
| --- | ---: | ---: | ---: |
| 总耗时 | 310.786 s | 341.004 s | 1.0972× |
| 模型推理 | 298.988 s | 334.402 s | 1.1184× |
| Denoise | 202.965 s | 198.751 s | 0.9792× |
| Decode/Trim | 29.996 s | 60.179 s | 2.0062× |
| VAE decode | 29.699 s | 59.919 s | 2.0175× |

CFG2×SP2 的 denoise 快 `2.12%`，但 SP4 的 VAE decode 约减半，足以使整体更快。SP4 的 SSIM mean/min 分别为 `0.984631 / 0.979781`，优于 CFG2×SP2 的 `0.984603 / 0.977484`；严格 comparator 因 `1/130` 帧低于阈值保留 `quality_failed`，但按与 R100 相同的已确认容差口径，专项质量验收通过。完整对比见 [纯 SP4 与 CFG2×SP2 四卡性能对比](../distribute/vividvr_vae_sp4_vs_cfg2_sp2_benchmark_20260717.md)。
