# VividVR 视频超分与修复：性能文档

本文档汇总当前 SGLang 原生 VividVR 集成已经完成的性能测试结果、测试口径和配置结论。结果是特定硬件与固定输入下的历史实测值，用于回归与容量估算，不应视为任意视频、任意机器上的延迟承诺。

## 1. 结论摘要

当前正式默认配置与实测结果如下：

| 本地资源 | 正式配置 | 正式 record | 总耗时（s） | 模型推理（s） | 相对 R0 模型加速比 |
| --- | --- | --- | ---: | ---: | ---: |
| 1 GPU | `single_gpu_fa_compile` | R2 | 941.516 | 936.101 | 1.18× |
| 2 GPU | `dual_gpu_fa_eager_compile` | R99 | 551.119 | 544.321 | 2.03× |

双卡 R99 相对单卡 R2 的模型推理耗时为约 `1.72×` 加速（`936.101 / 544.321`），Denoise 阶段为约 `2.03×` 加速（`772.493 / 380.176`）。因此，在两张可用 GPU 上，本地服务优先使用双卡 `FA + SP=2 + eager_global + torch.compile`；只有单卡资源时，使用单卡 `FA + torch.compile`。

四卡 R100 的总耗时为 `370.881 s`，可作为扩展性参考，但不属于当前正式默认服务配置，且其原始严格逐帧质量 record 状态为 `quality_failed`。不要仅按该耗时将四卡配置替换为日常默认。

## 2. 固定测试口径

### 2.1 请求与计时范围

| 项目 | 固定值/方式 |
| --- | --- |
| 输入视频 | 960×720、130 帧长视频 |
| 正式请求 | 20 denoising steps，`seed=42`，`num_temporal_process_frames=121` |
| prompt/caption | 固定的逐 temporal clip caption 文件 |
| 精度 | `bfloat16` |
| Warmup | 仅启用 `torch.compile` 的方案先执行一次 1-step warmup；warmup 不计入正式结果 |
| 并发 | 正式方案严格串行执行；单卡对比时同一时刻只有一个单卡推理进程 |
| 总耗时 | 从请求提交前到服务终态的 `total_runtime_seconds` |
| 模型推理耗时 | `pipeline.forward(...)` 覆盖的 `model_inference_runtime_seconds` |
| Denoise | `VividVRMultiClipDenoisingStage` 的同步累计耗时 |
| 质量检查 | 下载生成视频后与固定 reference 逐帧比较，并保留 SSIM 与失败帧比例 |

加速批次的固定输入、caption、reference 与输出结构见 [加速 benchmark 命令](../run_command/vividvr_acceleration_benchmark.md)。所有方案通过 FlowCut 服务完整生命周期执行，包含 caption、回调、对象存储上传、下载及质量比较。

### 2.2 测试环境

| 项目 | 实测环境（2026-07-16） |
| --- | --- |
| GPU | 8 × NVIDIA A100-SXM4-80GB；单个方案最多使用 4 张 |
| 服务器 | 6U GPU Server |
| GPU Driver | 550.90.07 |
| PyTorch / CUDA | `2.9.1+cu128` / PyTorch CUDA 12.8 |
| FlashAttention | FlashAttention 4 `4.0.0b19` |
| Python | `/home/zhiheng/sglang/.venv/bin/python`（Python 3.10.12） |
| 模型路径 | 当次批次为 `/home/zhiheng/ckpts/CogVideoX1.5-5B` 与 `/home/zhiheng/ckpts/Vivid-VR` |

模型路径只是该次测试的资源位置；当前本地使用文档也支持将 checkpoint 放在 `/home/zhiheng/Vivid-VR/ckpts/` 或其他目录，并通过启动参数指定。路径变化本身不代表性能变化。

## 3. 主加速测试结果

R0 是单卡 `SDPA eager` 基线。下表的“平均 step”来自 denoise 阶段；“相对 R0 Denoise 加速比”为 `R0 denoise 秒数 / 当前方案 denoise 秒数`。端到端总耗时还包含长视频准备、VAE decode/trim、stitch 等阶段。

| Record | GPU / 拓扑 | 关键配置 | 总耗时（s） | 模型推理（s） | Denoise（s） | 平均 step（s） | 相对 R0 Denoise 加速比 | 相对 R0 模型加速比 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R0 | 1 / 单卡 | SDPA，eager | 1111.828 | 1102.828 | 928.872 | 46.348 | 1.0000× | 1.00× |
| R1 | 1 / 单卡 | FA，eager | 1041.804 | 1030.589 | 870.188 | 43.424 | 1.0674× | 1.07× |
| R2 | 1 / 单卡 | FA，`torch.compile` | 941.516 | 936.101 | 772.493 | 38.621 | 1.2024× | 1.18× |
| R3 | 2 / SP=2 | FA-SP，`torch.compile` | 551.136 | 547.222 | 383.512 | 19.171 | 2.4220× | 2.02× |
| R99 | 2 / SP=2 | 已实现加速组合，FA-SP，`torch.compile` | 551.119 | 544.321 | 380.176 | 19.003 | 2.4433× | 2.03× |
| R4 | 4 / SP=4 | FA-SP，`torch.compile` | 380.823 | 374.242 | 201.709 | 10.081 | 4.6050× | 2.95× |
| R5 | 4 / CFG=2 × SP=2 | FA-SP，`torch.compile` | 380.814 | 369.751 | 194.807 | 9.733 | 4.7682× | 2.98× |
| R100 | 4 / CFG=2 × SP=2 | 已实现加速组合，FA-SP，`torch.compile` | 370.881 | 365.067 | 195.652 | 9.779 | 4.7476× | 3.02× |

原始正式记录：

- R0–R100：`Vivid_Acceptance/acceleration_benchmark/vividvr_accel_full_warmup1_20260716/records/`
- 批次汇总与全部 Stage 计时：[加速测试耗时总结](../docs_analysis/acceleration_benchmark_results_20260716.md)

## 4. 各加速项的可观察收益

以下为在同一主 benchmark 中逐项对照得到的结果，比较对象为模型推理耗时。

| 加速项 | 对比 | 增量模型加速比 | 结论 |
| --- | --- | ---: | --- |
| FlashAttention | R1 / R0 | 1.0701× | 单卡端到端耗时和 GPU·秒均降低。 |
| `torch.compile` | R2 / R1 | 1.1009× | 带来明确收益，但最大单卡峰值显存小幅增加。 |
| 双卡 SP | R3 / R2 | 1.7106× | 显著降低延迟；GPU·秒和最大单卡峰值显存增加。 |
| 双卡已实现加速组合 | R99 / R3 | 1.0053× | 在既有双卡拓扑上进一步小幅收益。 |
| 四卡已实现加速组合 | R100 / R5 | 1.0128× | 在四卡扩展实验上进一步小幅收益。 |
| modulation fusion | R6 / R2 | 0.9992× | 降低峰值显存，但本轮没有形成端到端加速。 |

Cache-DiT、TeaCache 和通用量化在该批次均未实现，因此没有正式性能结论。

## 5. 默认配置与实验配置的边界

| 配置/能力 | 状态 | 说明 |
| --- | --- | --- |
| `single_gpu_fa_compile` | 正式默认 | 单卡 `fa + torch.compile`，对应 R2 口径。 |
| `dual_gpu_fa_eager_compile` | 正式默认 | 双卡 `SP=2 + fa + eager_global + torch.compile`；运行时有效 backend 为 `fa_sp`，对应 R99 口径。 |
| `dual_gpu_sdpa_eager_compile` | 正式兼容验证 | 请求 backend 为 `sdpa`，双卡仍走 Ulysses 分布式 joint-attention，运行时为 `sdpa_sp`；不是日常默认。 |
| 4 GPU R100 | 扩展性参考 | 可降低固定输入延迟，但不替换当前单卡/双卡默认配置。 |
| `--vae-sp`（tiled decode） | 实验性 opt-in | 专项性能验收通过，但默认关闭。 |
| `--vae-encode-sp`（tiled encode） | 实验性 opt-in | bitwise 正确性通过，但正式性能门槛未通过，默认关闭。 |

双卡配置中的 `SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global` 是已验收的 full global control context 语义，不能为了追求历史实验数字改用 `deferred_global`。

## 6. VAE 空间 tile 并行专项结果

### 6.1 tiled decode：性能通过，但默认关闭

以下 treatment 以 R99/R100 为各自历史 control，均使用相同的 `130f / 20 step` 正式服务口径：

| Treatment | 拓扑 | Decode/Trim（s / speedup） | 模型推理（s / speedup） | 总耗时（s / speedup） | 验收结论 |
| --- | --- | ---: | ---: | ---: | --- |
| R99 + VAE SP | SP=2 | 58.938 / 1.7014× | 502.578 / 1.0831× | 510.931 / 1.0787× | 专项通过（人工质量豁免） |
| R100 + VAE SP | CFG=2 × SP=2 | 60.179 / 1.6914× | 334.402 / 1.0917× | 341.004 / 1.0876× | 专项通过（人工质量豁免） |

固定 latent 的 SP2、SP4、CFG2×SP2 均与串行结果 bitwise equal。正式记录的严格逐帧 comparator 仍标为 `quality_failed`，原因是阈值附近的少量 SSIM 差异；抽帧人工检查未发现 tile 接缝、闪烁、颜色漂移或 trim/stitch 边界异常，并按记录的人工豁免口径通过。

该能力继续保持 `vae_sp=False` 默认关闭：每个 SP rank 仍会恢复完整 decoded tile 集，gather staging 与 replicated merge 有额外显存成本。详情见 [VAE spatial tile 并行验收记录](../distribute/vividvr_vae_spatial_tile_parallel_acceptance_20260716.md)。

### 6.2 tiled encode：正确但性能门槛未通过

SP2、SP4、CFG2×SP2 的 posterior moments 和等价 sampled latents 都已 bitwise 验证正确，但三组正式 treatment 都存在 Long Clip Preparation 或 Decode/Trim 门槛失败。因此 `--vae-encode-sp` 不进入默认配置。

| Treatment | 拓扑 | 总耗时（s） | 相对 control 总耗时 | 性能门槛 |
| --- | --- | ---: | ---: | --- |
| R99_ENCODE_SP | SP=2 | 491.181 | 1.0402× | 未通过 |
| R100_ENCODE_SP | CFG=2 × SP=2 | 310.903 | 1.0968× | 未通过 |
| R101_ENCODE_SP4 | SP=4 | 270.743 | 1.1479× | 未通过 |

详见 [VAE tiled encode 并行验收记录](../distribute/vividvr_vae_spatial_tiled_encode_parallel_acceptance_20260717.md)。

## 7. 复测与回归建议

1. 日常性能回归固定使用相同 reference 对象的 `130f / 20 step` 口径；`50 step` 只用于阶段性最终回归。
2. 每个启用 `torch.compile` 的候选方案只做一次 `1 step` warmup；eager 方案不 warmup，正式请求仍固定为 `20 step`。
3. 比较单卡方案时，确保没有其他单卡推理进程并发占用 GPU。
4. 新结果必须同时记录 `total_runtime_seconds` 和 `model_inference_runtime_seconds`，并保留输入、caption、reference、命令、版本与质量报告。
5. 任何默认参数、backend、SP 语义或性能结论发生变化时，同步更新本文件、[基础使用文档](vividvr_basic_usage.md) 与 `AGENTS.md`。

完整 benchmark 命令、产物目录和 JSON 字段说明见 [Vivid-VR benchmark 说明](../run_vivid_benchmark.md) 与 [加速 benchmark 命令](../run_command/vividvr_acceleration_benchmark.md)。
