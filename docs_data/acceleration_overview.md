# VividVR 加速实验统计

本文用于统一记录 VividVR 当前已实现加速和后续正式加速实验的性能、资源效率与质量结果。

本文只保留能够回答关键加速模块收益的推荐组合，不对所有开关做全排列。主表固定使用同一组长视频正式输入；如果以后增加其他输入规格，应在附录中复制同一套表格，不再按分辨率扩展主表列。

## 1. 填写与计时约定

| 项目 | 约定 |
| --- | --- |
| 耗时单位 | 秒，保留至少 2 位小数；原始 JSON 中保留完整精度 |
| 显存单位 | GiB，记录每张 GPU 的峰值；主表填写最大单卡峰值 |
| 总耗时 | 从正式请求开始到结果文件写完，以 `total_runtime_seconds` 为准 |
| 模型推理耗时 | 以 `model_inference_runtime_seconds` 为准，是总体加速比的主计算字段 |
| Denoise 耗时 | 长视频主链中 `VividVRMultiClipDenoisingStage` 的同步耗时 |
| Stage 耗时 | 以同步后的 pipeline stage profiling 为准；多 clip 记录整次请求内各 Stage 累计值 |
| 正式计时 | 同一服务实例和同一配置先完整运行一次，再以第二次完整请求作为正式结果；首次运行不计入正式加速比 |
| 单卡公平性 | 正式单卡实验期间只允许一个单卡推理进程运行，避免并发占用造成不公平对比 |
| 质量口径 | 使用同一 reference 比较全部输出，至少记录 `SSIM mean`、`SSIM min` 和 `failed frame ratio` |
| N/A | 当前方案不涉及该指标，或者现有 profiler 无法可靠拆分；不得用估算值代替 |

### 1.1 加速与资源效率计算

所有累计加速比统一以 R0 原始性能基线计算：

```text
累计加速比 = R0 模型推理耗时 / 当前方案模型推理耗时
```

每个模块的增量收益按方案表中指定的对照计算：

```text
模块增量加速比 = 指定对照方案模型推理耗时 / 当前方案模型推理耗时
```

多卡方案同时记录资源成本：

```text
GPU·秒 = GPU 数量 × 当前方案模型推理耗时

相对 R0 资源效率 = R0 GPU·秒 / 当前方案 GPU·秒
```

解释多卡结果时，必须同时报告延迟加速比和 GPU·秒。延迟下降但 GPU·秒上升，表示单请求更快，但资源效率未必更高。

## 2. 固定实验口径

主表固定使用当前 Phase E 长视频正式 benchmark，不在每个方案中重复填写模型权重、推理步数和输入规格。

| 项目 | 固定值 |
| --- | --- |
| 输入视频 | `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4` |
| Caption | `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt` |
| Prompt | `/home/zhiheng/Vivid-VR/input/720p/prompt.txt` |
| Reference | `/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4` |
| 输入帧数 | 130 |
| Temporal process frames | 121 |
| 推理步数 | 20；50 step 仅用于阶段性最终回归，不进入本统计主表 |
| Seed | 42 |
| Guidance scale | 6 |
| Restoration guidance scale | -1.0 |
| Upscale | 1.0 |
| Dtype | `bfloat16` |
| Prompt embedding 长度 | 226；不得回退到 512 |
| VAE tiling | 保持当前正式配置，tile sample min height/width 为 240/360 |
| 双卡及四卡 SP connector | `SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global`，使用 full global control context |
| Compile 正式口径 | 固定形状 1-step warmup 后记录下一次完整请求；同时保存 compile mode 和实际生效状态 |
| 质量最低要求 | `pass_compare=true`，并检查质量指标是否相对稳定基线发生明显漂移 |

除被测加速模块外，所有方案必须保持上述输入、模型、dtype、seed、调度器、VAE、前后处理和服务请求语义一致。

## 3. 正式实验方案定义

下表是本轮固定的正式对比集合。R6–R9、R99 和 R100 虽然尚未全部实现，但同样属于正式实验计划，不再设置额外的候选层级。

| 编号 | 关键加速方案 | 增益对照 | 主要统计目标 | 当前状态 |
| --- | --- | --- | --- | --- |
| R0 | 单卡 SDPA eager | — | 原始性能基线 | 已实现，待按统一口径记录 |
| R1 | 单卡 FA eager | R0 | Attention backend 增益 | 已实现，待按统一口径记录 |
| R2 | 单卡 FA + `torch.compile` | R1 | `torch.compile` 增益 | 已实现；当前单卡正式配置 |
| R3 | 双卡 SP=2 + FA-SP + `torch.compile` | R2 | 双卡 SP 延迟收益和资源成本 | 已实现；当前双卡正式配置 |
| R4 | 四卡 SP=4 + FA-SP + `torch.compile` | R3 | SP 从 2 卡扩展到 4 卡的收益 | 既有实验未形成稳定主线，需统一复测 |
| R5 | 四卡 CFG=2 × SP=2 + FA-SP + `torch.compile` | R3、R4 | CFG parallel 增量收益及同资源拓扑对比 | 已实现并完成正式链路验收 |
| R6 | R2 + 算子融合 | R2 | 算子融合增益 | 待实现和正式实验 |
| R7 | R2 + Cache-DiT | R2 | Cache-DiT 增益及质量影响 | 待实现和正式实验 |
| R8 | R2 + TeaCache | R2 | TeaCache 增益及质量影响 | 待实现和正式实验 |
| R9 | R2 + 通用量化 | R2 | 量化增益、显存收益及质量影响 | 量化方法待定义，随后实现和正式实验 |
| R99 | 双卡综合最快方案 | R3 | 双卡全部有效加速的组合收益 | 待前置模块验证后正式实验 |
| R100 | 四卡综合最快方案 | R4、R5 中更快者 | 四卡全部有效加速的组合收益 | 待前置模块验证后正式实验 |

### 3.1 并行方案的比较关系

纯 SP 扩展效率：

```text
R2（单卡）→ R3（SP=2）→ R4（SP=4）
```

CFG parallel 增量收益：

```text
R3（双卡 SP=2）→ R5（四卡 CFG=2 × SP=2）
```

同为四卡时的并行拓扑对比：

```text
R4（SP=4）↔ R5（CFG=2 × SP=2）
```

R5 相对 R3 的结果用于描述“增加 CFG parallel 后的延迟变化”，但该比较同时从两卡增加到四卡，必须结合 GPU·秒解释。R4 与 R5 的比较使用相同 GPU 数量，是判断四卡拓扑效率的主要依据。

### 3.2 综合最快方案组合规则

R99 固定以 R3 的双卡 SP=2 + FA-SP + `torch.compile` 为基础；R100 先比较 R4 的 SP=4 与 R5 的 CFG=2 × SP=2，选择端到端耗时更短且质量达标的四卡基础拓扑。

R99 和 R100 只叠加已经验证满足以下条件的模块：

- 与对应多卡拓扑兼容。
- 质量结果达标。
- 端到端模型推理耗时有正收益。

算子融合与通用量化满足条件后可以加入。Cache-DiT、TeaCache 和关闭缓存固定三选一，不默认同时开启 Cache-DiT 与 TeaCache。某个模块在多卡下出现负收益时，不加入综合最快方案，但仍保留其独立正式实验结果。

## 4. 总体结果

| 方案 | 总耗时 | 模型推理耗时 | Denoise 耗时 | 相对 R0 加速比 | 模块增量加速比 | GPU·秒 | 相对 R0 资源效率 | 最大单卡峰值显存 | 质量结果 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| R0 |  |  |  | 1.0000× | — |  | 1.0000× |  |  |
| R1 |  |  |  |  | R0 / R1 |  |  |  |  |
| R2 |  |  |  |  | R1 / R2 |  |  |  |  |
| R3 |  |  |  |  | R2 / R3 |  |  |  |  |
| R4 |  |  |  |  | R3 / R4 |  |  |  |  |
| R5 |  |  |  |  | R3 / R5 |  |  |  |  |
| R6 |  |  |  |  | R2 / R6 |  |  |  |  |
| R7 |  |  |  |  | R2 / R7 |  |  |  |  |
| R8 |  |  |  |  | R2 / R8 |  |  |  |  |
| R9 |  |  |  |  | R2 / R9 |  |  |  |  |
| R99 |  |  |  |  | R3 / R99 |  |  |  |  |
| R100 |  |  |  |  | 最快(R4, R5) / R100 |  |  |  |  |

“质量结果”至少使用下面的紧凑格式：

```text
PASS/FAIL；SSIM mean=...；SSIM min=...；failed frame ratio=...
```

R4 与 R5 还需在结论表中单独计算同为四卡时的耗时比和 GPU·秒差异，不能只填写各自相对 R0 的累计值。

## 5. Stage 耗时明细

下表使用当前 VividVR 长视频 pipeline 的原生 Stage。单位统一为秒。

| Stage | 说明 | R0 | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R99 | R100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `VividVRInputValidationStage` | 输入和参数校验 |  |  |  |  |  |  |  |  |  |  |  |  |
| `VividVRPromptPreparationStage` | Prompt/caption 上下文准备 |  |  |  |  |  |  |  |  |  |  |  |  |
| `VividVRTemporalWindowPlanningStage` | 长视频 temporal clip 切分与窗口规划 |  |  |  |  |  |  |  |  |  |  |  |  |
| `VividVRLongClipPreparationStage` | 各 clip 的文本、condition、latent 和 tiling 准备 |  |  |  |  |  |  |  |  |  |  |  |  |
| `VividVRTimestepPreparationStage` | Scheduler timestep 准备 |  |  |  |  |  |  |  |  |  |  |  |  |
| `VividVRMultiClipDenoisingStage` | 多 clip timestep 级联合去噪和 latent merge |  |  |  |  |  |  |  |  |  |  |  |  |
| `VividVRMultiClipDecodeTrimStage` | 各 clip VAE decode、首 3 帧丢弃和 trim |  |  |  |  |  |  |  |  |  |  |  |  |
| `VividVRTemporalStitchPostprocessStage` | Temporal stitch、crop padding、AdaIN/reference color fix 及后处理 |  |  |  |  |  |  |  |  |  |  |  |  |
| 未归类开销 | Stage 间同步、调度、日志及无法归类的其他开销 |  |  |  |  |  |  |  |  |  |  |  |  |
| 模型推理总计 | `model_inference_runtime_seconds` |  |  |  |  |  |  |  |  |  |  |  |  |

未归类开销按下式计算：

```text
未归类开销 = model_inference_runtime_seconds - 所有已记录 Stage 耗时之和
```

如果 profiler 没有同步 CUDA，必须先修正统计方式，不能直接把异步提交耗时填入正式表格。单 clip 回归需要复用本表时，不存在的长视频 Stage 填 N/A。

## 6. Denoising 核心耗时明细

本表不再区分输入分辨率。`step` 指全局 denoising timestep；多 clip 必须在同一个全局 timestep 全部完成后才推进一次计数。

| 方案 | Temporal clip 数 | 推理步数 | Denoise 总耗时 | 占模型推理比例 | 平均每 step | Steady step 中位数 | SP 通信耗时 | CFG 通信耗时 | Cache 执行/跳过 | 最大单卡峰值显存 | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| R0 |  | 20 |  |  |  |  | N/A | N/A | N/A |  |  |
| R1 |  | 20 |  |  |  |  | N/A | N/A | N/A |  |  |
| R2 |  | 20 |  |  |  |  | N/A | N/A | N/A |  |  |
| R3 |  | 20 |  |  |  |  |  | N/A | N/A |  |  |
| R4 |  | 20 |  |  |  |  |  | N/A | N/A |  |  |
| R5 |  | 20 |  |  |  |  |  |  | N/A |  |  |
| R6 |  | 20 |  |  |  |  | N/A | N/A | N/A |  |  |
| R7 |  | 20 |  |  |  |  | N/A | N/A |  |  |  |
| R8 |  | 20 |  |  |  |  | N/A | N/A |  |  |  |
| R9 |  | 20 |  |  |  |  | N/A | N/A | N/A |  |  |
| R99 |  | 20 |  |  |  |  |  | N/A |  |  |  |
| R100 |  | 20 |  |  |  |  |  |  |  |  |  |

填写说明：

- `Denoise 总耗时` 对应 `VividVRMultiClipDenoisingStage`。
- `平均每 step` 使用 Denoise 总耗时除以全局 timestep 数；它用于总体核对，不代替 steady-state 统计。
- `Steady step 中位数` 应排除首次 compile、profiler flush 或其他已证明异常的 step。
- SP 通信耗时记录 Ulysses All-to-All、prefix gather 等能够可靠归类的通信。
- CFG 通信耗时记录 cond/uncond 合成相关 collective。
- R7 和 R8 必须填写实际执行与跳过的 timestep 数；其他方案填 N/A。
- 无法从 profiler 中可靠拆分的 SP/CFG 通信耗时填 N/A，不能用整段 Denoise 耗时估算。
- 不再统计 CFG cond/uncond forward 数和平均每 forward 耗时，因为串行 batch CFG 与 CFG parallel 的执行语义不同。

## 7. 实验环境与结论

### 7.1 实验环境记录

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

方案名称只能表达请求配置；正式报告还必须记录实际生效状态。

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

多卡请求 backend 仍只填写 `fa` 或 `sdpa`；Effective backend 应按实际运行时记录为 `fa_sp` 或 `sdpa_sp`。R6–R9 必须补充具体实现参数，不能只写“已开启”。

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

“正式结论”只能基于 compile 方案 1-step warmup 后的正式请求、有效 runtime 配置记录和质量比较填写。局部 kernel 或 collective 指标改善但端到端耗时没有改善时，应明确写为“局部优化生效，但未形成端到端收益”，不能只报告局部加速数字。