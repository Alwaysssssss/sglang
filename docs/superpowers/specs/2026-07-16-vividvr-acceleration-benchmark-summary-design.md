# VividVR 加速测试结果总结文档设计

## 目标

在 `docs_xzh/docs_analysis` 下维护一份正式 Markdown 总结文档，汇总批次
`vividvr_accel_full_warmup1_20260716` 的正式测试结果，使读者无需浏览全部 JSON
即可了解各加速方案的用时、实验环境、模块增量收益，以及单卡、双卡、四卡最快方案。

## 数据来源

- 批次汇总：`Vivid_Acceptance/acceleration_benchmark/vividvr_accel_full_warmup1_20260716/batch_summary.json`
- 正式记录：同批次 `records/*_formal.json`
- Warmup 记录：同批次 `records/*_warmup.json`
- 阶段指标：同批次 `requests/*/perf.json`
- 实验定义：`docs_xzh/docs_analysis/analysis_tables.md`
- 环境信息：正式记录的 `reproducibility`、服务日志、当前测试机的系统与 Python 环境查询结果

所有性能、显存和质量数值直接取自验收 JSON，不手工估算；机器与软件版本来自上述环境证据。Markdown 中耗时保留三位小数，加速比保留两位小数。

## 文档结构

目标文件为：

`docs_xzh/docs_analysis/acceleration_benchmark_results_20260716.md`

文档包含以下六部分：

1. 测试口径：仅列出输入帧数、正式推理步数、warmup 步数和耗时字段定义。
2. 总体耗时：用一个主表展示方案、关键加速条件、总耗时、模型推理耗时、denoise 耗时、平均 step 和相对 R0 模型加速比。
3. 完整 Stage 耗时：使用横向总表，以方案为行、8 个 Stage 为列，展示每个正式方案的完整 Stage 拆分。表前提供简化列名与代码 Stage 类名的对应关系。
4. 实验环境记录：填写机器、GPU、Driver、PyTorch、FlashAttention、Python、checkpoint、计时、显存采样、Stage profiling 和批次产物目录。
5. 模块收益结论：沿用 `analysis.md` 的 Treatment/Control 关系，填写延迟增量加速比、GPU·秒变化、最大单卡峰值显存变化、质量变化和正式结论。
6. 最快方案：汇报单卡、双卡、四卡的最低模型推理耗时。

完整 Stage 耗时固定读取正式记录的 `timings.stage_seconds`，并覆盖以下字段：

1. `VividVRInputValidationStage`
2. `VividVRPromptPreparationStage`
3. `VividVRTemporalWindowPlanningStage`
4. `VividVRLongClipPreparationStage`
5. `VividVRTimestepPreparationStage`
6. `VividVRMultiClipDenoisingStage`
7. `VividVRMultiClipDecodeTrimStage`
8. `VividVRTemporalStitchPostprocessStage`

## 结论表达原则

- R0 作为原始单卡基线。
- 单卡、双卡、四卡分别按模型推理耗时选择最快方案。
- 加速比默认以模型推理耗时计算，并在表头明确，避免与请求总耗时混用。
- R7–R9 没有产生推理耗时，不进入耗时结果表。
- 模块收益表中的延迟增量加速比统一使用 `Control 模型推理耗时 / Treatment 模型推理耗时`。
- GPU·秒变化统一使用 `Treatment GPU 数量 × Treatment 模型推理耗时 - Control GPU 数量 × Control 模型推理耗时`，同时报告百分比。
- 显存变化使用正式记录的 `max_single_gpu_peak_gib` 差值；质量变化使用 `ssim_mean` 差值。
- 本轮所有已执行视频结果按用户人工验收结论记为通过，同时保留 SSIM 差值，并明确它与 JSON 严格 `pass_compare` 门槛不是同一口径。
- R7 Cache-DiT、R8 TeaCache、R9 通用量化没有正式结果，相关收益字段填写 `N/A`，结论写明未实现、无法评价。
- R6 算子融合若增量加速比不超过 `1.00×`，正式结论必须明确为未形成端到端加速。
- 多卡结论同时描述延迟和 GPU·秒，不能只汇报延迟加速。
- 不汇报通信耗时或结果视频路径。
- Stage 横向总表中的耗时统一保留三位小数；不足 `0.0005` 秒的 Stage 显示为 `0.000`，不改变 JSON 原始数据。
- 不修改 `analysis_tables.md`，新文档仅总结已经生成的结果。

## 实验环境记录规则

- 机器型号记录为 `6U GPU Server`。
- GPU 记录为 `8 × NVIDIA A100-SXM4-80GB`，并注明本轮单次方案最多使用 4 张。
- Driver 记录为 `550.90.07`。
- CUDA 记录为 `PyTorch CUDA 12.8`；由于系统没有 `nvcc`，不得把它描述为已确认的 CUDA Toolkit 版本。
- PyTorch 记录为 `2.9.1+cu128`，FlashAttention 4 记录为 `4.0.0b19`。
- 本批次正式 JSON 未保存 Git commit，`sglang commit` 填写 `N/A（本批次记录缺失）`，不得按文件时间戳推测。
- Python、模型和 Vivid-VR checkpoint 路径从正式记录的 `reproducibility` 字段填写。
- Compile mode 按方案区分 eager 与 `torch.compile`，计时口径沿用正式 JSON 字段定义。
- 显存统计方式记录为 NVML 采样；Stage profiling 记录为同步计时。
- Perf/report 目录填写实际批次目录，而不是 `analysis.md` 模板中的通用目录。

## 验收标准

- 主表覆盖有正式推理耗时的 R0–R6、R99、R100。
- 数值与对应正式 JSON 一致。
- 横向总表覆盖九个正式方案和全部 8 个 Stage，且数值与 `timings.stage_seconds` 四舍五入到三位小数后一致。
- 实验环境表的可确认字段均有本地命令、正式 JSON 或服务日志依据，无法确认项明确写为 `N/A`。
- 模块收益表覆盖 `analysis.md` 定义的 13 组 Treatment/Control；其中 10 组有正式数据，R7–R9 三组明确标记为未实现。
- 10 组正式比较的增量加速比、GPU·秒、峰值显存和 SSIM mean 差值均能由正式 JSON 机械复算。
- 明确 warmup 均为 1 step，formal 均为 20 step。
- 明确总耗时、模型推理耗时和 denoise 耗时的口径。
- 正确列出单卡、双卡、四卡最快方案。
- 文档中不存在待定项、占位符或空表格。
