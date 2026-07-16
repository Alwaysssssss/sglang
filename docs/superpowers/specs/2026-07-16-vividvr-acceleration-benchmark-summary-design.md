# VividVR 加速测试结果总结文档设计

## 目标

在 `docs_xzh/docs_analysis` 下新增一份简洁的 Markdown 耗时总结文档，汇总批次
`vividvr_accel_full_warmup1_20260716` 的正式测试结果，使读者无需浏览全部 JSON
即可了解各加速方案的用时，以及单卡、双卡、四卡最快方案。

## 数据来源

- 批次汇总：`Vivid_Acceptance/acceleration_benchmark/vividvr_accel_full_warmup1_20260716/batch_summary.json`
- 正式记录：同批次 `records/*_formal.json`
- Warmup 记录：同批次 `records/*_warmup.json`
- 阶段指标：同批次 `requests/*/perf.json`
- 实验定义：`docs_xzh/docs_analysis/analysis_tables.md`

所有数值直接取自验收 JSON，不手工估算。Markdown 中耗时保留三位小数，加速比保留两位小数。

## 文档结构

目标文件为：

`docs_xzh/docs_analysis/acceleration_benchmark_results_20260716.md`

文档包含以下四部分：

1. 测试口径：仅列出输入帧数、正式推理步数、warmup 步数和耗时字段定义。
2. 总体耗时：用一个主表展示方案、关键加速条件、总耗时、模型推理耗时、denoise 耗时、平均 step 和相对 R0 模型加速比。
3. 完整 Stage 耗时：使用横向总表，以方案为行、8 个 Stage 为列，展示每个正式方案的完整 Stage 拆分。表前提供简化列名与代码 Stage 类名的对应关系。
4. 最快方案：汇报单卡、双卡、四卡的最低模型推理耗时。

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
- 不汇报质量、SSIM、显存、GPU·秒、通信耗时或产物路径。
- Stage 横向总表中的耗时统一保留三位小数；不足 `0.0005` 秒的 Stage 显示为 `0.000`，不改变 JSON 原始数据。
- 不修改 `analysis_tables.md`，新文档仅总结已经生成的结果。

## 验收标准

- 主表覆盖有正式推理耗时的 R0–R6、R99、R100。
- 数值与对应正式 JSON 一致。
- 横向总表覆盖九个正式方案和全部 8 个 Stage，且数值与 `timings.stage_seconds` 四舍五入到三位小数后一致。
- 明确 warmup 均为 1 step，formal 均为 20 step。
- 明确总耗时、模型推理耗时和 denoise 耗时的口径。
- 正确列出单卡、双卡、四卡最快方案。
- 文档中不存在待定项、占位符或空表格。
