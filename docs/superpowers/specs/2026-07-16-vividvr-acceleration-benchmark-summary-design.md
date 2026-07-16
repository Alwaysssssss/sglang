# VividVR 加速测试结果总结文档设计

## 目标

在 `docs_xzh/docs_analysis` 下新增一份简洁的 Markdown 总结文档，汇总批次
`vividvr_accel_full_warmup1_20260716` 的正式测试结果，使读者无需浏览全部 JSON
即可了解各加速模块的收益、单卡/双卡/四卡推荐方案、质量结论和验收产物位置。

## 数据来源

- 批次汇总：`Vivid_Acceptance/acceleration_benchmark/vividvr_accel_full_warmup1_20260716/batch_summary.json`
- 正式记录：同批次 `records/*_formal.json`
- Warmup 记录：同批次 `records/*_warmup.json`
- 阶段指标和质量指标：同批次 `requests/*/{perf.json,compare.json}`
- 实验定义：`docs_xzh/docs_analysis/analysis_tables.md`

所有数值直接取自验收 JSON，不手工估算。Markdown 中耗时保留三位小数，加速比保留两位小数，SSIM 保留六位小数。

## 质量口径

本总结按用户确认的正式口径，将所有已经完成推理并成功生成完整视频的 R0–R6、R99、R100 统一视为质量通过。

原始逐帧比较结果继续如实记录，包括 SSIM mean、SSIM min 和低于严格逐帧阈值的帧数，但不再用 `quality_failed` 描述这些实验。R7–R9 没有执行推理，保持 `unsupported`，不能记为质量通过。

## 文档结构

目标文件为：

`docs_xzh/docs_analysis/acceleration_benchmark_results_20260716.md`

文档包含以下五部分：

1. 测试结论：说明实验已完整结束，以及单卡、双卡、四卡推荐配置。
2. 测试口径：列出输入、帧数、推理步数、warmup、精度和质量解释。
3. 总体结果：用一个主表展示方案、关键加速条件、总耗时、模型耗时、denoise、step、显存、SSIM 和质量结论。
4. 加速模块收益：围绕 attention backend、torch.compile、SP、CFG×SP 和 modulation fusion 给出对照结果与简短判断。
5. 产物与验收：给出批次 summary、records、requests、日志目录以及验证结果。

## 结论表达原则

- R0 作为原始单卡基线。
- 单卡推荐以 R2 为主；R6 用于说明 modulation fusion 在本轮没有稳定收益。
- 双卡推荐 R99，同时说明它与 R3 的性能几乎相同，优势主要是覆盖全部已实现加速条件。
- 四卡最快方案为 R100；同时保留 R4 和 R5 的拓扑对照。
- 加速比默认以模型推理耗时计算，并在表头明确，避免与请求总耗时混用。
- 不把 Cache-DiT、TeaCache、量化写成已测试方案，只说明当前尚无可执行的 VividVR 路径。
- 不修改 `analysis_tables.md`，新文档仅总结已经生成的结果。

## 验收标准

- 主表覆盖 R0–R6、R99、R100，并单列 R7–R9 的 unsupported 状态。
- 所有已执行正式实验的质量结论均为“通过”。
- 数值与对应正式 JSON 一致。
- 明确 warmup 均为 1 step，formal 均为 20 step。
- 文档中的本地路径存在，且批次 summary、正式视频和指标 JSON 可定位。
- 文档中不存在待定项、占位符或空表格。
