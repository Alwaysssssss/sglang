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

## 最快方案

| GPU 数量 | 最快方案 | 总耗时（s） | 模型推理耗时（s） | Denoise（s） | 相对 R0 模型加速比 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | R2：FA + `torch.compile` | 941.516 | 936.101 | 772.493 | 1.18× |
| 2 | R99：SP=2 + FA-SP + `torch.compile` + modulation fusion | 551.119 | 544.321 | 380.176 | 2.03× |
| 4 | R100：CFG=2 × SP=2 + FA-SP + `torch.compile` + modulation fusion | 370.881 | 365.067 | 195.652 | 3.02× |
