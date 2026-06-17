# VividVR 长视频对齐 `WanVideoEditPipeline` Stage 风格改造计划

更新时间：2026-06-18

## 1. 目标

本文档用于规划把当前 VividVR 长视频推理路径，从
`VividVRPipeline._forward_temporal_windowed()` 中的“pipeline 内手工编排”
整理成更接近 `python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py`
的 stage 风格组织方式。

本次改造目标不是引入新语义，而是：

1. 保持当前 Phase D 已验收的长视频语义不变。
2. 让长视频路径具备更清晰的 stage 边界与 runtime contract。
3. 最大化复用现有短视频 10-stage 实现中的 helper 和局部能力。
4. 为后续 Phase E 的 profile、回归门禁和默认配置收口提供更稳定的结构基础。

## 2. 当前实现的准确定位

### 2.1 短视频路径

当 `original_num_frames <= num_temporal_process_frames` 时，VividVR 直接走
`ComposedPipelineBase.forward()`，是标准的线性 stage executor 风格。

当前短视频 stage 顺序为：

1. `VividVRInputValidationStage`
2. `VividVRPromptPreparationStage`
3. `VividVRTextEncodingStage`
4. `VividVRConditionEncodingStage`
5. `VividVRLatentPreparationStage`
6. `VividVRTilingPreparationStage`
7. `VividVRTimestepPreparationStage`
8. `VividVRDenoisingStage`
9. `VividVRDecodingStage`
10. `VividVROutputPostprocessStage`

### 2.2 长视频路径

当 `original_num_frames > num_temporal_process_frames` 时，VividVR 进入
`_forward_temporal_windowed()`。

这条路径的特点是：

- 开头仍直接调用 `input_validation_stage` 和 `prompt_preparation_stage`
- 之后不再走 executor 的线性 stage 链
- pipeline 自己维护 `window_plan`、`clip_states[]`、`denoising_states[]`
- 逐 clip 准备输入
- 在每个 timestep 上对所有 clip 同步推进
- 每一步 denoise 后做跨 clip latent merge
- 最后逐 clip decode、trim，再全局 stitch

因此，当前长视频实现不是“短视频 stage executor 的自然延伸”，而是
“stage helper 复用 + pipeline 手工 orchestration”的混合形态。

## 3. 为什么不能直接复用现有短视频 `forward()`

本次改造不能简单理解为“让长视频也走 `super().forward()`”。

原因有四个：

1. Phase D 已验收语义要求的是多 clip 的 timestep 级同步编排，而不是单 clip 串行跑完整条 10-stage。
2. 当前 `VividVRDenoisingStage.forward()` 默认是单 clip 完整 denoise loop，长视频需要的是“准备 state 后按 timestep 跨 clip 推进”。
3. 当前长视频 decode 不是单 clip 直接结束，而是 `decode -> trim -> stitch -> reference color fix -> optional postprocess`。
4. 当前 `VividVRSamplingParams.runtime_*` 大多仍以单 clip singleton 状态为核心，不能直接表达多 clip 全局执行态。

所以正确方向不是“强行让长视频复用现有短视频 10-stage 的 `forward()`”，
而是“把长视频 orchestration 自身拆成一组新的 stage，同时复用短视频 stage 已有 helper”。

## 4. 可复用能力盘点

### 4.1 可以直接复用为长视频 stage 的

- `VividVRInputValidationStage`
- `VividVRPromptPreparationStage`
- `VividVRTimestepPreparationStage`

这些 stage 的职责天然是全局性的，不依赖单 clip 完整闭环。

### 4.2 更适合复用 helper，而不是直接复用 `forward()` 的

- `VividVRTextEncodingStage`
- `VividVRConditionEncodingStage`
- `VividVRLatentPreparationStage`
- `VividVRTilingPreparationStage`
- `VividVRDenoisingStage`
- `VividVRDecodingStage`

建议复用的 helper / 子能力包括：

- `encode_prompt_pair(...)`
- `prepare_condition_inputs(...)`
- `prepare_latents(...)`
- `prepare_tiling_state(...)`
- `prepare_timesteps(...)`
- `prepare_denoising_state(...)`
- `run_denoising_step(...)`
- `decode_latents(...)`

### 4.3 不建议直接复用当前短视频末端 `forward()` 的

- `VividVROutputPostprocessStage.forward()`

原因是长视频需要先逐 clip 输出，再 trim/stitch，后处理边界与单 clip 不同。

## 5. 建议的长视频目标 stage 图

建议把长视频路径改造成下面这条专用 stage 链：

1. `VividVRInputValidationStage`
2. `VividVRPromptPreparationStage`
3. `VividVRTemporalWindowPlanningStage`
4. `VividVRLongClipPreparationStage`
5. `VividVRTimestepPreparationStage`
6. `VividVRMultiClipDenoisingStage`
7. `VividVRMultiClipDecodeTrimStage`
8. `VividVRTemporalStitchPostprocessStage`

这里的关键点是：

- 前两段直接复用短视频 stage
- timestep 仍保持全局一份
- clip preparation、multi-clip denoising、decode/stitch 改为长视频专用 stage
- 长视频路径不再把大量控制流留在 `VividVRPipeline._forward_temporal_windowed()` 里

## 6. 每个长视频 stage 的职责

### 6.1 `VividVRTemporalWindowPlanningStage`

职责：

- 基于 `original_num_frames` 和 `num_temporal_process_frames` 生成 `window_plan`
- 写入 clip specs、overlap、frame stride、全局 fps、高宽、原始帧数等 runtime 信息
- 明确标记 `runtime_execution_mode = "temporal_windowed"`

输出建议：

- `params.runtime_window_plan`
- `params.runtime_clip_specs`
- `params.runtime_num_temporal_overlapped_frames`
- `params.runtime_temporal_frame_stride`

### 6.2 `VividVRLongClipPreparationStage`

职责：

- 逐 clip 构造 `clip_video_info`
- 逐 clip 调 condition / latent / tiling / text helper
- 完成 caption cursor 消费和 tile 对齐
- 产出统一的 `clip_states[]`

这一 stage 内部应复用：

- `prepare_condition_inputs(...)`
- `prepare_latents(...)`
- `build_tiling_infos(...)`
- `encode_prompt_pair(...)`
- `prepare_tiling_state(...)`

输出建议：

- `params.runtime_clip_states`
- `params.runtime_clip_caption_records`

### 6.3 `VividVRMultiClipDenoisingStage`

职责：

- 基于全局 `timesteps` 为每个 clip 初始化 `denoising_state`
- 构造并保存 `temporal_merge_plan`
- 在每个 timestep 上遍历所有 clip 执行 `run_denoising_step(...)`
- 在每一步后执行跨 clip latent merge
- 更新全局进度与必要 debug 指标

这一 stage 是长视频语义核心，必须继续保持 Phase D 的：

- timestep 同步推进
- overlap latent merge
- CFG/restoration guidance/scheduler 参数路径
- tiling 遍历顺序和数值融合逻辑

输出建议：

- `params.runtime_denoising_states`
- `params.runtime_temporal_merge_plan`
- `params.runtime_progress`

### 6.4 `VividVRMultiClipDecodeTrimStage`

职责：

- 逐 clip 调用 `decode_latents(...)`
- resize 回原始分辨率
- 按 clip spec 执行 trim
- 产出 `trimmed_clips[]`

输出建议：

- `params.runtime_trimmed_output_clips`

### 6.5 `VividVRTemporalStitchPostprocessStage`

职责：

- stitch 所有 trimmed clips
- 应用 reference color fix
- 运行 optional postprocess modules
- 设置最终 `runtime_output_video`、`batch.output`、`batch.fps`

这一步同时负责补齐最终 debug/runtime 字段。

## 7. 运行态 contract 调整建议

本轮长视频 stage 化若要稳定落地，需要把当前散落在 pipeline 局部变量中的状态，
适度提升为显式 `runtime_*` contract。

建议新增或强化的字段包括：

- `runtime_window_plan`
- `runtime_clip_states`
- `runtime_clip_caption_records`
- `runtime_temporal_merge_plan`
- `runtime_denoising_states`
- `runtime_trimmed_output_clips`

约束：

- 不要破坏短视频当前已经稳定使用的 `runtime_*` 字段
- 长视频新增字段优先追加，不要随意重命名已被验收工具消费的字段
- 如需兼容现有 metrics/debug 输出，优先在 stage 内补充映射，不要直接移除历史字段

## 8. `VividVRPipeline` 应该收敛成什么样

改造完成后，`VividVRPipeline.forward()` 的职责应尽量简单：

1. 解析输入视频信息
2. 判断单 clip 还是 temporal windowed
3. 单 clip 继续走现有 `super().forward()`
4. 长视频改走一条“长视频 stage 列表”的 executor 路径

对应地，`_forward_temporal_windowed()` 最终应收敛为：

- 被删除
- 或缩成仅负责选择 long-video stage graph 的薄包装

无论采用哪种形式，都不应继续保留目前这种大段手工 orchestration 主逻辑。

## 9. 推荐实施顺序

### 阶段 1：只做结构外提，不改语义

- 提取 `TemporalWindowPlanningStage`
- 提取 `LongClipPreparationStage`
- 提取 `MultiClipDecodeTrimStage`
- 提取 `TemporalStitchPostprocessStage`
- 保持 `MultiClipDenoisingStage` 仍大量复用现有 helper

目标：

- 先把大函数拆掉
- 不改 Phase D 已验收数值路径

### 阶段 2：固化长视频 runtime contract

- 把 `clip_states[]` / `denoising_states[]` 明确写入 `runtime_*`
- 清理临时变量和重复 debug 拼接
- 让 perf/profile/diagnostic 工具更容易观测长视频状态

### 阶段 3：再考虑更深层复用

- 评估是否能让长视频 stage 内进一步调用通用 executor
- 评估部分 helper 是否还能继续抽象
- 这一步必须在 Phase D / Phase E 回归稳定之后再做

## 10. 不应在本次改造里顺手做的事

- 不要改长视频 `clip split / merge / trim / stitch` 语义
- 不要把长视频强行改成“单 clip 顺序跑完再下一个 clip”
- 不要顺手修改 prompt/caption 来源策略
- 不要引入新的 compile/backend 默认值变化
- 不要把 720p pool 回归、compile nondeterminism 与本次结构改造混在一起处理
- 不要为追求抽象纯度而重写短视频稳定路径

## 11. 验收建议

本次属于结构重组，验收重点不是新能力，而是“语义不回归”。

最低建议验收集：

1. Phase C 单 clip 基线不回归
2. Phase D 130f / 20 step 长视频公平 benchmark 不回归
3. 长视频 caption sidecar consumption 顺序不回归
4. 长视频 trim/stitch 后输出帧数与参考口径一致
5. 关键 metrics 字段和 debug 字段仍可被现有工具消费

如果本次 stage 化碰到 SP / compile 路径，还需要额外检查：

1. `pool=1` 长视频结果保持通过
2. 单卡与双卡长视频路径都能跑通
3. `model_inference_runtime_seconds` 的统计边界没有被结构改造破坏

## 12. 总结

长视频改造成更接近 `WanVideoEditPipeline` 的 stage 风格是合理方向，但正确落点不是
“让长视频直接复用短视频 10-stage 的整条 `forward()`”，而是：

- 前置全局 stage 直接复用
- 中间 clip orchestration 拆成长视频专用 stage
- 核心数值 helper 继续复用短视频阶段里已经沉淀下来的实现

这样可以在不破坏 Phase D 已验收语义的前提下，把当前的长视频手工编排逐步收敛成
更清晰、更可维护、也更便于 Phase E 回归治理的结构。
