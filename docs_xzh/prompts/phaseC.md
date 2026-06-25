# Phase C Recovery Prompt

你当前的任务是在 `Phase A` 和 `Phase B` 已恢复并通过验收的基础上，继续恢复 `Vivid-VR` 在 `sglang.multimodal_gen` 中的 `Phase C`，目标是完成单 clip MVP、严格对齐原版 `Vivid-VR` 的关键语义，并通过 `Phase C` 的 reference 验收。

## 1. 工作边界

- 代码仓库路径固定为：`/home/zhiheng/sglang`
- 只允许在分支：`sglang_Vivid` 上工作
- 必须以 `sglang` 内部原生实现为主，不允许把 `/home/zhiheng/Vivid-VR` 当成运行时 Python 依赖
- 允许把 `/home/zhiheng/Vivid-VR` 中的以下内容当成静态资源使用：
  - checkpoints
  - `input/720p/prompt.txt`
  - reference 视频 `result/720p_up1_result_vivid_ori/videos/test_video_960x720.mp4`
- 不允许：
  - import `/home/zhiheng/Vivid-VR` 里的 Python 代码
  - 复制 `Vivid-VR/src/diffusers` 整棵实现进 `sglang`
  - 通过 wrapper 方式把原仓库直接“包”进来

## 2. 先读这些文档，再动代码

- `/home/zhiheng/sglang/docs_xzh/add_strategy/README.md`
- `/home/zhiheng/sglang/docs_xzh/add_strategy/03_stage2_mvp_scope.md`
- `/home/zhiheng/sglang/docs_xzh/add_strategy/04_stage3_pipeline_mod_plan.md`
- `/home/zhiheng/sglang/docs_xzh/add_strategy/05_stage4_component_migration.md`
- `/home/zhiheng/sglang/docs_xzh/add_strategy/09_code_mod_order.md`
- `/home/zhiheng/sglang/docs_xzh/add_strategy/10_grouped_stage_acceptance.md`
- `/home/zhiheng/sglang/docs_xzh/hand_over/phase_bc_recovery_handover.md`

如需提取之前成功实现过的 Phase C patch，以这条 session 为主：

- `/home/zhiheng/.codex/sessions/2026/06/03/rollout-2026-06-03T08-44-22-019e8ca7-b803-7b41-a68b-a133925ddd80.jsonl`

## 3. 本阶段真正目标

不要把目标理解成“代码能跑通”。`Phase C` 的真实目标是：

- 恢复 `VividVR` 单 clip 端到端 pipeline
- 在固定 seed 下输出稳定、可重复的视频
- 在 `sglang` 内独立完成推理，不依赖原仓库运行时代码
- 与 reference 视频完成逐帧比对并过线
- 保存真实验收指标文件和真实输出视频，不能只在终端里口头说明“通过了”

## 4. 必须恢复的关键语义

这些点是之前 `Phase C` 成功通过验收的核心。不要在这些地方重踩坑：

### 4.1 prompt 来源固定

- caption 固定读取：`/home/zhiheng/Vivid-VR/input/720p/prompt.txt`
- 不允许在 `sglang` 环境里实时调用 `CogVLM2`
- 不允许把 auto caption 链路混入 `Phase C`

### 4.2 text encoder wrapper 语义

- 对齐原版 `Vivid-VR` 的调用方式
- 关键点是：忽略 `attention_mask`，只用 `input_ids` 调 T5
- 保持 rotary 兼容路径，不要退回到和原版不一致的 learned positional 语义

### 4.3 VAE tiling 默认值

这是之前最关键的坑之一，必须优先确认：

- `tile_sample_min_height = 240`
- `tile_sample_min_width = 360`

不要沿用 generic 的 `256 x 256` 默认值。之前正是这个差异导致 `960x720` 输入的 latent 从正确的 `90x120` 漂到错误的 `91x122`，后续再怎么调都很难对齐 reference。

### 4.4 preprocess 语义

- padding 前必须保留未 padding 的 `reference_video`
- control video 的 padding 规则要和之前成功版本一致
- prompt suffix 拼接和输入合同要与原版语义一致

### 4.5 decode 后的 wrapper 语义

decode 后不能直接把结果当最终输出，必须补回原版 `Vivid-VR` wrapper 的外层语义：

- 必要时丢掉前 `3` 帧
- 裁掉 padding 帧
- 再基于 `reference_video` 对输出执行 `AdaIN`

其中 `AdaIN` 是 `Phase C` 成功过线的关键收口项之一，不能漏。

## 5. 推荐优先关注的文件

优先恢复这些文件，不要一开始就在无关模块上浪费时间：

- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
- `python/sglang/multimodal_gen/runtime/vividvr/preprocess.py`
- `python/sglang/multimodal_gen/runtime/vividvr/postprocess.py`
- `python/sglang/multimodal_gen/runtime/vividvr/tiling.py`
- `python/sglang/multimodal_gen/configs/models/vaes/cogvideox.py`
- `python/sglang/multimodal_gen/test/unit/test_stage_c_vividvr_single_clip.py`

如果发现 `Phase A/B` 基线和这些文件的接口仍有轻微不匹配，可以做最小必要修正，但不要回头重做大范围重构。

## 6. 推荐恢复顺序

按语义顺序恢复，不要机械按 session 时间顺序回放 patch：

1. `vividvr_pipeline.py`
2. `runtime/vividvr/preprocess.py`
3. `runtime/vividvr/tiling.py`
4. `model_specific_stages/vividvr.py`
5. `runtime/vividvr/postprocess.py`
6. `configs/models/vaes/cogvideox.py`
7. `test_stage_c_vividvr_single_clip.py`
8. 最后补 acceptance 产物持久化逻辑

## 7. 必须主动规避的时间黑洞

下面这些方向在 `Phase C` 恢复中优先级很低，除非它们明确阻断 reference 对齐，否则不要过早投入：

- compile 优化
- offload 策略调优
- backend 性能微调
- 长视频 clip split / merge
- 可选 caption / OCR / ESRGAN / TextFixer
- 多卡 TP/SP

`Phase C` 的主问题是 correctness，不是性能。

## 8. 验收前必须人工核对的检查清单

在跑重型验收前，逐项人工确认：

- prompt 源确实是 `prompt.txt`
- 没有 live `CogVLM2`
- VAE tiling 默认值确实是 `240 / 360`
- `reference_video` 在 padding 前就被保留
- 丢前 `3` 帧逻辑仍在
- padding crop 逻辑仍在
- `AdaIN(reference_video)` 已在 decode 后应用
- 同 seed determinism 有测试覆盖
- 输出视频路径不是临时目录自动删除产物

最后一条尤其重要。之前真实验收虽然通过，但测试里的 `candidate.mp4` 曾保存在 `TemporaryDirectory()` 下。恢复时必须把正式验收产物持久化保存到固定目录，不能只留临时文件。

## 9. 验收产物保存要求

`Phase C` 恢复完成后，必须把真实验收结果持久化到以下固定目录：

- 指标文件目录：`/home/zhiheng/sglang/Vivid_Acceptance/indicator`
- 视频结果目录：`/home/zhiheng/sglang/Vivid_Acceptance/result_videos`

要求：

- 如果目录不存在，就创建
- 至少保存一份 candidate 视频到 `result_videos`
- 至少保存一份结构化指标文件到 `indicator`
- 指标文件里必须包含：
  - `ssim_mean`
  - `ssim_min`
  - `mse_mean`
  - `mse_max`
  - `mae_mean`
  - `mae_max`
  - `failed_frames`
  - `failed_frame_ratio`
  - `frame_count_delta`
  - `seed`
  - `prompt_path`
  - `reference_video_path`
  - `candidate_video_path`
  - `pass_compare`
  - 运行命令
  - 运行日期时间

建议：

- 指标保存为 `json`
- 文件名带时间戳和 `phase_c`
- 视频文件名带 `phase_c`、seed、分辨率

## 10. 验收标准

必须达到 `Phase C` 的默认宽松基线：

- `ssim_min >= 0.90`
- `mse_max <= 150.0`
- `mae_max <= 8.0`
- `failed_frame_ratio <= 0.05`
- `frame_count_delta <= 1`

并且要验证：

- 同 seed 两次推理结果一致
- 单 clip 端到端输出稳定
- 无空输出、错帧、明显崩溃

## 11. 建议执行顺序

建议严格按下面顺序推进：

1. 先确认 `Phase A/B` 当前基线测试仍通过
2. 从 session 中提取 `Phase C` 相关 patch
3. 先恢复语义关键点，不要急着跑重型验收
4. 先跑轻量 unit / contract test
5. 再跑单 clip reference 验收
6. 验收通过后，把真实视频和真实指标复制到固定目录
7. 最后再做 `git commit`
8. 然后 `git push origin sglang_Vivid`

## 12. 提交要求

在 `Phase C` 通过验收后：

- `git status` 必须清晰
- 以独立 commit 提交 `Phase C`
- push 到远程 `origin/sglang_Vivid`

最终汇报时必须同时给出：

- 修改了哪些关键文件
- 真实跑过哪些命令
- 真实指标结果
- 指标文件绝对路径
- 输出视频绝对路径
- commit hash
- push 是否成功

## 13. 最终执行要求

请直接动手恢复 `Phase C`，不要只停留在分析。过程中重点盯住语义对齐，而不是做泛化式重构。若遇到问题，优先怀疑以下几类语义漂移：

- VAE tiling
- text encoder wrapper 调用方式
- `reference_video` 的保留时机
- 前 `3` 帧裁剪
- padding crop
- `AdaIN` 是否漏掉

只要这些语义点重新对齐，`Phase C` 通过 reference 验收的概率最高。
