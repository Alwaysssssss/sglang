# VividVR Phase D Modular 改造与公平验收交接

更新时间：`2026-06-05 UTC`

## 1. 当前仓库状态

- 仓库路径：`/home/zhiheng/sglang`
- 当前分支：`sglang_Vivid`
- 当前 HEAD：`c714ebc5627b6926111a68bc0ceca28d60383ecc`
- 本轮工作结论：
  - `Phase C` 单 clip 基线仍然成立，语义和验收结果保持稳定。
  - `VividVR` 已经从原先更接近 hybrid 的结构，改造成更接近 `wan_videoedit` 的 modular 风格。
  - `Phase D` 的长视频路径和公平 benchmark 工具链已经实现并可运行。
  - 但 `Phase D` 在“原版 Vivid-VR 自己生成 caption，再回放给 sglang”的公平对比下，**仍未通过验收**。
- 当前工作区仍有未提交改动，原因是 `Phase D` 尚未通过最终验收，不应提前 `commit` / `push`。

当前工作区状态摘要：

- 已修改文件：
  - `.gitignore`
  - `AGENTS.md`
  - `docs_xzh/run_vivid_benchmark.md`
  - `python/sglang/multimodal_gen/configs/sample/vividvr.py`
  - `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
  - `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
  - `python/sglang/multimodal_gen/runtime/vividvr/__init__.py`
  - `python/sglang/multimodal_gen/runtime/vividvr/postprocess.py`
- 未跟踪文件：
  - `Vivid_Acceptance/indicator/phase_c_metrics_seed42_20260604T105047Z.json`
  - `Vivid_Acceptance/indicator/phase_d_metrics_seed42_20260604T150758Z.json`
  - `Vivid_Acceptance/indicator/phase_d_metrics_seed42_20260604T161753Z.json`
  - `python/sglang/multimodal_gen/runtime/vividvr/captioning.py`
  - `python/sglang/multimodal_gen/runtime/vividvr/windowing.py`
  - `python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py`
  - `python/sglang/multimodal_gen/tools/run_vividvr_phase_d_long_video.py`


## 2. 这次对话完成了什么

本轮对话从 `Phase A/B/C` 已恢复并验收通过的基线继续往前推进，主要做了以下几件事：

1. 阅读并确认了上一份交接文档，明确当前项目背景是“灾后恢复后的继续推进”，不是从零开发。
2. 新增并持续完善仓库根目录 `AGENTS.md`，把环境、tmux、验收产物路径、公平 benchmark 规则、阶段性验收后再 `commit/push` 等约束固定下来。
3. 分析了 `SKILL.md` 里的 modular / hybrid 定义，并确认原始 `vividvr_pipeline.py` 不是期望的 `wan_videoedit` 风格 modular 实现。
4. 先写了实施方案文档：
   - `docs_xzh/modular_style/vividvr_modular_refactor_plan.md`
5. 再按该方案把 `VividVR` 改造成更接近 `wan_videoedit` 的 modular stage 组合，并补上 `Phase D` 长视频时序编排路径。
6. 把仓库标准 Python/uv 环境统一到了：
   - `/home/zhiheng/sglang/.venv/bin/python`
7. 更新了 benchmark 文档，明确：
   - `sglang` 版本默认使用仓库 `.venv`
   - 原版 `Vivid-VR` 做公平对比时必须使用：
     - `/home/zhiheng/Vivid-VR/.venv/bin/python`
   - 公平 benchmark 以 `--upscale=1` 为准
8. 更新了验收脚本记录格式，使 JSON 中记录真实命令，并增加：
   - `total_runtime_seconds`
   - `model_inference_runtime_seconds`
9. 实际跑通了 `Phase D` 的公平 benchmark 流程，包括：
   - 复制原测试视频为 `x3` 长视频，保持 `960x720`
   - 用原版 `Vivid-VR` 自己的环境跑 reference
   - 从原版运行中提取每个 temporal clip 的 raw caption
   - 保存到 `/home/zhiheng/Vivid-VR/input/captions/test_video_960x720_x3.txt`
   - 再让 `sglang` 版本通过 `--caption-file` 回放这些 caption
10. 最终结论是：
    - 公平性问题已经纠正
    - 但 `Phase D` 结果仍未过线，当前主要矛盾已经不是 caption 路径，而是长视频 orchestration 语义本身仍有偏差


## 3. 代码实现现状

### 3.1 架构方向

本轮改造的目标不是“重写模型行为”，而是把 `VividVR` 的表达方式从“大一统 pre-denoising stage + 隐式 batch.extra 状态”改造成更接近 `wan_videoedit` 的：

- 多个小 stage 串联
- 显式 typed runtime contract
- 单 clip 和长视频路径都在原生 `sglang.multimodal_gen` pipeline 内完成

当前 `VividVR` 的 stage 结构已经改成：

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

对应入口在：

- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`


### 3.2 运行时 contract

`python/sglang/multimodal_gen/configs/sample/vividvr.py` 现在不再把主要运行态只塞进 `batch.extra["vividvr_runtime"]`，而是把关键状态显式放到了 `VividVRSamplingParams.runtime_*` 字段中，并提供 `reset_runtime()`。

已经显式化的内容包括：

- prompt / caption 相关：
  - `runtime_prompt_file_path`
  - `runtime_caption_file_path`
  - `runtime_caption_texts`
  - `runtime_raw_prompt_text`
  - `runtime_model_prompt_text`
  - `runtime_negative_prompt_text`
- input / condition 相关：
  - `runtime_control_video`
  - `runtime_reference_video`
  - `runtime_original_height`
  - `runtime_original_width`
  - `runtime_original_num_frames`
  - `runtime_num_padding_frames`
  - `runtime_padded_input_frames`
  - `runtime_fps`
- tensor / 调度相关：
  - `runtime_prompt_embeds`
  - `runtime_negative_prompt_embeds`
  - `runtime_control_latents`
  - `runtime_generator`
  - `runtime_latents`
  - `runtime_tiling_infos`
  - `runtime_timesteps`
  - `runtime_progress`
- 长视频相关：
  - `runtime_execution_mode`
  - `runtime_clip_specs`
  - `runtime_num_temporal_overlapped_frames`
  - `runtime_temporal_frame_stride`
  - `runtime_temporal_merge_plan`

同时，`caption_source` 现在允许两种模式：

- `prompt_file`
- `caption_file`

其中：

- `Phase C` 单 clip 仍以 `prompt_file` 为标准
- `Phase D` 公平 benchmark 使用 `caption_file`


### 3.3 Prompt / caption 处理

新增：

- `python/sglang/multimodal_gen/runtime/vividvr/captioning.py`

它负责：

- `caption_file_path` 解析
- `caption_file` 逐行读取
- 构造 prompt context
- 构造 long-video / tiled prompt list

这里的关键点是：

- 对新视频做公平 benchmark 时，不再直接让 `sglang` 自己猜 caption
- 而是重放原版 `Vivid-VR` 自己跑出来的 raw caption

这一步是为了把 caption backend 差异尽量从 `Phase D` 对比中剥离掉。


### 3.4 长视频 orchestration 实现

新增：

- `python/sglang/multimodal_gen/runtime/vividvr/windowing.py`

这个文件负责 `Phase D` 的核心 helper：

- `build_vividvr_temporal_window_plan(...)`
- `build_vividvr_temporal_latent_merge_plan(...)`
- `merge_vividvr_temporal_latent_states(...)`
- `trim_vividvr_temporal_output_clip(...)`
- `stitch_vividvr_temporal_output_clips(...)`

对应数据结构包括：

- `VividVRTemporalClipSpec`
- `VividVRTemporalWindowPlan`
- `VividVRTemporalLatentMergePlan`

当前 `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py` 已经有：

- `_forward_temporal_windowed(...)`

这条路径在输入帧数超过 `num_temporal_process_frames` 时生效，整体流程是：

1. 建立 temporal window plan
2. 为每个 clip 构建 clip-local video info
3. 每个 clip 单独做 condition encoding / latent preparation / tiling
4. 按 clip 组织 prompt list 与 text encoding
5. 统一生成 timesteps
6. 逐 timestep 对所有 clip 做 denoise
7. 逐 timestep 做跨 clip latent merge
8. 全部完成后 decode
9. 对每个 clip 做 trim
10. stitch 回最终长视频
11. 做 reference color fix / optional postprocess


### 3.5 Postprocess 与可选模块

`python/sglang/multimodal_gen/runtime/vividvr/postprocess.py` 当前已经整理成 helper 形式，包含：

- `adaptive_instance_normalization(...)`
- `decoded_video_to_frame_tensor(...)`
- `apply_reference_color_fix(...)`
- `run_optional_postprocess_modules(...)`

`enable_optional_caption_module`、`enable_optional_postprocess_module`、`allow_optional_module_fallback` 也已经进入采样参数 contract。

当前设计意图是：

- 保留 Phase C/原版要求的核心后处理语义
- 同时允许 optional module 在失败时按规则降级，而不是直接拖垮整个 pipeline


### 3.6 Phase D 工具脚本

新增：

- `python/sglang/multimodal_gen/tools/run_vividvr_phase_d_long_video.py`

这个脚本负责：

- 准备 `x3` 长视频输入，保持原分辨率不变
- 用 `caption_file` 模式发起 `Phase D` 请求
- 记录真实命令
- 记录：
  - `total_runtime_seconds`
  - `model_inference_runtime_seconds`
- 和原版参考视频做逐帧对比
- 将验收 JSON 写入：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator`


### 3.7 其他仓库规则与文档更新

- `AGENTS.md`
  - 已改成中文
  - 明确默认环境为 `/home/zhiheng/sglang/.venv/bin/python`
  - 明确长时间推理必须放进 `tmux`
  - 明确验收指标路径和候选视频路径
  - 明确新视频公平 benchmark 的原版 caption 提取规则
  - 明确原版 benchmark 必须使用 `/home/zhiheng/Vivid-VR/.venv/bin/python`
- `docs_xzh/run_vivid_benchmark.md`
  - 补充了当前 `sglang` 标准验收命令
  - 补充了原版 `Vivid-VR` 公平对比命令
  - 补充了 `Phase D` 长视频公平 benchmark 的完整流程
  - 已明确 benchmark 以 `--upscale=1` 为准
- `.gitignore`
  - 已补充对大视频产物的忽略规则，避免把 benchmark 生成的大文件意外纳入版本控制


## 4. Phase C 当前状态

### 4.1 语义状态

`Phase C` 单 clip 主链仍是当前项目的稳定基线，关键语义保持不变：

- prompt 来自：
  - `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`
- 不走 live `CogVLM2`
- `prompt_embed_shape` 仍为：
  - `[1, 226, 4096]`
- VAE tiling 默认值仍为：
  - `240 / 360`
- preprocess 仍保留未 padding 的 `reference_video`
- decode/postprocess 仍保留：
  - `drop first 3 frames`
  - 裁掉 padding
  - `AdaIN` / reference color fix


### 4.2 Phase C 验收基线

当前正式 gold baseline 仍然是：

- 指标：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_c_metrics_seed42_20260604T090642Z.json`
- 候选视频：
  - `/home/zhiheng/sglang/Vivid_Acceptance/result_videos/phase_c_candidate_seed42_20260604T090642Z.mp4`

关键指标：

- `pass_compare = true`
- `ssim_mean = 0.967716215299506`
- `ssim_min = 0.9473462237832677`
- `mse_mean = 39.878108160836355`
- `mse_max = 81.55096435546875`
- `mae_mean = 3.3365604979651313`
- `mae_max = 3.9912755489349365`
- `prompt_embed_shape = [1, 226, 4096]`


### 4.3 Phase C 新格式补充说明

后续更新了验收脚本，新增了：

- 真实 `command`
- `failed_frame_ratio`
- `total_runtime_seconds`
- `model_inference_runtime_seconds`

需要注意：

- `090642` 这份历史 baseline 本身**不包含**新加的时间字段
- 当前的格式要求是：
  - “以 `090642` 的字段结构和指标含义为基准”
  - “后续新产物额外包含两个时间字段”

一个已经符合新格式、且数值与基线一致的产物是：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_c_metrics_seed42_20260604T105047Z.json`

它的结果仍然是：

- `pass_compare = true`
- `failed_frame_ratio = 0.0`
- `ssim_mean = 0.967716215299506`
- `mse_mean = 39.878108160836355`


### 4.4 本轮重新执行的单测

本轮重新确认通过的相关测试：

- `PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/test_stage_c_vividvr_single_clip.py -q`
  - 结果：`3 passed, 1 skipped`

更早在 modular 改造阶段还跑通过：

- `python/sglang/multimodal_gen/test/unit/test_stage_b_vividvr_components.py`
- `python/sglang/multimodal_gen/test/unit/test_stage_c_vividvr_single_clip.py`

当时汇总结果为：

- `14 passed, 1 skipped, 1143 subtests passed`


## 5. Phase D 当前状态

### 5.1 已完成的部分

`Phase D` 当前不是“只有设计没有实现”，而是已经具备：

- temporal window planning
- clip split
- per-clip condition / latent preparation
- long-video denoise orchestration
- cross-clip latent merge
- clip trim
- clip stitch
- caption-file replay
- 对原版长视频 reference 的逐帧对比

配套单测文件：

- `python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py`

本轮重新确认通过：

- `PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py -q`
  - 结果：`15 passed`


### 5.2 公平 benchmark 的正确做法

这部分是本轮最重要的流程修正，必须传给下一位继续执行的人。

错误做法：

- 用 `/home/zhiheng/sglang/.venv/bin/python` 代跑原版 `Vivid-VR`
- 让原版在 `sglang` 的高版本 `transformers` 环境里跑 `CogVLM2`

这个做法会导致原版 caption 异常，之前已经出现过错误的中文 caption，因此不再有效。

正确做法：

1. 原版 `Vivid-VR` 必须使用：
   - `/home/zhiheng/Vivid-VR/.venv/bin/python`
2. benchmark 参数以：
   - `--upscale=1`
   为准
3. 如果是新视频，先在原版环境里跑 reference
4. 从原版日志里提取每个 temporal clip 的 raw caption
5. 保存到：
   - `/home/zhiheng/Vivid-VR/input/captions/<video_stem>.txt`
6. 再让 `sglang` 版本通过：
   - `--caption-file /home/zhiheng/Vivid-VR/input/captions/<video_stem>.txt`
   重跑
7. 最后用这两个结果做逐帧比较

当前 `Phase D` 已验证的公平基准使用：

- 输入视频：原测试视频按时间方向复制 `3x`
- 分辨率：保持 `960x720` 不变
- `seed = 42`
- `num_temporal_process_frames = 121`
- `num_inference_steps = 6`
- `guidance_scale = 6`
- `restoration_guidance_scale = -1.0`
- `upscale = 1`


### 5.3 Phase D 第一次长视频结果

第一次 long-video 对比使用了不正确的 caption 流程，结果很差：

- 指标：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_d_metrics_seed42_20260604T150758Z.json`

关键指标：

- `pass_compare = false`
- `ssim_mean = 0.5534662869523691`
- `ssim_min = 0.48017741297409916`
- `mse_mean = 589.2152419317337`
- `mse_max = 960.6524047851562`
- `mae_mean = 15.564659227643695`
- `mae_max = 19.79085350036621`
- `reference_frame_count = 210`
- `candidate_frame_count = 210`
- `total_runtime_seconds = 1026.443822`
- `model_inference_runtime_seconds = 800.860757`

这一次的主要问题后来确认不是单纯模型错误，而是公平性流程不正确，尤其是原版 caption 不是在原版自己的环境中生成。


### 5.4 Phase D 纠正公平性后的结果

随后重新按正确流程做了 fair rerun：

- 原版运行日志：
  - `/home/zhiheng/sglang/Vivid_Acceptance/logs/vividvr_ori_phase_d_vividenv_fixhdr3_6step_20260604T160137Z.log`
- 原版 reference 视频：
  - `/home/zhiheng/sglang/Vivid_Acceptance/result_videos/phase_d_original_up1_x3_vividenv_6step/videos/test_video_960x720_x3.mp4`
- 原版提取出的 caption file：
  - `/home/zhiheng/Vivid-VR/input/captions/test_video_960x720_x3.txt`
- `sglang` 重跑日志：
  - `/home/zhiheng/sglang/Vivid_Acceptance/logs/phase_d_long_fair_gpu1_6step_20260604T161742Z.log`
- `sglang` 候选视频：
  - `/home/zhiheng/sglang/Vivid_Acceptance/result_videos/phase_d_candidate_seed42_20260604T161753Z.mp4`
- 指标：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_d_metrics_seed42_20260604T161753Z.json`

关键结果：

- `pass_compare = false`
- `ssim_mean = 0.8597280224853191`
- `ssim_min = 0.7398718292919957`
- `mse_mean = 171.1507835024879`
- `mse_max = 438.7164001464844`
- `mae_mean = 8.070618184407552`
- `mae_max = 12.635977745056152`
- `failed_frame_ratio = 1.0`
- `reference_frame_count = 210`
- `candidate_frame_count = 210`
- `total_runtime_seconds = 853.88173`
- `model_inference_runtime_seconds = 801.66529`

debug 信息显示 fairness 输入已经正确：

- `caption_backend = "caption_file"`
- `caption_entry_count = 3`
- `caption_file_path = "/home/zhiheng/Vivid-VR/input/captions/test_video_960x720_x3.txt"`
- `num_clips = 3`
- `prompt_embed_shape = [1, 226, 4096]`
- `clip_latent_lengths = [32, 32, 26]`

clip 规划也已经记录在指标文件里：

- clip 0：
  - `start_frame=0`
  - `end_frame=121`
  - `original_num_frames=121`
  - `padded_num_frames=121`
  - `num_padding_frames=0`
  - `trim_front_frames=0`
  - `trim_back_frames=30`
- clip 1：
  - `start_frame=60`
  - `end_frame=181`
  - `original_num_frames=121`
  - `padded_num_frames=121`
  - `num_padding_frames=0`
  - `trim_front_frames=31`
  - `trim_back_frames=30`
- clip 2：
  - `start_frame=120`
  - `end_frame=210`
  - `original_num_frames=90`
  - `padded_num_frames=97`
  - `num_padding_frames=7`
  - `trim_front_frames=31`
  - `trim_back_frames=0`


### 5.5 Phase D 当前失败的准确判断

最重要的判断如下：

1. 这次 `Phase D` 失败，已经**不是**因为 caption 公平性路径错了。
2. 与错误 caption 的上一次结果相比，指标已经明显改善：
   - `ssim_mean` 从 `0.5534` 提升到 `0.8597`
   - `mse_mean` 从 `589.2` 降到 `171.15`
3. 但仍未过线，说明问题更接近：
   - 长视频 split / merge / temporal orchestration 语义仍和原版存在系统性偏差

分段统计也支持这一点。按 `0-59`、`60-119`、`120-209` 分段看：

- `0-59`
  - `ssim_mean = 0.863348380684115`
  - `mse_mean = 181.6495246887207`
  - `mae_mean = 8.296553444862365`
  - `failed = 60`
- `60-119`
  - `ssim_mean = 0.8604358918676648`
  - `mse_mean = 175.85430018107095`
  - `mae_mean = 8.181481734911602`
  - `failed = 60`
- `120-209`
  - `ssim_mean = 0.8568425374312244`
  - `mse_mean = 161.015944925944`
  - `mae_mean = 7.84608564376831`
  - `failed = 90`

这说明失败不是只集中在 clip seam 附近，更像整个长视频时序路径都存在稳定偏差。


## 6. 当前最值得继续追的点

下一轮如果继续修 `Phase D`，优先级建议如下：

1. 对照原版 `Vivid-VR` 的 `infer_split_clips()` 路径，逐步核对：
   - clip 切分
   - 每个 timestep 的多 clip 执行顺序
   - merge 发生时机
   - merge 覆盖策略
2. 重点检查：
   - `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py` 的 `_forward_temporal_windowed(...)`
   - `python/sglang/multimodal_gen/runtime/vividvr/windowing.py` 里的 latent ownership / trim / stitch 逻辑
3. 继续确认 caption replay 是否和原版 clip 消耗顺序完全一致。
4. 不要再把主要精力花在：
   - `prompt.txt` 路径
   - 单 clip `226` embedding 长度
   - 原版 env 选择
   上，因为这些关键公平性问题已经基本澄清。

当前更像需要继续追的是：

- temporal merge 语义
- clip 之间 latent ownership 的边界
- trim 前后时序对齐
- 长视频路径中每个 clip 的 prompt / tiling / denoise 组合方式是否真正和原版一致


## 7. 下一位接手时应该怎么做

推荐顺序：

1. 先读这份文档。
2. 再读：
   - `docs_xzh/hand_over/phase_abc_restore_and_next_stage_handover.md`
   - `docs_xzh/modular_style/vividvr_modular_refactor_plan.md`
   - `docs_xzh/run_vivid_benchmark.md`
3. 先确认当前工作区未提交改动仍在。
4. 如果要继续改 `Phase D`，先以当前 fair benchmark 为唯一有效 benchmark 流程。
5. 所有长时间推理仍放到 `tmux` 中执行。
6. 默认环境用：
   - `/home/zhiheng/sglang/.venv/bin/python`
7. 只有跑原版 `Vivid-VR` 时才切到：
   - `/home/zhiheng/Vivid-VR/.venv/bin/python`
8. 只有在 `Phase D` 公平 benchmark 真正通过后，才进行本阶段的 `commit` 和 `push`。


## 8. 一句话总结

这轮已经把 `VividVR` 从原来的非 modular 结构，推进到了接近 `wan_videoedit` 的 modular 形态，并把 `Phase D` 的长视频实现、公平 caption 回放、benchmark 命令和验收产物规范都搭起来了；当前剩下的核心问题不是环境或 caption，而是 **long-video orchestration 语义仍未和原版完全对齐**。
