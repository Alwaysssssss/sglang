# Phase E E3.2 Control 对齐 E2 的当前实现背景与问题交接

## 1. 文档目的

本交接文档用于给后续 Codex / 代码代理提供当前 `Vivid-VR -> sglang.multimodal_gen` 集成线的最新实现背景、已完成工作、正式验收结果、性能问题边界，以及下一步最合理的排查方向。

当前最重要的结论不是“E3.2 已经完全收口”，而是：

- `E3.2` 的 runtime 接线和观测字段已经接通
- 长视频画质 compare 已继续通过
- `denoise loop` 已经基本与 `E2` 对齐
- 但 `clip preparation` 和 `decode/postprocess` 仍然显著慢于 `E2`
- 所以当前还不能把这版 `E3.2 control` 当成已经完全对齐 `E2` 的 release-gate baseline

这意味着：下一轮如果继续做性能收口，不应该再优先怀疑 `denoise` 主循环，而应该集中看长视频入口准备和 decode 后处理路径。

## 2. 阶段背景

### 2.1 Phase C

`Phase C` 是当前单 clip 稳定基线，必须继续保护。

必须继续守住的关键语义包括：

- prompt 默认来自 `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`
- 不走 live `CogVLM2`
- `prompt_embed_shape` 保持 `226`
- VAE tiling 默认值保持 `240 / 360`
- preprocess 保留未 padding 的 `reference_video`
- decode / postprocess 保留 `drop first 3 frames + crop padding + AdaIN/reference color fix`

### 2.2 Phase D

`Phase D` 的重点是长视频语义对齐，已经完成正式验收，后续 `Phase E` 默认都建立在这个长视频语义基线之上。

必须继续保护的 `Phase D` 语义：

- clip split
- 多 clip 的 timestep 级时序编排
- 跨 clip latent merge
- clip trim / stitch
- 公平 benchmark 使用原版 caption sidecar

后续任何性能收口都不能把长视频路径改成“能跑但不等价”的近似实现。

### 2.3 Phase E

`Phase E` 的目标不是重新发明语义，而是在 `Phase D` 基线之上做：

- 默认配置收口
- runtime / backend 原生化
- profile 和性能结论沉淀
- regression 套件建设
- 逐步进入 release gate

当前默认 benchmark 口径固定为 `130f / 20 step` 长视频验收。

## 3. 当前正式基线

### 3.1 E2 当前正式最佳

当前最重要的单卡正式基线仍是 `E2 = FA + torch.compile`。

- 报告：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e2_130f_20step_compile_metrics_seed42_20260606T084506Z.json`
- 关键指标：
  - `model_inference_runtime_seconds = 923.9699`
  - `total_runtime_seconds = 1190.988609`

对应阶段时长：

- `vividvr_long_video_clip_preparation = 52027.46208384633 ms`
- `vividvr_long_video_denoising_state_preparation = 54.78476360440254 ms`
- `vividvr_long_video_denoising_loop = 773121.1420372128 ms`
- `vividvr_long_video_decode_postprocess = 97656.862270087 ms`

### 3.2 E3.2 当前最新正式验收件

本轮最新正式验收件：

- 报告：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e32_align_e2_final_130f_20step_compile_warmup_metrics_seed42_20260609T011232Z.json`
- 日志：
  - `/home/zhiheng/sglang/Vivid_Acceptance/logs/phase_e32_align_e2_final_130f_20step_compile_warmup_20260609T011222Z.log`
- 视频：
  - `/home/zhiheng/sglang/Vivid_Acceptance/result_videos/phase_e32_align_e2_final_130f_20step_compile_warmup_seed42_20260609T011232Z.mp4`

关键指标：

- `pass_compare = true`
- `model_inference_runtime_seconds = 943.064834`
- `total_runtime_seconds = 1246.5563`

对应阶段时长：

- `vividvr_long_video_clip_preparation = 62646.29367366433 ms`
- `vividvr_long_video_denoising_state_preparation = 158.4138683974743 ms`
- `vividvr_long_video_denoising_loop = 772137.6644484699 ms`
- `vividvr_long_video_decode_postprocess = 108062.53584474325 ms`

运行时配置确认：

- `attention_backend_effective = fa`
- `torch_compile_transformer = true`
- `torch_compile_controlnet = true`
- `attn_metadata_enabled = true`
- `attn_metadata_builder = FlashAttentionMetadataBuilder`
- `runai_model_streamer_enabled = true`
- `vividvr_vae_decode_tiling_config = true`
- `vae_tiling_enabled = true`
- `denoising_autocast_enabled = false`

## 4. E3.2 之前主要做了什么

`E3.2` 这条线的目标不是直接做大幅提速，而是把 `VividVR` 更原生地接到 `sglang` runtime 上，并且让这些状态可观测、可验收、可回归。

主要完成的内容：

- denoise 内接入 `attn_metadata builder`
- 长视频主链不再长期 `attn_metadata=None`
- runtime snapshot / 验收 JSON 增加 attention backend 和 metadata 观测
- decode 侧接入 `vae.enable_tiling()` 的配置与落盘
- 长视频主路径补齐 `step_profile` / `timestep_index` / runtime debug 等调用链
- 修复验收过程中暴露出的真实长视频主路径 bug

这条线的价值是“把 runtime/native path 收口”，不是“直接命中 DiT 主热点”。

## 5. 本轮已实现但尚未完全解决的问题

### 5.1 已做的实现调整

本轮实际落地了三类改动，目标都是在不破坏 `Phase D` 长视频语义的前提下，把 `E3.2 control` 尽量往 `E2` 靠。

#### A. pipeline 级控制视频缓存

文件：

- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`

已增加：

- `_build_control_video_cache_key(...)`
- `_resolve_input_video_info(...)`

策略：

- 以 `abspath(path) + st_mtime_ns + st_size` 构造 cache key
- 将 `load_control_video(...)` 的结果缓存在 pipeline 实例上
- `forward()` 改为先走 `_resolve_input_video_info(...)`

目的：

- 避免 warmup 和 formal 两次 `forward()` 重复解码同一份 control video

注意：

- warmup 请求来自 `Req.copy_as_warmup()`，内部会 `deepcopy(req)`
- 所以大体积视频缓存不能放在请求对象上，只能放在 pipeline 实例上

#### B. compare summary 直接携带 frame count

文件：

- `python/sglang/multimodal_gen/runtime/videoedit/compare.py`
- `python/sglang/multimodal_gen/tools/run_vividvr_inference.py`

已增加：

- `reference_frame_count`
- `candidate_frame_count`
- `frame_count_delta`

目的：

- 避免 compare 完成后，为了拿 frame count 再次单独读 reference / candidate 视频

#### C. video frame cache 路径已接通

文件：

- `python/sglang/multimodal_gen/runtime/videoedit/frame_cache.py`
- `python/sglang/multimodal_gen/runtime/videoedit/preprocess.py`

目的：

- 让 compare 首次读取的视频帧可以被后续 `load_video_frames(...)` 复用

### 5.2 这三类改动当前的真实效果

必须明确一个关键事实：

- `run_vividvr_inference.py` 记录 `total_runtime_seconds` 的位置在 `compare_videos(...)` 之前

因此：

- compare summary / frame cache 这些优化，并不会改变写进 JSON 的 `total_runtime_seconds`
- 它们只会改善 report 写盘之后的额外 wall clock，不会改善当前验收口径中的记录值

这点是本轮最重要的修正结论之一。之前如果把 compare 优化理解成可以直接压低 JSON 里的 `total_runtime_seconds`，那是错误的。

同样要明确：

- pipeline 级 `load_control_video(...)` 缓存也不会直接降低 `vividvr_long_video_clip_preparation`
- 因为 `clip_preparation` 这个 StageProfiler 是在 `_forward_temporal_windowed(...)` 内部才开始计时的
- 顶层 `forward()` 里的 `load_control_video(...)` 不在这个 stage metric 覆盖范围内

因此它更可能影响的是：

- `model_inference_runtime_seconds` 总值
- `total_runtime_seconds`

而不一定直接反映在 `clip_preparation` 分段指标上。

## 6. 当前最新结果的真实诊断

### 6.1 与 E2 的差距

和 `E2` 相比，当前最新 `E3.2` 验收件仍然有明显差距：

- `model_inference_runtime_seconds` 慢 `19.094934s`
- `total_runtime_seconds` 慢 `55.567691s`

分段差异：

- `clip_preparation` 慢 `10618.831590 ms`
- `denoising_state_preparation` 慢 `103.629105 ms`
- `denoising_loop` 快 `983.477589 ms`
- `decode_postprocess` 慢 `10405.673575 ms`

### 6.2 结论一：denoise loop 已经对齐

这一轮最明确的正面结论是：

- `denoising_loop` 已经与 `E2` 基本打平，甚至略快

正式日志里，20 step 的稳态进度最终是：

- `VividVR denoising: 100%|...| 20/20 [12:52<00:00, 38.61s/it]`

这与 `E2` 的 `step1-19` 均值 `38.71s` 已经在同一档位。

因此，后续不应该再把主要精力投到 denoise 主循环本身。

### 6.3 结论二：当前慢的核心在 pre / post

当前差距几乎全部集中在两段：

- `clip_preparation`
- `decode_postprocess`

并且两者量级非常接近：

- 都比 `E2` 大约慢 10 秒左右

这说明问题不是单一奇点，而更像是：

- 长视频入口准备路径仍有额外工作
- decode / resize / color-fix / postprocess 路径仍与 `E2` 不完全同构

### 6.4 结论三：decode 额外开销是真实发生在 VAE decode 内

本轮对运行中进程做了 `py-spy dump`，在 denoise 结束后主线程明确停在：

- `diffusers/models/autoencoders/autoencoder_kl_cogvideox.py:tiled_decode`
- 上层调用来自：
  - `VividVRDecodingStage.decode_latents(...)`
  - `VividVRPipeline._forward_temporal_windowed(...)`

这说明：

- denoise 结束后的长耗时不是 compare 或日志写盘卡住
- 当前确实在 `VAE decode / tiled_decode` 这条路径里消耗了较多时间

因此，`decode_postprocess` 的慢不是假象，而是真实热点。

## 7. 当前最合理的下一步

如果下一轮继续做性能收口，最合理的策略是：

- 保留 `E3.2` 的 runtime 接线和观测字段
- 不再优先动 denoise loop
- 重点检查 `clip preparation` 与 `decode/postprocess` 是否存在与 `E2` 不一致的执行方式

建议排查顺序：

1. 先核对长视频 `clip preparation` 路径

- 重点看 `condition_encoding_stage.prepare_condition_inputs(...)`
- 看长视频多 clip 循环里是否引入了 `E2` 没有的重复工作
- 看 caption / prompt 编码 / latent preparation / tiling preparation 是否存在额外重复

2. 再核对 `decode/postprocess` 路径

- 重点看 `VividVRDecodingStage.decode_latents(...)`
- 看 `vae.enable_tiling()` 这条路径是否与 `E2` 的实际 decode 行为一致
- 看 `decoded_video_to_frame_tensor(...)`
- 看 resize / `VideoProcessor.postprocess_video(...)` / `apply_reference_color_fix(...)`
- 明确当前长视频 decode 路径与 `E2` 是否完全同构

3. 继续守住 `Phase D` 长视频语义

- 不要为了压时间破坏 clip trim / stitch
- 不要把 reference color fix、caption sidecar、公平 compare 等语义绕掉

## 8. 本轮涉及文件

当前这条线涉及的关键文件包括：

- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
- `python/sglang/multimodal_gen/runtime/videoedit/compare.py`
- `python/sglang/multimodal_gen/runtime/videoedit/preprocess.py`
- `python/sglang/multimodal_gen/runtime/videoedit/frame_cache.py`
- `python/sglang/multimodal_gen/tools/run_vividvr_inference.py`

对应测试文件：

- `python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py`
- `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py`
- `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_runtime_snapshot.py`
- `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_offload.py`

## 9. 当前最终结论

截至本轮结束，可以确认：

- `E3.2` 的 runtime 接线和观测字段已经接通
- 最新 `E3.2` 正式件继续 `pass_compare = true`
- `denoise loop` 已经与 `E2` 对齐
- 但整体用时仍未与 `E2` 对齐
- 当前主要问题不在 denoise，而在 `clip preparation` 与 `decode/postprocess`

所以，后续正确口径应该是：

- 不要再把“E3.2 未对齐”理解成整个 runtime/native 化方向失败
- 也不要盲目进入 `E3.3`
- 应先把 `E3.2 control` 的 pre/post 两段继续收干净，直到它真正成为接近 `E2` 的干净 control

## 10. 本轮说明

本轮仅整理背景和问题，形成交接文档。

- 没有新增运行时实现修改
- 没有新增测试执行
- 没有新增验收运行

本轮文档使用的最新正式验收结论，来自已经落盘的：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e32_align_e2_final_130f_20step_compile_warmup_metrics_seed42_20260609T011232Z.json`
