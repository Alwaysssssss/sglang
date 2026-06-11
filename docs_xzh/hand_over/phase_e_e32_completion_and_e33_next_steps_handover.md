# VividVR Phase E3.2 完成态与 Phase E3.3 起点交接

更新时间：`2026-06-09 UTC`

## 1. 这份文档覆盖什么

这份交接文档用于给下一个 Codex 提供一份“当前最新、可直接开工”的项目状态摘要，重点说明：

- 当前哪些阶段已经完成，哪些必须继续保护
- `Phase E3.2` 到底完成到了什么程度
- 当前应该把哪一版当成正式 control / release-gate baseline
- `sglang` 接线中哪些能力被保留，哪些 hot path 包裹层被收回到 `E2` 形态
- 下一位 Codex 如果继续做 `Phase E3.3`，应该从哪里开始、先做什么、不该再踩什么坑

这份文档的状态结论，已经晚于并部分覆盖下面这些旧文档中的阶段性判断：

- `docs_xzh/hand_over/phase_e_e32_alignment_control_and_prepost_strategy_handover.md`
- `docs_xzh/hand_over/phase_e_e0_e3_acceptance_and_single_gpu_combo_handover.md`
- `docs_xzh/hand_over/phase_d_acceptance_completion_and_phase_e_benchmark_handover.md`


## 2. 当前总状态

### 2.1 阶段完成情况

- `Phase C`：
  - 单 clip 语义基线，已完成，必须继续保护
- `Phase D`：
  - 长视频 `clip split / timestep orchestration / merge / trim / stitch` 语义基线，已完成正式验收，必须继续保护
- `Phase E0 / E1 / E2 / E3`：
  - benchmark 口径、attention backend、`torch.compile`、fusion 类能力与验收链路已接通
- `Phase E3.2`：
  - 当前已经完成 runtime alignment 收口，可以作为后续 `Phase E3.3` 的直接起点

### 2.2 当前代码基线

- 当前推荐起点提交：
  - `a9058b25d`
  - `Complete VividVR Phase E3.2 runtime alignment`
- 当前分支：
  - `sglang_Vivid`

### 2.3 当前默认 benchmark / acceptance 口径

当前 `Phase E` 日常 benchmark 固定为：

- 输入视频：
  - `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4`
- caption sidecar：
  - `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt`
- reference 视频：
  - `/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4`
- step：
  - `20`
- seed：
  - `42`

`50 step` 只保留给阶段性最终回归，不作为日常性能迭代口径。


## 3. E3.2 的最终结论

### 3.1 不需要回退到 E2

当前结论不是“`E3.2` 失败，需要整条线回退到 `E2`”，而是：

- `E3.2` 的 `sglang` runtime/native 接线、观测字段、CLI 控制项和相关测试已经具备真实价值
- 真正需要收回的，只是 hot path 里偏离 `E2` 的运行时包裹层
- 这些包裹层收回后，当前分支已经与同机 fresh `E2` 基本打平

因此，当前正确策略是：

- 保留 `E2` 作为历史正式性能基线
- 保留当前 `E3.2` 作为后续 `E3.3` 的开发起点
- 不要再把整条 `E3.2` 路线视为失败分支

### 3.2 当前“完成”的准确口径

`E3.2` 当前可以视为完成，依据是：

- 长视频语义 compare 通过
- 模型推理时间与同机 fresh `E2` 基本打平
- 单测通过
- 当前 `HEAD` 已整理成可复用 checkpoint

但要注意，“完成”不等于“超越历史最优秒数”。更准确地说：

- 当前已经完成 `E3.2 runtime alignment`
- 当前还没有证明“历史 `2026-06-06` 那份 `923.97s` 一定还能作为今天唯一绝对 gate”


## 4. 关键验收结果

### 4.1 当前分支正式验收件

正式验收报告：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e32_runtime_e2_align_130f_20step_compile_metrics_seed42_20260609T025514Z.json`

对应结果：

- `pass_compare = true`
- `model_inference_runtime_seconds = 935.243947`
- `total_runtime_seconds = 1235.495857`

### 4.2 同机 fresh E2 复核件

为了避免被历史秒数误导，本轮重新用同机、同命令、fresh worktree 重跑了 `E2`。

复核报告：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e2_recheck_130f_20step_compile_metrics_seed42_20260609T031841Z.json`

对应结果：

- `pass_compare = true`
- `model_inference_runtime_seconds = 934.690479`
- `total_runtime_seconds = 1233.8452`

### 4.3 当前最重要的对比结论

当前分支相对 fresh `E2` 的差值只有：

- `model_inference_runtime_seconds = +0.553468s`
- `total_runtime_seconds = +1.650657s`

因此在当前机器、当前环境、当前测量条件下，可以认为：

- `E3.2` 已经完成与 `E2` 的 runtime 对齐

### 4.4 历史 E2 指标如何看待

历史正式 `E2` 报告：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e2_130f_20step_compile_metrics_seed42_20260606T084506Z.json`

其中：

- `model_inference_runtime_seconds = 923.9699`

这份历史结果仍然有参考价值，但当前不适合再把它当成唯一绝对 release gate。后续如果要做性能判断，默认应优先采用：

- same-machine fresh control
- 同一天 control / treatment A/B


## 5. E3.2 实际做了什么

### 5.1 保留下来的 sglang 接线

这轮没有把 `sglang` 侧能力撤掉。当前保留的接线包括：

- attention backend 接线
- `torch.compile` 接线
- `qkv fusion` 接线
- `qk_norm_rope fusion` 接线
- modulation fusion 接线
- runtime snapshot / 验收 JSON 观测字段
- inference tool 的 CLI 开关
- `runai model streamer` 观测开关
- VAE decode tiling 配置与观测
- control-video cache
- compare/frame cache

也就是说：

- `E3.2` 不是“把 `sglang` 改动全部删了”
- `E3.2` 是“保留接线和可观测性，同时把 hot path 包裹层收回到 `E2` 形态”

### 5.2 收回到 E2 形态的关键点

当前已对齐回 `E2` 的核心执行形态包括：

- denoise hot loop 不再在运行时强插 `attn_metadata`
- denoise hot loop 不再额外走 `DenoisingStage.step_profile(...)`
- decode 路径回到 `E2` 形态
- `vae_tiling_enabled` debug 字段改为反映真实 `vae.use_tiling`

### 5.3 与 E3.2 直接相关的主要文件

关键实现文件：

- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
- `python/sglang/multimodal_gen/tools/run_vividvr_inference.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py`
- `python/sglang/multimodal_gen/runtime/layers/attention/backends/flash_attn.py`
- `python/sglang/multimodal_gen/runtime/server_args.py`
- `python/sglang/multimodal_gen/runtime/videoedit/compare.py`
- `python/sglang/multimodal_gen/runtime/videoedit/preprocess.py`
- `python/sglang/multimodal_gen/runtime/videoedit/frame_cache.py`

相关测试文件：

- `python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py`
- `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py`
- `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_offload.py`
- `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_runtime_snapshot.py`
- `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py`


## 6. 本轮验证时发现的关键认识

### 6.1 之前的 pre / post 慢结论，不能直接照抄

旧的 `E3.2` 诊断文档里曾经得出：

- `clip_preparation` 比 `E2` 慢约 `10s`
- `decode_postprocess` 比 `E2` 慢约 `10s`

这个结论在当时是基于 formal stage metrics 做出的，但本轮进一步验证后，需要修正为：

- 这些 stage metrics 里有明显的异步 GPU 计时归因偏差
- 不能直接把 unsync stage metrics 当成真实热点定位依据

### 6.2 当前最重要的 profiling 注意事项

`StageProfiler` 默认并不会同步所有非-step 阶段，因此：

- `clip_preparation`
- `decode_postprocess`

这类指标可能会被上游异步 GPU 工作“串账”。

后续如果下一位 Codex 还要继续做热点诊断，默认做法应该是：

- 在微基准里显式 `torch.cuda.synchronize()`
- 或者只把 formal stage metrics 当成线索，而不是最终证据

### 6.3 compare / frame-cache 优化不会降低 JSON 里的 total_runtime

本轮还澄清了一个容易误判的点：

- `run_vividvr_inference.py` 记录 `total_runtime_seconds` 的位置在 `compare_videos(...)` 之前

因此：

- compare summary
- frame cache

这些优化不会直接降低 JSON 里的 `total_runtime_seconds`。

### 6.4 control-video cache 也未必反映在 clip_preparation 指标里

pipeline 顶层 `load_control_video(...)` 缓存发生在：

- `forward()` 顶层路径

而 `clip_preparation` stage metric 的计时范围在更后面，因此：

- 它可能改善总推理耗时
- 但不一定直接体现在 `clip_preparation` 分段指标里


## 7. 当前测试与可复现命令

### 7.1 单测

本轮最终通过的测试集合：

```bash
/home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_offload.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_runtime_snapshot.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py
```

结果：

- `47 passed`

### 7.2 正式验收命令口径

长视频正式件仍建议走 `tmux`，核心口径为：

- `130f / 20 step / seed=42`
- `--attention-backend fa`
- `--enable-torch-compile`
- `--warmup --warmup-steps 1`

后续如果下一位 Codex 继续做 `E3.3`，不要修改 benchmark 口径，除非任务明确要求。


## 8. 对下一个 Codex 的明确建议

### 8.1 开工前先做什么

第一步先确认：

- 当前 `HEAD` 是否仍是 `a9058b25d`
- benchmark 口径是否仍是 `130f / 20 step`
- `Phase C` / `Phase D` 基线没有被破坏

然后优先阅读：

- `docs_xzh/hand_over/phase_e_e32_completion_and_e33_next_steps_handover.md`
- `docs_xzh/add_strategy/11_phase_e_acceleration_implementation.md`
- `docs_xzh/hand_over/phase_e_e0_e3_acceptance_and_single_gpu_combo_handover.md`

### 8.2 E3.3 的正确起点

后续 `E3.3` 应默认从当前 `E3.2` 对齐态出发，而不是：

- 回退到旧 `E2`
- 也不是直接拿历史 `923.97s` 作为唯一 gate

正确做法是：

- 把当前 `a9058b25d` 当成 control
- 在这个 control 之上逐项尝试新的 runtime/native 优化
- 每次只引入一个主要变量

### 8.3 推荐推进顺序

建议下一位 Codex 按下面顺序推进：

1. 固化当前 `E3.2` control。
2. 选择一个单一优化变量做 `smoke -> formal` A/B。
3. 只有在 `pass_compare = true` 且正式件优于当前 control 时，才保留该变量。
4. 若收益只出现在 unsync stage metrics，而不出现在 `model_inference_runtime_seconds`，不要误判为真实收益。

### 8.4 E3.3 更值得优先尝试的方向

如果继续做 `Phase E3.3`，更合理的方向是：

- 在当前对齐基线上，逐步重新引入 `sglang` 原生优化能力
- 优先尝试已经接线、已经有测试、但还没有在当前 control 上正式收口的单项

比较适合做下一轮单变量 A/B 的候选包括：

- `qk_norm_rope fusion`
- 更谨慎地重新评估 runtime attn-metadata 方案是否有真实净收益
- 在不破坏 current control 的前提下，评估 VAE decode 相关配置是否能产生可重复收益

### 8.5 不要再重复的坑

下一位 Codex 不要再做下面这些事：

- 不要默认相信 unsync `clip_preparation` / `decode_postprocess` 指标就是最终热点
- 不要直接因为旧 `923.97s` 结果更快，就断言当前代码必须回退
- 不要同时引入多个主要加速变量再试图解释单项收益
- 不要破坏 `Phase C` 和 `Phase D` 已验收语义


## 9. 当前交接结论

一句话总结当前状态：

- `Phase C` 已完成并需保护
- `Phase D` 已完成并需保护
- `Phase E0-E3` 接线、测试、验收链路已具备
- `Phase E3.2` 已完成 runtime alignment，并与 same-machine fresh `E2` 基本打平
- 下一个 Codex 可以直接从 `a9058b25d` 进入 `Phase E3.3`

如果下一位 Codex 只记住一条规则，那就是：

- 把当前 `a9058b25d` 当成新的 control，用单变量 A/B 的方式继续推进，而不是再回到“怀疑整条 E3.2 路线是否成立”的阶段
