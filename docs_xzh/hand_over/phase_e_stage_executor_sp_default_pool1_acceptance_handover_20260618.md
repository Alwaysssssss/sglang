# VividVR Phase E: 长视频 Stage Executor + SP 默认禁用 Pooling 交接

更新时间：`2026-06-18 UTC`

## 1. 文档目的

本文档面向下一位接手 `Phase E` 的 Codex，记录当前这几件已经落地并完成验证的事实：

1. 长视频 `130f / 20 step` 推理路径已经整理为更标准的 stage 风格组织。
2. 双卡 `SP` 默认不再启用 control pooling，默认值已经从 `pool_size=2` 收口为 `pool_size=1`。
3. 默认 `pool=1` 的双卡正式长视频验收已经真实跑通并通过。
4. `pool=2` 仍然保留为显式性能实验开关，但不再作为默认质量口径。

---

## 2. 代码与仓库锚点

| 项 | 值 |
|----|-----|
| 分支 | `sglang_Vivid` |
| 当前 HEAD | `337a31932` |
| Python 环境 | `/home/zhiheng/sglang/.venv/bin/python` |
| GPU | `2 x NVIDIA A100-SXM4-80GB` |
| 当前长视频默认 benchmark | `130f / 20 step / seed=42` |
| 当前长视频默认 reference | `/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4` |

---

## 3. 本轮完成的默认值收口

### 3.1 目标

此前 `SP` Path B 的 control pooling 在 `130f` 长视频上可以通过门限，但在 `720p / 70f` 路径上已经确认会带来明显质量回归。  
因此当前正式默认不再追求 `pool=2` 的更快速度，而是优先保证双卡质量口径稳定。

### 3.2 代码改动

1. `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py`
   - `_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE_DEFAULT` 从 `2` 改为 `1`
   - 语义变成：默认不压缩 control state，只有显式设置环境变量时才启用 pooling

2. `python/sglang/multimodal_gen/tools/test_connector_remote_compress.py`
   - 工具脚本默认读取的 pool size 从 `"2"` 改为 `"1"`
   - 这样工具默认行为与运行时默认一致

3. `docs_xzh/run_vivid_benchmark.md`
   - 明确记录 `SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE` 默认值为 `1`
   - 明确说明 `=2` 等值只用于实验，不属于正式默认口径

4. `AGENTS.md`
   - 补充当前 `Phase E` 双卡 `SP` 默认质量口径要求：默认 `pool=1`
   - 同时补充 `tmux attach -r` 的只读查看建议，减少终端误发 `Ctrl-C` 中断推理

---

## 4. 当前长视频推理结构状态

当前长视频 `130f` 路径已经不再是 pipeline 内部的大段手工控制流，而是按 stage executor 风格组织为：

1. `VividVRInputValidationStage`
2. `VividVRPromptPreparationStage`
3. `VividVRTemporalWindowPlanningStage`
4. `VividVRLongClipPreparationStage`
5. `VividVRTimestepPreparationStage`
6. `VividVRMultiClipDenoisingStage`
7. `VividVRMultiClipDecodeTrimStage`
8. `VividVRTemporalStitchPostprocessStage`

也就是说，长视频路径现在已经和短视频一样，进入了明确的 stage 化组织；区别只在于长视频使用的是自己的 multi-clip orchestration stages，而不是直接复用短视频单 clip `forward()`。

---

## 5. 本轮正式验收结果

### 5.1 单卡基线

指标文件：

`/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e41_stage_executor_single_gpu_v2_130f_20step_compile_metrics_seed42_20260618T031644Z.json`

结果视频：

`/home/zhiheng/sglang/Vivid_Acceptance/result_videos/phase_e41_stage_executor_single_gpu_v2_130f_20step_compile_seed42_20260618T031644Z.mp4`

关键结果：

- `pass_compare=True`
- `ssim_mean=0.9847691341142859`
- `ssim_min=0.9800605706877014`
- `model_inference_runtime_seconds=936.494121`
- `total_runtime_seconds=1241.537855`

### 5.2 双卡旧默认 `pool=2` 对照

指标文件：

`/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e41_stage_executor_sp_v2_130f_20step_compile_metrics_seed42_20260618T034827Z.json`

结果视频：

`/home/zhiheng/sglang/Vivid_Acceptance/result_videos/phase_e41_stage_executor_sp_v2_130f_20step_compile_seed42_20260618T034827Z.mp4`

关键结果：

- `pass_compare=True`
- `ssim_mean=0.9813522104004836`
- `ssim_min=0.9696232467931043`
- `model_inference_runtime_seconds=441.984999`
- `total_runtime_seconds=718.795232`
- `control_context_shape_global=[2, 6600, 3072]`

说明：

- 这个结果证明 `pool=2` 在 `130f` 长视频上可以跑通并通过当前门限
- 但它不再代表正式默认质量口径

### 5.3 双卡新默认 `pool=1` 正式验收

tmux session：

- `vividvr_stage_sp_v2_default_pool1`

日志：

- `/home/zhiheng/sglang/Vivid_Acceptance/logs/phase_e41_stage_executor_sp_v2_default_pool1_130f_20step_compile_20260618T074048Z.log`

指标文件：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e41_stage_executor_sp_v2_default_pool1_130f_20step_compile_metrics_seed42_20260618T074059Z.json`

结果视频：

- `/home/zhiheng/sglang/Vivid_Acceptance/result_videos/phase_e41_stage_executor_sp_v2_default_pool1_130f_20step_compile_seed42_20260618T074059Z.mp4`

关键结果：

- `pass_compare=True`
- `ssim_mean=0.9845430592776135`
- `ssim_min=0.9783342201035932`
- `model_inference_runtime_seconds=537.016378`
- `total_runtime_seconds=967.607365`
- `control_context_shape_global=[2, 27000, 3072]`

日志确认点：

- 正式 stage 链完整执行到 `VividVRTemporalStitchPostprocessStage`
- 输出视频和指标 JSON 已正常保存
- 日志中没有再出现 `pool_size=2` 的 control pooling 压缩痕迹

### 5.4 三组结果对照

| 配置 | pass | SSIM mean | SSIM min | inference time |
|------|------|-----------|----------|----------------|
| 单卡 stage executor | ✅ | 0.984769 | 0.980061 | 936.49s |
| 双卡 SP `pool=2` | ✅ | 0.981352 | 0.969623 | 441.98s |
| 双卡 SP 默认 `pool=1` | ✅ | 0.984543 | 0.978334 | 537.02s |

当前结论：

- `pool=1` 的双卡结果在质量上基本贴近单卡基线
- `pool=2` 更快，但质量更低，且此前已经在 `720p` 路径暴露回归风险
- 因此当前正式默认应保持 `pool=1`

---

## 6. 为什么默认改成 `pool=1`

当前项目状态下，这个决策是为了把“正式默认口径”和“性能探索口径”拆开：

- 正式默认口径：
  - 追求通过长视频与短视频质量门禁
  - 优先保守、稳定、可复现
  - 当前默认使用 `pool=1`

- 性能探索口径：
  - 允许显式设置 `SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=2`
  - 只在明确做性能实验时启用
  - 结果不能自动外推为正式默认

这和前一份 handover 的结论一致：`pool=2` 不是“彻底不能用”，而是“不适合作为当前 release gate 的默认值”。

---

## 7. 当前推荐的正式双卡命令

```bash
tmux new-session -d -s vividvr_stage_sp_default_pool1 \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && export PYTHONUNBUFFERED=1 && export PYTHONPATH=python && export PYTORCH_ALLOC_CONF=expandable_segments:True && export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global && /home/zhiheng/sglang/.venv/bin/torchrun --nproc_per_node=2 --master_port=30084 python/sglang/multimodal_gen/tools/run_vividvr_inference.py --input-video /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4 --caption-file /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt --prompt-file /home/zhiheng/Vivid-VR/input/720p/prompt.txt --reference-video /home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4 --output-dir /home/zhiheng/sglang/Vivid_Acceptance/result_videos --report-dir /home/zhiheng/sglang/Vivid_Acceptance/indicator --artifact-prefix phase_e41_stage_executor_sp_v2_default_pool1_130f_20step_compile --phase-label E --mode-label temporal_windowed_stage_executor_sp_v2_default_pool1_compile --num-temporal-process-frames 121 --num-inference-steps 20 --guidance-scale 6 --restoration-guidance-scale -1.0 --seed 42 --num-gpus 2 --tp-size 1 --sp-degree 2 --ulysses-degree 2 --ring-degree 1 --dist-timeout 3600 --master-port 30084 --attention-backend fa --enable-torch-compile --warmup --warmup-steps 1 2>&1 | tee Vivid_Acceptance/logs/phase_e41_stage_executor_sp_v2_default_pool1_130f_20step_compile_$(date -u +%Y%m%dT%H%M%SZ).log'
```

只读查看：

```bash
tmux attach -r -t vividvr_stage_sp_default_pool1
```

注意：

- 不要显式设置 `SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE`
- 当前默认值已经是 `1`
- 如果终端环境容易误发 `Ctrl-C`，优先使用 `attach -r`

---

## 8. 下一步建议

1. 如果下一轮继续做 release gate 收口，默认沿用 `pool=1` 路径。
2. 如果下一轮专门做性能实验，可以单独重开 `pool=2` 或其他 pooling 策略测试，但不要覆盖默认口径。
3. 如果要进一步提高默认双卡速度，优先排查：
   - compile mode 是否需要改成更确定性的模式
   - decode 阶段是否还有可收口空间
   - 除 pooling 外的 SP 数据传输路径是否还能继续减轻开销
4. 如果要扩展质量门禁，下一优先项应是：
   - 用当前默认 `pool=1` 再补一轮 `720p` 双卡 formal 结果
   - 明确把短视频也纳入新的默认双卡回归套件

---

## 9. 本轮结论

当前可以认为：

1. 长视频 stage executor 风格改造已进入可验收状态。
2. 双卡 `SP` 默认 `pool=1` 已经真实验收通过。
3. `pool=2` 被降级为显式性能实验选项，不再作为默认质量配置。

