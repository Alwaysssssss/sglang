# VividVR Phase E E0-E3 验收与单卡组合测试交接

更新时间：`2026-06-06 UTC`

## 1. 这份交接文档覆盖什么

本文档总结本次对话中完成的全部关键工作，供下一个 Codex 快速接手：

- 阅读并确认最新 `Phase D` / `Phase E` 交接文档，收口当前项目背景。
- 讨论 `Phase E` 的加速方向，并明确“尽可能复用 `sglang` 底层已有加速工程实现”的原则。
- 新增并细化 `Phase E` 实施文档：
  - `docs_xzh/add_strategy/11_phase_e_acceleration_implementation.md`
- 在文档中补充单项 A/B 验收方法学，以及 `E0 -> E5` 的阶段推进顺序。
- 完成 `Phase E0 / E1 / E2 / E3` 的实现说明、正式验收和结果整理。
- 解释当前 `E3` 到底做了哪些高频算子融合。
- 实际跑通单卡 `E1 + E2 + E3` 全开组合测试，确认“极致加速”相对原版的收益和当前最优配置的真实关系。


## 2. 当前总状态

- 当前分支：
  - `sglang_Vivid`
- 当前代码 HEAD：
  - `3187e66f8`
- 已确认的重要阶段提交：
  - `4cd992dc2`
    - `Enable VividVR phase E0/E1 attention backend acceptance`
  - `3187e66f8`
    - `Enable VividVR phase E2/E3 acceleration acceptance`
- `Phase D` 仍是必须保护的长视频语义基线，`Phase E` 当前所有加速工作都建立在该基线上。
- `Phase E` 的日常 benchmark 口径已经固定为与 `Phase D` 相同 reference 对象的：
  - `130f / 20 step / seed=42`
- `50 step` 仍只保留给阶段性最终回归，不作为当前日常性能迭代口径。


## 3. 本次对话先完成了哪些文档工作

### 3.1 建立并细化了 Phase E 实施文档

本次对话中已经新写并更新了：

- `docs_xzh/add_strategy/11_phase_e_acceleration_implementation.md`

这份文档已经把 `Phase E` 收口为 4 条主线：

- `attention backend`
- 热点算子融合
- `torch.compile`
- 多卡并行推理

并明确强调：

- 优先把当前 `VividVR` 代码接入 `sglang` 现有底层加速体系。
- 不要优先写一套 `VividVR` 私有的平行加速框架。

### 3.2 文档中已加入的关键规则

当前 `Phase E` 文档已经明确了：

- 日常验收固定使用：
  - `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4`
  - `20 step`
  - `seed=42`
- 对比 reference 与 `Phase D` 相同，质量口径参考：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_d_130f_20step_metrics_seed42_20260605T083022Z.json`
- 每次加入一个新的主要加速项后，必须做单项 `control/treatment` A/B。
- 不允许同时引入多个新的主要加速变量，然后把组合结果当成某一个单项的独立收益结论。
- 文档里的实施顺序已经细化到：
  - `E0`：基线冻结与可观测性
  - `E1`：attention backend 单项接入
  - `E2`：`torch.compile` 单项接入
  - `E3`：按 fusion 类别逐项推进
  - `E4`：多卡分阶段推进
  - `E5`：所有单项通过后，再做组合回归


## 4. 固定 benchmark / acceptance 口径

当前 `Phase E` 默认沿用以下 reference 口径：

- 输入视频：
  - `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4`
- caption sidecar：
  - `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt`
- reference 视频：
  - `/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4`
- `Phase D` 对比基准指标：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_d_130f_20step_metrics_seed42_20260605T083022Z.json`
- 原版 `20 step` 模型推理时间基线：
  - `model_inference_runtime_seconds = 1047.001905`

验收时默认要求：

- 公平 benchmark 使用原版 caption sidecar 回放。
- `pass_compare` 必须为 `true`。
- 质量结果应与 `Phase D` 的 `20 step` 指标保持同量级。
- 主加速倍数口径优先看：
  - `original_model_inference_runtime_seconds / current_model_inference_runtime_seconds`


## 5. Phase E0 / E1 / E2 / E3 已完成并分别通过验收

### 5.1 E0 control

定位：

- 冻结 `Phase E` benchmark 口径。
- 保证 runtime report、debug 字段、后续 A/B 计算口径可复用。

formal report：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e0_130f_20step_control_metrics_seed42_20260606T042617Z.json`

结果：

- `pass_compare = true`
- `model_inference_runtime_seconds = 1073.231819`

说明：

- `E0` 不是优化阶段，而是后续所有单项加速验收的 control 基线。

### 5.2 E1 attention backend

正式通过的 `E1` 配置：

- `attention_backend = fa`

formal report：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e1_130f_20step_fa_metrics_seed42_20260606T045306Z.json`

结果：

- `pass_compare = true`
- `model_inference_runtime_seconds = 1016.256615`
- 相对 `E0 control` 的单项增益：
  - `1073.231819 / 1016.256615 = 1.056064x`
- 相对原版 `20 step`：
  - `1047.001905 / 1016.256615 = 1.030253x`

### 5.3 E2 torch.compile

正式通过的 `E2` 配置：

- `fa + torch.compile`
- `modulation_fusion = false`

formal report：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e2_130f_20step_compile_metrics_seed42_20260606T084506Z.json`

结果：

- `pass_compare = true`
- `model_inference_runtime_seconds = 923.9699`
- 相对 `E1` 的单项增益：
  - `1016.256615 / 923.9699 = 1.099881x`
- 相对原版 `20 step`：
  - `1047.001905 / 923.9699 = 1.133156x`

当前结论：

- 截至本次对话结束，`E2` 仍然是当前已测到的单卡最好正式结果。

### 5.4 E3 operator fusion

正式通过的 `E3` 配置：

- `fa + modulation_fusion`
- `compile = false`

formal report：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e3_130f_20step_modulation_fusion_both_metrics_seed42_20260606T082346Z.json`

结果：

- `pass_compare = true`
- `model_inference_runtime_seconds = 1007.328337`
- 相对 `E1` 的单项增益：
  - `1016.256615 / 1007.328337 = 1.008863x`
- 相对原版 `20 step`：
  - `1047.001905 / 1007.328337 = 1.039385x`

需要特别注意：

- 这次 formal 通过的 `E3` 主配置，是 `modulation / residual` 融合路径。
- `QKV fusion` 也已经实现并接线，但不是这次 `E3` formal 通过时采用的主配置。


## 6. 当前 E3 实际做了哪些高频算子融合

### 6.1 已正式通过验收的主路径：modulation / residual 融合

核心实现文件：

- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_operator_fusion.py`

当前主力融合的是 `CogVideoXBlock` 内部反复执行的小算子链：

- `LayerNorm + scale/shift`
  - 融合为 `LayerNormScaleShift`
- `residual add + gate + LayerNorm + scale/shift`
  - 融合为 `ScaleResidualLayerNormScaleShift`
- `residual + gate * ff_output`
  - 融合为 `MulAdd`

这条路径的收益逻辑是：

- 不是去改 attention 主核本体。
- 而是减少每个 block、每个 denoise step 都会反复触发的多个小 kernel launch。

### 6.2 已实现但不是当前 formal E3 主配置：QKV projection fusion

核心实现文件：

- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py`

当前接入方式：

- 把 self-attention 的 `to_q / to_k / to_v` 合并为一次 `to_qkv`
- 底层复用 diffusers 的 `fuse_projections(fuse=True)`
- 然后按输出维度拆回 `q / k / v`

这条路径已经能单独评估，但当前不是正式通过的 `E3` 主路径。


## 7. 关键实现文件

本次对话里明确涉及、且后续继续做 `Phase E` 时最需要关注的文件包括：

- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_controlnet.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_operator_fusion.py`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- `python/sglang/multimodal_gen/runtime/server_args.py`
- `python/sglang/multimodal_gen/tools/run_vividvr_inference.py`
- `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py`
- `scripts/gcc_python310_headers_wrapper.sh`

其中需要特别记住：

- `scripts/gcc_python310_headers_wrapper.sh` 是本机为 Triton launcher 编译补 `Python 3.10 headers` 的兼容 wrapper。
- 开启某些 fusion 路径做 formal 运行时，需要通过 `CC=/home/zhiheng/sglang/scripts/gcc_python310_headers_wrapper.sh` 来避免编译环境问题。


## 8. 验证与测试情况

本次对话中已经完成或确认过的验证包括：

- `E0` formal acceptance
- `E1` formal acceptance
- `E2` formal acceptance
- `E3` formal acceptance
- `py_compile`
- `PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m unittest python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py`

已知信息：

- `E0/E1` 阶段时该单测集通过过一轮 `5` 个测试。
- `E2/E3` 阶段后该单测集通过过一轮 `11` 个测试。
- 正式推理验收都按要求放在 `tmux` 中运行，日志在：
  - `/home/zhiheng/sglang/Vivid_Acceptance/logs`


## 9. 单卡 E1 + E2 + E3 全开组合测试结果

### 9.1 本次组合测试配置

这轮专门补跑了一次“单卡极致加速”组合测试，配置全开：

- `attention_backend = fa`
- `torch.compile = true`
- `qkv_fusion = true`
- `modulation_fusion = true`
- `qkv_fusion_targets = transformer,controlnet`
- `modulation_fusion_targets = transformer,controlnet`
- `num_gpus = 1`

`tmux` session：

- `vividvr_phase_e123_combo`

formal log：

- `/home/zhiheng/sglang/Vivid_Acceptance/logs/phase_e123_combo_20260606T091710Z.log`

formal report：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e123_130f_20step_fa_compile_qkv_modfusion_metrics_seed42_20260606T091719Z.json`

### 9.2 组合测试结果

结果：

- `pass_compare = true`
- `model_inference_runtime_seconds = 928.20107`
- `total_runtime_seconds = 1229.96182`
- `ssim_min = 0.9766703580534177`
- `mse_max = 20.64501190185547`
- `mae_max = 2.953833818435669`
- `failed_frame_ratio = 0.0`

从 report 的 `runtime_config` / `debug` 可确认：

- `FA`
- `torch.compile`
- `QKV fusion`
- `modulation fusion`

都真正生效了，而不是只打开了命令行开关。

### 9.3 单卡“极致加速”到底比原版快多少

相对原版 `20 step` 的模型推理时间：

- `1047.001905 / 928.20107 = 1.127990x`

也就是：

- 当前单卡把 `E1 / E2 / E3` 全部打开后，模型推理阶段约比原版快 `12.8%`。

### 9.4 但组合全开不是当前最快单卡配置

需要重点提醒下一个 Codex：

- 这次全开组合虽然通过质量验收，但它不是当前最快单卡结果。
- 之前正式通过的 `E2` 单项结果仍然更快：
  - `E2 model_inference_runtime_seconds = 923.9699`
  - `combo model_inference_runtime_seconds = 928.20107`
- 两者相对关系：
  - `923.9699 / 928.20107 = 0.995442x`

等价表述：

- 全开组合比 `E2` 单项最优结果慢约 `0.46%`。

当前实际结论应该写成：

- `全开组合` 相对原版：
  - `1.127990x`
- 当前已测到的单卡最好正式结果：
  - `E2 (FA + compile)`，约 `1.133156x`

不要把“所有加速都开”误写成“当前单卡最佳默认配置”。


## 10. 当前工作区状态提醒

本次整理交接前，`git status --short` 已经显示工作区非干净状态。可见项目里还保留着若干未提交的文档和验收产物，例如：

- `AGENTS.md`
- `docs_xzh/add_strategy/README.md`
- `docs_xzh/add_strategy/11_phase_e_acceleration_implementation.md`
- `docs_xzh/hand_over/phase_d_acceptance_completion_and_phase_e_benchmark_handover.md`
- 多个 `Vivid_Acceptance/indicator/*.json`

其中一些是正式验收产物，一些是 smoke / 中间 report。默认不要做破坏性清理，也不要因为它们是 untracked 就直接删除。


## 11. 建议下一个 Codex 的直接起手顺序

建议先按下面顺序建立上下文并继续：

1. 先读：
   - `docs_xzh/hand_over/phase_d_acceptance_completion_and_phase_e_benchmark_handover.md`
   - `docs_xzh/hand_over/phase_e_e0_e3_acceptance_and_single_gpu_combo_handover.md`
   - `docs_xzh/add_strategy/11_phase_e_acceleration_implementation.md`
2. 再确认当前单卡结论：
   - `E2` 是当前最好正式单卡结果
   - `E1+E2+E3` 全开组合不是当前最好结果
3. 若继续做单卡优化，优先补组合拆分测试：
   - `FA + compile + QKV`
   - `FA + compile + modulation`
   - 对比当前已测的 `FA + compile + QKV + modulation`
4. 若继续推进 `Phase E4`，保持同样的单项 A/B 验收纪律，不要跳过多卡单独口径。
5. 后续如果再产出新的交接文档，统一放到：
   - `/home/zhiheng/sglang/docs_xzh/hand_over`

