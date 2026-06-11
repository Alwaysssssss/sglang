# VividVR 当前实现状态与 Phase E4 下一步实施交接

更新时间：`2026-06-09 UTC`

## 1. 这份文档覆盖什么

这份交接文档面向下一位 Codex，目标是一次性说明清楚：

- 当前项目已经完成到了哪里
- 哪些阶段结论已经冻结，后续必须保护
- 当前单卡性能 control 应该采用哪一版
- 最近 `E3.3` 做了什么、为什么不建议再把它当主线
- 当前 worktree 里有哪些未提交实现
- 根据 `docs_xzh/add_strategy`，`E4` 多卡加速接下来应该怎么做

这份文档的结论晚于并部分覆盖下面这些旧文档：

- `docs_xzh/hand_over/phase_d_acceptance_completion_and_phase_e_benchmark_handover.md`
- `docs_xzh/hand_over/phase_e_e0_e3_acceptance_and_single_gpu_combo_handover.md`
- `docs_xzh/hand_over/phase_e_vividvr_acceleration_status_and_e3_execution_plan_handover.md`
- `docs_xzh/hand_over/phase_e_e32_completion_and_e33_next_steps_handover.md`


## 2. 当前总状态

### 2.1 已完成且必须继续保护的阶段

- `Phase C`
  - 单 clip 语义基线已完成，必须继续保护
- `Phase D`
  - 长视频 `clip split / timestep orchestration / latent merge / trim / stitch` 语义基线已完成正式验收，必须继续保护
- `Phase E0 / E1 / E2`
  - benchmark 口径、attention backend、`torch.compile`、runtime snapshot、验收链路都已接通
- `Phase E3.2`
  - 已完成 runtime alignment，可作为当前单卡主线 control

### 2.2 当前推荐起点

- 当前 `HEAD`：
  - `a9058b25d5df496a411ccc165bad9668efec3c3c`
- 当前推荐起点提交结论：
  - `E3.2 runtime alignment` 已完成
- 当前正确策略：
  - 冻结 `E3.2` 为单卡 control
  - 停止把 `E3.3` 当主线深挖
  - 转入 `E4` 多卡实施

### 2.3 当前默认 benchmark / acceptance 口径

当前 `Phase E` 日常 benchmark 固定为：

- 输入视频：
  - `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4`
- caption sidecar：
  - `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt`
- reference 视频：
  - `/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4`
- `num_inference_steps = 20`
- `seed = 42`

默认主判断字段仍然是：

- `pass_compare`
- `model_inference_runtime_seconds`

`50 step` 只保留给阶段性最终回归，不作为当前日常性能迭代口径。


## 3. 当前正式性能结论

### 3.1 原版与单卡演进锚点

- 原版 `20 step`：
  - `model_inference_runtime_seconds = 1047.001905`
  - reference report：
    - `/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f_report.json`
- `E1 = FA`：
  - `1016.256615`
- `E2 = FA + torch.compile`：
  - `923.9699`
- `E3.2 current control`：
  - `935.243947`
  - 指标：`/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e32_runtime_e2_align_130f_20step_compile_metrics_seed42_20260609T025514Z.json`

### 3.2 当前 single-card control 应该怎么认定

这里要分清两件事：

- 历史上单卡最好正式结果仍然是 `2026-06-06` 的 `E2 = 923.9699`
- 但当前继续开发时，应该把 `2026-06-09` 的 `E3.2 = 935.243947` 当成现实 control

原因：

- `E3.2` 保留了当前 `sglang` runtime / snapshot / CLI / 接线状态
- 同机 fresh `E2` 复核为 `934.690479`，与 `E3.2` 只差 `0.553468s`
- 当前更合理的比较方法是 same-day control / treatment A/B，而不是反复拿旧日期的 `923.9699` 作为唯一绝对 gate

fresh `E2` 复核件：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e2_recheck_130f_20step_compile_metrics_seed42_20260609T031841Z.json`
- `pass_compare = true`
- `model_inference_runtime_seconds = 934.690479`

### 3.3 最新 `E3.2` 复测

本轮又重跑了一次 `E3.2` formal，用于确认性能没有明显回退，同时检查日志收敛情况。

复测件：

- 指标：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e32_runtime_e2_align_recheck_130f_20step_compile_metrics_seed42_20260609T100112Z.json`
- 日志：
  - `/home/zhiheng/sglang/Vivid_Acceptance/logs/phase_e32_runtime_e2_align_recheck_130f_20step_compile_20260609T100100Z.log`
- 视频：
  - `/home/zhiheng/sglang/Vivid_Acceptance/result_videos/phase_e32_runtime_e2_align_recheck_130f_20step_compile_seed42_20260609T100112Z.mp4`

结果：

- `pass_compare = true`
- `model_inference_runtime_seconds = 938.206415`
- 相对 `E3.2 control` 慢 `2.962468s`
- 约 `0.32%`

当前判断：

- 这是一个很小的负向波动
- 更像同机重复测试噪声
- 当前没有证据说明 `E3.2` 发生了明确性能回退


## 4. E3.3 当前状态与结论

### 4.1 本轮 E3.3 实际做了什么

当前 dirty worktree 中已经落过两条 `E3.3` 候选：

- `QK norm fusion`
- `QKV fusion`

这两条都尽量复用了 `sglang` 已有底层能力，没有新写私有算子。

### 4.2 E3.3 formal 结果

`QK norm fusion`：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e33_qk_norm_130f_20step_compile_cc_metrics_seed42_20260609T064847Z.json`
- `pass_compare = true`
- `model_inference_runtime_seconds = 946.328917`

`QKV fusion` formal 1：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e33_qkv_130f_20step_compile_gpu1_metrics_seed42_20260609T071927Z.json`
- `pass_compare = true`
- `model_inference_runtime_seconds = 938.510198`

`QKV fusion` formal 2：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/test_video_long_960x720_130f_metrics_seed42_20260609T093234Z.json`
- `pass_compare = true`
- `model_inference_runtime_seconds = 941.64526`

### 4.3 为什么不建议继续把 E3.3 当主线

结论很清楚：

- 这两条 `E3.3` 路径都通过了 compare
- 但都没有赢过当前 `E3.2` control
- 说明当前单卡大头收益仍然主要来自：
  - `FA`
  - `torch.compile`

目前更合理的判断是：

- `torch.compile` 已经先吃掉了最值钱的一块单卡图级收益
- 后续手工 fusion 更容易落在局部结构重写、launch 数和 memory traffic 的边角优化上
- 这类优化在当前链路里可能只有几个点，甚至会与 compile 形成负交互

因此当前建议：

- 保留 `E3.3` 候选实现和测试，不要删
- 但不要继续把 `E3.3` 当成当前主线
- 现在应当转向 `E4`


## 5. 当前 worktree 现实状态

### 5.1 当前 dirty 文件

当前 `git status --short` 显示：

- `M python/sglang/multimodal_gen/runtime/loader/fsdp_load.py`
- `M python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py`
- `M python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- `M python/sglang/multimodal_gen/runtime/server_args.py`
- `M python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py`
- `M python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py`
- `M python/sglang/multimodal_gen/tools/run_vividvr_inference.py`
- `?? docs_xzh/hand_over/phase_e_e32_completion_and_e33_next_steps_handover.md`

### 5.2 这些未提交改动的含义

#### `fsdp_load.py`

这里只做了日志收敛，不是性能路径改写：

- 把启动初期那串极长的 unused key warning 收成简短摘要
- 真实错误分支没有改

换句话说：

- 这是低风险可保留改动
- 作用是让正式日志更干净

#### `cogvideox_attention_backend.py` / `vividvr_pipeline.py` / `server_args.py` / `run_vividvr_inference.py`

这些文件主要承载：

- `QKV fusion`
- `QK norm fusion`
- 单进程 model-parallel 初始化辅助
- 相关 CLI / runtime snapshot 字段

注意：

- 这些实现可以保留为候选分支
- 但它们当前不是性能 winning path

### 5.3 当前单测状态

最近一次相关单测已通过：

- `/home/zhiheng/sglang/.venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py`

结果：

- `26 passed`

另外还做过：

- `python -m py_compile python/sglang/multimodal_gen/runtime/loader/fsdp_load.py`


## 6. 当前 VividVR 与 E4 的真实缺口

### 6.1 现有基础设施已经具备什么

`sglang` 已有可直接复用的底层工程包括：

- distributed / model parallel 基础设施：
  - `python/sglang/multimodal_gen/runtime/distributed/parallel_state.py`
- `ServerArgs` 中已有的多卡参数：
  - `num_gpus`
  - `tp_size`
  - `sp_degree`
  - `ulysses_degree`
  - `ring_degree`
  - `dp_size`
  - `enable_cfg_parallel`

### 6.2 当前 VividVR 还没有真正接通什么

当前 `VividVR` 多卡线还没接通，不是“只差打开开关”，核心缺口至少有三个：

1. `run_vividvr_inference.py` 仍把多卡相关参数写死成 `1`
2. 当前没有真正可用的 `VividVR` 多进程启动方式
3. `VividVR` denoise 主链还没有和真实多进程并行组初始化接通

当前 `build_server_args()` 里仍然是：

- `num_gpus=1`
- `tp_size=1`
- `dp_size=1`
- `dp_degree=1`
- `sp_degree=1`

因此当前代码虽然已经能记录并行字段，也能在 `E3.3` 候选中做单进程 model-parallel 辅助，但这不是 `E4` 真正需要的多卡实现。


## 7. E4 的正确目标与顺序

这部分结论来自 `docs_xzh/add_strategy/11_phase_e_acceleration_implementation.md`，并结合本轮实现现状做了收口。

### 7.1 E4 的目标

`E4` 的目标不是做“看起来能用两张卡”的表面分发，而是：

- 在不破坏 `Phase D` 长视频主语义的前提下
- 让 `VividVR` 真正复用 `sglang` 已有 distributed / model parallel 基础设施
- 获得单视频单请求 latency 的真实下降

### 7.2 E4 的优先级顺序

对当前单视频 latency 场景，推荐顺序是：

1. `SP`
2. `TP`
3. 可选 `CFG parallel`
4. `DP` 不作为首要目标

原因：

- 当前 benchmark 是单视频、`batch=1`
- `DP` 对单请求 latency 几乎没有直接帮助
- `SP / TP` 才是当前模型真正能吃到的多卡降时延路线

### 7.3 E4 的语义红线

多卡实现必须继续保护下面这些语义：

- `Phase C` 单 clip 语义
- `Phase D` 长视频 `timestep` 级多 clip 同步推进
- overlap latent merge
- trim / stitch
- `drop first 3 frames + crop padding + AdaIN/reference color fix`

明确不建议的第一版实现方式：

- `clip0` 在 `GPU0` 独立完整跑完
- `clip1` 在 `GPU1` 独立完整跑完
- 最后再粗拼接

这类方案虽然工程上看起来简单，但容易偏离 `Phase D` 已验收主语义，也不符合“尽量复用 `sglang` 底层并行”的方向。


## 8. E4 首轮实施建议

### 8.1 建议从 `E4.1 = SP-only` 开始

第一轮只做最保守版本：

- 只做 `SP`
- 不叠 `TP`
- 不叠 `CFG parallel`
- 不混新的 backend / compile / fusion 变量

推荐首个尝试配置：

- `tp_size = 1`
- `sp_degree = num_gpus`
- `ulysses_degree = sp_degree`
- `ring_degree = 1`

如果后续要尝试 `ring_degree > 1`，则 attention backend 只能使用：

- `fa`
- `sage_attn`

### 8.2 启动方式建议

不要继续依赖当前写死目标脚本路径的老 `launch_distributed()`。

更合理的实现是二选一：

- 让 `run_vividvr_inference.py` 直接支持 `torchrun`
- 或把现有 `launch_distributed()` 泛化成可传目标脚本路径的公共工具，再让 `VividVR` 复用

当前更推荐第一种：

- `torchrun` 直接驱动 `run_vividvr_inference.py`
- 参数和验收入口更统一

### 8.3 建议优先改动的文件

优先关注：

- `python/sglang/multimodal_gen/tools/run_vividvr_inference.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`

必要时补：

- `VividVR` 专用 distributed entry wrapper


## 9. E4 具体实施 checklist

下一位 Codex 建议按这个顺序推进：

1. 先冻结当前 `E3.2` control，不要再改动 benchmark 口径。
2. 在 `run_vividvr_inference.py` 中把多卡相关参数从写死值改成真实 CLI / `ServerArgs` 透传。
3. 让 `run_vividvr_inference.py` 具备 `torchrun` 友好入口。
4. 在 `VividVR` pipeline / denoise 主链里接入真实的 distributed / model parallel 初始化与 rank 侧行为。
5. 第一轮只做 `SP-only`，保持 `FA + torch.compile` 不变，不混入新 fusion。
6. 先跑最小 smoke，确认：
   - 不死锁
   - 不 hang
   - rank 间 shape 一致
   - compare 继续通过
7. 再跑正式 `130f / 20 step / seed=42` formal。
8. formal 至少连续跑两次，确认秒数和稳定性。
9. 只有 `SP-only` 稳定后，再考虑 `TP`。
10. `TP` 必须以“已冻结 SP 配置”为 control，单独做 A/B。


## 10. E4 验收标准

`E4` 每个子阶段都应满足：

- `pass_compare = true`
- 同一配置至少连续两次稳定运行
- 无 deadlock
- 无 hang
- 无 rank mismatch
- report / indicator 中完整写出并行配置
- 主判断仍以 `model_inference_runtime_seconds` 为准

不建议把下面这些当成主要结论依据：

- unsync stage metric 的局部变化
- 单次偶然秒数尖峰


## 11. 当前不建议做的事

下一位 Codex 默认不建议：

- 继续把 `E3.3` 当成主线开放式深挖
- 在 `E4.1` 里同时混入新的 backend / fusion / compile 实验
- 把 `DP` 当成单视频 latency 第一优先级
- 先写 `VividVR` 私有多卡调度器，再考虑是否复用 `sglang` 基础设施
- 用“每个 temporal clip 各自完整跑完再拼接”的方式替代 `Phase D` 主语义


## 12. 下一位 Codex 接手时最重要的判断

一句话总结当前状态：

- `Phase C` 和 `Phase D` 已完成，语义基线稳定
- 当前单卡主线应冻结在 `E3.2`
- `E3.3` 候选实现已落地、可保留，但不是性能 winning path
- 现在最值得投入的工程方向是 `E4`，优先 `SP-only`

如果下一位 Codex 只记住三件事，应当是：

1. 当前 control 是 `E3.2`，不是继续开放式试单卡 fusion。
2. `E4` 的首轮目标是“稳定可复现的多卡 latency 降时延”，不是做复杂组合拳。
3. 任何多卡实现都不能破坏 `Phase D` 的长视频主语义。
