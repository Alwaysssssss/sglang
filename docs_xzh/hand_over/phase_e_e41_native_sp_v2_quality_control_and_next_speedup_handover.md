# VividVR E4.1 Native SP 当前状态与 v2 质量保真提速交接

更新时间：`2026-06-11 UTC`

## 1. 这份文档覆盖什么

这份 handover 面向下一位继续推进 `Phase E4.1 native SP` 的 Codex，目标是一次性说清：

- 当前项目应以哪一版代码和哪几份正式产物作为基线
- `native SP`、`v1`、`v2` 三条路径各自代表什么语义与速度/质量结论
- 为什么下一步不能再回到 `v1` 或更弱语义，而必须站在 `v2` 质量合同上继续提速
- 当前 worktree 里哪些东西是已冻结基线，哪些只是未完成实验
- 下一步继续优化时，什么情况下可以在 smoke 阶段直接判策略失败

这份文档晚于并补充：

- `docs_xzh/hand_over/phase_e_current_status_and_e4_next_steps_handover.md`
- `docs_xzh/add_strategy/12_phase_e_sp_native_acceleration_plan.md`
- `docs_xzh/add_strategy/13_phase_e_sp_quality_closure_plan.md`


## 2. 当前必须冻结的总判断

### 2.1 阶段语义基线

以下基线仍然必须继续保护：

- `Phase C`
  - 单 clip 主语义基线已验收，不能回归
- `Phase D`
  - 长视频 `clip split / timestep orchestration / latent merge / trim / stitch` 主语义已验收，不能回归
- `Phase E3.2`
  - 当前单卡 control 仍应认定为：
    - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e32_runtime_e2_align_130f_20step_compile_metrics_seed42_20260609T025514Z.json`
    - `model_inference_runtime_seconds = 935.243947`
    - `ssim_mean = 0.9844564849526698`
    - `ssim_min = 0.9799698863913181`

### 2.2 当前 E4.1 的冻结结论

`E4.1 native SP` 已经证明两件事：

1. `VividVR` 可以走真实双卡 `SP` 多进程推理，而不是“名义双卡、实质单卡”。
2. native `SP` 的主速度收益已经成立，但最激进的 fast 版本会损坏质量。

因此，当前主线不是“继续证明双卡能跑”，而是：

- 保留 `v2` 的质量语义
- 把 `v2` 的 denoise 成本压回接近 `v1`

### 2.3 当前推荐代码锚点

当前 `HEAD`：

- `c0008cd89c1fb56f348b2c29820703da8b6a511b`
- commit message：
  - `Restore vividvr native SP v1/v2 connector semantics`

这次提交的含义是：

- 代码已经收敛回当前确认有效的 `v1 / v2` 两条语义路径
- 下一步做提速时，应直接从这个提交继续
- 不要把此前那些未成型的 hybrid / gating / 分布式 attention 实验当成已冻结主线


## 3. 当前 benchmark / acceptance 口径

### 3.1 默认 formal 口径

当前 `Phase E` 日常正式 benchmark 仍固定为：

- 输入视频：
  - `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4`
- caption sidecar：
  - `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt`
- prompt：
  - `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`
- reference 视频：
  - `/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4`
- `num_inference_steps = 20`
- `seed = 42`
- `attention_backend = fa`
- `torch.compile = on`
- `num_gpus = 2`
- `tp_size = 1`
- `sp_degree = 2`
- `ulysses_degree = 2`
- `ring_degree = 1`

### 3.2 主判断字段

当前 formal 仍然至少要看：

- `pass_compare`
- `model_inference_runtime_seconds`
- `summary.ssim_mean`
- `summary.ssim_min`
- `request_metrics.steps`
- `vividvr_long_video_denoising_loop`
- `runtime_config.connector_context_mode`
- `runtime_config.control_context_shape_local`
- `runtime_config.control_context_shape_global`


## 4. 当前四个关键锚点

### 4.1 单卡 control：`E3.2`

正式产物：

- 指标：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e32_runtime_e2_align_130f_20step_compile_metrics_seed42_20260609T025514Z.json`

结论：

- `model_inference_runtime_seconds = 935.243947`
- `vividvr_long_video_denoising_loop = 771121.1627759039 ms`
- steady-state denoise 约 `38.6s/it`
- `ssim_mean = 0.9844564849526698`
- `ssim_min = 0.9799698863913181`

### 4.2 runtime-only 双卡 control：`E4.1 SP-only`

正式产物：

- 指标：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e41_sp_only_130f_20step_compile_metrics_seed42_20260611T041018Z.json`

结论：

- `model_inference_runtime_seconds = 933.725862`
- 速度几乎不变，说明“只接通 runtime / metadata”本身不会带来有效加速
- 质量仍接近单卡，说明 distributed 接线本身不是质量问题来源

这版的价值是：

- 它证明“质量问题不是双卡环境本身造成的”
- 它也是判断 native `SP` 质量退化是否来自模型内部 shard 语义改变的重要对照

### 4.3 最快 native `SP` fast path

正式产物：

- 指标：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e41_native_sp_only_130f_20step_compile_metrics_seed42_20260611T052918Z.json`
- 日志：
  - `/home/zhiheng/sglang/Vivid_Acceptance/logs/phase_e41_native_sp_formal_20260611T052907Z.log`
- 视频：
  - `/home/zhiheng/sglang/Vivid_Acceptance/result_videos/phase_e41_native_sp_only_130f_20step_compile_seed42_20260611T052918Z.mp4`

结论：

- `model_inference_runtime_seconds = 396.745880`
- `vividvr_long_video_denoising_loop = 241283.17333385348 ms`
- steady-state denoise 约 `12.06s/it`
- 相对 `E3.2` 加速约 `2.357x`
- `ssim_mean = 0.9627860811380421`
- `ssim_min = 0.9152052581958419`

这版的意义：

- 速度极好
- 但质量不可接受
- 因此不能作为后续 release 候选，只能作为“理论速度上界附近”的参考

### 4.4 `quality opt_v1`

首次正式产物：

- 指标：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e41_native_sp_quality_opt_v1_130f_20step_compile_metrics_seed42_20260611T081518Z.json`
- 日志：
  - `/home/zhiheng/sglang/Vivid_Acceptance/logs/phase_e41_native_sp_quality_opt_v1_formal_20260611T081507Z.log`

本轮 recheck 产物：

- 指标：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e41_native_sp_quality_opt_v1_130f_20step_compile_metrics_seed42_20260611T133530Z.json`
- 日志：
  - `/home/zhiheng/sglang/Vivid_Acceptance/logs/phase_e41_native_sp_quality_opt_v1_recheck_20260611T133517Z.log`
- 视频：
  - `/home/zhiheng/sglang/Vivid_Acceptance/result_videos/phase_e41_native_sp_quality_opt_v1_130f_20step_compile_seed42_20260611T133530Z.mp4`

冻结结论：

- `connector_context_mode = sp_exact_local_attention`
- `control_context_shape_local = [2, 13500, 3072]`
- `control_context_shape_global = null`
- 首次 formal：
  - `model_inference_runtime_seconds = 472.011449`
  - steady-state denoise 约 `15.64s/it`
  - `ssim_mean = 0.97853561647603`
  - `ssim_min = 0.9534618740006351`
- recheck：
  - `model_inference_runtime_seconds = 470.026965`
  - steady-state denoise 约 `15.7s/it`
  - `ssim_mean = 0.9785842231502916`
  - `ssim_min = 0.9529226959647417`

当前认定：

- `v1` 质量比 fast path 明显好
- 但仍未回到单卡/原版级别，尤其 seam 相关窗口仍偏弱
- 它不是后续最终质量目标，只是当前可接受的速度下界参考

### 4.5 `quality opt_v2`

首次正式产物：

- 指标：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e41_native_sp_quality_opt_v2_130f_20step_compile_metrics_seed42_20260611T084257Z.json`
- 日志：
  - `/home/zhiheng/sglang/Vivid_Acceptance/logs/phase_e41_native_sp_quality_opt_v2_formal_20260611T084245Z.log`

本轮 recheck 产物：

- 指标：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e41_native_sp_quality_opt_v2_130f_20step_compile_metrics_seed42_20260611T134903Z.json`
- 日志：
  - `/home/zhiheng/sglang/Vivid_Acceptance/logs/phase_e41_native_sp_quality_opt_v2_recheck_20260611T134851Z.log`
- 视频：
  - `/home/zhiheng/sglang/Vivid_Acceptance/result_videos/phase_e41_native_sp_quality_opt_v2_130f_20step_compile_seed42_20260611T134903Z.mp4`

冻结结论：

- `connector_context_mode = sp_exact_global_control_attention`
- `control_context_shape_local = [2, 13500, 3072]`
- `control_context_shape_global = [2, 27000, 3072]`
- 首次 formal：
  - `model_inference_runtime_seconds = 541.607954`
  - steady-state denoise 约 `19.11s/it`
  - `ssim_mean = 0.9845841448221968`
  - `ssim_min = 0.9800638150295611`
- recheck：
  - `model_inference_runtime_seconds = 539.324976`
  - steady-state denoise 约 `19.1s/it`
  - `ssim_mean = 0.9846050631221304`
  - `ssim_min = 0.9778964153159052`

当前认定：

- `v2` 已经基本回到单卡质量水平
- `v2` 是当前新的质量 control
- 下一步所有提速都必须站在这条语义合同上继续做


## 5. 当前三条语义路径到底代表什么

### 5.1 fast native `SP`

本质：

- video token 已做原生 shard
- connector 没有恢复 full control context 语义

结果：

- 最快
- 但最明显地损坏 `Vivid-VR` 的原版 guidance contract

### 5.2 `v1`

当前入口：

- 环境变量：
  - `SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=deferred_global`
- runtime snapshot 表现为：
  - `connector_context_mode = sp_exact_local_attention`

语义含义：

- connector attention 数学更接近原版 exact attention
- 但 connector 实际看到的 control context 仍是 local-only

结果：

- 速度能压到约 `15.6s/it`
- 质量明显优于 fast path
- 但仍达不到 `v2` / 单卡水准

### 5.3 `v2`

当前入口：

- 环境变量：
  - `SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global`
- runtime snapshot 表现为：
  - `connector_context_mode = sp_exact_global_control_attention`

语义含义：

- connector 的 local `q` 在数学上看到完整 control sequence 的 `k/v`
- `c_mlp` 仍只消费 local 对齐的 control context

结果：

- 质量已基本修回
- 速度显著慢于 `v1`

### 5.4 当前项目的核心共识

后续不能把问题定义成：

- “要不要在 `v1` 和 `v2` 之间取中间值”

当前正确问题是：

- “如何保留 `v2` 的 global connector 语义，同时把它的 denoise 成本压回接近 `v1`”

也就是说：

- `v1` 是速度参考
- `v2` 是质量参考
- 新实现必须尽量拿到“`v2` 语义 + `v1` 速度”


## 6. 当前为什么会卡在 `v2` 的速度上

### 6.1 速度差异几乎全在 denoise loop

三版的主要差异都集中在：

- `vividvr_long_video_denoising_loop`
- `request_metrics.steps`

不是：

- clip preparation
- decode / postprocess
- compare harness

因此后续优化重点必须继续放在 connector/control context 处理，不要先去动后处理。

### 6.2 当前最可能的成本来源

按现有代码和验收现象，`v2` 慢主要不是因为多了少量 Python 判断，而是因为：

- global control context 显式恢复
- `local -> global` 的 materialization / gather / layout 处理
- connector attention 读取完整 control sequence

当前最应该优先怀疑的成本段：

- packed local control states 的构造
- `sequence_model_parallel_all_gather(..., dim=1)` 或等价恢复
- global context 物化后的 `view / contiguous / dtype` 流程
- connector `to_q / to_k`
- connector exact attention kernel 本身

### 6.3 当前不应再走的方向

默认应判为错误方向的尝试：

- 回退到 local-only control context
- 用摘要 token / 局部窗口 / 稀疏近似静默替代完整 global control
- 把 `v2` 语义偷偷改弱，然后只看速度
- 在同一轮里同时引入：
  - 新 backend
  - 新 fusion
  - 新 compile 变量
  - TP / CFG parallel


## 7. 当前建议的下一步主线

### 7.1 先把 `v2` 当质量 control 冻结

下一位接手时，首先应确认：

- 当前质量 control 不是单卡，也不是 `v1`
- 而是 `v2`

只要一个新 patch 的质量明显掉回：

- `ssim_min < 0.97`
- 或者 seam 窗口肉眼又出现模糊 / 跳变

就应直接判它没有守住 `v2` 质量合同。

### 7.2 目标是把 `v2` 的速度往 `v1` 拉

当前最现实的目标不是直接回到 fast path 的 `12.06s/it`，而是：

- 尽量逼近 `v1` 的 `15.6s/it`
- 至少不要继续停在 `19.1s/it`

建议的阶段目标：

- 第一阶段：
  - 先争取把 steady-state denoise 拉到 `<= 16.0s/it`
- 第二阶段：
  - 再看 formal 的 `model_inference_runtime_seconds` 是否明显向 `v1` 靠拢

### 7.3 `16.0s/it` 作为 smoke 早停 gate

这是当前最重要的执行纪律之一。

后续任何新策略在 smoke 阶段，如果排除 warmup 后观察到：

- denoise steady-state 明显慢于 `16.0s/it`

就应默认：

- 这条策略当前不成功
- 不要继续浪费 formal 验收时间
- 应该中断并转向下一个优化思路

这条 gate 的含义不是“`16.0s/it` 就是最终成功线”，而是：

- 这是当前阶段足够有用的快速筛选线
- 因为 `v1` 已经证明 `15.6s/it` 左右是现实可达的

### 7.4 当前最值得做的优化方向

优先级最高的主线仍然是：

- 保持 `v2` 的 global connector 语义
- 但减少显式 global control tensor 的长期物化成本

更具体地说，优先尝试：

1. distributed exact connector attention
   - 目标是让 local `q` 仍然看到全局 `k/v`
   - 但不再把完整 global control 长时间常驻为大张量

2. chunked / short-lived global restore
   - 如果短期内做不到上面的 exact distributed 实现
   - 至少让 global restore 生命周期更短，减少额外物化与内存流量

3. 工程降本
   - 去掉重复的 `contiguous / view / reshape / cast`
   - 避免一个 step / tile 内重复构造相同 connector context


## 8. 可复现实验入口

### 8.1 `v1` formal 参考命令

```bash
tmux new-session -d -s vividvr_e41_v1_recheck \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && export PYTHONPATH=python && export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=deferred_global && /home/zhiheng/sglang/.venv/bin/torchrun --nproc_per_node=2 --master_port=30063 python/sglang/multimodal_gen/tools/run_vividvr_inference.py --input-video /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4 --caption-file /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt --prompt-file /home/zhiheng/Vivid-VR/input/720p/prompt.txt --reference-video /home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4 --num-inference-steps 20 --seed 42 --num-gpus 2 --tp-size 1 --sp-degree 2 --ulysses-degree 2 --ring-degree 1 --dist-timeout 3600 --master-port 30063 --attention-backend fa --enable-torch-compile --warmup --warmup-steps 1 --artifact-prefix phase_e41_native_sp_quality_opt_v1_130f_20step_compile 2>&1 | tee Vivid_Acceptance/logs/phase_e41_native_sp_quality_opt_v1_recheck_$(date -u +%Y%m%dT%H%M%SZ).log'
```

### 8.2 `v2` formal 参考命令

```bash
tmux new-session -d -s vividvr_e41_v2_recheck \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && export PYTHONPATH=python && export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global && /home/zhiheng/sglang/.venv/bin/torchrun --nproc_per_node=2 --master_port=30064 python/sglang/multimodal_gen/tools/run_vividvr_inference.py --input-video /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4 --caption-file /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt --prompt-file /home/zhiheng/Vivid-VR/input/720p/prompt.txt --reference-video /home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4 --num-inference-steps 20 --seed 42 --num-gpus 2 --tp-size 1 --sp-degree 2 --ulysses-degree 2 --ring-degree 1 --dist-timeout 3600 --master-port 30064 --attention-backend fa --enable-torch-compile --warmup --warmup-steps 1 --artifact-prefix phase_e41_native_sp_quality_opt_v2_130f_20step_compile 2>&1 | tee Vivid_Acceptance/logs/phase_e41_native_sp_quality_opt_v2_recheck_$(date -u +%Y%m%dT%H%M%SZ).log'
```

说明：

- 当前 `SP` 验收必须走真实 `torchrun --nproc_per_node=2`
- 不要再用单进程假设去跑 `sp_degree=2`
- 如果指定端口被占用，当前脚本会自动换到其他可用端口


## 9. 当前 worktree 状态与注意事项

### 9.1 当前 `git status --short`

截至写本文档时，工作区仍是脏的：

- `M docs_xzh/add_strategy/README.md`
- `M python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py`
- `M python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py`
- `?? docs_xzh/add_strategy/13_phase_e_sp_quality_closure_plan.md`

### 9.2 如何理解这些脏改动

当前这些 dirty 文件不应默认视为“已经验收的主线优化”。

更具体地说：

- `c0008cd89` 才是当前应继续站住的代码锚点
- `cogvideox_attention_backend.py` 相关脏改动更像中途优化实验残留
- `13_phase_e_sp_quality_closure_plan.md` 是质量闭环计划文档，本身有参考价值，但不是 runtime 改动

下一位接手时应先判断：

- 是在现有 dirty worktree 上继续做实验
- 还是先整理出干净分支 / 干净提交再推进

但在没有明确确认前：

- 不要把这些脏改动直接当成已冻结成功方案


## 10. 下一位 Codex 的默认行动顺序

建议按下面顺序接手，不要跳步：

1. 先读：
   - `docs_xzh/add_strategy/12_phase_e_sp_native_acceleration_plan.md`
   - `docs_xzh/add_strategy/13_phase_e_sp_quality_closure_plan.md`
   - 本文档

2. 先确认当前代码仍能重跑出：
   - `v1` 对齐 `081518Z`
   - `v2` 对齐 `084257Z`

3. 冻结共识：
   - `v2` 是质量 control
   - `v1` 是速度参考
   - fast path 只是速度上界参考，不是 release 候选

4. 新策略一律先跑 smoke：
   - 只看 denoise steady-state
   - `> 16.0s/it` 直接考虑止损

5. 只有 smoke 过线后，再跑 formal：
   - 同时检查 `ssim_mean / ssim_min`
   - 特别关注 seam 邻域窗口是否重新出现模糊 / 跳变


## 11. 最后一句结论

当前项目不是“还没做出真正双卡加速”，而是已经走到更细的阶段：

- 真正的 native `SP` 双卡加速已经做出来了
- `v2` 也已经把质量基本修回来了
- 当前唯一还没闭环的是：
  - **如何在不牺牲 `v2` 质量语义的前提下，把 denoise 从 `19.1s/it` 压回接近 `15.6s/it`**

后续所有实现、验收和止损判断，都应围绕这一个问题展开。
