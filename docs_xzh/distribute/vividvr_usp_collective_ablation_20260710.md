# VividVR USP collective 消融记录（2026-07-10）

## 结论摘要

- packed QKV A2A 在 CFG2 x SP2、`fa_sp`、`eager_global`、`torch.compile` 下正确生效：rank 0 的 3 个 profiler timestep 中，A2A 启动数从 `1152` 降到 `576`，下降 `50%`；对应 GPU annotation 总时长从 `505.611 ms` 降到 `404.369 ms`，下降 `20.02%`。
- prefix `all_gather_into_tensor` 开关没有减少当前 compile 正式路径的 collective：B0 和 P2 均为 `300` 次 gather，GPU annotation 总时长分别为 `46.734 ms` 和 `47.088 ms`。原因是旧的 tensor-list `dist.all_gather` 已被 Dynamo functionalize 为 coalesced tensor gather。
- P1/P2/P3 相对 B0 的 130 帧 smoke 视频均通过 `SSIM >= 0.98` 门槛；P3 的 `SSIM mean=0.991444`、`min=0.988110`，无失败帧。
- packed 优化降低了 launch count，但 5-step steady median 没有可测收益：B0 为 `9487.627 ms/step`，P1 为 `9523.407 ms/step`。profiler 中三步 A2A 总量只减少约 `101.242 ms`，不足以支持仅凭 smoke 推断正式耗时能从基线 `194.2424s` 降至 `175s`。
- 两个开关继续保持默认关闭。是否接受 P3 取决于后续严格按 `docs_xzh/run_command/mock_test.md` 执行的 warmup + formal 服务验收。

## 代码与 profiler 修复

本轮实现提交：

- `9d14482b6`：暴露两个默认关闭的 USP collective 开关。
- `2f7d10d77`：加入 packed QKV A2A primitive。
- `433cfd2d9`：在 USP attention processor 中选择优化路径。
- `75a61848d`：向 CogVideoX transformer/controlnet 传播开关。
- `df7ba8530`：接入 VividVR runtime 配置与报告。
- `be1d743d9`：加入双 rank NCCL bitwise-exact 测试。
- `79b65ed34`：直跑工具暴露 profiler 参数。
- `dbf6dc42c`：修复 profiler step hook，使其按全局 denoising timestep 推进。

首次消融发现 profiler hook 没有推进，产生的 trace 不包含有效 denoising timestep。该批无效产物保留在：

- `Vivid_Acceptance/logs/usp_ablation_invalid_profiler_hook_20260710`
- `Vivid_Acceptance/profiles/usp_ablation_invalid_profiler_hook_20260710`
- `Vivid_Acceptance/result_videos/usp_ablation_invalid_profiler_hook_20260710`
- `Vivid_Acceptance/indicator/usp_ablation_invalid_profiler_hook_20260710`

`dbf6dc42c` 修复后，单 clip 和 multi clip 都在一个全局 timestep 完成后只推进一次 profiler；本记录只使用修复后的产物。

## 环境与固定口径

- GPU：GPU0-3，`NVIDIA A100-SXM4-80GB`。
- 拓扑：GPU0-3 两两均为 `NV12`，CPU affinity `0-31,64-95`，NUMA 0。
- 并行：4 processes，CFG world size 2，SP/Ulysses world size 2，ring 1。
- attention：请求 `fa`，有效 transformer/controlnet backend 均为 `fa_sp`。
- connector：`SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global`。
- compile：transformer 和 controlnet 均启用 `torch.compile`。
- 输入：`/home/zhiheng/input/test_video_long_960x720_130f.mp4`。
- caption：`Vivid_Acceptance/captions/service_sidecars/quad-test-video-long-960x720-130f-run2-20260708T060202Z.txt`，2 行。
- 推理：130 帧，5 steps，seed 42，temporal process frames 121，upscale 1.0。
- profiler：1 个 wait/warmup 阶段后捕获 3 个全局 denoising timesteps。

四组均使用计划文档 Task 7 步骤 6 的完整 `torchrun` 命令，仅开关不同：

| 配置 | 追加参数 |
|---|---|
| B0 | 无 |
| P1 | `--enable-usp-packed-qkv-a2a` |
| P2 | `--enable-usp-prefix-all-gather-into-tensor` |
| P3 | `--enable-usp-packed-qkv-a2a --enable-usp-prefix-all-gather-into-tensor` |

日志位于 `Vivid_Acceptance/logs/usp_ablation/{B0,P1,P2,P3}.log`。四组均完成 compile、5 steps、decode 和报告写入。

## Profiler 结果

只统计 trace 中 `cat=gpu_user_annotation` 的 NCCL annotation，避免把 CPU wrapper 和 CUDA kernel 重复计数。

| 配置 | A2A count | A2A duration | 相对 B0 | gather count | gather duration | 相对 B0 |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 1152 | 505.611 ms | - | 300 | 46.734 ms | - |
| P1 | 576 | 404.369 ms | count -50.00%，duration -20.02% | 300 | 46.579 ms | duration -0.33% |
| P2 | 1152 | 505.876 ms | duration +0.05% | 300 | 47.088 ms | duration +0.76% |
| P3 | 576 | 408.650 ms | count -50.00%，duration -19.18% | 300 | 46.354 ms | duration -0.81% |

Trace 路径：

- `Vivid_Acceptance/profiles/usp_ablation/B0/profile_trace-3_steps-global-rank0.trace.json.gz`
- `Vivid_Acceptance/profiles/usp_ablation/P1/profile_trace-3_steps-global-rank0.trace.json.gz`
- `Vivid_Acceptance/profiles/usp_ablation/P2/profile_trace-3_steps-global-rank0.trace.json.gz`
- `Vivid_Acceptance/profiles/usp_ablation/P3/profile_trace-3_steps-global-rank0.trace.json.gz`

A2A count 与模型结构一致：`48 processors x 2 clips x 3 profiled steps x 4 collectives = 1152`；packed 路径把每层的三次 input A2A 合成一次，因此变为每层两次 A2A，共 `576`。

## 5-step smoke 时间

`model_inference_runtime_seconds` 包含 long-video preparation、denoising、profiler flush、decode 和 postprocess，不能作为正式 20-step 性能结论。第一步还包含 compile，因此额外列出 steps 2-5 的 median。

| 配置 | model inference | long preparation | denoising stage | decode | steady median |
|---|---:|---:|---:|---:|---:|
| B0 | 331.844 s | 65.979 s | 129.252 s | 106.997 s | 9.488 s/step |
| P1 | 333.632 s | 64.979 s | 135.306 s | 105.823 s | 9.523 s/step |
| P2 | 327.432 s | 65.015 s | 127.578 s | 105.711 s | 9.532 s/step |
| P3 | 323.723 s | 64.777 s | 125.935 s | 105.328 s | 9.534 s/step |

上述总时间的差异主要来自首次 compile 和 profiler flush，不代表优化收益。steady median 显示 P1/P3 没有可测的逐步加速。

## Smoke 数值等价

reference 为 B0 smoke 输出，candidate 为同轮 P1/P2/P3 输出，比较 130 帧。

| 配置 | SSIM mean | SSIM min | MSE mean | 失败帧 | 结论 |
|---|---:|---:|---:|---:|---|
| P1 | 0.991475 | 0.988179 | 3.133780 | 0 | PASS |
| P2 | 0.991500 | 0.988220 | 3.159122 | 0 | PASS |
| P3 | 0.991444 | 0.988110 | 3.152107 | 0 | PASS |

比较报告：

- `Vivid_Acceptance/indicator/usp_ablation/P1_vs_B0_compare.json`
- `Vivid_Acceptance/indicator/usp_ablation/P2_vs_B0_compare.json`
- `Vivid_Acceptance/indicator/usp_ablation/P3_vs_B0_compare.json`

## 下一步与接受条件

正式验收必须完全走 `docs_xzh/run_command/mock_test.md` 的 Moto S3、callback receiver、固定 caption sidecar mock、四卡服务、外部 FlowCut POST、进度轮询、callback、S3 上传与下载路径。先 warmup，再在同一服务实例上提交 formal。

P3 只有同时满足以下条件才通过第一阶段验收：

- `model_inference_runtime_seconds < 175.0`，基线为 `194.2424s`。
- 固定 July 8 reference 的 `SSIM mean >= 0.98` 且 `SSIM min >= 0.98`。
- CFG world size 2、SP world size 2、有效 backend `fa_sp`。
- transformer/controlnet 的两个 USP effective 开关均为 `true`。
- callback 终态 succeeded，Moto S3 对象存在并可下载。

任一门禁失败时，本轮只记录实验结论，不修改正式默认命令和 handover 基线；两个开关继续默认关闭。
