# VividVR USP collective 消融记录（2026-07-10）

## 结论摘要

- packed QKV A2A 在 CFG2 x SP2、`fa_sp`、`eager_global`、`torch.compile` 下正确生效：rank 0 的 3 个 profiler timestep 中，A2A 启动数从 `1152` 降到 `576`，下降 `50%`；对应 GPU annotation 总时长从 `505.611 ms` 降到 `404.369 ms`，下降 `20.02%`。
- prefix `all_gather_into_tensor` 开关没有减少当前 compile 正式路径的 collective：B0 和 P2 均为 `300` 次 gather，GPU annotation 总时长分别为 `46.734 ms` 和 `47.088 ms`。原因是旧的 tensor-list `dist.all_gather` 已被 Dynamo functionalize 为 coalesced tensor gather。
- P1/P2/P3 相对 B0 的 130 帧 smoke 视频均通过 `SSIM >= 0.98` 门槛；P3 的 `SSIM mean=0.991444`、`min=0.988110`，无失败帧。
- packed 优化降低了 launch count，但 5-step steady median 没有可测收益：B0 为 `9487.627 ms/step`，P1 为 `9523.407 ms/step`。profiler 中三步 A2A 总量只减少约 `101.242 ms`，不足以支持仅凭 smoke 推断正式耗时能从基线 `194.2424s` 降至 `175s`。
- 正式服务验收中，P3 去噪耗时为 `194.128929s`，相对 `194.2424s` 基线只减少 `0.113471s`（`0.0584%`），未达到 `<175s` 的性能门槛；质量仍通过，`SSIM mean=0.984879`、`min=0.980405`。
- 第一阶段结论是“collective launch 优化与数值正确性成立，但端到端性能不晋级”。两个开关继续保持默认关闭，不修改正式默认命令和 handover 基线。

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

## 正式服务验收

正式验收严格使用 `docs_xzh/run_command/mock_test.md` 的 Moto S3、callback receiver、固定 caption sidecar mock、四卡 `sglang serve` 和外部 FlowCut POST 链路。服务配置为 `CFG=2 x SP=2`、`fa_sp`、`eager_global`、`torch.compile`，并在 transformer/controlnet 上同时开启 packed QKV A2A 和 prefix `all_gather_into_tensor`。

- warmup task：`vividvr-usp-p3-warmup-20260710T052818Z`。
- formal task：`vividvr-usp-p3-formal-20260710T053636Z`。
- 服务日志：`Vivid_Acceptance/logs/vividvr_usp_p3_formal_service_20260710T052556Z.log`。
- callback 日志：`Vivid_Acceptance/logs/mock_callback_20260710T052514Z.jsonl`。
- formal perf：`Vivid_Acceptance/indicator/service_benchmark/vividvr-usp-p3-formal-20260710T053636Z_perf.json`。
- formal compare：`Vivid_Acceptance/indicator/service_benchmark/vividvr-usp-p3-formal-20260710T053636Z_compare.json`。
- acceptance summary：`Vivid_Acceptance/indicator/service_benchmark/vividvr-usp-p3-formal-20260710T053636Z_acceptance_summary.json`。
- S3 下载视频：`Vivid_Acceptance/result_videos/service_benchmark/downloads/vividvr-usp-p3-formal-20260710T053636Z.bridge-downloaded.mp4`。
- caption/manifest：`Vivid_Acceptance/captions/service_sidecars/vividvr-usp-p3-formal-20260710T053636Z.{txt,manifest.json}`。

warmup 用于完成 compile/cache，不计入性能。formal 在同一服务实例上执行，结果如下：

| 指标 | 结果 | 门槛 | 结论 |
|---|---:|---:|---|
| model inference / denoising | `194.128929s` | `<175.0s` | FAIL |
| 相对 `194.2424s` 基线 | `-0.113471s`（`-0.0584%`） | - | 基本持平 |
| total runtime | `353.864116s` | 记录项 | - |
| long preparation | `59.857291s` | 记录项 | - |
| decode | `98.368664s` | 记录项 | - |
| SSIM mean | `0.984879` | `>=0.98` | PASS |
| SSIM min | `0.980405` | `>=0.98` | PASS |
| 帧数 / 失败帧 | `130 / 0` | `130 / 0` | PASS |

运行时 debug 同时确认：

- `vividvr_parallel_mode=cfg_sp`、CFG world size 2、SP world size 2。
- transformer/controlnet effective backend 均为 `fa_sp`。
- transformer/controlnet 的 `packed_qkv_a2a` 和 `prefix_all_gather_into_tensor` 均为 `true`。
- 输出 130 帧，`prompt_embed_shape=[1,226,4096]`，未破坏既有 VividVR 语义。
- callback 终态为 `succeeded`；Moto S3 对象 `bridge-semantic-check/vividvr-usp-p3-formal-20260710T053636Z.mp4` 存在，大小 `5,751,035` bytes，并通过 boto3 下载后完成 SSIM。
- 请求临时目录在完成后已清理。

当前 Moto mock 对象 ACL 只有 owner `FULL_CONTROL`，因此返回的未签名裸 `result_url` 直接 `curl` 为 `403`；`mock_test.md` 提供的 boto3 认证下载路径通过。该现象与 USP collective 优化无关，但正式服务契约后续若要求裸 URL 可直接下载，需要单独修正 mock ACL 或改为 presigned URL。

## 验收结论与后续边界

P3 已证明 collective count、数值质量、并行拓扑和服务链路正确，但正式去噪耗时未低于 `175s`，因此第一阶段不通过性能验收。本轮不修改 `docs_xzh/run_command/vividvr_default_run_and_serve_commands.md` 和 handover 的正式基线，两个实验开关继续默认关闭。

profiler 显示 packed QKV A2A 仍有约 `20%` 的 A2A annotation duration 收益，但它在完整去噪中的占比不足。若继续优化，应另写独立计划评估按 head bucket 的异步 A2A/attention overlap，并重新定义 buffer 生命周期、stream/event 同步、compile graph break、bucket 消融和数值门槛；不得直接在当前默认路径上启用。
