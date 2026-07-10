# VividVR USP collective 优化实验交接

日期：2026-07-10

## 1. 交接范围

本文只交接本轮 VividVR USP collective 通信优化实验，不重复说明上一轮 CFG parallel v2 的实现过程，也不改变已经验收的 Phase C / D / E 默认配置与服务契约。

本轮实验建立在四卡 `CFG=2 x SP=2` 稳定基线上，目标是在保持输出质量和服务语义不变的前提下，评估以下两项通信优化能否把正式 130 帧、20 step 去噪耗时从基线 `194.2424s` 降到 `<175s`：

- packed QKV All-to-All：把每层 Q、K、V 三次输入 A2A 合并为一次。
- prefix `all_gather_into_tensor`：用 tensor gather 路径替换 prefix tensor-list gather。

## 2. 最终结论

本轮已经完成代码实现、单元与双 rank 分布式测试覆盖、profiler 修复、B0/P1/P2/P3 四组消融，以及完整 FlowCut mock 服务正式验收。

结论是：

1. packed QKV A2A 的通信优化真实生效。
   - rank 0 的 3 个 profiler timestep 中，A2A 启动数从 `1152` 降到 `576`，下降 `50%`。
   - A2A GPU annotation 总时长从 `505.611ms` 降到 `404.369ms`，下降 `20.02%`。
2. prefix `all_gather_into_tensor` 在当前 `torch.compile` 正式路径中没有可测收益。
   - B0 和 P2 都是 `300` 次 gather。
   - GPU annotation 总时长分别为 `46.734ms` 和 `47.088ms`。
   - 原因是旧的 tensor-list `dist.all_gather` 已经被 Dynamo functionalize 为 coalesced tensor gather。
3. 数值质量和服务链路均通过验收。
   - 正式 P3 输出 `SSIM mean=0.984879`、`SSIM min=0.980405`。
   - 比较 130 帧，无失败帧。
   - CFG/SP 拓扑、`fa_sp` backend、226 长度 prompt embedding、callback、S3 上传下载和请求清理均正常。
4. 端到端性能没有晋级。
   - 正式 P3 去噪耗时为 `194.128929s`。
   - 相对 `194.2424s` 基线只减少 `0.113471s`，即 `0.0584%`。
   - 没有达到 `<175s` 的性能门槛。

因此，本轮结论固定为：**collective launch 优化与数值正确性成立，但端到端性能不晋级。** 两个实验开关继续默认关闭，不加入正式默认命令，不修改既有 handover 性能基线。

## 3. 已完成实现

### 3.1 实验开关

`python/sglang/multimodal_gen/runtime/server_args.py` 和 `python/sglang/multimodal_gen/tools/run_vividvr_inference.py` 已增加：

```bash
--enable-usp-packed-qkv-a2a
--enable-usp-prefix-all-gather-into-tensor
```

两个开关默认值都是 `False`。未显式传入时，现有单卡、双卡 SP 和四卡 CFG/SP 路径的 collective 行为不变。

### 3.2 packed QKV A2A

`python/sglang/multimodal_gen/runtime/layers/usp.py` 新增 `_usp_input_all_to_all_qkv` primitive。

原路径每个 USP attention processor 包含三次输入 A2A 和一次输出 A2A，共四次 collective。packed 路径先把 Q、K、V 打包，只执行一次输入 A2A，再恢复三个 tensor，因此每层只剩一次输入 A2A 和一次输出 A2A，共两次 collective。

`python/sglang/multimodal_gen/runtime/layers/attention/layer.py` 根据实验开关选择原始路径或 packed 路径。默认关闭时仍走原始三次输入 A2A。

### 3.3 prefix gather 路径

`python/sglang/multimodal_gen/runtime/layers/usp.py` 新增 prefix tensor gather primitive；`python/sglang/multimodal_gen/runtime/layers/attention/layer.py` 根据开关选择该路径。

该实现和测试仍保留，方便后续在其他 compile/backend 组合下复用或继续研究，但本轮结果证明它不能改善当前正式 `torch.compile` 路径，因此不得仅凭实现存在就默认启用。

### 3.4 CogVideoX 与 VividVR 配置传播

`python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py` 已把两个选项传播到 CogVideoX attention processor。

`python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py` 已同时配置 transformer 和 controlnet，并把 requested/effective 状态写入 `vividvr_debug`。正式报告可以检查：

- `usp_packed_qkv_a2a_requested`
- `usp_prefix_all_gather_into_tensor_requested`
- `usp_transformer.packed_qkv_a2a`
- `usp_transformer.prefix_all_gather_into_tensor`
- `usp_controlnet.packed_qkv_a2a`
- `usp_controlnet.prefix_all_gather_into_tensor`

### 3.5 profiler 修复

首次消融暴露出 profiler step hook 没有随 denoising timestep 推进的问题，导致 trace 没有捕获有效去噪 step。

`dbf6dc42c` 修复后：

- 单 clip 在一个全局 denoising timestep 完成后推进一次 profiler。
- multi clip 也是所有 temporal clips 完成同一个全局 timestep 后才推进一次。
- 不会按 clip 重复推进 step。

首次无效 trace 和产物没有删除，统一保存在带 `usp_ablation_invalid_profiler_hook_20260710` 后缀的目录中。后续分析不得把这些无效产物混入正式统计。

## 4. 测试覆盖

本轮新增或扩展了以下测试：

- `python/sglang/multimodal_gen/test/unit/test_usp_packed_collectives.py`
  - packed QKV 的形状、布局和路径选择。
- `python/sglang/multimodal_gen/test/unit/test_usp_packed_collectives_distributed.py`
  - 双 rank NCCL 下 packed 与原路径 bitwise-exact 对比。
- `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py`
  - CogVideoX transformer/controlnet 配置传播和 effective 状态。
- `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py`
  - direct runner 参数、ServerArgs 传播与报告字段。
- `python/sglang/multimodal_gen/test/unit/test_diffusion_profiler.py`
  - profiler step 推进语义。
- `python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py`
  - multi-clip 全局 timestep 下 profiler 只推进一次。

除自动化测试外，B0/P1/P2/P3 四组都真实完成四卡 compile、5 step 推理、decode、报告写入和逐帧质量比较；P3 还完成了 20 step 正式服务闭环。

## 5. 消融口径

固定环境与输入：

- GPU：GPU0-3，`NVIDIA A100-SXM4-80GB`。
- 并行：4 processes，CFG world size 2，SP/Ulysses world size 2，ring 1。
- attention：请求 `fa`，transformer/controlnet 有效 backend 都是 `fa_sp`。
- connector：`SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global`。
- compile：transformer/controlnet 均启用 `torch.compile`。
- 输入：`/home/zhiheng/input/test_video_long_960x720_130f.mp4`。
- caption：`Vivid_Acceptance/captions/service_sidecars/quad-test-video-long-960x720-130f-run2-20260708T060202Z.txt`，共 2 行。
- 推理：130 帧、5 steps、seed 42、temporal process frames 121、upscale 1.0。
- profiler：1 个 wait/warmup 阶段后捕获 3 个全局 denoising timesteps。

四组开关矩阵：

| 配置 | packed QKV A2A | prefix `all_gather_into_tensor` |
|---|---|---|
| B0 | 关闭 | 关闭 |
| P1 | 开启 | 关闭 |
| P2 | 关闭 | 开启 |
| P3 | 开启 | 开启 |

## 6. Profiler 与 smoke 结果

### 6.1 Collective 结果

只统计 trace 中 `cat=gpu_user_annotation` 的 NCCL annotation，避免重复计算 CPU wrapper 和 CUDA kernel。

| 配置 | A2A count | A2A duration | 相对 B0 | gather count | gather duration |
|---|---:|---:|---:|---:|---:|
| B0 | 1152 | 505.611ms | - | 300 | 46.734ms |
| P1 | 576 | 404.369ms | count -50.00%，duration -20.02% | 300 | 46.579ms |
| P2 | 1152 | 505.876ms | duration +0.05% | 300 | 47.088ms |
| P3 | 576 | 408.650ms | count -50.00%，duration -19.18% | 300 | 46.354ms |

A2A count 与模型结构一致：`48 processors x 2 clips x 3 profiled steps x 4 collectives = 1152`；packed 路径把每层四次 collective 降为两次，因此为 `576`。

### 6.2 5-step 时间

| 配置 | model inference | long preparation | denoising stage | decode | steps 2-5 median |
|---|---:|---:|---:|---:|---:|
| B0 | 331.844s | 65.979s | 129.252s | 106.997s | 9.488s/step |
| P1 | 333.632s | 64.979s | 135.306s | 105.823s | 9.523s/step |
| P2 | 327.432s | 65.015s | 127.578s | 105.711s | 9.532s/step |
| P3 | 323.723s | 64.777s | 125.935s | 105.328s | 9.534s/step |

5-step 的 `model_inference_runtime_seconds` 包含 compile、profiler flush、decode 和 postprocess，不能直接当作正式性能结论。排除首步 compile 后，P1/P3 的 steady median 也没有显示逐 step 加速。

### 6.3 Smoke 数值质量

以同轮 B0 输出为 reference，比较 130 帧：

| 配置 | SSIM mean | SSIM min | MSE mean | 失败帧 | 结论 |
|---|---:|---:|---:|---:|---|
| P1 | 0.991475 | 0.988179 | 3.133780 | 0 | PASS |
| P2 | 0.991500 | 0.988220 | 3.159122 | 0 | PASS |
| P3 | 0.991444 | 0.988110 | 3.152107 | 0 | PASS |

三组实验路径都满足 `SSIM >=0.98`，说明 packed、prefix 和组合路径没有引入不可接受的画质偏差。

## 7. 正式服务验收

正式验收严格使用 `docs_xzh/run_command/mock_test.md` 的完整链路：

- Moto S3
- callback receiver
- 固定 caption sidecar mock
- 四卡 `sglang serve`
- 外部 FlowCut POST
- 进度轮询
- callback 终态
- S3 上传与认证下载
- 下载结果和固定 reference 的逐帧 SSIM

服务配置为 `CFG=2 x SP=2`、`fa_sp`、`eager_global`、`torch.compile`，transformer/controlnet 同时开启两个实验开关。先在同一个服务实例上执行完整 warmup，再执行 formal；warmup 不计入正式耗时。

任务：

- warmup：`vividvr-usp-p3-warmup-20260710T052818Z`
- formal：`vividvr-usp-p3-formal-20260710T053636Z`

正式结果：

| 指标 | 结果 | 门槛 | 结论 |
|---|---:|---:|---|
| model inference / denoising | 194.128929s | `<175.0s` | FAIL |
| 相对 194.2424s 基线 | -0.113471s（-0.0584%） | 记录项 | 基本持平 |
| total runtime | 353.864116s | 记录项 | - |
| long preparation | 59.857291s | 记录项 | - |
| decode | 98.368664s | 记录项 | - |
| SSIM mean | 0.984879 | `>=0.98` | PASS |
| SSIM min | 0.980405 | `>=0.98` | PASS |
| 帧数 / 失败帧 | 130 / 0 | 130 / 0 | PASS |

运行时报告同时确认：

- `vividvr_parallel_mode=cfg_sp`
- CFG world size 为 2，SP world size 为 2
- transformer/controlnet effective backend 都是 `fa_sp`
- transformer/controlnet 的两个 USP effective 开关都是 `true`
- 输出为 130 帧
- `prompt_embed_shape=[1,226,4096]`
- callback 终态为 `succeeded`
- S3 对象存在，大小为 `5,751,035` bytes
- boto3 认证下载和后续 SSIM 比较成功
- 请求临时目录在完成后已清理

## 8. 验收判定与默认配置决策

P3 通过了以下门禁：

- collective 实现真实生效
- 双 rank 数值等价测试
- 130 帧 smoke 质量
- 130 帧正式输出质量
- CFG/SP 并行拓扑
- `fa_sp` backend
- transformer/controlnet 配置传播
- callback、S3 和输入清理服务链路

P3 没有通过唯一的正式性能门禁：

- `model_inference_runtime_seconds <175.0`

所以必须继续保持以下决策：

1. `--enable-usp-packed-qkv-a2a` 默认关闭。
2. `--enable-usp-prefix-all-gather-into-tensor` 默认关闭。
3. 不把两个参数加入 `docs_xzh/run_command/vividvr_default_run_and_serve_commands.md` 的正式默认命令。
4. 不修改四卡 CFG/SP 的 `194.2424s` 正式基线。
5. 不因为 A2A 局部指标改善就宣称端到端提速。
6. 不影响既有 `single_gpu_fa_compile`、`dual_gpu_fa_eager_compile` 和 `dual_gpu_sdpa_eager_compile` 三条已验收配置。

## 9. 关键产物

### 9.1 计划与总结

- 计划：`docs_xzh/distribute/2026-07-10-vividvr-usp-packed-qkv-communication-optimization-plan.md`
- 完整消融记录：`docs_xzh/distribute/vividvr_usp_collective_ablation_20260710.md`

后续若需要引用具体数据，应优先以完整消融记录和 JSON 原始产物为准，本文用于说明交接边界和最终决策。

### 9.2 Smoke 日志、trace 和质量报告

- 日志：`Vivid_Acceptance/logs/usp_ablation/{B0,P1,P2,P3}.log`
- trace：`Vivid_Acceptance/profiles/usp_ablation/{B0,P1,P2,P3}/profile_trace-3_steps-global-rank0.trace.json.gz`
- P1 质量：`Vivid_Acceptance/indicator/usp_ablation/P1_vs_B0_compare.json`
- P2 质量：`Vivid_Acceptance/indicator/usp_ablation/P2_vs_B0_compare.json`
- P3 质量：`Vivid_Acceptance/indicator/usp_ablation/P3_vs_B0_compare.json`

无效 profiler hook 产物：

- `Vivid_Acceptance/logs/usp_ablation_invalid_profiler_hook_20260710`
- `Vivid_Acceptance/profiles/usp_ablation_invalid_profiler_hook_20260710`
- `Vivid_Acceptance/result_videos/usp_ablation_invalid_profiler_hook_20260710`
- `Vivid_Acceptance/indicator/usp_ablation_invalid_profiler_hook_20260710`

### 9.3 正式服务产物

- 服务日志：`Vivid_Acceptance/logs/vividvr_usp_p3_formal_service_20260710T052556Z.log`
- callback 日志：`Vivid_Acceptance/logs/mock_callback_20260710T052514Z.jsonl`
- perf：`Vivid_Acceptance/indicator/service_benchmark/vividvr-usp-p3-formal-20260710T053636Z_perf.json`
- compare：`Vivid_Acceptance/indicator/service_benchmark/vividvr-usp-p3-formal-20260710T053636Z_compare.json`
- acceptance summary：`Vivid_Acceptance/indicator/service_benchmark/vividvr-usp-p3-formal-20260710T053636Z_acceptance_summary.json`
- 下载视频：`Vivid_Acceptance/result_videos/service_benchmark/downloads/vividvr-usp-p3-formal-20260710T053636Z.bridge-downloaded.mp4`
- caption：`Vivid_Acceptance/captions/service_sidecars/vividvr-usp-p3-formal-20260710T053636Z.txt`
- caption manifest：`Vivid_Acceptance/captions/service_sidecars/vividvr-usp-p3-formal-20260710T053636Z.manifest.json`

## 10. 关键提交

本轮实现和验收提交范围为 `af8c66f62^..78faa0510`：

- `af8c66f62`：新增 USP collective 优化计划。
- `9d14482b6`：暴露两个默认关闭的优化开关。
- `2f7d10d77`：加入 packed QKV A2A primitive。
- `433cfd2d9`：USP attention processor 选择优化路径。
- `75a61848d`：向 CogVideoX transformer/controlnet 传播开关。
- `df7ba8530`：接入 VividVR runtime 配置和 effective 状态报告。
- `be1d743d9`：加入双 rank NCCL bitwise-exact 测试。
- `79b65ed34`：direct runner 暴露 profiler 参数。
- `dd4807e02`：按 mock 服务链调整正式验收口径。
- `b73f457d8`：修正规划文档中的 checkpoint 路径。
- `dbf6dc42c`：修复 profiler 按全局 denoising timestep 推进。
- `608d1ff39`：完成并记录 USP collective 四组 profiling。
- `78faa0510`：记录正式服务实验结论和默认配置决策。

## 11. 已知边界

### 11.1 Moto mock 裸 URL 返回 403

当前 Moto mock 对象 ACL 只有 owner `FULL_CONTROL`。服务返回的未签名裸 `result_url` 直接用 `curl` 下载会得到 `403`，但 `mock_test.md` 规定的 boto3 认证下载路径正常，且本轮正式视频已经通过认证下载和 SSIM 验收。

该问题与 USP collective 优化无关。本轮没有修改对象存储服务契约。如果后续明确要求裸 URL 可以匿名直接下载，应单独评估：

- 调整 mock 对象 ACL；或
- 返回 presigned URL。

不要把该问题与本轮性能不晋级混为同一个根因。

### 11.2 Smoke 总耗时不能代表正式收益

5-step smoke 的总耗时包含 compile 和 profiler flush，不可用于宣称性能提升。正式性能判断只能使用同一服务实例完整 warmup 后的 20-step formal 结果。

### 11.3 实现保留不等于正式启用

两个实验路径和测试会继续保留在代码中，但默认关闭。后续维护者不得因为代码存在或局部 profiler 数据变好，就绕过正式门禁将其加入默认配置。

## 12. 后续建议

packed QKV A2A 已经把 A2A annotation duration 降低约 `20%`，但 A2A 在完整去噪中的占比不足，继续只压缩 launch count 的收益上限很低。

若还要继续这个方向，建议单独立项评估按 head bucket 的异步 A2A/attention overlap，并在动代码前重新定义：

- bucket 划分与 tensor layout
- buffer 生命周期
- CUDA stream/event 同步
- `torch.compile` graph break 风险
- transformer/controlnet 一致性
- bucket 大小消融
- 双 rank 数值门槛
- 130 帧正式质量门槛
- 完整 warmup + formal 服务性能门槛

在新的方案完成独立计划和验收前，不得直接在当前正式默认路径上启用本轮开关。

## 13. 接手时的推荐阅读顺序

下一位维护者可以按以下顺序建立上下文：

1. 本交接文档：先确认最终决策和禁止事项。
2. `docs_xzh/distribute/vividvr_usp_collective_ablation_20260710.md`：查看完整数据、实验口径和产物位置。
3. `docs_xzh/distribute/2026-07-10-vividvr-usp-packed-qkv-communication-optimization-plan.md`：查看原始任务拆解和设计约束。
4. `Vivid_Acceptance/indicator/service_benchmark/vividvr-usp-p3-formal-20260710T053636Z_acceptance_summary.json`：核对正式数字。
5. `Vivid_Acceptance/indicator/service_benchmark/vividvr-usp-p3-formal-20260710T053636Z_perf.json`：核对每个 pipeline stage 和 runtime debug 字段。

接手后的默认动作不是继续打开两个开关，而是守住当前默认关闭状态；只有在提出新的端到端优化设计并获得明确授权后，才进入下一轮实验。
