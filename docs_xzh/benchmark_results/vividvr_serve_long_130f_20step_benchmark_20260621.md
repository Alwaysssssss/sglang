# VividVR Serve 长视频 Benchmark 与消融验收

日期：`2026-06-21 UTC`

基准口径：`130f / 20 step / serve + curl / warmup excluded`

## 1. 测试目的

本轮目标不是再做单一“双组对比”，而是对当前已集成到 `sglang` 的 `Vivid-VR` 加速模块做一次正式消融验收，明确回答下面几个问题：

- 单卡与双卡在 `serve` 路径下的真实正式耗时分别是多少。
- `FA`、`torch.compile`、`SP=2` 各自能带来多少速度收益。
- 哪些组合虽然更快，但已经偏离单卡质量基线。
- 哪个双卡组合作为默认配置最合适。

本轮正式结论只保留 `dual eager_global + pool=1` 口径，不再纳入 `deferred_global`。原因很直接：当前只有 `eager_global` 的双卡路径能稳定保持与单卡基本一致的质量。

## 2. 固定测试条件

### 2.1 固定输入

- 输入视频：`/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4`
- caption 文件：`/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt`
- prompt 文件：`/home/zhiheng/Vivid-VR/input/720p/prompt.txt`
- reference 视频：`/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4`

### 2.2 固定请求参数

- `num_inference_steps=20`
- `seed=42`
- `num_temporal_process_frames=121`
- 请求方式固定为 `serve` 启动服务后，通过 `curl` 调用 `/v1/videos/repairs`

### 2.3 执行纪律

- 每个配置都先跑 1 次 `warmup`，再跑第 2 次正式请求。
- 正式 benchmark 只统计第 2 次请求，`warmup` 时间不计入正式结果。
- 单卡正式验收严格串行执行，同一时刻只允许 1 个推理进程，避免双进程抢占导致单卡数据失真。
- 本文采用的单卡数据全部来自串行重跑结果。更早那轮并行单卡尝试已经废弃，不计入任何正式结论。
- 双卡全部固定 `SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global` 与 `SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1`。

### 2.4 术语说明

- 文中 `sdpa` 是表格简写；实际服务启动参数使用的是 `--attention-backend torch_sdpa`。
- 文中双卡 `fa` 组表示“请求 backend 为 `fa`”；在 `SP=2` 下运行时实际走的是 `fa_sp` 路径。

## 3. 环境信息

| 项目 | 值 |
| --- | --- |
| 仓库 | `/home/zhiheng/sglang` |
| git commit | `9cc2729d4` |
| Python | `3.10.12` |
| OS | `Linux 5.14.0-284.25.1.el9_2.x86_64 x86_64 GNU/Linux` |
| GPU | `NVIDIA A100-SXM4-80GB` x2 |
| driver | `550.90.07` |
| torch | `2.9.1+cu128` |
| diffusers | `0.37.0` |
| transformers | `5.3.0` |
| numpy | `2.2.6` |
| opencv-python | `4.10.0` |

## 4. 正式配置矩阵

| 标识 | GPU | attention backend | context mode | compile | control pool | 备注 |
| --- | ---: | --- | --- | --- | ---: | --- |
| `single_gpu_sdpa_no_compile` | 1 | `torch_sdpa` | `N/A` | off | 1 | 单卡串行正式值 |
| `single_gpu_fa_no_compile` | 1 | `fa` | `N/A` | off | 1 | 单卡串行正式值 |
| `single_gpu_sdpa_compile` | 1 | `torch_sdpa` | `N/A` | on | 1 | 单卡串行正式值 |
| `single_gpu_fa_compile` | 1 | `fa` | `N/A` | on | 1 | 单卡串行正式值 |
| `dual_gpu_sdpa_eager_no_compile` | 2 | `torch_sdpa` | `eager_global` | off | 1 | 质量较单卡明显漂移 |
| `dual_gpu_fa_eager_no_compile` | 2 | `fa -> fa_sp` | `eager_global` | off | 1 | 质量与单卡基本一致 |
| `dual_gpu_sdpa_eager_compile` | 2 | `torch_sdpa` | `eager_global` | on | 1 | 最快，但质量漂移最大 |
| `dual_gpu_fa_eager_compile` | 2 | `fa -> fa_sp` | `eager_global` | on | 1 | 推荐默认双卡配置 |

## 5. 正式结果总表

下表均为排除 `warmup` 后的正式请求结果。

| 标识 | 配置 | 总耗时（s） | 模型推理（s） | warmup 推理（s） | 峰值显存（MB） | SSIM mean | SSIM min | MSE mean | MAE mean | Pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `single_gpu_sdpa_no_compile` | single / sdpa / no_compile | 1081.51 | 1075.43 | 1083.33 | 27741.82 | 0.987546 | 0.981780 | 2.927101 | 1.004737 | `true` |
| `single_gpu_fa_no_compile` | single / fa / no_compile | 1021.30 | 1020.00 | 1022.01 | 27743.05 | 0.987548 | 0.983061 | 2.864804 | 1.007994 | `true` |
| `single_gpu_sdpa_compile` | single / sdpa / compile | 1006.39 | 991.02 | 1147.35 | 27742.12 | 0.987092 | 0.983171 | 2.991987 | 1.027670 | `true` |
| `single_gpu_fa_compile` | single / fa / compile | 961.31 | 950.01 | 1015.55 | 27742.06 | 0.987081 | 0.981336 | 3.016236 | 1.023840 | `true` |
| `dual_gpu_sdpa_eager_no_compile` | dual eager / sdpa / no_compile | 540.75 | 527.12 | 532.35 | 27742.67 | 0.967059 | 0.917686 | 19.903753 | 1.987248 | `true` |
| `dual_gpu_fa_eager_no_compile` | dual eager / fa / no_compile | 615.95 | 604.49 | 611.53 | 27742.28 | 0.987517 | 0.982830 | 2.918570 | 1.012828 | `true` |
| `dual_gpu_sdpa_eager_compile` | dual eager / sdpa / compile | 495.72 | 483.37 | 574.35 | 27742.67 | 0.966271 | 0.916820 | 20.259482 | 2.010831 | `true` |
| `dual_gpu_fa_eager_compile` | dual eager / fa / compile | 540.79 | 538.43 | 538.69 | 27744.68 | 0.987086 | 0.982025 | 3.077819 | 1.023633 | `true` |

## 6. 主结论

- 8 组配置全部通过当前 compare gate，但 compare gate 本身不足以区分“只是能过线”和“与单卡质量基本一致”。
- 单卡最快配置是 `single_gpu_fa_compile`，正式总耗时 `961.31s`。
- 双卡最快配置是 `dual_gpu_sdpa_eager_compile`，正式总耗时 `495.72s`，但 `ssim_mean=0.966271`，相对单卡 `sdpa+compile` 下降约 `0.020821`，不适合作为默认质量安全配置。
- 真正兼顾速度与质量的双卡配置是 `dual_gpu_fa_eager_compile`。它与 `single_gpu_fa_compile` 的 `ssim_mean` 只差 `+0.000004`，可以视为同一质量水平，同时端到端加速达到 `1.7776x`。
- 如果只看质量安全路径，双卡 `FA` 是必要条件；如果只看绝对速度，双卡 `sdpa` 更快，但质量会明显掉出“与单卡基本一致”的范围。

## 7. FA 消融

这里的“FA 增益”按 `sdpa -> fa` 定义；数值大于 `1` 表示切到 `FA` 后更快，小于 `1` 表示切到 `FA` 后更慢。

| 场景 | 对比 | 端到端速度比 | 推理速度比 | SSIM 变化 |
| --- | --- | ---: | ---: | ---: |
| single no_compile | `single_gpu_sdpa_no_compile` -> `single_gpu_fa_no_compile` | 1.0590x | 1.0543x | +0.000001 |
| single compile | `single_gpu_sdpa_compile` -> `single_gpu_fa_compile` | 1.0469x | 1.0432x | -0.000010 |
| dual eager no_compile | `dual_gpu_sdpa_eager_no_compile` -> `dual_gpu_fa_eager_no_compile` | 0.8779x | 0.8720x | +0.020458 |
| dual eager compile | `dual_gpu_sdpa_eager_compile` -> `dual_gpu_fa_eager_compile` | 0.9167x | 0.8977x | +0.020814 |

解读：

- 单卡上，`FA` 明确优于 `sdpa`，速度提升约 `1.05x`，质量基本不变。
- 双卡 `eager_global` 上，`FA` 比 `sdpa` 更慢，但它把 `ssim_mean` 从约 `0.966` 拉回到约 `0.987`，这是默认双卡配置必须接受的代价。
- 因此双卡里 `FA` 不是“更激进的加速项”，而是“质量安全前提”。

## 8. Compile 消融

这里的“compile 增益”按 `no_compile -> compile` 定义。

| 场景 | 对比 | 端到端加速比 | 推理加速比 | SSIM 变化 |
| --- | --- | ---: | ---: | ---: |
| single sdpa | `single_gpu_sdpa_no_compile` -> `single_gpu_sdpa_compile` | 1.0746x | 1.0852x | -0.000454 |
| single fa | `single_gpu_fa_no_compile` -> `single_gpu_fa_compile` | 1.0624x | 1.0737x | -0.000466 |
| dual eager sdpa | `dual_gpu_sdpa_eager_no_compile` -> `dual_gpu_sdpa_eager_compile` | 1.0908x | 1.0905x | -0.000788 |
| dual eager fa | `dual_gpu_fa_eager_no_compile` -> `dual_gpu_fa_eager_compile` | 1.1390x | 1.1227x | -0.000432 |

解读：

- `torch.compile` 在 4 条主线上都有效，收益大致在 `1.06x` 到 `1.14x` 之间。
- 对默认双卡质量安全路径 `dual_gpu_fa_eager_*` 而言，`compile` 带来的是这轮最扎实的额外收益：端到端 `1.1390x`，纯推理 `1.1227x`。
- 编译路径的首请求开销不能忽略，因此 benchmark 必须坚持“先 warmup，再记正式值”。

## 9. 单卡 vs 双卡加速

这里按“相同 backend、相同 compile 状态”的单卡与双卡配对做对比。

| 场景 | 对比 | 端到端加速比 | 推理加速比 | SSIM 变化 |
| --- | --- | ---: | ---: | ---: |
| sdpa no_compile | `single_gpu_sdpa_no_compile` -> `dual_gpu_sdpa_eager_no_compile` | 2.0000x | 2.0402x | -0.020487 |
| fa no_compile | `single_gpu_fa_no_compile` -> `dual_gpu_fa_eager_no_compile` | 1.6581x | 1.6874x | -0.000030 |
| sdpa compile | `single_gpu_sdpa_compile` -> `dual_gpu_sdpa_eager_compile` | 2.0302x | 2.0503x | -0.020821 |
| fa compile | `single_gpu_fa_compile` -> `dual_gpu_fa_eager_compile` | 1.7776x | 1.7644x | +0.000004 |

解读：

- 如果只追求更高 speedup，双卡 `sdpa` 能做到约 `2.0x`，但质量下降过大，不适合作为正式默认模式。
- 如果要求双卡输出与单卡基本一致，`fa` 是正确主线。此时 `dual_gpu_fa_eager_compile` 能稳定提供约 `1.78x` 端到端加速。
- 从阶段耗时看，主要收益集中在 denoising。以推荐默认对比为例：
  - `single_gpu_fa_compile` denoise：`771.79s`
  - `dual_gpu_fa_eager_compile` denoise：`380.69s`
  - denoise 段加速约 `2.03x`
  - 端到端最终落在 `1.78x`，原因是 prep / decode / postprocess 并没有同比例缩短

## 10. 推荐默认配置

### 10.1 双卡默认

推荐默认双卡配置为 `dual_gpu_fa_eager_compile`：

- `SP=2`
- `SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global`
- `SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1`
- `attention backend = fa`，运行时实际进入 `fa_sp`
- `torch.compile = on`

推荐理由：

- 与单卡 `single_gpu_fa_compile` 的 `ssim_mean` 基本完全一致：`0.987081 -> 0.987086`
- 正式端到端耗时从 `961.31s` 降到 `540.79s`
- 正式模型推理耗时从 `950.01s` 降到 `538.43s`
- 是当前双卡 `eager_global` 路径里最快的质量安全配置

### 10.2 单卡默认

如果只能用单卡，推荐 `single_gpu_fa_compile`：

- 是本轮 4 组单卡里正式耗时最低的组合
- 相比 `single_gpu_sdpa_no_compile` 端到端加速 `1.1240x`
- 质量仍稳定在 `ssim_mean ~= 0.98708`

### 10.3 不推荐默认的配置

- `dual_gpu_sdpa_eager_no_compile`
- `dual_gpu_sdpa_eager_compile`

这两组虽然更快，但 `ssim_mean` 都跌到 `0.966x`，与单卡基线不再属于“基本一致”的质量水平，因此不应作为默认推理模式。

## 11. 产物与脚本

正式指标目录：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator`

结果视频目录：

- `/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark`

本轮使用的辅助脚本：

- `/home/zhiheng/sglang/Vivid_Acceptance/tmp/run_vividvr_service_ablation.sh`
- `/home/zhiheng/sglang/Vivid_Acceptance/tmp/collect_vividvr_benchmark_metrics.py`

8 组正式 JSON：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr-service-benchmark-long-130f-20step-single_gpu_sdpa_no_compile-20260621T113553Z.json`
- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr-service-benchmark-long-130f-20step-single_gpu_fa_no_compile-20260621T121313Z.json`
- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr-service-benchmark-long-130f-20step-single_gpu_sdpa_compile-20260621T124833Z.json`
- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr-service-benchmark-long-130f-20step-single_gpu_fa_compile-20260621T132544Z.json`
- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr-service-benchmark-long-130f-20step-dual_gpu_sdpa_eager_no_compile-20260621T095026Z.json`
- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr-service-benchmark-long-130f-20step-dual_gpu_fa_eager_no_compile-20260621T100947Z.json`
- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr-service-benchmark-long-130f-20step-dual_gpu_sdpa_eager_compile-20260621T110928Z.json`
- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr-service-benchmark-long-130f-20step-dual_gpu_fa_eager_compile-20260621T085249Z.json`

## 12. 最终结论

当前 `Vivid-VR` 在 `sglang serve` 下的正式性能结论可以收敛为：

- 单卡最佳正式配置：`single_gpu_fa_compile`
- 双卡最佳质量安全配置：`dual_gpu_fa_eager_compile`
- 双卡质量安全端到端加速：`1.7776x`
- 双卡质量安全纯推理加速：`1.7644x`
- 双卡最快但不推荐默认的激进配置：`dual_gpu_sdpa_eager_compile`

如果后续只允许保留一条双卡默认路径，应直接固定为 `FA + eager_global + pool=1 + compile`，而不是继续使用 `sdpa` 路径追求更高但不稳定的表观速度。
