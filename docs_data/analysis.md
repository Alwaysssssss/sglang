# VideoEdit 加速实验统计

更新时间：2026-07-17。

本文只记录 VideoEdit 模型的推理性能、显存、Stage 耗时和后续加速实验结果。已删除原先旧模型相关表格；尚未做过实验的 VideoEdit 方案保留定义和空结果，后续复测时直接补表。

## 1. 填写与计时约定

| 项目 | 约定 |
| --- | --- |
| 耗时单位 | 秒，保留至少 2 位小数；原始 JSON 中保留完整精度 |
| 显存单位 | GiB；主表填写最大单卡峰值显存，括号内可记录总峰值 |
| 总耗时 | 以 benchmark 脚本 `elapsed_s` 为准 |
| 模型推理耗时 | 以服务端 `perf/*.json` 的 `total_duration_ms / 1000` 为准 |
| Denoise 耗时 | 对应 `VideoEditDenoisingStage` |
| Decode 耗时 | 对应 `VideoEditDecodingStage` |
| Stage 耗时 | 以服务端 `perf/*.json` 的 `steps` 为准；若 record/summary 与 perf 不一致，优先使用 perf |
| 显存统计 | 以 benchmark record 中 `nvidia-smi` 采样结果为准；`peak_run_gpu_memory.max_gpu_memory_used_mb` 转 GiB |
| 正式计时 | 同一服务实例和同一配置先 warmup，再记录正式请求；没有 warmup 产物的结果必须在表中标注 |
| 质量口径 | 使用同一输入、同一 seed 和输出视频做主观/指标检查；未做质量检查时留空，不写 PASS |
| N/A | 当前方案不涉及该指标，或者该指标与基线非同口径；不得用估算值代替 |

### 1.1 加速与资源效率计算

同一步数和同一 guidance 口径下，累计加速比统一以 VE0 计算：

```text
累计加速比 = VE0 模型推理耗时 / 当前方案模型推理耗时
```

模块增量收益按方案定义中的对照计算：

```text
模块增量加速比 = 指定对照方案模型推理耗时 / 当前方案模型推理耗时
```

多卡方案同时记录资源成本：

```text
GPU·秒 = GPU 数量 × 当前方案模型推理耗时

相对 VE0 资源效率 = VE0 GPU·秒 / 当前方案 GPU·秒
```

4-step / guidance=1.0 与 40-step / guidance=5.0 不属于同一步数口径；可以记录耗时和显存，但累计加速比只作为非同口径参考，不能直接作为同质量加速结论。

## 2. 固定实验口径

| 项目 | 固定值 |
| --- | --- |
| 输入视频 | `/home/tyx/workspace/1080/1080.mp4` |
| Mask | `/home/tyx/workspace/1080/mask_1080_merged.mp4` |
| Reference image | `/home/tyx/workspace/1080/local.png` |
| Prompt | `一个男人在舞台演讲，背后有两排文字。` |
| 输出帧配置 | `num_frames=80`，`infer_len=81`，`overlap=5` |
| 默认推理参数 | `num_inference_steps=40`，`guidance_scale=5.0`，`seed=42`，`dtype=bf16` |
| 低步数补测参数 | `num_inference_steps=4`，`guidance_scale=1.0`，`dynamic_cfg=false` |
| 编辑区域参数 | `bbox_expand_scale=1.6`，`bbox_padding=0`，`dilate_px=0`，`mask_scale=1.0`，`feather_px=0`，`adain_boundary_dilate=0` |
| Callback | `http://127.0.0.1:18080/videoedit/callback`；callback 失败不影响本地结果和 perf dump |
| 主要输出根目录 | `/home/tyx/workspace/zhouhao6/outputs` |

除被测加速模块、并行拓扑、步数和 guidance 外，所有方案必须保持输入、mask、reference、seed、dtype、VAE、后处理和请求语义一致。

## 3. 正式实验方案定义

| 编号 | 关键方案 | 增益对照 | 主要统计目标 | 当前状态 |
| --- | --- | --- | --- | --- |
| VE0 | 单卡 native，40-step，guidance=5.0 | — | VideoEdit 原始基线 | 已完成 |
| VE1 | VE0 + TeaCache | VE0 | 单卡 TeaCache 收益、显存风险和质量影响 | 已完成 |
| VE2 | VE0 + `torch.compile` | VE0 | 单卡 compile 收益 | 未完成 |
| VE3 | VE0 + `torch.compile` + TeaCache | VE2、VE1 | 单卡 compile 与 TeaCache 组合收益 | 未完成 |
| VE4 | 双卡 SP=2 native，40-step，guidance=5.0 | VE0 | 双卡 SP 延迟收益和 GPU·秒成本 | 已完成 |
| VE5 | VE4 + TeaCache | VE4 | 双卡 TeaCache 收益和显存占用 | 已完成 |
| VE6 | VE4 + `torch.compile` | VE4 | 双卡 compile 收益 | 已完成 |
| VE7 | VE4 + `torch.compile` + TeaCache | VE6、VE5 | 双卡 compile 与 TeaCache 组合收益 | 未完成 |
| VE8 | 双卡 SP=2 native，4-step，guidance=1.0，no TeaCache | VE4 仅作非同口径参考 | DMD/低步数模型延迟和显存 | 已完成 |
| VE9 | 双卡 SP=2，4-step，guidance=1.0 + TeaCache | VE8 | 4-step 下 TeaCache 是否仍有收益 | 未纳入本轮数据 |

### 3.1 比较关系

TeaCache 单卡收益：

```text
VE0 -> VE1
```

双卡 SP 收益：

```text
VE0 -> VE4
```

双卡 TeaCache 收益：

```text
VE4 -> VE5
```

双卡 `torch.compile` 收益：

```text
VE4 -> VE6
```

4-step 方案：

```text
VE8 只和同为 4-step / guidance=1.0 的方案比较；VE9 暂不纳入本轮正式表。
```

## 4. 总体结果

| 方案 | 请求配置 | 总耗时 | 模型推理耗时 | Denoise 耗时 | Decode 耗时 | 相对 VE0 加速比 | 模块增量加速比 | GPU·秒 | 相对 VE0 资源效率 | 最大单卡峰值显存 | 质量结果 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| VE0 | 1 GPU；40-step；gs=5.0；no TeaCache；no compile | 2047.58 | 2042.64 | 1956.34 | 21.98 | 1.0000× | — | 2042.64 | 1.0000× | 50.73 |  |
| VE1 | 1 GPU；40-step；gs=5.0；TeaCache；no compile | 770.98 | 762.83 | 676.07 | 22.08 | 2.6777× | VE0 / VE1 = 2.6777× | 762.83 | 2.6777× | 79.29 |  |
| VE2 | 1 GPU；40-step；gs=5.0；no TeaCache；compile |  |  |  |  |  | VE0 / VE2 |  |  |  |  |
| VE3 | 1 GPU；40-step；gs=5.0；TeaCache；compile |  |  |  |  |  | VE2 / VE3 |  |  |  |  |
| VE4 | 2 GPU SP=2；40-step；gs=5.0；no TeaCache；no compile | 1076.38 | 1084.28 | 1011.42 | 15.24 | 1.8839× | VE0 / VE4 = 1.8839× | 2168.55 | 0.9419× | 32.93（总 65.52） |  |
| VE5 | 2 GPU SP=2；40-step；gs=5.0；TeaCache；no compile | 450.64 | 439.56 | 351.84 | 16.54 | 4.6470× | VE4 / VE5 = 2.4667× | 879.13 | 2.3235× | 33.32（总 66.62） |  |
| VE6 | 2 GPU SP=2；40-step；gs=5.0；no TeaCache；compile | 1092.68 | 1088.07 | 1003.06 | 15.57 | 1.8773× | VE4 / VE6 = 0.9965× | 2176.15 | 0.9386× | 33.32（总 66.37） |  |
| VE7 | 2 GPU SP=2；40-step；gs=5.0；TeaCache；compile |  |  |  |  |  | VE6 / VE7 |  |  |  |  |
| VE8 | 2 GPU SP=2；4-step；gs=1.0；no TeaCache；no compile | 135.47 | 129.84 | 73.56 | 15.47 | 15.7315×（非同口径） | N/A | 259.69 | 7.8657×（非同口径） | 32.69（总 65.33） |  |
| VE9 | 2 GPU SP=2；4-step；gs=1.0；TeaCache |  |  |  |  |  | VE8 / VE9 |  |  |  |  |

说明：

- VE1 峰值显存为 79.29 GiB，接近 80 GiB 卡容量，后续复测需要重点观察稳定性。
- VE5 的空闲显存包含上一条双卡 native 请求结束后的残留占用，因此显存比较优先看峰值显存。
- VE6 是只开 `torch.compile`、不叠 TeaCache 的结果；目前没有 completed 的 VE7 数据。
- `bench_1080_dual_gpu_retest` 目录下的两条双卡请求是 failed 状态，没有进入正式表。
- VE4 的 record/summary 与 `perf/` JSON 有一次不一致；本表模型推理耗时和 Stage 明细按 `perf/` JSON 填，脚本总耗时和显存按 record 填。
- VE8 已替换为 2026-07-17 01:31:55 的 `lh_38` 新模型 4-step no TeaCache 成功请求；2026-07-16 的旧 4-step 数据不再作为正式表数据。
- `lh_38` 目录下同时存在一条 4-step TeaCache completed 产物，但 VE9 暂不纳入本轮正式表。

## 5. Stage 耗时明细

单位：秒。

| Stage | VE0 | VE1 | VE2 | VE3 | VE4 | VE5 | VE6 | VE7 | VE8 | VE9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `VideoEditWindowValidationStage` | 0.000 | 0.000 |  |  | 0.000 | 0.000 | 0.000 |  | 0.000 |  |
| `VideoEditTextEncodingStage` | 1.077 | 1.095 |  |  | 1.492 | 0.991 | 0.982 |  | 1.261 |  |
| `VideoEditImageEncodingStage` | 1.890 | 1.621 |  |  | 1.009 | 0.926 | 0.828 |  | 0.898 |  |
| `VideoEditConditionEncodingStage` | 26.782 | 27.448 |  |  | 20.313 | 46.348 | 28.787 |  | 18.139 |  |
| `VideoEditLatentPreparationStage` | 0.071 | 0.068 |  |  | 0.067 | 0.029 | 0.043 |  | 0.028 |  |
| `VideoEditTimestepPreparationStage` | 0.001 | 0.001 |  |  | 0.000 | 0.000 | 0.000 |  | 0.000 |  |
| `VideoEditLatentInitStage` | 0.000 | 0.000 |  |  | 0.000 | 0.000 | 0.000 |  | 0.000 |  |
| `VideoEditDenoisingStage` | 1956.337 | 676.066 |  |  | 1011.421 | 351.845 | 1003.056 |  | 73.556 |  |
| `VideoEditDecodingStage` | 21.978 | 22.078 |  |  | 15.238 | 16.541 | 15.570 |  | 15.467 |  |
| `VideoEditWindowPostprocessStage` | 0.000 | 0.000 |  |  | 0.000 | 0.000 | 0.000 |  | 0.000 |  |
| 未归类开销 | 34.50 | 34.46 |  |  | 34.74 | 22.88 | 38.81 |  | 20.49 |  |
| 模型推理总计 | 2042.64 | 762.83 |  |  | 1084.28 | 439.56 | 1088.07 |  | 129.84 |  |

未归类开销按下式计算：

```text
未归类开销 = 模型推理耗时 - 所有已记录 Stage 耗时之和
```

## 6. Denoising 核心耗时明细

| 方案 | Window 数 | 推理步数 | Denoise 总耗时 | 占模型推理比例 | 平均每 step | Steady step 中位数 | SP 通信耗时 | Cache 执行/跳过 | 最大单卡峰值显存 | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| VE0 | 1 | 40 | 1956.34 | 95.78% | 48.908 |  | N/A | N/A | 50.73 |  |
| VE1 | 1 | 40 | 676.07 | 88.63% | 16.902 |  | N/A |  | 79.29 | TeaCache 开启，执行/跳过明细当前未记录 |
| VE2 |  | 40 |  |  |  |  | N/A | N/A |  |  |
| VE3 |  | 40 |  |  |  |  | N/A |  |  |  |
| VE4 | 1 | 40 | 1011.42 | 93.28% | 25.286 |  |  | N/A | 32.93 | SP=2 |
| VE5 | 1 | 40 | 351.84 | 80.04% | 8.796 |  |  |  | 33.32 | SP=2，TeaCache 开启 |
| VE6 | 1 | 40 | 1003.06 | 92.19% | 25.076 |  |  | N/A | 33.32 | SP=2，compile 开启 |
| VE7 |  | 40 |  |  |  |  |  |  |  |  |
| VE8 | 1 | 4 | 73.56 | 56.65% | 18.389 |  |  | N/A | 32.69 | 4-step / gs=1.0，非 40-step 同口径；`lh_38` 新模型 |
| VE9 |  | 4 |  |  |  |  |  |  |  |  |

填写说明：

- `平均每 step` 使用 Denoise 总耗时除以请求步数，仅用于总体核对。
- `Steady step 中位数` 需要有逐 step profiler 后再填；当前 perf dump 只有 stage 总耗时。
- SP 通信耗时和 Cache 执行/跳过明细当前无法可靠拆分时留空，不用整段 Denoise 耗时估算。

## 7. 实验环境与运行快照

| 项目 | 值 |
| --- | --- |
| 机器型号 |  |
| GPU 型号与可用数量 |  |
| CUDA 版本 |  |
| Driver 版本 |  |
| PyTorch 版本 |  |
| sglang commit | `7c8e62eaa930a7e458734533981df2e17813945c`（来自 perf dump） |
| Python 路径 |  |
| 基础模型路径 | `/home/tyx/workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model` |
| Transformer 路径 |  |
| Dtype | `bf16` |
| 计时方式 | benchmark 脚本 `elapsed_s` + 服务端 `perf_dump_path` |
| 显存统计方式 | benchmark 脚本采样 `nvidia-smi` |
| Stage profiling 是否同步 |  |
| 结果根目录 | `/home/tyx/workspace/zhouhao6/outputs` |

### 7.1 方案运行时快照

| 方案 | Requested config | Effective config | Compile 生效 | 并行拓扑 | Cache 配置 | Warmup/Formal 产物 |
| --- | --- | --- | --- | --- | --- | --- |
| VE0 | no TeaCache；no compile |  | 否 | 1 GPU | 无 | `/home/tyx/workspace/zhouhao6/outputs/bench_1080_single_gpu_native` |
| VE1 | TeaCache；no compile |  | 否 | 1 GPU | TeaCache | `/home/tyx/workspace/zhouhao6/outputs/bench_1080_single_gpu_teacache` |
| VE2 | no TeaCache；compile |  |  | 1 GPU | 无 |  |
| VE3 | TeaCache；compile |  |  | 1 GPU | TeaCache |  |
| VE4 | no TeaCache；no compile |  | 否 | 2 GPU, SP=2 | 无 | `/home/tyx/workspace/zhouhao6/outputs/bench_1080_current_bbox` |
| VE5 | TeaCache；no compile |  | 否 | 2 GPU, SP=2 | TeaCache | `/home/tyx/workspace/zhouhao6/outputs/bench_1080_current_bbox` |
| VE6 | no TeaCache；compile |  | 是 | 2 GPU, SP=2 | 无 | `/home/tyx/workspace/zhouhao6/outputs/bench_1080_dual_gpu_torchcompile_native` |
| VE7 | TeaCache；compile |  |  | 2 GPU, SP=2 | TeaCache |  |
| VE8 | no TeaCache；no compile；4-step / gs=1.0 |  | 否 | 2 GPU, SP=2 | 无 | `/home/tyx/workspace/zhouhao6/outputs/lh_38_1080_dual_gpu_4step_gs1` |
| VE9 | TeaCache；4-step / gs=1.0 |  |  | 2 GPU, SP=2 | TeaCache |  |

## 8. 模块收益结论

| 加速模块 | Treatment | Control | 延迟增量加速比 | GPU·秒变化 | 显存变化 | 质量变化 | 正式结论 |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 单卡 TeaCache | VE1 | VE0 | 2.6777× | 2042.64 -> 762.83 | 50.73 -> 79.29 |  | 延迟收益明显，但显存接近 80 GiB，需要补质量检查和稳定性复测 |
| 双卡 SP | VE4 | VE0 | 1.8839× | 2042.64 -> 2168.55 | 50.73 -> 32.93（总 65.52） |  | 单请求延迟下降，但 GPU·秒略升 |
| 双卡 TeaCache | VE5 | VE4 | 2.4667× | 2168.55 -> 879.13 | 32.93 -> 33.32 |  | 双卡下 TeaCache 端到端收益明显 |
| 双卡 `torch.compile` | VE6 | VE4 | 0.9965× | 2168.55 -> 2176.15 | 32.93 -> 33.32 |  | 当前数据未形成端到端收益 |
| 单卡 `torch.compile` | VE2 | VE0 |  |  |  |  |  |
| 单卡 `torch.compile` + TeaCache | VE3 | VE2/VE1 |  |  |  |  |  |
| 双卡 `torch.compile` + TeaCache | VE7 | VE6/VE5 |  |  |  |  |  |
| 4-step / guidance=1.0 | VE8 | VE4 | N/A | N/A | 32.93 -> 32.69 |  | 非同一步数口径，只记录低步数模型运行结果 |
| 4-step TeaCache | VE9 | VE8 |  |  |  |  |  |

“正式结论”需要基于同一输入、有效 runtime 配置、服务端 perf dump、显存记录和质量检查填写。局部 Stage 或 kernel 改善但端到端模型推理耗时没有改善时，应明确写为“局部优化生效，但未形成端到端收益”。

## 9. 原始产物索引

| 方案 | Summary | Stage summary | Perf dump | Record |
| --- | --- | --- | --- | --- |
| VE0 | `/home/tyx/workspace/zhouhao6/outputs/bench_1080_single_gpu_native/summary.csv` | `/home/tyx/workspace/zhouhao6/outputs/bench_1080_single_gpu_native/stage_summary.csv` | `/home/tyx/workspace/zhouhao6/outputs/bench_1080_single_gpu_native/perf/videoedit_bench_single_gpu_native_1080_current_bbox_20260716_070005_single_gpu_native.json` | `/home/tyx/workspace/zhouhao6/outputs/bench_1080_single_gpu_native/records/videoedit_bench_single_gpu_native_1080_current_bbox_20260716_070005_single_gpu_native.json` |
| VE1 | `/home/tyx/workspace/zhouhao6/outputs/bench_1080_single_gpu_teacache/summary.csv` | `/home/tyx/workspace/zhouhao6/outputs/bench_1080_single_gpu_teacache/stage_summary.csv` | `/home/tyx/workspace/zhouhao6/outputs/bench_1080_single_gpu_teacache/perf/videoedit_bench_single_gpu_1080_current_bbox_20260716_074935_single_gpu_teacache.json` | `/home/tyx/workspace/zhouhao6/outputs/bench_1080_single_gpu_teacache/records/videoedit_bench_single_gpu_1080_current_bbox_20260716_074935_single_gpu_teacache.json` |
| VE4 | `/home/tyx/workspace/zhouhao6/outputs/bench_1080_current_bbox/summary.csv` | `/home/tyx/workspace/zhouhao6/outputs/bench_1080_current_bbox/stage_summary.csv` | `/home/tyx/workspace/zhouhao6/outputs/bench_1080_current_bbox/perf/videoedit_bench_1080_current_bbox_20260716_090006_dual_gpu.json` | `/home/tyx/workspace/zhouhao6/outputs/bench_1080_current_bbox/records/videoedit_bench_1080_current_bbox_20260716_090006_dual_gpu.json` |
| VE5 | `/home/tyx/workspace/zhouhao6/outputs/bench_1080_current_bbox/summary.csv` | `/home/tyx/workspace/zhouhao6/outputs/bench_1080_current_bbox/stage_summary.csv` | `/home/tyx/workspace/zhouhao6/outputs/bench_1080_current_bbox/perf/videoedit_bench_1080_current_bbox_20260716_090006_dual_gpu_teacache.json` | `/home/tyx/workspace/zhouhao6/outputs/bench_1080_current_bbox/records/videoedit_bench_1080_current_bbox_20260716_090006_dual_gpu_teacache.json` |
| VE6 | `/home/tyx/workspace/zhouhao6/outputs/bench_1080_dual_gpu_torchcompile_native/torchcompile_native_summary.csv` | `/home/tyx/workspace/zhouhao6/outputs/bench_1080_dual_gpu_torchcompile_native/torchcompile_native_stage_summary.csv` | `/home/tyx/workspace/zhouhao6/outputs/bench_1080_dual_gpu_torchcompile_native/perf/videoedit_bench_torchcompile_native_1080_current_bbox_20260716_062119_dual_gpu_torchcompile.json` | `/home/tyx/workspace/zhouhao6/outputs/bench_1080_dual_gpu_torchcompile_native/records/videoedit_bench_torchcompile_native_1080_current_bbox_20260716_062119_dual_gpu_torchcompile.json` |
| VE8 | `/home/tyx/workspace/zhouhao6/outputs/lh_38_1080_dual_gpu_4step_gs1/summary_4step_gs1.csv` | `/home/tyx/workspace/zhouhao6/outputs/lh_38_1080_dual_gpu_4step_gs1/stage_summary_4step_gs1.csv` | `/home/tyx/workspace/zhouhao6/outputs/lh_38_1080_dual_gpu_4step_gs1/perf/videoedit_bench_4step_gs1_1080_current_bbox_20260717_013155_dual_gpu_4step_gs1.json` | `/home/tyx/workspace/zhouhao6/outputs/lh_38_1080_dual_gpu_4step_gs1/records/videoedit_bench_4step_gs1_1080_current_bbox_20260717_013155_dual_gpu_4step_gs1.json` |
