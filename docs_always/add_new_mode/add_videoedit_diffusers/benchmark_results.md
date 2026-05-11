# VideoEdit 优化 Benchmark 结果总结

本文档汇总 `sglang/outputs/videoedit_optimizer_bench_summary.json` 中本轮实际运行结果。所有候选输出均与固定 reference 视频对比：

```text
/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/reference/15108907_3840_2160_50fps.mp4
```

## 1. 测试口径

固定输入与采样参数：

- 视频：`15108907_3840_2160_50fps_short.mp4`
- Mask：`15108907_3840_2160_50fps_No_bbox_mask.mp4`
- Prompt：`A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video.`
- `num_frames=81`
- `infer_len=81`
- `overlap=0`
- `num_inference_steps=20`
- `guidance_scale=5.0`
- `dynamic_cfg=true`
- `dynamic_cfg_max_step=15`
- `seed=42`
- `dtype=bf16`
- `enable_paste_back=true`
- `drop_reference_frame=true`

统计口径：

- CLI `wall_seconds` 包含模型加载、warmup、compile、worker 启停和推理。
- CLI `forward_s` 来自 `--perf-dump-path`，只统计请求 forward 侧。
- Serve 每个阶段跑两次请求，第一次为 warmup request，第二次为有效统计。
- Serve `wall_seconds` 是第二次请求从提交到完成的时间，`inference_time_s` / `forward_s` 为服务返回或 perf dump 中的推理耗时。
- `peak_allocated_mb` 来自 perf dump 的 `after_forward.peak_allocated_mb`。
- 质量对比阈值：`min_ssim=0.90`、`max_mse=150.0`、`max_mae=8.0`、`max_failed_frame_ratio=0.05`。

## 2. 总体结论

最佳稳定吞吐方案是 `sp2_no_offload_compile_fa_cache_fast`：

- Serve 第二次请求 `forward=112.53s`，相对 `sp1_offload` serve 基线 `311.47s` 加速 `2.77x`。
- 相对无 cache 的 `sp2_no_offload_fa` serve 基线 `177.88s` 加速 `1.58x`。
- 逐帧 compare 通过，`SSIM mean/min = 0.9865/0.9842`。
- 峰值显存约 `44.7GB`。

如果不希望使用 Cache-DiT，推荐 `sp2_no_offload_compile_fa` 或 `sp2_no_offload_fa`：

- `sp2_no_offload_fa` serve `forward=177.88s`，相对 `sp1_offload` 加速 `1.75x`。
- `sp2_no_offload_compile_fa` serve `forward=176.17s`，仅比不 compile 快约 `1%`，收益很小。
- Torch compile 在 serve 模式的第二次请求可以稳定运行，但首个 warmup request 很慢；CLI 冷启动 wall time 不适合作为收益判断。

算子对比结论：

- `fa` 是当前 A100 上最优的非 cache attention backend。
- `torch_sdpa` 比 `fa` 慢约 `3%`。
- `sage_attn` 已能作为实际后端运行，但本轮慢于 `fa`，serve `forward=208.38s`。
- `sage_attn_3` 未实际启用，CLI 中回退到 `torch_sdpa`，serve 中 `sp2_no_offload_sage3` 请求超时失败；A100 不应把 `sage_attn_3` 作为有效成绩。

并行对比结论：

- SP2 Ulysses + FA 最优：serve `forward=177.88s`。
- TP2 + FA 更省显存，峰值约 `34.3GB`，但 serve `forward=194.29s`，比 SP2 FA 慢约 `9.2%`。
- Ring SP2 CLI 可以跑通，但 serve 启动失败，需要单独排查服务启动和端口/分布式初始化日志。

质量结论：

- 所有成功完成的候选输出均通过 reference 逐帧 compare。
- Cache-DiT fast 的 SSIM 最低，但仍明显高于阈值：`SSIM mean/min = 0.9865/0.9842`。
- 无 cache 方案的 SSIM 基本稳定在 `0.9871-0.9872`。

## 3. Serve 有效结果

Serve 结果以下表的第二次请求为准，未列 warmup request。

| 阶段 | actual backend | forward(s) | denoise(s) | VAE encode(s) | decode(s) | wall(s) | peak MB | speedup vs SP1 offload | compare |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `sp1_offload` | fa | 311.47 | 283.02 | 9.34 | 7.96 | 315.22 | 14224 | 1.00x | pass |
| `sp1_no_offload` | fa | 310.01 | 282.79 | 9.37 | 7.95 | 315.31 | 55185 | 1.00x | pass |
| `sp1_no_offload_compile` | fa | 308.99 | 281.81 | 9.23 | 8.12 | 315.31 | 55169 | 1.01x | pass |
| `sp1_no_offload_compile_sage3` | torch_sdpa fallback | 321.10 | 293.65 | 9.78 | 8.05 | 325.27 | 55169 | 0.97x | pass |
| `sp2_no_offload_torch_sdpa` | torch_sdpa | 183.14 | 156.95 | 8.43 | 7.56 | 190.12 | 44041 | 1.70x | pass |
| `sp2_no_offload_fa` | fa | 177.88 | 151.51 | 8.79 | 7.27 | 180.12 | 44041 | 1.75x | pass |
| `sp2_no_offload_sage_attn` | sage_attn | 208.38 | 181.50 | 8.97 | 7.44 | 215.18 | 44041 | 1.49x | pass |
| `tp2_no_offload_fa` | fa | 194.29 | 161.45 | 12.53 | 10.22 | 200.17 | 34299 | 1.60x | pass |
| `sp2_no_offload_compile_fa` | fa | 176.17 | 148.66 | 9.58 | 7.59 | 180.22 | 44032 | 1.77x | pass |
| `sp2_no_offload_compile_torch_sdpa` | torch_sdpa | 181.91 | 154.84 | 9.43 | 7.39 | 185.21 | 44032 | 1.71x | pass |
| `sp2_no_offload_compile_fa_teacache` | fa | 175.32 | 148.73 | 9.21 | 7.22 | 180.20 | 44198 | 1.78x | pass |
| `sp2_no_offload_compile_fa_cache_rdt010` | fa | 143.11 | 116.42 | 8.98 | 7.41 | 145.13 | 44713 | 2.18x | pass |
| `sp2_no_offload_compile_fa_cache_rdt012` | fa | 142.30 | 116.37 | 8.59 | 7.16 | 145.20 | 44713 | 2.19x | pass |
| `sp2_no_offload_compile_fa_cache_rdt018` | fa | 138.78 | 112.39 | 8.87 | 7.32 | 145.16 | 44712 | 2.24x | pass |
| `sp2_no_offload_compile_fa_cache_fast` | fa | 112.53 | 86.55 | 8.53 | 7.31 | 115.09 | 44713 | 2.77x | pass |
| `offload_branch` | fa | 311.45 | 282.92 | 9.41 | 8.12 | 315.28 | 14224 | 1.00x | pass |

## 4. CLI 结果

CLI 更适合验证命令能否跑通，不适合评估 compile 的最终收益，因为每次都会重新拉起进程。

| 阶段 | actual backend | forward(s) | denoise(s) | wall(s) | peak MB | compare |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `sp1_offload` | default | 310.22 | 282.62 | 595.24 | 14224 | pass |
| `sp1_no_offload` | fa | 309.13 | 282.55 | 538.23 | 55185 | pass |
| `sp1_no_offload_compile` | fa | 307.68 | 281.36 | 542.46 | 55168 | pass |
| `sp1_no_offload_compile_sage3` | torch_sdpa fallback | 320.19 | 293.51 | 556.62 | 55168 | pass |
| `sp2_no_offload_torch_sdpa` | torch_sdpa | 183.22 | 156.68 | 434.16 | 44040 | pass |
| `sp2_no_offload_fa` | fa | 177.51 | 150.93 | 426.00 | 44040 | pass |
| `sp2_no_offload_sage_attn` | sage_attn | 207.89 | 181.54 | 460.77 | 44040 | pass |
| `sp2_no_offload_sage3` | torch_sdpa fallback | 183.02 | 156.93 | 435.15 | 44040 | pass |
| `sp2_ring_no_offload_fa` | fa | 182.79 | 157.22 | 434.87 | 44040 | pass |
| `tp2_no_offload_fa` | fa | 192.23 | 161.03 | 459.93 | 34299 | pass |
| `sp2_no_offload_compile_fa` | fa | 174.37 | 148.53 | 924.94 | 44030 | pass |
| `sp2_no_offload_compile_torch_sdpa` | torch_sdpa | 179.91 | 154.44 | 938.44 | 44030 | pass |
| `sp2_no_offload_compile_fa_teacache` | fa | 174.56 | 148.53 | 921.37 | 44198 | pass |
| `sp2_no_offload_compile_fa_cache_rdt010` | fa | 141.19 | 116.25 | 904.84 | 44712 | pass |
| `sp2_no_offload_compile_fa_cache_rdt012` | fa | 142.52 | 116.61 | 889.01 | 44712 | pass |
| `sp2_no_offload_compile_fa_cache_rdt018` | fa | 138.44 | 112.54 | 883.02 | 44712 | pass |
| `sp2_no_offload_compile_fa_cache_fast` | fa | 113.47 | 86.72 | 888.76 | 44713 | pass |
| `offload_branch` | fa | 310.26 | 282.44 | 599.01 | 14224 | pass |

## 5. 质量对比

所有成功完成的方案都通过了逐帧 compare。关键方案的质量指标如下：

| 阶段 | SSIM mean | SSIM min | MSE mean | MAE mean | max abs diff | pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sp1_offload` | 0.9872 | 0.9854 | 3.55 | 1.17 | 52 | true |
| `sp2_no_offload_fa` | 0.9872 | 0.9854 | 3.63 | 1.18 | 68 | true |
| `sp2_no_offload_compile_fa` | 0.9872 | 0.9853 | 3.61 | 1.18 | 65 | true |
| `sp2_no_offload_compile_fa_teacache` | 0.9872 | 0.9853 | 3.61 | 1.18 | 65 | true |
| `sp2_no_offload_compile_fa_cache_rdt010` | 0.9870 | 0.9852 | 3.80 | 1.21 | 62 | true |
| `sp2_no_offload_compile_fa_cache_rdt012` | 0.9870 | 0.9852 | 3.80 | 1.21 | 62 | true |
| `sp2_no_offload_compile_fa_cache_rdt018` | 0.9870 | 0.9851 | 3.80 | 1.21 | 63 | true |
| `sp2_no_offload_compile_fa_cache_fast` | 0.9865 | 0.9842 | 4.03 | 1.25 | 72 | true |

Cache-DiT fast 质量指标最低，但仍远高于当前宽松质量阈值。若用于默认生产配置，建议额外做人工抽检，重点看 mask 边缘闪烁、纹理漂移和窗口边界。

## 6. 失败与异常项

本轮失败项：

- `serve sp2_no_offload_sage3`：两次请求均失败，错误为 `Scheduler did not respond in time`。CLI 中同名方案实际回退到 `torch_sdpa` 并成功，因此该 serve 失败不代表 SageAttention3 真实性能。
- `serve sp2_ring_no_offload_fa`：服务未通过 health check。CLI 版本可跑通，说明 Ring SP 本身不一定不可用，但 serve 启动路径仍需单独排查。

需要注意的 fallback：

- `sp1_no_offload_compile_sage3` 和 `sp2_no_offload_sage3` 的 CLI 结果实际是 `torch_sdpa` fallback，不应记录为 SageAttention3 成绩。
- `sage_attn` 已实际启用，但本轮性能慢于 `fa`。

## 7. 推荐配置

### 高质量默认配置

推荐使用 `sp2_no_offload_fa`：

```text
num_gpus=2
sp_degree=2
ulysses_degree=2
ring_degree=1
attention_backend=fa
offload=false
num_inference_steps=20
dynamic_cfg_max_step=15
```

理由：

- serve `forward=177.88s`，相对 `sp1_offload` 加速 `1.75x`。
- compare 通过，质量和其他无 cache 方案基本一致。
- 没有 Cache-DiT 的质量 tradeoff。

### 吞吐优先配置

推荐使用 `sp2_no_offload_compile_fa_cache_fast`：

```text
num_gpus=2
sp_degree=2
ulysses_degree=2
ring_degree=1
attention_backend=fa
enable_torch_compile=true
Cache-DiT: FN=1, BN=0, WARMUP=2, RDT=0.24, MC=3, SCM_PRESET=fast
```

理由：

- serve `forward=112.53s`，本轮最快。
- 相对 `sp2_no_offload_fa` 加速 `1.58x`。
- compare 通过。

风险：

- Cache-DiT 属于允许质量 tradeoff 的优化项，虽然本轮自动指标通过，但需要人工抽检。
- `SCM_PRESET=fast` 比 RDT 0.10/0.12/0.18 更激进。

### 显存受限配置

推荐 `sp1_offload` 或 `offload_branch`：

- 峰值显存约 `14.2GB`。
- serve `forward≈311s`，速度明显慢于 SP2。
- 适合只要求稳定跑通、GPU 显存紧张的场景。

### 不推荐作为主线

- `sage_attn_3`：A100 上未启用，CLI fallback 到 Torch SDPA，serve 超时失败。
- `sage_attn`：本轮可用但慢于 FA。
- `tp2_no_offload_fa`：显存更低但速度慢于 SP2 FA。
- CLI compile wall time：冷启动包含编译成本，不适合作为最终性能判断。

## 8. 后续建议

1. 若要将 Cache-DiT fast 设为默认吞吐配置，建议补充更多视频/mask 类型：小 mask、大 mask、快速运动、复杂纹理。
2. 对 `sp2_no_offload_compile_fa_cache_fast` 做人工质量检查，重点检查 mask 边缘、身份一致性、纹理稳定性和窗口边界。
3. 排查 `serve sp2_ring_no_offload_fa` health check 失败原因，确认是端口/进程清理问题，还是 Ring SP serve 路径问题。
4. 不再优先投入 `sage_attn_3` 于 A100；如果要继续测 SageAttention，优先围绕 `sage_attn` 与 FA 做同环境对比。
5. 如果显存目标低于 44GB，可继续评估 TP2 + FA 或量化分支。
