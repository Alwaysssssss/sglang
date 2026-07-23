# VideoEdit L20 W8A8 与低精度 Attention 实验总结

> 更新时间：2026-07-22  
> 目标：在 NVIDIA L20 上加速 VideoEdit 的 DiT 降噪阶段，并验证实际执行的是低精度计算，而不是仅以 FP8 格式保存权重。

## 1. 结论

当前最优配置为：

```text
Linear:
  Weight: FP8 E4M3FN，per-output-channel，FP32 scale
  Activation: 动态 FP8 E4M3FN，per-token，对称 max-abs scale
  GEMM: Triton row-wise scaled_mm
  QKV/KV: fused projections

Attention:
  Self-attention: SageAttention qk_int8_pv_fp8_cuda
  Text cross-attention: FlashAttention BF16
  Image cross-attention: FlashAttention BF16

Parallel/offload:
  2 x L20
  SP=2, Ulysses=2, Ring=1
  dit_cpu_offload=false
  dit_layerwise_offload=true
```

在当前主对比场景 `profile81`（81 帧推理窗口、4 个 denoise steps）中：

| 指标 | BF16 baseline | 最终 FP8+Sage | 加速比 | 耗时下降 |
|---|---:|---:|---:|---:|
| DiT 降噪中位数 | 298.457 s | 178.895 s | **1.668x** | 40.06% |
| 请求总耗时中位数 | 348.060 s | 228.223 s | **1.525x** | 34.43% |
| Self-attention GPU 时间/被采样 timestep | 50.375 s | 28.261 s | **1.782x** | 43.90% |
| 全部 Attention GPU 时间/被采样 timestep | 50.986 s | 28.872 s | **1.766x** | 43.37% |
| 全部 GPU kernel 时间/被采样 timestep | 68.118 s | 45.992 s | **1.481x** | 32.48% |

因此，当前结论不是“L20 没有 FP8 加速能力”，而是：

1. L20 的 FP8 Tensor Core 能在真实大矩阵上达到约 `1.7x-2.0x`。
2. 原始 SGL CUTLASS row-wise FP8 kernel 只达到约 `1.2x`，是最初收益低的主要原因之一。
3. 切换 Triton dynamic W8A8 后，三个主导 Linear shape 达到约 `1.69x-1.80x`。
4. Self-attention 使用 Sage 8-bit kernel 后达到 `1.782x`。
5. scale、动态激活量化、低精度 Attention 量化、SP 通信、Norm、残差和其他 kernel 都包含在最终耗时中，因此整个 DiT 没有达到理论 `2x` 是正常的。

最终 DiT 降噪达到 **1.668x**，已经接近项目希望得到的 `1.7x` 实际加速区间。

## 2. 实验环境与固定输入

### 2.1 硬件和软件

| 项目 | 配置 |
|---|---|
| GPU | 2 x NVIDIA L20 |
| Compute Capability | 8.9（SM89） |
| 单卡显存 | 45,457.6 MiB |
| SM 数量 | 92/卡 |
| PyTorch | 2.11.0+cu130 |
| CUDA | 13.0 |
| SageAttention | 2.2.0 |
| 模型 | `VideoEdit-diffusers-model` |
| Transformer | `/mnt/nas/models/DifusserEdit/pretrain_models/VideoEdit-diffusers-model/transformer` |

### 2.2 输入和请求

| 项目 | 值 |
|---|---|
| 视频 | `/sgl-workspace/sglang/demo/1080.mp4` |
| Mask | `/sgl-workspace/sglang/demo/mask_1080_acc.mp4` |
| Reference | `/sgl-workspace/sglang/demo/local.png` |
| 原始 mask 分辨率 | 1920 x 1080 |
| Mask bbox | `[74, 187, 1793, 775]` |
| bbox crop | 1719 x 588 |
| 对齐后模型输入 | **1728 x 592** |
| `bbox_expand_scale` | 0.3 |
| Prompt seed | 42 |
| Guidance scale | 5.0 |
| TeaCache | false |
| Paste-back | true |

主性能矩阵统一使用：

```text
num_frames=80
infer_len=81
overlap=0
num_inference_steps=4
warmups=1
runs=3
```

`smoke` profiler 场景虽然只输出 16 帧，但会反射填充到 `infer_len=81`；其 DiT Attention shape 与单个 81 帧窗口一致。正式速度结论使用无 profiler 的 `profile81` 三次运行。

## 3. 最终无 Profiler 长测

最终运行目录：

```text
videoedit_phase15_diagnostics/phase15_20260722_063759
```

配置：Triton dynamic W8A8、fused QKV/KV、self Sage FP8、cross FlashAttention、layerwise offload。

| 正式运行 | DiT 降噪 | 请求总耗时 | 状态 |
|---|---:|---:|---|
| run00 | 178.895 s | 228.085 s（最小值） | completed |
| run01 | 178.878 s | 228.223 s | completed |
| run02 | 178.902 s | 228.264 s（最大值） | completed |
| **中位数** | **178.895 s** | **228.223 s** | **completed** |
| **p95/最大正式值** | **178.902 s** | **228.264 s** | - |

三次降噪耗时极差只有约 `23.6 ms`，说明最终结果稳定。服务 ready 耗时为 `72.03 s`，不计入请求耗时。

数据源：

- [最终 matrix summary](../videoedit_phase15_diagnostics/phase15_20260722_063759/matrix_summary.json)
- [最终 benchmark summary](../videoedit_phase15_diagnostics/phase15_20260722_063759/fp8_layerwise/benchmark/phase0_20260722_063912.summary.json)
- [最终 Attention audit](../videoedit_phase15_diagnostics/phase15_20260722_063759/fp8_layerwise/attention_runtime_audits.json)
- [最终 quantization audit](../videoedit_phase15_diagnostics/phase15_20260722_063759/fp8_layerwise/quantization_audits.json)

## 4. 4-Step 性能演进

以下数据都使用双卡 SP2、81 帧窗口、4 steps、1 次 warmup 和 3 次正式运行。Profiler smoke 数据不放入此表。

| 阶段 | Linear backend | Fused QKV/KV | Self Attention | 总耗时 median | 总耗时 p95 | 降噪 median | 降噪 p95 | 降噪相对 BF16 |
|---|---|---|---|---:|---:|---:|---:|---:|
| BF16 baseline | BF16 cuBLAS | 否 | Flash BF16 | 348.060 s | 348.118 s | 298.457 s | 298.467 s | 1.000x |
| 初始在线 FP8 | SGL CUTLASS dynamic W8A8 | 否 | Flash BF16 | 333.042 s | 337.235 s | 283.143 s | 283.146 s | **1.054x** |
| Triton FP8 | Triton dynamic W8A8 | 否 | Flash BF16 | 316.562 s | 317.134 s | 266.580 s | 266.677 s | **1.120x** |
| Triton FP8 + K3 | Triton dynamic W8A8 | 是 | Flash BF16 | 316.492 s | 316.854 s | 267.359 s | 267.422 s | **1.116x** |
| 最终 FP8 + Sage | Triton dynamic W8A8 | 是 | Sage 8-bit | **228.223 s** | **228.264 s** | **178.895 s** | **178.902 s** | **1.668x** |

阶段结论：

- 初始在线 FP8 只让 DiT 降噪加速 `1.054x`，符合最初“收益很低”的观察。
- Triton backend 将降噪提升到 `1.120x`；Linear 的真实 FP8 Tensor Core 收益开始转化到服务。
- Fused QKV/KV 没有产生可分辨的端到端增益，`266.580 s` 和 `267.359 s` 的差异约 0.3%，应视为中性。它仍减少了重复 activation quant，并将量化 Linear 实例数从 488 合并为 328。
- 最终 Sage self-attention 相对 Triton+K3 的 DiT 降噪额外加速 **1.495x**。
- 最终配置相对 BF16 的 DiT 降噪为 **1.668x**，端到端请求为 **1.525x**。

主要数据源：

- [BF16/初始 FP8 矩阵](../videoedit_phase15_diagnostics/phase15_20260721_034005/matrix_summary.json)
- [Triton FP8 矩阵](../videoedit_phase15_diagnostics/phase15_20260721_074527/matrix_summary.json)
- [Triton+K3 矩阵](../videoedit_phase15_diagnostics/phase15_20260721_094946/matrix_summary.json)
- [最终 FP8+Sage 矩阵](../videoedit_phase15_diagnostics/phase15_20260722_063759/matrix_summary.json)

### 4.1 No-Offload 与 Layerwise

FP8 Transformer 可以在双卡上运行 no-offload，但三轮矩阵都没有显示出稳定的降噪收益：

| Linear 配置 | No-offload 总耗时 | Layerwise 总耗时 | No-offload 降噪 | Layerwise 降噪 |
|---|---:|---:|---:|---:|
| 初始 SGL dynamic W8A8 | 332.823 s | 333.042 s | 282.972 s | 283.143 s |
| Triton dynamic W8A8 | 317.368 s | 316.562 s | 266.573 s | 266.580 s |
| Triton dynamic W8A8 + K3 | 317.408 s | 316.492 s | 267.373 s | 267.359 s |

降噪差异均不到 0.1%；请求总耗时差异也不足以建立稳定优势。BF16 no-offload 会失败，而 layerwise 能给 FP8 留出更多显存余量，因此最终配置保留 `dit_layerwise_offload=true`。

## 5. 历史 40-Step Phase0/Phase1

这组数据来自较早的 Phase0/Phase1 代码和服务配置，用于说明最初在线量化为什么看起来几乎没有收益。它不与上面的最新 4-step 矩阵混合计算最终加速比。

### 5.1 单个 81 帧窗口

| 配置 | Steps | 正式次数 | DiT 降噪 median | 请求总耗时 median | 总耗时 p95 | 降噪加速 |
|---|---:|---:|---:|---:|---:|---:|
| BF16 Phase0 | 40 | 5 | 2052.696 s（34m12.7s） | 2102.707 s（35m02.7s） | 2103.014 s | 1.000x |
| 初始在线 FP8 Phase1 | 40 | 5 | 1946.638 s（32m26.6s） | 2103.654 s（35m03.7s） | 2307.416 s | **1.054x** |

初始 FP8 的降噪确实快了约 5.4%，但其他阶段和运行波动抵消了收益，因此请求总耗时中位数几乎没有变化。

### 5.2 完整 126 帧视频

`full` 场景使用 `infer_len=81`、`overlap=10`、40 steps：

| 配置 | 请求总耗时 median | 约合时间 | 总耗时加速 |
|---|---:|---:|---:|
| BF16 Phase0 | 4189.067 s | 1h09m49s | 1.000x |
| 初始在线 FP8 Phase1 | 4025.310 s | 1h07m05s | **1.041x** |

历史数据源：

- [Phase0 BF16 summary](../videoedit_phase0_outputs/phase0_20260717_043315.summary.json)
- [Phase1 FP8 summary](../videoedit_phase1_fp8_dynamic_outputs/phase0_20260720_101810.summary.json)

当前最终 FP8+Sage 配置尚未执行 40-step 三次正式回归，因此不能把 4-step 的 `1.668x` 直接声明为 40-step 实测结果。

## 6. Linear W8A8 验证

### 6.1 最终运行审计

最终 Transformer 审计结果：

| 审计项 | 结果 |
|---|---:|
| Linear 实例数 | 328 |
| `Fp8LinearMethod` 数量 | 328/328 |
| FP8 weight 数量 | 328/328 |
| 预测 true W8A8 数量 | 328/328 |
| 实际路由 | `dynamic_per_token_fp8+triton_scaled_mm` |
| 未量化 Transformer Linear | 0 |
| BF16 dequant fallback | 0 |
| Weight dtype | `torch.float8_e4m3fn` |
| FP8 weight bytes | 16,388,587,520 bytes（15.263 GiB） |

328 个实例是 fused projection 后的数量：self Q/K/V、text K/V 和 image added K/V 分别合并为 QKV/KV。它不表示有层被漏量化。

当前 Linear 量化方式为：

```text
Weight:
  FP8 E4M3FN
  per-output-channel
  symmetric max(abs(weight)) scale
  FP32 scale

Activation:
  runtime dynamic FP8 E4M3FN
  per-token
  symmetric max(abs(activation)) scale

Output/bias/residual/norm:
  BF16/FP32
```

### 6.2 L20 真实 Shape Backend Microbenchmark

每个 shape 预热 3 次、正式 10 次。`dynamic W8A8` 包含 activation scale 和 FP8 quant；`GEMM-only` 不包含动态量化。

| Shape `(M,K,N)` | 主要层 | BF16 median | SGL dynamic | Triton GEMM-only | Triton dynamic | FP8 硬件上限* |
|---|---|---:|---:|---:|---:|---:|
| `(41958,5120,13824)` | FFN up | 50.650 ms | 1.234x | 1.866x | **1.804x** | 1.926x |
| `(41958,13824,5120)` | FFN down | 51.589 ms | 1.257x | 1.931x | **1.691x** | 2.026x |
| `(41958,5120,5120)` | Attention/output projection | 18.767 ms | 1.222x | 1.855x | **1.699x** | 1.979x |
| `(83916,5120,64)` | `proj_out` 窄矩阵 | 1.351 ms | 0.512x | 1.798x | **0.516x** | 1.923x |

\* FP8 硬件上限使用 `torch._scaled_mm` scalar scale + fast accumulation，只用于证明 L20 的 FP8 计算能力，不等同于生产 per-token/per-channel 量化方案。

结论：

- L20 硬件和 CUDA 栈能够在合适的大矩阵上接近 `2x`，不存在“L20 FP8 完全不加速”的硬件问题。
- 原始 SGL CUTLASS dynamic W8A8 只有约 `1.22x-1.26x`。
- Triton dynamic W8A8 在三个主导大矩阵上达到 `1.69x-1.80x`。
- `proj_out` 的 `N=64` 很窄，动态量化耗时高于 GEMM 收益；后续可以为该 shape 增加 BF16/hybrid 路由，但它只有一个实例，不是当前主要瓶颈。

数据源：[L20 FP8 backend benchmark](../videoedit_phase15_diagnostics/phase15_20260721_034005/l20_fp8_backend_bench.json)。

## 7. Attention 低精度实验

### 7.1 实际 Shape

| 角色 | Q shape | K/V shape | 最终 backend |
|---|---|---|---|
| Self | `[1,83916,20,128]` | `[1,83916,20,128]` | Sage 8-bit |
| Text cross | `[1,41958,40,128]` | `[1,512,40,128]` | Flash BF16 |
| Image cross | `[1,41958,40,128]` | `[1,257,40,128]` | Flash BF16 |

### 7.2 Attention Microbenchmark

| 角色 | Flash BF16 | Sage QK INT8 + PV FP16 | 加速 | Sage QK INT8 + PV FP8 | 加速 | FP8 cosine | FP8 relative L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Self | 629.830 ms | 554.342 ms | 1.136x | **353.361 ms** | **1.782x** | 0.999246 | 0.03882 |
| Text cross | 4.346 ms | 4.994 ms | 0.870x | 3.571 ms | 1.217x | 0.999327 | 0.03668 |
| Image cross | 3.335 ms | 3.802 ms | 0.877x | 2.804 ms | 1.189x | 0.999342 | 0.03626 |

Self-attention 占绝大部分 Attention 时间，Sage FP8 达到 `1.782x` 且通过数值 gate。Cross-attention 的绝对耗时很小，切换 Sage 收益有限，因此最终只优化 self，cross 保持 FlashAttention。

Sage self-attention 的实际参数为：

```text
kernel=qk_int8_pv_fp8_cuda
qk_quant_gran=per_thread
pv_accum_dtype=fp32+fp16
smooth_k=true
```

这里没有 Attention weight。低精度对象是 Q/K/V activation：Q/K 进入 INT8 QK 路径，PV 使用 FP8 路径。

数据源：[Attention microbenchmark](../videoedit_phase15_diagnostics/attention_microbench/attention_bench_20260722_053358.json)。

## 8. Operator Profile

Profiler 各采样一个 denoise timestep，调用数和 shape 一致。下面对比 FP8+Flash 与最终 FP8+Sage；除 self-attention 外配置相同。

| GPU 类别 | FP8+Flash | FP8+Sage | 变化/加速 |
|---|---:|---:|---:|
| FP8 activation quant | 0.715 s | 0.715 s | 基本不变 |
| FP8 GEMM | 9.204 s | 9.204 s | 基本不变 |
| Self-attention | 50.375 s | 28.261 s | **1.782x** |
| Text cross-attention | 0.346 s | 0.346 s | 基本不变 |
| Image cross-attention | 0.265 s | 0.265 s | 基本不变 |
| SP communication | 3.096 s | 3.088 s | 基本不变 |
| Other GPU kernels | 4.118 s | 4.113 s | 基本不变 |
| **全部 Attention** | **50.986 s** | **28.872 s** | **1.766x** |
| **GPU kernel 总时间** | **68.118 s** | **45.992 s** | **1.481x** |

最终 profile 中的占比：

| 类别 | 当前 GPU 时间占比 |
|---|---:|
| Self-attention | 61.45% |
| FP8 GEMM | 20.01% |
| Other GPU kernels | 8.94% |
| SP communication | 6.71% |
| FP8 activation quant | 1.56% |
| Text + image cross-attention | 1.33% |

这解释了为什么整个 DiT 没有达到 2x：即使主导 Linear 和 self-attention 都接近 `1.7x-1.8x`，通信、量化、Norm、残差和其他 kernel 并没有同步加速。

数据源：

- [FP8+Flash profile](../videoedit_phase15_diagnostics/phase15_20260722_050218/matrix_summary.json)
- [FP8+Sage profile](../videoedit_phase15_diagnostics/phase15_20260722_060314/matrix_summary.json)
- [FP8+Sage operator breakdown](../videoedit_phase15_diagnostics/phase15_20260722_060314/fp8_layerwise/torch_profiler/operator_breakdown.json)

## 9. 显存

相同诊断场景、layerwise offload 下的 `nvidia-smi` 峰值：

| 配置 | GPU 0 峰值 | GPU 1 峰值 | 相对 BF16 |
|---|---:|---:|---:|
| BF16 layerwise | 39,589 MiB | 39,563 MiB | baseline |
| FP8 Triton+K3+Flash | 36,909 MiB | 36,907 MiB | GPU 0 减少 2,680 MiB（6.77%） |
| FP8 Triton+K3+Sage | 36,909 MiB | 36,907 MiB | 与 FP8+Flash 相同 |

Transformer FP8 weight 本身约为 15.263 GiB，但服务峰值还包含 activation、Attention workspace、其他模型组件和 layerwise staging，因此不能按 weight 减半直接推导服务显存减半。

BF16 no-offload 在诊断矩阵中失败；FP8 no-offload 可以运行，但与 layerwise 的降噪速度几乎相同。为保留显存余量，最终继续使用 layerwise offload。

## 10. 正确性和质量

已经确认：

- 最终 3 次视频任务全部 `completed`。
- 328/328 Transformer Linear 实际为 FP8 W8A8 路径，无 BF16 dequant fallback。
- Self-attention 实际 backend 为 `SageAttentionImpl`，没有静默回退到 FlashAttention。
- Text/image cross-attention 实际为 `FlashAttentionImpl`。
- Sage FP8 self-attention microbenchmark 输出 finite，cosine 为 `0.999246`，relative L2 为 `0.03882`。
- 生产 adapter 小 shape 验证 cosine 为 `0.999284`，relative L2 为 `0.03783`。
- 36 项定向单测通过，包括角色路由和“指定 Sage 后禁止静默回退”测试。
- Phase1 Linear 量化视频由人工观察未见明显崩坏。

尚未完成：

- 最终 FP8+Sage 视频与 BF16 的正式盲测/并排人工评分。
- PSNR、SSIM、LPIPS、时序一致性等量化质量指标。
- 最终配置的 40-step 三次正式性能和质量回归。

因此当前可以确认性能和 kernel 路径，但最终生产质量验收仍需补充完整视频对照。

## 11. 无效实验和已排除问题

以下失败运行不进入性能统计：

| 问题 | 原因 | 处理 |
|---|---|---|
| FP8 加载时报 Triton CPU pointer | quant-after-load 时权重仍在 CPU/offload 路径 | `dit_cpu_offload=false`，保留 layerwise offload |
| `Can't disable Kineto profiler when it's not running` | 嵌套 profiler 生命周期错误 | 修复 profiler ownership/停止逻辑后重跑 |
| Scheduler timeout | 旧的有限 scheduler 等待时间 | 长任务使用无限 task timeout |
| `/tmp/sglang-videoedit-inputs` permission denied | 服务和客户端目录 ownership 不一致 | 诊断脚本使用每次运行独立的 workspace input/output 目录 |
| `.cache` performance record permission warning | 仓库 `.cache` 不可写 | 不影响 perf JSON、视频和 profiler 结果 |

`phase15_20260722_013842`、`021633`、`023805`、`024439` 等 profiler 失败目录不用于最终速度结论。

## 12. 最终复现命令

```bash
python scripts/videoedit_phase15_diagnose.py \
    --model-path /mnt/nas/models/DifusserEdit/pretrain_models/VideoEdit-diffusers-model \
    --transformer-path /mnt/nas/models/DifusserEdit/pretrain_models/VideoEdit-diffusers-model/transformer \
    --video /sgl-workspace/sglang/demo/1080.mp4 \
    --mask /sgl-workspace/sglang/demo/mask_1080_acc.mp4 \
    --reference /sgl-workspace/sglang/demo/local.png \
    --variants fp8_layerwise \
    --scenarios profile81 \
    --bbox-expand-scale 0.3 \
    --out-root /sgl-workspace/sglang/videoedit_phase15_diagnostics \
    --skip-microbench \
    --server-extra-arg=--transformer-fp8-gemm-backend \
    --server-extra-arg=triton \
    --server-extra-arg=--transformer-fp8-fused-projections \
    --server-extra-arg=true \
    --server-extra-arg=--videoedit-self-attention-backend \
    --server-extra-arg=sage_attn \
    --server-extra-arg=--videoedit-cross-attention-backend \
    --server-extra-arg=fa
```

## 13. 后续建议

按优先级：

1. 使用最终配置跑 40-step `single81`，至少 1 次 warmup、3 次正式运行，确认 `1.668x` 是否能延续到完整 denoise schedule。
2. 对 BF16 与最终 FP8+Sage 输出做固定 seed 的人工并排评估，并补充时序质量指标。
3. 为 `(83916,5120,64)` 的窄 `proj_out` 增加 hybrid BF16 路由，避免动态 quant 开销超过 GEMM 收益。
4. 如果仍需继续接近 2x，下一目标应是 self-attention kernel、SP 通信和其他融合 kernel，而不是继续量化 text/image cross-attention；cross 当前只占 GPU 时间约 1.33%。
5. 性能和质量通过后，再把在线量化 weight 转成离线 FP8 checkpoint，以缩短启动时间并降低加载峰值。离线权重不会自动进一步加速 forward。
