# VideoEdit L20 FP8 W8A8 Kernel 性能定位与优化计划

## 1. 当前结论

当前问题不是“没有做到 W8A8”，而是“W8A8 kernel 没有充分利用 L20 的 FP8 Tensor Core”。

Phase 1.5 已确认：

- Transformer 中 488/488 个 Linear 都使用 `Fp8LinearMethod`；
- weight dtype 全部为 `torch.float8_e4m3fn`；
- BF16 activation 在每次 Linear 前由 `sglang_per_token_quant_fp8` 动态量化为 FP8；
- GEMM 进入 `sgl_kernel.fp8_scaled_mm`，没有 dequant 到 BF16 的 fallback；
- 当前路径是实际的 dynamic per-token activation + per-channel weight W8A8。

真实 VideoEdit shape 的单卡微基准为：

| Shape `(M, K, N)` | FP8 GEMM-only | dynamic W8A8 |
|---|---:|---:|
| `(41958, 5120, 13824)` | `1.264x` | `1.236x` |
| `(41958, 13824, 5120)` | `1.389x` | `1.258x` |
| `(41958, 5120, 5120)` | `1.301x` | `1.220x` |
| `(83916, 5120, 64)` | `1.978x` | `0.566x` |

三个主导 shape 上，BF16 已达到约 `115-117 TFLOPS`，dynamic FP8 只有约
`143-145 TFLOPS`。BF16 cuBLAS 路径效率较高，当前 SM89 FP8 CUTLASS 路径没有
接近硬件理论上限。

按 40 层和每个 timestep 两次 CFG forward 估算，主要 Linear 只占当前 denoise
时间约 23%。即使全部主要 Linear 达到 `2x`，整体 DiT 在当前条件下也只有约
`1.13x`。整体接近 `2x` 还需要优化 attention、序列并行通信，或减少
CFG/denoise 工作量。

## 2. 本计划要回答的问题

1. 当前机器上的 L20 使用独立且成熟的 FP8 GEMM backend 时，能否在真实 shape
   上明显超过 BF16？
2. 如果硬件上限正常，瓶颈位于当前 `sgl_kernel` CUTLASS GEMM、动态 activation
   量化，还是 backend 分发？
3. 在不降低 VideoEdit 数值质量的前提下，能否把主导 Linear 的 dynamic W8A8
   提升到至少 `1.6x`？

这些问题解决前，不进入 offline FP8 checkpoint 的性能开发。离线权重只能降低
启动量化时间和 checkpoint/显存体积，不会改变当前 forward kernel。

## 3. 验收标准

### 3.1 硬件能力

使用独立于 diffusion runtime 的 benchmark：

- BF16 基线使用 cuBLAS/cuBLASLt，输出 BF16；
- FP8 上限使用预量化 FP8 A/B 和 `torch._scaled_mm` 或直接 cuBLASLt；
- 使用相同 `(M, K, N)`、输出 dtype、warmup 和 CUDA Event 计时；
- 同时记录 SM clock、power、temperature 和 throttling reason。

判定：

- FP8 cuBLASLt 相对 BF16 `>=1.7x`：硬件和工具链正常，当前 SGL kernel 是主因；
- `1.4x-1.7x`：继续检查 accumulation、layout、clock 和 shape 利用率；
- `<1.4x`：先排查机器功耗/时钟、CUDA/cuBLASLt 和实际 SM89 指令路径。

`torch._scaled_mm` 在 SM89 上未必支持生产所需的 per-token/per-channel rowwise
scale，因此 scalar-scale 只作为硬件上限，不直接作为生产方案。

### 3.2 Linear kernel

对三个主导 shape：

- FP8 GEMM-only 相对 BF16 目标 `>=1.7x`；
- dynamic activation quant + FP8 GEMM 目标 `>=1.6x`；
- 十次正式测量 p95 波动不超过 5%；
- cosine similarity 不低于当前实现；
- 无 BF16 fallback，无 NaN/Inf。

### 3.3 DiT

- 488 个 Linear 的 dtype、scale 和 backend 审计完整；
- `profile81` 至少一次 warmup、三次正式运行；
- 报告 denoise median/p95；
- FP8 layerwise 与 no-offload 单独报告；
- 质量门槛沿用原 W8A8 计划。

## 4. Phase K0：独立硬件上限基准

新增：

```text
scripts/videoedit_l20_fp8_backend_bench.py
```

必须测试：

```text
bf16_cublas
fp8_torch_scaled_mm_scalar_gemm_only
fp8_sgl_cutlass_rowwise_gemm_only
fp8_sgl_cutlass_dynamic_w8a8
fp8_triton_rowwise_gemm_only
fp8_triton_dynamic_w8a8
```

shape 从 `linear_runtime_audits.json` 读取。输出 JSON 至少包含 backend、M/K/N、
dtype、scale scheme、quant/gemm/total 时间、TFLOPS、speedup、误差和 GPU 状态。

Phase K0 不修改服务代码，只负责给“硬件问题还是当前 kernel 问题”一个可复现的
二分结论。

## 5. Phase K1：显式 backend 和 profiling

### 5.1 CLI

在 `python/sglang/multimodal_gen/runtime/server_args.py` 增加：

```text
--transformer-fp8-gemm-backend auto|sgl_cutlass|triton|torch_scaled_mm
```

要求默认行为不变，不支持当前 scale scheme 时启动报错，禁止静默 fallback，并在
启动审计中同时输出请求 backend 和实际 backend。

### 5.2 拆分 quant 和 GEMM

修改：

```text
python/sglang/srt/layers/quantization/fp8_utils.py
python/sglang/multimodal_gen/runtime/layers/quantization/fp8.py
```

拆成向后兼容的两个接口：

```python
quantize_fp8_activation(input, scheme) -> (qinput, input_scale)
fp8_linear_gemm(qinput, weight, input_scale, weight_scale, backend) -> output
```

为 quant 和 GEMM 分别增加 NVTX range，记录 scale shape、weight layout 和实际
backend。profiling 开关不得改变数值路径。

### 5.3 现有 Triton 对照

重构前先使用现有开关补跑三个真实 shape：

```bash
USE_TRITON_W8A8_FP8_KERNEL=1
```

Triton 明显更快时先做 shape-based 分发，否则进入 K2。

## 6. Phase K2：优化 SM89 CUTLASS GEMM

修改 `sgl-kernel/csrc/gemm/fp8_gemm_kernel.cu`。

当前 `M>512` 主要使用固定的 `CTA 128x128x64`、`Warp 64x32x64` 和
2/3 stages。执行：

1. 用 CUTLASS profiler 和 Nsight Compute 采集 Tensor Core active、occupancy、
   waves、DRAM、L2 hit rate 和 stall reason。
2. 对 `N<=8192` 与 `N>8192` 搜索 `128x128`、`128x256`、`256x128`
   及 2/3/4 stages。
3. 离线生成最佳配置表，按 `(M bucket, K, N)` 分发，不在请求中 autotune。
4. 检查每次创建 workspace tensor 的开销，必要时改为 worker 级复用。
5. 保持 FP32 accumulator 和 BF16 output。降低累加精度必须独立验证质量。

扩展 `sgl-kernel/tests/test_fp8_gemm.py` 和
`sgl-kernel/benchmark/bench_fp8_gemm.py`，加入 VideoEdit 真实大 M。

## 7. Phase K3：消除重复 activation 量化

当前 Wan block 会重复量化相同 activation：

- self-attention Q/K/V 共用 normalized input，却分别量化三次；
- cross-attention K/V 共用 text context；
- added K/V 共用 image context。

优先使用合并投影：

1. self-attention 改成 merged QKV Linear，一次 quant、一次大 GEMM，输出 split；
2. cross-attention 合并 K/V；
3. image cross-attention 合并 added K/V；
4. 保持旧 checkpoint loader 兼容；
5. BF16 和 FP8 都做逐层一致性测试。

涉及 `wanvideo.py`、`wan_videoedit.py` 和 `linear.py`。不首选 tensor 指针
全局缓存，因为 storage 复用、in-place 修改、CUDA stream 和生命周期容易导致错误。

## 8. Phase K4：量化 kernel 和静态 activation

只有 K2/K3 后 quant 仍占主导 Linear 时间 5% 以上才执行：

1. 优化 `sglang_per_token_quant_fp8` reduction 和 vectorized load/store；
2. 确认 merged QKV/KV 只量化一次；
3. 比较 static per-tensor activation；
4. static 必须通过原计划的 calibration、holdout 和饱和率门槛。

不能为了接入 `torch._scaled_mm` 直接把所有层改成 per-tensor scale。backend
性能和量化精度变化必须分开实验。

## 9. Phase K5：服务验证和决策

固定矩阵：

```text
BF16 layerwise
FP8 current sgl_cutlass layerwise
FP8 optimized backend layerwise
FP8 optimized backend no-offload
```

`bf16_nooffload` 当前会 OOM，不作为必须完成的服务对照；纯 compute 由 K0 隔离。

Go 条件：

- 独立基准证明 L20 FP8 明显快于 BF16；
- 主导 shape dynamic W8A8 `>=1.6x`；
- 服务中 488/488 FP8 Linear 无 fallback；
- `profile81` denoise 稳定增益且质量不回退；
- 无新增 OOM、超时和跨 rank 不一致。

No-go 条件：

- cuBLASLt FP8 上限本身低于 `1.4x`；
- 优化后 dynamic W8A8 仍低于 `1.4x`；
- 必须使用不可接受的 scale scheme 或累加精度；
- Linear 加速没有转化为可重复的 DiT 收益。

No-go 后转向 attention、SP/Ulysses 通信、CFG parallel、TeaCache/Cache-DiT 和
inference step 优化。

## 10. 实施顺序

| 顺序 | 工作 | 主要文件 | 改生产路径 |
|---|---|---|---|
| 1 | 独立硬件/backend benchmark | 新 benchmark 脚本 | 否 |
| 2 | backend 参数与启动审计 | `server_args.py`, diffusion `fp8.py` | 默认不变 |
| 3 | quant/GEMM 拆分与 NVTX | `fp8_utils.py`, diffusion `fp8.py` | 是 |
| 4 | SM89 CUTLASS 配置搜索 | `fp8_gemm_kernel.cu` | 是 |
| 5 | merged QKV/KV | `wanvideo.py`, `wan_videoedit.py` | 是 |
| 6 | 动态量化优化/静态实验 | FP8 quant kernel、校准工具 | 可选 |
| 7 | `profile81` 和 full 回归 | Phase 1.5/Phase 0 脚本 | 否 |

第一轮只执行 1 到 3。得到硬件上限和 backend profile 后，再决定是调 CUTLASS、
切 Triton，还是补 cuBLASLt rowwise 路径。不要在没有 K0 数据时直接重写 kernel。
