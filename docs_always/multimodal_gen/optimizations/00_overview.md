# multimodal_gen 优化技术总览

> 本系列为 `python/sglang/multimodal_gen` 子系统中所有与「性能 / 显存 / 精度 / 吞吐」相关的优化技术做的深度拆解，每篇文档聚焦一个模块，包含源码路径、关键类、调用链、默认值、环境变量、参数开关与适配矩阵，方便排查性能问题或为新模型接入做特性选型。
>
> 上游总体架构请先阅读 [`../01_architecture_overview.md`](../01_architecture_overview.md) 与 [`../07_disaggregation_and_optimization.md`](../07_disaggregation_and_optimization.md)。

## 0. 为什么要分这么多篇

与通用 LLM 推理不同，`multimodal_gen` 需要在 **扩散模型固有的多轮去噪** 和 **大尺寸隐空间 / 长序列视觉 token** 之上做优化；同时为视频 / 图像 / 3D 等多种任务提供一套统一后端。整个子系统实际可以拆成 **九大正交优化维度**，它们分别对应这系列文档的九篇主文：

| 维度 | 关键词 | 主文 |
|------|--------|------|
| 并行 | TP / CFG Parallel / USP(Ulysses + Ring) / FSDP / DiT 与 VAE 分组 | [`01_parallelism.md`](./01_parallelism.md) |
| 量化 | FP8 (dynamic/static/block) / ModelOpt FP8 & NVFP4 / ModelSlim W8A8 / Nunchaku SVDQuant W4A4 | [`02_quantization.md`](./02_quantization.md) |
| 缓存 | TeaCache (多项式 rescale + 累加阈值) / cache-dit + DBCache / TaylorSeer / SCM | [`03_cache.md`](./03_cache.md) |
| 注意力 | FA2/3/4、SageAttention 2/3、STA、VSA、VMoBA、SVG2、Sparse Linear、AITer、Ascend FA、XPU | [`04_attention_backends.md`](./04_attention_backends.md) |
| 算子 / kernel 融合 | RMSNorm / LayerNorm / SiluAndMul / RoPE (FlashInfer / AITer) / Elementwise / QK-Norm-RoPE / 渲染 | [`05_kernels.md`](./05_kernels.md) |
| 编译 | `torch.compile` + Inductor + `reorder_for_compute_comm_overlap` + NVFP4 JIT 预热 + NPU torchair | [`06_torch_compile.md`](./06_torch_compile.md) |
| 显存 / Offload | `dit_cpu_offload` / `dit_layerwise_offload` / 组件级 offload / `pin_cpu_memory` / VAE tiling / slicing | [`07_offload.md`](./07_offload.md) |
| 服务化解耦 (PD 分离) | 四角色拆分 / ZMQ 控制面 / Mooncake RDMA 数据面 / capacity-aware 调度 | [`08_disaggregation.md`](./08_disaggregation.md) |
| 后处理 | RIFE 插帧 / Real-ESRGAN 超分 | [`09_postprocessing.md`](./09_postprocessing.md) |

## 1. 优化维度在一个请求里的位置

以一个典型 `DiffGenerator.generate()` 请求为例，优化点沿着下面的生命周期分布：

```text
   ┌────────────┐       ┌────────────────┐
   │ ServerArgs │──────▶│ _adjust_parallelism │ ① 并行维度推断 (TP/SP/CFG)
   └────────────┘       │ _adjust_quant_config│ ② 量化方案推断 (Nunchaku)
                        │ _adjust_offload      │ ③ Offload 默认值
                        │ _adjust_platform_*   │ ④ 平台默认值 (Wan/MOVA 自动 layerwise)
                        └────────────────┘
                                │
                                ▼
                  ┌────────────────────────────┐
                  │   ComposedPipelineBase      │ ⑤ Disagg 角色过滤 stage/module
                  └────────────────────────────┘
                                │
                    ┌───────────┼─────────────┐
                    ▼           ▼             ▼
               Encoding       Denoising      Decoding
                (TE/IE)       (DiT loop)      (VAE)
                                │
       ┌────────────────────────┼───────────────────────────┐
       ▼                        ▼                           ▼
  _maybe_enable_cache_dit    set_forward_context        vae_tiling/slicing
  _maybe_enable_torch_compile ├─ 选定 Attention backend   vae_cpu_offload
  NVFP4 JIT prewarm           ├─ USPAttention A2A + Ring
  (transformer.compile)        ├─ RMSNorm/SiLU/RoPE kernel
                                ├─ LayerwiseOffload prefetch
                                └─ TeaCache skip / cache-dit hit
                                │
                                ▼
                        save_outputs (RIFE + Real-ESRGAN)
```

## 2. 优化开关的三种来源

同一类优化可能由多处触发，记住这三种来源能帮助快速定位：

1. **`ServerArgs` CLI / Python 参数**：如 `--tp-size`、`--enable-cfg-parallel`、`--enable-svdquant`、`--dit-layerwise-offload`、`--enable-torch-compile`、`--attention-backend`、`--cache-dit-config`。文件 [`runtime/server_args.py`](../../python/sglang/multimodal_gen/runtime/server_args.py)。
2. **环境变量（`envs.py`）**：性能相关特性大量通过 `os.environ` 暴露，尤其是 cache-dit（`SGLANG_CACHE_DIT_*`）、torch.compile 模式（`SGLANG_TORCH_COMPILE_MODE`）、attention backend 强制（`SGLANG_DIFFUSION_ATTENTION_BACKEND`）、AITer（`SGLANG_USE_AITER`）、MLX 融合（`SGLANG_USE_MLX`）。
3. **`SamplingParams` 请求参数**：面向单条请求的细粒度控制，如 `enable_teacache` / `teacache_params`、`enable_frame_interpolation`、`enable_upscaling`、`true_cfg_scale`。文件 [`configs/sample/sampling_params.py`](../../python/sglang/multimodal_gen/configs/sample/sampling_params.py)。

## 3. 快速索引（按使用场景）

| 我想…… | 先看 |
|--------|------|
| 把 DiT 延迟降下来 | `01_parallelism.md` + `04_attention_backends.md` + `06_torch_compile.md` |
| 在 24GB/40GB 卡上跑大视频模型 | `07_offload.md`（layerwise）+ `02_quantization.md`（Nunchaku/ModelOpt） |
| 减少去噪步数的计算 | `03_cache.md`（TeaCache + cache-dit + SCM） |
| 在 Blackwell / Hopper 上跑 FLUX2-NVFP4 | `02_quantization.md` + `06_torch_compile.md`（NVFP4 JIT 预热） |
| 跑视频稀疏注意力 | `04_attention_backends.md`（VSA / VMoBA / SVG2 / STA） |
| 大规模部署需要拆 encoder / denoiser / decoder | `08_disaggregation.md` |
| 生成后要 2×/4× 插帧 + 超分 | `09_postprocessing.md` |
| 在 ROCm / NPU / MPS 上复用同一代码 | `01_parallelism.md` + `04_attention_backends.md` + `05_kernels.md` + `06_torch_compile.md`（对应章节分别列了平台差异）|

## 4. 阅读建议

- 如果是 **首次阅读**，按编号顺序看即可，每篇都能独立成章。
- 如果是 **排查特定性能问题**，直接跳到 §3 表对应主题。
- 如果是 **给新模型接特性**，建议先读 `01 / 04 / 06 / 07`，再按需看 `02 / 03`。

> 每篇文档里引用的代码行号基于当前仓库 HEAD，随代码演化可能漂移，请以类 / 函数名为准。
