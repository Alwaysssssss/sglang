# Disaggregation 与优化能力

## 1. 为什么要看这一层

如果只看普通 Pipeline，你会觉得 `multimodal_gen` 是“一个结构清晰的扩散框架”。但如果继续往下看 `runtime/disaggregation/`、`runtime/cache/`、`runtime/platforms/`、`runtime/postprocess/`，会发现它真正的目标是“可部署、可优化、可扩展的生成系统”。

## 2. Disaggregation：把 Pipeline 拆成角色

位置：`runtime/disaggregation/`

角色定义在 `roles.py`：

- `MONOLITHIC`
- `ENCODER`
- `DENOISER`
- `DECODER`
- `SERVER`

这意味着一条扩散链路可以拆成：

- 编码阶段服务
- 去噪阶段服务
- 解码阶段服务
- 一个前端路由服务

## 3. 角色过滤如何作用到 Pipeline

`roles.filter_modules_for_role()` 会根据组件名判断归属：

- text/image encoder、tokenizer、processor -> `ENCODER`
- transformer -> `DENOISER`
- vae/vocoder -> `DECODER`

`ComposedPipelineBase.__init__()` 会在 disagg 模式下直接过滤 `_required_config_modules`；`add_stage()` 也会根据 stage 的 `role_affinity` 决定是否注册。

这意味着：

- 不是“同一个完整 pipeline 跑三遍”；
- 而是“构造出角色裁剪后的 pipeline 子集”。

## 4. `DiffusionServer`：全局编排器

位置：`runtime/disaggregation/orchestrator.py`

它是 server/head node，职责包括：

- 接收前端请求；
- 管理 encoder/denoiser/decoder 实例池；
- 做 capacity-aware dispatch；
- 跟踪 request state；
- 管理 TTA 队列；
- 管理跨角色 tensor transfer。

它更像“多阶段流水线调度器”，而不是简单负载均衡器。

## 5. transport 子系统

目录：`runtime/disaggregation/transport/`

从文件名可以看出，它单独抽出了：

- `allocator.py`
- `buffer.py`
- `codec.py`
- `engine.py`
- `manager.py`
- `protocol.py`

在 orchestrator 代码中还能看到：

- `TransferAllocMsg`
- `TransferPushMsg`
- `TransferReadyMsg`

这说明 disagg 模式不是把中间 tensor 序列化成 Python 对象乱传，而是认真设计了控制面与数据面协议。

## 6. 为什么 disagg 值得注意

扩散模型天然具有阶段结构：

- 编码部分相对轻
- 去噪部分最重
- 解码部分通常又是另一种资源模式

把它们拆开后，可以针对不同角色做独立扩缩容和资源配置。这对大规模服务特别关键。

## 7. Stage 级优化能力

除了 disagg，这个项目还有大量 stage/运行时优化。

## 7.1 attention backend 选择

相关模块：

- `runtime/layers/attention/selector.py`
- `runtime/layers/attention/backends/*`
- `runtime/platforms/*`

支持多种 backend，例如：

- flash attention
- sage attention
- sdpa
- video sparse attention
- vmoba
- ascend fa

`ServerArgs._adjust_attention_backend()` 会结合平台、并行模式和模型情况自动选择默认值。

## 7.2 cache-dit

相关模块：

- `runtime/cache/cache_dit_integration.py`
- `DenoisingStage._maybe_enable_cache_dit()`

其作用是在去噪循环里缓存 transformer 的部分上下文，减少重复计算。`DenoisingStage` 会根据环境变量和当前请求状态做启用、刷新、关闭控制。

## 7.3 LoRA

虽然 LoRA 更像功能性特性，但这里它已经被深度优化到运行时里：

- 支持动态 set/merge/unmerge；
- 支持多 target；
- 兼容 layerwise offload；
- 能作为服务运行时控制指令操作。

## 7.4 torch.compile

`DenoisingStage` 会根据 `server_args.enable_torch_compile` 选择是否编译 transformer，还专门处理了：

- NPU backend
- inductor overlap 设置
- NVFP4 JIT 预热

这是一种“把编译优化嵌进运行时”的设计。

## 8. Offload 与显存策略

`ServerArgs` 和 `GPUWorker` 中可以看到非常多显存相关逻辑：

- `dit_cpu_offload`
- `dit_layerwise_offload`
- `text_encoder_cpu_offload`
- `image_encoder_cpu_offload`
- `vae_cpu_offload`
- `pin_cpu_memory`

值得注意的是：

- 这些参数不只是用户手动设置；
- 系统还会根据模型类型、设备总显存、平台类型自动推断默认值。

其中 `dit_layerwise_offload` 是一个很典型的性能/显存权衡项，尤其针对 Wan/MOVA 这类大视频模型。

## 9. 平台适配

平台目录：

- `runtime/platforms/cuda.py`
- `runtime/platforms/rocm.py`
- `runtime/platforms/mps.py`
- `runtime/platforms/npu.py`
- `runtime/platforms/musa.py`
- `runtime/platforms/xpu.py`

平台抽象不仅影响 device type，还会影响：

- attention backend 可用性
- VAE 优化
- FSDP 可用性
- kernel 选择
- 某些参数的自动默认值

所以这个项目不是简单写了几句 `if cuda else cpu`，而是专门做了一层平台抽象。

## 10. 后处理能力

目录：`runtime/postprocess/`

包括：

- `rife_interpolator.py`
- `realesrgan_upscaler.py`

这些功能在 `save_outputs(...)` 链路里被统一集成，因此：

- 生成结果可以直接插帧；
- 也可以直接超分；
- 不需要用户额外手工串联处理脚本。

这也是“产品化运行时”的一个特征。

## 11. Profiling 与性能观测

相关模块：

- `runtime/utils/perf_logger.py`
- `runtime/utils/profiler.py`
- `PipelineExecutor.profile_execution()`
- `PipelineStage` 中的 `StageProfiler`

可观测能力至少覆盖：

- 请求总时长
- stage 级时长
- 去噪 step 级记录
- 显存快照
- benchmark report dump

这对调优和 CI 性能回归都很重要。

## 12. 还有哪些“系统级”模块值得继续深挖

如果你要继续深入而不是停留在架构层，下一批建议读：

- `runtime/cache/*`
- `runtime/distributed/*`
- `runtime/layers/attention/*`
- `runtime/layers/quantization/*`
- `runtime/utils/layerwise_offload.py`
- `benchmarks/*`
- `test/server/*`

这些模块基本决定了它为什么比“纯 diffusers 调用脚本”更像一个真正的推理系统。

## 13. 总结

`multimodal_gen` 的高级能力可以概括为三类：

1. 架构级：disaggregation、角色化 pipeline、统一服务化链路
2. 运行时级：多并行模式、offload、cache-dit、LoRA、torch.compile
3. 产品级：HTTP/OpenAI API、后处理、观测与诊断

这也是为什么阅读这个目录时，不应该只盯着某个模型 Pipeline。真正的价值在于：它已经把“扩散模型推理”提升成了一套完整系统工程。
