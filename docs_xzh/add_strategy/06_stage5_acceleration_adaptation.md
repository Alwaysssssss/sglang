# Stage 5: 加速路径适配

## 1. 先给结论

Vivid-VR 当前实际使用到的“加速 / 省显存”能力主要是：

- VAE slicing
- VAE tiling
- diffusers model CPU offload
- 可选 sequential CPU offload

明确没有看到它在主恢复链里实际启用：

- TeaCache
- CacheDiT
- torch.compile
- CUDA Graph
- 自定义 Triton 推理内核
- 多卡 TP/SP

注意：

- `CogVLM2` checkpoint 内部有 Triton/flash 相关代码，但那是 caption 模型自己的实现，不代表 VividVR 主恢复链已经接入这些能力。

## 2. 原始仓库现状

来自 `VRDiT/inference.py` 的真实调用：

- `vae.enable_slicing()`
- `vae.enable_tiling()`
- `pipe.enable_model_cpu_offload()`
- 注释中保留 `pipe.enable_sequential_cpu_offload()`

这说明原项目的主优化策略是：

- 依赖 diffusers 的 offload
- 依赖 VAE 的分块

而不是原生 runtime kernel 优化。

## 3. 在 SGLang 中哪些能力可以自动获得

若采用原生 pipeline 路线，可自动继承或较容易复用：

- `TransformerLoader` / `VAELoader` 的组件加载链
- `torch.compile` 开关能力
- CPU offload / layerwise offload 框架
- attention backend 抽象
- profiler / 日志

前提：

- 组件已经原生化到 SGLang runtime
- forward 合同稳定

## 4. 哪些能力需要主动适配

| 能力 | 是否自动获得 | 说明 |
| --- | --- | --- |
| VAE tiling / slicing | 否，需要组件实现 | `CogVideoX` VAE 类本身要支持 |
| model CPU offload | 部分可复用 | 需要 pipeline / server_args 对应挂载 |
| layerwise offload | 需要主动接入 | 取决于 transformer 是否适配 SGLang offload mixin |
| attention backend | 需要主动适配 | `CogVideoX` attention 实现必须接入 SGLang 抽象 |
| torch.compile | 部分可复用 | 但前提是 forward 图稳定 |
| TP/SP | 不会自动获得 | 需要 `CogVideoX` 原生并行实现 |
| CacheDiT / TeaCache | 不会自动获得 | 需要专门 adapter 和 block 结构支持 |

## 5. 各能力的接入阶段建议

### 5.1 MVP 阶段

只保留：

- `bf16` / `fp16`
- VAE tiling
- VAE slicing
- 最基础的 CPU offload

### 5.2 正确性稳定后

再评估：

- `torch.compile`
- layerwise offload
- attention backend 优化

### 5.3 最后阶段

再评估：

- TP/SP
- CacheDiT / TeaCache
- kernel 级定制

## 6. 为什么不建议第一版做 TP/SP

原因：

- `sglang` 当前没有 `CogVideoX` 并行基座
- VividVR 还额外引入 connector / control_hidden_states
- 多卡切分会放大调试复杂度

因此正确顺序是：

1. 先单卡数值对齐
2. 再分析瓶颈
3. 再决定是否设计 `CogVideoX` 并行化

## 7. 为什么不建议第一版做 CacheDiT / TeaCache

原因：

- 当前没有现成的 `CogVideoX` / VividVR cache adapter
- VividVR timestep 状态里多了 `old_pred_original_sample`
- spatial tile 级 denoise 会改变缓存边界

结论：

- 这不是“接一下开关”就能用
- 必须等模型结构稳定后再考虑

## 8. attention backend 风险点

VividVR 的 transformer 和 controlnet 使用 `CogVideoXBlock` 及其 attention 语义。

风险在于：

- SGLang 当前 attention 抽象主要为已有模型族调优
- `CogVideoX` 的 patch / rotary / temporal layout 不一定能无缝套上

建议：

- 第一版先保守用最稳的 PyTorch 路径
- 再评估是否映射到 SGLang 的高性能 backend

## 9. 加速验证文档需要回答的问题

后续真正接加速前，必须先回答：

- VAE decode 是不是主瓶颈
- tile 数量对总时长影响多大
- caption 是否已经成为瓶颈
- 长视频路径中，是 denoise 慢还是 merge 慢
- compile 对 tile 形状是否稳定

## 10. 本阶段的建议结论

- 第一版不追求“完整继承 SGLang 全部加速能力”
- 第一版只要保证结构上不阻断后续加速即可
- 真正值得规划的优化优先级是：
  1. VAE tiling/slicing
  2. 基础 offload
  3. compile
  4. attention backend
  5. TP/SP
