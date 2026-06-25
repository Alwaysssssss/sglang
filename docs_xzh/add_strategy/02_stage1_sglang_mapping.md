# Stage 1: 与 SGLang 架构映射

## 1. 映射结论

Vivid-VR 接入 SGLang 时，组件应分成三类：

| 类别 | 组件 | 结论 |
| --- | --- | --- |
| 可直接复用 | `ComposedPipelineBase`、`Req`、`PipelineExecutor`、`registry.py` 主流程、`TransformerLoader`、`VAELoader`、`TextEncodingStage`、T5 tokenizer / encoder 加载链 | 直接复用 |
| 可轻度改造 | `PipelineConfig`、`SamplingParams`、model registry 自动发现、标准组件 loader 的使用方式 | 通过新增 config / sampling / EntryClass 复用 |
| 必须新建或重写 | `CogVideoX` transformer、VividVR transformer、VividVR controlnet、CogVideoX VAE、CogVideoX DPM scheduler、VividVR model-specific stages、long-video orchestration helpers | 新增本地原生实现 |

## 2. 为什么不能直接走 DiffusersPipeline generic wrapper

`sglang.multimodal_gen/runtime/pipelines/diffusers_pipeline.py` 能做的是：

- 包裹一个完整 diffusers pipeline
- 自动透传 kwargs
- 规范化 output tensor

它不能稳定承接 VividVR 的核心诉求：

- 单独调用 `pre_denoise_process()`
- 自定义长视频 clip 级 timestep orchestration
- 跨 clip latent merge
- 后续与 SGLang 原生 runtime 的加速能力做结构级对接

因此：

- generic wrapper 只适合作为对照实现和 smoke test
- 最终生产路径必须是 `sglang.multimodal_gen` 原生 pipeline

## 3. 与 add_rules 的风格对齐

### 3.1 modular 规则仍然生效

仍然必须遵守：

- pipeline 只做编排
- 私有逻辑局部化
- 复用公共 registry / loader / executor
- 进入 denoising 之前合同清晰

### 3.2 但不应机械套用 Wan VideoEdit 的细粒度 stage

不建议把 VividVR 拆成大量细粒度 stages，原因有三点：

1. `pre_denoise_process()` 天然是强耦合块
2. prompt 不是单条，而是 tile 列表
3. 长视频时间聚合必须在 pipeline 级别掌控

推荐组织方式：

- `VividVRInputValidationStage`
- `VividVRBeforeDenoisingStage`
- `VividVRDenoisingStage`
- `VividVRDecodingStage`

这仍然符合 add_rules，只是更偏 hybrid。

## 4. SGLang 现有能力对应表

| Vivid-VR 功能 | SGLang 可复用能力 | 建议 |
| --- | --- | --- |
| pipeline 编排 | `ComposedPipelineBase` | 直接复用 |
| 阶段执行 | `PipelineExecutor` | 直接复用 |
| 请求批结构 | `Req` + `SamplingParams.runtime_*` | 直接复用 |
| T5 文本编码 | `TextEncodingStage.encode_text()` | 直接复用 |
| diffusers 组件目录解析 | component loaders | 直接复用 |
| pipeline 自动发现 | `runtime/pipelines/*` 的 `EntryClass` | 直接复用 |
| model 自动发现 | `runtime/models/*` 的 `EntryClass` | 直接复用 |
| `CogVideoX` Transformer | 无现成实现 | 必须新建 |
| `CogVideoX` VAE | 无现成实现 | 必须新建 |
| `CogVideoX` DPM scheduler | 无现成实现 | 必须新建 |
| VividVR controlnet | 无现成组件类型 | 第一版局部实现，不先公共化 |
| 长视频时间聚合 | 无现成通用模块 | 在 `runtime/vividvr/windowing.py` 新建 |
| auto caption | 有部分 VL encoder 基础，但当前 `sglang` 环境中的 `CogVLM2` caption 输出乱码 | 当前阶段禁止实时接入，统一读取 `prompt.txt` |

## 5. 目标架构

推荐的 SGLang 落地结构：

```text
Req
  └─ sampling_params = VividVRSamplingParams
        ├─ 请求字段
        └─ runtime_* 私有状态

VividVRPipeline.forward()
  ├─ 短视频：单 clip 执行
  └─ 长视频：多 clip orchestration + timestep 级 merge

stages:
  validation
  -> before_denoising
  -> denoising
  -> decoding

modules:
  tokenizer
  text_encoder
  vae (CogVideoX)
  transformer (VividVR variant)
  scheduler (CogVideoX DPM vividvr variant)
  controlnet (VividVR private module)
```

caption 输入策略补充：

- 当前集成阶段不走 `CogVLM2` 实时 caption
- 统一读取：
  - `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`
- `runtime/vividvr/captioning.py` 只保留占位或 sidecar 文件读取逻辑

## 6. 哪些层不要动

第一版明确不建议改：

- `ComposedPipelineBase` 主流程
- `ModelRegistry` 主流程
- `TransformerLoader` 主流程
- `VAELoader` 主流程
- 公共 `DenoisingStage` 主逻辑

原因：

- 这些是 SGLang 现有稳定骨架
- 改它们会把局部模型接入问题扩大成 runtime 框架问题

## 7. 第一版 controlnet 的处理建议

`sglang` 当前没有独立 `runtime/models/controlnet/` 命名空间。

因此第一版建议：

- 把 `CogVideoXVividVRControlNetModel` 放在 `runtime/models/dits/` 下，视作 VividVR transformer 伴随组件
- 在 `VividVRPipeline.initialize_pipeline()` 中显式构造和挂载 `self.modules["controlnet"]`
- 暂不新增公共 `controlnet_loader.py`

这样可以把改动局部化在 VividVR 命名空间。

## 8. 待确认问题

- `controlnet` 是作为 `Pipeline._required_config_modules` 新增正式组件，还是第一版先作为 pipeline 局部私有模块加载。
- 若 `CogVideoX` VAE 原生移植成本过高，是否允许 MVP 阶段使用 `auto_map`/自定义类加载的过渡方案。文档建议是最终仍要落到原生 runtime 实现。
