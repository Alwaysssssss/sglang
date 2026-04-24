# 组件加载器与模型实现

## 1. 为什么需要 Loader 层

`multimodal_gen` 支持的模型来源很多：

- diffusers 目录
- transformers 模型目录
- safetensors 单文件
- FSDP / 量化权重
- 自定义 SGLang 模型实现

如果把这些逻辑直接写进 Pipeline，会非常混乱。所以框架单独抽出 `runtime/loader/`。

## 2. `ComponentLoader`：统一加载模板

位置：`runtime/loader/component_loaders/component_loader.py`

它定义了统一模板：

1. 记录加载前可用显存；
2. 先尝试 `load_customized()`；
3. 若失败则回退 `load_native()`；
4. 做 `.eval()`、device 放置、显存统计、日志输出。

这个模板体现了框架的总体策略：

- 优先 SGLang 自定义高性能实现；
- 失败时回退官方/原生库；
- 保证功能性优先，性能次之。

## 3. Loader 的自动注册

`ComponentLoader.__init_subclass__()` 会按 `component_names` 自动注册 loader。后续 `for_component_type()` 通过组件名选择对应 loader。

例如：

- `text_encoder` -> `TextEncoderLoader`
- `vae` / `video_vae` -> `VAELoader`
- `transformer` -> `TransformerLoader`
- `tokenizer` -> `TokenizerLoader`

这使得 `ComposedPipelineBase.load_modules()` 不需要知道每种组件的加载细节。

## 4. `TransformerLoader`

位置：`runtime/loader/component_loaders/transformer_loader.py`

这是最关键的 loader 之一。

它负责：

- 读取 diffusers component config；
- 根据 `component_name` 更新 `dit_config`；
- 解析权重文件列表；
- 解析量化装载规格；
- 从 `ModelRegistry` 获取模型类；
- 调 `maybe_load_fsdp_model()` 进行实例化与装载；
- 执行量化后处理 hook。

几个重要点：

### 4.1 配置先于模型实例化

它会先用 HF config 更新 `server_args.pipeline_config.dit_config`，然后再实例化模型类。这保证模型结构与权重一致。

### 4.2 `ModelRegistry` 负责找类，Loader 负责装载权重

这是清晰的职责分工：

- registry 解决“用哪个类”
- loader 解决“如何把权重装进去”

### 4.3 支持量化与 FSDP

装载路径里已经把以下问题统一考虑进去了：

- `transformer_weights_path`
- nunchaku 量化
- FSDP 推理
- CPU offload

## 5. `VAELoader`

位置：`runtime/loader/component_loaders/vae_loader.py`

它做的事包括：

- 读取 `_class_name`
- 更新 `vae_config`
- 处理 custom `auto_map` 类
- 从 `ModelRegistry` 获取标准 VAE 类
- 加载 safetensors 权重
- 执行平台侧 VAE 优化
- 按需转成 `channels_last_3d`

这说明 VAE 不是简单 `from_pretrained()`，而是走统一的 SGLang 运行时装载流程。

## 6. `TextEncoderLoader`

位置：`runtime/loader/component_loaders/text_encoder_loader.py`

相比 VAE/DiT，它的复杂点在于：

- 可能存在多个 text encoder；
- 既要支持 transformers 原生类，也要支持 SGLang 自定义类；
- 权重来源可能不只一处；
- 还要考虑 CPU offload 和 FSDP shard。

它内部把权重来源抽象成 `Source`，并支持 safetensors / pt 两种权重迭代器。

## 7. `ModelRegistry`：类发现与懒加载

位置：`runtime/models/registry.py`

这个模块的重点不是注册配置，而是注册“模型类实现”。

### 7.1 它扫描什么

它会扫描 `runtime/models/*/*.py`，解析 AST，寻找：

- 类定义
- `EntryClass`
- `_aliases`

这样就能建立：

- 架构名 -> `(component_dir, module_name, class_name)`
- alias -> canonical class name

### 7.2 为什么用 AST + 懒加载

原因在注释里已经写得很明确：避免主进程提前 import 导致 CUDA 初始化，尤其是在 fork / subprocess 环境中。

因此它区分：

- `_RegisteredModel`
- `_LazyRegisteredModel`

必要时甚至用子进程来 inspect model class。

这是一个很典型的“为多进程/CUDA 安全让步”的工程设计。

## 8. Loader 与 ModelRegistry 的分工

可以把它们理解成：

- `ModelRegistry`：类名解析器
- `ComponentLoader`：权重与实例装载器

两者配合后，框架就能支持：

- 同一套加载逻辑对接多个模型类；
- 同一类模型由多个不同 loader 路径加载；
- 同时保持 Pipeline 层足够干净。

## 9. 平台与设备适配

装载过程中会经常看到 `current_platform`，它负责处理平台差异，例如：

- CUDA
- ROCm
- MPS
- NPU
- MUSA
- XPU

Loader 不直接写死“怎么优化”，而是把平台相关优化下沉到 `runtime/platforms/*` 中。

## 10. 组件级 fallback 策略

这是整个系统里一个非常实际的工程点。

如果某个组件没有定制实现，框架不会直接报错，而是尝试：

- `load_customized()` 失败
- fallback 到 `load_native()`

并打印“performance may be sub-optimal”。

这意味着：

- 框架允许“部分原生、部分官方”混合运行；
- 适配工作可以渐进式推进，而不必一次性做完所有组件。

## 11. `memory_usages` 的作用

`ComposedPipelineBase.load_modules()` 会记录各组件加载时消耗的显存，放到 `pipeline.memory_usages`。

`GPUWorker.do_mem_analysis()` 会用它估算：

- 哪些模块其实可以常驻 GPU；
- 哪些 offload 参数可能可以关闭。

这是一种“基于最近一次请求工作负载”的经验性建议机制。

## 12. LoRA 与加载层的关系

LoRA 不是 loader 直接处理的，而是在 Pipeline 构建完成后由 `LoRAPipeline` 把目标层替换成 LoRA 版本，并支持：

- 设置 adapter
- merge / unmerge
- 多 target
- 多 adapter

但它又和加载层深度耦合，因为：

- 某些模型需要参数名映射；
- layerwise offload 与 LoRA 需要协调；
- 二次 transformer（如 `transformer_2`）需要特殊处理。

## 13. 典型的加载路径

以 transformer 为例，加载链通常是：

1. Pipeline 决定需要 `transformer`
2. `ComposedPipelineBase.load_modules()`
3. `PipelineComponentLoader.load_component(...)`
4. `ComponentLoader.for_component_type("transformer", ...)`
5. `TransformerLoader.load_customized(...)`
6. `ModelRegistry.resolve_model_cls(cls_name)`
7. FSDP/量化/权重装载
8. 返回 `torch.nn.Module`

## 14. 整体评价

Loader 层是这个项目工程成熟度很高的一部分。它的价值不在于“代码优雅”，而在于把下面这些复杂性都收口了：

- 模型类发现
- 权重文件选择
- 量化装载
- FSDP
- CPU offload
- 平台优化
- 原生/官方 fallback

没有这一层，Pipeline 很快就会退化成一堆和业务逻辑无关的加载胶水代码。

