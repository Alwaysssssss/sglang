# `python/sglang/srt/model_loader` 模块分析

## 定位

`model_loader` 负责把模型配置解析成 SRT 可执行的 `nn.Module`，并把本地/HuggingFace/远端/量化/分片权重加载到模型参数。它是 `ModelRunner` 初始化阶段的关键依赖。

## 关键文件

- `loader.py`：loader 类中心。定义 `BaseModelLoader`、`DefaultModelLoader`、`LayeredModelLoader`、`QuantizedRLModelLoader`、`DummyModelLoader`、`ShardedStateLoader`、`BitsAndBytesModelLoader`、`GGUFModelLoader`、`RemoteInstanceModelLoader`、`RemoteModelLoader`、`ModelOptModelLoader`、`RunaiModelStreamerLoader`，以及 `get_model_loader`。
- `utils.py`：模型架构解析，决定使用 SRT 原生模型、transformers backend、sequence classification backend 等；提供 `get_model_architecture`、`get_resolved_model_impl`、`post_load_weights`。
- `weight_utils.py`：权重下载、safetensors/pt/gguf/RunAI 迭代器、分片 loader、默认 loader、KV cache scale loader、dummy weights 初始化等。
- `remote_instance_weight_loader_utils.py`：remote instance 权重传输、memory region 注册和 transfer engine 信息获取。
- `ci_weight_validation.py`：CI/缓存场景的下载完整性校验和坏缓存清理。
- `__init__.py`：对外 `get_model` 入口。

## 运行流程

`ModelRunner` 先通过 `configs.model_config.ModelConfig` 取得模型架构和 load config，再调用 `get_model_loader(load_config)` 选择 loader。默认路径会用 `utils.get_model_architecture` 找到 `models` 目录中的原生类，初始化模型 skeleton，再通过 `weight_utils` 的迭代器逐个读取 tensor，并调用参数对象或默认 loader 完成 TP/PP/量化分片加载。加载后执行 `post_load_weights`，再进入 CUDA graph、KV cache、sampler 等初始化。

## 依赖关系

该模块向上被 `model_executor.model_runner` 使用，向下依赖 `models` registry、`configs`、`layers.parameter`/量化方法、`utils.runai_utils`、HuggingFace/safetensors/torch load 生态以及 remote instance/disaggregation transfer 相关工具。

## 设计要点和风险

- loader 与模型参数命名强耦合；新增模型时 `load_weights`、stacked weight mapping、KV scale 名称映射要与 `weight_utils` 规则对齐。
- 量化 loader 不只是 dtype 转换，还涉及 layout、scale、zero point、packed weight、Marlin/GGUF/bitsandbytes 等差异。
- remote/streaming loader 会改变权重可用时序，必须和 `ModelRunner` 的同步、memory registration 和错误处理配合。
- CI validation 侧重缓存完整性，不等同于模型语义正确；运行时仍需 loader 处理缺失/额外 tensor。
