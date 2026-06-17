# SGLang `python/sglang/srt` 模块分析

本文档面向 `python/sglang/srt`，按顶层 Python 模块/子包梳理 SGLang Runtime 的实现。这里的“模块”采用当前目录的一层边界：`srt` 下的顶层 `.py` 文件和顶层子包；每篇文档再覆盖其内部关键子模块。

## 阅读顺序

1. [00_srt_architecture.md](./00_srt_architecture.md) - SRT 总体架构、请求链路和模块分层
2. [srt_server_args.md](./srt_server_args.md) - 服务参数、后端选择和运行时配置入口
3. [srt_entrypoints.md](./srt_entrypoints.md) - Python API、HTTP/OpenAI/Ollama/Anthropic 入口
4. [srt_managers.md](./srt_managers.md) - Tokenizer、Scheduler、Detokenizer 和批调度核心
5. [srt_model_executor.md](./srt_model_executor.md) - ModelRunner、ForwardBatch、CUDA graph 和实际执行
6. [srt_mem_cache.md](./srt_mem_cache.md) - KV cache、Radix cache、HiCache、Mamba/SWA cache
7. [srt_layers.md](./srt_layers.md) - Attention、Linear、MoE、量化、采样等模型层
8. [srt_models.md](./srt_models.md) - 模型实现与注册体系
9. [srt_model_loader.md](./srt_model_loader.md) - 权重下载、格式适配、分片加载与远端加载

## 模块索引

| 模块 | 文档 | 主要职责 |
| --- | --- | --- |
| `constants.py` | [srt_constants.md](./srt_constants.md) | 全局常量、特殊 rid、内存类型标签 |
| `environ.py` | [srt_environ.md](./srt_environ.md) | 环境变量声明、默认值和类型化读取 |
| `server_args.py` | [srt_server_args.md](./srt_server_args.md) | `ServerArgs` / `PortArgs`，服务启动参数和运行时校验 |
| `server_args_config_parser.py` | [srt_server_args_config_parser.md](./srt_server_args_config_parser.md) | 配置文件到启动参数的解析 |
| `batch_invariant_ops` | [srt_batch_invariant_ops.md](./srt_batch_invariant_ops.md) | 与 batch 大小无关的算子包装 |
| `batch_overlap` | [srt_batch_overlap.md](./srt_batch_overlap.md) | prefill/decode batch overlap 的操作编排 |
| `checkpoint_engine` | [srt_checkpoint_engine.md](./srt_checkpoint_engine.md) | 运行中 checkpoint/权重更新工作进程 |
| `compilation` | [srt_compilation.md](./srt_compilation.md) | `torch.compile`、piecewise graph、FX pass 和 split op |
| `configs` | [srt_configs.md](./srt_configs.md) | HF 配置扩展、设备/加载配置、模型配置修正 |
| `connector` | [srt_connector.md](./srt_connector.md) | Redis/S3/remote instance 等外部连接器 |
| `constrained` | [srt_constrained.md](./srt_constrained.md) | 结构化输出 grammar 后端和跳转前进 |
| `debug_utils` | [srt_debug_utils.md](./srt_debug_utils.md) | dump、对比、截断、日志解析和调度模拟 |
| `disaggregation` | [srt_disaggregation.md](./srt_disaggregation.md) | prefill/decode/encoder 分离和 KV 传输 |
| `distributed` | [srt_distributed.md](./srt_distributed.md) | TP/PP/DP/EP 进程组和通信原语 |
| `dllm` | [srt_dllm.md](./srt_dllm.md) | Diffusion LLM / D-LLM 调度混入和算法 |
| `elastic_ep` | [srt_elastic_ep.md](./srt_elastic_ep.md) | 弹性专家并行与 expert backup |
| `entrypoints` | [srt_entrypoints.md](./srt_entrypoints.md) | Engine、HTTP server、OpenAI/Ollama/Anthropic API |
| `eplb` | [srt_eplb.md](./srt_eplb.md) | Expert Parallel Load Balancing |
| `function_call` | [srt_function_call.md](./srt_function_call.md) | Tool/function call 增量检测和格式解析 |
| `grpc` | [srt_grpc.md](./srt_grpc.md) | gRPC 入口占位和 Rust/proto 对接边界 |
| `hardware_backend` | [srt_hardware_backend.md](./srt_hardware_backend.md) | NPU/MLX 等非 CUDA 后端适配 |
| `layers` | [srt_layers.md](./srt_layers.md) | 模型层、attention 后端、MoE、量化与采样 |
| `lora` | [srt_lora.md](./srt_lora.md) | LoRA 注册、内存池、加载、层替换与 batch 信息 |
| `managers` | [srt_managers.md](./srt_managers.md) | 请求对象、调度、tokenizer/detokenizer、DP 控制 |
| `mem_cache` | [srt_mem_cache.md](./srt_mem_cache.md) | KV 内存池、prefix cache、HiCache、sparsity |
| `model_executor` | [srt_model_executor.md](./srt_model_executor.md) | 模型执行、CUDA graph、forward metadata、hook |
| `model_loader` | [srt_model_loader.md](./srt_model_loader.md) | 模型架构解析、权重迭代器和 loader 实现 |
| `models` | [srt_models.md](./srt_models.md) | 各模型族的 SRT 原生实现 |
| `multimodal` | [srt_multimodal.md](./srt_multimodal.md) | 图像/视频/音频输入处理、视觉图捕获、EVS |
| `multiplex` | [srt_multiplex.md](./srt_multiplex.md) | prefill/decode multiplex 上下文和 scheduler mixin |
| `observability` | [srt_observability.md](./srt_observability.md) | metrics、trace、request timing、CPU monitor |
| `parser` | [srt_parser.md](./srt_parser.md) | chat template、reasoning、Harmony、conversation 解析 |
| `ray` | [srt_ray.md](./srt_ray.md) | Ray 入口、scheduler actor、HTTP server 封装 |
| `sampling` | [srt_sampling.md](./srt_sampling.md) | SamplingParams、batch sampling metadata、penaltylib |
| `speculative` | [srt_speculative.md](./srt_speculative.md) | EAGLE、standalone、N-gram speculative decoding |
| `tokenizer` | [srt_tokenizer.md](./srt_tokenizer.md) | tokenizer 扩展，目前主要是 tiktoken 包装 |
| `utils` | [srt_utils.md](./srt_utils.md) | 平台检测、日志、网络、profile、offload、补丁工具 |
| `weight_sync` | [srt_weight_sync.md](./srt_weight_sync.md) | 权重同步 bucket 和张量传输工具 |

## 范围说明

`python/sglang/srt` 内含大量模型文件、Triton kernel、量化配置 JSON 和平台特化实现。本文档不逐个展开所有 JSON/每个模型文件，而是在对应顶层模块文档中说明共同结构、核心类、关键运行链路和需要重点阅读的代表文件。
