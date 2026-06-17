# `python/sglang/srt/server_args.py` 模块分析

## 定位

`server_args.py` 是 SRT 服务级配置入口。`ServerArgs` 聚合模型、HTTP、SSL、量化、内存调度、并行、LoRA、kernel backend、speculative、expert parallel、disaggregation、observability、debug 等几乎所有运行时参数；`PortArgs` 则负责内部 IPC/服务端口规划。

## 关键内容

- backend choice 常量：`LOAD_FORMAT_CHOICES`、`QUANTIZATION_CHOICES`、`ATTENTION_BACKEND_CHOICES`、`GRAMMAR_BACKEND_CHOICES`、`MOE_RUNNER_BACKEND_CHOICES` 等。
- `add_*_choices` 函数：允许外部扩展 load format、attention backend、grammar backend、MoE/GEMM backend 等选项。
- `ServerArgs`：dataclass 主体。字段按模型、HTTP、量化、内存调度、runtime、日志、API、DP/multinode、LoRA、kernel backend、speculative、EP、disaggregation 等分组。
- `PortArgs`：派生 tokenizer/scheduler/detokenizer/RPC 等内部端口和 IPC 名称。
- 全局 server args helper：让 scheduler/model runner 等下游模块访问当前配置。

## 运行流程

CLI 或 Python API 先构造 `ServerArgs`。字段初始化后，后续校验/派生逻辑会结合硬件、模型配置、环境变量和用户显式选项推导默认后端，例如 attention backend、sampling backend、grammar backend、LoRA backend、disaggregation backend、KV cache dtype、parallelism 参数等。`Engine` 使用它启动子进程；`ModelConfig.from_server_args` 和 `ModelRunner` 再读取其中的模型、dtype、并行、backend 与 cache 参数。

## 依赖关系

`server_args.py` 依赖 `environ`、`connector`、`function_call`、`parser`、`lora`、`utils.common`、`utils.network`、`utils.runai_utils` 等。它向下被 `entrypoints`、`managers`、`model_executor`、`configs`、`layers` 几乎全量使用。

## 设计要点和风险

- `ServerArgs` 字段顺序要求与 `add_cli_args` 保持一致；新增字段时要同步 CLI、校验、序列化和文档。
- choices 常量是用户可见 API；扩展后端时要加入对应 choice、默认推导、实际实现和错误提示。
- `ServerArgs` 的默认值会被很多模块隐式依赖；修改默认值比局部代码修改影响面更大。
- 参数相容性是主要风险，例如 speculative + chunked prefill + LoRA + disaggregation + CUDA graph 的组合，需要明确禁用或测试。
