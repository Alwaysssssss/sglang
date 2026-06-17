# `python/sglang/srt/entrypoints` 模块分析

## 定位

`entrypoints` 是 SRT 的协议入口层。它既提供进程内 Python API `Engine`，也提供 HTTP server、OpenAI-compatible API、Ollama API、Anthropic API、gRPC server 入口以及一些模板/工具调用辅助逻辑。它不直接执行模型 forward，而是负责把外部请求转换为 `managers` 定义的内部请求对象，并把输出转换回协议响应。

## 关键文件

- `engine.py`：核心 Python API。`Engine` 解析 `ServerArgs`，启动 tokenizer/scheduler/detokenizer 子进程，建立 ZMQ RPC socket，并提供 generate/embedding/score/update-weight/LoRA/RPC 等方法。
- `http_server.py`：FastAPI 服务启动入口，创建全局状态，注册路由，初始化 OpenAI/Ollama/Anthropic serving 对象，处理 warmup、健康检查、metrics、admin API。
- `http_server_engine.py`：把 HTTP server 包成 `EngineBase` 兼容的外部进程适配器。
- `openai/`：OpenAI protocol Pydantic 模型和 serving 实现，覆盖 chat/completions/embeddings/rerank/score/classify/tokenize/responses/transcription。
- `ollama/`、`anthropic/`：协议模型和 serving 适配。
- `context.py`、`harmony_utils.py`、`tool.py`：对 Harmony 格式、tool 执行上下文和消息渲染做协议辅助。
- `grpc_server.py`：Python 侧 gRPC server 入口。
- `engine_info_bootstrap_server.py`：为 transfer/disaggregation 场景暴露 engine 信息。

## 运行流程

`Engine.__init__` 是最短主线：构造 `ServerArgs` -> 注册 shutdown -> `_launch_subprocesses` -> 初始化 `TokenizerManager` 和 template -> 等待 scheduler ready -> 创建 RPC DEALER socket -> 可选 tracing。实际用户调用会落到 tokenizer manager，scheduler 进程通过 IPC 接收 tokenized batch，detokenizer 再把结果返回主进程。

HTTP 模式多一层 FastAPI：`launch_server` 先准备 `ServerArgs` 和 `PortArgs`，再初始化 `Engine`/tokenizer manager，并把不同路由交给 OpenAI/Ollama/Anthropic serving 类。OpenAI 子模块的重点是把协议字段映射成 `GenerateReqInput`、`EmbeddingReqInput`、`TokenizeRequest` 等内部结构，同时处理 streaming、usage、logprobs、tool calls、reasoning 内容。

## 依赖关系

入口层向下依赖 `server_args`、`managers.io_struct`、`TokenizerManager`、`TemplateManager`、`scheduler` 进程函数、`detokenizer_manager`、`observability.trace`、`utils.network`。它也直接消费 `parser` 和 `function_call` 能力，但尽量不理解模型层细节。

## 设计要点和风险

- `Engine` 同时是用户 API 和进程编排器，启动参数、端口、子进程清理、watchdog 都集中在这里。
- `openai/protocol.py` 模型很多，是协议兼容性的主要维护点；新增 API 字段时要确认内部 `GenerateReqInput` 是否能表达。
- streaming 输出要同时处理 detokenizer 状态、finish reason、usage 和 tool/reasoning 增量解析，容易出现边界不一致。
- 多协议共享同一个 tokenizer/scheduler 后端，入口层应避免引入协议特有状态污染内部请求。
