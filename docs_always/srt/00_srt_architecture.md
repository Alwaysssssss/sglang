# `python/sglang/srt` 架构总览

`srt` 是 SGLang 的在线 LLM 推理运行时。它把 HTTP/OpenAI/Ollama/Anthropic API、tokenizer、动态 batch scheduler、模型执行、KV cache、采样、结构化输出、LoRA、多模态、分布式并行、prefill/decode 分离和观测能力整合到一个多进程服务架构里。

## 进程与请求主线

典型请求链路是：

`entrypoints/http_server.py` 或 `entrypoints/engine.py`
-> `TokenizerManager`
-> `Scheduler`
-> `TpModelWorker` / `ModelRunner`
-> `ScheduleBatch` / `ForwardBatch`
-> `layers` / `models`
-> `Sampler`
-> `DetokenizerManager`
-> API response。

`Engine` 文档里明确了三组件结构：主进程里的 `TokenizerManager`，scheduler 子进程，以及 detokenizer 子进程。它们之间通过 ZMQ IPC 通信，主进程还负责 HTTP server、模板管理、RPC 控制、生命周期清理和 tracing 初始化。

## 分层结构

| 层 | 代表模块 | 职责 |
| --- | --- | --- |
| 入口层 | `entrypoints`, `server_args`, `parser`, `function_call` | 接收请求、解析协议、构造 `ServerArgs`/请求对象、处理 chat/template/tool/reasoning |
| 管理层 | `managers` | tokenizer、scheduler、detokenizer、batch 对象、会话、DP 控制、请求输入/输出结构 |
| 执行层 | `model_executor`, `model_loader`, `models` | 初始化分布式环境，加载模型和权重，构建 forward batch，执行模型 forward |
| 算子层 | `layers`, `sampling`, `compilation`, `batch_overlap` | attention、linear、MoE、量化、sampler、CUDA graph、`torch.compile`、overlap 编排 |
| 状态层 | `mem_cache`, `lora`, `speculative`, `constrained` | KV cache、prefix cache、LoRA adapter、speculative decoding、grammar 状态 |
| 横向能力 | `distributed`, `disaggregation`, `observability`, `utils`, `debug_utils` | 并行通信、PD 分离、metrics/trace、平台工具、调试工具 |

## 关键数据结构

- `ServerArgs`：服务级配置入口，集中决定模型路径、并行度、attention backend、sampling backend、grammar backend、LoRA、speculative、disaggregation、quantization、HTTP 行为等。
- `GenerateReqInput` / `TokenizedGenerateReqInput`：入口请求结构，定义从 API 到 tokenizer/scheduler 的边界。
- `Req`：scheduler 内部的单请求状态，持有 token、采样参数、KV cache 长度、完成原因、多模态信息、grammar 状态等。
- `ScheduleBatch`：scheduler 侧批对象，负责把请求组织成 prefill/decode/extend 可执行 batch。
- `ModelWorkerBatch`：scheduler 发送给 worker 的批结构。
- `ForwardBatch`：model executor 和 attention/backend 使用的执行元数据，包含 forward mode、positions、seq lens、cache loc、attention backend metadata 等。
- `SamplingBatchInfo`：采样阶段的批量参数与惩罚状态。

## 关键控制面

1. `server_args.py` 先把 CLI/API 参数归一化，并基于硬件、模型配置和用户选项推导默认值。
2. `Engine._launch_subprocesses` 根据 `ServerArgs` 和 `PortArgs` 启动 tokenizer、scheduler、detokenizer 及可选 data-parallel controller。
3. `Scheduler` 接收 tokenized request，结合 `SchedulePolicy`、prefix cache、grammar、LoRA、speculative、disaggregation 等状态决定每一轮 forward。
4. `TpModelWorker` 包装一个或多个 `ModelRunner`，负责真正的 rank-local 模型执行。
5. `ModelRunner` 初始化分布式进程组、加载模型、创建 KV cache、选择 attention backend、启用 CUDA graph/compile/offload/LoRA/EPLB。
6. `DetokenizerManager` 把 token 输出转回字符串，并维护流式输出、finish reason、logprobs 等上层响应信息。

## 设计特点

- **参数中心化**：绝大多数后端选择、特性开关和平台差异先进入 `ServerArgs`，再被下游模块读取，降低入口分散度。
- **多进程隔离**：tokenizer、scheduler、detokenizer 独立运行，避免模型执行阻塞协议层，同时用 watchdog 和 IPC 做生命周期管理。
- **批调度为核心**：`Scheduler` 是最密集的控制点，所有 prefix cache、chunked prefill、grammar、speculative、LoRA、disagg 都围绕 batch 调度集成。
- **后端可替换**：attention、sampling、MoE、LoRA、transfer backend、grammar backend 都通过 registry/choices/adapter 实现多后端。
- **性能路径多层叠加**：CUDA graph、piecewise compile、batch overlap、DP attention、radix cache、HiCache、FlashInfer/FA3/FlashMLA/DeepGEMM 等都可组合，但也带来配置相容性风险。

## 阅读建议

抓主线时建议按下面顺序读：

1. `server_args.py`
2. `entrypoints/engine.py`
3. `entrypoints/http_server.py`
4. `managers/io_struct.py`
5. `managers/tokenizer_manager.py`
6. `managers/scheduler.py`
7. `managers/schedule_batch.py`
8. `model_executor/model_runner.py`
9. `model_executor/forward_batch_info.py`
10. `mem_cache/memory_pool.py` 和 `mem_cache/radix_cache.py`
11. `layers/radix_attention.py`、`layers/linear.py`、`layers/moe/*`
12. 一个代表模型，如 `models/llama.py` 或 `models/deepseek_v2.py`
