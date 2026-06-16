# Qwen3.6-27B Agent SGLang 子模块文档索引

本目录按 `sglang_qwen36_27b_agent_call_flow.md` 的端到端方案，把 Qwen3.6-27B Agent 在 SGLang 内部的调用链拆成可单独阅读的子模块文档。拆分时使用多个只读 subagent 分别核查启动拓扑、OpenAI 请求、scheduler/cache、模型执行、日志排查链路，主线程统一整合并按当前 worktree 校正脚本默认值。

当前 `docs_always/qwen3.6-27b/start_qwen36_27b_agent.sh` 的关键默认值以当前文件为准：

| 项 | 当前默认值 |
| --- | --- |
| `SERVED_MODEL_NAME` | `qwen3.6-27b` |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` |
| `TP_SIZE` | `4` |
| `CONTEXT_LENGTH` | `262144` |
| `MAX_OUTPUT_TOKENS` | `128000` |
| `TOOL_CALL_PARSER` | `qwen3_coder` |
| `SCHEDULE_POLICY` | `lpm` |
| `RADIX_EVICTION_POLICY` | `lru` |
| `LOG_REQUESTS_LEVEL` | `3` |

## 阅读顺序

1. `sglang_qwen36_27b_agent_call_flow.md`
   - 端到端总览，从启动脚本一路到 OpenAI JSON/SSE 响应。

2. `sglang_qwen36_27b_bootstrap_serverargs_topology.md`
   - 启动脚本默认值、显存/并发推导、`launch_server` 分支、`ServerArgs`、Engine 子进程拓扑和 ZMQ/IPC。

3. `sglang_qwen36_27b_openai_tokenizer_toolcalls.md`
   - `/v1/chat/completions`、鉴权、`ChatCompletionRequest`、chat template、`TokenizerManager`、`qwen3_coder` tool call parser。

4. `sglang_qwen36_27b_scheduler_prefill_radix_cache.md`
   - Scheduler 入队、`waiting_queue`、`PrefillAdder`、chunked prefill、decode retract、`lpm`、radix KV cache。

5. `sglang_qwen36_27b_model_tp_forward_output.md`
   - Qwen3.6 config 到 SGLang Qwen3.5 架构映射、TP worker、`ModelRunner`、FlashInfer attention/sampling、输出回传。

6. `sglang_qwen36_27b_observability_troubleshooting.md`
   - 启动日志、server log、request log、request-time stats、Prometheus `/metrics`、验证命令和故障分类排查。

7. `sglang_qwen36_27b_server_args_and_request_schema.md`
   - 更完整的 `ServerArgs` 参数表和 `/v1/chat/completions` 请求体结构说明。

## 端到端链路缩写

```text
start_qwen36_27b_agent.sh
  -> python -m sglang.launch_server
  -> prepare_server_args()
  -> http_server.launch_server()
  -> Engine._launch_subprocesses()
     -> Scheduler x TP_SIZE
     -> DetokenizerManager
     -> TokenizerManager
  -> FastAPI /v1/chat/completions
  -> OpenAIServingChat
  -> TokenizerManager.generate_request()
  -> Scheduler waiting_queue / prefill / decode
  -> ModelRunner.forward()
  -> Sampler.forward()
  -> DetokenizerManager
  -> TokenizerManager._wait_one_response()
  -> OpenAI JSON or SSE
```
