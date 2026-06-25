# Qwen3.6-27B 模型加载、TP 执行与输出回传

本文覆盖 Qwen3.6-27B 在 SGLang 中的模型配置映射、TP worker、`ModelRunner`、FlashInfer attention/sampling、forward、sampling、detokenizer 回传链路。

## 1. 当前启动参数

当前脚本默认：

```bash
MODEL_PATH=/mnt/shanhai-ai/wenhy/models/Qwen/Qwen/Qwen3___6-27B
SERVED_MODEL_NAME=qwen3.6-27b
CUDA_VISIBLE_DEVICES=0,1,2,3
TP_SIZE=4
CONTEXT_LENGTH=262144
DTYPE=bfloat16
ATTENTION_BACKEND=flashinfer
SAMPLING_BACKEND=flashinfer
SCHEDULE_POLICY=lpm
RADIX_EVICTION_POLICY=lru
DISABLE_PIECEWISE_CUDA_GRAPH=1
```

默认值在 `docs_always/qwen3.6-27b/start_qwen36_27b_agent.sh:4-49`，server 命令在 `start_qwen36_27b_agent.sh:282-350`。脚本未显式设置 DP/PP，因此无额外 `EXTRA_SERVER_ARGS` 时通常是 `dp_size=1`、`pp_size=1`、`tp_size=4`。

## 2. 本地 config 到 Qwen3.5 架构映射

本地模型 config 路径：

```text
/mnt/shanhai-ai/wenhy/models/Qwen/Qwen/Qwen3___6-27B/config.json
```

关键字段：

```json
{
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "model_type": "qwen3_5",
  "language_model_only": false,
  "text_config": {
    "model_type": "qwen3_5_text",
    "dtype": "bfloat16",
    "max_position_embeddings": 262144,
    "num_hidden_layers": 64,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "full_attention_interval": 4
  }
}
```

当前 `text_config.layer_types` 是 48 个 `linear_attention` 和 16 个 `full_attention`。按 `full_attention_interval=4`，0-based full-attention 层是 `3, 7, 11, ..., 63`。

SGLang config：

- `Qwen3_5Config` / `Qwen3_5TextConfig`：`python/sglang/srt/configs/qwen3_5.py`
- `Qwen3NextConfig.layers_block_type`：按 full attention interval 生成层类型，见 `configs/qwen3_next.py`
- `Qwen3_5ForConditionalGeneration`：`python/sglang/srt/models/qwen3_5.py:1312-1324`
- `Qwen3_5ForCausalLM`：`qwen3_5.py:881-1023`
- `EntryClass`：`qwen3_5.py:1724`，供 `ModelRegistry` 收集

模型类解析路径：

```text
get_model_architecture()
  -> model_config.hf_config.architectures
  -> "Qwen3_5ForConditionalGeneration"
  -> ModelRegistry.resolve_model_cls()
```

见 `python/sglang/srt/model_loader/utils.py:193-228`。

## 3. TP rank 与 GPU 创建链路

启动入口：

```text
start_qwen36_27b_agent.sh
  -> python -m sglang.launch_server
  -> launch_server.run_server()
  -> http_server.launch_server()
  -> Engine._launch_subprocesses()
  -> Engine._launch_scheduler_processes()
```

关键位置：

- `python/sglang/launch_server.py:59-62`：解析并运行 server。
- `launch_server.py:43-47`：默认进入 HTTP server。
- `http_server.py:2135-2157`：说明 SRT engine 由 TokenizerManager、Scheduler、DetokenizerManager 组成。
- `http_server.py:2158-2170`：调用 `Engine._launch_subprocesses()`。

TP scheduler 子进程创建在 `python/sglang/srt/entrypoints/engine.py:514-579`：

```text
for pp_rank in pp_rank_range:
  for tp_rank in tp_rank_range:
    gpu_id = base_gpu_id
             + ((pp_rank % pp_size_per_node) * tp_size_per_node)
             + (tp_rank % tp_size_per_node) * gpu_id_step
    mp.Process(target=run_scheduler_process, args=(..., gpu_id, tp_rank, ...))
```

当前单节点、`pp_size=1`、`tp_size=4` 时，`tp_rank` 通常是 `0..3`，`gpu_id` 通常是 `0..3`。

`run_scheduler_process()` 在 `python/sglang/srt/managers/scheduler.py:3560-3616`，把 `gpu_id`、`tp_rank`、`pp_rank` 传入 `Scheduler`。

## 4. Scheduler 到 TpModelWorker / ModelRunner

`Scheduler.__init__()` 初始化主线在 `scheduler.py:362-424`：

```text
init_model_config()
init_ipc_channels()
init_tokenizer()
init_moe_gemm_config()
init_mamba_backend()
init_model_worker()
init_cache_with_memory_pool()
init_running_status()
init_chunked_prefill()
init_schedule_policy()
init_overlap()
```

`Scheduler.init_model_worker()` 在 `scheduler.py:631-683`：

```text
init_tp_model_worker()
  -> TpModelWorker(...)
maybe_init_draft_worker()
self.model_worker = self.tp_worker
```

`TpModelWorker.__init__()` 在 `python/sglang/srt/managers/tp_worker.py:221-262` 保存 `server_args`、`tp_size`、`tp_rank`、`gpu_id`、`pp_rank` 等，然后：

```text
_init_model_config()
_init_model_runner()
```

`_init_model_config()` 调 `ModelConfig.from_server_args()`，见 `tp_worker.py:323-339`。`_init_model_runner()` 构造 `ModelRunner(...)`，见 `tp_worker.py:341-362`。

`ModelRunner.__init__()` 在 `python/sglang/srt/model_executor/model_runner.py:286-428` 保存 `mem_fraction_static`、`device`、`gpu_id`、`tp_rank`、`tp_size`、`server_args` 等，并调用 `init_torch_distributed()`：

- 设置 CUDA device。
- 初始化 distributed environment。
- 初始化 tensor/pipeline/expert/context parallel groups。
- 最后调用 `initialize(pre_model_load_memory)`。

## 5. 模型加载

`ModelRunner.initialize()` 在 `model_runner.py:461-675`：

```text
create_sampler()
load_model()
configure_kv_cache_dtype()
init_memory_pool(pre_model_load_memory)
init_attention_backend()
kernel_warmup()
init_device_graphs()
init_piecewise_cuda_graphs()
```

`ModelRunner.load_model()` 在 `model_runner.py:1071-1172`：

```text
LoadConfig(...)
get_model_loader(load_config, model_config)
loader.load_model(model_config=..., device_config=DeviceConfig(...))
```

`get_model_loader()` 在 `python/sglang/srt/model_loader/loader.py:3105-3192`。当前脚本未设置特殊 `--load-format` 时 fallback 是 `DefaultModelLoader`。

默认 loader：

```text
DefaultModelLoader.load_model()
  -> _get_quantization_config()
  -> set_default_torch_dtype(model_config.dtype)
  -> _initialize_model()
     -> get_model_architecture()
     -> Qwen3_5ForConditionalGeneration(...)
  -> load_weights_and_postprocess()
     -> model.load_weights(weights)
```

Qwen3.6 权重加载方法：

- `Qwen3_5ForConditionalGeneration.load_weights()`：`python/sglang/srt/models/qwen3_5.py:1361-1395`
- `Qwen3_5ForCausalLM.load_weights()`：`qwen3_5.py:1023-1060`

这些方法处理 `q_proj/k_proj/v_proj -> qkv_proj`、`gate_proj/up_proj -> gate_up_proj`、GDN 投影、`language_model` 到 `model` 的命名兼容等。

## 6. KV / memory pool

`ModelRunner` 继承 `ModelRunnerKVCacheMixin`，见：

- `python/sglang/srt/model_executor/model_runner.py:286`
- `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`

关键步骤：

- `profile_max_num_token()`：用可用显存和每 token KV cell size 推导 token capacity。
- `get_cell_size_per_token()`：按 KV dtype、KV head 数、head dim、有效层数估算每 token KV。
- `_resolve_token_capacity()`：处理 `--max-total-tokens` 和 page 对齐。
- `_resolve_max_num_reqs()`：推导并发请求上限。
- `_init_pools()`：创建 request pool、KV pool、allocator。

Qwen3.6 / Qwen3.5 hybrid GDN 的 pool 形态：

- `ModelRunner.hybrid_gdn_config` 识别 Qwen3.5 config。
- `mambaish_config` 会返回 hybrid GDN config。
- `_init_pools()` 对 hybrid 模型创建 `HybridReqToTokenPool`。
- hybrid linear/GDN 模型使用 `HybridLinearKVPool`。

Scheduler 侧通过 `Scheduler.init_cache_with_memory_pool()` 获取 memory pool，再创建 `MambaRadixCache` 或 `RadixCache`，见 `scheduler.py:704-818`。

## 7. FlashInfer attention

启动参数：

```bash
--attention-backend flashinfer
```

`ServerArgs.get_attention_backends()` 会返回 prefill/decode backend；当前脚本显式传入 `flashinfer`，不走默认选择。

`ModelRunner.init_attention_backend()` 在 `model_runner.py:1959-1971`：

```text
_get_attention_backend()
  -> server_args.get_attention_backends()
  -> _get_attention_backend_from_str("flashinfer")
  -> ATTENTION_BACKENDS["flashinfer"](runner)
  -> attn_backend_wrapper(runner, full_attention_backend)
```

FlashInfer 注册表在 `python/sglang/srt/layers/attention/attention_registry.py:12-45`。Qwen3.6 不是 MLA 路径，full-attention 层使用 `FlashInferAttnBackend`。

hybrid GDN wrapper 在 `attention_registry.py:180-235`：

- `runner.mambaish_config` 存在时构造 linear attention backend。
- hybrid GDN 创建 `GDNAttnBackend`。
- 最终返回 `HybridLinearAttnBackend(full_attn_backend, linear_attn_backend, full_attn_layers)`。

因此当前模型：

- full-attention 层使用 FlashInfer。
- linear/GDN 层使用 GDN backend。
- `HybridLinearAttnBackend` 根据 full-attention layer ids 选择后端。

## 8. FlashInfer sampling

启动参数：

```bash
--sampling-backend flashinfer
```

`ModelRunner.initialize()` 调 `create_sampler()`，见 `model_runner.py:505-507`。`create_sampler()` 在 `python/sglang/srt/layers/sampler.py:426-445`，`flashinfer` 走内置 `Sampler`。

`Sampler.forward()` 在 `sampler.py:77-187`：

- 全 greedy：直接 `torch.argmax`。
- 非 greedy：temperature 缩放、softmax，再 `_sample_from_probs()`。
- simple sampling case 可能仍走 torch。
- 需要 top-k/top-p/min-p 且 backend 是 `flashinfer` 时，走 FlashInfer sampling kernel。
- `_sync_token_ids_across_tp()` 在特殊场景同步 TP token ids。

结论：`--sampling-backend flashinfer` 主要影响非 greedy 且需要 top-k/top-p/min-p 的采样路径；全 greedy 不触发 FlashInfer sampling kernel。

## 9. `--disable-piecewise-cuda-graph`

当前脚本默认：

```bash
DISABLE_PIECEWISE_CUDA_GRAPH=1
--disable-piecewise-cuda-graph
```

脚本追加位置在 `start_qwen36_27b_agent.sh:340-342`。

`ModelRunner.init_piecewise_cuda_graphs()` 在 `model_runner.py:2464-2569`。如果 `server_args.disable_piecewise_cuda_graph` 为真，`piecewise_cuda_graph_runner = None` 并直接返回。

含义：

- 只关闭分段 prefill/extend 图优化路径。
- 不等于关闭普通 CUDA graph。
- 普通 decode CUDA graph 由 `--disable-cuda-graph` 等参数控制，入口是 `init_device_graphs()`。

## 10. `Scheduler.run_batch` 到 forward / sampling

`Scheduler.run_batch()` 在 `python/sglang/srt/managers/scheduler.py:2636-2735`：

```text
batch.get_model_worker_batch()
self.model_worker.forward_batch_generation(...)
batch.output_ids = next_token_ids or future_indices
```

`TpModelWorker.forward_batch_generation()` 在 `python/sglang/srt/managers/tp_worker.py:444-522`：

```text
ForwardBatch.init_new(model_worker_batch, self.model_runner)
self.model_runner.forward(forward_batch)
self.model_runner.sample(logits_output, forward_batch)
GenerationBatchResult(...)
```

`ModelRunner.forward()` 在 `model_runner.py:2725-2782`，内部调用 `_forward_raw()` 并按 `forward_batch.forward_mode` 分支：

- `forward_decode()`
- `forward_split_prefill()`
- `forward_extend()`
- `forward_idle()`

Qwen 模型 forward：

- `Qwen3_5ForCausalLM.forward()`：`python/sglang/srt/models/qwen3_5.py:959-1021`
- full-attention layer：`qwen3_5.py:822-872`
- linear/GDN layer：`qwen3_5.py:569-585`

sampling：

```text
ModelRunner.sample()
  -> _preprocess_logits()
  -> self.sampler(...)
  -> Sampler.forward()
  -> next_token_ids
```

见 `model_runner.py:2884-2921` 和 `sampler.py:77-187`。

## 11. GenerationBatchResult 到 detokenizer

`GenerationBatchResult` 定义在 `python/sglang/srt/managers/utils.py`，包含 logits、`next_token_ids`、CUDA graph 能力、可选 delayed sample function、expert/spec metrics。

Scheduler 结果处理：

```text
Scheduler.process_batch_result()
  -> process_batch_result_prefill() or process_batch_result_decode()
  -> stream_output()
  -> stream_output_generation()
  -> BatchTokenIDOutput
  -> send_to_detokenizer
```

关键代码：

- `process_batch_result()`：`scheduler.py:2811-2832`
- prefill 处理：`scheduler_output_processor_mixin.py:123-330`
- decode 处理：`scheduler_output_processor_mixin.py:374-555`
- `stream_output_generation()`：`scheduler_output_processor_mixin.py:924-1201`

prefill 结果会 append 首 token、检查 finished、释放或 cache KV，然后 stream。decode 结果会 append next token、更新 metrics/reasoning、检查 finished、释放 KV，然后 stream。

`BatchTokenIDOutput` 定义在 `python/sglang/srt/managers/io_struct.py:961-1025`，包含 rids、finish reasons、decode ids、output ids、token counts、logprobs、hidden states、cache details、time stats 等。

## 12. DetokenizerManager 回传

`DetokenizerManager` 在 `python/sglang/srt/managers/detokenizer_manager.py:73-145`：

```text
recv_from_scheduler = PULL detokenizer_ipc_name
send_to_tokenizer = PUSH tokenizer_ipc_name
event_loop()
  -> recv_from_scheduler.recv_pyobj()
  -> _request_dispatcher(recv_obj)
  -> send_to_tokenizer.send_pyobj(output)
```

`BatchTokenIDOutput` 由 `handle_batch_token_id_out()` 处理：

```text
_decode_batch_token_id_output()
  -> DecodeStatus
  -> tokenizer.batch_decode/decode
  -> incremental output_strs
BatchStrOutput(...)
```

见 `detokenizer_manager.py:216-365`。Detokenizer 会维护 `read_offset` / `sent_offset`，避免半个 unicode 或不可打印文本提前发出。

## 13. TokenizerManager 到 OpenAI response

TokenizerManager 接收回包：

```text
TokenizerManager.handle_loop()
  -> recv_from_detokenizer.recv_pyobj()
  -> _result_dispatcher(...)
  -> _handle_batch_output(...)
  -> state.out_list.append(out_dict)
  -> state.event.set()
```

见 `python/sglang/srt/managers/tokenizer_manager.py:1523-1687`。

`_handle_batch_output()` 对 `BatchStrOutput` 构造：

```python
{
  "text": output_text,
  "output_ids": output_token_ids,
  "meta_info": {
    "id": rid,
    "finish_reason": ...,
    "prompt_tokens": ...,
    "completion_tokens": ...,
    "reasoning_tokens": ...,
    "cached_tokens": ...,
    "weight_version": ...,
    ...
  }
}
```

OpenAI streaming：

```text
OpenAIServingChat._handle_streaming_request()
  -> _generate_chat_stream()
  -> async for content in tokenizer_manager.generate_request(...)
  -> ChatCompletionStreamResponse
  -> StreamingResponse(text/event-stream)
```

见 `python/sglang/srt/entrypoints/openai/serving_chat.py:604-915`。

OpenAI non-streaming：

```text
OpenAIServingChat._handle_non_streaming_request()
  -> await tokenizer_manager.generate_request(...).__anext__()
  -> _build_chat_response()
  -> ChatCompletionResponse
```

见 `serving_chat.py:917-1050`。如果请求包含 tools 且启用 `tool_call_parser`，`_process_tool_calls()` 会把文本解析成 OpenAI tool calls。

## 14. 子链路摘要

```text
Engine._launch_subprocesses()
  -> Scheduler subprocess x TP
     -> Scheduler.__init__(gpu_id, tp_rank)
     -> TpModelWorker
     -> ModelRunner
        -> init_torch_distributed()
        -> create_sampler()
        -> load_model()
           -> Qwen3_5ForConditionalGeneration
           -> Qwen3_5ForCausalLM
        -> init_memory_pool()
           -> HybridReqToTokenPool
           -> HybridLinearKVPool
        -> init_attention_backend()
           -> FlashInferAttnBackend
           -> GDNAttnBackend
           -> HybridLinearAttnBackend
        -> init_device_graphs()
        -> init_piecewise_cuda_graphs()  # 当前脚本关闭

Scheduler.run_batch()
  -> TpModelWorker.forward_batch_generation()
  -> ForwardBatch.init_new()
  -> ModelRunner.forward()
  -> ModelRunner.sample()
  -> GenerationBatchResult
  -> BatchTokenIDOutput
  -> DetokenizerManager
  -> BatchStrOutput
  -> TokenizerManager
  -> OpenAI JSON or SSE
```
