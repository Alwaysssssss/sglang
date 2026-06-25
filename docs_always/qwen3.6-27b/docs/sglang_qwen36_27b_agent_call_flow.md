# Qwen3.6-27B Agent SGLang 调用流程

本文基于当前仓库代码和 `docs_always/qwen3.6-27b/start_qwen36_27b_agent.sh`，从启动脚本向下梳理 Qwen3.6-27B Agent 服务在 SGLang 内部的调用流程。分析采用多个只读 subagent 分别检查入口/bootstrap、OpenAI HTTP、scheduler/KV cache 链路，并由主线程补充模型执行链路核查。

## 1. 顶层启动流程

启动脚本默认入口是：

```bash
docs_always/qwen3.6-27b/start_qwen36_27b_agent.sh
```

脚本默认值集中在 `start_qwen36_27b_agent.sh:4-49`：

| 变量 | 默认值 | 下游作用 |
| --- | --- | --- |
| `SGLANG_PY` | `/mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/python3` | 运行 `python -m sglang.launch_server` |
| `MODEL_PATH` | `/mnt/shanhai-ai/wenhy/models/Qwen/Qwen/Qwen3___6-27B` | `--model-path` |
| `SERVED_MODEL_NAME` | `qwen3.6-27b` | OpenAI `/v1/models` 和请求中的模型名 |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | 限定 4 张 GPU |
| `TP_SIZE` | `4` | `--tensor-parallel-size 4`，启动 4 个 TP scheduler worker |
| `CONTEXT_LENGTH` | `262144` | `--context-length`，单请求上下文上限 |
| `MAX_OUTPUT_TOKENS` | `128000` | 写入 client defaults，客户端侧输出上限参考 |
| `CHUNKED_PREFILL_SIZE` | `8192` | 长 prompt 分块 prefill |
| `MAX_PREFILL_TOKENS` | `16384` | 单轮 prefill batch token 预算 |
| `SCHEDULE_POLICY` | `lpm` | longest prefix match 调度 |
| `RADIX_EVICTION_POLICY` | `lru` | radix cache 淘汰策略 |
| `DTYPE` | `bfloat16` | 模型 dtype |
| `ATTENTION_BACKEND` | `flashinfer` | attention kernel 后端 |
| `SAMPLING_BACKEND` | `flashinfer` | top-k/top-p 采样后端 |
| `TOOL_CALL_PARSER` | `qwen3_coder` | OpenAI tool_calls 输出解析 |

启动前检查包括 Python 可执行、模型目录、API key、`MAX_OUTPUT_TOKENS < CONTEXT_LENGTH`，见 `start_qwen36_27b_agent.sh:131-149`。脚本会用 `nvidia-smi` 读取可见 GPU 的总显存和已用显存，并按 `MEMORY_TARGET_FRACTION` 与 `RESPECT_CURRENT_GPU_USAGE` 估算每卡服务预算，见 `start_qwen36_27b_agent.sh:156-199`。

如果没有显式设置 `MAX_RUNNING_REQUESTS`，脚本会用以下近似逻辑推导并发：

```text
service_budget_mib = min_visible_gpu_total_mib * MEM_FRACTION_STATIC
kv_budget_mib = service_budget_mib - model_shard_mib - STATIC_OVERHEAD_MIB
estimated_total_tokens = kv_budget_mib * 1048576 / KV_BYTES_PER_TOKEN_PER_GPU
MAX_RUNNING_REQUESTS = estimated_total_tokens / CONTEXT_LENGTH
MAX_RUNNING_REQUESTS <= MAX_RUNNING_REQUESTS_CAP
```

对应代码在 `start_qwen36_27b_agent.sh:202-245`。随后默认 `MAX_QUEUED_REQUESTS=MAX_RUNNING_REQUESTS*8`，`PREFILL_MAX_REQUESTS=MAX_RUNNING_REQUESTS`，见 `start_qwen36_27b_agent.sh:245-246`。

脚本最终拼出的核心命令在 `start_qwen36_27b_agent.sh:282-350`：

```bash
"$SGLANG_PY" -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host "$SGLANG_HOST" \
  --port "$SGLANG_PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size "$TP_SIZE" \
  --context-length "$CONTEXT_LENGTH" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --max-queued-requests "$MAX_QUEUED_REQUESTS" \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --prefill-max-requests "$PREFILL_MAX_REQUESTS" \
  --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
  --schedule-policy "$SCHEDULE_POLICY" \
  --radix-eviction-policy "$RADIX_EVICTION_POLICY" \
  --dtype "$DTYPE" \
  --attention-backend "$ATTENTION_BACKEND" \
  --sampling-backend "$SAMPLING_BACKEND" \
  --sampling-defaults "$SAMPLING_DEFAULTS" \
  --tool-call-parser "$TOOL_CALL_PARSER" \
  --log-requests \
  --enable-request-time-stats-logging \
  --enable-metrics \
  --export-metrics-to-file \
  --disable-piecewise-cuda-graph
```

`MAX_TOTAL_TOKENS` 只有在环境变量存在时才追加为 `--max-total-tokens`，见 `start_qwen36_27b_agent.sh:311-312`。`EXTRA_SERVER_ARGS` 会在最后追加，见 `start_qwen36_27b_agent.sh:348-350`。

实际启动使用：

```bash
setsid env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" OPENAI_API_KEY="$OPENAI_API_KEY" "${server_cmd[@]}"
```

见 `start_qwen36_27b_agent.sh:397-401`。脚本写入 PID 后，默认循环请求 `/health` 直到 ready 或超时，见 `start_qwen36_27b_agent.sh:412-436`。

## 2. Python 入口和 ServerArgs

`python -m sglang.launch_server` 进入 `python/sglang/launch_server.py`。`__main__` 调用：

```text
prepare_server_args(sys.argv[1:])
run_server(server_args)
```

证据在 `python/sglang/launch_server.py:50-64`。

`run_server()` 的分支选择在 `python/sglang/launch_server.py:15-47`：

```text
encoder_only -> encode server
grpc_mode    -> grpc server
use_ray      -> ray http server
default      -> sglang.srt.entrypoints.http_server.launch_server
```

本脚本没有传 `--encoder-only`、`--grpc-mode`、`--use-ray`，因此走默认 HTTP 分支：`sglang.srt.entrypoints.http_server.launch_server(server_args)`。

CLI 解析由 `prepare_server_args()` 完成：创建 argparse，注册 `ServerArgs.add_cli_args`，可选合并 `--config`，然后 `parse_args` 并转成 `ServerArgs`，见 `python/sglang/srt/server_args.py:6515-6539`。`ServerArgs` 字段定义在 `python/sglang/srt/server_args.py:287-442`，其中与本脚本直接相关的字段包括：

| CLI | ServerArgs 字段 | 证据 |
| --- | --- | --- |
| `--model-path` | `model_path` | `server_args.py:297-306` |
| `--host`, `--port` | `host`, `port` | `server_args.py:312-316` |
| `--context-length` | `context_length` | `server_args.py:306` |
| `--dtype` | `dtype` | `server_args.py:329-334` |
| `--mem-fraction-static` | `mem_fraction_static` | `server_args.py:342-347` |
| `--max-running-requests` | `max_running_requests` | `server_args.py:342-345` |
| `--max-queued-requests` | `max_queued_requests` | `server_args.py:342-346` |
| `--max-total-tokens` | `max_total_tokens` | `server_args.py:342-347` |
| `--chunked-prefill-size` | `chunked_prefill_size` | `server_args.py:342-348` |
| `--max-prefill-tokens` | `max_prefill_tokens` | `server_args.py:349` |
| `--prefill-max-requests` | `prefill_max_requests` | `server_args.py:350` |
| `--schedule-policy` | `schedule_policy` | `server_args.py:351` |
| `--radix-eviction-policy` | `radix_eviction_policy` | `server_args.py:362` |
| `--served-model-name` | `served_model_name` | `server_args.py:428-432` |
| `--tool-call-parser` | `tool_call_parser` | `server_args.py:438-441` |
| `--api-key` | `api_key` | `server_args.py:428-430` |
| metrics/logging flags | `enable_metrics`, `log_requests`, `export_metrics_to_file` | `server_args.py:393-426` |

`Engine._launch_subprocesses()` 启动前会调用 `server_args.check_server_args()` 做进一步校验，见 `python/sglang/srt/entrypoints/engine.py:639-643`。

## 3. 服务进程拓扑

`http_server.launch_server()` 的注释直接描述 SRT server 拓扑：HTTP server、Engine、TokenizerManager 在主进程，Scheduler 和 DetokenizerManager 在子进程，进程间通过 ZMQ IPC 通信，见 `python/sglang/srt/entrypoints/http_server.py:2135-2157`。

本脚本默认 `dp_size=1`、`pp_size=1`、`tp_size=4`、`tokenizer_worker_num=1`，实际拓扑是：

```text
主进程
  python -m sglang.launch_server
    -> launch_server.run_server()
    -> http_server.launch_server()
    -> Engine._launch_subprocesses()
       -> Scheduler subprocess x 4
          - tp_rank = 0..3
          - gpu_id  = 0..3
       -> DetokenizerManager subprocess x 1
       -> TokenizerManager in main process
    -> FastAPI / uvicorn on 127.0.0.1:30000

请求数据流
  HTTP/OpenAI endpoint
    -> TokenizerManager
    -> ZMQ scheduler_input_ipc
    -> Scheduler rank0 recv input, TP ranks cooperate
    -> ModelRunner forward + sampler
    -> ZMQ detokenizer_ipc
    -> DetokenizerManager
    -> ZMQ tokenizer_ipc
    -> TokenizerManager
    -> HTTP JSON or SSE response
```

关键代码：

- `http_server.launch_server()` 调 `Engine._launch_subprocesses()`，再 `_setup_and_run_http_server()`：`python/sglang/srt/entrypoints/http_server.py:2158-2181`
- `_launch_scheduler_processes()` 在 `dp_size == 1` 时遍历 TP rank，按 rank 创建 `mp.Process(target=run_scheduler_process_func, ...)`：`python/sglang/srt/entrypoints/engine.py:513-579`
- `_launch_subprocesses()` 顺序是配置环境、分配 `PortArgs`、启动 scheduler、启动 detokenizer、初始化 tokenizer manager、等待 scheduler ready、启动 watchdog：`python/sglang/srt/entrypoints/engine.py:620-754`
- `run_scheduler_process()` 创建 `Scheduler` 并通过 pipe 发送 ready 信息：`python/sglang/srt/managers/scheduler.py:3560-3615`
- `TokenizerManager` 主进程初始化 model config、tokenizer、IPC、日志、metrics、dispatcher：`python/sglang/srt/managers/tokenizer_manager.py:178-225`
- `DetokenizerManager` 子进程从 scheduler 接 token id 输出，再回推字符串输出给 TokenizerManager：`python/sglang/srt/managers/detokenizer_manager.py:73-145`
- HTTP server setup 设置全局状态、metrics middleware、auth middleware，然后 `uvicorn.run()`：`python/sglang/srt/entrypoints/http_server.py:1972-2094`

## 4. FastAPI 和 OpenAI 路由

FastAPI app 在 `python/sglang/srt/entrypoints/http_server.py:373-389` 创建并注册 router。与本服务最相关的路由：

| 路由 | 代码位置 | 作用 |
| --- | --- | --- |
| `GET /health` | `http_server.py:476-548` | readiness/health check |
| `POST /v1/chat/completions` | `http_server.py:1388-1395` | OpenAI chat completion |
| `GET /v1/models` | `http_server.py:1498-1527` | 返回 `served_model_name` 和 context length |
| `/metrics` | metrics middleware | `enable_metrics` 时挂载 Prometheus endpoint |

`/v1/chat/completions` 的入口只做路由转发：

```text
openai_v1_chat_completions(request, raw_request)
  -> raw_request.app.state.openai_serving_chat.handle_request(request, raw_request)
```

见 `http_server.py:1388-1395`。

鉴权在单 tokenizer 模式下由 HTTP setup 添加 middleware，见 `http_server.py:2004-2022`。本脚本传了 `--api-key`，因此普通 API 请求需要 `Authorization: Bearer <key>`；`/health` 和 `/metrics` 在 auth 层放行，见 `python/sglang/srt/utils/auth.py:74-146`。

## 5. OpenAI Chat 请求到 TokenizerManager

OpenAI chat 主链路：

```text
/v1/chat/completions
  -> OpenAIServingChat.handle_request()
  -> validate request
  -> _process_messages()
  -> chat template / tokenizer.apply_chat_template()
  -> GenerateReqInput(input_ids=..., sampling_params=..., stream=...)
  -> TokenizerManager.generate_request()
```

关键点：

- FastAPI 先校验 JSON content-type：`http_server.py:457-470`
- `ChatCompletionRequest` 协议模型定义在 `python/sglang/srt/entrypoints/openai/protocol.py`
- 通用 OpenAI adapter 在 `OpenAIServingBase.handle_request()` 中记录 received time、校验请求、转换内部请求，并按 `request.stream` 分叉：`python/sglang/srt/entrypoints/openai/serving_base.py:73-109`
- Chat 专用逻辑校验 messages、tools、tool_choice、max tokens 等：`python/sglang/srt/entrypoints/openai/serving_chat.py:194-240`
- Chat adapter 生成 `GenerateReqInput`：`serving_chat.py:242-327`
- HF/Jinja chat template 路径会调用 `tokenizer.apply_chat_template(..., tokenize=True, tools=...)` 得到 `prompt_ids`：`serving_chat.py:484-492`
- conversation template 路径会生成 prompt 字符串再 `tokenizer.encode(prompt)`：`serving_chat.py:547-593`

`TokenizerManager.generate_request()` 的主逻辑在 `python/sglang/srt/managers/tokenizer_manager.py:478-525`：

```text
normalize request
set default priority
validate rid
init request stats
log received request
wait if generation paused
validate/resolve LoRA
tokenize one request or batch
send tokenized request to scheduler
wait for response and yield
```

对于 chat adapter 已经生成 `input_ids` 的普通文本请求，TokenizerManager 不再重复文本 tokenize；它会创建 `TokenizedGenerateReqInput`，见 `tokenizer_manager.py:930-1003`，然后通过 `self.send_to_scheduler.send_pyobj(tokenized_obj)` 发送给 scheduler，见 `tokenizer_manager.py:1095-1102`。

TokenizerManager 后台 `handle_loop()` 持续从 detokenizer 收结果，再分发到对应 `ReqState`，见 `tokenizer_manager.py:1498-1528` 和 `tokenizer_manager.py:1532-1585`。等待协程 `_wait_one_response()` 对 streaming 逐块 yield，对 non-streaming 等完成后 yield 最终结果，见 `tokenizer_manager.py:1120-1215`。

## 6. tool_call_parser=qwen3_coder

脚本默认 `TOOL_CALL_PARSER=qwen3_coder`，见 `start_qwen36_27b_agent.sh:27`，并传给 `--tool-call-parser`，见 `start_qwen36_27b_agent.sh:302`。

SGLang 内部映射：

```text
ServerArgs.tool_call_parser
  -> OpenAIServingChat.tool_call_parser
  -> FunctionCallParser(..., "qwen3_coder")
  -> Qwen3CoderDetector
```

证据：

- `OpenAIServingChat` 保存 parser 名称：`python/sglang/srt/entrypoints/openai/serving_chat.py:89-101`
- parser 注册表中 `qwen3_coder -> Qwen3CoderDetector`：`python/sglang/srt/function_call/function_call_parser.py:30-66`
- `Qwen3CoderDetector` 识别 `<tool_call>`、`<function=...>`、`<parameter=...>` 格式：`python/sglang/srt/function_call/qwen3_coder_detector.py:18-38`
- 非流式输出在 `_process_tool_calls()` 中把文本解析成 OpenAI `ToolCall`：`serving_chat.py:1000-1014`、`serving_chat.py:1124-1195`
- 流式输出在 `_process_tool_call_stream()` 中生成 delta `tool_calls`：`serving_chat.py:765-790`、`serving_chat.py:1316-1425`

因此，`qwen3_coder` 主要影响模型输出如何被转换为 OpenAI 兼容 `tool_calls`；请求侧如果指定 `tool_choice=required` 或具体工具，还会在 chat adapter 中构造 tool 相关约束，见 `serving_chat.py:341-369`。

## 7. Scheduler 入队和调度

Scheduler 初始化时保存 TP/rank/device、调度参数、model config、IPC、model worker、cache/memory pool、chunked prefill 和 schedule policy，见 `python/sglang/srt/managers/scheduler.py:273-420`。

Scheduler 主循环：

```text
event_loop_normal()
  -> recv_requests()
  -> process_input_requests()
  -> get_next_batch_to_run()
  -> run_batch()
  -> process_batch_result()
```

证据在 `scheduler.py:1303-1320`。启用 overlap schedule 时走 `event_loop_overlap()`，见 `scheduler.py:1331-1362`。

接收请求：

- 只有 `pp_rank == 0 && attn_tp_rank == 0 && attn_cp_rank == 0` 的 scheduler rank 直接从 tokenizer ZMQ PULL socket 收请求，其它 TP rank 通过分布式通信拿到广播后的请求，见 `scheduler.py:1423-1475`
- dispatcher 把 `TokenizedGenerateReqInput` 分发给 `handle_generate_request()`，见 `scheduler.py:1204-1210`
- `handle_generate_request()` 将 tokenized request 转成内部 `Req`，保存 input ids、sampling params、stream 标记、LoRA、priority、metrics collector 等，并校验输入长度，见 `scheduler.py:1724-1901`
- `_add_request_to_queue()` 做优先级检查、队列上限检查、可选 cache prefetch，然后把请求加入 `waiting_queue`，见 `scheduler.py:1936-1944`
- `max_queued_requests` 超限时会 abort incoming request，见 `scheduler.py:1985-2032`

## 8. Prefill、Decode 和关键参数

`get_next_batch_to_run()` 的策略是优先尝试 prefill 新请求，不能 prefill 时再推进 decode；它会合并上轮 prefill 产物到 `running_batch`，见 `scheduler.py:2177-2282`。

新 prefill batch 的关键逻辑在 `_get_new_batch_prefill_raw()`：

```text
if running batch full and no chunked req:
    return None
policy.calc_priority(waiting_queue, running_batch)
adder = PrefillAdder(... max_prefill_tokens, chunked_prefill_size,
                    max_running_requests, prefill_max_requests ...)
for req in waiting_queue:
    req.init_next_round_input(tree_cache)
    adder.add_one_req(req)
remove can_run_list from waiting_queue
new_batch = ScheduleBatch.init_new(...)
new_batch.prepare_for_extend()
```

证据在 `scheduler.py:2338-2518`。

decode batch 更新在 `update_running_batch()`：

- 过滤已完成请求
- 检查 decode KV 空间
- KV 不足时 retract 部分请求回队列
- 调 `prepare_for_decode()`

见 `scheduler.py:2547-2625`。

关键启动参数的生效点：

| 参数 | 生效位置 | 说明 |
| --- | --- | --- |
| `max_running_requests` | `model_runner_kv_cache_mixin.py`, `tp_worker.py:293-300`, `scheduler.py:641-647` | 限制运行态请求数，也影响 request pool size |
| `max_queued_requests` | `scheduler.py:1985-2032` | 等待队列满时拒绝或替换请求 |
| `chunked_prefill_size=8192` | `scheduler.py:862-883`, `scheduler.py:2370-2393`, `schedule_policy.py:803-843` | 长 prompt prefill 分块 |
| `prefill_max_requests` | `scheduler.py:2389-2391`, `schedule_policy.py:748-749` | 单轮 prefill 最多纳入多少请求 |
| `max_prefill_tokens=16384` | `scheduler.py:2378-2393`, `schedule_policy.py:398-407` | 单轮 prefill token 预算 |
| `context_length=262144` | `model_config.py:379-408`, `tp_worker.py:301-305` | 单请求上下文和 `max_req_input_len` 基准 |
| `mem_fraction_static` | `model_runner.py:289-313`, KV cache mixin | 影响可分配 KV token capacity |
| `max_total_tokens` | KV cache mixin | 显式限制 KV pool 总 token，脚本默认不传 |

## 9. lpm 调度和 radix KV cache

本脚本传 `--schedule-policy lpm`。在 `SchedulePolicy` 中，`lpm` 是 cache-aware policy，含义是 longest prefix match，见 `python/sglang/srt/managers/schedule_policy.py:80-84`。

调度时：

```text
SchedulePolicy.calc_priority()
  -> _compute_prefix_matches()
     -> tree_cache.match_prefix(RadixKey(token_ids, extra_key))
     -> 写入 req.prefix_indices / last_node / last_host_node / host_hit_length
  -> _sort_by_longest_prefix()
```

证据在 `schedule_policy.py:117-159` 和 `schedule_policy.py:185-256`。

Radix cache 关键结构：

- `RadixKey` 用 token ids 加 `extra_key` 作为命名空间，避免 LoRA/cache_salt 等场景错误复用，见 `python/sglang/srt/mem_cache/radix_cache.py:71-83`
- `TreeNode` 保存 key、KV value、lock ref、last access time、host value 等，见 `radix_cache.py:121-181`
- `RadixCache.match_prefix()` 返回最长命中的 KV indices 和最后节点，相关代码在 `radix_cache.py:374-444`
- `Req.init_next_round_input()` 用 tree cache 做实际 prefix match，并计算本轮需要 prefill 的未命中部分，相关代码在 `schedule_batch.py:940-1022`
- `ScheduleBatch.prepare_for_extend()` 只对未命中 token 做 extend，并分配新 KV，见 `python/sglang/srt/managers/schedule_batch.py:1560-1618`

`--radix-eviction-policy lru` 会让 radix cache 使用 LRU 优先级，相关代码在 `radix_cache.py:285-331` 和 `evict_policy.py:16-18`。空间不足时，SGLang 通过 `evict_from_tree_cache()` 淘汰 radix cache 中可释放的叶子节点，相关代码在 `python/sglang/srt/mem_cache/common.py:229-253` 和 `radix_cache.py:582-609`。

## 10. 模型加载、TP 和执行

本地模型配置文件 `/mnt/shanhai-ai/wenhy/models/Qwen/Qwen/Qwen3___6-27B/config.json` 当前关键信息：

```json
{
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "model_type": "qwen3_5",
  "text_config": {
    "model_type": "qwen3_5_text",
    "max_position_embeddings": 262144,
    "num_hidden_layers": 64,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256
  }
}
```

因此模型会映射到 SGLang 的 Qwen3.5 实现：

- `Qwen3_5Config` / `Qwen3_5TextConfig` 定义在 `python/sglang/srt/configs/qwen3_5.py`
- `Qwen3_5ForConditionalGeneration` 定义在 `python/sglang/srt/models/qwen3_5.py:1312-1324`
- 模型架构解析通过 `get_model_architecture()` 查询 `ModelRegistry`，见 `python/sglang/srt/model_loader/utils.py:193-228`

TP worker 创建链路：

```text
Engine._launch_scheduler_processes()
  -> run_scheduler_process(server_args, gpu_id, tp_rank, ...)
  -> Scheduler.__init__()
  -> Scheduler.init_model_worker()
  -> TpModelWorker(...)
  -> TpModelWorker._init_model_runner()
  -> ModelRunner(...)
```

关键证据：

- `Engine._launch_scheduler_processes()` 按 `tp_rank_range` 创建 scheduler 子进程并计算 `gpu_id`：`python/sglang/srt/entrypoints/engine.py:529-579`
- `run_scheduler_process()` 创建 `Scheduler`：`python/sglang/srt/managers/scheduler.py:3560-3615`
- `Scheduler.__init__()` 记录 `tp_rank`、`tp_size`、`gpu_id`，并调用 `init_model_worker()`：`scheduler.py:288-386`
- `TpModelWorker` 保存 `tp_size/tp_rank/gpu_id`，创建 `ModelRunner`：`python/sglang/srt/managers/tp_worker.py:218-362`
- `ModelRunner.__init__()` 保存 `mem_fraction_static`、`gpu_id`、`tp_rank`、`tp_size`、`server_args` 等：`python/sglang/srt/model_executor/model_runner.py:286-347`

模型加载和后端初始化：

```text
ModelRunner.initialize()
  -> create_sampler()
  -> load_model()
     -> get_model_loader(load_config, model_config)
     -> loader.load_model(...)
  -> configure_kv_cache_dtype()
  -> init_memory_pool(...)
  -> init_attention_backend()
  -> init_device_graphs()
  -> init_piecewise_cuda_graphs()
```

证据：

- `ModelRunner.initialize()` 创建 sampler 并加载模型：`model_runner.py:461-508`
- `load_model()` 构造 `LoadConfig`，选择 loader 并加载权重：`model_runner.py:1071-1172`
- 默认 loader fallback 是 `DefaultModelLoader`：`python/sglang/srt/model_loader/loader.py:3105-3192`
- 加载后初始化 memory pool、attention backend、device graph、piecewise CUDA graph：`model_runner.py:620-675`

`--attention-backend flashinfer` 生效点：

- `ModelRunner.init_attention_backend()` 调 `_get_attention_backend()`：`model_runner.py:1959-1971`
- `_get_attention_backend()` 读取 `server_args.get_attention_backends()` 并最终调用 `_get_attention_backend_from_str()`：`model_runner.py:1972-2024`
- `ATTENTION_BACKENDS["flashinfer"]` 注册到 `FlashInferAttnBackend` 或 MLA 版本：`python/sglang/srt/layers/attention/attention_registry.py:23-45`

`--sampling-backend flashinfer` 生效点：

- `Sampler` 从全局 server args 读取 sampling backend，见 `python/sglang/srt/layers/sampler.py:41-57`
- 非 greedy 且需要 top-k/top-p/min-p 时，`backend == "flashinfer"` 会走 flashinfer 采样 kernel，见 `sampler.py:189-227`

`--disable-piecewise-cuda-graph` 生效点：

- 脚本在 `start_qwen36_27b_agent.sh:340-342` 追加该参数
- `ModelRunner.init_piecewise_cuda_graphs()` 如果 `server_args.disable_piecewise_cuda_graph` 为真，直接不创建 piecewise graph runner；相关入口在 `model_runner.py:2464-2482`
- 这不等于完全关闭普通 CUDA graph；普通 CUDA graph 由 `disable_cuda_graph` 等参数控制，piecewise graph 是额外分段图优化路径。

## 11. Forward、Sampling 和输出回传

模型执行从 scheduler 的 batch 开始：

```text
Scheduler.run_batch(batch)
  -> batch.get_model_worker_batch()
  -> self.model_worker.forward_batch_generation(...)
  -> TpModelWorker.forward_batch_generation(...)
  -> ModelRunner.forward(...)
  -> Sampler.forward(...)
  -> next_token_ids
```

证据：

- `Scheduler.run_batch()` 调 `model_worker.forward_batch_generation()`：`python/sglang/srt/managers/scheduler.py:2636-2728`
- `TpModelWorker.forward_batch_generation()` 调 `ModelRunner.forward()` 并 sample next token，相关代码在 `python/sglang/srt/managers/tp_worker.py:444-504`
- `ModelRunner.forward()` 主体在 `python/sglang/srt/model_executor/model_runner.py:2725-2782`
- `Sampler.forward()` 对 logits 做温度、top-k/top-p/min-p、greedy 或 flashinfer/pytorch 采样，并同步 TP token ids：`python/sglang/srt/layers/sampler.py:77-187`

输出回传分两段：

```text
Scheduler.process_batch_result()
  -> stream_output_generation()
  -> BatchTokenIDOutput
  -> DetokenizerManager
  -> BatchStrOutput
  -> TokenizerManager.handle_loop()
  -> ReqState event
  -> OpenAI serving layer
  -> HTTP JSON or SSE
```

细节：

- prefill result 会 append 首 token，未完成请求会 cache unfinished，然后 stream：相关代码在 `scheduler_output_processor_mixin.py:123-325`
- decode result 会 append next token、检查 finished，完成时释放 KV cache，然后 stream：相关代码在 `scheduler_output_processor_mixin.py:374-548`
- `stream_output_generation()` 构造 `BatchTokenIDOutput` 发给 detokenizer：相关代码在 `scheduler_output_processor_mixin.py:924-1201`
- Detokenizer 将 token ids 增量 decode 成字符串并回推给 TokenizerManager：`python/sglang/srt/managers/detokenizer_manager.py:137-145`、`detokenizer_manager.py:216-328`
- TokenizerManager `_handle_batch_output()` 更新 text、output_ids、finish_reason、prompt/completion/cached tokens 等 meta 信息：`python/sglang/srt/managers/tokenizer_manager.py:1532-1585`
- OpenAI chat streaming 返回 `StreamingResponse(text/event-stream)`：`python/sglang/srt/entrypoints/openai/serving_chat.py:604-630`
- OpenAI chat non-streaming 取生成器首个最终结果后构造 `ChatCompletionResponse`：`serving_chat.py:917-1050`

## 12. 日志、metrics 和排查入口

脚本默认打开：

- `--log-requests`
- `--log-requests-level 3`
- `--log-requests-format json`
- `--log-requests-target "$REQUEST_LOG_DIR"`
- `--enable-request-time-stats-logging`
- `--enable-metrics`
- `--export-metrics-to-file`

请求日志初始化在 `TokenizerManager.init_request_logging_and_dumping()`，相关代码在 `python/sglang/srt/managers/tokenizer_manager.py:344-366`；收到请求时记录在 `tokenizer_manager.py:507-509`；完成请求时记录在 `tokenizer_manager.py:1191-1195`。

metrics 相关：

- HTTP response tracking middleware 在 `server_args.enable_metrics` 为真时添加：`python/sglang/srt/entrypoints/http_server.py:1989-1990`
- `/metrics` 由 Prometheus middleware 暴露，相关代码在 `python/sglang/srt/utils/common.py:1331-1341`
- 每请求 metrics 文件导出在完成后异步写入，相关代码在 `tokenizer_manager.py:1197-1201`

上线排查时建议按以下顺序看：

1. `START_LOG_FILE`：确认脚本推导出的 `MEM_FRACTION_STATIC`、`MAX_RUNNING_REQUESTS`、启动命令和 ready check。
2. `SERVER_LOG_FILE`：确认 `server_args`、TP rank 启动、模型加载、KV cache capacity、uvicorn startup。
3. `REQUEST_LOG_DIR`：按 rid 查请求进入、完成、finish_reason。
4. `METRICS_FILE_DIR` 和 `/metrics`：查 TTFT、ITL、E2E、队列和 cache 命中等运行指标。

## 13. 端到端摘要

端到端调用链可以压缩为：

```text
start_qwen36_27b_agent.sh
  -> setsid env CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server ...
  -> launch_server.prepare_server_args()
  -> launch_server.run_server()
  -> http_server.launch_server()
  -> Engine._launch_subprocesses()
     -> Scheduler x4
        -> TpModelWorker
        -> ModelRunner
        -> Qwen3_5ForConditionalGeneration
        -> FlashInfer attention backend
        -> Sampler(flashinfer)
     -> DetokenizerManager x1
     -> TokenizerManager in main process
  -> FastAPI / uvicorn
  -> /v1/chat/completions
  -> OpenAIServingChat
  -> GenerateReqInput
  -> TokenizerManager.generate_request()
  -> TokenizedGenerateReqInput
  -> Scheduler waiting_queue
  -> lpm + radix cache prefix match
  -> PrefillAdder / ScheduleBatch.prepare_for_extend()
  -> Scheduler.run_batch()
  -> ModelRunner.forward()
  -> Sampler.forward()
  -> BatchTokenIDOutput
  -> DetokenizerManager
  -> BatchStrOutput
  -> TokenizerManager._wait_one_response()
  -> OpenAI JSON or SSE response
```
