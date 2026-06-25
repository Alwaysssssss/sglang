# Qwen3.6-27B SGLang 启动、ServerArgs 与进程拓扑

本文覆盖 `docs_always/qwen3.6-27b/start_qwen36_27b_agent.sh` 到 SGLang HTTP server、Engine、Scheduler、TokenizerManager、DetokenizerManager 的启动链路。内容基于当前 worktree，尤其注意当前脚本默认 `SERVED_MODEL_NAME=qwen3.6-27b`、`LOG_REQUESTS_LEVEL=3`。

## 1. 顶层脚本默认值

启动入口：

```bash
docs_always/qwen3.6-27b/start_qwen36_27b_agent.sh
```

脚本默认值集中在 `start_qwen36_27b_agent.sh:4-49`：

| 变量 | 当前默认值 | 下游作用 |
| --- | --- | --- |
| `ROOT_DIR` | `/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang` | 启动前进入仓库根目录 |
| `SGLANG_PY` | `/mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/python3` | 运行 `python -m sglang.launch_server` |
| `MODEL_PATH` | `/mnt/shanhai-ai/wenhy/models/Qwen/Qwen/Qwen3___6-27B` | `--model-path` |
| `SGLANG_HOST` | `127.0.0.1` | `--host` |
| `SGLANG_PORT` | `30000` | `--port` |
| `SERVED_MODEL_NAME` | `qwen3.6-27b` | `--served-model-name`，影响 `/v1/models` 和请求 `model` |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | 启动进程环境变量 |
| `TP_SIZE` | `4` | `--tensor-parallel-size` |
| `CONTEXT_LENGTH` | `262144` | `--context-length` |
| `MAX_OUTPUT_TOKENS` | `128000` | 写入 client defaults，不直接传给 server |
| `MEMORY_TARGET_FRACTION` | `0.90` | 显存预算推导输入 |
| `RESPECT_CURRENT_GPU_USAGE` | `1` | 估算预算时是否扣除当前已用显存 |
| `CHUNKED_PREFILL_SIZE` | `8192` | `--chunked-prefill-size` |
| `MAX_PREFILL_TOKENS` | `16384` | `--max-prefill-tokens` |
| `TOOL_CALL_PARSER` | `qwen3_coder` | `--tool-call-parser` |
| `SCHEDULE_POLICY` | `lpm` | `--schedule-policy` |
| `RADIX_EVICTION_POLICY` | `lru` | `--radix-eviction-policy` |
| `LOG_REQUESTS_LEVEL` | `3` | `--log-requests-level 3`，记录完整输入/输出 |

启动前脚本会检查 Python 可执行、模型目录、API key、`MAX_OUTPUT_TOKENS < CONTEXT_LENGTH`，见 `start_qwen36_27b_agent.sh:131-148`。

## 2. 显存与并发推导

脚本使用 `nvidia-smi` 获取可见 GPU 的显存快照，见 `start_qwen36_27b_agent.sh:150-199`。核心计算：

```text
target_budget_mib = gpu_total_mib * MEMORY_TARGET_FRACTION
if RESPECT_CURRENT_GPU_USAGE == 1:
    target_budget_mib -= gpu_used_mib

auto_mem_fraction_static = min_service_budget_mib / min_gpu_total_mib
MEM_FRACTION_STATIC = 显式环境变量或 auto_mem_fraction_static
```

自动并发推导在 `start_qwen36_27b_agent.sh:219-246`：

```text
model_shard_mib = ceil(MODEL_SIZE_MIB / TP_SIZE)
kv_budget_mib = service_budget_mib - model_shard_mib - STATIC_OVERHEAD_MIB
estimated_total_tokens = kv_budget_mib * 1048576 / KV_BYTES_PER_TOKEN_PER_GPU
MAX_RUNNING_REQUESTS = estimated_total_tokens / CONTEXT_LENGTH
MAX_RUNNING_REQUESTS = clamp(MAX_RUNNING_REQUESTS, 1, MAX_RUNNING_REQUESTS_CAP)
MAX_QUEUED_REQUESTS = MAX_RUNNING_REQUESTS * 8
PREFILL_MAX_REQUESTS = MAX_RUNNING_REQUESTS
```

`KV_BYTES_PER_TOKEN_PER_GPU=16384` 只用于脚本侧估算；SGLang 内部 KV pool 容量由 `ModelRunner` 根据模型结构、KV dtype、显存 profiling 和 `mem_fraction_static` 重新计算。

## 3. 下游 CLI 命令

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
  --log-level "$LOG_LEVEL" \
  --log-level-http "$LOG_LEVEL_HTTP" \
  --decode-log-interval "$DECODE_LOG_INTERVAL" \
  --uvicorn-access-log-exclude-prefixes /health /metrics \
  --crash-dump-folder "$CRASH_DUMP_FOLDER" \
  --api-key "$OPENAI_API_KEY"
```

条件追加项包括 `--max-total-tokens`、`--log-requests`、`--enable-request-time-stats-logging`、`--enable-metrics`、`--export-metrics-to-file`、`--disable-piecewise-cuda-graph`、`EXTRA_SERVER_ARGS`，见 `start_qwen36_27b_agent.sh:311-350`。

实际启动使用：

```bash
setsid env \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  OPENAI_API_KEY="$OPENAI_API_KEY" \
  "${server_cmd[@]}"
```

见 `start_qwen36_27b_agent.sh:397-401`。

## 4. `launch_server` 分支

`python -m sglang.launch_server` 进入 `python/sglang/launch_server.py`。`__main__` 调用：

```text
prepare_server_args(sys.argv[1:])
run_server(server_args)
```

见 `launch_server.py:50-64`。

`run_server()` 的分支在 `launch_server.py:15-47`：

```text
encoder_only -> encoder server
grpc_mode    -> grpc server
use_ray      -> ray http server
default      -> sglang.srt.entrypoints.http_server.launch_server()
```

当前脚本没有传 `--encoder-only`、`--grpc-mode`、`--use-ray`，因此走默认 HTTP 分支。

## 5. `ServerArgs` 解析与校验

`prepare_server_args()` 在 `python/sglang/srt/server_args.py:6515-6539`：

```text
ArgumentParser(prog="sglang serve")
ServerArgs.add_cli_args(parser)
如果带 --config，合并配置
parse_args(argv)
ServerArgs.from_cli_args(raw_args)
```

与当前脚本强相关的字段定义在 `ServerArgs` dataclass 中：

| 参数组 | 字段 | 位置 |
| --- | --- | --- |
| 模型 | `model_path`, `tokenizer_path`, `context_length` | `server_args.py:297-306` |
| HTTP | `host`, `port`, `grpc_mode`, `nccl_port` | `server_args.py:312-320` |
| dtype/量化 | `dtype`, `kv_cache_dtype` | `server_args.py:329-334` |
| 内存/调度 | `mem_fraction_static`, `max_running_requests`, `max_queued_requests`, `max_total_tokens`, `chunked_prefill_size`, `max_prefill_tokens`, `prefill_max_requests`, `schedule_policy`, `radix_eviction_policy` | `server_args.py:342-362` |
| 并行 | `tp_size`, `pp_size`, `dp_size` | `server_args.py:369-375`, `server_args.py:443-445` |
| OpenAI | `api_key`, `served_model_name`, `tool_call_parser`, `sampling_defaults` | `server_args.py:428-441` |

CLI alias 在 `from_cli_args()` 中映射，例如 `--tensor-parallel-size` / `--tp-size` 映射到 `tp_size`，见 `server_args.py:5965-5975`。

最终参数校验在 Engine 启动子进程前调用：

```text
Engine._launch_subprocesses()
  -> _set_envs_and_config()
  -> server_args.check_server_args()
```

见 `python/sglang/srt/entrypoints/engine.py:639-643`。

## 6. Engine 子进程拓扑

`http_server.launch_server()` 的注释说明 SRT 拓扑：HTTP server、Engine、TokenizerManager 在主进程，Scheduler 和 DetokenizerManager 在子进程，通过 ZMQ IPC 通信，见 `http_server.py:2135-2157`。

当前默认拓扑：

```text
主进程
  python -m sglang.launch_server
    -> http_server.launch_server()
    -> Engine._launch_subprocesses()
       -> Scheduler subprocess x 4
          - tp_rank = 0..3
          - gpu_id = 0..3
       -> DetokenizerManager subprocess x 1
       -> TokenizerManager in main process
    -> FastAPI / uvicorn on 127.0.0.1:30000
```

`Engine._launch_subprocesses()` 顺序见 `engine.py:620-754`：

1. 配置 logger、环境变量和参数校验。
2. `PortArgs.init_new(server_args)` 分配 IPC/NCCL 端口。
3. `_launch_scheduler_processes()` 启动 scheduler。
4. `_launch_detokenizer_process()` 启动 detokenizer。
5. 创建 TokenizerManager。
6. 等 scheduler ready。
7. 启动 `SubprocessWatchdog`。

Scheduler 子进程由 `_launch_scheduler_processes()` 创建，见 `engine.py:513-617`。单 DP 默认路径会为每个 TP rank 创建 `mp.Process(target=run_scheduler_process, ...)`。

## 7. ZMQ / IPC 数据流

`PortArgs` 定义在 `server_args.py:6546-6565`，关键 IPC 名称：

| 字段 | 用途 |
| --- | --- |
| `scheduler_input_ipc_name` | TokenizerManager 发送 tokenized request 到 scheduler rank 0 |
| `detokenizer_ipc_name` | Scheduler 发送 token id 输出到 DetokenizerManager |
| `tokenizer_ipc_name` | DetokenizerManager/Scheduler 回传字符串或控制输出到 TokenizerManager |
| `rpc_ipc_name` | Engine 与 scheduler 的 RPC |
| `metrics_ipc_name` | scheduler metrics 输出 |

当前未启用 DP attention，`PortArgs.init_new()` 使用 `ipc://<tempfile>`，见 `server_args.py:6585-6595`。

请求数据流：

```text
POST /v1/chat/completions
  -> http_server.openai_v1_chat_completions()
  -> OpenAIServingChat.handle_request()
  -> TokenizerManager.generate_request()
  -> send_to_scheduler.send_pyobj(TokenizedGenerateReqInput)
  -> Scheduler.recv_requests()
  -> Scheduler.run_batch()
  -> stream_output_generation()
  -> DetokenizerManager.handle_batch_token_id_out()
  -> TokenizerManager.handle_loop()
  -> HTTP JSON or SSE
```

TokenizerManager 初始化 IPC 在 `tokenizer_manager.py:310-318`：`recv_from_detokenizer` 为 PULL/bind，`send_to_scheduler` 为 PUSH/bind。Scheduler 只有 `pp_rank == 0 && attn_tp_rank == 0 && attn_cp_rank == 0` 的 rank 直接接收 tokenizer ZMQ 请求，其它 TP rank 通过分布式通信接收广播，见 `scheduler.py:1423-1511`。

## 8. Ready 与 `/health`

脚本默认 `WAIT_FOR_READY=1`，启动后每 2 秒请求 `/health`，最多等待 `READY_TIMEOUT_SECONDS=900`，见 `start_qwen36_27b_agent.sh:412-436`。

SGLang `/health` 在 `http_server.py:476-548`：

- `gracefully_exit` 返回 503。
- `ServerStatus.Starting` 返回 503。
- 默认普通 `/health` 在非 Starting 时直接返回 200。
- 如果开启 `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` 或访问 `/health_generate`，会发一个 1-token 内部请求并等待回包。

鉴权层对 `/health` 和 `/metrics` 放行，见 `python/sglang/srt/utils/auth.py:74-146`。脚本 ready check 仍带 Authorization header，不影响结果。

## 9. 排查重点

优先看：

- `START_LOG_FILE`：脚本检查、显存推导、最终 redacted command、ready 失败时的 server log 尾部。
- `SERVER_LOG_FILE`：`server_args`、TP rank 启动、模型加载、KV capacity、uvicorn startup。
- `REQUEST_LOG_DIR`：当前默认 `LOG_REQUESTS_LEVEL=3`，请求输入输出会很完整。
- `METRICS_FILE_DIR` 和 `/metrics`：TTFT、ITL、E2E、队列、cache hit。

常见定位：

- 模型名不匹配：当前默认是 `qwen3.6-27b`，客户端 `model` 应与 `/v1/models` 返回一致。
- ready 卡住：区分模型加载、warmup、scheduler 子进程提前退出。
- scheduler 初始化失败：`Engine._wait_for_scheduler_ready()` 会检测子进程退出，OS OOM killer 常见于 SIGKILL。
- 版本问题：CUDA + FlashInfer 路径会检查 `flashinfer_python` 和 `sglang-kernel` 版本，见 `engine.py:1126-1141`。
