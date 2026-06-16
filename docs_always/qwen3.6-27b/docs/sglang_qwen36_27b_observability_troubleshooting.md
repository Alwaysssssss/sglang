# Qwen3.6-27B 日志、Metrics、Health 与排查手册

本文覆盖当前 `docs_always/qwen3.6-27b/start_qwen36_27b_agent.sh` 的日志、metrics、health/readiness、request logging、验证脚本和常见故障排查。当前脚本默认 `SERVED_MODEL_NAME=qwen3.6-27b`、`LOG_REQUESTS_LEVEL=3`。

## 1. 脚本输出文件

日志与观测目录在 `start_qwen36_27b_agent.sh:32-49`、`120-128`：

| 项 | 默认路径模式 | 说明 |
| --- | --- | --- |
| `LOG_DIR` | `${ROOT_DIR}/logs/qwen36_27b_agent` | 该服务日志根目录 |
| `PID_FILE` | `${LOG_DIR}/qwen36_27b_agent.pid` | 后台主进程 PID |
| `START_LOG_FILE` | `${LOG_DIR}/qwen36_27b_agent_start_${stamp}.log` | 启动脚本日志 |
| `SERVER_LOG_FILE` | `${LOG_DIR}/qwen36_27b_agent_tp${TP_SIZE}_256k_${stamp}.log` | SGLang stdout/stderr |
| `REQUEST_LOG_DIR` | `${LOG_DIR}/requests_${stamp}` | request logging 目录 |
| `METRICS_FILE_DIR` | `${LOG_DIR}/metrics_${stamp}` | 每请求 metrics JSONL 目录 |
| `CRASH_DUMP_FOLDER` | `${LOG_DIR}/crash_dumps_${stamp}` | crash 前 dump |
| `CLIENT_DEFAULTS_FILE` | `${LOG_DIR}/qwen36_27b_agent_client_defaults_${stamp}.json` | 客户端默认配置 |

当前关键默认值：

| 变量 | 默认值 |
| --- | --- |
| `SGLANG_HOST` | `127.0.0.1` |
| `SGLANG_PORT` | `30000` |
| `SERVED_MODEL_NAME` | `qwen3.6-27b` |
| `CONTEXT_LENGTH` | `262144` |
| `MAX_OUTPUT_TOKENS` | `128000` |
| `WAIT_FOR_READY` | `1` |
| `READY_TIMEOUT_SECONDS` | `900` |
| `LOG_REQUESTS` | `1` |
| `LOG_REQUESTS_LEVEL` | `3` |
| `LOG_REQUESTS_FORMAT` | `json` |
| `ENABLE_REQUEST_TIME_STATS_LOGGING` | `1` |
| `ENABLE_METRICS` | `1` |
| `EXPORT_METRICS_TO_FILE` | `1` |

`LOG_REQUESTS_LEVEL=3` 会记录完整输入/输出，适合复盘 agent 长上下文，但线上需注意日志体积与敏感内容。

## 2. 启动日志

`START_LOG_FILE` 是第一入口。脚本 `log()` 写 UTC 时间戳，见 `start_qwen36_27b_agent.sh:101-105`。

启动日志包含：

- `ROOT_DIR`、`MODEL_PATH`、`SERVED_MODEL_NAME`
- `CUDA_VISIBLE_DEVICES`、`TP_SIZE`
- `MEM_FRACTION_STATIC`、`MAX_RUNNING_REQUESTS`、`MAX_QUEUED_REQUESTS`
- `CONTEXT_LENGTH`、`MAX_OUTPUT_TOKENS`
- `CHUNKED_PREFILL_SIZE`、`MAX_PREFILL_TOKENS`
- `REQUEST_LOG_DIR`、`METRICS_FILE_DIR`、`CRASH_DUMP_FOLDER`
- GPU snapshot
- redacted launch command
- ready 成功或失败信息

常用命令：

```bash
LOG_DIR=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/logs/qwen36_27b_agent
ls -lt "$LOG_DIR" | head
tail -n 200 "$LOG_DIR"/qwen36_27b_agent_start_*.log
grep -E "MEM_FRACTION_STATIC|MAX_RUNNING_REQUESTS|MAX_QUEUED_REQUESTS|Launch command|Service is ready|Timed out" "$LOG_DIR"/qwen36_27b_agent_start_*.log
```

启动脚本直接失败时，先看 `ERROR:`：

- Python 不可执行：`start_qwen36_27b_agent.sh:131-133`
- 模型目录不存在：`135-137`
- API key 为空：`139-144`
- `MAX_OUTPUT_TOKENS >= CONTEXT_LENGTH`：`146-148`
- 可见 GPU 数少于 `TP_SIZE`：`194-197`
- GPU 显存预算不足：`181-216`
- 端口占用：`256-264`

## 3. Server 日志

`SERVER_LOG_FILE` 是实际服务进程 stdout/stderr，见 `start_qwen36_27b_agent.sh:397-401`。

重点搜索：

| 关键词 | 含义 |
| --- | --- |
| `server_args=` | 确认启动参数 |
| `tool_call_parser='qwen3_coder'` | 确认 tool parser |
| `max_total_num_tokens` | KV capacity 初始化 |
| `chunked_prefill_size` | chunked prefill 生效 |
| `max_running_requests` | running 并发上限 |
| `available_gpu_mem` | 可用显存 |
| `KV cache pool is full. Retract requests.` | 运行期 KV 压力 |
| `Req Time Stats` | request-time stats |
| `Tool call parsing error` | tool call 解析异常 |
| `Traceback`, `OutOfMemory`, `CUDA` | 异常 |

脚本传了 `--uvicorn-access-log-exclude-prefixes /health /metrics`，因此健康检查和 metrics 拉取不会刷屏。过滤器在 `python/sglang/srt/utils/common.py:2291-2421`。

```bash
tail -f "$SERVER_LOG_FILE"
grep -E "server_args=|tool_call_parser|max_total_num_tokens|available_gpu_mem|KV cache pool is full|Req Time Stats|Tool call parsing error|Traceback|OutOfMemory|CUDA" "$SERVER_LOG_FILE"
```

## 4. Health 与 readiness

脚本 ready loop 在 `start_qwen36_27b_agent.sh:412-436`：

```bash
curl --noproxy '*' -fsS \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  "http://${SGLANG_HOST}:${SGLANG_PORT}/health"
```

`/health` 实现在 `python/sglang/srt/entrypoints/http_server.py:476-548`：

- `gracefully_exit=True` 返回 503。
- `server_status == ServerStatus.Starting` 返回 503。
- 默认普通 `/health` 在非 Starting 且未 shutdown 时返回 200。
- `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` 或 `/health_generate` 会发 1-token 内部请求等待回包。
- health 生成请求设置 `log_metrics=False`，不会作为普通用户请求写 metrics 文件。

排查语义：

- `/health=200`：HTTP server 与基本 runtime ready。
- `/health=503`：可能仍在 Starting、正在退出，或 generation health probe 未收到回包。
- `/health` 不通：查进程、端口、`SERVER_LOG_FILE`。

## 5. `/v1/models`

`/v1/models` 在 `http_server.py:1498-1527`，返回：

- `TokenizerManager.served_model_name`，当前默认 `qwen3.6-27b`
- `TokenizerManager.model_config.context_len`，当前默认 `262144`
- 可选 LoRA adapter model card

命令：

```bash
curl --noproxy '*' -sS \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  http://127.0.0.1:30000/v1/models | jq .
```

`/health` 用于 readiness/liveness；`/v1/models` 用于确认 OpenAI API 鉴权、模型名和上下文长度。

## 6. Auth 排查

脚本优先使用环境变量 `OPENAI_API_KEY`；未设置则读取 `/etc/sglang/qwen36_openai_api_key`；都没有时默认拒绝启动，见 `start_qwen36_27b_agent.sh:112-144`。

middleware 添加点：

- `python/sglang/srt/entrypoints/http_server.py:2004-2022`
- `python/sglang/srt/utils/auth.py:149-208`

`decide_request_auth()` 在 `auth.py:74-146`：

- `/health` 与 `/metrics` 放行。
- 普通 endpoint 需要 `Authorization: Bearer <api_key>`。
- admin 强制 endpoint 需要 admin key。

401 排查：

```bash
curl --noproxy '*' -i http://127.0.0.1:30000/health
curl --noproxy '*' -i http://127.0.0.1:30000/v1/models
curl --noproxy '*' -i -H "Authorization: Bearer ${OPENAI_API_KEY}" http://127.0.0.1:30000/v1/models
```

如果本机 `/health` 正常但公网 `/health` 401，优先查 nginx、Cloudflare tunnel 或上层网关。

## 7. Request Logging

当前脚本默认追加：

```bash
--log-requests
--log-requests-level 3
--log-requests-format json
--log-requests-target "$REQUEST_LOG_DIR"
```

位置见 `start_qwen36_27b_agent.sh:315-321`。

初始化链路：

- `TokenizerManager.init_request_logging_and_dumping()`：`tokenizer_manager.py:344-366`
- `RequestLogger`：`python/sglang/srt/utils/request_logger.py:46-233`
- `create_log_targets()`：`python/sglang/srt/utils/log_utils.py:15-45`

日志文件名通常是：

```text
${REQUEST_LOG_DIR}/${hostname}_${rank}.log
```

`LOG_REQUESTS_LEVEL`：

| level | 含义 |
| --- | --- |
| `0` | metadata，不含 sampling params/text/input_ids/output |
| `1` | metadata + sampling params |
| `2` | metadata + sampling params + 截断 input/output |
| `3` | 完整 input/output，当前默认 |

查询：

```bash
REQ_LOG="$REQUEST_LOG_DIR/$(hostname)_0.log"
grep '"event": "request.received"' "$REQ_LOG" | tail
grep '"event": "request.finished"' "$REQ_LOG" | tail
grep '"rid": "你的-rid"' "$REQ_LOG"
```

## 8. Request-time stats 与文件 metrics

脚本默认打开：

```bash
--enable-request-time-stats-logging
--export-metrics-to-file
--export-metrics-to-file-dir "$METRICS_FILE_DIR"
```

完成请求时 TP rank 0 会记录 `Req Time Stats(...)`，路径：

- `python/sglang/srt/managers/scheduler_output_processor_mixin.py:1147-1152`
- `python/sglang/srt/managers/schedule_batch.py:1239-1251`
- `python/sglang/srt/observability/req_time_stats.py`

文件 exporter：

- `RequestMetricsExporterManager` 初始化：`tokenizer_manager.py:362-366`
- 完成请求后异步写出：`tokenizer_manager.py:1197-1201`
- 实现：`python/sglang/srt/observability/request_metrics_exporter.py:72-156`
- 文件名：`sglang-request-metrics-${YYYYMMDD_HH}.log`
- health check 请求不写入文件

```bash
ls -lh "$METRICS_FILE_DIR"
tail -n 20 "$METRICS_FILE_DIR"/sglang-request-metrics-*.log
grep '"finish_reason"' "$METRICS_FILE_DIR"/sglang-request-metrics-*.log | tail
```

## 9. Prometheus `/metrics`

脚本默认 `ENABLE_METRICS=1`。

初始化路径：

- `Engine._set_envs_and_config()` 调 `set_prometheus_multiproc_dir()`：`engine.py:1119-1121`
- FastAPI lifespan 添加 `/metrics` route：`http_server.py:260-276`
- `add_prometheus_middleware()`：`python/sglang/srt/utils/common.py:1313-1341`
- HTTP response tracking middleware：`common.py:1364-1427`

常看指标：

| 指标 | 用途 |
| --- | --- |
| `sglang:http_requests_total` | endpoint 请求数 |
| `sglang:http_responses_total` | endpoint/status 响应数 |
| `sglang:http_requests_active` | 当前 HTTP active |
| `sglang:num_running_reqs` | running requests |
| `sglang:num_queue_reqs` | waiting queue |
| `sglang:token_usage` | KV/token pool 使用率 |
| `sglang:cache_hit_rate` | prefix cache 命中率 |
| `sglang:time_to_first_token_seconds` | TTFT |
| `sglang:inter_token_latency_seconds` | ITL |
| `sglang:e2e_request_latency_seconds` | E2E latency |
| `sglang:num_aborted_requests_total` | abort 请求数 |

```bash
curl --noproxy '*' -sS http://127.0.0.1:30000/metrics | grep -E 'sglang:(num_queue_reqs|num_running_reqs|token_usage|time_to_first_token|inter_token_latency|e2e_request_latency)'
```

`/metrics` 在 auth 层按前缀放行。

## 10. 验证脚本

主要验证脚本：

```bash
docs_always/qwen3.6-27b/verify_qwen36_27b.py
```

能力包括 `/health`、`/v1/models`、错误 key、非流式 chat、流式 chat、并发、长上下文。注意该脚本默认值与当前 agent 启动脚本不完全一致：

| 项 | verify 默认 | 当前 agent 脚本默认 |
| --- | --- | --- |
| base URL | `http://127.0.0.1:18080/v1` | `http://127.0.0.1:30000/v1` |
| expected context | `131072` | `262144` |
| model | `qwen3.6-27b` | `qwen3.6-27b` |

当前本地服务建议显式传参：

```bash
python docs_always/qwen3.6-27b/verify_qwen36_27b.py \
  --base-url http://127.0.0.1:30000/v1 \
  --model qwen3.6-27b \
  --expected-context-length 262144 \
  --skip-long-context
```

长上下文验收：

```bash
python docs_always/qwen3.6-27b/verify_qwen36_27b.py \
  --base-url http://127.0.0.1:30000/v1 \
  --model qwen3.6-27b \
  --expected-context-length 262144 \
  --target-tokens 100000 \
  --long-max-tokens 64 \
  --long-timeout 900
```

当前 `docs_always/qwen3.6-27b/openai_call.py` 是空文件，不应作为验证入口。

## 11. 故障分类

### 启动失败

先看：

```bash
tail -n 200 "$START_LOG_FILE"
grep "ERROR:" "$START_LOG_FILE"
```

常见原因：Python 不可执行、模型路径不存在、API key 为空、GPU 数不足、GPU 已占用过多、端口占用、`ServerArgs` 校验失败。

### Ready 超时

排查：

```bash
kill -0 "$(cat "$PID_FILE")"
tail -n 200 "$SERVER_LOG_FILE"
grep -E "server_args=|max_total_num_tokens|available_gpu_mem|Traceback|OutOfMemory|Health check failed" "$SERVER_LOG_FILE"
curl --noproxy '*' -i http://127.0.0.1:30000/health
```

代码入口：ready loop 在 `start_qwen36_27b_agent.sh:412-436`，`/health` 在 `http_server.py:476-548`。

### 队列满

`MAX_QUEUED_REQUESTS = MAX_RUNNING_REQUESTS * 8`，见 `start_qwen36_27b_agent.sh:245`。队列满判断在 scheduler 层：

- `_add_request_to_queue()`：`python/sglang/srt/managers/scheduler.py:1936-1958`
- `_abort_on_queued_limit()`：`scheduler.py:1985-2032`

```bash
grep -R "The request queue is full" "$REQUEST_LOG_DIR" "$METRICS_FILE_DIR" "$SERVER_LOG_FILE"
curl --noproxy '*' -sS http://127.0.0.1:30000/metrics | grep 'sglang:num_queue_reqs'
```

提高 `MAX_QUEUED_REQUESTS` 只增加排队，不增加实际吞吐；真正容量取决于 `MAX_RUNNING_REQUESTS`、KV cache、prompt 长度和输出长度。

### OOM / KV 压力

启动期 OOM 看 `SERVER_LOG_FILE` 的 `CUDA out of memory`、`OutOfMemory`、`Traceback`，并结合 `START_LOG_FILE` 的 GPU snapshot、`MEM_FRACTION_STATIC`、`MODEL_SHARD_MIB_ESTIMATE`。

运行期 KV 压力看：

- `KV cache pool is full. Retract requests.`
- `/metrics` 的 `sglang:token_usage`、`sglang:num_running_reqs`、`sglang:num_queue_reqs`
- scheduler 初始化日志中的 `max_total_num_tokens`

处理方向：降低客户端并发、降低 `MAX_RUNNING_REQUESTS`、降低请求侧 `max_tokens`、避免多个 256K 请求同时跑。

### 首 token 慢

分解 TTFT：

```bash
grep "Req Time Stats" "$SERVER_LOG_FILE" | tail -n 20
curl --noproxy '*' -sS http://127.0.0.1:30000/metrics | grep -E 'time_to_first_token|queue_time|num_queue_reqs|cache_hit_rate|token_usage'
grep '"event": "request.finished"' "$REQUEST_LOG_DIR"/*.log | tail -n 20
```

判断：

- `queue_duration` 高：排队或 running slots 不足。
- `forward_duration` 高：prefill/decode 计算重。
- `cache_hit_rate` 低：prefix cache 未命中。
- `token_usage` 高：KV 压力大，可能伴随 retract。

### tool_calls 异常

排查：

1. `grep "tool_call_parser" "$SERVER_LOG_FILE" | head -n 1` 确认 `qwen3_coder`。
2. 请求是否传 `tools`，且 `tool_choice` 不是 `none`。
3. 非流式响应看 `choices[0].message.tool_calls`。
4. 流式响应累加 `delta.tool_calls[*].function.arguments`。
5. 查 `Tool call parsing error`。
6. SGLang 只解析 OpenAI `tool_calls` 字段，不执行工具；上层 Agent 需要执行工具并回填 `role=tool`。

## 12. 一次请求的排查路径

```text
HTTP /v1/chat/completions
  -> OpenAIServingChat
  -> TokenizerManager.generate_request()
  -> RequestLogger request.received
  -> tokenize / validate context length
  -> _send_one_request()
  -> Scheduler waiting_queue / running batch
  -> ModelRunner forward / decode
  -> DetokenizerManager
  -> TokenizerManager _handle_batch_output()
  -> RequestLogger request.finished
  -> RequestMetricsExporter JSONL
  -> Prometheus counters/histograms
```

关键代码：

| 环节 | 代码 |
| --- | --- |
| `/health` | `python/sglang/srt/entrypoints/http_server.py:476-548` |
| `/v1/models` | `http_server.py:1498-1545` |
| Auth | `python/sglang/srt/utils/auth.py:74-208` |
| request logging | `python/sglang/srt/managers/tokenizer_manager.py:344-366` |
| token 长度校验 | `tokenizer_manager.py:785-835` |
| dispatch scheduler | `tokenizer_manager.py:1095-1102` |
| request finished | `tokenizer_manager.py:1183-1201` |
| metrics exporter | `python/sglang/srt/observability/request_metrics_exporter.py:72-156` |
| Prometheus route | `python/sglang/srt/utils/common.py:1313-1341` |
| scheduler queue full | `python/sglang/srt/managers/scheduler.py:1985-2032` |
| KV retract | `scheduler.py:2561-2605` |
| tool parser registry | `python/sglang/srt/function_call/function_call_parser.py` |
| qwen3_coder detector | `python/sglang/srt/function_call/qwen3_coder_detector.py` |
