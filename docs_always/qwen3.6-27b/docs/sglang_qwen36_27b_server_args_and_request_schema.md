# Qwen3.6-27B SGLang 服务启动参数与请求结构

本文面向 `docs_always/qwen3.6-27b/start_qwen36_27b_agent.sh` 和 SGLang OpenAI 兼容接口，说明两类参数：

1. 启动 LLM 服务时传给 `python -m sglang.launch_server` / `sglang serve` 的 `ServerArgs`。
2. 调用 `/v1/chat/completions` 时，curl `-d` 里的 JSON 请求体结构。

主要代码依据：

- `ServerArgs` 字段定义：`python/sglang/srt/server_args.py:287-606`
- CLI 参数注册：`python/sglang/srt/server_args.py:3633-5963`
- CLI 到 dataclass 的映射：`python/sglang/srt/server_args.py:5965-5975`
- 参数解析入口：`python/sglang/srt/server_args.py:6515-6539`
- OpenAI chat 请求模型：`python/sglang/srt/entrypoints/openai/protocol.py:418-760`
- `/v1/chat/completions` 路由：`python/sglang/srt/entrypoints/http_server.py:1388-1396`
- Qwen3.6-27B agent 启动脚本：`docs_always/qwen3.6-27b/start_qwen36_27b_agent.sh`

## 1. 最小启动命令

裸启动形态：

```bash
python -m sglang.launch_server \
  --model-path /path/to/model \
  --host 0.0.0.0 \
  --port 30000
```

Qwen3.6-27B agent 当前脚本实际围绕下面这类命令构造：

```bash
python -m sglang.launch_server \
  --model-path /mnt/shanhai-ai/wenhy/models/Qwen/Qwen/Qwen3___6-27B \
  --host 127.0.0.1 \
  --port 30000 \
  --served-model-name qwen3.6-27b \
  --tensor-parallel-size 4 \
  --context-length 262144 \
  --mem-fraction-static 0.90 \
  --max-running-requests 1 \
  --max-queued-requests 8 \
  --chunked-prefill-size 8192 \
  --prefill-max-requests 1 \
  --max-prefill-tokens 16384 \
  --schedule-policy lpm \
  --radix-eviction-policy lru \
  --dtype bfloat16 \
  --attention-backend flashinfer \
  --sampling-backend flashinfer \
  --sampling-defaults model \
  --tool-call-parser qwen3_coder \
  --log-level info \
  --log-level-http warning \
  --api-key "$OPENAI_API_KEY" \
  --log-requests \
  --log-requests-level 3 \
  --log-requests-format json \
  --enable-request-time-stats-logging \
  --enable-metrics \
  --export-metrics-to-file \
  --disable-piecewise-cuda-graph
```

脚本会根据 GPU 显存、模型目录大小、`CONTEXT_LENGTH` 和 `KV_BYTES_PER_TOKEN_PER_GPU` 自动估算 `MEM_FRACTION_STATIC`、`MAX_RUNNING_REQUESTS`、`MAX_QUEUED_REQUESTS` 等值，所以实际日志里的命令可能和上面略有差异。

## 2. 参数从 CLI 到服务的流向

启动入口是 `python/sglang/launch_server.py`。它调用 `prepare_server_args(sys.argv[1:])` 得到 `ServerArgs`，再调用 `run_server(server_args)`。

`prepare_server_args()` 的流程：

1. 创建 `argparse.ArgumentParser(prog="sglang serve")`。
2. 调用 `ServerArgs.add_cli_args(parser)` 注册所有 CLI 参数。
3. 如果命令行带 `--config`，先把 YAML 配置合并进 CLI 参数。
4. `parser.parse_args(argv)`。
5. `ServerArgs.from_cli_args(raw_args)` 把 argparse namespace 映射成 dataclass。

注意几个别名映射：

| CLI 参数 | dataclass 字段 |
| --- | --- |
| `--tensor-parallel-size`, `--tp-size` | `tp_size` |
| `--pipeline-parallel-size`, `--pp-size` | `pp_size` |
| `--data-parallel-size`, `--dp-size` | `dp_size` |
| `--expert-parallel-size`, `--ep-size`, `--ep` | `ep_size` |
| `--attention-context-parallel-size`, `--attn-cp-size` | `attn_cp_size` |
| `--moe-data-parallel-size`, `--moe-dp-size` | `moe_dp_size` |

`ServerArgs.__post_init__()` 会做大量自动推断和校验，例如：

- `tokenizer_path` 未设置时默认等于 `model_path`。
- `served_model_name` 未设置时默认等于 `model_path`。
- `device` 未设置时自动探测。
- `random_seed` 未设置时随机生成。
- 根据 GPU 显存自动推断 `chunked_prefill_size`、`cuda_graph_max_bs`、`mem_fraction_static`。
- 根据模型结构、设备、KV dtype、MoE/LoRA/PD 等开关选择或禁用部分 backend。

## 3. Qwen3.6 agent 启动脚本变量

这些变量不是 `ServerArgs` 字段本身，而是脚本层用于拼装 CLI 的环境变量。

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `ROOT_DIR` | `/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang` | 进入仓库根目录后启动 |
| `SGLANG_PY` | `/mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/python3` | Python 解释器 |
| `MODEL_PATH` | `/mnt/shanhai-ai/wenhy/models/Qwen/Qwen/Qwen3___6-27B` | 映射到 `--model-path` |
| `SGLANG_HOST` | `127.0.0.1` | 映射到 `--host` |
| `SGLANG_PORT` | `30000` | 映射到 `--port` |
| `SERVED_MODEL_NAME` | `qwen3.6-27b` | 映射到 `--served-model-name` |
| `API_KEY_FILE` | `/etc/sglang/qwen36_openai_api_key` | 未设置 `OPENAI_API_KEY` 时读取 |
| `ALLOW_EMPTY_API_KEY` | `0` | 是否允许本地空 key 测试 |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | 控制可见 GPU |
| `TP_SIZE` | `4` | 映射到 `--tensor-parallel-size` |
| `CONTEXT_LENGTH` | `262144` | 映射到 `--context-length` |
| `MAX_OUTPUT_TOKENS` | `128000` | 写入客户端默认配置；请求侧仍需传 `max_tokens` 或 `max_completion_tokens` |
| `MEMORY_TARGET_FRACTION` | `0.90` | 脚本计算 `MEM_FRACTION_STATIC` 的目标显存比例 |
| `RESPECT_CURRENT_GPU_USAGE` | `1` | 估算可用显存时是否扣除当前已用显存 |
| `MAX_RUNNING_REQUESTS_CAP` | `8` | 自动估算并发的上限 |
| `KV_BYTES_PER_TOKEN_PER_GPU` | `16384` | 每卡每 token KV cache 字节估算 |
| `STATIC_OVERHEAD_MIB` | `2048` | 每卡静态开销估算 |
| `CHUNKED_PREFILL_SIZE` | `8192` | 映射到 `--chunked-prefill-size` |
| `MAX_PREFILL_TOKENS` | `16384` | 映射到 `--max-prefill-tokens` |
| `DTYPE` | `bfloat16` | 映射到 `--dtype` |
| `ATTENTION_BACKEND` | `flashinfer` | 映射到 `--attention-backend` |
| `SAMPLING_BACKEND` | `flashinfer` | 映射到 `--sampling-backend` |
| `TOOL_CALL_PARSER` | `qwen3_coder` | 映射到 `--tool-call-parser` |
| `SCHEDULE_POLICY` | `lpm` | 映射到 `--schedule-policy` |
| `RADIX_EVICTION_POLICY` | `lru` | 映射到 `--radix-eviction-policy` |
| `SAMPLING_DEFAULTS` | `model` | 映射到 `--sampling-defaults` |
| `LOG_DIR` | `${ROOT_DIR}/logs/qwen36_27b_agent` | 日志、PID、metrics、请求日志目录 |
| `WAIT_FOR_READY` | `1` | 启动后是否等待 `/health` ready |
| `READY_TIMEOUT_SECONDS` | `900` | 等待 ready 超时时间 |
| `DRY_RUN` | `0` | 只打印命令，不启动 |
| `EXTRA_SERVER_ARGS` | 未设置 | 追加任意额外 `ServerArgs` CLI |

脚本级校验包括：

- Python 可执行文件必须存在。
- 模型目录必须存在。
- `OPENAI_API_KEY` 默认不能为空，除非显式 `ALLOW_EMPTY_API_KEY=1`。
- `MAX_OUTPUT_TOKENS` 必须小于 `CONTEXT_LENGTH`。
- 启动前检查端口是否已被监听。

## 4. 核心启动参数分组

### 4.1 模型与 tokenizer

| CLI | 字段 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--model-path`, `--model` | `model_path` | 必填 | 模型权重路径，可以是本地目录或 Hugging Face repo id |
| `--tokenizer-path` | `tokenizer_path` | `None` | tokenizer 路径；未设置时默认等于 `model_path` |
| `--tokenizer-mode` | `tokenizer_mode` | `auto` | `auto` 优先 fast tokenizer；`slow` 强制慢 tokenizer |
| `--tokenizer-worker-num` | `tokenizer_worker_num` | `1` | tokenizer manager worker 数；必须大于 0 |
| `--skip-tokenizer-init` | `skip_tokenizer_init` | `False` | 跳过 tokenizer 初始化；请求需要传 token ids |
| `--load-format` | `load_format` | `auto` | 权重加载格式，常见 `auto`、`pt`、`safetensors`、`gguf`、`bitsandbytes`、`dummy`、`layered` |
| `--model-loader-extra-config` | `model_loader_extra_config` | `{}` | 传给对应 model loader 的额外 JSON 配置 |
| `--trust-remote-code` | `trust_remote_code` | `False` | 允许加载 Hub 上自定义模型代码 |
| `--context-length` | `context_length` | `None` | 覆盖模型最大上下文；未设时使用模型 config |
| `--is-embedding` | `is_embedding` | `False` | 把 CausalLM 当 embedding 模型使用 |
| `--enable-multimodal` | `enable_multimodal` | `None` | 启用多模态功能；非多模态模型无效果 |
| `--revision` | `revision` | `None` | Hugging Face 分支、tag 或 commit |
| `--model-impl` | `model_impl` | `auto` | `auto`、`sglang`、`transformers`、`mindspore` |

Qwen3.6-27B agent 当前最关键的是：

```bash
--model-path "$MODEL_PATH"
--context-length 262144
--dtype bfloat16
--model-impl auto
```

### 4.2 HTTP 服务与 TLS

| CLI | 字段 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--host` | `host` | `127.0.0.1` | HTTP 监听地址；公网或容器内对外暴露常用 `0.0.0.0` |
| `--port` | `port` | `30000` | HTTP 端口 |
| `--fastapi-root-path` | `fastapi_root_path` | 空字符串 | 服务在反向代理子路径下时设置 |
| `--grpc-mode` | `grpc_mode` | `False` | 使用 gRPC server 而不是 HTTP server |
| `--skip-server-warmup` | `skip_server_warmup` | `False` | 跳过 warmup |
| `--warmups` | `warmups` | `None` | 指定启动前 warmup 函数列表 |
| `--nccl-port` | `nccl_port` | `None` | NCCL 初始化端口；未设时随机端口 |
| `--checkpoint-engine-wait-weights-before-ready` | `checkpoint_engine_wait_weights_before_ready` | `False` | 等初始权重通过 checkpoint/update 机制加载后才 ready |
| `--ssl-keyfile` | `ssl_keyfile` | `None` | SSL key 文件 |
| `--ssl-certfile` | `ssl_certfile` | `None` | SSL certificate 文件 |
| `--ssl-ca-certs` | `ssl_ca_certs` | `None` | CA 证书文件 |
| `--ssl-keyfile-password` | `ssl_keyfile_password` | `None` | keyfile 密码 |
| `--enable-ssl-refresh` | `enable_ssl_refresh` | `False` | cert/key 文件变化时热加载；要求同时设置 cert 和 key |

TLS 校验规则：

- `--ssl-keyfile` 和 `--ssl-certfile` 必须成对设置。
- 未启用 SSL 时，`--ssl-ca-certs` 和 `--ssl-keyfile-password` 没有效果并会报错。
- `--enable-ssl-refresh` 要求 SSL cert/key 都存在。

### 4.3 量化与 dtype

| CLI | 字段 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--dtype` | `dtype` | `auto` | 权重和激活 dtype；可选 `auto`、`half`、`float16`、`bfloat16`、`float`、`float32` |
| `--quantization` | `quantization` | `None` | 权重量化方法，例如 AWQ/GPTQ/GGUF 等，具体取决于 `QUANTIZATION_CHOICES` |
| `--quantization-param-path` | `quantization_param_path` | `None` | KV cache FP8 scaling factor JSON；FP8 KV cache 建议提供 |
| `--kv-cache-dtype` | `kv_cache_dtype` | `auto` | KV cache dtype；支持 `auto`、`fp8_e5m2`、`fp8_e4m3`、`bf16`、`bfloat16`、`fp4_e2m1` |
| `--enable-fp32-lm-head` | `enable_fp32_lm_head` | `False` | LM head logits 使用 FP32 |
| `--modelopt-quant` | `modelopt_quant` | `None` | NVIDIA ModelOpt 量化配置，如 `fp8`、`int4_awq`、`w4a8_awq`、`nvfp4` |
| `--modelopt-checkpoint-restore-path` | `modelopt_checkpoint_restore_path` | `None` | 从已保存的 ModelOpt 量化 checkpoint 恢复 |
| `--modelopt-checkpoint-save-path` | `modelopt_checkpoint_save_path` | `None` | 量化后保存 ModelOpt checkpoint |
| `--modelopt-export-path` | `modelopt_export_path` | `None` | 量化后导出 HuggingFace 格式 |
| `--quantize-and-serve` | `quantize_and_serve` | `False` | ModelOpt 量化后直接启动服务 |
| `--rl-quant-profile` | `rl_quant_profile` | `None` | `--load-format flash_rl` 时所需 profile |

Qwen3.6-27B 当前用 `--dtype bfloat16`。如果显存不足，再考虑 KV cache FP8 或权重量化；精度敏感服务不建议先从量化入手。

### 4.4 显存、KV cache 与调度

| CLI | 字段 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--mem-fraction-static` | `mem_fraction_static` | `None` | 静态显存比例，覆盖模型权重和 KV cache pool；OOM 时先调小 |
| `--max-running-requests` | `max_running_requests` | `None` | 同时运行的请求上限 |
| `--max-queued-requests` | `max_queued_requests` | `None` | 排队请求上限；PD disaggregation 下忽略 |
| `--max-total-tokens` | `max_total_tokens` | `None` | KV cache token pool 上限；未设时按显存比例估算 |
| `--chunked-prefill-size` | `chunked_prefill_size` | `None` | chunked prefill 单块最大 token 数；`-1` 表示禁用 |
| `--enable-dynamic-chunking` | `enable_dynamic_chunking` | `False` | pipeline parallel 下动态调整 chunk size |
| `--max-prefill-tokens` | `max_prefill_tokens` | `16384` | prefill batch token 上限；实际约束会考虑模型最大上下文 |
| `--prefill-max-requests` | `prefill_max_requests` | `None` | prefill batch 请求数上限 |
| `--schedule-policy` | `schedule_policy` | `fcfs` | 调度策略：`lpm`、`random`、`fcfs`、`dfs-weight`、`lof`、`priority`、`routing-key` |
| `--enable-priority-scheduling` | `enable_priority_scheduling` | `False` | 启用请求优先级调度 |
| `--disable-priority-preemption` | `disable_priority_preemption` | `False` | 禁用优先级抢占 |
| `--default-priority-value` | `default_priority_value` | `None` | 未显式传 priority 的请求默认优先级 |
| `--abort-on-priority-when-disabled` | `abort_on_priority_when_disabled` | `False` | 未启用优先级调度时，拒绝带 priority 的请求 |
| `--schedule-low-priority-values-first` | `schedule_low_priority_values_first` | `False` | 优先调度较小 priority 值 |
| `--priority-scheduling-preemption-threshold` | `priority_scheduling_preemption_threshold` | `10` | 触发抢占所需优先级差值 |
| `--schedule-conservativeness` | `schedule_conservativeness` | `1.0` | 调度保守程度；频繁 retract 时调大 |
| `--page-size` | `page_size` | `None` | KV cache page token 数 |
| `--swa-full-tokens-ratio` | `swa_full_tokens_ratio` | `0.8` | SWA 层 KV token / full layer KV token 比例 |
| `--disable-hybrid-swa-memory` | `disable_hybrid_swa_memory` | `False` | 禁用混合 SWA memory pool |
| `--radix-eviction-policy` | `radix_eviction_policy` | `lru` | radix tree 淘汰策略：`lru`、`lfu`、`slru` 等 |
| `--enable-prefill-delayer` | `enable_prefill_delayer` | `False` | DP attention 下延迟 prefill，减少 idle |
| `--prefill-delayer-max-delay-passes` | `prefill_delayer_max_delay_passes` | `30` | prefill 最多延迟多少个 forward pass |
| `--prefill-delayer-token-usage-low-watermark` | `prefill_delayer_token_usage_low_watermark` | `None` | token 使用低水位线 |
| `--prefill-delayer-forward-passes-buckets` | `prefill_delayer_forward_passes_buckets` | `None` | forward pass histogram bucket |
| `--prefill-delayer-wait-seconds-buckets` | `prefill_delayer_wait_seconds_buckets` | `None` | 等待时间 histogram bucket |

长上下文服务的主要调参顺序：

1. OOM：先降低 `--mem-fraction-static`，或降低 `--max-running-requests`。
2. 首 token 慢：检查 `--chunked-prefill-size`、`--max-prefill-tokens` 和 `--schedule-policy`。
3. Prefix cache 命中重要：使用 `--schedule-policy lpm` 和合适的 `--radix-eviction-policy`。
4. 多并发导致排队：提高 `--max-running-requests` 之前先确认 KV cache 预算够。

### 4.5 运行时与并行

| CLI | 字段 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--device` | `device` | `None` | `cuda`、`xpu`、`hpu`、`npu`、`cpu`；未设时自动探测 |
| `--tensor-parallel-size`, `--tp-size` | `tp_size` | `1` | Tensor parallel 大小；单机多 GPU 最常用 |
| `--attention-context-parallel-size`, `--attn-cp-size` | `attn_cp_size` | `1` | attention context parallel 大小 |
| `--moe-data-parallel-size`, `--moe-dp-size` | `moe_dp_size` | `1` | MoE data parallel 大小 |
| `--pipeline-parallel-size`, `--pp-size` | `pp_size` | `1` | Pipeline parallel 大小 |
| `--pp-max-micro-batch-size` | `pp_max_micro_batch_size` | `None` | PP 最大 micro batch size |
| `--pp-async-batch-depth` | `pp_async_batch_depth` | `0` | PP async batch depth |
| `--stream-interval` | `stream_interval` | `1` | 流式输出多少 token flush 一次；越小越平滑，越大吞吐更好 |
| `--incremental-streaming-output` | `incremental_streaming_output` | `False` | 流式输出为互不重叠片段 |
| `--stream-response-default-include-usage` | `stream_response_default_include_usage` | `False` | stream 响应默认包含 usage |
| `--enable-streaming-session` | `enable_streaming_session` | `False` | 启用 streaming session 和 SessionAwareCache |
| `--random-seed` | `random_seed` | `None` | 随机种子；未设时随机生成 |
| `--constrained-json-whitespace-pattern` | `constrained_json_whitespace_pattern` | `None` | outlines/llguidance JSON 约束输出空白正则 |
| `--constrained-json-disable-any-whitespace` | `constrained_json_disable_any_whitespace` | `False` | xgrammar/llguidance 下强制紧凑 JSON |
| `--watchdog-timeout` | `watchdog_timeout` | `300` | forward batch 超时后 crash，避免 hang |
| `--soft-watchdog-timeout` | `soft_watchdog_timeout` | `None` | soft 超时，只 dump debug 信息 |
| `--dist-timeout` | `dist_timeout` | `None` | torch.distributed 初始化超时 |
| `--download-dir` | `download_dir` | `None` | HF/ModelScope 下载目录 |
| `--model-checksum` | `model_checksum` | `None` | 模型文件完整性校验 |
| `--base-gpu-id` | `base_gpu_id` | `0` | 分配 GPU 的起始 id |
| `--gpu-id-step` | `gpu_id_step` | `1` | GPU id 步长，如 `2` 表示用 `0,2,4,...` |
| `--sleep-on-idle` | `sleep_on_idle` | `False` | idle 时降低 CPU 占用 |
| `--use-ray` | `use_ray` | `False` | 用 Ray actor 管理 scheduler 进程 |
| `--custom-sigquit-handler` | `custom_sigquit_handler` | `None` | Engine 场景自定义 SIGQUIT 清理逻辑 |

并行约束：

- `tp_size * pp_size` 必须能被 `nnodes` 整除。
- `pp_size > 1` 时要求禁用 overlap schedule，且不能和 speculative decoding、mixed chunked prefill 同用。
- 多机 data parallel 默认不支持，除非启用 DP attention。

### 4.6 日志、metrics 与 tracing

| CLI | 字段 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--log-level` | `log_level` | `info` | 全局日志级别 |
| `--log-level-http` | `log_level_http` | `None` | HTTP server 日志级别；未设时复用 `--log-level` |
| `--log-requests` | `log_requests` | `False` | 记录请求元数据、输入、输出 |
| `--log-requests-level` | `log_requests_level` | `2` | `0` 元数据；`1` 加采样参数；`2` 加部分输入输出；`3` 完整输入输出 |
| `--log-requests-format` | `log_requests_format` | `text` | `text` 或 `json` |
| `--log-requests-target` | `log_requests_target` | `None` | `stdout` 或目录路径，可多个 |
| `--uvicorn-access-log-exclude-prefixes` | `uvicorn_access_log_exclude_prefixes` | 默认列表 | 排除指定路径前缀的 uvicorn access log，如 `/health`、`/metrics` |
| `--crash-dump-folder` | `crash_dump_folder` | `None` | crash 前最近请求 dump 目录 |
| `--show-time-cost` | `show_time_cost` | `False` | 打印自定义 mark 的耗时 |
| `--enable-metrics` | `enable_metrics` | `False` | 开启 Prometheus metrics |
| `--enable-mfu-metrics` | `enable_mfu_metrics` | `False` | 开启 MFU 估算 metrics |
| `--enable-metrics-for-all-schedulers` | `enable_metrics_for_all_schedulers` | `False` | 所有 TP scheduler 记录 metrics；DP attention 场景有用 |
| `--tokenizer-metrics-custom-labels-header` | `tokenizer_metrics_custom_labels_header` | `x-custom-labels` | 从哪个 header 读取 tokenizer metrics 自定义 labels |
| `--tokenizer-metrics-allowed-custom-labels` | `tokenizer_metrics_allowed_custom_labels` | `None` | 允许的自定义 label 名 |
| `--extra-metric-labels` | `extra_metric_labels` | `None` | 服务级固定 metrics labels |
| `--bucket-time-to-first-token` | `bucket_time_to_first_token` | `None` | TTFT histogram buckets |
| `--bucket-inter-token-latency` | `bucket_inter_token_latency` | `None` | ITL histogram buckets |
| `--bucket-e2e-request-latency` | `bucket_e2e_request_latency` | `None` | E2E latency histogram buckets |
| `--collect-tokens-histogram` | `collect_tokens_histogram` | `False` | 收集 prompt/generation token histogram |
| `--prompt-tokens-buckets` | `prompt_tokens_buckets` | `None` | prompt token bucket 规则 |
| `--generation-tokens-buckets` | `generation_tokens_buckets` | `None` | generation token bucket 规则 |
| `--gc-warning-threshold-secs` | `gc_warning_threshold_secs` | `0.0` | GC 超过阈值时 warning；`0` 关闭 |
| `--decode-log-interval` | `decode_log_interval` | `40` | decode batch 日志和 metrics 间隔 |
| `--enable-request-time-stats-logging` | `enable_request_time_stats_logging` | `False` | 记录单请求耗时统计 |
| `--kv-events-config` | `kv_events_config` | `None` | NVIDIA dynamo KV event 发布配置 |
| `--enable-trace` | `enable_trace` | `False` | 开启 OpenTelemetry trace |
| `--otlp-traces-endpoint` | `otlp_traces_endpoint` | `localhost:4317` | OTLP collector 地址 |
| `--export-metrics-to-file` | `export_metrics_to_file` | `False` | 每请求性能指标写本地文件 |
| `--export-metrics-to-file-dir` | `export_metrics_to_file_dir` | `None` | metrics 文件目录；开启导出时必填 |

请求日志等级建议：

- 生产默认：不开 `--log-requests`，或只开 level 0/1。
- 调试 prompt / tool call：用 level 2。
- level 3 会记录完整输入输出，容易产生大量日志并暴露敏感数据。

### 4.7 OpenAI API 与模板

| CLI | 字段 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--api-key` | `api_key` | `None` | 服务 API key；OpenAI 兼容接口鉴权使用 |
| `--admin-api-key` | `admin_api_key` | `None` | 管理端点 key；设置后敏感管理端点不接受普通 `api_key` |
| `--served-model-name` | `served_model_name` | `None` | `/v1/models` 返回的模型名；未设时默认 `model_path` |
| `--weight-version` | `weight_version` | `default` | 权重版本标识 |
| `--chat-template` | `chat_template` | `None` | 内置 chat template 名或模板文件路径 |
| `--hf-chat-template-name` | `hf_chat_template_name` | `None` | HF tokenizer 有多个 chat template 时指定名字 |
| `--completion-template` | `completion_template` | `None` | completion template 名或文件路径；主要用于代码补全 |
| `--file-storage-path` | `file_storage_path` | `sglang_storage` | 后端文件存储路径 |
| `--enable-cache-report` | `enable_cache_report` | `False` | 在 OpenAI usage 里返回 cached token 数 |
| `--reasoning-parser` | `reasoning_parser` | `None` | reasoning 模型输出解析器 |
| `--tool-call-parser` | `tool_call_parser` | `None` | tool call 输出解析器 |
| `--tool-server` | `tool_server` | `None` | `demo` 或 tool server URL 列表 |
| `--sampling-defaults` | `sampling_defaults` | `model` | 默认采样参数来源：`model` 使用模型 `generation_config.json`，`openai` 使用 SGLang/OpenAI 默认值 |

Qwen3.6 agent 当前设置 `--tool-call-parser qwen3_coder`，适合 agent/tool call 输出解析。

### 4.8 Data parallel、多机与模型覆盖

| CLI | 字段 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--data-parallel-size`, `--dp-size` | `dp_size` | `1` | data parallel 大小 |
| `--load-balance-method` | `load_balance_method` | `auto` | DP 负载均衡：`auto`、`round_robin`、`follow_bootstrap_room`、`total_requests`、`total_tokens` |
| `--dist-init-addr`, `--nccl-init-addr` | `dist_init_addr` | `None` | 分布式初始化地址，如 `192.168.0.2:25000` |
| `--nnodes` | `nnodes` | `1` | 节点数 |
| `--node-rank` | `node_rank` | `0` | 当前节点 rank |
| `--json-model-override-args` | `json_model_override_args` | `{}` | 用 JSON 覆盖模型 config |
| `--preferred-sampling-params` | `preferred_sampling_params` | `None` | `/get_model_info` 返回的推荐采样参数 |

`load_balance_method=auto` 会根据 disaggregation mode 自动选择：

- 非 PD：`round_robin`
- PD prefill：`follow_bootstrap_room`
- PD decode：`round_robin`

### 4.9 LoRA

| CLI | 字段 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--enable-lora` | `enable_lora` | `None` | 启用 LoRA；传了 `--lora-paths` 时会自动启用 |
| `--enable-lora-overlap-loading` | `enable_lora_overlap_loading` | `None` | 异步加载 LoRA 权重，用 H2D transfer overlap GPU compute |
| `--max-lora-rank` | `max_lora_rank` | `None` | LoRA adapter 最大 rank；未设时从 adapter 推断 |
| `--lora-target-modules` | `lora_target_modules` | `None` | LoRA 作用模块；可用 `all` 表示全部支持模块 |
| `--lora-paths` | `lora_paths` | `None` | adapter 列表，格式 `<PATH>`、`<NAME>=<PATH>` 或 JSON |
| `--max-loaded-loras` | `max_loaded_loras` | `None` | CPU 内存同时加载的 LoRA 数上限 |
| `--max-loras-per-batch` | `max_loras_per_batch` | `8` | 单个 running batch 内 adapter 数上限，包含 base-only 请求 |
| `--lora-eviction-policy` | `lora_eviction_policy` | `lru` | adapter 淘汰策略：`lru` 或 `fifo` |
| `--lora-backend` | `lora_backend` | `csgmv` | multi-LoRA kernel backend |
| `--max-lora-chunk-size` | `max_lora_chunk_size` | `16` | CSGMV backend chunk size，可选 `16/32/64/128` |
| `--experts-shared-outer-loras`, `--no-experts-shared-outer-loras` | `experts_shared_outer_loras` | `None` | MoE 模型共享 outer LoRA 模式，默认从 adapter 权重自动探测 |

LoRA 相关校验：

- `max_loras_per_batch` 必须大于 0。
- 开启 overlap loading 时，`max_loaded_loras` 必须不超过 `2 * max_loras_per_batch`。
- 当前 LoRA 只兼容 `NGRAM` speculative decoding 或不启用 speculative decoding。
- 没有初始 `lora_paths` 时，必须同时指定 `max_lora_rank` 和 `lora_target_modules`。

### 4.10 Kernel backend

| CLI | 字段 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--attention-backend` | `attention_backend` | `None` | attention kernel 总开关 |
| `--prefill-attention-backend` | `prefill_attention_backend` | `None` | prefill attention backend，优先级高于 `attention_backend` |
| `--decode-attention-backend` | `decode_attention_backend` | `None` | decode attention backend，优先级高于 `attention_backend` |
| `--sampling-backend` | `sampling_backend` | `None` | sampling kernel backend |
| `--grammar-backend` | `grammar_backend` | `None` | grammar-guided decoding backend |
| `--mm-attention-backend` | `mm_attention_backend` | `None` | 多模态 attention backend |
| `--fp8-gemm-backend` | `fp8_gemm_runner_backend` | `auto` | FP8 GEMM runner backend |
| `--fp4-gemm-backend` | `fp4_gemm_runner_backend` | `auto` | NVFP4 GEMM runner backend |
| `--nsa-prefill-backend` | `nsa_prefill_backend` | `None` | NSA prefill backend，默认按硬件和 KV dtype 自动选择 |
| `--nsa-decode-backend` | `nsa_decode_backend` | `None` | NSA decode backend，默认按硬件和 KV dtype 自动选择 |
| `--disable-flashinfer-autotune` | `disable_flashinfer_autotune` | `False` | 禁用 FlashInfer autotune |
| `--mamba-backend` | `mamba_backend` | `triton` | Mamba SSM backend，常见 `triton`、`flashinfer` |

Qwen3.6 agent 当前设置：

```bash
--attention-backend flashinfer
--sampling-backend flashinfer
```

### 4.11 Speculative decoding

| CLI | 字段 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--speculative-algorithm` | `speculative_algorithm` | `None` | `EAGLE`、`EAGLE3`、`NEXTN`、`STANDALONE`、`NGRAM` |
| `--speculative-draft-model-path`, `--speculative-draft-model` | `speculative_draft_model_path` | `None` | draft model 路径或 HF repo id |
| `--speculative-draft-model-revision` | `speculative_draft_model_revision` | `None` | draft model 版本 |
| `--speculative-draft-load-format` | `speculative_draft_load_format` | `None` | draft model 加载格式；未设时跟随 `load_format` |
| `--speculative-num-steps` | `speculative_num_steps` | `None` | speculative decoding 步数 |
| `--speculative-eagle-topk` | `speculative_eagle_topk` | `None` | EAGLE 每步采样 token 数 |
| `--speculative-num-draft-tokens` | `speculative_num_draft_tokens` | `None` | draft token 数 |
| `--speculative-accept-threshold-single` | `speculative_accept_threshold_single` | `1.0` | 单 token 接受阈值 |
| `--speculative-accept-threshold-acc` | `speculative_accept_threshold_acc` | `1.0` | 累计接受阈值 |
| `--speculative-token-map` | `speculative_token_map` | `None` | draft model 小词表映射 |
| `--speculative-attention-mode` | `speculative_attention_mode` | `prefill` | speculative 操作使用 prefill 或 decode attention |
| `--speculative-draft-attention-backend` | `speculative_draft_attention_backend` | `None` | draft 阶段 attention backend |
| `--speculative-moe-runner-backend` | `speculative_moe_runner_backend` | `None` | speculative MoE runner backend |
| `--speculative-moe-a2a-backend` | `speculative_moe_a2a_backend` | `None` | speculative MoE A2A backend |
| `--speculative-draft-model-quantization` | `speculative_draft_model_quantization` | `None` | draft model 量化方式；未设时跟随主模型量化 |
| `--speculative-ngram-min-bfs-breadth` | `speculative_ngram_min_bfs_breadth` | `1` | NGRAM BFS 最小宽度 |
| `--speculative-ngram-max-bfs-breadth` | `speculative_ngram_max_bfs_breadth` | `10` | NGRAM BFS 最大宽度 |
| `--speculative-ngram-match-type` | `speculative_ngram_match_type` | `BFS` | `BFS` 或 `PROB` |
| `--speculative-ngram-max-trie-depth` | `speculative_ngram_max_trie_depth` | `18` | NGRAM trie 最大深度 |
| `--speculative-ngram-capacity` | `speculative_ngram_capacity` | `10000000` | NGRAM cache 容量 |
| `--enable-multi-layer-eagle` | `enable_multi_layer_eagle` | `False` | 启用 multi-layer EAGLE |

Speculative decoding 通常会改变显存占用和 CUDA graph 策略。Qwen3.6 当前启动脚本没有启用。

### 4.12 Expert parallelism / MoE

| CLI | 字段 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--expert-parallel-size`, `--ep-size`, `--ep` | `ep_size` | `1` | expert parallel 大小 |
| `--moe-a2a-backend` | `moe_a2a_backend` | `none` | MoE all-to-all backend，如 `deepep`、`mooncake`、`nixl`、`flashinfer` |
| `--moe-runner-backend` | `moe_runner_backend` | `auto` | MoE runner backend |
| `--flashinfer-mxfp4-moe-precision` | `flashinfer_mxfp4_moe_precision` | `default` | FlashInfer MXFP4 MoE 计算精度 |
| `--enable-flashinfer-allreduce-fusion` | `enable_flashinfer_allreduce_fusion` | `False` | FlashInfer allreduce + Residual RMSNorm fusion |
| `--enforce-disable-flashinfer-allreduce-fusion` | `enforce_disable_flashinfer_allreduce_fusion` | `False` | 强制关闭 FlashInfer allreduce fusion |
| `--enable-aiter-allreduce-fusion` | `enable_aiter_allreduce_fusion` | `False` | Aiter allreduce fusion |
| `--deepep-mode` | `deepep_mode` | `auto` | DeepEP 模式：`normal`、`low_latency`、`auto` |
| `--ep-num-redundant-experts` | `ep_num_redundant_experts` | `0` | EP 中冗余 expert 数 |
| `--ep-dispatch-algorithm` | `ep_dispatch_algorithm` | `None` | 冗余 expert rank 选择算法 |
| `--init-expert-location` | `init_expert_location` | `trivial` | expert 初始位置 |
| `--enable-eplb` | `enable_eplb` | `False` | 启用 EPLB |
| `--eplb-algorithm` | `eplb_algorithm` | `auto` | EPLB 算法 |
| `--eplb-rebalance-num-iterations` | `eplb_rebalance_num_iterations` | `1000` | 触发 rebalance 的迭代数 |
| `--eplb-rebalance-layers-per-chunk` | `eplb_rebalance_layers_per_chunk` | `None` | 每次 forward rebalance 的层数 |
| `--eplb-min-rebalancing-utilization-threshold` | `eplb_min_rebalancing_utilization_threshold` | `1.0` | 触发 EPLB 的 GPU 平均利用率阈值 |
| `--expert-distribution-recorder-mode` | `expert_distribution_recorder_mode` | `None` | expert 分布记录模式 |
| `--expert-distribution-recorder-buffer-size` | `expert_distribution_recorder_buffer_size` | `None` | 记录 buffer 大小；`-1` 表示无限 |
| `--enable-expert-distribution-metrics` | `enable_expert_distribution_metrics` | `False` | 记录 expert balance metrics |
| `--deepep-config` | `deepep_config` | `None` | DeepEP 调优配置，JSON 字符串或文件 |
| `--moe-dense-tp-size` | `moe_dense_tp_size` | `None` | MoE dense MLP 层 TP size；当前只支持 `1` 或 `None` |
| `--elastic-ep-backend` | `elastic_ep_backend` | `None` | elastic EP collective backend：`mooncake`、`nixl` |
| `--enable-elastic-expert-backup` | `enable_elastic_expert_backup` | `False` | 启用 elastic expert backup |
| `--mooncake-ib-device` | `mooncake_ib_device` | `None` | Mooncake backend InfiniBand 设备 |

这组参数主要用于 MoE 模型。普通 dense Qwen3.6-27B 服务通常不用改。

### 4.13 Mamba、linear attention 与 cache 扩展

| CLI | 字段 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--max-mamba-cache-size` | `max_mamba_cache_size` | `None` | Mamba cache 最大大小 |
| `--mamba-ssm-dtype` | `mamba_ssm_dtype` | `None` | Mamba SSM state dtype；未设时从模型 config 读取 |
| `--mamba-full-memory-ratio` | `mamba_full_memory_ratio` | `0.9` | Mamba state memory / full KV cache memory 比例 |
| `--mamba-scheduler-strategy` | `mamba_scheduler_strategy` | `auto` | Mamba radix cache 调度策略 |
| `--mamba-track-interval` | `mamba_track_interval` | `256` | decode 中追踪 Mamba state 的间隔 |
| `--linear-attn-backend` | `linear_attn_backend` | `triton` | linear attention 默认 backend |
| `--linear-attn-decode-backend` | `linear_attn_decode_backend` | `None` | decode 阶段 linear attention backend |
| `--linear-attn-prefill-backend` | `linear_attn_prefill_backend` | `None` | prefill/extend 阶段 linear attention backend |
| `--enable-hierarchical-cache` | `enable_hierarchical_cache` | `False` | 启用分层 KV cache |
| `--hicache-ratio` | `hicache_ratio` | `2.0` | host KV cache pool / device pool 大小比例 |
| `--hicache-size` | `hicache_size` | `0` | host KV cache pool 大小，GB；设置后覆盖 ratio |
| `--hicache-write-policy` | `hicache_write_policy` | `write_through` | `write_back`、`write_through`、`write_through_selective` |
| `--hicache-io-backend` | `hicache_io_backend` | `kernel` | CPU/GPU KV cache transfer backend |
| `--hicache-mem-layout` | `hicache_mem_layout` | `layer_first` | host memory pool layout |
| `--hicache-storage-backend` | `hicache_storage_backend` | `None` | L3 storage backend：`file`、`mooncake`、`hf3fs`、`nixl`、`aibrix`、`dynamic`、`eic` |
| `--hicache-storage-prefetch-policy` | `hicache_storage_prefetch_policy` | `best_effort` | storage prefetch 停止策略 |
| `--hicache-storage-backend-extra-config` | `hicache_storage_backend_extra_config` | `None` | storage backend 额外 JSON/YAML/TOML 配置 |
| `--enable-hisparse` | `enable_hisparse` | `False` | 启用 hierarchical sparse attention |
| `--hisparse-config` | `hisparse_config` | `None` | HiSparse JSON 配置 |
| `--enable-lmcache` | `enable_lmcache` | `False` | 使用 LMCache 作为替代分层 cache 方案 |

HiSparse 当前校验要求 DSA 模型，并要求 `--disable-radix-cache`，同时 NSA backend 必须是 `flashmla_sparse` 或未显式设置。

### 4.14 KTransformers、DLLM、Double Sparsity、Offload

| CLI | 字段 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--kt-weight-path` | `kt_weight_path` | `None` | KTransformers AMX kernel 量化 expert 权重路径 |
| `--kt-method` | `kt_method` | `AMXINT4` | CPU 执行量化格式 |
| `--kt-cpuinfer` | `kt_cpuinfer` | `None` | CPUInfer 线程数 |
| `--kt-threadpool-count` | `kt_threadpool_count` | `2` | NUMA thread pool 数 |
| `--kt-num-gpu-experts` | `kt_num_gpu_experts` | `None` | GPU experts 数 |
| `--kt-max-deferred-experts-per-token` | `kt_max_deferred_experts_per_token` | `None` | 每 token 最多延迟到 CPU 的 expert 数 |
| `--dllm-algorithm` | `dllm_algorithm` | `None` | Diffusion LLM 算法，如 `LowConfidence` |
| `--dllm-algorithm-config` | `dllm_algorithm_config` | `None` | Diffusion LLM YAML 配置 |
| `--enable-double-sparsity` | `enable_double_sparsity` | `False` | 启用 double sparsity attention |
| `--ds-channel-config-path` | `ds_channel_config_path` | `None` | double sparsity channel config |
| `--ds-heavy-channel-num` | `ds_heavy_channel_num` | `32` | heavy channel 数 |
| `--ds-heavy-token-num` | `ds_heavy_token_num` | `256` | heavy token 数 |
| `--ds-heavy-channel-type` | `ds_heavy_channel_type` | `qk` | heavy channel 类型 |
| `--ds-sparse-decode-threshold` | `ds_sparse_decode_threshold` | `4096` | decode 序列达到多少后切 sparse kernel |
| `--cpu-offload-gb` | `cpu_offload_gb` | `0` | CPU offload 预留 RAM，GB |
| `--offload-group-size` | `offload_group_size` | `-1` | offload 每组层数 |
| `--offload-num-in-group` | `offload_num_in_group` | `1` | 每组 offload 层数 |
| `--offload-prefetch-step` | `offload_prefetch_step` | `1` | offload 预取步数 |
| `--offload-mode` | `offload_mode` | `cpu` | offload 模式 |

这几组都是特定模型或特定部署形态才需要的参数。Qwen3.6 常规 GPU 服务一般不启用。

### 4.15 优化与 debug

| CLI | 字段 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--disable-radix-cache` | `disable_radix_cache` | `False` | 关闭 RadixAttention prefix cache |
| `--cuda-graph-max-bs` | `cuda_graph_max_bs` | `None` | CUDA graph 最大 batch size |
| `--cuda-graph-bs` | `cuda_graph_bs` | `None` | CUDA graph capture batch size 列表 |
| `--disable-cuda-graph` | `disable_cuda_graph` | `False` | 禁用 CUDA graph |
| `--disable-cuda-graph-padding` | `disable_cuda_graph_padding` | `False` | 需要 padding 时不用 CUDA graph |
| `--enable-profile-cuda-graph` | `enable_profile_cuda_graph` | `False` | profile CUDA graph capture |
| `--enable-cudagraph-gc` | `enable_cudagraph_gc` | `False` | CUDA graph capture 期间启用 GC |
| `--enable-layerwise-nvtx-marker` | `enable_layerwise_nvtx_marker` | `False` | 开启逐层 NVTX 标注 |
| `--enable-nccl-nvls` | `enable_nccl_nvls` | `False` | 可用时对 prefill heavy 请求启用 NCCL NVLS |
| `--enable-symm-mem` | `enable_symm_mem` | `False` | 启用 NCCL symmetric memory |
| `--disable-flashinfer-cutlass-moe-fp4-allgather` | `disable_flashinfer_cutlass_moe_fp4_allgather` | `False` | 禁用 FlashInfer CUTLASS MoE FP4 allgather 前量化 |
| `--enable-tokenizer-batch-encode` | `enable_tokenizer_batch_encode` | `False` | 文本批量 tokenization；不要和图片、预 token ids、input_embeds 混用 |
| `--disable-tokenizer-batch-decode` | `disable_tokenizer_batch_decode` | `False` | 多 completion decode 时禁用批量 decode |
| `--disable-outlines-disk-cache` | `disable_outlines_disk_cache` | `False` | 禁用 outlines 磁盘 cache |
| `--disable-custom-all-reduce` | `disable_custom_all_reduce` | `False` | 禁用自定义 all-reduce，回退 NCCL |
| `--enable-mscclpp` | `enable_mscclpp` | `False` | 小消息 all-reduce 使用 mscclpp |
| `--enable-torch-symm-mem` | `enable_torch_symm_mem` | `False` | 使用 torch symmetric memory all-reduce |
| `--pre-warm-nccl` | `pre_warm_nccl` | AMD/HIP 默认 true | 启动时预热 NCCL/RCCL |
| `--disable-overlap-schedule` | `disable_overlap_schedule` | `False` | 禁用 CPU scheduler 与 GPU worker overlap |
| `--enable-mixed-chunk` | `enable_mixed_chunk` | `False` | chunked prefill 时混合 prefill 和 decode |
| `--enable-dp-attention` | `enable_dp_attention` | `False` | attention 使用 DP、FFN 使用 TP |
| `--enable-dp-lm-head` | `enable_dp_lm_head` | `False` | DP attention 下 vocab parallel，减少 all-gather |
| `--enable-two-batch-overlap` | `enable_two_batch_overlap` | `False` | 两个 micro batch overlap |
| `--enable-single-batch-overlap` | `enable_single_batch_overlap` | `False` | 单 micro batch 内计算通信 overlap |
| `--tbo-token-distribution-threshold` | `tbo_token_distribution_threshold` | `0.48` | TBO 中 two-batch / two-chunk overlap 判定阈值 |
| `--enable-torch-compile` | `enable_torch_compile` | `False` | 使用 `torch.compile` 优化模型，实验特性 |
| `--disable-piecewise-cuda-graph` | `disable_piecewise_cuda_graph` | `False` | 禁用 extend/prefill piecewise CUDA graph |
| `--enforce-piecewise-cuda-graph` | `enforce_piecewise_cuda_graph` | `False` | 跳过自动禁用条件，强制启用，主要测试用 |
| `--enable-torch-compile-debug-mode` | `enable_torch_compile_debug_mode` | `False` | torch compile debug |
| `--torch-compile-max-bs` | `torch_compile_max_bs` | `32` | torch compile 最大 batch size |
| `--piecewise-cuda-graph-max-tokens` | `piecewise_cuda_graph_max_tokens` | `None` | piecewise CUDA graph 最大 token 数 |
| `--piecewise-cuda-graph-tokens` | `piecewise_cuda_graph_tokens` | `None` | piecewise CUDA graph capture token 列表 |
| `--piecewise-cuda-graph-compiler` | `piecewise_cuda_graph_compiler` | `eager` | `eager` 或 `inductor` |
| `--torchao-config` | `torchao_config` | 空字符串 | torchao 优化配置 |
| `--enable-nan-detection` | `enable_nan_detection` | `False` | 已废弃；改用环境变量 |
| `--enable-p2p-check` | `enable_p2p_check` | `False` | 检查 GPU P2P access |
| `--triton-attention-reduce-in-fp32` | `triton_attention_reduce_in_fp32` | `False` | Triton attention 中间结果转 FP32 |
| `--triton-attention-num-kv-splits` | `triton_attention_num_kv_splits` | `8` | Triton flash decoding KV split 数 |
| `--triton-attention-split-tile-size` | `triton_attention_split_tile_size` | `None` | deterministic inference 用 split KV tile size |
| `--num-continuous-decode-steps` | `num_continuous_decode_steps` | `1` | 连续 decode 多步，降低调度开销但可能增加 TTFT |
| `--delete-ckpt-after-loading` | `delete_ckpt_after_loading` | `False` | 加载后删除 checkpoint |
| `--enable-memory-saver` | `enable_memory_saver` | `False` | 允许释放/恢复 memory occupation |
| `--enable-weights-cpu-backup` | `enable_weights_cpu_backup` | `False` | 释放权重占用时备份主模型和 draft 权重到 CPU |
| `--enable-draft-weights-cpu-backup` | `enable_draft_weights_cpu_backup` | `False` | 只备份 draft 权重到 CPU |
| `--allow-auto-truncate` | `allow_auto_truncate` | `False` | 请求超长时自动截断，而不是报错 |
| `--enable-custom-logit-processor` | `enable_custom_logit_processor` | `False` | 允许请求传自定义 logit processor；默认关闭以保证安全 |
| `--flashinfer-mla-disable-ragged` | `flashinfer_mla_disable_ragged` | `False` | FlashInfer MLA 不使用 ragged prefill wrapper |
| `--disable-shared-experts-fusion` | `disable_shared_experts_fusion` | `False` | 关闭 DeepSeek v3/r1 shared experts fusion |
| `--disable-chunked-prefix-cache` | `disable_chunked_prefix_cache` | `False` | 关闭 DeepSeek chunked prefix cache |
| `--disable-fast-image-processor` | `disable_fast_image_processor` | `False` | 使用基础 image processor |
| `--keep-mm-feature-on-device` | `keep_mm_feature_on_device` | `False` | 多模态 feature 留在 device 上，减少 D2H copy |
| `--enable-return-hidden-states` | `enable_return_hidden_states` | `False` | 允许响应返回 hidden states |
| `--enable-return-routed-experts` | `enable_return_routed_experts` | `False` | 允许响应返回每层 routed experts |
| `--scheduler-recv-interval` | `scheduler_recv_interval` | `1` | scheduler poll 请求间隔 |
| `--numa-node` | `numa_node` | `None` | 为子进程指定 NUMA node |
| `--enable-deterministic-inference` | `enable_deterministic_inference` | `False` | batch invariant deterministic inference |
| `--rl-on-policy-target` | `rl_on_policy_target` | `None` | 要对齐的 RL on-policy 训练系统 |
| `--enable-attn-tp-input-scattered` | `enable_attn_tp_input_scattered` | `False` | 仅 TP 时允许 attention 输入分散，降低 qkv latent 等计算 |
| `--gc-threshold` | `gc_threshold` | `None` | Python GC 阈值，接受 1 到 3 个整数 |
| `--enable-nsa-prefill-context-parallel` | `enable_nsa_prefill_context_parallel` | `False` | DeepSeek v3.2 长序列 prefill context parallel |
| `--nsa-prefill-cp-mode` | `nsa_prefill_cp_mode` | `round-robin-split` | NSA prefill CP token split 模式 |
| `--enable-fused-qk-norm-rope` | `enable_fused_qk_norm_rope` | `False` | 融合 qk norm 和 rope |
| `--enable-precise-embedding-interpolation` | `enable_precise_embedding_interpolation` | `False` | embedding grid resize 使用 corner alignment |
| `--enable-fused-moe-sum-all-reduce` | `enable_fused_moe_sum_all_reduce` | `False` | 启用 fused MoE Triton 和 sum all reduce |
| `--enable-prefill-context-parallel` | `enable_prefill_context_parallel` | `False` | 通用 prefill context parallel |
| `--prefill-cp-mode` | `prefill_cp_mode` | `in-seq-split` | 通用 prefill CP token split 模式 |
| `--enable-dynamic-batch-tokenizer` | `enable_dynamic_batch_tokenizer` | `False` | 并发请求下异步动态批量 tokenizer |
| `--dynamic-batch-tokenizer-batch-size` | `dynamic_batch_tokenizer_batch_size` | `32` | 动态 tokenizer batch size |
| `--dynamic-batch-tokenizer-batch-timeout` | `dynamic_batch_tokenizer_batch_timeout` | `0.002` | 动态 tokenizer batching 超时秒数 |
| `--debug-tensor-dump-output-folder` | `debug_tensor_dump_output_folder` | `None` | tensor dump 输出目录 |
| `--debug-tensor-dump-layers` | `debug_tensor_dump_layers` | `None` | dump 层 id；未设表示全部层 |
| `--debug-tensor-dump-input-file` | `debug_tensor_dump_input_file` | `None` | tensor dump 输入文件 |
| `--debug-tensor-dump-inject` | `debug_tensor_dump_inject` | `False` | 把 JAX 输出注入为每层输入 |

Qwen3.6 启动脚本默认 `--disable-piecewise-cuda-graph`，这是为了长上下文 agent 服务更保守地避开 piecewise CUDA graph 的兼容性变量。

### 4.16 PD disaggregation、encoder disaggregation 与权重加载

| CLI | 字段 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--disaggregation-mode` | `disaggregation_mode` | `null` | PD 分离模式：`null`、`prefill`、`decode` |
| `--disaggregation-transfer-backend` | `disaggregation_transfer_backend` | `mooncake` | PD transfer backend |
| `--disaggregation-bootstrap-port` | `disaggregation_bootstrap_port` | `8998` | prefill server bootstrap port |
| `--disaggregation-ib-device` | `disaggregation_ib_device` | `None` | PD transfer InfiniBand 设备 |
| `--disaggregation-decode-enable-offload-kvcache` | `disaggregation_decode_enable_offload_kvcache` | `False` | decode server 异步 offload KV cache |
| `--num-reserved-decode-tokens` | `num_reserved_decode_tokens` | `512` | decode 新请求预留 token 数 |
| `--disaggregation-decode-polling-interval` | `disaggregation_decode_polling_interval` | `1` | decode server poll 间隔 |
| `--encoder-only` | `encoder_only` | `False` | MLLM encoder-only server |
| `--language-only` | `language_only` | `False` | VLM 只加载 language model 权重 |
| `--encoder-transfer-backend` | `encoder_transfer_backend` | 首个 `ENCODER_TRANSFER_BACKEND_CHOICES` | encoder disaggregation transfer backend |
| `--encoder-urls` | `encoder_urls` | `[]` | encoder server URL 列表 |
| `--enable-adaptive-dispatch-to-encoder` | `enable_adaptive_dispatch_to_encoder` | `False` | 多图请求发 encoder，单图本地处理 |
| `--custom-weight-loader` | `custom_weight_loader` | `None` | 自定义权重更新 loader import path |
| `--weight-loader-disable-mmap` | `weight_loader_disable_mmap` | `False` | safetensors 加载时禁用 mmap |
| `--remote-instance-weight-loader-seed-instance-ip` | `remote_instance_weight_loader_seed_instance_ip` | `None` | remote instance 权重加载 seed IP |
| `--remote-instance-weight-loader-seed-instance-service-port` | `remote_instance_weight_loader_seed_instance_service_port` | `None` | seed service port |
| `--remote-instance-weight-loader-send-weights-group-ports` | `remote_instance_weight_loader_send_weights_group_ports` | `None` | 权重发送通信 group ports |
| `--remote-instance-weight-loader-backend` | `remote_instance_weight_loader_backend` | `nccl` | `transfer_engine`、`nccl`、`modelexpress` |
| `--remote-instance-weight-loader-start-seed-via-transfer-engine` | `remote_instance_weight_loader_start_seed_via_transfer_engine` | `False` | 通过 transfer engine 启动 seed server |
| `--engine-info-bootstrap-port` | `engine_info_bootstrap_port` | `6789` | engine info bootstrap server 端口 |
| `--modelexpress-config` | `modelexpress_config` | `None` | ModelExpress P2P 权重加载 JSON 配置 |
| `--enable-pdmux` | `enable_pdmux` | `False` | PD-Multiplexing |
| `--pdmux-config-path` | `pdmux_config_path` | `None` | PD-Multiplexing 配置文件 |
| `--sm-group-num` | `sm_group_num` | `8` | SM partition group 数 |
| `--config` | 无 dataclass 字段 | `None` | 从 YAML 文件读取 CLI 选项 |

PD-Multiplexing 校验要求：

- `pp_size == 1`
- `chunked_prefill_size == -1`
- `disaggregation_mode == "null"`
- `disable_overlap_schedule == True`

### 4.17 多模态、加密 checkpoint 与 forward hooks

| CLI | 字段 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--enable-broadcast-mm-inputs-process` | `enable_broadcast_mm_inputs_process` | `False` | scheduler 中启用 broadcast mm-inputs process |
| `--mm-process-config` | `mm_process_config` | `None` | 多模态预处理 JSON，例如 `image`、`video`、`audio` |
| `--mm-enable-dp-encoder` | `mm_enable_dp_encoder` | `False` | 多模态 encoder data parallel；dp size 自动设为 tp size |
| `--limit-mm-data-per-request` | `limit_mm_data_per_request` | `None` | 每请求多模态输入数上限，如 `{"image": 1, "video": 1}` |
| `--enable-prefix-mm-cache` | `enable_prefix_mm_cache` | `False` | 启用 prefix multimodal cache，目前仅支持 mm-only |
| `--enable-mm-global-cache` | `enable_mm_global_cache` | `False` | 全局多模态 embedding cache，跳过重复 ViT inference |
| `--decrypted-config-file` | `decrypted_config_file` | `None` | checkpoint 解密配置 |
| `--decrypted-draft-config-file` | `decrypted_draft_config_file` | `None` | draft checkpoint 解密配置 |
| `--forward-hooks` | `forward_hooks` | `None` | JSON 格式 forward hook 规格 |

Qwen3.6-27B 文本 agent 服务通常不需要这些多模态参数。

## 5. Qwen3.6-27B 推荐关注的参数

### 5.1 必须确认

| 参数 | 建议 |
| --- | --- |
| `--model-path` | 指向真实 Qwen3.6-27B 本地模型目录 |
| `--served-model-name` | 和客户端 `model` 字段保持一致，便于排查 |
| `--api-key` | 生产环境不要为空 |
| `--host` | 本机反向代理用 `127.0.0.1`；容器或公网入口用 `0.0.0.0` |
| `--port` | 默认 `30000`，多实例要错开 |
| `--tp-size` | 和实际可用 GPU 数匹配 |
| `--context-length` | 长上下文 agent 当前使用 `262144` |

### 5.2 首先调优

| 场景 | 优先看 |
| --- | --- |
| 启动 OOM | `MEMORY_TARGET_FRACTION`、`--mem-fraction-static`、`--max-running-requests` |
| 请求中途 OOM | `--max-running-requests`、`--max-total-tokens`、请求 `max_tokens` |
| 首 token 慢 | `--chunked-prefill-size`、`--max-prefill-tokens`、`--schedule-policy` |
| prefix cache 命中低 | `--schedule-policy lpm`、`--radix-eviction-policy lru/slru` |
| 长输出被截断 | 请求侧 `max_tokens` / `max_completion_tokens`，以及 `CONTEXT_LENGTH` |
| tool call 解析异常 | `--tool-call-parser`、请求 `tools`、`tool_choice` |
| 需要排查单请求耗时 | `--enable-request-time-stats-logging`、`--log-requests-level 2` |

### 5.3 不建议随意打开

| 参数 | 原因 |
| --- | --- |
| `--trust-remote-code` | 会执行模型仓库自定义代码，只在可信模型源上使用 |
| `--enable-custom-logit-processor` | 请求可传自定义逻辑，默认关闭是安全选择 |
| `--delete-ckpt-after-loading` | 会删除 checkpoint，调试和共享模型目录风险很高 |
| `--log-requests-level 3` | 完整记录输入输出，日志量和数据泄露风险都高 |
| `--enforce-piecewise-cuda-graph` | 跳过自动禁用条件，主要测试用 |
| `--allow-auto-truncate` | 可能悄悄截断输入，影响 agent 正确性 |

## 6. `/v1/chat/completions` 请求参数结构

你的 curl：

```bash
curl --noproxy '*' -sS http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [
      {"role": "system", "content": "你是助手，都用中文回答"},
      {"role": "user", "content": "用三句话介绍诗人李白。"}
    ],
    "max_tokens": 256,
    "temperature": 0
  }'
```

服务端路由：

```text
POST /v1/chat/completions
request model = ChatCompletionRequest
handler = openai_serving_chat.handle_request()
```

对应 Pydantic 模型是 `ChatCompletionRequest`。

### 6.1 最小请求体

`messages` 是必填字段。`model` 在 Pydantic 中有默认值 `default`，但实际调用建议显式传，并和启动时 `--served-model-name` 对齐。

```json
{
  "model": "qwen3.6-27b",
  "messages": [
    {"role": "system", "content": "你是助手，都用中文回答"},
    {"role": "user", "content": "用三句话介绍诗人李白。"}
  ],
  "max_tokens": 256,
  "temperature": 0
}
```

如果启动脚本仍使用默认：

```bash
--served-model-name qwen3.6-27b
```

则请求里更推荐：

```json
"model": "qwen3.6-27b"
```

SGLang 还支持通过 model 字段选择 LoRA adapter：

```json
"model": "qwen3.6-27b:adapter-name"
```

服务端会把冒号后的 `adapter-name` 解析为 LoRA adapter 名。

### 6.2 顶层字段

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `messages` | `List[ChatCompletionMessageParam]` | 必填 | 对话消息数组 |
| `model` | `string` | `default` | 模型名；建议与 `--served-model-name` 一致；支持 `base:adapter` LoRA 写法 |
| `frequency_penalty` | `float` | `0.0` | 频率惩罚 |
| `logit_bias` | `dict[str, float]` | `null` | token id 到 bias 的映射 |
| `logprobs` | `bool` | `false` | 是否返回 logprobs |
| `top_logprobs` | `int | null` | `null` | 每 token 返回多少 top logprobs |
| `max_tokens` | `int | null` | `null` | 最大生成 token 数；代码标注已被 `max_completion_tokens` 替代 |
| `max_completion_tokens` | `int | null` | `null` | 最大 completion token 数，包含可见输出和 reasoning tokens |
| `n` | `int` | `1` | 每个请求生成多少个 completion |
| `presence_penalty` | `float` | `0.0` | 存在惩罚 |
| `response_format` | `ResponseFormat | StructuralTagResponseFormat | null` | `null` | 约束输出格式，如 JSON object / JSON schema |
| `seed` | `int | null` | `null` | 采样随机种子 |
| `stop` | `string | list[string] | null` | `null` | 停止字符串 |
| `stream` | `bool` | `false` | 是否 SSE 流式返回 |
| `stream_options` | `object | null` | `null` | 流式返回配置 |
| `temperature` | `float | null` | `null` | 采样温度；`null` 时按模型 generation config 或默认值 |
| `top_p` | `float | null` | `null` | nucleus sampling |
| `user` | `string | null` | `null` | 终端用户标识 |
| `tools` | `list[Tool] | null` | `null` | 工具定义 |
| `tool_choice` | `ToolChoice | "auto" | "required" | "none"` | 自动规则 | 工具选择；无 tools 且未传时归一为 `none` |
| `parallel_tool_calls` | `bool` | `true` | 是否允许并行 tool calls |
| `return_hidden_states` | `bool` | `false` | 是否返回 hidden states；需要服务端允许 |
| `return_routed_experts` | `bool` | `false` | 是否返回 routed experts；需要服务端允许 |
| `return_cached_tokens_details` | `bool` | `false` | 是否返回 cache token 明细 |
| `reasoning_effort` | `"none" | "low" | "medium" | "high" | null` | `null` | reasoning 模型努力程度 |

### 6.3 SGLang 扩展字段

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `top_k` | `int | null` | `null` | top-k sampling；`null` 时按模型 generation config 或默认值 |
| `min_p` | `float | null` | `null` | min-p sampling |
| `min_tokens` | `int` | `0` | 最少生成 token 数 |
| `regex` | `string | null` | `null` | regex constrained decoding |
| `ebnf` | `string | null` | `null` | EBNF constrained decoding |
| `repetition_penalty` | `float | null` | `null` | 重复惩罚 |
| `stop_token_ids` | `list[int] | null` | `null` | 停止 token id |
| `stop_regex` | `string | list[string] | null` | `null` | 停止正则 |
| `no_stop_trim` | `bool` | `false` | 不裁掉 stop 内容 |
| `ignore_eos` | `bool` | `false` | 忽略 EOS |
| `continue_final_message` | `bool` | `false` | 如果最后一条是 assistant，把它作为续写前缀 |
| `skip_special_tokens` | `bool` | `true` | decode 时跳过 special tokens |
| `lora_path` | `string | list[string|null] | null` | `null` | 请求级 LoRA adapter；优先级低于 `model` 里的 `base:adapter` |
| `session_params` | `dict | null` | `null` | session 参数 |
| `separate_reasoning` | `bool` | `true` | reasoning 内容和正文分离 |
| `stream_reasoning` | `bool` | `true` | stream 时是否输出 reasoning |
| `chat_template_kwargs` | `dict | null` | `null` | 传给 tokenizer chat template 的额外参数 |
| `max_dynamic_patch` | `int | null` | `null` | 多模态 tiling 控制 |
| `min_dynamic_patch` | `int | null` | `null` | 多模态 tiling 控制 |
| `custom_logit_processor` | `string | list[string|null] | null` | `null` | 自定义 logit processor；服务端需开启 |
| `custom_params` | `dict | null` | `null` | 透传自定义参数 |
| `rid` | `string | list[string] | null` | `null` | 请求 id |
| `extra_key` | `string | list[string] | null` | `null` | 请求分类 key，例如 cache salt |
| `cache_salt` | `string | list[string] | null` | `null` | request cache salt |
| `priority` | `int | null` | `null` | 请求优先级；服务端需启用 priority scheduling |
| `bootstrap_host` | `string | list[string] | null` | `null` | PD disaggregation 用 |
| `bootstrap_port` | `int | list[int|null] | null` | `null` | PD disaggregation 用 |
| `bootstrap_room` | `int | list[int] | null` | `null` | PD disaggregation 用 |
| `routed_dp_rank` | `int | null` | `null` | 外部 router 指定 DP worker |
| `disagg_prefill_dp_rank` | `int | null` | `null` | PD decode 侧提示 KV cache 所在 prefill DP worker |
| `data_parallel_rank` | `int | null` | `null` | 已废弃，迁移到 `routed_dp_rank` |

采样参数优先级：

```text
请求显式值 > 模型 generation_config.json > OpenAI/SGLang 默认值
```

其中 SGLang/OpenAI 默认值包括：

```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "top_k": -1,
  "min_p": 0.0,
  "repetition_penalty": 1.0
}
```

### 6.4 `messages` 数据结构

`messages` 是一个数组。每条消息按 role 分两类。

#### user 消息

```json
{
  "role": "user",
  "content": "用三句话介绍诗人李白。"
}
```

结构：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `role` | `"user"` | 用户消息 |
| `content` | `string | list[ContentPart]` | 文本或多模态内容数组；user 消息不能为空 |

#### system / developer / assistant / tool / function 消息

```json
{
  "role": "system",
  "content": "你是助手，都用中文回答"
}
```

结构：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `role` | `"system" | "developer" | "assistant" | "tool" | "function"` | role 大小写会归一成小写 |
| `content` | `string | list[ContentPart] | null` | 内容，可为空 |
| `tool_call_id` | `string | null` | tool 响应关联的 tool call id |
| `name` | `string | null` | function/tool 名 |
| `reasoning_content` | `string | null` | reasoning 内容 |
| `tool_calls` | `list[ToolCall] | null` | assistant 发起的 tool calls |
| `tools` | `list[Tool] | null` | 可附着在消息上的工具定义 |

#### 多模态 content part

文本：

```json
{"type": "text", "text": "描述这张图"}
```

图片：

```json
{
  "type": "image_url",
  "image_url": {
    "url": "https://example.com/a.png",
    "detail": "auto",
    "max_dynamic_patch": 12,
    "min_dynamic_patch": 1
  },
  "modalities": "image"
}
```

视频：

```json
{
  "type": "video_url",
  "video_url": {
    "url": "file:///tmp/a.mp4",
    "max_dynamic_patch": 12,
    "min_dynamic_patch": 1
  }
}
```

音频：

```json
{
  "type": "audio_url",
  "audio_url": {
    "url": "file:///tmp/a.wav"
  }
}
```

Qwen3.6-27B 文本 agent 请求通常只用字符串 content，不需要多模态 part。

### 6.5 tools / tool_choice 数据结构

工具定义：

```json
{
  "type": "function",
  "function": {
    "name": "search_docs",
    "description": "搜索内部文档",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string"}
      },
      "required": ["query"]
    },
    "strict": false
  }
}
```

`tool_choice` 可选：

```json
"tool_choice": "none"
```

```json
"tool_choice": "auto"
```

```json
"tool_choice": "required"
```

```json
{
  "type": "function",
  "function": {
    "name": "search_docs"
  }
}
```

校验规则：

- `tool_choice="required"` 时必须传 `tools`。
- 指定具体 tool 时，`tools` 里必须存在同名 function。
- tool function 的 `parameters` 必须是合法 JSON Schema。

### 6.6 response_format 数据结构

普通文本：

```json
{"type": "text"}
```

JSON object：

```json
{"type": "json_object"}
```

JSON schema：

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "Answer",
    "schema": {
      "type": "object",
      "properties": {
        "answer": {"type": "string"}
      },
      "required": ["answer"]
    },
    "strict": true
  }
}
```

兼容写法：如果 `response_format` 顶层传了 `schema`，服务端会转换成 `json_schema.schema`。

### 6.7 stream_options

```json
{
  "stream": true,
  "stream_options": {
    "include_usage": true,
    "continuous_usage_stats": false
  }
}
```

字段：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `include_usage` | `false` | stream 结束时包含 usage |
| `continuous_usage_stats` | `false` | 流式过程中持续输出 usage stats |

服务启动参数 `--stream-response-default-include-usage` 可以让 stream 响应默认带 usage。

### 6.8 你的 curl 字段逐项解释

```json
{
  "model": "qwen3.6-27b",
  "messages": [
    {"role": "system", "content": "你是助手，都用中文回答"},
    {"role": "user", "content": "用三句话介绍诗人李白。"}
  ],
  "max_tokens": 256,
  "temperature": 0
}
```

| 字段 | 含义 | 当前值影响 |
| --- | --- | --- |
| `model` | 请求模型名 | 建议改成启动时 `--served-model-name`，当前脚本默认是 `qwen3.6-27b` |
| `messages[0].role=system` | 系统指令 | 要求助手全部中文回答 |
| `messages[1].role=user` | 用户问题 | 要求三句话介绍李白 |
| `max_tokens=256` | 最多生成 256 个 token | 足够三句话；代码中更推荐新字段 `max_completion_tokens` |
| `temperature=0` | 确定性输出 | 降低随机性，适合验证服务 |

等价的新字段写法：

```json
{
  "model": "qwen3.6-27b",
  "messages": [
    {"role": "system", "content": "你是助手，都用中文回答"},
    {"role": "user", "content": "用三句话介绍诗人李白。"}
  ],
  "max_completion_tokens": 256,
  "temperature": 0
}
```

### 6.9 非流式响应结构

非流式响应模型是 `ChatCompletionResponse`，大致结构：

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1730000000,
  "model": "qwen3.6-27b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "李白是唐代著名浪漫主义诗人..."
      },
      "logprobs": null,
      "finish_reason": "stop",
      "matched_stop": null
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "reasoning_tokens": 0
  }
}
```

可能出现的 `finish_reason`：

- `stop`
- `length`
- `tool_calls`
- `content_filter`
- `function_call`
- `abort`

### 6.10 流式响应结构

`stream=true` 时响应对象是 `ChatCompletionStreamResponse`，SSE 每个 chunk 大致为：

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion.chunk",
  "created": 1730000000,
  "model": "qwen3.6-27b",
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "李"
      },
      "finish_reason": null
    }
  ],
  "usage": null
}
```

如果开启 `include_usage`，最后或过程中会带 usage。

## 7. 推荐调用模板

本机非流式 smoke test：

```bash
curl --noproxy '*' -sS http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [
      {"role": "system", "content": "你是助手，都用中文回答"},
      {"role": "user", "content": "用三句话介绍诗人李白。"}
    ],
    "max_completion_tokens": 256,
    "temperature": 0
  }'
```

流式调用：

```bash
curl --noproxy '*' -N -sS http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [
      {"role": "user", "content": "用三句话介绍诗人李白。"}
    ],
    "max_completion_tokens": 256,
    "temperature": 0,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

JSON schema 约束输出：

```bash
curl --noproxy '*' -sS http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [
      {"role": "user", "content": "用 JSON 给出李白的姓名、朝代、代表作。"}
    ],
    "max_completion_tokens": 256,
    "temperature": 0,
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "Poet",
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "dynasty": {"type": "string"},
            "works": {
              "type": "array",
              "items": {"type": "string"}
            }
          },
          "required": ["name", "dynasty", "works"]
        },
        "strict": true
      }
    }
  }'
```

## 8. 常见排错

| 现象 | 优先检查 |
| --- | --- |
| `401 Unauthorized` | `OPENAI_API_KEY`、`--api-key`、Authorization header |
| 连接失败 | `--host`、`--port`、进程是否 ready、反向代理配置 |
| `/health` 不 ready | server log、模型路径、显存、NCCL 初始化 |
| 返回模型名不符合预期 | `--served-model-name` 和请求 `model` |
| `max_completion_tokens is too large` | 请求 `max_tokens/max_completion_tokens` 是否超过 `--context-length` |
| 请求排队或超时 | `--max-running-requests`、`--max-queued-requests`、GPU 显存和 KV cache |
| 长 prompt 首 token 慢 | `--chunked-prefill-size`、`--max-prefill-tokens`、prefix cache 命中率 |
| JSON schema 报错 | `response_format.json_schema.schema` 是否存在且为合法 JSON Schema |
| tool call 没解析 | `--tool-call-parser`、`tools`、`tool_choice`、模型模板 |
| priority 不生效 | 是否启动了 `--enable-priority-scheduling` |
