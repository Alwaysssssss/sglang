# Feature: 面向长程任务 Agent 的 Qwen3.6-27B SGLang 启动配置说明

本文基于 `docs_always/qwen3.6-27b/start_qwen36_27b.sh`、已验证的上线记录、SGLang 当前参数定义和本地模型配置，说明如何把 Qwen3.6-27B 部署成适合长程任务 Agent 的 OpenAI 兼容服务。重点回答三个问题：

- 如何配置更长上下文。
- 在长上下文下如何支持多用户、多请求同时发起，以及 SGLang 会如何调度。
- 如何更好地使用和扩展 KV cache，提升长上下文和多轮 Agent 的效率。

## 0. 结论先行

当前脚本的主线配置是合理的生产基线：

```bash
docs_always/qwen3.6-27b/start_qwen36_27b.sh
```

默认关键参数：

| 参数 | 当前脚本默认值 | 作用 |
| --- | --- | --- |
| `--model-path` | `/mnt/shanhai-ai/wenhy/models/Qwen/Qwen/Qwen3___6-27B` | 本地模型目录。 |
| `--served-model-name` | `qwen3.6-27b` | 对外 OpenAI 兼容模型名。 |
| `--tensor-parallel-size` | `4` | 使用 4 张 A100 做 TP。 |
| `--context-length` | `131072` | 128K 上下文窗口。 |
| `--max-total-tokens` | `1048576` | GPU KV pool 总 token 容量。 |
| `--max-running-requests` | `8` | 同时进入运行态的请求上限。 |
| `--chunked-prefill-size` | `8192` | 长 prompt 分块 prefill，每块最多 8K tokens。 |
| `--max-prefill-tokens` | `16384` | 单个 prefill batch 的 token 上限。 |
| `--dtype` | `bfloat16` | A100 友好，匹配模型 dtype。 |
| `--attention-backend` | `flashinfer` | 当前环境已安装并已验证可启动。 |
| `--sampling-backend` | `flashinfer` | 采样后端。 |
| `--tool-call-parser` | `qwen3_coder` | 将 Qwen 工具调用文本解析成 OpenAI `tool_calls`。 |
| `--disable-piecewise-cuda-graph` | 已启用 | 当前脚本固定关闭 piecewise CUDA graph，避免长上下文和混合后端下的额外复杂度。 |

已验证日志中有以下关键证据：

- `context_len=131072`
- `max_total_num_tokens=1048576`
- `chunked_prefill_size=8192`
- `max_prefill_tokens=16384`
- `max_running_requests=8`
- 每个 TP rank 分配 `#tokens: 1048576`，`K size: 8.00 GB`，`V size: 8.00 GB`
- 历史验收覆盖 `/health`、`/v1/models`、错误 key、非流式、流式、8 并发、100K token 长上下文

因此，推荐把当前配置作为 `128K / tp=4 / agent baseline`。不要一开始就把上下文拉到模型上限 256K，也不要盲目提高并发。长程 Agent 的瓶颈不是 HTTP 并发数，而是活跃请求占用的 KV token 总量、prefill 峰值、decode 阶段显存和单请求的超长等待时间。

## 1. 当前模型和机器基线

本地模型配置文件显示：

| 项目 | 值 |
| --- | --- |
| 架构 | `Qwen3_5ForConditionalGeneration` |
| `model_type` | `qwen3_5` |
| `text_config.max_position_embeddings` | `262144` |
| `num_hidden_layers` | `64` |
| `num_attention_heads` | `24` |
| `num_key_value_heads` | `4` |
| `head_dim` | `256` |
| `dtype` | `bfloat16` |
| 本轮上线目标 | `131072` tokens |
| GPU | 4 x `NVIDIA A100-SXM4-80GB` |
| 模型大小 | 约 `52G` |

这意味着：

- 模型配置允许到 `256K` 级别的位置长度。
- 当前上线选择 `128K` 是保守、已验收、可承载 8 并发测试的配置。
- 如果切到 `256K`，即使模型位置长度允许，也会显著减少可同时运行的长请求数量，并拉长 prefill 时间，需要重新做显存、并发、流式、Agent 工具调用验收。

## 2. 推荐启动方式

### 2.1 正常启动

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

docs_always/qwen3.6-27b/start_qwen36_27b.sh
```

如果当前 GPU 0-2 已有其它服务占用，SGLang 可能触发 TP 显存不均衡检查。确认剩余显存足够后再显式关闭该检查：

```bash
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
  docs_always/qwen3.6-27b/start_qwen36_27b.sh
```

不要默认关闭该检查。只有在确认其它进程属于预期占用、每张卡仍有足够 free memory 时才使用。

### 2.2 启动前检查

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

test -x /mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/python3
test -d /mnt/shanhai-ai/wenhy/models/Qwen/Qwen/Qwen3___6-27B
lsof -nP -iTCP:30000 -sTCP:LISTEN || true
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
```

### 2.3 启动后确认

```bash
LOG_FILE=$(ls -t logs/qwen36_27b/qwen36_27b_tp4_128k_*.log | head -n 1)

grep -E "server_args=|KV Cache is allocated|max_total_num_tokens|Application startup complete" "$LOG_FILE"
```

应重点确认：

- `tool_call_parser='qwen3_coder'`。如果日志里是 `tool_call_parser=None`，说明当前运行进程不是由最新脚本启动，需要重启。
- `context_length=131072`。
- `max_total_tokens=1048576`。
- `max_running_requests=8`。
- `chunked_prefill_size=8192`。
- `disable_radix_cache=False`，表示 radix/prefix cache 未被关闭。
- `page_size=1`，表示 prefix cache 以 token 级粒度匹配，适合 Agent 重复系统提示词和长上下文前缀复用。

## 3. 长上下文配置

### 3.1 为什么当前选择 128K

`context-length` 是单请求的最大上下文窗口，`max-total-tokens` 是所有运行请求共享的 KV token pool。两者不是同一个概念。

当前配置：

```text
context_length = 131072
max_total_tokens = 1048576
```

粗略估算：

```text
理论满上下文并发数 ~= max_total_tokens / context_length
                  ~= 1048576 / 131072
                  ~= 8
```

这正好对应脚本中的：

```text
MAX_RUNNING_REQUESTS=8
```

实际运行时还要加上输出 token、系统提示词、工具调用历史、chat template token、内部保留 token 和调度开销，所以不要把这个公式理解成严格 SLA。它的意义是：当前配置是为“最多 8 个 128K 级别请求同时在运行态”设计的保守基线。

### 3.2 128K Agent 场景推荐

保持当前默认值：

```bash
CONTEXT_LENGTH=131072
MAX_TOTAL_TOKENS=1048576
MAX_RUNNING_REQUESTS=8
CHUNKED_PREFILL_SIZE=8192
MAX_PREFILL_TOKENS=16384
```

适用场景：

- 单个 Agent 任务可能带很长历史、文档、代码上下文或多轮工具返回。
- 多用户同时使用，但并发规模还处于个位数到十来个活跃请求。
- 优先保证长上下文可用性，而不是极限短问答吞吐。

### 3.3 64K 高并发优先配置

如果实际业务中大多数请求不需要 128K，而是更多用户同时短问答或中等上下文，建议新起一个单独实例或调整启动参数：

```bash
CONTEXT_LENGTH=65536 \
MAX_TOTAL_TOKENS=1048576 \
MAX_RUNNING_REQUESTS=16 \
CHUNKED_PREFILL_SIZE=4096 \
MAX_PREFILL_TOKENS=8192 \
docs_always/qwen3.6-27b/start_qwen36_27b.sh
```

含义：

- 单请求上下文降到 64K。
- 同样的 KV pool 可以容纳更多中等长度请求。
- `chunked_prefill_size` 降到 4K，降低 prefill 峰值显存，但长 prompt prefill 时间可能增加。
- 需要重新跑验收脚本，尤其是并发、流式和真实 Agent 请求。

### 3.4 256K 实验配置

模型配置中 `max_position_embeddings=262144`，但 256K 不应直接作为默认生产配置。若确实要试验：

```bash
CONTEXT_LENGTH=262144 \
MAX_TOTAL_TOKENS=1048576 \
MAX_RUNNING_REQUESTS=4 \
CHUNKED_PREFILL_SIZE=8192 \
MAX_PREFILL_TOKENS=16384 \
READY_TIMEOUT_SECONDS=900 \
docs_always/qwen3.6-27b/start_qwen36_27b.sh
```

注意：

- 1 个满 256K 请求约占当前 token pool 的 1/4。
- `MAX_RUNNING_REQUESTS` 应先降到 `4`，再压测。
- Nginx、客户端、Agent 网关的超时都要放宽。
- 必须补充 200K 到 240K token 级别的长上下文验收，不能用 100K 验收代替。

## 4. SGLang 如何处理多用户和同时请求

### 4.1 请求生命周期

一个 OpenAI chat 请求进入 SGLang 后，主要经过：

1. HTTP 接入和鉴权。
2. tokenizer 和 chat template 处理。
3. scheduler 入队。
4. prefill：把输入 prompt 计算成 KV cache。
5. decode：逐 token 生成输出。
6. 流式或非流式返回。

长上下文 Agent 的瓶颈主要在第 4 步和第 5 步：

- 超长 prompt 的 prefill 会占用大量 GPU 计算和激活显存。
- decode 阶段的每个活跃请求都要持有自己的 KV cache。
- 多个请求一起运行时，SGLang 会把它们组织成 batch，但 batch 能装多少，受 `max_total_tokens`、`max_running_requests`、`max_prefill_tokens`、`chunked_prefill_size` 和显存共同限制。

### 4.2 当前 8 并发的含义

`MAX_RUNNING_REQUESTS=8` 不是“整个服务最多只能收到 8 个 HTTP 请求”。它表示 scheduler 中同时运行的请求上限。更多请求到达时会排队等待，或在上游代理、客户端超时后失败。

当前 Nginx 配置还有限流：

```nginx
limit_req_zone $binary_remote_addr zone=qwen36_limit:10m rate=2r/s;
limit_req zone=qwen36_limit burst=16 nodelay;
```

因此，多用户同时发消息时会发生：

- Nginx 先限制每个来源 IP 的突发速率。
- 通过 Nginx 的请求进入 SGLang。
- SGLang scheduler 只让不超过 `max_running_requests=8` 个请求进入运行态。
- 如果运行请求的 KV token 占满 `max_total_tokens`，新请求即使数量没到 8，也要等待。
- 如果很多请求都是 100K 到 128K 长 prompt，实际瓶颈会是 KV token pool，而不是 HTTP 请求数。

### 4.3 多用户支持策略

生产上不要只依赖 SGLang 自身排队。建议分三层做保护：

| 层 | 目的 | 建议 |
| --- | --- | --- |
| Nginx | 防突发流量打爆服务 | 保留 `limit_req`，按业务调整 `rate` 和 `burst`。 |
| Agent/API Gateway | 做用户级限额 | 按 user_id/API key 限制并发、队列长度、最大 prompt tokens、最大输出 tokens。 |
| SGLang | GPU 内部调度 | 用 `max-running-requests`、`max-total-tokens`、`chunked-prefill-size` 控制显存和 batch。 |

建议网关策略：

- 普通用户：每人最多 1 到 2 个运行中请求。
- 高优用户或后台任务：单独队列，避免和交互式请求互相影响。
- 每个请求必须设置合理的 `max_tokens`，不要让客户端默认无限生成。
- 进入 SGLang 前估算 prompt token 数，超过业务上限直接拒绝或要求用户压缩上下文。
- 对 64K 以上的长任务使用异步任务模式，而不是同步 HTTP 等待到完成。

### 4.4 并发容量估算

用下面的简单公式做第一轮容量估算：

```text
每个请求占用 tokens ~= prompt_tokens + 已生成 tokens + 保守余量
可运行请求数 ~= floor(max_total_tokens / 每请求平均占用 tokens)
```

在当前 `max_total_tokens=1048576` 下：

| 平均 prompt | 输出上限 | 粗略可运行请求数 | 推荐 `max-running-requests` |
| --- | --- | --- | --- |
| 8K | 1K | 100+ | 不建议直接设太高，先 16 到 32 压测 |
| 32K | 2K | 30 左右 | 8 到 16 |
| 64K | 2K | 15 左右 | 8 到 12 |
| 100K | 2K | 10 左右 | 6 到 8 |
| 128K | 1K | 7 到 8 | 4 到 8 |
| 256K | 1K | 3 到 4 | 2 到 4 |

这些数字只是容量估算。真实值要用 `verify_qwen36_27b.py --concurrency N` 和真实 Agent 请求回放验证。

### 4.5 调参顺序

如果目标是“更多并发”：

1. 先降低单请求上下文需求，例如 128K 降到 64K。
2. 限制客户端 `max_tokens` 和工具返回内容长度。
3. 根据日志里的 `available_gpu_mem` 调整 `MAX_TOTAL_TOKENS` 或 `--mem-fraction-static`。
4. 逐步提高 `MAX_RUNNING_REQUESTS`，每次只加一档，例如 `8 -> 12 -> 16`。
5. 每一档都做并发、长上下文、流式和 Agent 工具调用验收。

如果出现 prefill OOM：

- 先把 `CHUNKED_PREFILL_SIZE` 从 `8192` 降到 `4096`。
- 再把 `MAX_PREFILL_TOKENS` 从 `16384` 降到 `8192`。
- 如果仍然 OOM，再降低 `MAX_RUNNING_REQUESTS` 或 `CONTEXT_LENGTH`。

如果出现 decode OOM：

- 优先降低 `MAX_RUNNING_REQUESTS`。
- 限制每个请求的 `max_tokens`。
- 检查是否有超长请求长期占用 KV pool。

## 5. KV cache 配置和优化

### 5.1 当前 baseline 已经做了什么

当前服务没有关闭 radix cache：

```text
disable_radix_cache=False
page_size=1
radix_eviction_policy='lru'
```

这对 Agent 很重要。典型 Agent 请求会重复包含：

- 固定 system prompt。
- 固定工具说明。
- 固定输出格式要求。
- 多轮对话历史的前缀。
- RAG 或代码仓库任务中的共享文档前缀。

Radix/prefix cache 可以复用这些共享前缀对应的 KV，减少重复 prefill。`page_size=1` 的粒度最细，匹配率最高，适合 prompt 前缀相似但不完全按大页对齐的 Agent 请求。

### 5.2 当前 KV pool 规模

历史日志显示每个 TP rank：

```text
KV Cache is allocated. #tokens: 1048576, K size: 8.00 GB, V size: 8.00 GB
```

也就是每个 rank 约 `16 GB` KV cache。4 张卡合计约 `64 GB` KV cache 存储。这个配置已经比默认自动估算更明确，适合固定 128K 上下文服务。

### 5.3 `max-total-tokens` 与 `mem-fraction-static`

SGLang 的内存可以粗略理解为：

```text
GPU memory = model weights + KV cache pool + activations + CUDA graph buffers
```

`--mem-fraction-static` 表示模型权重和 KV cache pool 占 GPU 总显存的比例。当前脚本没有显式设置 `MEM_FRACTION_STATIC`，而是直接固定：

```text
MAX_TOTAL_TOKENS=1048576
```

这让 KV pool 的 token 容量更可控。调优时建议二选一：

- 固定 `MAX_TOTAL_TOKENS`：适合本文这种固定 128K 目标的服务。
- 使用 `--mem-fraction-static`：适合根据剩余显存自动推导 KV pool 的场景。

不要同时随意改两者。若要增加 KV pool，优先做小步实验，例如：

```bash
MAX_TOTAL_TOKENS=1179648 \
docs_always/qwen3.6-27b/start_qwen36_27b.sh
```

然后看启动日志：

```bash
grep -E "KV Cache is allocated|max_total_num_tokens|available_gpu_mem" "$LOG_FILE"
```

经验判断：

- `available_gpu_mem` 还有 5 到 8GB 左右，通常比较健康。
- 如果 `available_gpu_mem` 很高，例如 10 到 20GB，可以考虑增加 KV pool。
- 如果 `available_gpu_mem` 很低，后续 prefill/decode 更容易 OOM，应降低 KV pool 或并发。

### 5.4 是否使用 FP8 KV cache

SGLang 支持：

```bash
--kv-cache-dtype fp8_e4m3
--kv-cache-dtype fp8_e5m2
```

可选收益：

- KV cache 单 token 显存下降。
- 同样显存下可以放更多 tokens 或更多并发。

风险：

- 可能影响长上下文精度、工具参数稳定性、引用细节和推理质量。
- 需要为本模型、本硬件、本任务重新验收。

建议：

- 生产 baseline 继续用 `auto`，即 BF16 KV cache。
- 如果业务确实被 KV 容量限制，再单独做 `fp8_e4m3` 实验，因为它通常比 `fp8_e5m2` 更偏精度。
- 实验验收必须包含长上下文、工具调用 JSON 参数、代码任务、流式输出和并发，不要只测短问答。

### 5.5 是否启用 HiCache

HiCache 是 SGLang 的分层 KV cache，可以把 GPU KV cache 扩展到 CPU 内存和外部存储。相关参数包括：

```bash
--enable-hierarchical-cache
--page-size 64
--hicache-ratio 2
--hicache-size 100
--hicache-io-backend kernel
--hicache-write-policy write_through
--hicache-storage-backend file|mooncake|hf3fs|nixl|aibrix
```

它适合：

- 大量请求共享长前缀。
- 多轮 Agent 反复访问同一批长文档、代码仓库、工具说明。
- GPU KV pool 命中率不足，但机器 CPU 内存或外部 KV 存储充足。

它不适合作为第一版 baseline：

- 配置复杂度高。
- CPU/GPU KV 传输会影响延迟。
- `page-size`、host memory、storage backend、预取策略都需要单独压测。
- 当前 128K baseline 已经有明确 GPU KV pool 和通过记录。

推荐路线：

1. 第一阶段：保持当前 GPU radix cache，开启缓存命中观测。
2. 第二阶段：如果重复前缀多但 GPU KV pool 不够，再启用 HiCache host memory，不接外部存储。
3. 第三阶段：多实例、多机器或跨服务共享 KV 时，再评估 Mooncake/HF3FS/NIXL 等 L3 backend。

### 5.6 建议增加 cache 可观测性

如果要量化 Agent 前缀复用效果，建议在实验配置中加入：

```bash
--enable-cache-report
```

该参数会在 OpenAI 请求的 `usage.prompt_tokens_details` 中返回缓存 token 信息，便于判断：

- 固定 system prompt 是否被复用。
- 工具说明是否被复用。
- 多轮对话是否稳定复用前缀。
- 修改 prompt template 后 cache 命中是否下降。

当前 `start_qwen36_27b.sh` 还没有暴露这个参数。如果要纳入生产脚本，建议用环境变量控制，例如 `ENABLE_CACHE_REPORT=1` 时才追加，避免默认响应结构变化影响客户端。

### 5.7 何时 flush cache

不要把 flush cache 当成常规操作。只有以下情况才考虑：

- 更新模型权重。
- 改了 chat template 或工具提示词格式，旧 cache 不再有意义。
- 调试缓存异常或怀疑脏缓存。
- 做基准测试时需要消除缓存命中影响。

如果服务有运行中请求，flush 可能失败或影响延迟。应先确认服务空闲，再执行。

## 6. Agent 工具调用配置

### 6.1 `--tool-call-parser` 解决什么问题

Qwen tokenizer 中包含：

```text
<tool_call>
</tool_call>
<tool_response>
</tool_response>
```

模型可能自然生成如下文本：

```text
<tool_call> ... </tool_call>
```

但这只是模型输出文本。Agent 真正调用工具需要三件事同时成立：

1. 服务端启用合适的 parser，例如当前脚本的 `--tool-call-parser qwen3_coder`。
2. 客户端请求中传入 OpenAI 兼容 `tools` schema，并设置合适的 `tool_choice`。
3. Agent 框架读取响应里的 `message.tool_calls`，自己执行工具，再把工具结果以 `role=tool` 发回模型。

也就是说，SGLang 不会因为模型吐出 `<tool_call>` 文本就自动调用本机工具。SGLang 的职责是尽量把模型生成的工具调用格式解析成 OpenAI `tool_calls` 字段；真正执行工具的是上层 Agent runtime。

### 6.2 如何判断当前 parser 是否生效

重启后检查日志：

```bash
LOG_FILE=$(ls -t logs/qwen36_27b/qwen36_27b_tp4_128k_*.log | head -n 1)
grep "tool_call_parser" "$LOG_FILE" | head -n 1
```

预期包含：

```text
tool_call_parser='qwen3_coder'
```

如果是：

```text
tool_call_parser=None
```

说明当前运行进程不是最新脚本拉起，或者启动时覆盖了 `TOOL_CALL_PARSER`。需要停止旧进程并重启。

### 6.3 最小工具调用验收

请求必须包含 `tools`。只让模型自由输出“我要调用工具”不够。

```bash
export OPENAI_BASE_URL=http://127.0.0.1:18080/v1
export OPENAI_API_KEY="$(tr -d '[:space:]' < /etc/sglang/qwen36_openai_api_key)"

curl --noproxy '*' -sS "$OPENAI_BASE_URL/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [
      {"role": "user", "content": "请调用 get_weather 查询北京天气。"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "查询城市天气",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string"}
            },
            "required": ["city"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "max_tokens": 256,
    "temperature": 0
  }'
```

验收标准：

- 成功时响应应包含 `choices[0].message.tool_calls`。
- 如果只在 `choices[0].message.content` 里看到 `<tool_call>...</tool_call>`，说明 parser 未生效、请求没有使用 tools schema、chat template 不匹配，或模型没有按 parser 期望格式生成。
- 即使有 `tool_calls`，工具也不会由 SGLang 自动执行。上层 Agent 必须执行工具并追加 `role=tool` 消息。

### 6.4 是否需要 reasoning parser

Qwen 3.5/3 系列支持：

```bash
--reasoning-parser qwen3
--tool-call-parser qwen3_coder
```

当前脚本只启用了 `--tool-call-parser qwen3_coder`。如果 Agent 需要把 `<think>...</think>` 或推理内容拆到 `reasoning_content`，建议后续把启动脚本扩展为可选：

```bash
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
```

并在启动命令中追加：

```bash
--reasoning-parser "$REASONING_PARSER"
```

但这会改变响应字段，需要确认 Agent 客户端是否兼容 `reasoning_content`。

## 7. 推荐的生产分层架构

当前最小架构：

```text
Agent clients
  -> Nginx 0.0.0.0:18080
  -> SGLang 127.0.0.1:30000
  -> Qwen3.6-27B tp=4
  -> 4 x A100 80GB
```

建议生产上补齐：

```text
Agent clients
  -> API Gateway / Agent Gateway
      - per-user API key
      - per-user concurrency
      - prompt token budget
      - max_tokens policy
      - async long job queue
      - request id / trace id
  -> Nginx
      - rate limit
      - long timeout
      - streaming proxy
      - TLS if exposed outside trusted network
  -> SGLang
      - tp=4
      - 128K context
      - radix cache
      - tool call parser
  -> GPU
```

如果未来需要更高吞吐，优先扩展为多副本：

```text
Gateway / Router
  -> SGLang replica A, tp=4
  -> SGLang replica B, tp=4
  -> SGLang replica C, tp=4
```

多副本时要注意：

- 每个副本的 GPU KV cache 不共享。
- 如果同一用户的多轮请求被打到不同副本，prefix cache 命中率会下降。
- 应使用 session affinity，例如按 user_id、conversation_id 或 routing key 做粘性路由。
- 若需要跨副本 cache-aware 路由，可评估 SGLang Model Gateway 的 cache-aware policy。

## 8. 运维与监控

### 8.1 必看日志字段

```bash
LOG_FILE=$(ls -t logs/qwen36_27b/qwen36_27b_tp4_128k_*.log | head -n 1)

grep -E "server_args=|KV Cache is allocated|max_total_num_tokens|available_gpu_mem|Application startup complete|error|OOM|CUDA out of memory" "$LOG_FILE"
```

关注：

- `available_gpu_mem`：太低会增加 OOM 风险，太高说明 KV pool 可增加。
- `max_total_num_tokens`：实际 KV pool token 容量。
- `tool_call_parser`：确认工具调用 parser 是否生效。
- `disable_radix_cache`：应为 `False`。
- `page_size`：baseline 期望为 `1`。
- `Application startup complete`：服务已就绪。

### 8.2 GPU 观测

```bash
nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
```

判断：

- GPU 显存接近打满且伴随 OOM：降低并发、prefill chunk 或上下文。
- GPU 利用率低但请求排队：可能受 tokenizer、上游限流、长请求串行、网络或 Agent 工具执行阻塞影响。
- 只有部分 GPU 忙：检查 TP 进程、NCCL、是否有旧服务或其它任务占卡。

### 8.3 一键验收

外部入口：

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

export OPENAI_BASE_URL=http://127.0.0.1:18080/v1
docs_always/qwen3.6-27b/verify_qwen36_27b.py
```

更高并发实验：

```bash
docs_always/qwen3.6-27b/verify_qwen36_27b.py \
  --base-url http://127.0.0.1:18080/v1 \
  --concurrency 12
```

更长上下文实验：

```bash
docs_always/qwen3.6-27b/verify_qwen36_27b.py \
  --base-url http://127.0.0.1:18080/v1 \
  --target-tokens 120000 \
  --long-timeout 1200
```

256K 实验不应只用 `100000` token 验收。至少增加到 `200000` 以上，并观察 prefill 延迟和显存。

## 9. 配置档位建议

| 档位 | 目标 | 推荐参数 | 风险 |
| --- | --- | --- | --- |
| `128K baseline` | 长程 Agent 默认 | `context=131072`、`max_total=1048576`、`running=8`、`chunk=8192` | 已验证，吞吐不是最高。 |
| `64K concurrency` | 更多用户同时在线 | `context=65536`、`running=16`、`chunk=4096` | 单请求上下文减半，需要业务接受。 |
| `128K conservative` | 减少 OOM 风险 | `context=131072`、`running=4`、`chunk=4096` | 长 prompt 更慢，但更稳。 |
| `128K higher pool` | 提高 KV 容量 | `max_total` 小步提高到 `1179648` 或更高 | 需要看 `available_gpu_mem`，过高会 OOM。 |
| `128K fp8 KV experiment` | 用 KV 量化换容量 | 加 `--kv-cache-dtype fp8_e4m3` | 可能影响精度，必须重新验收。 |
| `256K experiment` | 极长单任务 | `context=262144`、`running=2..4` | prefill 慢、并发低、需新验收。 |
| `HiCache experiment` | 重复长前缀很多 | `--enable-hierarchical-cache`、`--page-size 64`、host memory | 配置和延迟复杂，先离线压测。 |

## 10. 最终建议

1. 保持 `start_qwen36_27b.sh` 当前 128K baseline 作为主生产入口。
2. 重启后确认日志中的 `tool_call_parser='qwen3_coder'`，否则 Agent 工具调用只会停留在文本层面。
3. 对外不要直接暴露 SGLang，继续使用 Nginx 或更上层 Gateway。
4. 多用户并发不要只提高 `MAX_RUNNING_REQUESTS`，要同时控制 prompt tokens、`max_tokens`、队列长度和每用户并发。
5. KV cache 优化先用现有 radix cache 和 `page_size=1`，再考虑 `--enable-cache-report` 观测命中率。
6. 如果 KV pool 明显不足，优先小步提高 `MAX_TOTAL_TOKENS` 或实验 `fp8_e4m3` KV cache。
7. 只有在大量长前缀重复且 GPU KV pool 不够时，再把 HiCache 作为第二阶段方案。
8. 任何上下文、并发、KV dtype、HiCache 调整，都必须重新运行 `verify_qwen36_27b.py`，并补充真实 Agent 工具调用验收。
