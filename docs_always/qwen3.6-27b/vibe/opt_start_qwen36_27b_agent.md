# Feature: 优化 Qwen3.6-27B Agent 启动方案

> 本文是实施方案，不包含实现代码。后续如执行，需要按本文范围修改脚本、测试和文档，并在真实 GPU 环境完成验收。

## 0. 需求归纳

原始需求有三项：

- 日志更丰富：能够定位每次用户输入、模型输出、耗时、排队、显存和关键启动参数。
- 内存优化：保持 `256K` 上下文，目标最大运行并发为 `6`。需要比较两种方向：
  - 方案一：静态预算方式，保持 `256K` 上下文，将最大运行并发固定或收敛到 `6`。
  - 方案二：只常驻模型主体显存，其它 KV/request 资源尽量在请求使用时分配，仍保持 `256K` 上下文和 `6` 并发目标。
- 前端不再显示 `</think>` 或 `<think>...</think>` 推理标记，避免用户看到残留 reasoning 标签。

## 1. 当前事实

目标启动入口是 `docs_always/qwen3.6-27b/start_qwen36_27b_agent.sh`。当前脚本已经具备一部分能力：

| 能力 | 当前状态 | 影响 |
| --- | --- | --- |
| 上下文长度 | `CONTEXT_LENGTH=262144` | 已是 256K 目标。 |
| 输出上限 | `MAX_OUTPUT_TOKENS=128000` | 写入 client defaults，不直接限制 server 全局生成。 |
| TP | `TP_SIZE=4` | 面向 4 张 A100 80GB。 |
| 显存预算 | `MEMORY_TARGET_FRACTION=0.90`，并默认扣除当前 GPU 已用显存 | 能避免已有进程导致的错误估算。 |
| 运行并发 | 自动根据显存估算，且 `MAX_RUNNING_REQUESTS_CAP=8` | 与需求的 `6` 并发目标不一致。 |
| request log | `LOG_REQUESTS=1`、`LOG_REQUESTS_LEVEL=3`、`LOG_REQUESTS_FORMAT=json` | 已能记录完整输入输出，但有日志体积和敏感信息风险。 |
| metrics | request-time stats、Prometheus metrics、metrics 文件导出默认开启 | 已具备性能排查基础。 |
| tool call parser | `TOOL_CALL_PARSER=qwen3_coder` | 已支持 Qwen tool call 解析方向。 |
| reasoning parser | 当前未暴露 | `<think>` / `</think>` 治理缺少服务端解析层。 |

相邻文档已经覆盖启动拓扑、OpenAI 请求链路、scheduler/prefill/radix cache、observability 和参数说明，实施时应同步参考：

- `docs_always/qwen3.6-27b/docs/sglang_qwen36_27b_bootstrap_serverargs_topology.md`
- `docs_always/qwen3.6-27b/docs/sglang_qwen36_27b_openai_tokenizer_toolcalls.md`
- `docs_always/qwen3.6-27b/docs/sglang_qwen36_27b_scheduler_prefill_radix_cache.md`
- `docs_always/qwen3.6-27b/docs/sglang_qwen36_27b_observability_troubleshooting.md`
- `docs_always/qwen3.6-27b/docs/sglang_qwen36_27b_server_args_and_request_schema.md`

## 2. 推荐结论

优先实施方案一，并把 `6` 作为明确的运行态并发上限。理由：

- 当前 SGLang serving 路径会在启动阶段按 `mem_fraction_static` profiling 并初始化 KV pool；“只加载模型，其它完全用时再分配”不是当前脚本已证明支持的默认语义。
- 256K 长上下文下，稳定性比抢占更多瞬时显存更重要。固定 `MAX_RUNNING_REQUESTS=6` 比继续让脚本最高估到 `8` 更符合需求，也更利于验收和容量规划。
- 当前日志、metrics、request-time stats 基础已经存在，主要缺口是并发目标收敛、reasoning parser 暴露、测试断言同步和验收场景补齐。

方案二不应直接作为首轮生产变更。它可以作为第二阶段调研项，先确认当前 SGLang 是否支持更接近“按需分配”的 KV/cache 模式，或是否需要依赖 HiCache、CPU/offload、disaggregated serving、降低 `mem_fraction_static`、限制 `MAX_TOTAL_TOKENS` 等替代手段。未证明前，不应在方案中承诺“只加载模型内存”。

## 3. 目标状态

### 3.1 日志与可观测性

- 启动日志必须包含关键配置：模型路径、服务模型名、TP、可见 GPU、256K 上下文、输出上限、显存预算、运行并发、排队上限、prefill 参数、request log 目录、metrics 目录、server log 路径和 redacted launch command。
- request logging 在调试/内网验收阶段保留 `level 3`，确保可看到用户输入和模型输出。
- request log 必须使用 JSON 格式，方便按 `rid`、时间、事件类型、输入输出字段检索。
- request-time stats 和 metrics 文件导出保持开启，用于确认 TTFT、E2E、queue、prefill/decode、finish reason 等指标。
- 明确安全边界：`level 3` 会记录完整输入输出，生产长期运行需配套访问权限、留存周期、脱敏策略，或降级到 `level 1/2`。

### 3.2 内存与并发

- 默认保持 `CONTEXT_LENGTH=262144`。
- 将目标运行态并发收敛为 `MAX_RUNNING_REQUESTS=6` 或将自动推导 cap 调整为 `6`，避免脚本在空闲 A100 上继续给出 `8`。
- `MAX_QUEUED_REQUESTS` 随运行并发按固定倍数推导，建议保持 `MAX_RUNNING_REQUESTS * 8`，即默认 `48`。
- `PREFILL_MAX_REQUESTS` 默认跟随运行并发，避免单轮 prefill 请求数超过运行态设计。
- `CHUNKED_PREFILL_SIZE=8192`、`MAX_PREFILL_TOKENS=16384` 先保持不变，降低一次性扩大变量带来的定位难度。
- 启动后必须从 server log 读取真实 `max_total_num_tokens`。如果真实 KV token pool 小于 `6 * 262144` 再加必要 headroom，则不能宣称“6 个满 256K 请求同时运行”，只能宣称“最多 6 个请求进入 running 状态，实际长上下文并发受 KV pool 和请求长度约束”。

### 3.3 `</think>` 不显示

首选服务端 reasoning parser 加客户端渲染约束：

- 启动脚本增加可配置的 `REASONING_PARSER`，默认建议为 `qwen3`，并传给 SGLang `--reasoning-parser`。
- 对 OpenAI chat 响应，客户端或 Agent Gateway 只渲染 `content`，不把 `reasoning_content` 展示给普通用户。
- 流式响应场景下，如果产品不希望展示推理过程，应设置或约定 `stream_reasoning=false`，只向 UI 发送普通内容 delta。
- 如果当前前端或网关还不能消费 `reasoning_content`，短期兜底是在前端/网关渲染层过滤 `<think>...</think>`、孤立 `<think>` 和孤立 `</think>`。兜底过滤只作为兼容措施，不作为首选根因修复。
- 不推荐用 `stop` 或 `stop_regex` 简单截停 `</think>`。这可能在模型刚结束 reasoning 时直接截断回答，导致用户拿不到最终正文。

## 4. 范围

### 4.1 需要修改的文件

| 文件 | 修改目的 |
| --- | --- |
| `docs_always/qwen3.6-27b/start_qwen36_27b_agent.sh` | 收敛 256K/6 并发默认值；暴露 reasoning parser；保留和补强日志/metrics 配置；确保启动摘要可审计。 |
| `docs_always/qwen3.6-27b/test_start_qwen36_27b_agent.py` | 同步 dry-run 断言：256K、6 并发、request log level、reasoning parser、metrics 参数、redacted command。 |
| `docs_always/qwen3.6-27b/test_verify_qwen36_27b.py` | 如验证脚本新增 reasoning 或隐藏 think 的检查，需要补充对应单元测试。 |
| `docs_always/qwen3.6-27b/verify_qwen36_27b.py` | 补充运行验收项：6 并发、长上下文、流式、非流式、`</think>` 不出现在用户可见 content。 |
| `docs_always/qwen3.6-27b/start_qwen36_27b_agent.md` | 更新启动说明、配置档位、reasoning parser、日志安全说明和验收命令。 |
| `docs_always/qwen3.6-27b/docs/*.md` | 只同步受影响的参数表和排查说明，不做无关扩写。 |
| 前端或 Agent Gateway 渲染层 | 如果用户可见内容仍来自未解析的 raw text，需要在展示层忽略 reasoning 或做兜底过滤。当前仓库未定位到该前端文件，实施前需要确认代码位置。 |

### 4.2 不应修改的文件

| 文件或范围 | 原因 |
| --- | --- |
| `python/sglang/srt/*` | 首轮只调整部署脚本和调用策略，不改 SGLang 内核行为。 |
| 模型权重目录 `/mnt/shanhai-ai/wenhy/models/Qwen/Qwen/Qwen3___6-27B` | 模型文件不是本次需求范围。 |
| Nginx / Cloudflare 配置 | 除非验收发现上游超时或限流阻断 256K/6 并发，否则不在首轮修改。 |
| 无关启动脚本 `start_qwen36_27b.sh` | 本需求指向 agent 启动入口，避免扩大影响面。 |

## 5. 设计方案

### 5.1 日志方案

启动脚本继续分三类输出：

| 类型 | 目标 | 验收点 |
| --- | --- | --- |
| start log | 记录启动前检查、显存估算、最终参数、ready 结果 | 能定位脚本参数错误、端口占用、API key、GPU 显存预算、ready 超时。 |
| server log | 记录 SGLang server_args、KV capacity、scheduler、异常、request-time stats | 能确认 `context_length=262144`、`max_running_requests=6`、`tool_call_parser`、`reasoning_parser`、KV pool。 |
| request log | 记录请求输入输出 | JSON 中能按 rid 找到用户输入、模型输出、finish reason 和错误。 |

实施时不要只“打开更多 stdout”。更重要的是让同一次请求能跨 request log、metrics、server log 关联。建议要求客户端传入稳定 `rid`；若客户端不传，服务端生成的 rid 也要能在日志中检索。

### 5.2 内存方案一：256K + 6 并发静态预算

这是首选方案。

目标行为：

- `CONTEXT_LENGTH` 保持 `262144`。
- `MAX_RUNNING_REQUESTS` 默认固定为 `6`，或自动估算后最多 cap 到 `6`。
- `MAX_RUNNING_REQUESTS_CAP` 默认改为 `6`，避免脚本在显存充足时扩大到 `8`。
- `MAX_QUEUED_REQUESTS` 默认 `48`。
- `PREFILL_MAX_REQUESTS` 默认 `6`。
- 保留 `RESPECT_CURRENT_GPU_USAGE=1`，避免有旧进程占卡时高估可用显存。
- 保留启动前模型 shard、service budget、KV budget、estimated tokens 的日志，帮助判断 6 并发是否只是 running slots，还是能承载接近满上下文的 6 请求。

关键验收判断：

- 如果 server log 显示真实 `max_total_num_tokens` 足够覆盖 `6 * 262144` 并留有输出和调度余量，才能把容量表述为“接近 6 个 256K 请求”。
- 如果不足，应在文档中明确业务 admission policy：例如单用户 1 个长任务、限制 prompt tokens、限制输出 tokens、长任务进入异步队列。

### 5.3 内存方案二：模型常驻，其它按需分配

这是调研/实验方案，不建议首轮直接承诺。

需要先回答：

- 当前 SGLang 版本是否支持不在启动阶段预留大 KV pool 的 serving 模式。
- `mem_fraction_static` 降低后，是否只是减少 KV pool 容量，而不是实现真正按需增长。
- `MAX_TOTAL_TOKENS` 是否能作为硬上限帮助减少预留，同时不牺牲 6 并发验收。
- HiCache、hierarchical cache、host memory、KV cache dtype 或 disaggregated serving 是否能满足“降低 GPU 常驻显存”的真实目标。
- 这些能力对 TTFT、decode、cache hit、稳定性和故障恢复有什么代价。

实验策略：

- 不和方案一混在一次上线里做。
- 单独开实验档位，固定 256K/6 的请求模型，逐项比较启动显存、首 token 延迟、E2E、OOM、KV retract、queue。
- 只有在真实压测优于方案一，并且故障模式清晰时，才作为生产候选。

### 5.4 `</think>` 治理方案

分三层处理：

| 层 | 做法 | 说明 |
| --- | --- | --- |
| 服务端解析 | 启用 `qwen3` reasoning parser | 将 reasoning 从普通 content 中拆出，减少 `<think>` 标签直接进入正文。 |
| API/Gateway | 统一响应契约 | 普通用户响应只返回或只渲染 content；内部调试可保留 reasoning_content。 |
| 前端兜底 | 过滤 think 标签残留 | 处理历史服务、未启 parser、流式半包等兼容问题。 |

验收必须同时覆盖非流式和流式：

- 非流式 `choices[0].message.content` 不包含 `<think>` 或 `</think>`。
- 流式所有展示给 UI 的 delta 拼接后不包含 `<think>` 或 `</think>`。
- 如果服务端返回 `reasoning_content`，前端不展示给普通用户。
- tool call 场景仍能正常返回 `tool_calls`，不因 reasoning parser 影响 `qwen3_coder` tool parser。

## 6. 实施步骤

### Step 1: 明确基线和配置语义

- 记录当前 `start_qwen36_27b_agent.sh` 默认值。
- 记录当前 dry-run 输出中的 context、并发、日志参数和 parser 参数。
- 记录真实启动日志中的 `max_total_num_tokens`、`max_running_requests`、`chunked_prefill_size`、`max_prefill_tokens`、`available_gpu_mem`。
- 明确“6 并发”在方案中指 SGLang running requests 上限；是否能同时承载 6 个满 256K 请求必须由真实 KV capacity 证明。

### Step 2: 收敛 256K/6 并发配置

- 将 agent 启动入口的默认并发目标收敛到 `6`。
- 保留显存自动估算，但 cap 不应超过 `6`。
- 保留 `RESPECT_CURRENT_GPU_USAGE=1` 和预算不足时 fail fast。
- 确认 `MAX_QUEUED_REQUESTS`、`PREFILL_MAX_REQUESTS` 与运行并发同步。
- 更新启动摘要，让日志直接显示最终并发来源：显式设置、自动估算、cap 后结果。

### Step 3: 保留并治理完整请求日志

- 保留 request log JSON 输出和 request-time stats。
- 明确 `LOG_REQUESTS_LEVEL=3` 是调试/验收默认，生产长期运行需要权限和留存策略。
- 确认 launch command、server log、request log 不泄露 API key。
- 要求验收请求传入 `rid`，便于串联用户输入、模型输出、metrics 和 server log。

### Step 4: 增加 reasoning parser 配置

- 给启动脚本增加可配置 reasoning parser。
- 默认建议启用 `qwen3`，但保留环境变量关闭能力，以便兼容不支持 `reasoning_content` 的客户端。
- 启动后从 server log 确认 `reasoning_parser` 生效。
- 同时确认 `tool_call_parser=qwen3_coder` 仍然生效。

### Step 5: 处理前端或 Gateway 展示

- 找到当前展示 `</think>` 的前端或 Agent Gateway 代码位置。
- 将普通用户可见内容限定为 OpenAI content 字段。
- 对 `reasoning_content` 只进入调试面板或内部日志，不进入普通回答区域。
- 对 raw text 历史兼容路径增加 think 标签兜底过滤。
- 覆盖流式半包场景，避免先显示 `</think>` 后再被清理造成闪烁。

### Step 6: 更新验证脚本和测试断言

- dry-run 测试断言应与新默认值一致：256K、6 并发、request log、metrics、tool parser、reasoning parser。
- 当前 `test_start_qwen36_27b_agent.py` 中存在 `--log-requests-level 1` 断言，而脚本默认是 `3`；实施时必须二选一对齐，不能保留矛盾。
- 验证脚本补充 `</think>` 不展示检查。
- 6 并发验收应覆盖非流式和流式，避免只测短请求。

### Step 7: 同步文档

- 更新 `start_qwen36_27b_agent.md` 的默认值、配置档位和最终建议。
- 更新 `docs/README.md` 和参数表中受影响的默认值。
- 在 observability 文档中补充日志安全说明和按 rid 排查方法。
- 在 OpenAI/tool call 文档中补充 reasoning parser 与 tool parser 同时启用时的响应契约。

### Step 8: 上线和回滚

- 先 dry-run 验证启动命令。
- 再在空闲 GPU 上启动并等待 ready。
- 通过 `/health` 和 `/v1/models` 确认服务可用、模型名和 256K context。
- 运行短请求、长上下文请求、6 并发请求、流式请求、tool call 请求、think 标签请求。
- 若出现 OOM、ready 超时、KV pool 不足或前端兼容问题，回滚到上一版启动脚本，并保留日志用于定位。

## 7. 验收标准

### 7.1 启动验收

- dry-run 输出中包含 `CONTEXT_LENGTH=262144`。
- dry-run 输出中最终运行并发为 `6`，且 launch command 传入的 running request 上限也是 `6`。
- dry-run 输出中 request logging、request-time stats、metrics file export 参数存在。
- dry-run 输出中 API key 被 redacted。
- 真实启动后 `/health` 返回成功。
- `/v1/models` 返回模型名 `qwen3.6-27b`，context length 为 `262144`。

### 7.2 日志验收

- start log 能看到 GPU snapshot、显存估算、最终并发、request log 目录、metrics 目录、server log 路径。
- server log 能确认 `context_length=262144`、`max_running_requests=6`、`tool_call_parser='qwen3_coder'`、reasoning parser 生效状态。
- request log 能按 rid 找到用户输入和模型输出。
- metrics 文件能看到完成请求的耗时和 token 统计。
- API key 不出现在日志明文中。

### 7.3 内存和并发验收

- 6 个并发请求进入运行/排队行为符合预期，不出现进程崩溃。
- 长上下文请求覆盖至少 `200K` 级别 token；不能只用 `100K` 证明 256K 配置。
- 压测期间观察 GPU 显存、`KV cache pool is full`、request retract、queue length、TTFT 和 E2E。
- 若实际 KV capacity 不足以支持 6 个满 256K 请求，验收报告必须明确真实容量边界和业务限流策略。

### 7.4 `</think>` 验收

- 普通非流式回答的用户可见内容不包含 `<think>` 或 `</think>`。
- 普通流式回答在 UI 拼接后的可见内容不包含 `<think>` 或 `</think>`。
- 如果返回 `reasoning_content`，它不会进入普通用户回答区域。
- tool call 请求仍然能产生 OpenAI 兼容 `tool_calls`，或在失败时有明确日志可定位 parser/chat template 问题。

### 7.5 回归验收

- 现有健康检查、模型列表、错误 API key、短请求、流式请求、长上下文请求仍通过。
- 现有 tool call parser 行为不退化。
- 旧的客户端默认配置文件仍能被使用，或文档明确说明新的字段和兼容策略。

## 8. 风险与应对

| 风险 | 表现 | 应对 |
| --- | --- | --- |
| 6 个满 256K 请求超出真实 KV capacity | OOM、retract、长时间排队、ready 后压测失败 | 不把 running slots 等同于满上下文并发；增加 admission control；降低输出上限或限制 prompt tokens。 |
| request log level 3 暴露敏感数据 | 用户输入、工具结果、模型输出完整落盘 | 限制日志目录权限；设置留存周期；生产长期运行改 level 1/2；必要时做脱敏。 |
| reasoning parser 改变响应结构 | 客户端忽略或错误展示 `reasoning_content` | 先在 staging 验证；保留关闭 parser 的环境变量；前端按 content/reasoning_content 分字段处理。 |
| 前端只拿 raw text | 即使服务端解析，UI 仍可能显示历史 raw text | 找到真实渲染层做兜底过滤；流式按状态机处理 think 标签。 |
| 测试与脚本默认值不一致 | CI 失败或误导后续维护 | 实施时同步更新 `test_start_qwen36_27b_agent.py`，尤其是 log level 和并发断言。 |
| 方案二语义不成立 | 以为按需分配，实际只是缩小 KV pool 或降低容量 | 单独调研 SGLang 当前能力；未证明前不作为生产承诺。 |

## 9. 回滚策略

- 启动脚本改动必须保持环境变量可覆盖。若新默认值导致问题，可临时通过环境变量恢复旧并发或关闭 reasoning parser。
- 如果服务启动失败，使用旧脚本版本重启，并保留失败的 start log 和 server log。
- 如果只是前端显示 `</think>` 异常，优先回滚展示层或关闭 reasoning parser，不要同时回滚已验证稳定的 256K/6 并发配置。
- 如果出现显存压力，优先降低 `MAX_RUNNING_REQUESTS`、`PREFILL_MAX_REQUESTS` 或输出上限，而不是关闭日志和 metrics。

## 10. 开放问题

- 当前真实生产入口是否一定是 `start_qwen36_27b_agent.sh`，还是还有 systemd、supervisor、Nginx wrapper 或平台脚本覆盖环境变量。
- 当前显示 `</think>` 的前端或 Gateway 代码位置在哪里；本仓库当前未定位到明确前端实现。
- 产品是否希望完全关闭 reasoning，还是只是不展示 reasoning。前者应考虑请求级 `reasoning_effort=none` 或 chat template 参数；后者应保留 reasoning parser 并隐藏展示。
- 6 并发是指 6 个任意长度请求、6 个 256K 输入请求，还是 6 个长程 Agent 任务。三者需要不同的验收负载。
