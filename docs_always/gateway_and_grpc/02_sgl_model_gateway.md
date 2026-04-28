# `sgl-model-gateway`：SGLang 模型网关（路由器 + 控制/数据面）

## 1. 作用定位

`sgl-model-gateway` 是 SGLang 栈里**面向客户端的第一跳**，对外提供 OpenAI 兼容的 HTTP API，对内负责把请求分发给一组异构的 SGLang Worker（HTTP 或 gRPC）。它既是**控制面**（管理 Worker 生命周期、服务发现、策略配置），也是**数据面**（路由、限流、熔断、Tokenize、工具解析）。

一句话概括（摘自 `sgl-model-gateway/README.md` 的自述）：

> *High-performance model routing control and data plane for large-scale LLM deployments. The gateway orchestrates fleets of workers, balances traffic across HTTP and gRPC backends, and exposes OpenAI-compatible APIs with pluggable history storage and tool integrations—while remaining deeply optimized for the SGLang serving runtime.*

## 2. 构建形态与命名

- crate 名：`sgl-model-gateway`，版本 `0.3.2`
- Rust 2021 edition
- 一个 `rlib`（供 bindings 复用）+ 三个 `bin`（见 `Cargo.toml`）：
  - `sgl-model-gateway` — 正式名
  - `smg` — 短名（CLI 最常用）
  - `amg` — "all model gateway"，历史别名
- 发布形式：
  - Rust 原生二进制（`cargo build --release`）
  - Python wheel：`bindings/python/`，对应包名 `sglang_router`，入口 `python -m sglang_router.launch_router`
  - Golang 客户端：`bindings/golang/`
  - Docker：`lmsysorg/sgl-model-gateway:latest`（多架构）

## 3. 目录结构

```
sgl-model-gateway/
├── Cargo.toml
├── build.rs
├── Makefile
├── README.md
├── .cargo/config.toml        # 构建/增量编译配置
├── src/
│   ├── main.rs               # CLI 入口、配置解析 (~1200 行)
│   ├── lib.rs                # 对外 re-export 的 rlib 门面
│   ├── app_context.rs        # 全局上下文：共享状态、Tokenizer、连接池
│   ├── server.rs             # axum 路由表、启动服务器
│   ├── middleware.rs         # Tower 中间件：tracing / RBAC / 限流 / 请求 ID
│   ├── service_discovery.rs  # Kubernetes Pod 发现 + Informer
│   ├── version.rs
│   ├── config/               # 配置 builder / 类型 / 校验
│   ├── core/                 # 熔断、重试、令牌桶、Job Queue、Worker 注册表
│   ├── policies/             # 负载均衡策略 (random/RR/cache/p2c/...)
│   ├── routers/              # HTTP / gRPC / OpenAI / Mesh / Parse / Tokenize 路由器
│   ├── observability/        # 日志 / Prometheus / OTel / Events
│   └── wasm/                 # 基于 wasmtime 的 WASM 中间件
├── bindings/
│   ├── python/               # maturin + PyO3：sglang_router 包
│   └── golang/               # Go 客户端 / SDK
├── benches/                  # 一致性哈希、策略、WASM、路由注册表基准
├── tests/                    # Rust 集成测试（api/mcp/otel/reliability/routing/security/...）
├── e2e_test/                 # pytest 驱动的端到端黑盒测试
├── examples/                 # WASM 中间件样例
├── scripts/                  # release notes、sccache、vision golden 等脚本
└── LICENSE -> ../LICENSE
```

关键子模块拆解见后文各节。

## 4. 核心依赖栈

| 能力 | 选型 |
| --- | --- |
| HTTP 服务 | `axum 0.8`（+ `axum-server` 0.8 TLS） |
| 中间件 | `tower` 0.5 / `tower-http` 0.6（含 tracing、cors、compression、request-id） |
| gRPC 客户端 | `tonic 0.14` + `prost 0.14` + 私有 crate `smg-grpc-client` |
| 反向代理 HTTP | `reqwest 0.12`（stream/json/rustls-tls） |
| 并发 / 数据结构 | `tokio 1.42` / `dashmap` / `parking_lot` / `arc-swap` |
| Tokenization | 私有 crate `llm-tokenizer`（重导出为 `tokenizer`） |
| Reasoning / Tool / OpenAI Schema | `reasoning-parser` / `tool-parser` / `openai-protocol`（均为私有 crate） |
| Auth | `smg-auth`（RBAC + API key）+ `jsonwebtoken 9`（JWT/OIDC） |
| MCP | `rmcp 0.8` + 私有 `smg-mcp`（支持 stdio/sse/streamable-http） |
| K8s | `kube 1.1` + `k8s-openapi` |
| 观测 | `tracing` + `tracing-opentelemetry 0.28` + `opentelemetry-otlp 0.27` + `metrics-exporter-prometheus 0.17` |
| WASM 中间件 | `wasmtime 38`（component-model + async） |
| 数据库 | `redis 0.27`；Oracle / Postgres 走私有 `data-connector` |
| Mesh | `crdts 7.3` + Redis（`smg-mesh` 私有 crate） |

> 上面带"私有 crate"标记的，如 `smg-auth`、`smg-grpc-client`、`llm-tokenizer`、`reasoning-parser`、`tool-parser`、`openai-protocol`、`smg-mcp`、`wfaas`、`data-connector`、`smg-mesh`、`smg-wasm`，都是 `=1.0.0` 精确版本，说明它们来自私有 registry 或同源仓库。

构建细节：

- `Cargo.toml` 定义了 `release` / `ci` / `dev` / `dev-opt` 四套 profile，发布版以**体积**为优化目标（`opt-level="z"`、`lto="fat"`），CI 用 `thin-LTO` 折中。
- `.cargo/config.toml` + 可选 `sccache` 用于 CI 分布式缓存。
- 提供 `vendored-openssl` feature 以便在老发行版上打包。

## 5. 架构总览：控制面 + 数据面

```
                        客户端
                          │  OpenAI 兼容 HTTP（/v1/**、/generate 等）
                          ▼
           ┌────────────────────────────────┐
           │  axum + tower 中间件链          │
           │   tracing / CORS / RequestId    │
           │   RBAC(API Key / JWT-OIDC)      │
           │   限流（令牌桶 + 队列）         │
           │   超时 / 压缩 / WASM hook       │
           └──────────────┬─────────────────┘
                          ▼
                 ┌────────────────────┐
                 │ Router Manager     │  ← 多模型 IGW
                 └──┬──────┬──────┬───┘
                    ▼      ▼      ▼
             HTTP 路由  gRPC 路由  OpenAI 远端代理
             (regular /  (regular /
              PD)         PD)
                    │      │      │
                    ▼      ▼      ▼
                 ┌────────────────────┐
                 │ Policy Registry    │  random / RR / cache_aware
                 │                    │  power_of_two / bucket / manual
                 └────────┬───────────┘
                          ▼
                 ┌────────────────────┐
                 │ Worker Registry    │  health / load / CB / priority
                 │ + Job Queue        │
                 └──┬──────┬──────┬───┘
                    ▼      ▼      ▼
        HTTP Worker  gRPC Worker  OpenAI 上游
                           ↑
                   (由 rust/sglang-grpc 暴露端口)
```

### 控制面

- **Worker Registry / Manager**：维护 `WorkerConfig`、worker 类型（regular / prefill / decode）、连接模式（HTTP/gRPC）、优先级、权重、标签。
- **Job Queue**（`src/core/job_queue.rs`）：把 `POST /workers`、`PUT /workers/{id}`、`DELETE /workers/{id}` 异步化，避免阻塞客户端；通过 `GET /workers/{id}` 查询 pending/processing/failed 状态。
- **Health Checker + Load Monitor**：后台探活并反馈给熔断器与 cache-aware / power-of-two 策略。
- **Service Discovery**（`src/service_discovery.rs`，1400+ 行）：基于 `kube::runtime::controller` 的 Informer 监听 Pod，支持独立的 prefill/decode selector，支持 bootstrap port 注解 (`sglang.ai/bootstrap-port`)。
- **RBAC**：双通道认证：静态 API Key（`--control-plane-api-keys id:name:role:key`）+ JWT/OIDC（`--jwt-issuer`、`--jwt-audience`、`--jwt-role-mapping`）；角色 `admin` / `user`；可选 audit log。

### 数据面

三种路由器同台并存（由 `Router Manager` 统一编排）：

1. **SGLang HTTP Router**
   - 子形态：`regular`（单段）和 `PD`（prefill/decode 解耦，带 bootstrap port、metadata merge、stream fan-in）
   - 全部 OpenAI 兼容端点：`/generate`、`/v1/chat/completions`、`/v1/completions`、`/v1/embeddings`、`/v1/responses`、`/v1/rerank`、`/v1/classify`、`/v1/tokenize`、`/v1/detokenize`
2. **SGLang gRPC Router**（`src/routers/grpc/`）
   - 业界首个全 Rust OpenAI 兼容 gRPC 推理网关
   - 本地 Rust Tokenizer（HF/json/chat-template）、Rust Reasoning Parser（DeepSeek-R1、Qwen3、GLM4/4.7、Kimi、Step-3、MiniMax 等）、Rust Tool Parser（JSON/Pythonic/XML）
   - 同时支持单段 / PD 两种 worker 拓扑，对应 `regular/`、`pd_router.rs`、`pipeline.rs`
   - `harmony/` 子目录处理 gpt-oss 系列的 Harmony chat template
3. **OpenAI Router**（`src/routers/openai/`）
   - 把本网关当作**前置代理**转发到 OpenAI / xAI / Gemini 等兼容后端
   - 保留 SSE/流式语义，支持 `/v1/responses` 背景任务（创建/取消/删除/列项）
   - 会话（`/v1/conversations`）与响应落库到插件化 **History Backend**（见下）
4. **Mesh Router**（`src/routers/mesh/`）
   - 多实例 Gateway 之间通过 Redis + CRDT 同步全局状态（worker 视图、tokenizer registry、rate limiter token bucket 等），用于水平扩展部署
5. **Parse / Tokenize Router**
   - 暴露 `/parse/reasoning`、`/parse/function_call`、`/v1/tokenize`、`/v1/detokenize`、`/v1/tokenizers` CRUD API

## 6. 路由策略（`src/policies/`）

| 策略 | 说明 |
| --- | --- |
| `random` | 纯随机 |
| `round_robin` | 原子计数器轮询 |
| `cache_aware` | 前缀树记录历史 prompt，命中相同前缀优先复用（`--cache-threshold` 等调优） |
| `power_of_two` | 抽两台，挑负载轻者；配合 `LoadMonitor` |
| `bucket` | 按请求特征分桶 |
| `consistent_hashing` / `prefix_hash` | 适配会话粘性 / 缓存亲和 |
| `manual` | 通过请求 Header/字段手动指定目标 Worker |

`registry.rs` 提供 per-model 策略覆写，IGW 模式下不同模型可各自独立策略。

## 7. 可靠性与流量治理（`src/core/`）

- **Retry**（`retry.rs`）：指数退避 + jitter，默认最多 5 次；仅在 408/429/5xx 触发。
- **Circuit Breaker**（`circuit_breaker.rs`）：per-worker 状态机（closed/open/half-open），失败阈值、滚动窗口可配。
- **Token Bucket + Queue**（`token_bucket.rs`）：`--max-concurrent-requests` 决定容量，`--rate-limit-tokens-per-second` 决定填充率；超过时入队（FIFO + 超时 + cancel 传递）。
- **Health Check**：`--health-check-endpoint`、`--health-check-interval-secs`，失败/成功阈值可配。
- **Cache Flush**：`POST /flush_cache` 批量清空 PD worker 的 KV 缓存。

## 8. OpenAI Responses / Conversations 与 History Backend

`sgl-model-gateway` 是少见的**把 OpenAI `/v1/responses` 与 `/v1/conversations` 做成一等公民**的推理网关：

- `/v1/responses`：支持后台任务、取消、删除、list input items；适合 agentic 多轮场景。
- `/v1/conversations`：在网关侧集中管理对话历史，避免数据泄露到第三方。
- **可插拔后端**：
  - `memory`（默认）— 进程内哈希表
  - `none` — 不存，省资源
  - `oracle` — Oracle ATP（通过 `ATP_DSN` 或 TNS alias + Wallet）
  - `postgres`
  - `redis`（`REDIS_URL` + `REDIS_RETENTION_DAYS`）

对应代码：`src/routers/openai/responses/`、`src/routers/conversations/`，底层复用私有 crate `data-connector`。

## 9. MCP（Model Context Protocol）集成

网关内置 MCP **客户端**，让工具调用循环可以跨 STDIO / SSE / Streamable HTTP 透明进行：

- 配置：`--mcp-config-path /path/to/mcp.yaml`
- 字段：`servers`、`pool`、`proxy`、`inventory`（见 README 示例）
- 私有 crate `smg-mcp` 封装工具注册/发现/刷新；底层走 `rmcp 0.8` 的三种 transport
- 监控：`smg_mcp_*` 一类 Prometheus 指标

## 10. WASM 中间件（`src/wasm/` + `examples/wasm/`）

- 基于 `wasmtime 38` 组件模型（`component-model + async`）
- 允许把自定义的请求 / 响应改写逻辑以 `.wasm` 组件形式热加载
- 基准测试 `benches/wasm_middleware_latency.rs` 量化开销
- 典型用途：敏感词拦截、header 注入、租户级策略

## 11. 观测性

### 结构化日志

`tracing` 主干 + JSON/Plain formatter（`tracing-subscriber`）；可选文件 sink `--log-dir`；`chrono` feature 带 ISO8601 时间戳。

### Prometheus（40+ 指标）

通过 `--prometheus-host/--prometheus-port` 暴露（默认 `0.0.0.0:29000`）。分层：

| 层 | 前缀 |
| --- | --- |
| HTTP 入口 | `smg_http_*` |
| Router | `smg_router_*`（含 `ttft`、`tpot`、`generation_duration`、`tokens_total`） |
| Worker | `smg_worker_*`（池大小、选择事件、连接） |
| Circuit Breaker | `smg_worker_cb_*` |
| Retry | `smg_worker_retries_*` |
| Service Discovery | `smg_discovery_*` |
| MCP | `smg_mcp_*` |
| DB | `smg_db_*` |

Duration bucket 针对 LLM 特征定制：1ms→240s。

### OpenTelemetry Tracing

- OTLP/gRPC exporter（默认 4317 端口）
- W3C Trace Context 向上游 worker 注入
- batch span processor（500ms/64 span）
- `--enable-trace` + `--otlp-traces-endpoint` 开启

### 请求 ID

`--request-id-headers x-request-id ...`；返回时回写 `x-request-id`，便于跨系统关联。

## 12. 安全

- **Router API Key**（`--api-key`）+ 初始 Worker 继承 / 动态 Worker 手动指定
- **TLS 终止**：rustls + ring；`--tls-cert-path`、`--tls-key-path`
- **mTLS 对后端**：`--client-cert-path`、`--client-key-path`、`--ca-cert-path`
- **控制面 RBAC**：API Key（哈希存储）+ JWT/OIDC（签名+iss+aud+exp+roles/groups 映射）
- **Audit Log**：`--control-plane-audit-enabled`，记录 principal / role / action / outcome

## 13. 部署与运行

### 启动形态

| 场景 | 命令（节选自 README） |
| --- | --- |
| 单模型 HTTP | `sgl-model-gateway --worker-urls http://w1:8000 --policy cache_aware` |
| PD 解耦 | `sgl-model-gateway --pd-disaggregation --prefill http://p1:30001 9001 --decode http://d1:30011` |
| 多模型 IGW | `sgl-model-gateway --enable-igw --policy cache_aware`（+ `POST /workers`） |
| gRPC | `sgl-model-gateway --worker-urls grpc://w:31001 --tokenizer-path ... --reasoning-parser ...` |
| OpenAI 代理 | `python3 -m sglang_router.launch_router --backend openai --worker-urls https://api.openai.com` |
| K8s 服务发现 | `--service-discovery --selector app=sglang-worker role=inference` |

### Python bindings

- 位置：`bindings/python/`
- 构建：`cd bindings/python && maturin develop`（开发）/ `maturin build --release --features vendored-openssl`（发版）
- ABI3 支持 Python 3.8+
- 命令行入口：`python -m sglang_router.launch_router`（仅网关）、`python -m sglang_router.launch_server`（网关 + 自动拉起若干 SGLang Worker）

### Docker

```bash
docker pull lmsysorg/sgl-model-gateway:latest
```

## 14. 测试矩阵

- **Rust 单元 / 集成测试**：`tests/` 下按主题分目录（`api/`、`reliability/`、`routing/`、`security/`、`spec/`、`wasm_test.rs`、`mcp_test.rs`、`otel_tracing_test.rs`、`metrics_aggregator_test.rs`、`inflight_tracker_test.rs`、`load_guard_raii_test.rs` 等）
- **基准**：`benches/` 含一致性哈希、手动策略、前缀树、WASM 中间件延迟、请求处理等
- **端到端 pytest**：`e2e_test/`（`chat_completions`、`embeddings`、`responses`、`router`、`benchmarks`）
- **Golden 数据**：`scripts/generate_vision_golden.py` 生成视觉 golden 用于回归

## 15. 发布流程

`Makefile` 提供 `make release-notes PREV=... CURR=...`：

- 过滤 commit 范围：`sgl-model-gateway/`、`python/sglang/srt/grpc/`、`python/sglang/srt/entrypoints/grpc_server.py`
- 自动提取作者 / PR / 新贡献者
- `CREATE_RELEASE=1` 调 `gh release create`；`DRAFT=0` 直接发布

Tag 规范：使用 `gateway-*` 或 `router-*` 前缀以便 CI 过滤。

## 16. 小结

`sgl-model-gateway` 是一整套**生产级 LLM 流量前端**，核心价值：

1. **OpenAI 兼容 API** + **SGLang 原生生态**同台：既能当 OpenAI 代理，也能做 SGLang 集群调度。
2. **全 Rust gRPC 推理路径**：tokenizer / reasoning / tool 调用全部本地执行，尾延迟显著低于 Python 实现。
3. **可靠性原语完整**：重试、熔断、限流、队列、健康检查、cache-aware / power-of-two、策略化 PD。
4. **控制面扎实**：K8s 服务发现、Job Queue、RBAC、审计、TLS/mTLS、JWT。
5. **深度可观测**：40+ Prometheus 指标 + OTLP + 结构化日志 + 请求 ID 全链路。
6. **可扩展**：WASM 中间件、MCP 工具循环、可插拔 History backend、Mesh 多实例 CRDT 同步。
