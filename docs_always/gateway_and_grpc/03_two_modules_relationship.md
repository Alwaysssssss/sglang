# `rust/` 与 `sgl-model-gateway/` 的协作关系

本篇把前两篇内容串起来，从**一次真实请求**和**一次上线部署**两个角度，说明两个 Rust 模块如何共同完成服务目标。

## 1. 一次 gRPC 请求的生命周期

以 `/v1/chat/completions` （gRPC 后端）为例：

```
[Client]
   │ POST /v1/chat/completions   (HTTP, OpenAI JSON)
   ▼
[sgl-model-gateway]  ─── src/server.rs (axum)
   │
   ├─ middleware.rs:  tracing / request-id / RBAC / token-bucket
   │
   ├─ routers/grpc/router.rs:
   │     1) llm-tokenizer  对 prompt 做本地 tokenize（Rust 原生）
   │     2) Policy Registry 选出目标 worker
   │     3) 若命中 PD：按 regular 或 pd_router.rs 分流
   │
   ├─ smg-grpc-client (tonic 0.14 Client)
   │     ChatComplete(OpenAIRequest) → stream<OpenAIStreamChunk>
   ▼
   ──── gRPC over HTTP/2 ────────────────────────────────
   ▼
[sglang worker process]
   │
   ├─ python 解释器
   │   └─ sglang.srt.grpc 模块动态 import `_core`
   │         ←── rust/sglang-grpc 构建出来的 cdylib
   │
   ├─ rust/sglang-grpc 通过 tonic 0.12 Server
   │   在独立 OS 线程 + tokio 运行时监听端口，收到 RPC
   │
   ├─ 通过 pyo3::Python::with_gil 回调 Python RuntimeHandle
   │   执行真正的 scheduler → model forward → sampler 循环
   │
   └─ stream 回吐 token / OpenAI chunk
   ▼
[sgl-model-gateway]  ─── routers/grpc/pipeline.rs
   │
   ├─ reasoning-parser:  拆 <think>…</think>
   ├─ tool-parser:       解析函数调用
   ├─ 聚合为 OpenAI SSE chunk
   ▼
[Client]  (text/event-stream)
```

两个模块在这条链路上严格分工：

| 阶段 | 角色 | 归属模块 |
| --- | --- | --- |
| HTTP 入口、OpenAI Schema 解析 | 网关 | `sgl-model-gateway` |
| Tokenizer / Reasoning / Tool 解析 | 网关 | `sgl-model-gateway`（`llm-tokenizer` / `reasoning-parser` / `tool-parser`） |
| Worker 选择 / 熔断 / 重试 | 网关 | `sgl-model-gateway` |
| gRPC 客户端 | 网关 | `sgl-model-gateway`（tonic 0.14 + `smg-grpc-client`） |
| **gRPC 服务端 + 监听 loop** | Worker | **`rust/sglang-grpc`**（tonic 0.12 + PyO3） |
| 实际模型 forward | Worker | SGLang Python Runtime |

## 2. 共享的 proto 契约

仓库里只有一份 proto：`proto/sglang/runtime/v1/sglang.proto`。

- `rust/sglang-grpc/build.rs` 编译出的是 **server 代码**（`build_server(true).build_client(false)`）。
- `sgl-model-gateway` 则通过 `smg-grpc-client`（及其自身 `build.rs`/私有 crate）编译出 **client 代码**。

这意味着 **proto 一旦变更，两个 crate 都需要重新构建**——这是它们最硬的耦合点，但也是唯一的契约。除此之外两者完全独立。

## 3. 版本差异 & 边界

| 维度 | `rust/sglang-grpc` | `sgl-model-gateway` |
| --- | --- | --- |
| Rust edition | 2024 | 2021 |
| tonic 版本 | 0.12 | 0.14 |
| prost 版本 | 0.13 | 0.14 |
| tokio 版本 | 1.x（未钉） | 1.42 |
| 输出类型 | `cdylib` | `rlib` + 三个 `bin` |
| 发布单位 | 嵌入到 `sglang` Python 包里 | 独立二进制 + `sglang_router` wheel + Go SDK + Docker |

不同 tonic 版本只要 **wire protocol（HTTP/2 + protobuf）**一致即可互通；二者通过 gRPC wire 通信而不是共享类型，所以即使版本错位也没问题。

## 4. 部署拓扑样例

### 小集群（单机多 GPU）

```
┌────────────────────────────────────────────────┐
│ 主机                                            │
│                                                │
│  python -m sglang_router.launch_server \       │
│     --grpc-mode --dp-size 8                    │
│                                                │
│  ├─ 进程 1: sgl-model-gateway (:30000)         │
│  │    ↑ bind axum，开启 gRPC router            │
│  │                                             │
│  └─ 进程 2..9: sglang worker (:20001..20008)   │
│       └─ 每个进程加载 rust/sglang-grpc         │
│         监听本地 gRPC 端口                      │
└────────────────────────────────────────────────┘
```

### 生产 K8s 集群

```
Ingress / LoadBalancer
       │
       ▼
 ┌───────────────────────────┐
 │ Deployment: sgl-gateway   │  ← sgl-model-gateway 镜像
 │  副本 N，无状态            │
 │  --service-discovery       │
 │  --selector app=sglang-worker
 │  --enable-trace            │
 └──────┬────────────────────┘
        │ gRPC
        ▼
 ┌───────────────────────────┐
 │ Deployment/StatefulSet:   │
 │   sglang-worker            │  ← 包含 rust/sglang-grpc.so
 │   annotation:             │
 │     sglang.ai/bootstrap-  │
 │     port: 9001             │
 └───────────────────────────┘
```

网关通过 `kube::runtime::controller` 监听 Pod Informer 把新 worker 自动注册到 Worker Registry；worker 端由 `rust/sglang-grpc` 提供 gRPC 端口 + Python Runtime 处理实际计算。

## 5. 两者都**不**关心的事

澄清边界可以避免误用：

- **`rust/sglang-grpc` 不做**：请求路由、限流、Tokenizer 选型、OpenAI JSON 兼容、工具解析。这些都由 gateway 负责。
- **`sgl-model-gateway` 不做**：模型 forward、KV Cache、采样、实际推理。它只调度请求，从不执行模型。

## 6. 修改建议（给开发者）

| 改动目标 | 修改位置 |
| --- | --- |
| 新增/修改 RPC 方法 | 1) `proto/sglang/runtime/v1/sglang.proto`；2) `rust/sglang-grpc/src/lib.rs`（server 实现）；3) `sgl-model-gateway/src/routers/grpc/*`（client 调用） |
| 新增负载均衡策略 | `sgl-model-gateway/src/policies/` 新增实现 + `registry.rs` 注册 + `factory.rs` 分发 |
| 新增 OpenAI 端点 | `sgl-model-gateway/src/routers/{http,grpc,openai}/` 对应路由器 + `server.rs` 注册 axum route |
| 新增观测指标 | `sgl-model-gateway/src/observability/metrics.rs` |
| Worker 端性能优化（分词 / 流式） | `rust/sglang-grpc`（搬更多 Python 逻辑到 Rust） |
| gRPC 客户端池优化 | `sgl-model-gateway/src/routers/grpc/client.rs` |

## 7. 延伸阅读

- 本目录：
  - [README.md](./README.md) — 总览
  - [01_rust_sglang_grpc.md](./01_rust_sglang_grpc.md)
  - [02_sgl_model_gateway.md](./02_sgl_model_gateway.md)
- 原始资料：
  - `sgl-model-gateway/README.md`
  - `rust/sglang-grpc/{Cargo.toml,build.rs,src/lib.rs}`
  - `proto/sglang/runtime/v1/sglang.proto`
  - 官方文档：[docs.sglang.io/advanced_features/sgl_model_gateway.html](https://docs.sglang.io/advanced_features/sgl_model_gateway.html)
