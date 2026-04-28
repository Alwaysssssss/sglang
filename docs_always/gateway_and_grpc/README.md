# SGLang Rust 组件：`rust/` 与 `sgl-model-gateway/`

本文档面向仓库中两处独立的 Rust 工程：

- `rust/`：轻量级 **进程内 gRPC 扩展**，通过 PyO3 被 Python 侧的 SGLang 运行时直接加载。
- `sgl-model-gateway/`：完整的 **模型网关（路由器 / 控制面 + 数据面）**，是一整套独立发布的高性能服务进程。

这两个模块在"服务 LLM 请求"的大图里处于**不同层级**：

```
┌────────────────────────────────────────────────────────────────┐
│   Client (OpenAI SDK / curl / 应用)                             │
└─────────────────┬──────────────────────────────────────────────┘
                  │ HTTP / OpenAI API
                  ▼
┌────────────────────────────────────────────────────────────────┐
│   sgl-model-gateway                                             │
│   └─ 控制面：Worker 注册 / 健康检查 / K8s 服务发现 / RBAC       │
│   └─ 数据面：路由策略 / 限流 / 熔断 / Tokenizer / Tool 解析     │
└─────────┬───────────────────┬──────────────────────────────────┘
          │ HTTP              │ gRPC (tonic client)
          ▼                   ▼
    ┌─────────────┐      ┌────────────────────────────────────┐
    │ SGLang      │      │ SGLang worker（gRPC 模式）         │
    │ worker      │      │   └─ rust/sglang-grpc (PyO3 扩展)  │
    │ (HTTP)      │      │       └─ tokio + tonic server      │
    │             │      │       └─ 调用 Python RuntimeHandle │
    └─────────────┘      └────────────────────────────────────┘
```

一句话区分：

| 维度 | `rust/sglang-grpc` | `sgl-model-gateway` |
| --- | --- | --- |
| 定位 | **Worker 端** 的 gRPC 服务端 extension | **路由器 / 网关**，对外接受请求、对内调度多个 Worker |
| 形态 | `cdylib`，被 Python 以 `import _core` 加载 | 独立二进制（`sgl-model-gateway`/`smg`/`amg`），同时提供 Python 包装 |
| 运行位置 | 跑在每个 SGLang Worker 进程内部 | 跑在集群前端，可单实例/多实例横向扩展 |
| 依赖栈 | `pyo3` + `tokio` + `tonic` | `axum` + `tower` + `tonic` + `kube` + `opentelemetry` + `wasmtime` |
| 代码规模 | 单文件几十行（骨架），随 PR 扩充 | `src/` 20+ 模块，万级行数，含路由/策略/观测/WASM |
| 对外协议 | 由 Python 侧实现业务，Rust 提供 gRPC 监听骨架 | OpenAI 兼容 HTTP + 内部 gRPC + Prometheus + OTLP |

## 阅读顺序

1. [01_rust_sglang_grpc.md](./01_rust_sglang_grpc.md) — `rust/sglang-grpc` 的定位、构建与生命周期
2. [02_sgl_model_gateway.md](./02_sgl_model_gateway.md) — `sgl-model-gateway` 的整体架构、控制面 / 数据面、运维要点
3. [03_two_modules_relationship.md](./03_two_modules_relationship.md) — 二者如何在一次请求中协作

## 关键事实速查

- `rust/sglang-grpc/build.rs` 编译的是仓库级 proto：`proto/sglang/runtime/v1/sglang.proto`，与 gateway 端的 gRPC 客户端共享同一份 Schema。
- `sgl-model-gateway` 的 gRPC 客户端走 `smg-grpc-client` 这个私有 crate，连接的正是 worker 侧由 `rust/sglang-grpc` 启动的端口。
- Gateway 的 `gRPC 路由器` 是业界少有的 **全 Rust 实现 OpenAI 兼容推理网关**：Tokenizer、Reasoning Parser、Tool Parser 全部在 Rust 进程内完成。
- Gateway 同时以 `smg`、`amg`、`sgl-model-gateway` 三个可执行名发布（见 `Cargo.toml` 的 `[[bin]]` 段）。

> 注：仓库根下仅有 `rust/` 和 `sgl-model-gateway/` 两处 Rust 工程，但它们各自独立构建、不共享 `Cargo.toml`，也不形成 workspace——是两个完全解耦的 crate。
