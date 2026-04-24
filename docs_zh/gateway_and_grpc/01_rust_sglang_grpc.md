# `rust/sglang-grpc`：Worker 端的 Rust gRPC 扩展

## 1. 作用定位

`rust/sglang-grpc` 是一个**被 Python 加载的 Rust 扩展模块**，目标是在每个 SGLang Worker 进程内嵌一个**高性能 gRPC 服务器**，替代纯 Python 实现的 gRPC 入口。

- crate 名：`sglang-grpc`（见 `rust/sglang-grpc/Cargo.toml`）
- 构建产物：`cdylib`，模块名 `_core`——即 Python 里 `from sglang.srt.grpc import _core` 可直接加载的 `.so`
- 语言版本：Rust 2024 edition
- 依赖核心：
  - `pyo3`：绑定到 Python，暴露 `start_server` / `GrpcServerHandle`
  - `tokio`：运行时
  - `tonic 0.12`：gRPC 服务端
  - `prost 0.13` / `tonic-build 0.12`：Proto 代码生成
  - `tokenizers`：HuggingFace Tokenizer 的 Rust 实现（用于后续 PR 中的本地分词加速）
  - `tracing` / `tracing-subscriber`：结构化日志
  - `crossbeam-channel` / `async-stream` / `tokio-stream`：与 Python 侧运行时做跨线程消息传递所需的原语

## 2. 目录与文件

```
rust/sglang-grpc/
├── Cargo.toml     # 依赖与 cdylib 配置
├── build.rs       # 基于 tonic-build 编译共享 proto
└── src/
    └── lib.rs     # PyO3 扩展入口 + gRPC server handle 骨架
```

核心源码长这样（来自 `src/lib.rs`）：

```1:8:rust/sglang-grpc/src/lib.rs
use pyo3::prelude::*;
use std::sync::Arc;
use tokio::sync::Notify;

pub mod proto {
    tonic::include_proto!("sglang.runtime.v1");
}
```

```10:33:rust/sglang-grpc/src/lib.rs
/// Handle returned by `start_server` — used to shut down the gRPC server.
#[pyclass]
pub struct GrpcServerHandle {
    shutdown: Arc<Notify>,
    join_handle: Option<std::thread::JoinHandle<()>>,
}

#[pymethods]
impl GrpcServerHandle {
    /// Signal the server to stop and wait for the background thread to exit.
    fn shutdown(&mut self) {
        self.shutdown.notify_one();
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }

    /// Returns `true` while the server thread is still running.
    fn is_alive(&self) -> bool {
        self.join_handle
            .as_ref()
            .map_or(false, |h| !h.is_finished())
    }
}
```

对外暴露的 Python API 只有两个符号：

- `start_server(host, port, runtime_handle) -> GrpcServerHandle`
- `GrpcServerHandle.shutdown()` / `GrpcServerHandle.is_alive()`

## 3. gRPC 协议契约

扩展编译时通过 `build.rs` 把**仓库级 proto**（位于 `proto/sglang/runtime/v1/sglang.proto`）编译进来：

```1:16:rust/sglang-grpc/build.rs
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_path = "../../proto/sglang/runtime/v1/sglang.proto";

    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .protoc_arg("--experimental_allow_proto3_optional")
        .file_descriptor_set_path(
            std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap())
                .join("sglang_descriptor.bin"),
        )
        .compile_protos(&[proto_path], &["../../proto"])?;
```

几个关键点：

1. **只编译 server，不编译 client**（`build_server(true).build_client(false)`）——因为 client 生成放在 gateway 那边。
2. **启用 proto3 可选字段**（`--experimental_allow_proto3_optional`），与 gateway 侧 tonic 0.14 生成的客户端兼容。
3. 生成 `file_descriptor_set`，为未来开启 gRPC reflection 做准备。

对应的 `service SglangService`（`proto/sglang/runtime/v1/sglang.proto`）是两边共同遵守的契约，关键 RPC 分三组：

- **SGLang 原生 RPC（typed proto）**：`TextGenerate`、`Generate`、`TextEmbed`、`Embed`、`Classify`、`Tokenize`、`Detokenize`、`HealthCheck`、`GetModelInfo`、`GetServerInfo`、`ListModels`、`GetLoad`、`Abort`、`FlushCache`、`PauseGeneration`、`ContinueGeneration`
- **OpenAI 兼容 RPC（JSON 透传）**：`ChatComplete`、`Complete`、`OpenAIEmbed`、`OpenAIClassify`、`Score`、`Rerank`
- **运维 RPC**：`StartProfile`、`StopProfile`、`UpdateWeightsFromDisk`

## 4. 运行时拓扑

启动流程（`start_server` 内部）：

```42:77:rust/sglang-grpc/src/lib.rs
fn start_server(host: String, port: u16, runtime_handle: PyObject) -> PyResult<GrpcServerHandle> {
    let _ = &runtime_handle; // Will be used in Phase 1 PR 2
    let shutdown = Arc::new(Notify::new());
    let shutdown_clone = shutdown.clone();

    let addr_str = format!("{}:{}", host, port);
    let addr: std::net::SocketAddr = addr_str
        .parse()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Bad address: {e}")))?;

    let join_handle = std::thread::Builder::new()
        .name("grpc-server".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(4)
                .enable_all()
                .build()
                .expect("Failed to build Tokio runtime");

            rt.block_on(async move {
                tracing::info!("gRPC server listening on {}", addr);
                // Server implementation will be added in PR 2.
                // For now, just wait for shutdown signal.
                shutdown_clone.notified().await;
                tracing::info!("gRPC server shutting down");
            });
        })
```

特点：

- **独立 OS 线程 + 独立 tokio 运行时**：避免影响 Python 侧的 asyncio 事件循环，也绕开 GIL 冲突。
- **4 个 worker thread**：默认配置，可按需扩展。
- **通过 `Arc<Notify>` 做优雅关闭**：Python 端显式调用 `handle.shutdown()` 时发信号，线程退出。
- `runtime_handle: PyObject`：Python 侧会传一个 `RuntimeHandle`（见仓库内 `python/sglang/srt/grpc/grpc_bridge.py` 体系），Rust 侧将在后续 PR 中通过 `pyo3::Python::with_gil` 回调 Python 执行真正的生成逻辑。

## 5. 在 SGLang 中的集成点

Python 侧会把这个 `.so` 打包到 `sglang.srt.grpc` 命名空间下，由 `grpc_server.py` 决定使用哪种实现：

- 纯 Python 版（`grpcio` + `asyncio`）：默认，便于调试
- Rust 版（`rust/sglang-grpc`）：通过 env 开关或 `--grpc-mode` 组合启用，追求更高吞吐 / 更低 CPU 开销

在 `sgl-model-gateway/README.md` 中也提到："The gRPC router streams tokenized requests directly to SRT gRPC workers"——这里的"SRT gRPC worker"所监听的端口，就是由这个 Rust 扩展提供服务的端口。

## 6. 设计考量

1. **为什么必须是 Rust 扩展，而不是单独进程？**
   - gRPC 到 Python 业务逻辑之间的开销要尽可能小；嵌入进程内后，`pyo3::Python::with_gil` 就能直接调用 Python 端的调度器，不需要再跨进程做一次 RPC。
2. **为什么用 tonic 0.12 而非更高？**
   - 与 `tokenizers 0.21`、PyO3 0.23 等并存时的 ABI/版本约束选择；gateway 侧独立进程使用 tonic 0.14，两边互不干扰。
3. **为什么只构建 `cdylib` 不构建 `rlib`？**
   - 该 crate 只作为 Python 扩展使用，不会被其它 Rust 库 link。`crate-type = ["cdylib"]`（见 `Cargo.toml`）是标准 PyO3 写法。
4. **为什么在编译产物里保存 file descriptor set？**
   - 为 tonic gRPC reflection 留好接入点，将来运维可以直接用 `grpcurl` 无模型文件发现服务。

## 7. 开发者速查

```bash
# 在子 crate 目录中单独构建（需要系统安装 protoc）
cd rust/sglang-grpc
cargo build --release

# 在 CI / 本地迭代（debug 更快）
cargo build

# 作为 Python 扩展安装（仓库根执行）
# 通常由上层 sglang 的 maturin 或 setup.py 触发
```

预期产物：`target/{debug,release}/libsglang_grpc.so`，随后被 Python 打包脚本更名为 `_core.cpython-*.so` 放到 `sglang/srt/grpc/` 下。

## 8. 小结

`rust/sglang-grpc` 是 SGLang Worker 端的 **gRPC 服务骨架**：

- 把 gRPC 服务端从 Python 搬到 Rust + tokio，降低尾延迟
- 通过 PyO3 与 Python RuntimeHandle 打通，业务逻辑仍在 Python 侧
- 与 `sgl-model-gateway` 共享同一份 proto，形成闭环：**Gateway 的 Rust gRPC 客户端 → Worker 的 Rust gRPC 服务端**
