# `python/sglang/srt/grpc` 模块分析

## 定位

`grpc` 是 SRT Python 包内的 gRPC 边界占位模块。当前目录只有 `__init__.py`，实际 gRPC 服务入口主要在 `entrypoints/grpc_server.py`、`disaggregation/encode_grpc_server.py` 以及仓库 Rust gRPC 扩展中。

## 当前状态

该顶层包本身不承载业务逻辑。它的存在更多是为了包路径稳定和后续扩展；SRT 中真正的 gRPC 相关实现分散在：

- `entrypoints/grpc_server.py`：推理服务 gRPC 入口。
- `disaggregation/encode_grpc_server.py`：多模态 encoder 分离的 gRPC 服务。
- `rust/sglang-grpc`：PyO3/tonic 侧 gRPC 扩展。
- `proto/sglang/runtime/v1/`：协议定义。

## 设计要点和风险

- 不应把 `grpc` 空包误认为完整服务实现；阅读 gRPC 路径时要从 entrypoints/disaggregation/Rust/proto 联合看。
- 后续若在该包下新增实现，要避免与已有 `entrypoints/grpc_server.py` 职责重复。
