# `python/sglang/srt/hardware_backend` 模块分析

## 定位

`hardware_backend` 收纳非标准 CUDA 路径的硬件适配，当前重点是 NPU/Ascend 和 MLX。它覆盖 memory pool、graph runner、attention backend、MoE/linear 量化方法、stream 工具和 processor patch。

## 关键子包

- `npu/`：Ascend/NPU 后端。包含 `utils.py`、NPU memory pool/allocator、Ascend attention backend、MLA preprocess、NPU graph runner、EAGLE draft graph runner、ViT graph runner、fused MoE/linear quantization 方法、Qwen VL processor patch、CMO stream 等。
- `mlx/`：Apple MLX 路径。包含 `model_runner.py`、`model_runner_stub.py`、`tp_worker.py`。

## 运行流程

平台检测到 NPU/MLX 后，`ModelRunner` 或 `TpModelWorker` 选择对应 backend。NPU 路径初始化 torch_npu/ACL 相关状态，替换 memory pool、attention backend、graph runner 和量化方法；MLX 路径通过专门的 model runner 和 tensor bridge 执行。

## 依赖关系

该模块被 `model_executor`、`layers.attention`、`layers.quantization`、`mem_cache`、`managers.tp_worker` 和 `utils.tensor_bridge` 使用。它依赖硬件厂商库，如 torch_npu、MLX。

## 设计要点和风险

- 平台后端通常无法完全覆盖 CUDA 功能矩阵，ServerArgs 中的 backend/quantization/speculative 需要显式限制。
- NPU tensor format/stream/graph runner 与 CUDA 语义不同，不能假设 CUDA 路径可直接复用。
- processor patch 会影响 transformers 预处理行为，应限制在对应平台/模型条件下启用。
