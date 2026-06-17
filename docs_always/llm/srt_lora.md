# `python/sglang/srt/lora` 模块分析

## 定位

`lora` 为 SRT 提供 LoRA adapter 的注册、加载、缓存、内存池、层包装和多后端执行。它支持在线加载/卸载 adapter、batch 内多 LoRA、overlap loading、LoRA MoE 和 LM head mixing。

## 关键文件与子包

- `lora_config.py`：`LoRAConfig`，adapter 配置。
- `lora_registry.py`：`LoRARef`、`LoRARegistry`，adapter 引用和注册表。
- `lora.py`：`LoRALayer`、`LoRAAdapter`。
- `lora_manager.py`：`LoRAManager`，模型侧 LoRA 生命周期和 batch 状态管理。
- `layers.py`：把 embedding、LM head、Column/Row/QKV linear、FusedMoE 包装成带 LoRA 的层。
- `mem_pool.py`：`LoRAMemoryPool`，adapter 权重池。
- `eviction_policy.py`：LRU/FIFO eviction。
- `lora_overlap_loader.py`：异步/overlap 加载 adapter。
- `backend/`：Triton、chunked SGMV、Torch native、Ascend、FlashInfer backend 注册。
- `triton_ops/`、`torch_ops/`：LoRA A/B、embedding、QKV、gate/up、fused MoE kernel。

## 运行流程

启用 LoRA 后，`ModelRunner` 创建 `LoRAManager`，扫描模型目标层并替换为 `layers.py` 中的 LoRA wrapper。请求指定 adapter 时，scheduler/manager 生成 `LoRABatchInfo`，后端根据 batch 中不同 adapter 的 segment 和权重池位置执行 LoRA A/B 低秩增量。在线加载时，LoRA registry 记录路径或 tensor，memory pool 分配 slot，eviction policy 在超出容量时卸载旧 adapter。

## 依赖关系

LoRA 与 `server_args`、`managers.io_struct` 的 load/unload 请求、`model_executor.model_runner`、`layers.linear`、`layers.moe`、`model_loader.weight_utils` 和 scheduler batch 状态耦合。

## 设计要点和风险

- target module 自动检测与模型命名强相关，新模型需验证目标层覆盖。
- batch 内多 LoRA 依赖 segment 构造和 memory pool slot，filter/merge batch 时必须同步更新。
- overlap loading 可减少延迟，但会引入 adapter 可见性和并发卸载时序问题。
- LoRA + quantization + MoE + CUDA graph 的组合需要逐后端确认支持。
