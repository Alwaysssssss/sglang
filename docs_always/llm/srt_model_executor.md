# `python/sglang/srt/model_executor` 模块分析

## 定位

`model_executor` 是模型实际执行层。它在每个 rank 上初始化分布式环境和模型对象，创建 KV cache 与 attention backend，管理 CUDA graph / piecewise CUDA graph / CPU graph / MindSpore runner，构造 `ForwardBatch`，并把模型输出转换为 scheduler 可消费的 logits、hidden states、embedding 或 pooler 输出。

## 关键文件

- `model_runner.py`：核心 `ModelRunner`。负责分布式初始化、设备/dtype、模型加载、attention backend、sampler、LoRA、EPLB、CUDA graph、offloader、权重更新和 forward。
- `forward_batch_info.py`：定义 `ForwardMode`、`ForwardBatch`、`CaptureHiddenMode`、`PPProxyTensors` 等执行元数据。
- `cuda_graph_runner.py`：decode/prefill 相关 CUDA graph capture、input buffer、replay 和 torch compile 配置。
- `piecewise_cuda_graph_runner.py`：piecewise capture/compile 路径，把更细粒度的图段接入 prefill/decode。
- `cpu_graph_runner.py`：CPU graph 路径。
- `model_runner_kv_cache_mixin.py`：KV cache 初始化、memory pool 配置和 cache 相关辅助。
- `input_buffers.py`：forward 输入 buffer 抽象。
- `hook_manager.py`：按配置给模型注册 forward hook。
- `mindspore_runner.py`：MindSpore/HCCL 场景的 runner 初始化。

## 运行流程

`TpModelWorker` 创建 `ModelRunner` 后，`ModelRunner` 首先依据 `ServerArgs` 初始化 `torch.distributed`、TP/PP/DP/EP 进程组和平台后端，然后通过 `model_loader` 解析模型架构并加载权重。初始化阶段还会创建 KV cache、sampler、attention backend、LoRA manager、CUDA graph runner、EPLB manager 等。

forward 阶段，scheduler 传入 `ModelWorkerBatch`，executor 将其转成 `ForwardBatch`。`ForwardBatch` 带着 forward mode、positions、seq lens、cache loc、sampling info、attention metadata 进入模型。模型内部的 `RadixAttention` 等层通过 `forward_batch.attn_backend` 执行 attention，并将 KV 写入 `mem_cache`。最后 `ModelRunnerOutput` 返回 logits/next token/hidden states/embedding 等结果。

## 依赖关系

该模块向下调用 `models`、`layers`、`model_loader`、`mem_cache`、`distributed`、`sampling`、`lora`、`eplb`、`elastic_ep`、`utils`。它向上被 `managers.tp_worker` 调用。`forward_batch_info` 同时被 attention backend、speculative、batch overlap、CUDA graph 等模块读取，因此属于跨模块契约。

## 设计要点和风险

- `ModelRunner` 同时管理模型生命周期和性能路径，初始化顺序很关键：分布式组、dtype、模型加载、KV cache、attention backend、graph capture 之间存在隐式依赖。
- CUDA graph 和 `torch.compile` 对 tensor shape、buffer 复用、forward mode 有强约束；新增模型层或 attention backend 要确认 capture 路径不产生 graph break。
- 权重更新、remote instance loading、LoRA、offload 和 EPLB 都可能在运行中改模型状态，需要严格同步 rank-local 与全局状态。
- `ForwardBatch` 字段是性能关键路径，不应轻易引入 Python 对象或动态结构，否则会影响 compile/cudagraph。
