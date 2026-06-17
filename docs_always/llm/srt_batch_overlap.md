# `python/sglang/srt/batch_overlap` 模块分析

## 定位

`batch_overlap` 负责 MoE/attention 推理中的 overlap 优化，包含 two-batch overlap(TBO) 和 single-batch overlap(SBO)。它的目标是在同一轮 forward 中交错执行 attention、dispatch、MoE、combine、shared expert 等阶段，提高 GPU/通信资源利用率。

## 关键文件

- `operations.py`：定义 `YieldOperation` 和 `execute_overlapped_operations`，把模型子步骤组织成可 yield 的 operation 序列。
- `operations_strategy.py`：`OperationsStrategy` 决定如何初始化 TBO/SBO 策略。
- `two_batch_overlap.py`：TBO 主实现，包含 `TboForwardBatchPreparer`、`TboDPAttentionPreparer`、`TboCudaGraphRunnerPlugin`、`model_forward_maybe_tbo`。
- `single_batch_overlap.py`：SBO 标志、stream/event/signal 和 overlap 参数计算。

## 运行流程

模型层把 attention、MoE dispatch、expert compute、combine 等步骤拆成 operation 列表，并在可交错处插入 `YieldOperation`。TBO 会把一个 `ForwardBatch` 切成两个 child batch，按 `tbo_delta_stages` 交错执行两个 stage executor，最后合并 hidden state/residual。SBO 则在单 batch 内为 combine/down-gemm/shared expert 使用 alternate stream、event 和 SMS 配额，使通信与计算尽量重叠。

## 依赖关系

`batch_overlap` 强依赖 `model_executor.forward_batch_info.ForwardBatch`、`managers.schedule_batch.ScheduleBatch`、DP attention、layer communicator、TBO attention backend、DeepEP/Mooncake/Mori/Nixl dispatchers 和 MoE runner。它主要被 DeepSeek、Qwen3 MoE、GLM MoE、MiMo、MiniMax 等 MoE 模型层使用。

## 设计要点和风险

- TBO 只覆盖特定 sparse MoE layer 类型；dense layer 或未适配模型会禁用或 assert。
- `TboForwardBatchPreparer.filter_batch` 对 `ForwardBatch` 字段做白名单复制，新增字段若未处理会直接失败。
- extend/chunked prefill 场景需要重写 seq len、extend len、prefix len 和 cache loc，边界复杂。
- DP attention 下要求各 rank forward mode 一致，否则 TBO 不应启用。
- overlap 优化对 stream/event 生命周期敏感，错误通常表现为隐性数值错误或 hang。
