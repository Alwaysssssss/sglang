# `python/sglang/srt/layers` 模块分析

## 定位

`layers` 是 SRT 原生模型实现的算子和模型层库。它提供 attention 抽象与多后端实现、tensor parallel linear、embedding、layernorm、activation、MoE、量化方法、sampler、pooler、多模态 hash/conv、Mamba/linear attention 后端，以及若干平台特化工具。

## 关键文件与子包

- `radix_attention.py`：`RadixAttention` 层，是模型 attention 调用的统一入口。它根据 `ForwardBatch` 的 attention backend 执行 prefill/decode，并在 compile context 中支持 `unified_attention_with_output` split op。
- `attention/`：FlashInfer、FA3/FA4、FlashMLA、Triton、Cutlass MLA、TRTLLM、NSA、Mamba/linear/hybrid attention、Intel AMX、Ascend 等后端。
- `linear.py`：`ReplicatedLinear`、`ColumnParallelLinear`、`QKVParallelLinear`、`RowParallelLinear` 等 TP linear 层和权重分片加载逻辑。
- `moe/`：fused MoE、DeepGEMM、Triton/FlashInfer/Cutlass runner、routing、expert parallel 相关实现。
- `quantization/`：AWQ、GPTQ、FP8/FP4、bitsandbytes、compressed-tensors、modelopt、Marlin、GGUF 等量化方法。
- `sampler.py`：token 采样层，支持后端注册、torch/ascend 实现、logprob 提取和自定义 logit processor。
- `layernorm.py`、`activation.py`、`elementwise.py`：RMSNorm、LayerNorm、SwiGLU/GELU、fused elementwise Triton kernel。
- `vocab_parallel_embedding.py`、`logits_processor.py`、`pooler.py`、`sparse_pooler.py`：embedding、logits 处理、embedding/pooling 输出。
- `communicator.py`、`dp_attention.py`：attention TP/CP/DP 通信、allreduce/rmsnorm fusion、DP gather/scatter。

## 运行流程

模型文件通常组合 `linear.py` 中的 TP linear、`layernorm.py` 中的 norm、`activation.py` 中的激活、`moe/` 中的专家层和 `RadixAttention`。forward 时，`RadixAttention.forward` 先 reshape Q/K/V，再通过 `forward_batch.attn_backend.forward(...)` 调用实际后端；KV cache 写入由 attention backend 与 `mem_cache` 协作完成。采样阶段由 `Sampler` 消费 logits 和 `SamplingBatchInfo`，输出 next token 和 logprob 信息。

## 依赖关系

`layers` 被 `models` 和 `model_executor` 广泛使用，同时依赖 `distributed` 进程组、`compilation` split op、`mem_cache`、`sampling`、`utils`、`sgl_kernel`/Triton/FlashInfer/DeepGEMM 等底层库。量化层还与 `model_loader.weight_utils` 的权重加载函数耦合。

## 设计要点和风险

- attention backend 数量多，`ServerArgs.attention_backend`、模型结构、GPU 架构、KV cache dtype、MLA/MHA、sliding window、chunked prefix cache 之间有复杂兼容矩阵。
- linear/quantization 参数类承担权重切片和 layout 转换，loader 与层定义必须保持同一分片语义。
- MoE runner 与 EP/EPLB/elastic EP 共享 expert metadata，动态迁移或负载均衡路径需要关注专家位置一致性。
- 许多函数是 Triton custom op 或 `torch.compile` 路径的一部分，新增 Python 分支可能导致 graph break。
