# `python/sglang/srt/sampling` 模块分析

## 定位

`sampling` 管理请求级采样参数、批级采样张量、惩罚项、grammar mask、logit bias 和自定义 logit processor。它连接入口参数、scheduler batch、模型 logits 和最终 token 选择。

## 关键文件

- `sampling_params.py`：`SamplingParams`，包含 temperature、top_p/top_k/min_p、max_new_tokens、stop、logprobs、penalty、grammar、thinking budget 等请求级配置和校验。
- `sampling_batch_info.py`：`SamplingBatchInfo`，把一批请求的采样参数转成 tensor 状态，支持 filter/merge/copy。
- `custom_logit_processor.py`：自定义 logits processor 的序列化和执行。
- `penaltylib/`：frequency/presence/repetition/min-new-tokens penalizer 和 orchestrator。

## 运行流程

Tokenizer 或 scheduler 创建并校验 `SamplingParams`。`ScheduleBatch` 将多个请求合成 `SamplingBatchInfo`。模型 forward 后，`ModelRunner` 在采样前更新 grammar vocab mask、累计并应用 penalties、应用 logit bias 和自定义 processor。随后 `layers.sampler.Sampler` 按 greedy/top-k/top-p/min-p 等策略选择 next token，并把 logprob/top logprob 等信息返回 scheduler。

## 依赖关系

`sampling` 被 `managers.schedule_batch`、`model_executor.model_runner`、`layers.sampler`、`constrained`、`speculative` 使用。它读取 `server_args` 中 deterministic/custom processor 等开关，并与 tokenizer vocab/eos/stop 语义绑定。

## 设计要点和风险

- penalty 张量常是 `batch_size x vocab_size`，大 batch/大 vocab 下显存压力明显。
- 自定义 logit processor 可能通过 `dill` 反序列化 callable，安全边界依赖入口开关和部署控制。
- grammar、penalty、logit bias、自定义 processor 的应用顺序会影响结果，改动时要明确语义。
- overlap/speculative 路径可能复制或预累计 sampling 状态，要求 filter/merge/copy 与 batch 调度保持一致。
