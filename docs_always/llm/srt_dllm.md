# `python/sglang/srt/dllm` 模块分析

## 定位

`dllm` 为 diffusion language model / D-LLM 类推理提供配置、请求和 scheduler mixin。它扩展传统自回归调度，让模型可以按置信度、阈值等策略更新多个 token。

## 关键文件

- `config.py`：`DllmConfig`，D-LLM 运行配置。
- `algorithm/base.py`：算法抽象。
- `algorithm/low_confidence.py`、`algorithm/joint_threshold.py`：基于低置信度或联合阈值的 token 更新策略。
- `mixin/req.py`：请求侧 D-LLM 状态混入。
- `mixin/scheduler.py`：scheduler 侧 D-LLM 行为混入。

## 运行流程

请求启用 D-LLM 后，`ReqDllmMixin` 在请求对象上维护 D-LLM 相关状态。scheduler mixin 根据 `DllmConfig` 和算法输出决定每轮要更新哪些 token、何时结束、如何与普通 batch 调度协同。算法模块根据 logits/置信度判断 mask 或 token 位置。

## 依赖关系

`dllm` 被 `managers.schedule_batch.Req` 和 `managers.scheduler.Scheduler` 混入，依赖 sampling/model output 和 request 状态。它与传统 decode、finish reason、batch filter/merge 有交集。

## 设计要点和风险

- D-LLM 的“一个请求每轮更新多个位置”不同于自回归 next-token，不应复用普通 decode 假设。
- finish 判定、max token、logprob 和 streaming 输出需要与算法状态一致。
- 与 speculative、grammar、prefix cache 的组合需要明确支持边界。
