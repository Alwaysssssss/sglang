# `python/sglang/srt/constrained` 模块分析

## 定位

`constrained` 实现结构化/受限解码。它把请求中的 JSON schema、regex、EBNF、structural tag 或 reasoning grammar 编译成 token mask，并在采样前修改 logits，使模型输出满足约束。

## 关键文件

- `base_grammar_backend.py`：`BaseGrammarObject`、`BaseGrammarBackend` 抽象。
- `grammar_manager.py`：scheduler 侧 grammar 编译、缓存、轮询和 ready 管理。
- `xgrammar_backend.py`：默认/主力 xgrammar 后端。
- `llguidance_backend.py`：llguidance 后端。
- `outlines_backend.py`、`outlines_jump_forward.py`：Outlines 后端和 jump-forward 支持。
- `reasoner_grammar_backend.py`：reasoning parser 相关 grammar 后端。
- `triton_ops/bitmask_ops.py`：logits bitmask 应用的 Triton kernel。
- `utils.py`：约束解码工具函数。

## 运行流程

入口请求把 `json_schema`、`regex`、`ebnf` 或 structural tag 写入 `SamplingParams`。Scheduler 调 `GrammarManager.process_req_with_grammar` 后，grammar backend 异步编译并缓存 grammar。请求 ready 后进入正常调度。采样前 `SamplingBatchInfo.update_regex_vocab_mask()` 让 grammar 对象填充 vocab mask，再由 backend 的 `apply_vocab_mask` 修改 logits。speculative decoding 场景会对 draft tree 生成逐节点 bitmask，保证候选 token 也符合 grammar 状态。

## 依赖关系

该模块被 `managers.scheduler`、`sampling`、`speculative` 使用，依赖 tokenizer、vocab size、`ServerArgs.grammar_backend`、Triton/sgl_kernel 和可选第三方库 xgrammar/llguidance/outlines。

## 设计要点和风险

- `server_args` 中默认优先 xgrammar，不支持 tokenizer 或依赖缺失时会降级/禁用。
- 多种约束同时传入时存在优先级路径；调用方应尽量保证互斥。
- grammar 编译异步执行，timeout/cancel 不一定能立即终止底层编译线程。
- speculative + grammar 会在 draft tree 上做额外 DFS/mask 生成，长树和大 vocab 下成本明显。
