# SRT 源码分析持续迭代路线图

本文定义 `docs_always/srt` 的持续维护方式。目标是把当前目录从“模块速览”演进为完整的 SRT 技术参考，覆盖架构、核心流程、模块细节、扩展点、风险和 troubleshooting。

## 文档分层

| 层级 | 文件 | 用途 |
| --- | --- | --- |
| 总览 | `README.md`、`00_srt_architecture.md` | 导航、全局架构、阅读路径 |
| 深度专题 | `01_request_lifecycle.md`、`02_scheduler_batching.md`、`03_model_execution.md`、`04_cache_and_memory.md` | 横跨多模块的核心流程 |
| 模块文档 | `srt_*.md` | 每个顶层模块的职责、关键类、扩展点 |
| 专项指南 | 后续新增 | 新模型、新 backend、新 cache、新 API 的开发指南 |

## 统一模板

后续扩写每篇模块文档时建议使用以下结构：

```markdown
# `python/sglang/srt/<module>` 模块分析

## 定位
说明模块解决什么问题，位于哪一层。

## 架构图
用 Mermaid 表示主要对象和调用关系。

## 关键文件与类
列出文件、类、函数、职责。

## 核心流程
按请求/初始化/执行/释放等真实路径解释。

## 关键数据结构
说明字段含义、不变量、跨模块契约。

## 扩展点
说明如何新增模型、backend、配置、协议或能力。

## 风险与排查
列出常见坑、日志/metrics、测试建议。
```

## 已完成

- 建立 `docs_always/srt` 模块索引。
- 完成首版 `00_srt_architecture.md`。
- 完成首批深度专题：
  - `01_request_lifecycle.md`
  - `02_scheduler_batching.md`
  - `03_model_execution.md`
  - `04_cache_and_memory.md`
  - `05_configuration_and_extensions.md`
- 完成第一批核心模块深挖：
  - `srt_layers.md`
  - `srt_lora.md`
  - `srt_disaggregation.md`
  - `srt_observability.md`
  - `srt_multimodal.md`
- 完成顶层子目录逐项 subagent 分析与复写覆盖：
  - `batch_invariant_ops`、`batch_overlap`、`checkpoint_engine`、`compilation`
  - `configs`、`connector`、`constrained`、`debug_utils`
  - `disaggregation`、`distributed`、`dllm`、`elastic_ep`
  - `entrypoints`、`eplb`、`function_call`、`grpc`
  - `hardware_backend`、`layers`、`lora`、`managers`
  - `mem_cache`、`model_executor`、`model_loader`、`models`
  - `multimodal`、`multiplex`、`observability`、`parser`
  - `ray`、`sampling`、`speculative`、`tokenizer`
  - `utils`、`weight_sync`

## 下一阶段优先级

1. 编写专项开发指南：新增模型、新 attention backend、新 LoRA backend、新 multimodal processor、新 transfer backend、新 speculative 算法。
2. 建立跨模块调用链专题：在线权重更新、disaggregation KV transfer、grammar constrained decoding、speculative decoding、LoRA serving。
3. 将 `srt_model_loader.md` 与 `srt_models.md` 抽象为“新模型接入手册”，补充权重命名、config 修正、quant/load format、测试矩阵。
4. 将 `srt_server_args.md`、`srt_configs.md`、`srt_environ.md` 联动成配置参考，覆盖默认值推导、参数冲突、环境变量迁移策略。
5. 为高风险路径补充 troubleshooting：KV cache 错位、weight update 部分失败、multi-node rendezvous、Transformers 版本回归、CUDA graph capture 失败。
6. 建立变更同步机制：每次修改 `python/sglang/srt/<module>` 后检查对应 `docs_always/srt/srt_<module>.md` 是否需要更新。

## 分析方法

- 每轮选择 1 到 3 个独立模块或 1 条跨模块链路，优先由 subagent 并行做只读源码分析。
- 主线程统一整合结论，保证术语、架构图和跨文档链接一致。
- 每完成一个模块立即写入或更新对应 Markdown。
- 所有事实以当前 `python/sglang/srt` 代码为准，避免只复述历史认知。
- 对高风险路径标注“不变量”和“修改建议”，便于后续开发。

## 覆盖检查清单

- 请求入口：Python API、HTTP、OpenAI、Ollama、Anthropic、RPC。
- 管理链路：TokenizerManager、Scheduler、DetokenizerManager、DP controller。
- 执行链路：TpModelWorker、ModelRunner、ForwardBatch、CUDA graph、sampler。
- 状态管理：KV cache、prefix cache、HiCache、session、LoRA、grammar、speculative。
- 配置系统：ServerArgs、ModelConfig、LoadConfig、环境变量、CLI parser。
- 扩展生态：models registry、model loader、attention backend、quantization、disaggregation backend。
- 可观测性：metrics、trace、request stats、profile、日志、debug utils。
- 维护实践：测试矩阵、常见故障、性能排查、相容性风险。
