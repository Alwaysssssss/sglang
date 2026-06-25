# Qwen3.6-27B Agent Online 启停与显存优化整体方案

> 本文是上线方案文档，不是实现代码。它基于当前 `docs_always/qwen3.6-27b/start_qwen36_27b_agent.sh` 和 `stop_qwen36_27b_agent.sh`，把 256K agent 服务、默认并发 4、基础日志、reasoning 展示治理、ChatOpenAI 调用、以及“降低常驻显存/更多模型共存”的目标整理到同一张路线图里。

## 0. 结论

- `start_qwen36_27b_agent_online.sh` 负责真实 SGLang 参数拼装、显存估算、ready check、日志脱敏；`stop_qwen36_27b_agent_online.sh` 负责按 PID/端口安全停止。
- 需求里的 `start_qwen36_27b_agent_online.sh` 和 `stop_qwen36_27b_agent_online.sh` 不应该复制一份完整启动逻辑，建议做成 thin wrapper：只固化 online 默认环境变量，然后 `exec` 调用现有脚本。即使当前工作树已有同名文件，也必须按这个原则校正内容，不能把 128K 普通服务脚本复制后改名当作 agent online 交付。
- Online 默认画像应保留 `CONTEXT_LENGTH=262144`，显式设置 `MAX_RUNNING_REQUESTS=4`，并关闭 request body 级日志和每请求 metrics 文件导出。
- “只常驻模型主体显存”不能仅靠当前脚本完全实现。SGLang 的 `mem_fraction_static` 定义包含 `model weights + KV cache pool`，因此当前可落地目标是降低固定 KV pool 预算、限制并发和输出上限、关闭不必要的观测面；真正按请求动态分配 KV/request 资源需要后续改 SGLang allocator 或引入可验证的新运行模式。

## 1. 目标和非目标

### 目标

- 提供一套真实线上可用的 Qwen3.6-27B agent 启停方案。
- 服务保持 256K 上下文：`--context-length 262144`。
- 默认运行并发收敛为 4：启动时直接传 `--max-running-requests 4`，而不是只依赖自动估算 cap。
- 日志收敛到基础运行日志：启动摘要、SGLang stdout/stderr、PID、ready/health；默认不记录完整请求体和完整响应体。
- 显存策略优先支持同机多模型共存：降低静态显存目标、限制并发、避免显式放大 `--max-total-tokens`。
- 给 LangChain `ChatOpenAI` 调用方明确 base URL、模型名、超时、流式、输出 token 和 reasoning 字段处理方式。

### 非目标

- 不修改模型权重、不转换量化权重。
- 不把 256K agent 服务直接暴露到公网；对外入口仍应由 Nginx/API gateway/TLS/API key 策略控制。
- 不承诺完全按需分配 KV cache；当前只能通过 SGLang 参数降低静态预算。
