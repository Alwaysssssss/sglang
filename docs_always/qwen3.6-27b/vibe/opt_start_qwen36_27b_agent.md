# Feature: Qwen3.6-27B Agent 启停与显存优化整体方案，服务真实的上线版本

> 本文是整体实施方案，不是实现代码。目标是把 `docs_always/qwen3.6-27b/start_qwen36_27b_agent.sh`、`stop_qwen36_27b_agent.sh`、日志观测、reasoning 展示治理、以及“只常驻模型主体显存/更多模型共存”的目标放到同一张路线图里。

## 0. 需求归纳

当前需求不是单纯“启动一个 256K 服务”，而是同时满足以下目标：

- 新增并完善一组 agent 专用启停脚本：
  - `start_qwen36_27b_agent_online.sh`
  - `stop_qwen36_27b_agent_online.sh`
- 保持 Qwen3.6-27B agent 服务的 `256K` 上下文能力。
- 默认运行并发目标收敛为 `4`，避免空闲 GPU 上自动估算到更高并发。
- 只需要保留最基本的日志
- 降低服务常驻显存：理想目标是“只常驻模型主体显存，其它 KV/request 资源尽量在请求使用时分配”，从而让同一批 GPU 能容纳更多模型或更多服务实例。

