# Vivid-VR benchmark 单步 warmup 设计

## 目标

缩短 compile 实验的 warmup 时间：warmup 请求固定使用 1 个推理 step，formal 请求继续使用固定的 20 个推理 step。

## 设计

- 保持 `build_request_payload` 生成正式工作负载的现有行为，不修改其接口。
- `FlowCutRequestExecutor` 已经收到当前请求的 `RunRole`。构造 payload 后，如果角色是 `RunRole.WARMUP`，直接把 `num_inference_steps` 覆盖为 `1`。
- eager 方案仍不执行 warmup；compile 方案仍执行一次 warmup 后再执行 formal。
- 记录到结果 JSON 中的 `request_payload` 保留实际发送的 step 数，因此 warmup 记录为 1，formal 记录为 20。

## 验证

- 单元测试分别执行 warmup 和 formal 请求，断言实际提交 payload 的 `num_inference_steps` 为 1 和 20。
- 重跑 benchmark runner 完整单测文件并执行 `git diff --check`。
