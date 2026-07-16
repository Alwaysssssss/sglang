# Vivid-VR benchmark 单步 warmup 设计

## 目标

缩短加速组合完整测试的准备时间：所有启用 `torch.compile` 的方案仍执行一次 warmup，但 warmup 请求固定只运行 1 个推理 step。

## 行为边界

- `RunRole.WARMUP` 请求的 `num_inference_steps` 固定为 `1`。
- `RunRole.FORMAL` 请求继续使用固定正式口径 `20` step。
- eager 方案继续只执行 formal 请求，不新增 warmup。
- warmup 与 formal 的视频、性能 JSON、请求记录和失败处理路径保持不变。

## 实现与测试

`build_request_payload` 显式接收 `RunRole`，由请求角色决定 step，避免根据 task ID 猜测或在 payload 创建后隐式覆盖。单元测试分别覆盖 warmup=1 与 formal=20，并运行 benchmark runner 的完整单测文件回归。
