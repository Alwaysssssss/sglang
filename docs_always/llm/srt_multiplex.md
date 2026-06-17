# `python/sglang/srt/multiplex` 模块分析

## 定位

`multiplex` 支持 prefill/decode multiplex 的调度和 CUDA stream/SM 资源划分。它允许在同一 GPU 上用不同 stream group 和 SM 配额并行推进不同类型工作。

## 关键文件

- `pdmux_context.py`：`PDMuxConfig`、配置加载、SM 划分、stream group 初始化、当前 stream index 管理。
- `multiplexing_mixin.py`：`SchedulerMultiplexMixin`，把 multiplex 行为接入 scheduler。

## 运行流程

服务启动后根据配置加载 PDMux 分组，结合 GPU compute capability 和总 SM 数划分每组资源，并创建对应 CUDA streams。scheduler 在不同任务之间切换当前 stream index，使 prefill/decode 或多路工作在指定 stream/SM 配额下运行。

## 依赖关系

该模块依赖 CUDA stream、GPU 架构信息和 scheduler。它与 `batch_overlap`、`model_executor`、attention backend 等性能路径存在隐式交互。

## 设计要点和风险

- SM 划分不合理会降低总吞吐，配置必须结合硬件和 workload 测。
- stream group 是全局状态，嵌套/并发切换需要谨慎。
- 与 CUDA graph、batch overlap、通信 stream 同时使用时，事件同步顺序是主要风险。
