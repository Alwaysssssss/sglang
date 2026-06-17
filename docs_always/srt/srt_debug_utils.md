# `python/sglang/srt/debug_utils` 模块分析

## 定位

`debug_utils` 是 SRT 的调试工具箱，覆盖 tensor dump、dump 加载/比较、文本输出比较、CUDA coredump、模型截断、日志解析、source patch、调度模拟和跨并行维度的 tensor comparator。

## 关键文件与子包

- `dumper.py`：核心 dump 框架，支持 hook/替换函数、HTTP/ZMQ 控制、跨 rank 元数据、过滤表达式和非侵入式 dump。
- `dump_loader.py`、`dump_comparator.py`、`text_comparator.py`：读取 dump、比较 tensor/文本输出。
- `tensor_dump_forward_hook.py`：给模型注册 forward hook dump。
- `model_truncator.py`：裁剪模型 config/index/safetensors，用于小模型复现。
- `log_parser.py`：解析日志。
- `cuda_coredump.py`：CUDA coredump 环境注入、清理和报告。
- `comparator/`：更系统的 tensor comparator，包括维度规格、unshard/reorder/token align、可视化。
- `schedule_simulator/`：离线 scheduler/router 模拟器。
- `source_patcher/`：源码 patch 工具。

## 运行流程

调试时可以通过 dumper 配置或控制接口启用 dump，模型 forward hook 或替换函数会把中间 tensor 和元数据写到目录。之后 `dump_loader` 建立索引，comparator 根据维度注释、并行信息和 token 对齐计划把不同 rank/不同实现的 tensor 还原到可比较形态，并输出数值差异或图。调度模拟器则读取请求日志或合成请求，离线评估路由/调度策略。

## 依赖关系

该模块被 `model_executor.model_runner` 的 dumper/hook 路径使用，也可作为独立 CLI 工具。它依赖 torch、polars、rich/matplotlib 等调试生态，并读取 SRT 的并行元数据和请求日志。

## 设计要点和风险

- dump 热路径会显著增加内存、磁盘和同步开销，不应在线上无界开启。
- 跨 rank comparator 的正确性依赖维度规格和并行元数据；错误 unshard/reorder 会产生误导性 diff。
- HTTP/ZMQ 控制接口用于调试，部署时要注意暴露范围。
- 模型截断会修改 config/权重索引，适合复现，不等于真实模型行为。
