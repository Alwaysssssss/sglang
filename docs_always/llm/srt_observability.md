# `python/sglang/srt/observability` 模块分析

## 定位

`observability` 提供 SRT 的 metrics、trace、请求耗时分段、启动耗时、函数计时、CPU monitor 和请求 metrics 导出。它贯穿 API server、scheduler、DP controller 和 storage/cache 等路径。

## 关键文件

- `metrics_collector.py`：Scheduler/Tokenizer/Storage/RadixCache/ExpertDispatch 等 metrics collector。
- `scheduler_metrics_mixin.py`：把 scheduler stats、prefill stats、KV metrics 接入 scheduler。
- `req_time_stats.py`：请求阶段时间戳、跨线程时间转换、API/DP/Scheduler stage stats。
- `trace.py`：OpenTelemetry tracing、trace context、span attributes、thread info。
- `request_metrics_exporter.py`：文件导出请求 metrics。
- `startup_func_log_and_timer.py`、`func_timer.py`：启动/函数耗时。
- `cpu_monitor.py`：后台 CPU 使用监控。
- `utils.py`、`label_transform.py`：histogram bucket 和 label 转换。

## 运行流程

服务启动时根据 `ServerArgs.enable_metrics/enable_trace/...` 和相关环境变量初始化 Prometheus multiprocess、metrics collector、bucket 配置或 OTLP exporter。请求经过 API、tokenizer、scheduler、detokenizer 时记录 stage 时间。scheduler/tokenizer 在关键阶段更新 queue、token、cache、EPLB、prefill/decode 等指标；需要时 request metrics exporter 写文件。trace 开启时，各线程设置 trace thread info，并把请求 headers 中的 trace context 传入后续 span。

## 依赖关系

该模块被 `entrypoints`、`managers.scheduler`、`data_parallel_controller`、`mem_cache`、`eplb`、`utils.profile_utils` 使用。它依赖 Prometheus/OpenTelemetry 和 SRT 的请求结构。

## 设计要点和风险

- metrics label 需要控制基数，尤其是自定义 header label、routing key 和 model name。
- 请求时间跨线程/跨进程时需要校准，不应混用 realtime 和 monotonic。
- tracing 不能在热路径引入大量同步或字符串构造。
- Prometheus multiprocess 目录和子进程初始化顺序/生命周期要一致，否则会残留旧指标或缺指标。
- 文件导出属于 I/O 路径，采样或限流策略需要结合线上吞吐评估。
