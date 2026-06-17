# `python/sglang/srt/ray` 模块分析

## 定位

`ray` 为 SRT 提供 Ray 部署模式，把 scheduler/engine 进程放入 Ray actor 和 placement group，以支持多节点资源调度。

## 关键文件

- `engine.py`：`RayEngine` 继承 `Engine`，返回 `RaySchedulerInitResult`，处理 placement group 和 rank/bundle 查找。
- `scheduler_actor.py`：`SchedulerActor`，Ray actor 形式运行 scheduler。
- `http_server.py`：Ray 模式下的 HTTP server 启动封装。
- `__init__.py`：包入口。

## 运行流程

当 `ServerArgs.use_ray` 开启时，入口层要求 placement group，`RayEngine` 查找与 Engine 同节点的 bundle，并为每个 TP/PP rank 启动 `SchedulerActor`。actor 绑定对应资源 bundle，内部仍运行 SRT scheduler/worker event loop。`get_info` 握手完成后，HTTP/tokenizer/detokenizer 主流程基本复用普通 Engine。

## 依赖关系

该模块依赖 Ray、`entrypoints.engine`、`managers.scheduler` 和分布式 rank/port 配置。它与多节点 `ServerArgs`、`PortArgs` 和 placement group 资源声明绑定。

## 设计要点和风险

- Ray placement group 的 bundle/rank 映射必须和 SRT TP/PP/DP rank 一致。
- 当前 Ray 路径对 `dp_size > 1` 支持有限或未实现，DP 场景不能直接按本地模式推断。
- rank0/Engine 同节点假设、Ray runtime GPU 分配和 NCCL 初始化地址都比较敏感。
- Ray actor 异常传播和本地 subprocess watchdog 语义不同，需要额外关注失败恢复。
- 端口和 NCCL/rendezvous 地址在 Ray 多节点环境中更容易冲突或不可达。
