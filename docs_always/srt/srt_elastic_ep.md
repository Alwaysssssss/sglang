# `python/sglang/srt/elastic_ep` 模块分析

## 定位

`elastic_ep` 为 expert parallel 提供弹性状态和 expert backup 机制。它服务于 MoE expert 迁移、备份、恢复和动态 EP 状态管理。

## 关键文件

- `elastic_ep.py`：`ElasticEPState`、`ElasticEPStateManager`，维护弹性 EP 状态。
- `expert_backup_manager.py`：`ExpertBackupManager`，解析 expert 参数名并管理 expert backup 服务进程。
- `expert_backup_client.py`：`ExpertBackupClient`，向 backup manager 请求/发送 expert 权重。

## 运行流程

`ModelRunner` 初始化或 EPLB 相关路径创建 `ElasticEPStateManager`，维护 active rank 状态。当需要备份或恢复 expert 权重时，backup manager 从磁盘加载本节点 expert 权重到 CPU 连续 buffer，并可注册 Mooncake 内存；client 根据当前 expert location 和参数名解析 layer/expert id，从指针表拉回本地 GPU 参数。备份服务可作为独立进程由 `run_expert_backup_manager` 启动。

## 依赖关系

它与 `eplb`、`layers.moe`、`model_executor.model_runner`、`distributed` 和权重命名规则耦合。参数名解析函数必须匹配模型 MoE 权重命名。

## 设计要点和风险

- 参数名解析是脆弱点，新模型/新 MoE layer 命名不匹配会导致 backup 找不到 expert。
- backup 状态要和 EPLB 的 expert location 版本同步，否则可能恢复到旧位置或旧权重。
- manager/client 跨进程通信错误需要清晰传播，不能只在某个 rank 静默失败。
- 当前状态管理偏薄，故障检测不主要在本模块；端口按 node rank 推导时要避免部署冲突。
