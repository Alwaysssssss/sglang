# `python/sglang/srt/distributed` 模块分析

## 定位

`distributed` 接管 SRT 的 PyTorch distributed 状态，建立 world、tensor parallel、pipeline parallel、attention TP/CP、MoE DP/EP/TP 等进程组，并封装 all-reduce、all-gather、reduce-scatter、broadcast、P2P、共享内存广播和自定义 communicator。

## 关键文件

- `parallel_state.py`：核心状态模块。定义 `GroupCoordinator`、全局 group registry、`init_distributed_environment`、`initialize_model_parallel`、`get_tp_group/get_pp_group/...`、graph capture 通信 custom op 等。
- `communication_op.py`：tensor/attention/MoE parallel 通信函数。
- `utils.py`：tensor split、PP indices、全局 TCP store、stateless process group 等辅助。
- `device_communicators/`：PyNccl、custom allreduce、QuickAllReduce、Mscclpp、torch symmetric memory、Mooncake transfer engine、NPU/HPU/XPU communicator、共享内存广播等。
- `naive_distributed.py`：简化分布式实现/状态。

## 运行流程

`ModelRunner` 初始化时调用分布式环境初始化，再基于 `ServerArgs` 的 TP/PP/DP/EP/CP 参数创建各类进程组。模型层和 executor 通过 `get_tp_group()` 等 helper 获取 `GroupCoordinator`。通信 op 既可以走 torch.distributed，也可根据设备、tensor size、graph capture 和配置路由到 PyNccl、custom allreduce、Mscclpp 或 symmetric memory 实现。

## 依赖关系

`distributed` 被 `model_executor`、`layers`、`managers.scheduler`、`disaggregation`、`eplb`、`elastic_ep` 使用。它依赖 `environ`、`utils`、平台检测和可选通信库。`parallel_state` 还注册了 custom op，以便在 compile/cudagraph 路径中保留通信调用。

## 设计要点和风险

- `GroupCoordinator` 是通信语义中心，既持有 CPU/device process group，也持有可选 communicator；销毁顺序和全局 weakref registry 要一致。
- graph capture 中的通信必须用可捕获/可 replay 的路径，普通 torch.distributed 调用可能不安全。
- TP/PP/DP/EP/CP group 组合复杂，rank mapping 错误会表现为 hang 而不是明确异常。
- 非 CUDA 后端有专门 options/communicator，新增平台要确认 backend、设备流、P2P、共享内存和 timeout 语义。
