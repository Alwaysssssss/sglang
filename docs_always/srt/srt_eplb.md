# `python/sglang/srt/eplb` 模块分析

## 定位

`eplb` 是 Expert Parallel Load Balancing，负责 MoE expert 的负载统计、专家位置计算、逻辑/物理 expert 映射和运行时权重迁移，降低 expert parallel 下的热点和跨节点通信成本。

## 关键文件

- `eplb_manager.py`：`EPLBManager`，周期性触发负载重平衡。
- `expert_distribution.py`：`ExpertDistributionRecorder`、metrics、gatherer、accumulator，记录各 expert/token/layer 的使用分布。
- `expert_location.py`：`ExpertLocationMetadata`、初始 expert 位置计算、逻辑到物理映射。
- `expert_location_dispatch.py`：把 topk logical expert id 转成 physical expert/rank dispatch 信息。
- `expert_location_updater.py`：`ExpertLocationUpdater` 和 expert 权重迁移逻辑。
- `eplb_algorithms/`：DeepSeek 等重平衡算法实现。
- `eplb_simulator/`：离线模拟/读取工具。

## 运行流程

模型执行时 MoE 层记录 expert 命中分布。EPLB manager 汇总一段窗口内的负载指标，可 dump 统计；算法根据 token 分布、节点/GPU 拓扑、冗余 expert 等信息生成新的 `physical_to_logical_map`。`ExpertLocationUpdater` 再按层/专家、分 chunk 迁移权重，并更新全局 expert location metadata。dispatch 阶段用映射把 logical topk 转成实际物理 expert。

## 依赖关系

`eplb` 被 `model_executor.model_runner`、`layers.moe`、`elastic_ep` 和 `distributed` 使用。它依赖 torch distributed/P2P、MoE runner、expert weight 命名、`ServerArgs` 的 EP/MoE 参数。

## 设计要点和风险

- expert placement 是全局共享状态，所有 rank 必须在同一版本上执行 dispatch。
- 权重迁移涉及 P2P/copy/canary buffer，失败可能造成部分 expert 权重不一致。
- 负载窗口过短会抖动，过长会响应慢；算法参数直接影响吞吐。
- 逻辑/物理 expert 映射要和 MoE topk、EP rank、冗余 expert 数量保持一致。
- recorder buffer、迭代周期和动态 dispatch 的随机选择会影响可复现性；线上调参要保留足够观测。
