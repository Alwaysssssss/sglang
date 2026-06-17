# `python/sglang/srt/weight_sync` 模块分析

## 定位

`weight_sync` 提供运行时权重同步的张量 bucket 工具。它用于把多个 tensor 展平、打包、传输和还原，支撑 update weights、remote instance 和分布式权重同步场景。

## 关键文件

- `tensor_bucket.py`：`FlattenedTensorMetadata`、`FlattenedTensorBucket`，记录 tensor 名称/shape/dtype/offset，并提供 flatten/unflatten 能力。
- `utils.py`：`_preprocess_tensor_for_update_weights`，在同步前规范 tensor。

## 运行流程

权重更新路径收集若干 named tensor，必要时先处理 `DeviceMesh` / `DTensor`，再构造 `FlattenedTensorBucket`，把 tensor 数据拼接到连续 buffer 以减少传输次数。rank0 聚合各 TP rank 的本地序列化 tensor 后，通过 engine 的 update weights 请求下发。接收端根据 metadata 切片并恢复原 tensor shape/dtype，再交给模型参数更新逻辑。

## 依赖关系

它被 `model_executor.model_runner`、remote instance loader、update weight API 等路径间接使用，依赖 torch tensor dtype/device 语义。

## 设计要点和风险

- flatten bucket 要保证 offset、dtype、shape 与原 tensor 一致，任何 metadata 错误都会造成权重错写。
- rank0 聚合和大模型权重 bucket 可能成为内存/延迟瓶颈，需要关注峰值内存和跨设备拷贝。
- 各 rank tensor 顺序必须严格一致；DTensor materialize full tensor 时会放大显存/内存压力。
- 同步前预处理必须保留量化/packed tensor 的布局语义。
