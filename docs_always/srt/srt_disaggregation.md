# `python/sglang/srt/disaggregation` 模块分析

## 定位

`disaggregation` 实现 SRT 的分离式推理：prefill/decode 分离、encoder 分离、多模态 embedding 服务，以及跨实例 KV cache 传输。它把 scheduler、memory pool、transfer backend 和请求元数据连接起来。

## 关键文件与子包

- `utils.py`：`DisaggregationMode`、`TransferBackend`、metadata buffer、KV page/index 工具、abort helper。
- `prefill.py`：`PrefillBootstrapQueue`、`SchedulerDisaggregationPrefillMixin`，prefill 侧 bootstrap 和请求释放。
- `decode.py`：`DecodeReqToTokenPool`、`DecodeRequest`、`DecodePreallocQueue`、`DecodeTransferQueue`、`SchedulerDisaggregationDecodeMixin`。
- `decode_schedule_batch_mixin.py`：decode 模式下 `ScheduleBatch` 的附加行为。
- `decode_kvcache_offload_manager.py`：decode KV cache offload 管理。
- `encode_server.py`、`encode_receiver.py`、`encode_grpc_server.py`：多模态 encoder 服务端和 scheduler/tokenizer 侧接收器。
- `base/conn.py`：KV manager/sender/receiver/bootstrap server 抽象。
- `common/`：通用 KV transfer、staging buffer/handler。
- `mooncake/`、`nixl/`、`mori/`、`ascend/`、`fake/`：具体 transfer backend。
- `kv_events.py`：KV block store/remove/clear 事件发布。

## 运行流程

prefill 节点接收请求并计算 prompt KV，随后通过 bootstrap/transfer backend 把 KV 元数据、首 token 相关状态和页数据传给 decode 节点。decode 节点预分配 KV/metadata buffer，创建 receiver，poll transfer 成功后构造 prebuilt batch 接管请求继续 decode。多模态 encoder 分离时，encoder server 处理图像/音频/视频 embedding，再通过 HTTP/gRPC 或共享内存路径传回主推理服务。

## 依赖关系

`disaggregation` 与 `managers.scheduler`、`schedule_batch`、`mem_cache`、`distributed`、`model_executor`、`multimodal`、`utils.network` 深度耦合。具体 backend 依赖 Mooncake、NIXL、Mori、Ascend 等外部传输库。

## 设计要点和风险

- KV page/index 是跨实例协议，page size、TP/CP rank、MLA/MHA cache layout 必须一致。
- staging buffer 解决跨 head/rank 的布局和水位问题，但引入额外异步状态。
- abort/timeout 必须同时清理 scheduler 请求、tree cache、metadata buffer、prealloc queue 和远端 transfer 状态。
- TP/CP/DP/PP rank 映射和 bootstrap room/metadata buffer 隔离是高风险区域；串扰会导致 decode 端接错 KV。
- fake backend 适合测试，不应误用于性能或真实跨节点传输判断。
