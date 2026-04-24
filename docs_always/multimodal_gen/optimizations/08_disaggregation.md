# 08. 分离式部署（Disaggregation / PD 分离）

> 源码位置：`python/sglang/multimodal_gen/runtime/disaggregation/`（`roles.py`、`orchestrator.py`、`dispatch_policy.py`、`request_state.py`、`scheduler_mixin.py`、`metrics.py`、`disagg_args.py`）、`runtime/disaggregation/transport/`（`allocator.py`、`buffer.py`、`codec.py`、`engine.py`、`manager.py`、`protocol.py`）、`runtime/pipelines_core/composed_pipeline_base.py`。

## 1. 为什么要做分离式

扩散模型的三个主要阶段资源特性差异极大：

- **Encoder**（T5 / CLIP / VAE Encoder / Image Processor）：计算量小，**对内存带宽敏感**，**加载期长**（大 T5 模型）；
- **Denoiser**（DiT transformer 循环）：**计算密集**，常用 TP / SP / CFG / Ring；
- **Decoder**（VAE Decoder）：**显存峰值高**（高分辨率视频），延迟敏感；

把它们拆到不同进程 / 不同机器，可以：

- **按角色独立扩缩容**（denoiser 副本 × 4，vae × 1 即可）；
- **让 encoder 常驻 CPU / 小 GPU**，把显存省给 DiT；
- **异构卡搭配**（H100 跑 DiT，4090 跑 VAE）。

## 2. 四角色定义

`runtime/disaggregation/roles.py`：

```python
class RoleType(Enum):
    MONOLITHIC = ...   # 单体，默认
    ENCODER = ...      # text_encoder / image_encoder / tokenizer / processor
    DENOISER = ...     # transformer
    DECODER = ...      # vae / vocoder
    SERVER = ...       # 无 GPU 的 head node，做路由和调度
```

### 2.1 模块归属：`get_module_role`

`roles.py` 约 31–57 行：

- Encoder 侧：`text_encoder*`、`tokenizer`、`image_encoder*`、`image_processor`、`processor`、`connectors`
- Denoiser：`transformer*`
- Decoder：`vae*`、`vocoder`

### 2.2 角色过滤：`filter_modules_for_role`

`roles.py` 约 61–78 行：

- 非单体模式只保留 **本角色模块 + `module_role is None`（共享）**；
- **特例**：Encoder 角色**也会保留 Decoder 类模块**（注释提到 ImageVAEEncoding 等 stage 需要 VAE）。

### 2.3 在 `ComposedPipelineBase` 中的落地

`runtime/pipelines_core/composed_pipeline_base.py`：

- `__init__` 约 102–114 行对 `_required_config_modules` 调用 `filter_modules_for_role`；
- 对跳过的模块用 `_init_skipped_component_configs` 只读 HF 配置来补全 `pipeline_config`（约 340–343、198–253 行）——**Encoder 进程也能知道 DiT / VAE 的 shape，但不加载权重**；
- `add_stage` 约 449–460 行：若 `stage.role_affinity != self._disagg_role` 则**跳过该 stage**；
- MoE 双塔场景：动态加入 `transformer_2` 时再次按 disagg role 判断是否追加（约 300–317 行）。

## 3. 请求状态机

文件 `runtime/disaggregation/request_state.py`。

### 3.1 状态枚举

约 13–31 行：

```
PENDING
  → ENCODER_WAITING → ENCODER_RUNNING → ENCODER_DONE
  → DENOISING_WAITING → DENOISING_RUNNING → DENOISING_DONE
  → DECODER_WAITING → DECODER_RUNNING → DONE
  ▼
  FAILED / TIMED_OUT（可从任意 active 状态转入）
```

合法转移矩阵：`_VALID_TRANSITIONS`（约 39–55 行）；
异常分支：`FAILED` / `TIMED_OUT` 可从任意 active 状态进入（约 108–114 行）。

### 3.2 `RequestTracker`

约 76–165 行，接口：`submit` / `transition` / `find_timed_out` / `snapshot`。**所有状态转移经过 `transition`，非法转移会被记录与报错**。

### 3.3 与编排器联动

`DiffusionServer`（`orchestrator.py`）：

- 接收客户端请求 `submit` + 入队 `encoder_tta`；
- 各阶段完成时 `transition` 并维护每实例 free slot；
- 通过 TTA（time-to-arrival）队列对下一阶段派发。

## 4. DiffusionServer 编排器

文件 `runtime/disaggregation/orchestrator.py`。角色：**SERVER（head node，无 GPU）**。

职责：

1. 通过 **ZMQ ROUTER** 接受前端请求；
2. 对 encoder / denoiser / decoder 各自维护实例池（数量由 `disagg_args` 指定）；
3. 通过 **ZMQ PUSH** 向各角色派发工作、**PULL** 收结果；
4. 跟踪 request state；
5. 管理每角色的 TTA 队列（`deque`）；
6. 管理跨角色的 tensor transfer（见 §6）；
7. 超时检测、metrics 聚合。

`get_stats`（约 1054–1076 行）：返回 slot / 队列深度 / peer 状态。

HTTP 端口 settle：`server_args._adjust_network_ports` 仅 `MONOLITHIC` 与 `SERVER` 需要（约 530–533 行）——encoder / denoiser / decoder 不面向外部。

## 5. 调度策略

文件 `runtime/disaggregation/dispatch_policy.py`。

| 策略 CLI 名 | 类 | 规则 |
|-------------|-----|------|
| `round_robin` | `RoundRobin.select_with_capacity`（约 46–53 行） | 有 free slot 的实例间轮转 |
| `max_free_slots` | `MaxFreeSlotsFirst.select_with_capacity`（约 96–112 行） | 选 `free_slots` 最大的实例（capacity-aware） |

`PoolDispatcher`（约 115–152 行）：encoder / denoiser / decoder **三套独立策略**。`disagg_args.py` 约 129–133 行限定 `choices=["round_robin", "max_free_slots"]`。

编排器侧 `DiffusionServer` 还使用 **每角色 TTA 队列 + 每实例 `_*_free_slots`**（约 477–514 行）：只有 `select_*_with_capacity` 返回非空索引时才出队派发 —— 是**队列 + 容量**的组合，而非单独命名的 "queue-aware" 策略。

## 6. Transport 层

### 6.1 控制面：ZMQ

`runtime/disaggregation/scheduler_mixin.py` 约 201–229 行：前端 ROUTER + 各角色 PUSH/PULL。

协议帧定义：`protocol.py`

- `TRANSFER_MAGIC`（约 15–16 行）+ JSON（约 122–131 行）；
- `TransferMsgType`（约 18–31 行）：`transfer_staged` / `transfer_allocated` / `transfer_pushed` / `transfer_done` / `transfer_alloc` / `transfer_push` / `transfer_ready` / `transfer_register` 等；
- 关键消息类（约 51–95 行）：
  - `TransferAllocMsg`：slow path，请求接收端分配 buffer；
  - `TransferPushMsg`：fast path，发送端直接推；
  - `TransferReadyMsg`：DS 通知接收端数据到达可消费。

Orchestrator 的处理：`_send_slow_path_alloc`、`_handle_transfer_allocated`、`_handle_transfer_pushed`、发送 `TransferReadyMsg`（约 846–898 行）。

### 6.2 数据面：Mooncake RDMA

`transport/engine.py::create_transfer_engine` 约 118–126 行：**mooncake 不可用时直接抛错**。

- `MooncakeDiffusionEngine` 包装 `MooncakeTransferEngine`（约 59–64 行），`supports_gpu_direct=True`；
- **不使用** NCCL / HTTP / gRPC 传大 tensor payload——全部走 RDMA（with GPU Direct）。

### 6.3 Buffer 分配

`transport/buffer.py::TransferTensorBuffer`（约 49–52 行）：

- `device=="cpu"` 时用 **`pin_memory=True`** 的 uint8 pool；
- 在 scheduler 里若检测到 `engine.supports_gpu_direct`，则直接用 **`cuda:{physical_gpu_id}`** 池（`scheduler_mixin.py` 约 385–389 行）—— **不需要先经 CPU 落地**。

### 6.4 Buffer 注册

`transport/manager.py::DiffusionTransferManager.register_buffer(pool_data_ptr, pool_size)`（约 50 行）：注册给 RDMA；`stage_tensors` 约 73–116 行把 GPU tensor 写入 pool 并同步。

### 6.5 完整流程（以 Encoder → Denoiser 为例）

```text
Encoder worker:                Orchestrator (SERVER):        Denoiser worker:
   finish encode               │                             │
   ──▶ TransferStagedMsg ─────▶│                             │
                               │ select_denoiser             │
                               │ (fast path?)                │
                               │──▶ TransferPushMsg (fast) ─────▶ rdma write
                               │   or                        │
                               │──▶ TransferAllocMsg (slow) ────▶ alloc buffer
                               │◀─ TransferAllocatedMsg ────│
                               │──▶ TransferPushMsg ────────────▶ rdma write
                               │◀─ TransferPushedMsg ───────│
                               │──▶ TransferReadyMsg ─────────────▶ start denoise
```

Denoiser 完成后再走同样流程把 latent 推给 Decoder（约 928–971 行）。

## 7. Metrics

`disaggregation/metrics.py::DisaggMetrics`（约 45–128 行）：每角色 completed / failed / timeout、in_flight、queue_depth、latency、RPS；与 orchestrator 的 `get_stats`（偏 DS 全局 slot / 队列 / peer）互补。

## 8. `disagg_args.py` / CLI

`runtime/disaggregation/disagg_args.py`：

- `role`：`monolithic | encoder | denoiser | decoder | server`；
- `disagg_scheduler_policy`：`round_robin | max_free_slots`；
- `encoder_pool_size`、`denoiser_pool_size`、`decoder_pool_size`；
- 各角色绑定的 ZMQ 地址、Mooncake endpoint；
- 请求超时、buffer 池大小。

Server 端还会解析 peer 地址列表，连接其它角色进程。

## 9. `SERVER` vs `MONOLITHIC` 区别

| 项 | `MONOLITHIC` | `SERVER` |
|----|--------------|----------|
| HTTP 入口 | 是（OpenAI 风格）| 是 |
| GPU 占用 | 有 | **无** |
| 包含 encoder / denoiser / decoder | 全部 | 都不含（通过 peer 连接其它角色进程）|
| 适合场景 | 单机单节点 | 多机 / 异构卡部署 |

`server_args._adjust_network_ports`（约 530–533 行）：**只有 `MONOLITHIC` 和 `SERVER` 需要 settle HTTP 端口**。

## 10. 与其它优化的关系

- **并行**：每个 worker 进程内部仍可自己再走 TP / SP / USP / CFG 并行（见 [`01_parallelism.md`](./01_parallelism.md)）。注意 encoder 的 TP 常配 `_get_folding_tp_group` 选择不同 PG。
- **量化**：每个角色独立决定量化（Encoder 可能 BF16，DiT 上 FP8/NVFP4）。
- **Offload**：角色拆分后显存压力分散，可关闭 layerwise offload 提高延迟。
- **缓存**：cache-dit / TeaCache 都绑定在 DiT，随 denoiser role 部署，不需要跨角色同步。
- **Post-processing**：RIFE / Real-ESRGAN 运行在 decoder 或 server 侧，由 `save_outputs` 控制。

## 11. 关键文件索引

| 主题 | 路径 |
|------|------|
| 角色与模块过滤 | `runtime/disaggregation/roles.py` |
| 编排 + transfer 状态机 | `runtime/disaggregation/orchestrator.py` |
| 调度策略 | `runtime/disaggregation/dispatch_policy.py` |
| 请求状态机 | `runtime/disaggregation/request_state.py` |
| Scheduler 侧 transfer / ZMQ | `runtime/disaggregation/scheduler_mixin.py` |
| 协议 | `runtime/disaggregation/transport/protocol.py` |
| RDMA engine | `runtime/disaggregation/transport/engine.py` |
| Buffer / Allocator | `runtime/disaggregation/transport/{buffer,allocator,manager}.py` |
| CLI / Args | `runtime/disaggregation/disagg_args.py` |
| Metrics | `runtime/disaggregation/metrics.py` |
| Pipeline role / stage 过滤 | `runtime/pipelines_core/composed_pipeline_base.py` |

## 12. 调优 & 部署建议

- **首次搭建**先单机 `MONOLITHIC` 跑通；
- 视频大模型（Wan / Hunyuan）：**denoiser 多副本、encoder 单副本** 即可；
- **DiT 与 VAE 异构**：Decoder 单独 4090 + GPU Direct RDMA，DiT 侧 H100；
- **超时要单独监控**：RDMA 慢 path 可能把 DS 的队列拉高；
- **不要跨网段拉 RDMA**：Mooncake 需要高带宽低延迟网络；
- **扩缩容操作**：调整 `*_pool_size` 并重启对应角色；server 会自动重建 peer；
- **failure recovery**：默认没有自动 retry，失败请求立即回 `FAILED`；上层 router 需要做 retry policy。

---

> 若要继续深入 Mooncake RDMA 的内部实现，需要查看 mooncake 包（第三方）源码；本篇只覆盖 `multimodal_gen` 内部的控制面 + buffer 抽象。
