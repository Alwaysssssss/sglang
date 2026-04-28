# 01. 并行（Parallelism）

> 源码位置：`python/sglang/multimodal_gen/runtime/distributed/`、`runtime/layers/usp.py`、`runtime/layers/attention/layer.py`、`runtime/managers/gpu_worker.py`、`runtime/pipelines_core/executors/parallel_executor.py`、`runtime/server_args.py`。

## 1. 为什么要有一套独立的并行栈

LLM 侧的 `srt` 已经提供 TP/DP/PP/EP 等并行，但 **扩散模型** 天然拥有一些 LLM 所没有的特征：

1. 视觉 token 可能达到 `10^5` 量级（视频 DiT 甚至更多），**序列并行比张量并行更能省显存**；
2. **Classifier-Free Guidance (CFG)** 需要一条请求同时算 cond 和 uncond，可以并行化为两个 rank；
3. VAE 和 DiT 的计算负载、通信模式完全不同，可以使用不同的通信组；
4. Ulysses + Ring 的组合（USP）对长序列注意力非常有效。

因此 `multimodal_gen` 基于 FastVideo / xDiT / vLLM parallel_state 的思路实现了一套独立的分布式抽象。

## 2. 支持的并行维度

| 并行类型 | 参数 / 变量 | 对应 PG 名字 | 作用 |
|----------|-------------|--------------|------|
| Tensor Parallel | `tp_size` | `_TP` / `get_tp_group()` | 线性层 / Norm 列或行切分 |
| Sequence Parallel (总度) | `sp_degree = ulysses × ring` | `_SP` / `get_sp_group()` | 沿序列维切分 token |
| Ulysses 子组 | `ulysses_degree` | `PROCESS_GROUP.ULYSSES_PG` | 序列 ↔ 头维 `all_to_all` |
| Ring 子组 | `ring_degree` | `PROCESS_GROUP.RING_PG` | `_templated_ring_attention` |
| CFG Parallel | `enable_cfg_parallel` ⇒ `cfg_degree=2` | `get_cfg_group()` | cond/uncond 拆到 2 个 rank |
| Data Parallel | `dp_size` | `get_dp_group()` | batch / 请求 replica |
| Pipeline Parallel | 代码预留，**worker 固定 PP=1** | `PipelineGroupCoordinator` | 分层流水（尚未经 CLI 暴露）|
| DiT / VAE 组 | 自动派生 | `_DIT` / `_VAE` | 不同角色间通信隔离 |
| FSDP / HSDP | `use_fsdp_inference`、`hsdp_replicate_dim`、`hsdp_shard_dim` | — | 权重分片推理，降低单卡模型显存 |

上面 9 种里，**只有 USP（Ulysses + Ring）和 CFG Parallel 是 diffusion 特有**；其它与 LLM 并行类似，但参数名、行为和耦合方式独立。

## 3. World 的数学划分：`RankGenerator`

`initialize_model_parallel`（`runtime/distributed/parallel_state.py` 约 351–358 行）写死了正交顺序 **`"tp-sp-pp-cfg-dp"`**：

```351:358:python/sglang/multimodal_gen/runtime/distributed/parallel_state.py
    rank_generator: RankGenerator = RankGenerator(
        tensor_parallel_degree,
        sequence_parallel_degree,
        pipeline_parallel_degree,
        classifier_free_guidance_degree,
        data_parallel_size,
        "tp-sp-pp-cfg-dp",
    )
```

`RankGenerator` 位于 `runtime/utils/distributed.py` 约 165–234 行，底层用 `generate_masked_orthogonal_rank_groups` 按 mask 生成每个维度的 rank 列表。`get_ranks("tp")`、`get_ranks("sp")` 即得到对应维度的 **rank 划分**。

`dit_parallel_size` 的约束（约 332–338 行）：

```332:338:python/sglang/multimodal_gen/runtime/distributed/parallel_state.py
    dit_parallel_size = (
        data_parallel_size
        * classifier_free_guidance_degree
        * sequence_parallel_degree
        * pipeline_parallel_degree
        * tensor_parallel_degree
    )
```

同时通过 `init_dit_group` / `init_vae_group` 再为 DiT / VAE 建立额外的大组，VAE rank 位于 `dit_parallel_size` 之后（约 810–835 行）。

## 4. 两级 Process Group 与设备通信器

- **GroupCoordinator**（`group_coordinator.py` 约 124–217 行）：对每个 `group_ranks` 调用 `torch.distributed.new_group` 建 **NCCL 的 `device_group`**，并再建 **Gloo 的 `cpu_group`**（barrier/pyobj 广播用）。
- **CudaCommunicator**（`device_communicators/cuda_communicator.py`）：`world_size>1` 时包装成 PyNCCL；内部 `PyNcclCommunicator` 必须绑在 `cpu_group` 上，因为 NCCL group 不能用于 pynccl 初始化（`pynccl.py` 约 46–51 行注释）。
- **`DistributedAutograd.AllToAll4D`**（`base_device_communicator.py` 约 102–195 行）：提供 **可反传**的 4D all_to_all，是 Ulysses 并行的核心 primitive。

## 5. SP 再切子组：Ulysses + Ring

`parallel_groups.py` 的 `set_seq_parallel_pg_by_sp_groups`（约 25–91 行）在**每个 SP 组的本地 rank 顺序**上再切：

- `sp_degree = ring × ulysses`；
- 默认 `use_ulysses_low=True`：先按连续块建 **Ulysses 组**，再按跨步下标建 **Ring 组**；
- 这种设计允许 TP>1 时 **SP 组 rank 不连续** 也能得到正确子组（文件头注释 33–35 行）。

`SequenceParallelGroupCoordinator`（`group_coordinator.py` 约 1208–1239 行）存下：`ulysses_group` / `ring_group`、`ulysses_world_size` / `ulysses_rank`、`ring_world_size` / `ring_rank`。

## 6. USP 的三段式数据流

USP = Unified Sequence Parallel，公式 `sp_degree = ulysses_degree × ring_degree`。

在 Attention 层（`runtime/layers/attention/layer.py`），`USPAttention.forward` 分三步（约 520–544 行）：

1. **Ulysses 子组 A2A**：`_usp_input_all_to_all`（`usp.py` 约 36–44 行），通过 `get_sp_group().ulysses_group` 做 `ft_c.all_to_all_single`，把 Q/K/V 从 **序列维** 交换到 **头维**；
2. **Ring 子组 Ring Attention**：`ring_attn`（`usp.py` 约 161–252 行）调用 PyTorch `_templated_ring_attention`，在 `get_sp_group().ring_group` 内分块传递 K/V；若 `ring_degree==1` 则退化为本地 `attn_impl.forward`；
3. **Ulysses A2A 回放**：`_usp_output_all_to_all` 再把结果从头维换回序列维。

额外的 `UlyssesAttention` / `UlyssesAttention_VSA` 是纯 Ulysses 版本（不带 Ring），其中 `UlyssesAttention_VSA` 会同时 A2A 传 `gate_compress`，给 VSA 稀疏 attention 用（约 159–220 行）。

## 7. CFG 并行

`enable_cfg_parallel` 开启时 `cfg_degree=2`，cond 和 uncond 两次 forward 被拆到同一 SP 组内的两个 rank 上同时跑：

- 通信入口：`cfg_model_parallel_all_gather` / `cfg_model_parallel_all_reduce`（`runtime/distributed/communication_op.py`）；
- 执行器：`parallel_executor.py` 的 `StageParallelismType.CFG_PARALLEL` 分支（约 64–93 行），负责 `broadcast_pyobj`、收集结果；
- 自动启用条件（`server_args._adjust_parallelism` 约 600–640 行）：
  - 未显式指定 `--enable-cfg-parallel`；
  - GPU 数 ≥ 2 且整除 `dp_size × tp_size × 2`；
  - `ulysses` / `ring` 也都未显式指定；
  - `_model_default_uses_cfg()`：默认 sampling_params 里 `guidance_scale>1` 且带 `negative_prompt`；
  - 模型不在黑名单（LTX 等显式禁用）。

## 8. 并行落点一览

| 机制 | 主要落点 |
|------|----------|
| **TP** | `runtime/layers/linear.py`、`lora/linear.py`、`layernorm.py`、`vocab_parallel_embedding.py`，以及 `communication_op.tensor_model_parallel_all_reduce` / `all_gather` |
| **SP A2A (4D)** | `UlyssesAttention` 的 `sequence_model_parallel_all_to_all_4D`（`attention/layer.py` 约 121、153–154 行）|
| **USP + Ring** | `USPAttention`、`usp.ring_attn` |
| **SP all_gather** | 许多 DiT（`wanvideo.py`、`helios.py`、`zimage.py`、`ltx_2.py`）对 hidden / freqs 做 `sequence_model_parallel_all_gather` |
| **CFG** | `cfg_model_parallel_*`；`ParallelExecutor` 分支 |
| **FSDP** | `runtime/loader/fsdp_load.py`，各 `component_loaders` |
| **Text Encoder Folding** | `_get_folding_tp_group`（`distributed/__init__.py` 61–71 行）可让 text encoder 选择 `sp / ulysses / ring / tp` 作 folding |

## 9. 与 `ServerArgs` 的自动推断

关键字段（`server_args.py` 约 147–194 行）：`tp_size`、`sp_degree`、`ulysses_degree`、`ring_degree`、`dp_size`、`enable_cfg_parallel`、`hsdp_replicate_dim`、`hsdp_shard_dim`、`use_fsdp_inference`。

`_adjust_parallelism`（约 550–620 行）的默认值推断顺序：

1. `tp_size` 未指定 → `1`；
2. 若未指定 `--enable-cfg-parallel` 且条件满足（见 §7），**自动开启 CFG parallel**；
3. 若未指定 `sp_degree`：`num_gpus_per_group = dp_size * tp_size`，开启 CFG 时 `×2`；整除则 `sp_degree = num_gpus // num_gpus_per_group`；
4. 若 `ulysses` 和 `ring` 都未指定且 `sp_degree != 1`：`ulysses_degree = sp_degree`（纯 Ulysses 路径），否则默认 1；
5. 校验 `sp_degree == ring_degree * ulysses_degree` 以及 `num_gpus` 的整除关系（约 1383–1428 行）。

`GPUWorker.init_device_and_model`（`runtime/managers/gpu_worker.py` 约 94–115 行）把这些参数传给 `maybe_init_distributed_environment_and_model_parallel(...)` 完成进程内 PG 创建。**注意：`pipeline_parallel_degree` 目前不经过 worker 路径暴露，故 PP 恒为 1。**

## 10. 与 `srt`（LLM）并行的区别

| 维度 | `multimodal_gen` (diffusion) | `srt` (LLM) |
|------|------------------------------|-------------|
| 并行语义 | CFG parallel、SP = Ulysses × Ring、DiT/VAE 分组 | TP / DP / PP / EP，无 CFG |
| 序列并行 | `tp-sp-pp-cfg-dp` 顺序，SP 内 ULYSSES_PG + RING_PG | CP / SP 设计各异 |
| USP | 显式 USPAttention（Ulysses+Ring 拼装） | 通常无同名模块 |
| 权重切分 | 更强调 `use_fsdp_inference` + HSDP mesh | EP / 多级 FSDP |
| 执行层 | `ParallelExecutor` 按 stage `StageParallelismType` 做 CFG / main-rank 分支 | Scheduler / token 路径不同 |

## 11. 引用清单

| 文件 | 符号 | 作用 |
|------|------|------|
| `runtime/distributed/parallel_state.py` | `initialize_model_parallel`、`maybe_init_distributed_environment_and_model_parallel`、`get_*_group`、`init_dit_group`、`init_vae_group` | 全局 PG 初始化 |
| `runtime/utils/distributed.py` | `RankGenerator`、`generate_masked_orthogonal_rank_groups` | World 正交划分 |
| `runtime/distributed/parallel_groups.py` | `set_seq_parallel_pg_by_sp_groups`、`PROCESS_GROUP` | SP 内 Ulysses/Ring |
| `runtime/distributed/group_coordinator.py` | `GroupCoordinator`、`PipelineGroupCoordinator`、`SequenceParallelGroupCoordinator` | 通信封装 / PP P2P |
| `runtime/distributed/communication_op.py` | `tensor_model_parallel_*`、`sequence_model_parallel_*`、`cfg_model_parallel_*` | 上层通信入口 |
| `runtime/distributed/device_communicators/base_device_communicator.py` | `DistributedAutograd`、`DeviceCommunicatorBase` | all_to_all / all_gather 反传 |
| `runtime/distributed/device_communicators/cuda_communicator.py` | `CudaCommunicator` | NCCL 实装 |
| `runtime/layers/usp.py` | `_usp_input_all_to_all`、`_usp_output_all_to_all`、`ring_attn` | USP + Ring |
| `runtime/layers/attention/layer.py` | `UlyssesAttention`、`UlyssesAttention_VSA`、`USPAttention` | 注意力并行策略 |
| `runtime/pipelines_core/executors/parallel_executor.py` | `ParallelExecutor.collect_from_main`、`_execute` | 请求/阶段与 CFG、SP 广播 |
| `runtime/managers/gpu_worker.py` | `GPUWorker.__init__`、`init_device_and_model` | 进程内分布式 |
| `runtime/server_args.py` | `ServerArgs`、`_adjust_parallelism`、`_validate_parallelism` | CLI 与默认值 |

## 12. 调优建议

- **单机 8 卡、追求最大吞吐**：优先 `tp_size=1 + sp_degree=8 + enable_cfg_parallel`（若模型 CFG）→ 系统会选 `cfg=2, sp=4, ulysses=4, ring=1` 或类似组合。
- **长视频 / 显存紧张**：`ring_degree>1` 减少峰值 KV 占用；与 `dit_layerwise_offload` 搭配进一步省显存（详见 [`07_offload.md`](./07_offload.md)）。
- **小显存卡跑 Flux2**：`use_fsdp_inference=True` + 合适的 HSDP mesh。
- **需要 attention backend 支持 Ring**：仅 FA 和 SageAttention 两个 backend 能与 Ring 组合（详见 [`04_attention_backends.md`](./04_attention_backends.md)）。
- **Warning**：`SP + TP 混用` 并开启 cache-dit 会 warning，需在 `cache_dit_integration._patch_cache_dit_similarity` 里 patch `CachedContextManager.similarity` 做多卡 `all_reduce`。
