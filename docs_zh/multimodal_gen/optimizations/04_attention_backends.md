# 04. 注意力后端与稀疏注意力

> 源码位置：`python/sglang/multimodal_gen/runtime/layers/attention/`（`__init__.py`、`layer.py`、`selector.py`、`turbo_layer.py`、`STA_configuration.py`、`backends/*`）、`runtime/layers/attention/backends/*`、`runtime/platforms/*`、`runtime/server_args.py`、`csrc/attn/vmoba_attn/`。

扩散模型里的 attention 特点：**长序列（视频 token 可达 10⁵ 量级）、稀疏可用性高（相邻帧强相关、空间邻近块强相关）、需要并行（USP / Ring）**。`multimodal_gen` 为此提供了一个 **可插拔 attention backend** 框架，并集成多种稀疏注意力算法。

## 1. 抽象框架

| 文件 | 作用 |
|------|------|
| `attention/__init__.py`（约 4–30 行）| 导出 `AttentionBackend`、`AttentionMetadata`、`LocalAttention`、`UlyssesAttention`、`UlyssesAttention_VSA`、`USPAttention`、`get_attn_backend`、`MinimalA2AAttnOp` |
| `attention/attention_backend.py` | 抽象基类：`AttentionBackend`、`AttentionMetadata`、`AttentionImpl`、`wrap_attention_impl_forward`（约 174–179 行内核调试包装） |
| `attention/selector.py` | **全局后端解析**：`get_attn_backend`、`global_force_attn_backend`、环境变量 `SGLANG_DIFFUSION_ATTENTION_BACKEND`（`STR_BACKEND_ENV_VAR`，见 `utils.py` 约 63 行）|
| `attention/layer.py` | 多头封装：`LocalAttention`、`UlyssesAttention`、`UlyssesAttention_VSA`、`USPAttention` |
| `attention/turbo_layer.py` | TurboDiffusion 的 CP A2A：`MinimalA2AAttnOp`（仅允许 SLA / Sage-SLA 实现）|
| `attention/STA_configuration.py` | STA 离线 mask 搜索与策略选择（`configure_sta`、`select_best_mask_strategy`）|

## 2. 支持的 backend 清单

枚举定义在 `runtime/platforms/interface.py` 约 27–55 行：

| 枚举名 | CLI 字符串 | 类 | 平台 |
|--------|-----------|-----|------|
| `FA` | `fa` | `FlashAttentionBackend`（`flash_attn.py`） | CUDA 为主，Hopper→FA3, Blackwell→FA4 |
| `FA2` | `fa2` | `FlashAttention2Backend`（`flash_attn_2.py`）| 同上 |
| `TORCH_SDPA` | `torch_sdpa` | `SDPAImpl`（`sdpa.py`）| 通用兜底 |
| `SAGE_ATTN` | `sage_attn` | `SageAttentionBackend`（`sage_attn.py`）| CUDA |
| `SAGE_ATTN_3` | `sage_attn_3` | `SageAttention3Backend`（`sage_attn3.py`）| Blackwell |
| `SLIDING_TILE_ATTN` | `sliding_tile_attn` | `SlidingTileAttentionBackend`（`sliding_tile_attn.py`）| 视频 DiT |
| `VIDEO_SPARSE_ATTN` | `video_sparse_attn` | `VideoSparseAttentionBackend`（`video_sparse_attn.py`）| 视频 DiT |
| `SPARSE_VIDEO_GEN_2_ATTN` | `sparse_video_gen_2_attn` | `SparseVideoGen2AttentionBackend`（`sparse_video_gen_2_attn.py`）| 视频 DiT |
| `VMOBA_ATTN` | `vmoba_attn` | `VmobaAttentionBackend`（`vmoba.py`）| 视频 DiT |
| `AITER` | `aiter` | AMD AITer（`aiter.py`）| ROCm |
| `AITER_SAGE` | `aiter_sage` | AITer + Sage wrapper（`aiter_sage.py`）| ROCm |
| `SLA_ATTN` | `sla_attn` | `SparseLinearAttentionBackend`（`sparse_linear_attn.py`）| TurboDiffusion 路线 |
| `SAGE_SLA_ATTN` | `sage_sla_attn` | Sage + SLA 组合 | 同上 |
| — | — | `AscendFABackend`（`ascend_fa.py`）| 华为 NPU |
| — | — | `XPUAttentionBackend`（`xpu_backend.py`）| Intel XPU |
| `NO_ATTENTION` | `no_attention` | 占位 | — |

CLI 字符串经 `selector.py` 转成枚举（约 124–128 行）：`AttentionBackendEnum[server_args.attention_backend.upper()]`。`server_args.py` 约 464–466 行把 `fa3`/`fa4` 归一成 `fa`。

## 3. 选型流程：`selector.get_attn_backend`

流程（约 89–161 行）：

1. `forced_attn_backend`（`global_force_attn_backend`）优先级最高；
2. `ServerArgs.attention_backend` → `AttentionBackendEnum`；
3. 模型层 `supported_attention_backends` 约束：若 CLI 选择与集合冲突则置 `None`；
4. `current_platform.get_attn_backend_cls_str(selected_backend, head_size, dtype)` 得到类名字符串；
5. `resolve_obj_by_qualname` 动态导入。

因此 **同一枚举名在不同平台** 可以解析到完全不同的类（例如 CUDA FA vs NPU AscendFA vs XPU 专用模块）。

## 4. 各 backend 详解

### 4.1 FlashAttention（`fa` / `fa2`）

- `flash_attn.py` 从 `sglang.jit_kernel.flash_attention` 引入 `flash_attn_varlen_func`（约 8 行）；
- `jit_kernel/flash_attention.py` 按 GPU 架构分发：Hopper 走 FA3（`sgl_kernel.flash_attn`）、Blackwell 走 FA4（`flash_attn.cute`，`jit_kernel/flash_attention_v3.py` 约 60–67 行）；
- 平台降级：非 sm80+/非 fp16/bf16/head 尺寸不支持时 `cuda.py` 约 419–455 行会降级到 `TORCH_SDPA`；
- **USP Ring 路径仅允许** `FA` 与 `SAGE_ATTN`（`layer.py` 约 364–374 行）。

### 4.2 SageAttention 2 / 3

- `sage_attn.py` 调用 `sageattention.sageattn`，布局 `NHD`（约 61–70 行）；CUDA，未装 Sage 包时 `cuda.py` 约 258–274 行回退到 FA；
- `sage_attn3.py` 使用 `sageattn3_blackwell`（约 7、90 行）；GQA/MQA 时退回 SDPA（约 67–88 行）；回退到 SDPA（`cuda.py` 约 275–288 行）。

### 4.3 Torch SDPA

兜底；`LocalAttention` / `USPAttention` 在带 mask 或 skip SP 时也走 SDPA（`layer.py` 约 288–315、429–499 行）。

### 4.4 AITer / AITer-Sage（ROCm）

- `aiter.py` 调 `aiter.flash_attn_func`（约 84–90 行）；
- `aiter_sage.py` 调 `aiter.ops.triton.attention.fav3_sage.fav3_sage_wrapper_func`（约 52–80 行）；
- ROCm 默认 backend 通常为 `aiter`（`server_args._set_default_attention_backend` 约 1446–1454 行）。

### 4.5 Ascend FA（华为 NPU）

`ascend_fa.py`：`torch.ops.npu.npu_fused_infer_attention_score`（约 91–100 行）。NPU 平台在 `selected_backend==FA` 时指向 `AscendFABackend`（`npu.py` 约 119–121 行）。

### 4.6 Intel XPU

`xpu_backend.py` **直接** `from sgl_kernel.flash_attn import flash_attn_varlen_func`（约 11–14 行），变长拼接后调用。XPU 平台默认在 fp16/bf16 且 head 在白名单时用 `XPUAttentionBackend`，否则 SDPA。

## 5. 稀疏注意力：五种机制对比

| 方法 | 稀疏依据 | 块结构 | 关键参数（代码字段）|
|------|----------|--------|---------------------|
| **VSA** (`video_sparse_attn.py`) | 块级 top-k + gate 压缩权重 | 固定 **4×4×4** token tile | `VSA_sparsity` → `cur_topk = ceil((1-sparsity) * (T/tile_vol))`（约 313–328 行）|
| **VMoBA** (`vmoba.py`) | 块间相似度门控 + varlen flash | 随层切换 temporal / spatial / spatiotemporal chunk | `moba_topk`、`moba_chunk_size`、`simsum_threshold`、`select_mode`（metadata 约 49–70 行，forward 约 190–213 行）|
| **SVG2 / SAP** (`sparse_video_gen_2_attn.py`) | K-means 聚类标签驱动动态块 | 帧/空间划分 + 动态图 | `num_q_centroids`、`top_p_kmeans`、`kmeans_iter_*`、`first_layers_fp` 等（metadata 约 90–107 行）|
| **STA** (`sliding_tile_attn.py`) | 手工/搜索得到的 3D 滑窗 mask | 基 tile `[6,8,8]`，序列 reshape + `sliding_tile_attention` | `STA_param` 每层每头窗口；search 模式对比 L1/L2；`mask_strategy_file_path` JSON |
| **SLA** (`sparse_linear_attn.py`) | 块均值打分 top-k（Sage 式 smooth-k） | `BLKQ` / `BLKK` 池化块 | `topk_ratio`（默认 0.1，约 238–243 行）|

### 5.1 VSA 深挖

`VideoSparseAttention` 关键调用是 `vsa.video_sparse_attn`（约 322–330 行）。由于固定 4×4×4 tile，使用前需保证 `(num_frames, H, W)` 能被 4 整除。与 `UlyssesAttention_VSA` 共同工作时，`gate_compress` 会和 Q/K/V 一起 `all_to_all_4D`（`layer.py` 约 159–220 行）。

### 5.2 VMoBA 深挖

层索引通过 `prefix` 解析（`vmoba.py` 约 169–173 行），按层循环选择 temporal / spatial / st 三种 chunk；稀疏度由 `moba_config.moba_threshold` 与 `moba_topk` 联合控制。底层 Python 实现在 `csrc/attn/vmoba_attn/vmoba/vmoba.py`（约 1087 行大模块），内含：

- `calc_chunks`（约 34–70 行）：chunk 划分；
- `_select_threshold_query_head`（约 76 行起）：阈值式选块；
- 基于 `flash_attn_varlen` 的 `moba_attn_varlen`。

### 5.3 SVG2 深挖

本质是 online k-means 聚类 + 动态块稀疏 + FlashInfer。`SparseVideoGen2AttentionMetadata` 约 90–107 行存储 centroid、kmeans 迭代数、`first_layers_fp`、`first_times_fp`（前若干层/步保持 full attention）。

### 5.4 STA 深挖

STA 分三模式（`configure_sta` 约 14–250 行）：

- `STA_searching`：生成候选 mask 网格；
- `STA_tuning` / `STA_tuning_cfg`：读 JSON、按 `skip_time_steps` 选最优；
- `STA_inference`：直接加载 JSON。

`select_best_mask_strategy`（约 301–357 行）：前若干步用全注意力 mask；之后对每个 `(t, layer, head)` 从候选中选 **L2 最小** 的策略；输出 overall sparsity = `1 - total_tokens/total_length`（约 341–355 行）。

运行时 `SlidingTileAttentionImpl.forward`（约 200–315 行）根据 `STA_param` 决定当前步/层/头使用什么窗口；search 模式会附加 loss 计算写入 `forward_batch`。

### 5.5 SLA 深挖

`SparseLinearAttentionImpl`（约 227–338 行）用 Triton `get_block_map` + top-k 块稀疏 + 线性注意力残差；`SageSparseLinearAttentionImpl`（约 388–402 行）依赖 `spas_sage_attn`。

Turbo 场景下 `MinimalA2AAttnOp.forward`（`turbo_layer.py` 约 228–274 行）会把当前 backend 强制回落到 `SparseLinearAttentionBackend` / `SageSparseLinearAttentionBackend`。

## 6. TurboLayer 与 STA 配置

### 6.1 `turbo_layer.py`

- `_SeqAllToAll` / `async_a2a_communicate`（约 44–124 行）实现**序列维 ↔ 头维** 的 all-to-all，与 Transformer Engine 的 CP 类似；
- `DistributedAttention.forward`（约 214–222 行）封装 A2A；
- `MinimalA2AAttnOp`（约 228–274 行）：获取 `get_attn_backend` 的实现类；若不是 SLA / Sage-SLA 会 warning 并回落；forward 末尾 `rearrange(..., "b ... h l -> b ... (h l)")` 匹配 Turbo 期望形状。

### 6.2 `STA_configuration.py`

离线 mask 搜索工具（见 §5.4）。`STA_configuration` 产出 JSON 供推理加载；运行时参数 `--mask-strategy-file-path`（`server_args.py` 约 904–907 行）。

## 7. USP + Ring 与各 backend 的协作

- `UlyssesAttention`（`layer.py` 约 84–156 行）：先 `sequence_model_parallel_all_to_all_4D` → `attn_impl.preprocess_qkv` → `forward` → `postprocess_output` → 再 A2A 还原。任何实现 `preprocess_qkv`/`postprocess_output` 的 backend（例如 VSA 的 tile 规整）都能接入。
- `UlyssesAttention_VSA`（约 159–220 行）：A2A 的是 `q,k,v,gate` 四维 cat，要求无 replicated QKV。
- `USPAttention`（约 321–545 行）：`skip_sequence_parallel` 或 world size 1 时直接 `attn_impl.forward`；否则 Ulysses A2A + Ring（`ring_degree>1`）。**Ring 路径仅允许 FA 与 SAGE_ATTN**（约 364–374、528–535 行）。
- 带 `attn_mask` 时倾向 SDPA / 自定义 gather 路径（约 429–499 行），Ring 未实现。
- Turbo 的 `DistributedAttention` 是独立的 CP process group + A2A，与 Ulysses / USP 不同。

## 8. 稀疏度控制字段速查

- **VSA**：`VSA_sparsity` 越大保留块越少。
- **STA**：由窗口三维尺寸决定；`configure_sta` 打印 overall_sparsity；推理用 JSON `mask_strategy_file_path`。
- **VMoBA**：`temporal_topk` / `spatial_topk` / `st_topk` + `moba_threshold`；`first_full_step` 前可强制 dense。
- **SVG2**：`num_q_centroids`、`top_p_kmeans`、`min_kc_ratio`。
- **SLA**：`topk_ratio`（metadata 默认 0.1）；Turbo 路径通过构造参数 `topk` 传入 `SparseLinearAttentionImpl`。

## 9. 与 `sgl-kernel` 的对接

- **FlashAttention**（CUDA 主路径）经 `sglang.jit_kernel.flash_attention`，其在 Hopper 会调 `sgl_kernel.flash_attn`；
- **Intel XPU** 直接 `from sgl_kernel.flash_attn import flash_attn_varlen_func`；
- **Sage 2/3 / Sage-SLA** 目前对接的是独立包（`sageattention`、`sageattn3`、`spas_sage_attn`）或 ROCm AITer 里的 `fav3_sage`，**不**直接从 `sgl-kernel` 引入 Sage 模块。

## 10. `csrc/attn/vmoba_attn`

`csrc/attn/vmoba_attn/vmoba/vmoba.py` 是 VMoBA 的 Python 实现入口（体量约 1087 行），被打包成 `kernel.attn.vmoba_attn` 命名空间。backend `vmoba.py` 约 10–14 行 `from kernel.attn.vmoba_attn.vmoba import ...` 引用打包结果。

## 11. 整体协作简图

```text
ServerArgs.attention_backend + attention_backend_config
        │
        ▼
   selector.get_attn_backend
        │
        ▼
 current_platform.get_attn_backend_cls_str
        │
        ▼
 *Backend 类 (FlashAttentionBackend / VSA / STA / VMoBA / ...)
        │
        ▼
 *Impl.forward（或 preprocess_qkv/postprocess_output 配合 USP）
        ▲
        │
 Layer 级：LocalAttention / Ulysses* / USPAttention / MinimalA2AAttnOp
```

## 12. 调优建议

| 场景 | 推荐 backend |
|------|-------------|
| 通用 DiT 推理（Hopper） | `fa` |
| Blackwell | `fa`（自动 FA4）或 `sage_attn_3` |
| ROCm | `aiter` 或 `aiter_sage` |
| 视频 DiT（Wan / HunyuanVideo） | `video_sparse_attn`（首选）、`vmoba_attn`、`sparse_video_gen_2_attn` |
| 长视频 + 低延迟 | `sliding_tile_attn` + 预先 `STA_searching` |
| Turbo 系列（TurboWan 等） | `sla_attn` / `sage_sla_attn`（由 `MinimalA2AAttnOp` 绑定）|
| Intel XPU | `xpu_backend` 自动绑定 |
| NPU | `ascend_fa` |
| 出现维度/dtype 不兼容 | `torch_sdpa` 兜底 |

---

> 若要从某个具体 pipeline（例如 Wan VSA 或 STA_searching）追踪 `denoising.py` → `AttentionMetadataBuilder.build` 的**数据流**，可以结合 [`../10_wan2_1_end_to_end.md`](../10_wan2_1_end_to_end.md) 一起看。
