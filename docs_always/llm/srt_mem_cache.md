# `python/sglang/srt/mem_cache` 模块分析

## 定位

`mem_cache` 管理 SRT 的 KV cache 与 prefix cache。它既负责物理 KV tensor 的分配/释放，也负责请求到 token 位置的映射、Radix prefix tree 命中、session-aware cache、SWA/Mamba/linear state cache、HiCache 分层存储和稀疏缓存策略。

## 关键文件

- `memory_pool.py`：核心内存池。`ReqToTokenPool` 映射请求到 token 位置；`MambaPool`、KV pool 相关类管理不同 cache 形态；底层提供写入 KV cache 的 Triton/JIT/fallback 路径。
- `allocator.py`：token-to-KV pool allocator 抽象。
- `radix_cache.py`：普通 Radix prefix cache，管理 prefix tree、lock ref、eviction 和 cache hit。
- `chunk_cache.py`：chunked cache 变体。
- `session_aware_cache.py`：按 session 管理 prefix cache。
- `hiradix_cache.py`、`hi_mamba_radix_cache.py`、`hicache_storage.py`、`hybrid_cache/`、`storage/`：HiCache/分层存储和混合 cache。
- `swa_memory_pool.py`、`swa_radix_cache.py`：sliding-window attention 的 cache 结构。
- `base_prefix_cache.py`、`evict_policy.py`、`flush_cache.py`：cache 抽象、淘汰策略和清理。
- `sparsity/`：针对 NSA/Quest 等稀疏 attention 的 cache backend adapter 和算法。

## 运行流程

Scheduler 为请求分配 `ReqToTokenPool` slot，并从 `BaseTokenToKVPoolAllocator` 获取 KV cache loc。模型 forward 时 attention 层把新 K/V 写入 token loc 对应的物理 KV cache。请求完成或被 abort 后，scheduler 释放请求 slot，并根据 prefix cache 策略保留或释放 KV token。

Prefill 时，`RadixCache` 用输入 token 查找可复用 prefix，命中部分会减少实际 extend 长度；decode 时新 token 继续追加到 cache loc。HiCache/remote storage 场景会把部分 KV 在 GPU、host、外部存储或 disaggregation transfer backend 之间搬运。

## 依赖关系

`mem_cache` 被 `managers.scheduler`、`schedule_batch`、`model_executor`、`disaggregation` 和 attention backend 直接使用。它依赖 `layers.radix_attention`、`configs.mamba_utils`、`utils`、Triton/JIT kernel 和平台检测逻辑。

## 设计要点和风险

- cache loc 是 scheduler、model executor、attention backend 和 disaggregation 的共享坐标系，任何重排都必须同步。
- `ReqToTokenPool.alloc` 支持 chunked prefill 复用 `req_pool_idx`，这类路径对请求状态不变量要求很高。
- Radix cache 的 lock/ref/evict 语义直接影响吞吐和显存安全；abort/retract/session close 容易产生泄漏或重复释放。
- HiCache 与 disaggregation 会引入异步搬运和 staging buffer，水位、prefetch、free 时序需要和 scheduler 一致。
