# Qwen3.6-27B Scheduler、Prefill/Decode 与 Radix KV Cache

本文覆盖 `Scheduler` 入队、prefill/decode 调度、`lpm`、chunked prefill、decode retract、radix KV cache。当前脚本相关默认值：`CONTEXT_LENGTH=262144`、`MAX_PREFILL_TOKENS=16384`、`CHUNKED_PREFILL_SIZE=8192`、`SCHEDULE_POLICY=lpm`、`RADIX_EVICTION_POLICY=lru`。

## 1. 脚本参数入口

| 脚本变量 | server 参数 | 作用 |
| --- | --- | --- |
| `TP_SIZE=4` | `--tensor-parallel-size` | scheduler / TP worker 组织 |
| `CONTEXT_LENGTH=262144` | `--context-length` | 请求上下文上限 |
| `MEM_FRACTION_STATIC` | `--mem-fraction-static` | KV pool capacity profiling |
| `MAX_RUNNING_REQUESTS` | `--max-running-requests` | running batch 请求上限 |
| `MAX_QUEUED_REQUESTS` | `--max-queued-requests` | `waiting_queue` 上限 |
| `CHUNKED_PREFILL_SIZE=8192` | `--chunked-prefill-size` | 单个长请求一次 prefill chunk 上限 |
| `MAX_PREFILL_TOKENS=16384` | `--max-prefill-tokens` | 单轮 prefill batch token 预算 |
| `PREFILL_MAX_REQUESTS` | `--prefill-max-requests` | 单轮 prefill 最多请求数 |
| `SCHEDULE_POLICY=lpm` | `--schedule-policy` | longest prefix match 调度 |
| `RADIX_EVICTION_POLICY=lru` | `--radix-eviction-policy` | radix cache 驱逐策略 |
| `MAX_TOTAL_TOKENS` | `--max-total-tokens` | 可选 KV pool token 上限 |

脚本中的 `KV_BYTES_PER_TOKEN_PER_GPU=16384` 只用于启动前估算 `MAX_RUNNING_REQUESTS`。真实 KV pool 容量由 `ModelRunnerKVCacheMixin` 根据模型、KV dtype、显存和 `mem_fraction_static` 计算。

## 2. Scheduler 初始化

`Scheduler` 定义在 `python/sglang/srt/managers/scheduler.py:273` 附近，构造函数约 `scheduler.py:288-435`。初始化主线：

```text
Scheduler.__init__()
  -> init_model_config()
  -> init_ipc_channels()
  -> init_tokenizer()
  -> init_model_worker()
  -> init_cache_with_memory_pool()
  -> init_running_status()
  -> init_chunked_prefill()
  -> init_schedule_policy()
  -> init_overlap()
  -> init_request_dispatcher()
```

`init_model_worker()` 在 `scheduler.py:631-702`，创建 `TpModelWorker` 并读取 worker info：

- `max_total_num_tokens`
- `max_prefill_tokens`
- `max_running_requests`
- `max_queued_requests`
- `max_req_len`
- `max_req_input_len`
- `device`
- `forward_stream`

`TpModelWorker.__init__()` 在 `python/sglang/srt/managers/tp_worker.py:221-323`，会把 `model_runner.max_total_num_tokens`、`model_runner.max_running_requests` 等能力回传给 scheduler。

## 3. Memory Pool 与 Radix Cache

`Scheduler.init_cache_with_memory_pool()` 在 `scheduler.py:704-844`：

```text
tp_worker.get_memory_pool()
  -> model_runner.req_to_token_pool
  -> model_runner.token_to_kv_pool_allocator
CacheInitParams(...)
RadixCache(params) 或 MambaRadixCache(params)
```

KV pool 初始化在 `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`：

- `profile_max_num_token()`：用显存和每 token KV cell size 估算 capacity。
- `_resolve_token_capacity()`：应用 `max_total_tokens` 并按 page 对齐。
- `_resolve_max_num_reqs()`：用 token capacity 和 `max_running_requests` 得到请求数上限。
- `_init_pools()`：创建 `ReqToTokenPool`、KV pool、allocator。

Qwen3.6 的 config 是 hybrid GDN/linear attention 形态，scheduler 可能使用 `MambaRadixCache`；普通场景 fallback 是 `RadixCache`，见 `scheduler.py:801-818`。

## 4. 运行状态

`Scheduler.init_running_status()` 在 `scheduler.py:845-860`：

```python
self.waiting_queue = []
self.running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
self.cur_batch = None
self.last_batch = None
self.forward_ct = 0
self.num_retracted_reqs = 0
self.session_controller = SessionController(self.tree_cache)
```

含义：

- `waiting_queue`：请求进入 prefill 前的等待队列。
- `running_batch`：continuous batching 中正在 decode 的请求集合。
- `last_batch`：上一轮 prefill batch，会在下一轮合入 running batch。
- `chunked_req`：当前被拆分 prefill 的长请求。

`init_chunked_prefill()` 在 `scheduler.py:862-899`，会把 `chunked_prefill_size <= 0` 视为禁用；当前脚本默认启用 8192。

## 5. Scheduler 主循环

进程入口 `run_scheduler_process()` 在 `scheduler.py:3560-3616`：

```text
Scheduler(...)
pipe_writer.send(scheduler.get_init_info())
scheduler.run_event_loop()
```

普通模式 `event_loop_normal()` 在 `scheduler.py:1303-1320`：

```text
while True:
  recv_reqs = recv_requests()
  process_input_requests(recv_reqs)
  batch = get_next_batch_to_run()
  if batch:
      result = run_batch(batch)
      process_batch_result(batch, result)
  last_batch = batch
```

overlap 模式在 `scheduler.py:1331-1383`，把 forward 和上一轮结果处理错开，但阶段仍是接收、入队、取 batch、执行、处理结果。

## 6. 请求接收与入队

`recv_requests()` 在 `scheduler.py:1423-1558`。普通 TP 下，只有 `pp_rank == 0 && attn_tp_rank == 0 && attn_cp_rank == 0` 的 scheduler rank 从 tokenizer/RPC ZMQ socket 接收对象，其它 TP rank 通过 `broadcast_pyobj()` 获取广播后的请求。

`process_input_requests()` 在 `scheduler.py:1589-1610`，将请求交给 `_request_dispatcher`。dispatcher 在 `scheduler.py:1204-1258` 注册：

- `TokenizedGenerateReqInput -> handle_generate_request`
- `TokenizedEmbeddingReqInput -> handle_embedding_request`
- `BatchTokenizedGenerateReqInput -> handle_batch_generate_request`
- `AbortReq -> abort_request`
- 权重更新、session、profile、LoRA 等控制请求

生成请求入队：

```text
handle_generate_request()
  -> Req(...)
  -> init_req_max_new_tokens()
  -> validate_input_length()
  -> grammar_manager.process_req_with_grammar()
  -> _add_request_to_queue()
```

`_add_request_to_queue()` 在 `scheduler.py:1936-1958`：

```text
_set_or_validate_priority(req)
_abort_on_queued_limit(req)
_prefetch_kvcache(req)
waiting_queue.append(req)
req.time_stats.set_wait_queue_entry_time()
```

`max_queued_requests` 限制在 `scheduler.py:1985-2032`。默认队列满时 abort incoming request；启用 priority scheduling 时可能替换低优先级等待请求。

## 7. `get_next_batch_to_run`

核心调度入口在 `scheduler.py:2177-2282`：

```text
get_next_batch_to_run()
  -> _abort_on_waiting_timeout()
  -> _abort_on_running_timeout()
  -> 合并上一轮 extend batch 到 running_batch
  -> 清理 prefill-only 已完成请求
  -> get_new_batch_prefill()
  -> 如果有 new_batch，优先返回 prefill batch
  -> 否则 update_running_batch(running_batch)，返回 decode batch
```

关键点：

- 能组出新的 prefill batch 时优先跑 prefill。
- 上一轮 extend batch 会在下一轮合入 `running_batch`。
- 当没有新 prefill 可跑时，才推进 decode。
- chunked request 会在下一轮通过 cache 继续扩展，避免重复占用。

## 8. `lpm` 调度

`SchedulePolicy` 在 `python/sglang/srt/managers/schedule_policy.py`。`lpm` 是 cache-aware policy，含义是 longest prefix match。

`SchedulePolicy.calc_priority()` 在 `schedule_policy.py:117-159`：

```text
calc_priority(waiting_queue, running_batch)
  -> _determine_active_policy()
  -> _compute_prefix_matches(waiting_queue, LPM)
  -> _sort_by_longest_prefix(waiting_queue, temporary_deprioritized)
```

`_compute_prefix_matches()` 会为每个等待请求构造：

```text
prefix_ids = origin_input_ids + output_ids
extra_key = req.extra_key
tree_cache.match_prefix(RadixKey(token_ids=prefix_ids, extra_key=extra_key))
```

并写回：

- `req.prefix_indices`
- `req.last_node`
- `req.last_host_node`
- `req.host_hit_length`

`_sort_by_longest_prefix()` 按 `len(req.prefix_indices)` 从大到小排序。效果是优先调度 KV cache 命中更长的请求，适合共享 system prompt、工具模板或多轮前缀的 agent 流量。

## 9. Prefill batch 构造

`_get_new_batch_prefill_raw()` 在 `scheduler.py:2324-2545`：

```text
policy.calc_priority(waiting_queue, running_batch)
adder = PrefillAdder(... max_prefill_tokens, chunked_prefill_size,
                     max_running_requests, prefill_max_requests ...)
if self.chunked_req:
    adder.add_chunked_req(self.chunked_req)
for req in waiting_queue:
    req.init_next_round_input(tree_cache)
    adder.add_one_req(req)
remove can_run_list from waiting_queue
ScheduleBatch.init_new(...)
new_batch.prepare_for_extend()
```

`Req.init_next_round_input()` 在 `schedule_batch.py:940-1022`：

- `fill_ids = origin_input_ids + output_ids`
- 通过 `tree_cache.match_prefix()` 找命中前缀。
- `extend_input_len = len(fill_ids) - len(prefix_indices)`

因此真正进入 extend forward 的只是未命中 suffix。

`ScheduleBatch.prepare_for_extend()` 在 `schedule_batch.py:1560-1735`：

- 设置 `forward_mode = EXTEND`。
- `input_ids = fill_ids[len(prefix_indices):]`。
- 统计 prefix/extend/seq lens。
- `alloc_for_extend()` 分配 KV。
- 写入 `req_to_token_pool`。

## 10. `PrefillAdder` 与 chunked prefill

`PrefillAdder` 定义在 `schedule_policy.py:375-440`。Scheduler 创建它的位置在 `scheduler.py:2377-2393`。

关键预算：

| 预算 | 来源 | 含义 |
| --- | --- | --- |
| `rem_total_tokens` | KV allocator + radix evictable tokens | 本轮可用总 KV token |
| `rem_input_tokens` | `max_prefill_tokens` | 单轮 prefill 输入 token |
| `rem_chunk_tokens` | `chunked_prefill_size` | 单个 chunk token |
| `prefill_max_requests` | server args | 单轮请求数 |
| `max_running_requests` | worker info | running 请求上限 |

当请求未命中部分超过 chunk 预算时，`add_one_req()` 会截断本轮 `extend_input_len`，设置 `new_chunked_req`，见 `schedule_policy.py:816-843`。

chunked prefill 结果处理：

- `req.is_chunked > 0` 时，prefill 未完成，不立即 stream 输出。
- 下一轮把已完成 chunk 写入 radix cache。
- 后续 chunk 通过 prefix match 复用前面 KV。

相关代码在 `scheduler_output_processor_mixin.py:250-278`、`scheduler.py:2191-2196`、`radix_cache.py:510-573`。

## 11. Decode 更新与 retract

没有新 prefill batch 时，`update_running_batch()` 在 `scheduler.py:2547-2625` 推进 decode：

```text
batch.filter_batch()
tree_cache.flush_write_through_acks()
batch.check_decode_mem()
if KV 不足:
    batch.retract_decode()
    retracted req -> _add_request_to_queue(is_retracted=True)
batch.prepare_for_decode()
```

`ScheduleBatch.check_decode_mem()` 在 `schedule_batch.py:1937-1940`，会尝试 `evict_from_tree_cache()` 释放可驱逐 prefix cache。

`ScheduleBatch.retract_decode()` 在 `schedule_batch.py:1950-2019`：

- 选择部分 running 请求释放 KV。
- 至少保留一个请求，除非最后一个也无法满足内存。
- 更新 `new_token_ratio`，后续调度更保守。
- retracted 请求回到等待队列，后续重新 prefix match / prefill。

`prepare_for_decode()` 在 `schedule_batch.py:2062-2174`，为每个请求分配下一 token 的 KV slot，并设置 `forward_mode=DECODE`。

## 12. RadixKey、TreeNode、match/insert/evict

`RadixKey` 在 `python/sglang/srt/mem_cache/radix_cache.py:71-99`：

- `token_ids`：token 序列。
- `extra_key`：cache namespace，例如 LoRA、cache salt。
- `is_bigram`：EAGLE 相关。

`extra_key` 避免不同 adapter 或 cache namespace 之间错误复用 KV。

`TreeNode` 在 `radix_cache.py:121-181`，包含 `children`、`parent`、`key`、`value`、`lock_ref`、`last_access_time`、`hit_count`、`host_value` 等字段。

`RadixCache.__init__()` 在 `radix_cache.py:285-334`，根据 `radix_eviction_policy` 选择策略。当前脚本 `lru` 对应 `LRUStrategy`，优先驱逐最久未访问的叶子，见 `evict_policy.py:16-18`。

`match_prefix()` 在 `radix_cache.py:374-444`：

```text
match_prefix(RadixKey(...))
  -> page 对齐截断
  -> _match_prefix_helper()
  -> 拼接命中 KV indices
  -> 返回 MatchResult
```

`cache_finished_req()` 会把完成请求 committed KV 插入 radix cache，见 `radix_cache.py:463-508`。`cache_unfinished_req()` 用于未完成请求和 chunked prefill，见 `radix_cache.py:510-573`。

`evict()` 在 `radix_cache.py:582-609`，从可驱逐叶子按策略释放 KV。`inc_lock_ref()` / `dec_lock_ref()` 用于保护正在被请求引用的 prefix，见 `radix_cache.py:611-645`。

## 13. KV 驱逐辅助

`evict_from_tree_cache()` 在 `python/sglang/srt/mem_cache/common.py:229-253`：

- allocator 可用空间不足时调用 `tree_cache.evict()`。
- SWA hybrid allocator 会分别检查 full / swa 空间。

常见调用点：

- `alloc_token_slots()`
- `alloc_paged_token_slots_extend()`
- `ScheduleBatch.check_decode_mem()`
- `ScheduleBatch.release_req()`

因此 prefill extend、decode step、decode retract 都会先尝试从 radix cache 驱逐可释放 KV。

## 14. 调度链路摘要

```text
TokenizedGenerateReqInput
  -> Scheduler.recv_requests()
  -> process_input_requests()
  -> handle_generate_request()
  -> _add_request_to_queue()
  -> waiting_queue

get_next_batch_to_run()
  -> SchedulePolicy(lpm).calc_priority()
     -> RadixCache.match_prefix(RadixKey(token_ids, extra_key))
  -> PrefillAdder.add_one_req()
  -> ScheduleBatch.prepare_for_extend()
     -> alloc_for_extend()
     -> evict_from_tree_cache()
  -> Scheduler.run_batch()
  -> process_batch_result_prefill()
     -> cache_unfinished_req() or release_kv_cache()

decode:
  -> update_running_batch()
  -> check_decode_mem()
  -> retract_decode() if KV 不足
  -> prepare_for_decode()
  -> run_batch()
  -> process_batch_result_decode()
```

## 15. 对当前 Agent 配置的结论

- `context_length=262144` 打开 256K 级上下文。
- 自动 `MAX_RUNNING_REQUESTS` 以整段 context length 为分母，长上下文会显著压低并发。
- `lpm` 优先复用共享 prefix KV cache。
- `radix lru` 在 KV 紧张时驱逐最近最少使用的可驱逐 prefix。
- `max_prefill_tokens=16384` 与 `chunked_prefill_size=8192` 限制超长 prompt 的单轮 prefill 峰值。
- decode 阶段 KV 不足时先驱逐 radix cache，仍不足才 retract 部分 running 请求回队列。
