# `python/sglang/srt/managers` 模块分析

## 定位

`managers` 是 SRT 的运行时控制核心。它定义请求/响应数据结构，负责 tokenizer 与 detokenizer 进程，维护 scheduler 主循环，把请求组织成可执行 batch，并把 LoRA、grammar、speculative、disaggregation、DP/PP attention、多模态、session、profile、权重更新等能力接入同一调度循环。

## 关键文件

- `io_struct.py`：内部 IPC 的请求/响应类型中心，包含生成、embedding、权重更新、LoRA、profile、cache、健康检查、load metrics、tool parsing 等 dataclass/Pydantic 风格结构。
- `tokenizer_manager.py`：接收上层请求，完成 tokenizer、模板、多模态预处理、请求分发和输出收集。
- `detokenizer_manager.py`：把 token 流增量 detokenize，处理 finish reason、logprobs、skip special tokens、健康检查等。
- `scheduler.py`：最核心的调度进程。持有 waiting/running batch、cache、grammar、LoRA、worker、metrics，并在 event loop 中选择 prefill/decode/extend。
- `schedule_batch.py`：定义 `Req`、`ScheduleBatch`、`ModelWorkerBatch` 和 finish reason，是 scheduler 与 model executor 的批状态边界。
- `schedule_policy.py`：cache-aware/cache-agnostic 调度策略和 `PrefillAdder`。
- `tp_worker.py`：`TpModelWorker` 包装 `ModelRunner`，暴露 forward、profile、权重更新、cache 操作等 worker 能力。
- `data_parallel_controller.py`：DP rank 的负载汇总、路由、限流和负载均衡。
- `scheduler_*_mixin.py`：把 DP attention、PP、profile、output processing、runtime checker、update weights 等横向能力混入 `Scheduler`。
- `mm_utils.py`、`multimodal_processor.py`：多模态 embedding、共享内存传输和 processor 注册。

## 运行流程

入口层先把请求转换成 `GenerateReqInput` 或 tokenized 结构。`TokenizerManager` 负责 tokenizer、chat template、多模态 processor 和 DP/多 tokenizer 路由，然后把 `TokenizedGenerateReqInput` 等对象发给 scheduler。`Scheduler` 把请求转换为 `Req`，根据 prefix cache 命中、预算、grammar、batch 状态、chunked prefill 策略和 speculative 状态，形成 `ScheduleBatch`。`TpModelWorker` 将 batch 转成 worker 可执行形式并调用 `ModelRunner.forward`。输出回到 scheduler 后，经 `SchedulerOutputProcessorMixin` 处理采样结果、finish、cache release、metrics，再发给 detokenizer。

## 依赖关系

`managers` 向下依赖 `model_executor`、`mem_cache`、`sampling`、`constrained`、`disaggregation`、`distributed`、`lora`、`multimodal`、`observability` 和 `utils`。它向上为 `entrypoints` 暴露稳定的请求/响应边界。由于 scheduler 同时读写大部分运行时状态，该模块是 SRT 中耦合最密集的部分。

## 设计要点和风险

- `Req` 和 `ScheduleBatch` 是调度状态的真实来源；修改字段要同步 tokenizer、scheduler、worker、detokenizer 多处语义。
- scheduler event loop 中 prefill、decode、retract、abort、pause、continue、profile、cache flush、LoRA 更新等路径共享状态，容易因边界条件产生泄漏或重复释放。
- chunked prefill、disaggregation、speculative、grammar 同时开启时会改变 cache loc、accepted length 和 finish 判定，测试矩阵需要覆盖组合。
- `io_struct.py` 类型很多，新增控制请求应经过 `_check_all_req_types` 一类校验，避免 IPC 反序列化或 dispatcher 漏处理。
