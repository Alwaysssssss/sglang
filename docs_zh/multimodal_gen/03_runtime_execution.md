# 运行时执行链路

## 1. 总体执行模型

`multimodal_gen` 的执行模型是“客户端 + scheduler + worker + pipeline”的结构：

- 客户端：`DiffGenerator` / HTTP route / CLI
- 调度端：`Scheduler`
- 执行端：`GPUWorker` / `CPUWorker`
- 具体逻辑：`Pipeline`

这意味着即使是本地离线调用，底层依然尽量复用服务化链路，而不是单独维护一套完全不同的执行路径。

## 2. `DiffGenerator`：统一用户入口

位置：`runtime/entrypoints/diffusion_generator.py`

`DiffGenerator` 是外部最常见的入口类。

### 2.1 `from_pretrained()`

它负责：

1. 把用户 kwargs 转成 `ServerArgs`
2. 调 `from_server_args()`

### 2.2 `from_server_args()`

这里会决定运行模式：

- `local_mode=True`
  - 本地直接拉起 scheduler/worker 进程
- `local_mode=False`
  - 连接远端已有 scheduler

### 2.3 `generate()`

`generate()` 的流程是：

1. 解析 prompt / prompt file；
2. 构造 `SamplingParams`；
3. 逐条 prompt 生成 `Req`；
4. 用 scheduler client 发给 scheduler；
5. 收到 `OutputBatch` 后封装 `GenerationResult`；
6. 在需要时保存文件、插帧、超分、返回路径或内存对象。

注意它现在是“逐请求发送”，注释里也写了 TODO，说明批量发送还不是主路径。

## 3. `prepare_request()`：请求对象成型

位置：`runtime/entrypoints/utils.py`

`prepare_request(server_args, sampling_params)` 负责把 `SamplingParams` 包成 `Req`，并补充：

- `VSA_sparsity`
- request extra 字段
- size 调整
- prompt/尺寸合法性检查

这是“从 API 参数跨入执行框架”的真正入口。

## 4. `launch_server()`：本地服务拉起

位置：`runtime/launch_server.py`

### 4.1 进程拓扑

`launch_server()` 会：

1. 按 GPU 数创建 master/slave 通信 pipe；
2. 为每个 GPU 启动一个 `run_scheduler_process` 子进程；
3. rank 0 作为主 worker；
4. 等待所有 worker 报 ready；
5. 可选启动 HTTP server。

这里的命名稍有迷惑：`run_scheduler_process()` 启动的不是独立“纯 scheduler 进程”，而是每个 worker 进程内部各自构造 `Scheduler`；其中 rank 0 的 `Scheduler` 对外接收请求。

### 4.2 本地离线模式

`DiffGenerator` 在本地模式下调用 `launch_server(..., launch_http_server=False)`，因此只启动内部 worker/scheduler，不起 FastAPI。

## 5. `Scheduler`：rank 0 的事件循环核心

位置：`runtime/managers/scheduler.py`

`Scheduler` 负责：

- 创建 ZMQ ROUTER socket；
- 初始化 `GPUWorker` 或 `CPUWorker`；
- 维护 `request_handlers`；
- 处理 warmup；
- 维护等待队列；
- 将结果回发给客户端。

### 5.1 请求类型

它不仅处理生成请求 `Req` / `List[Req]`，还处理：

- `SetLoraReq`
- `MergeLoraWeightsReq`
- `UnmergeLoraWeightsReq`
- `ListLorasReq`
- `ShutdownReq`
- RL/post-training 相关权重更新请求
- disagg stats 请求

所以 scheduler 不只是“转发生成”，也承担控制平面职责。

### 5.2 warmup 机制

`prepare_server_warmup_reqs()` 会在启动时插入 warmup 请求。它会根据任务类型判断是否需要构造最小输入图像，并把 warmup request 提前塞进等待队列。

这说明 warmup 被视作正式执行链的一部分，而不是独立脚本。

## 6. `GPUWorker`：真实执行者

位置：`runtime/managers/gpu_worker.py`

### 6.1 初始化职责

`init_device_and_model()` 会做很多事：

- 设定当前 device；
- 设置 `MASTER_ADDR` / `RANK` / `WORLD_SIZE` 等环境变量；
- 初始化 distributed / model parallel；
- 设置进程标题；
- `build_pipeline(server_args)`；
- 按需启用 layerwise offload。

也就是说，worker 初始化已经把“设备、并行环境、模型、pipeline”全部准备好了。

### 6.2 `execute_forward()`

核心流程：

1. 重置峰值显存统计；
2. 记录性能/内存基线；
3. `req.log()` 打印本次请求信息；
4. `self.pipeline.forward(req, server_args)`；
5. 将 `Req` 或结果包装为 `OutputBatch`；
6. 按需保存输出文件并只返回路径；
7. 输出性能与显存统计；
8. 异常时统一封装错误信息。

### 6.3 worker 还承担运行中控制操作

`GPUWorker` 还暴露：

- LoRA 设置/合并/反合并/列举；
- 从磁盘原位更新权重；
- 权重 checksum 计算。

这说明 worker 持有的 pipeline 是“可动态修改”的，而不是只读。

## 7. `build_pipeline()`：选 Pipeline 并实例化

位置：`runtime/pipelines_core/__init__.py`

逻辑很直接：

1. 如果用户显式指定 `pipeline_class_name`，直接从 `_PIPELINE_REGISTRY` 查；
2. 否则让 `registry.get_model_info()` 决定；
3. 用 `pipeline_cls(model_path, server_args)` 实例化。

所以 Pipeline 的选择权既可以自动，也可以手工强制指定。

## 8. Executor：stage 的执行者

### 8.1 `PipelineExecutor`

位置：`runtime/pipelines_core/executors/pipeline_executor.py`

它定义统一接口：

- `execute()`
- `execute_with_profiling()`

profile 逻辑被放在 executor 层，而不是散落在每个 stage 里。

### 8.2 `ParallelExecutor`

默认 executor 是 `ParallelExecutor`。它根据 stage 声明的 `parallelism_type` 决定执行方式：

- `REPLICATED`
- `MAIN_RANK_ONLY`
- `CFG_PARALLEL`

这让“stage 逻辑”和“分布式执行语义”解耦。

## 9. `Req` 与 `OutputBatch` 的关系

- `Req`：中间态，可不断被 stage 改写；
- `OutputBatch`：返回态，更接近服务接口输出。

`GPUWorker.execute_forward()` 支持 Pipeline 直接返回 `Req`，再由 worker 统一转换为 `OutputBatch`。这可以减少各个 stage/pipe 的重复包装代码。

## 10. 典型离线调用时序

```text
DiffGenerator.generate
  -> SamplingParams.from_user_sampling_params_args
  -> prepare_request
  -> sync_scheduler_client.forward
  -> Scheduler._handle_generation
  -> GPUWorker.execute_forward
  -> Pipeline.forward
  -> Executor.execute
  -> Stage1 -> Stage2 -> ... -> StageN
  -> OutputBatch
  -> save_outputs / GenerationResult
```

## 11. 这套执行模型的优点

### 11.1 API 与服务复用同一执行内核

无论来自 Python API、CLI 还是 HTTP，请求最后都尽量落到统一的 `Req -> Scheduler -> Worker -> Pipeline` 链路上。

### 11.2 扩展点集中

想新增模型或功能，通常不需要改 scheduler/worker 本身，而是：

- 加新的 config 注册；
- 加新的 pipeline；
- 必要时加新的 stage 或 loader。

### 11.3 服务化能力天然存在

因为内部默认就是 scheduler/worker 结构，所以 HTTP/OpenAI API 只是外层封装，不需要重写推理主循环。

