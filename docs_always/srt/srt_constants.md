# `python/sglang/srt/constants.py` 模块分析

## 定位

`constants.py` 是 SRT 的极小全局常量模块。它目前集中定义 GPU 内存用途标签和健康检查请求 id 前缀，供 memory saver、scheduler、detokenizer 和健康检查路径共享。

## 关键内容

- `GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"`：KV cache 显存区域标签。
- `GPU_MEMORY_TYPE_WEIGHTS = "weights"`：模型权重显存区域标签。
- `GPU_MEMORY_TYPE_CUDA_GRAPH = "cuda_graph"`：CUDA graph capture/replay buffer 显存区域标签。
- `GPU_MEMORY_ALL_TYPES`：上述三类的全集。
- `HEALTH_CHECK_RID_PREFIX = "HEALTH_CHECK"`：健康检查请求的 rid 前缀。

## 使用方式

内存标签主要被 `TorchMemorySaverAdapter` 和 `model_executor`/`mem_cache` 使用，用于把权重、KV cache、CUDA graph 的显存占用分区标记。健康检查前缀被 `scheduler` 和 `detokenizer_manager` 识别，用来把探活请求与普通生成请求区分。

## 设计要点和风险

该模块简单，但常量是跨模块协议。修改字符串值会影响日志、内存统计、health check 识别和已有外部脚本；新增内存类型时要同步 `GPU_MEMORY_ALL_TYPES` 以及所有统计/过滤逻辑。
