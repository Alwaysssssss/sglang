# `python/sglang/srt/utils` 模块分析

## 定位

`utils` 是 SRT 的通用工具层，覆盖平台检测、网络/端口/ZMQ、日志、auth、profile、offload、watchdog、NUMA、模型文件校验、torch/tokenizer patch、视频解码、MLX tensor bridge、CUDA IPC transport、request logging、memory saver、慢 rank 检测等。

## 关键文件

- `common.py`：最大型的通用工具集合，包含设备/平台检测、包版本、序列化、进程控制、dtype/backend 判断、JSON/list 解析、CUDA/ROCm/NPU/CPU helper 等。
- `network.py`：端口探测、bind、ZMQ socket、local IP、`NetworkAddress`。
- `auth.py`：API/admin key 中间件和 auth level 判定。
- `log_utils.py`、`request_logger.py`、`json_response.py`、`http_middleware_patch.py`：日志、请求日志和 HTTP 响应工具。
- `profile_utils.py`、`profile_merger.py`、`rpd_utils.py`、`nvtx_pytorch_hooks.py`、`device_timer.py`：profiling 和计时。
- `offloader.py`：参数/模块 offload，支持 CPU/meta/shared buffer 等。
- `watchdog.py`：进程和子进程 watchdog。
- `torch_memory_saver_adapter.py`：按内存类型标记显存区域。
- `patch_torch.py`、`patch_tokenizer.py`：运行时 monkey patch。
- `numa_utils.py`、`multi_stream_utils.py`、`poll_based_barrier.py`：系统和并发辅助。
- `model_file_verifier.py`、`weight_checker.py`：模型文件/权重校验。
- `tensor_bridge.py`：torch/MLX tensor 转换。
- `video_decoder.py`：视频解码封装。

## 运行流程

多数模块在初始化时读取 `utils.common` 的平台能力，选择 backend、设备、dtype 和内存策略。`entrypoints` 使用 network/auth/log/watchdog；`model_executor` 使用 profile/offload/patch/memory saver；`model_loader` 使用 file verifier 和 HF 工具；`managers` 使用 request logger、barrier、network 和进程控制。

## 依赖关系

`utils` 被几乎所有 SRT 模块依赖，是基础层。它同时依赖 torch、psutil、zmq、FastAPI/Starlette、平台库和可选 profiling 库。

## 设计要点和风险

- `common.py` 功能很宽，新增工具前应确认是否会造成循环 import。
- monkey patch 和全局 offloader/watchdog 状态会影响整个进程，必须可控、可恢复或只在启动期调用。
- 网络端口探测和 bind 存在 TOCTOU，生产部署应优先显式分配端口。
- profile/offload/debug 工具可能改变性能和内存行为，不应默认开启。
