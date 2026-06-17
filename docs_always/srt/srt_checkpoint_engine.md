# `python/sglang/srt/checkpoint_engine` 模块分析

## 定位

`checkpoint_engine` 对接外部 `checkpoint-engine` 包，用于运行中的 SGLang worker 通过 IPC/ZMQ 接收权重更新。它服务于在线训练/权重热更新等场景。

## 关键文件

- `checkpoint_engine_worker.py`：定义 `SGLangCheckpointEngineWorkerExtension` 和实现类，封装 checkpoint-engine worker extension。
- `update.py`：外部更新脚本入口，负责加载或切分 safetensors，启动 `ParameterServer`，等待 SGLang 服务并触发 `/update_weights_from_ipc`。
- `__init__.py`：包入口。

## 运行流程

外部进程通常通过 `torchrun python -m sglang.srt.checkpoint_engine.update` 启动。脚本读取 checkpoint，按 rank/inference parallel size 准备参数并注册到 `ParameterServer`，然后轮询 SGLang `/ping`，再通过 HTTP POST 触发服务端的 IPC 权重更新。SGLang 侧 `ModelRunner.update_weights_from_ipc` 创建 worker extension，根据当前 GPU UUID 查找 ZMQ handle，从 checkpoint-engine 拉取权重并写入模型参数，最后执行量化 post hook。

## 依赖关系

该模块依赖外部 `checkpoint_engine`、ZMQ、HTTPX、torch.distributed、safetensors。它被 HTTP 权重更新接口和 `model_executor/model_runner.py` 的 update weight 路径触发。

## 设计要点和风险

- 当前实现假设 CUDA GPU UUID 可用，非 CUDA/NPU/CPU 后端不一定适用。
- `checkpoint_engine_worker.py` 在 import 时依赖外部包，缺包会直接失败。
- 量化 post hook 的异常目前偏 warning 路径，可能导致权重已更新但量化后处理未完整完成。
- 外部脚本的 rank 切分必须和 SGLang 服务端并行配置一致，否则容易出现 silent mismatch 或 hang。
