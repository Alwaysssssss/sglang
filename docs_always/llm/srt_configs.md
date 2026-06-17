# `python/sglang/srt/configs` 模块分析

## 定位

`configs` 承接 HuggingFace config 与 SRT runtime config 的转换。它既定义大量模型专用 `PretrainedConfig` 扩展，也提供 `ModelConfig`、`LoadConfig`、`DeviceConfig`、量化/modelopt 配置、Mamba/Kimi linear state 参数和运行前配置修正逻辑。

## 关键文件

- `model_config.py`：核心 `ModelConfig`。读取 HF config/generation config，判断模型类型、多模态、encoder-decoder、sliding window、NSA、dtype、context length、量化和 transformers 版本。
- `load_config.py`：`LoadFormat` 与 `LoadConfig`，描述权重加载格式和附加配置。
- `device_config.py`：设备配置封装。
- `update_config.py`：针对 CPU/TP、MoE padding、head padding 等场景修改 config。
- `mamba_utils.py`：Mamba/Kimi linear state shape/dtype/cache 参数。
- `modelopt_config.py`：ModelOpt 量化配置。
- 各模型文件：`qwen3_vl.py`、`qwen3_next.py`、`deepseekvl2.py`、`janus_pro.py`、`internvl.py`、`kimi_vl.py`、`step3_vl.py` 等，为 transformers 不完全支持或 SRT 需要特殊字段的模型提供 config/processor。
- `utils.py`：注册 image processor / processor。

## 运行流程

`ModelConfig.from_server_args` 从 `ServerArgs` 取模型路径、revision、dtype、quantization、override args 等，读取 HF config 并派生 SRT 运行所需字段。之后 `model_loader` 用它选择模型架构和 loader，`ModelRunner` 用它创建 KV cache、attention backend、sampler 和模型执行路径。

## 依赖关系

`configs` 依赖 transformers、`utils.hf_transformers_utils`、`layers.quantization`、`server_args`、`environ`。它向下影响 `model_loader`、`model_executor`、`layers.attention`、`mem_cache`、`multimodal`。

## 设计要点和风险

- `ModelConfig` 是模型行为的派生事实来源；同一个 HF 字段可能影响多模态、cache、attention backend、dtype、context length 多条路径。
- 模型专用 config 常常包含 processor 代码，升级 transformers 后要检查字段兼容性。
- `update_config.py` 修改 config 结构以适配硬件/TP padding，可能影响权重加载 shape；必须和 loader 的分片逻辑配套。
