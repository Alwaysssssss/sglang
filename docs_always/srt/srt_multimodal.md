# `python/sglang/srt/multimodal` 模块分析

## 定位

`multimodal` 处理 LLM/VLM 的图像、视频、音频等输入，提供 processor、预处理工具、视觉 encoder CUDA graph runner、InternVL/Qwen/Llava/Pixtral 等模型适配，以及 EVS token 压缩模块。

## 关键文件与子包

- `processors/base_processor.py`：`BaseMultimodalProcessor`、`BaseMultiModalProcessorOutput`、特殊 token 定义。
- `processors/`：各模型族 processor，如 Qwen VL、Llava、InternVL、Mllama、Phi4MM、Whisper、DeepSeek VL/OCR、Gemma3、Kimi VL、Pixtral 等。
- `mm_utils.py`：图片 resize/pad/patch、base64 加载、anyres 处理、DP sharded vision model 等通用工具。
- `customized_mm_processor_utils.py`：自定义 processor 注册。
- `vit_cuda_graph_runner.py`、`internvl_vit_cuda_graph_runner.py`：视觉 encoder 的 CUDA graph runner。
- `internvl_utils.py`：InternVL 动态预处理。
- `evs/`：EVS token retention、processor 和模块实现。

## 运行流程

入口请求携带多模态数据后，`TokenizerManager` 或 scheduler 侧的 multimodal processor 根据模型 config 构造图像/视频/音频 tensor、grid、bounds 和特殊 token 信息。模型 forward 前，`managers.mm_utils` 把多模态 embedding 插入文本 token 对应位置，必要时使用共享内存或 encoder disaggregation。VLM 模型文件再消费这些 embedding、position 或 mrope 信息。

## 依赖关系

`multimodal` 被 `managers.tokenizer_manager`、`managers.mm_utils`、`models`、`disaggregation.encode_*` 和 `configs` 使用。它依赖 PIL/torch/transformers processor、模型专用 config 和 CUDA graph runner。

## 设计要点和风险

- 多模态 token 对齐是主要风险：processor 输出长度、special token、grid、mrope position、embedding mask 必须一致。
- 不同模型族的 resize/patch 规则差异很大，不能用通用 processor 盲目替代。
- 视觉 CUDA graph 对输入 shape 更敏感，动态分辨率/动态 batch 需要走 fallback 或预设 capture size。
- encoder disaggregation/共享内存路径会改变 tensor 生命周期，必须避免提前释放或重复拷贝。
