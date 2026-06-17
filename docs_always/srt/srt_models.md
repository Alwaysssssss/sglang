# `python/sglang/srt/models` 模块分析

## 定位

`models` 存放 SRT 原生模型实现。每个模型文件通常把 HuggingFace config 映射到 SRT 的高性能层：TP linear、RadixAttention、MoE runner、量化参数、pooler、视觉 encoder、多模态 projector、MTP/EAGLE/head 等。它是 `model_loader` 解析模型架构后的实际实例化目标。

## 关键结构

- `registry.py`：模型架构注册与查找入口。
- `utils.py`：模型通用工具，包括权重加载辅助、embedding/pooler/position 等公共逻辑。
- `llama.py`、`qwen2.py`、`qwen3.py`、`deepseek_v2.py`、`deepseek.py` 等：主流 decoder-only 架构的代表实现。
- `deepseek_common/`：DeepSeek MLA/MHA attention forward method、attention backend handler 和权重 loader。
- 多模态模型：`qwen2_vl.py`、`qwen3_vl.py`、`mllama.py`、`mllama4.py`、`internvl.py`、`kimi_vl.py`、`dots_vlm.py`、`glm4v.py`、`paddleocr_vl.py` 等。
- MoE/混合模型：`mixtral.py`、`deepseek_v2.py`、`qwen3_moe.py`、`bailing_moe.py`、`nemotron_h.py`、`longcat_flash.py`、`minimax_m2.py` 等。
- draft/MTP/reward/classification/embedding 变体：`*_mtp.py`、`*_eagle.py`、`*_reward.py`、`*_classification.py`、`llama_embedding.py`。

## 运行流程

`model_loader.utils.get_model_architecture` 根据 `ModelConfig` 的 architecture 字段选择一个模型类。模型类构造时创建 embedding、decoder layer、norm、LM head/pooler 等组件。forward 接收 `input_ids`、`positions`、`ForwardBatch` 等参数，逐层调用 attention/MLP/MoE，并把 logits 或 hidden states 交给 `LogitsProcessor` / pooler。权重加载通常由模型类的 `load_weights` 实现，内部把 HF tensor 名称映射到 SRT 参数名和 TP 分片。

## 依赖关系

`models` 强依赖 `layers`、`model_executor.forward_batch_info`、`model_loader.weight_utils`、`configs`、`distributed` 进程组和 `multimodal` processor。它向上由 `model_loader` 和 `ModelRunner` 调用，向旁侧支持 `speculative`、`lora`、`eplb`、`sampling` 等能力。

## 设计要点和风险

- 模型文件数量很大，但共同模式稳定：`Attention` + `MLP/MoE` + `DecoderLayer` + `Model` + `ForCausalLM/ConditionalGeneration` + `load_weights`。
- 模型 forward 必须遵守 `ForwardBatch` 和 attention backend 的契约，例如 cache 写入、positions、extend/decode 模式、多模态 embedding 插入。
- 新模型最容易出错的部分是权重名映射、TP 切分、QKV 合并/拆分、MoE expert layout、rope/position scaling、多模态 token 对齐。
- 多模型复用父类能减少重复，但也隐藏配置差异；继承 Llama/DeepSeek/Qwen 路径时要确认特殊字段不会被父类默认逻辑吞掉。
