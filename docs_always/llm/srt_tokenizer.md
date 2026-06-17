# `python/sglang/srt/tokenizer` 模块分析

## 定位

`tokenizer` 目前主要提供 tiktoken tokenizer 兼容包装，使 SRT 能以统一 tokenizer 接口处理 tiktoken 模型。

## 关键文件

- `tiktoken_tokenizer.py`：`TiktokenTokenizer` 和 `TiktokenProcessor`。

## 运行流程

当 `utils.hf_transformers_utils.get_tokenizer` 遇到 tiktoken JSON 路径时，会读取 tokenizer JSON，构造 `tiktoken.Encoding`，并包装为 `TiktokenTokenizer`。该 wrapper 暴露 encode/decode、batch decode、`__call__`、`apply_chat_template`、特殊 token 和 processor 接口，使其能接入 SRT 的模板渲染、tokenized request、xgrammar tokenizer info 和 detokenizer 流程。

## 依赖关系

该模块被 `utils.hf_transformers_utils`、`TokenizerManager` 或 tokenizer 初始化路径间接使用，依赖 tiktoken 包和模型 tokenizer 配置。

## 设计要点和风险

- tiktoken 与 transformers tokenizer 在 special tokens、offset、decode 行为上可能不同，stop token 和流式 detokenize 要重点验证。
- 当前入口主要识别 `.json` tokenizer 路径；默认 special/control token 映射存在硬编码。
- xgrammar 对特殊 token 有额外替换逻辑，结构化输出场景要单独验证。
- wrapper 应尽量模拟 SRT 需要的 tokenizer 最小接口，避免让调用方依赖 tiktoken 私有细节。
