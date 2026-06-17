# `python/sglang/srt/parser` 模块分析

## 定位

`parser` 负责文本协议和模板解析，包括 chat conversation 模板、code completion FIM 模板、Jinja chat template 内容格式检测、Harmony 消息解析和 reasoning 内容拆分。

## 关键文件

- `conversation.py`：`Conversation`、`SeparatorStyle`、模板注册和按模型路径匹配模板。
- `code_completion_parser.py`：FIM/code completion 模板注册和 prompt 生成。
- `jinja_template_utils.py`：检测 Jinja chat template 是否使用 string/openai/multimodal content 格式，并处理消息 content。
- `harmony_parser.py`：Harmony token/event parser。
- `reasoning_parser.py`：`ReasoningParser` 和 DeepSeek/Qwen/Kimi/GLM/GPT-OSS/MiniMax/Nemotron/Mistral 等 reasoning detector。

## 运行流程

`TemplateManager` 和 entrypoints 根据模型路径、显式 chat template 或 completion template 选择 parser。普通 chat 请求先规范化文本/多模态 content，再通过 conversation 或 HF chat template 渲染为 prompt；code completion 根据 FIM 模板插入 prefix/suffix；reasoning parser 在输出阶段按模型 reasoning token/tag 把 reasoning text 与 final answer 分离；Harmony parser 则解析 Harmony channel、事件流、assistant action 和剩余状态。

## 依赖关系

`parser` 被 `entrypoints`、`TokenizerManager`、OpenAI serving、`server_args` 和 `function_call` 使用。它依赖 tokenizer/HF chat template、OpenAI protocol 消息结构和模型命名约定。

## 设计要点和风险

- 模板匹配顺序会影响默认 prompt，新增模板需避免抢占其他模型。
- Jinja AST 检测只是启发式，复杂模板可能误判 content 格式。
- reasoning 流式拆分要处理 tag 被 token 边界切开的情况。
- prompt 模板变更会影响缓存命中、stop token、tool call 和输出格式。
- 多模态占位符数量、`continue_final_message`、reasoning/tool token 交叉时边界复杂，入口和 detokenizer 要保持一致。
