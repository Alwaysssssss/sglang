# `python/sglang/srt/function_call` 模块分析

## 定位

`function_call` 负责把模型输出解析成 tool/function call，支持多种模型家族的 tool 调用格式、流式增量解析、JSON array 解析、schema 约束生成和 OpenAI-style tool 结构。

## 关键文件

- `function_call_parser.py`：`FunctionCallParser`，按 parser 名称选择 detector。
- `base_format_detector.py`：`BaseFormatDetector`，格式检测和流式解析抽象。
- `core_types.py`：`ToolCallItem`、`StreamingParseResult`、`StructureInfo`。
- 各 detector：`qwen25_detector.py`、`qwen3_coder_detector.py`、`deepseekv3_detector.py`、`deepseekv31_detector.py`、`deepseekv32_detector.py`、`hermes_detector.py`、`mistral_detector.py`、`glm4_moe_detector.py`、`glm47_moe_detector.py`、`gpt_oss_detector.py`、`kimik2_detector.py`、`mimo_detector.py`、`step3_detector.py` 等。
- `json_array_parser.py`：JSON array tool call parser。
- `utils.py`：partial JSON 解析、schema defs、schema constraint 生成和类型推断。

## 运行流程

入口层根据 `ServerArgs.tool_call_parser` 创建 `FunctionCallParser`，再选择具体 detector。非流式路径通常先 `has_tool_call`，再 `detect_and_parse` 得到普通文本和 calls；流式路径通过 `parse_streaming_increment` 维护状态，返回普通文本增量或 tool call 增量。对于结构化 tag 或 schema 场景，utils 可生成 grammar/json schema 约束辅助受限解码。

## 依赖关系

该模块被 OpenAI serving、parser/reasoning、`server_args` 和 `constrained` 间接使用。它依赖 `managers.io_struct.Tool/Function` 类型和 partial JSON 解析库。

## 设计要点和风险

- 不同模型 tool 格式差异很大，detector 状态机应隔离，避免一个模型的容错污染另一个模型。
- 流式解析必须处理 token 边界切断 JSON 或特殊 tag 的情况。
- partial JSON、结束标签跨 chunk、tool index、未知工具转发都需要在“尽量恢复”和“不要误报 tool call”之间取平衡。
- tool schema 约束和最终 parser 要保持一致，否则模型可能被约束生成 parser 不接受的格式。
- `structural_tag` 更适合 JSON 类格式，不能假定所有模型 tool 格式都能被同一种约束表达。
