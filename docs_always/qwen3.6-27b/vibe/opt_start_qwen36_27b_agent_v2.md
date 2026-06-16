# Feature: 实际agent在使用过程中，经常出现qwen-27b停止工作的情况，需要点击继续，qwen-27b才会继续运行

> 本文是实施方案，不包含实现代码。后续如执行，需要按本文范围修改脚本、测试和文档，并在真实 GPU 环境完成验收。

## 0. 需求归纳

- 检查当前docs_always/qwen3.6-27b/start_qwen36_27b_agent.sh的配置
- 发出请求的主要是 langchain 的 ChatOpenAI

找到具体的原因，并提出解决办法
