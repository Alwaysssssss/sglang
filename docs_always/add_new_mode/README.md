# 在 SGLang 中新增模型（LLM / VLM / DiT）

本目录提供**三篇详细的中文文档**，覆盖在 SGLang 中接入新模型的三条主要路径。每一篇都可以独立使用，但三者之间存在依赖：VLM 建立在 LLM 的基础之上；DiT / 扩散模型则走完全独立的运行时。

---

## 三篇文档

| # | 文档 | 适用场景 | 所在运行时 |
|---|------|---------|-----------|
| 1 | [01_add_llm.md](./01_add_llm.md) | 新增 **纯文本 LLM**（Causal LM / Seq2Seq / Reward / Embedding / Classification）、MoE、MLA、线性注意力、Speculative / MTP | `sglang.srt` |
| 2 | [02_add_vlm.md](./02_add_vlm.md) | 新增 **多模态 LLM**：图像理解 / 视频理解 / 音频理解 / 全模态融合；ViT + LLM、M-RoPE 等 | `sglang.srt` + `srt/multimodal` |
| 3 | [03_add_dit.md](./03_add_dit.md) | 新增 **扩散模型**：Text→Image、Text→Video、Image→Image、Image→Video、3D 等；DiT/UNet + VAE + TextEncoder + Scheduler | `sglang.multimodal_gen` |

---

## 选择指南

### 如果你在问自己 "我要新增的模型属于哪一类？"

```
┌─ 是扩散模型（输出是图像 / 视频 / 3D mesh，核心是多步去噪）？
│  └─ 是 → 读 03_add_dit.md
│
├─ 是生成文本的模型（Causal LM / Reward / Embedding / Classification）？
│  ├─ 带图像 / 视频 / 音频输入？
│  │  └─ 是 → 读 02_add_vlm.md（会引用 01）
│  └─ 纯文本 → 读 01_add_llm.md
│
└─ 其他（例如纯语音合成 TTS、音频生成）
   └─ 多数仍在 sglang.multimodal_gen 范畴，参考 03_add_dit.md
```

### 工作量预期

| 场景 | 预计改动规模 |
|------|-----------|
| 基于已有 LLM 的小改（继承 Llama 后加 QK Norm / 换 RoPE） | 1 个新文件（200~500 行），1~3 小时 |
| 从 vLLM 移植一个 LLM | 1 个新文件（500~1500 行），半天~1 天 |
| 新增一个全新的 VLM | 3~5 个新文件（LLM + ViT + Processor + 多模态白名单登记），1~3 天 |
| 新增一个全新的 DiT | 5~10 个新文件（DiT 模型 / 配置 / SamplingParams / PipelineConfig / BeforeDenoisingStage / Pipeline / registry 登记），3~7 天（不含精度对齐） |

---

## 三条路径的相同原则

1. **找参考先于写代码**：SGLang 仓库里几乎已有每种架构的近似实现，先读源码再动手。
2. **能继承就不要复制**：差异用子类 / 回调表达，避免从空白开始。
3. **对齐是关键**：LLM 要对齐 HF Transformers 的 prefill logits；VLM 要对齐占位符展开与 ViT token 数；DiT 要与 Diffusers **逐张量**对齐。
4. **测试必须实跑**：PR 里附上 GSM8K / MMLU / MMMU / Component-Accuracy 结果，而不是"跑通了"三个字。
5. **不写死，要可组合**：TP / SP / 量化 / LoRA 尽量走框架提供的开关式能力，而不是在模型文件里硬编码。

---

## 相关文档

- 仓库整体目录解读：[`docs_zh/path.md`](../path.md)
- 扩散模型架构系列（深入运行时原理）：[`docs_zh/multimodal_gen/`](../multimodal_gen/)
- CI 测试编排：[`test/README.md`](../../test/README.md)
- 官方英文支持文档：
  - [`docs/supported_models/extending/support_new_models.md`](../../docs/supported_models/extending/support_new_models.md)
  - [`docs/diffusion/support_new_models.md`](../../docs/diffusion/support_new_models.md)
