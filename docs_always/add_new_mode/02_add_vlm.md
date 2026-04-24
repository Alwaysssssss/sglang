# 在 SGLang 中新增一个 VLM（多模态 LLM）

> 目标读者：希望在 `sglang/srt` 中接入带图像 / 视频 / 音频输入的 **多模态 LLM**（MLLM / VLM），例如 Qwen2-VL、LLaVA、InternVL、MiniCPM-V、Gemma3-VL、Pixtral、Kimi-VL 等。
>
> 相关参考：
> - 视觉/音频模型目录：[`python/sglang/srt/models/`](../../python/sglang/srt/models/)
> - 多模态处理器目录：[`python/sglang/srt/multimodal/processors/`](../../python/sglang/srt/multimodal/processors/)
> - MM 工具函数：[`python/sglang/srt/managers/mm_utils.py`](../../python/sglang/srt/managers/mm_utils.py)
> - 多模态模型白名单：[`python/sglang/srt/configs/model_config.py`](../../python/sglang/srt/configs/model_config.py) 的 `multimodal_model_archs`
>
> **先决条件**：在读本文档之前，请先确认你已理解 [`01_add_llm.md`](./01_add_llm.md) 中关于 LLM 接入的全部内容——**VLM = LLM + 视觉/音频编码器 + Processor + 占位符展开**。

---

## 0. VLM 相较 LLM 多出来的五件事

SGLang 的 VLM 架构本质上是"语言模型 + 多模态编码器 + 数据处理器"。相较纯文本 LLM，必须额外做以下五件事：

| 多出的职责 | 对应代码位置 | 为什么必须 |
|---|---|---|
| **① 声明这个架构是多模态** | `srt/configs/model_config.py` 的 `multimodal_model_archs` 列表 | 调度器据此启用图像 / 音频 pipeline，准备 `mm_inputs`，开启 RadixAttention 的多模态 hash 机制 |
| **② 注册一个 Chat Template**（可选） | `srt/parser/conversation.py` | 默认 template 可能不支持 `<image>` / `<video>` 占位符 |
| **③ 写一个 Multimodal `Processor`** | `srt/multimodal/processors/{model}.py`，继承 `BaseMultimodalProcessor` | 把用户传的图像 / 视频 / 音频转成 `MultimodalDataItem`，供模型使用 |
| **④ 实现 `pad_input_ids`** | 模型类中 | 把 prompt 里的 `<image>` / `<video>` 等单占位符展开成足够多的占位 token，并用 multimodal-data-hash 填充，保证 RadixAttention 能区分不同图像 |
| **⑤ 实现 `get_image_feature` / `get_video_feature` / `get_audio_feature`** | 模型类中 | 将视觉编码器输出的特征投影成 LM embedding 维度，再由 `general_mm_embed_routine` 拼到 token embedding 里 |

另外，ViT 内部的多头 `Attention` 要用 SGLang 的 `VisionAttention` 替换（继承了 TP，并与框架 FP8/BF16 调度一致）。

---

## 1. 推荐工作流（十步）

### Step 1 — 研读参考实现

**选一个结构最接近的现成 VLM** 作为起点：

| 架构特征 | 推荐参考 |
|---|---|
| ViT + MLP Projector + LLM（最标准的 LLaVA 范式） | [`models/llava.py`](../../python/sglang/srt/models/llava.py) |
| ViT + Qwen2 LM（Qwen2-VL） | [`models/qwen2_vl.py`](../../python/sglang/srt/models/qwen2_vl.py)、[`models/qwen2_5_vl.py`](../../python/sglang/srt/models/qwen2_5_vl.py) |
| ViT + Qwen3 LM（Qwen3-VL / Qwen3-VL-MoE） | [`models/qwen3_vl.py`](../../python/sglang/srt/models/qwen3_vl.py) |
| 带 M-RoPE（多维度 rotary，视觉 token 需要 3D 位置） | 同上 Qwen2-VL 系列 |
| InternViT + LLM（InternVL） | [`models/internvl.py`](../../python/sglang/srt/models/internvl.py) |
| ViT + Gemma LM | [`models/gemma3_mm.py`](../../python/sglang/srt/models/gemma3_mm.py) |
| Pixtral（Mistral + 自研 ViT） | [`models/pixtral.py`](../../python/sglang/srt/models/pixtral.py) |
| DeepSeek VL / VL2 | [`models/deepseek_vl2.py`](../../python/sglang/srt/models/deepseek_vl2.py) |
| Kimi-VL（MoonViT + DeepSeek-like LM） | [`models/kimi_vl.py`](../../python/sglang/srt/models/kimi_vl.py) |
| 音频 + LM（Whisper / Voxtral / Qwen2-Audio / Qwen3-ASR） | [`models/qwen2_audio.py`](../../python/sglang/srt/models/qwen2_audio.py)、[`models/voxtral.py`](../../python/sglang/srt/models/voxtral.py) |
| 全能多模态（图 / 音 / 视频联合） | [`models/gemma4_mm.py`](../../python/sglang/srt/models/gemma4_mm.py)、[`models/qwen3_omni.py`](../../python/sglang/srt/models/qwen3_omni.py) |

**研读重点**：
1. HuggingFace 官方 `modeling_*.py` 里视觉 / 音频编码器的结构（重点是 `VisionConfig` 如何传入）
2. HuggingFace 官方 `processing_*.py` 里如何把 PIL.Image 切 patch、生成 `pixel_values` / `image_grid_thw`
3. `chat_template.jinja` 里图像 / 音频占位符的格式（例如 Qwen 是 `<|vision_start|><|image_pad|><|vision_end|>`）

### Step 2 — 在多模态白名单里登记

编辑 [`python/sglang/srt/configs/model_config.py`](../../python/sglang/srt/configs/model_config.py)，在 `multimodal_model_archs` 列表里加一行：

```python
multimodal_model_archs = [
    # ... existing ...
    "YourModelForConditionalGeneration",   # 与 config.json 的 architectures[0] 一致
]
```

以及（如果你的模型只做图像理解而不做生成）在 `is_generation_model()` 里做相应判断。

**该列表的作用**：被 `is_multimodal_model()` 查表使用，驱动 TokenizerManager 在前置 pipeline 里启用 `MMProcessor`、打开多模态 RadixAttention hash 模式等。漏登记会直接跳过多模态路径，走纯文本推理，出现"忽略图片、只理解 prompt 文字"的现象。

### Step 3 — 写 Processor（`srt/multimodal/processors/{model}.py`）

Processor 是**上行的数据管道**：接收用户 HTTP 请求里的 `image_data` / `video_data` / `audio_data`，预处理后以 `MultimodalDataItem` 形式挂到 `Req` 上。

最小模板：

```python
# python/sglang/srt/multimodal/processors/your_model.py

from sglang.srt.managers.schedule_batch import (
    Modality, MultimodalDataItem, MultimodalInputs,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor, MultimodalSpecialTokens,
)
from sglang.srt.models.your_model import YourModelForConditionalGeneration


class YourModelImageProcessor(BaseMultimodalProcessor):
    models = [YourModelForConditionalGeneration]   # 关联模型类

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        # 从 hf_config 读取特殊 token id
        self.IM_TOKEN_ID = hf_config.image_token_id
        self.IM_START_TOKEN_ID = hf_config.vision_start_token_id
        self.IM_END_TOKEN_ID = hf_config.vision_end_token_id

        # 构造 special tokens 结构（供框架识别）
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<image>",
            image_token_id=self.IM_TOKEN_ID,
        ).build(_processor)

    async def process_mm_data_async(
        self,
        image_data,         # List[PIL.Image | str | bytes]
        audio_data=None,
        input_text=None,    # 未展开占位符的 prompt
        request_obj=None,
        *args,
        **kwargs,
    ):
        # 1. 调用 HuggingFace 官方 processor 把 PIL.Image 变 pixel_values
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )
        inputs = self._processor(
            text=[base_output.input_text],
            images=base_output.images,
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]

        # 2. 构造 MultimodalDataItem
        items = [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        ]

        # 3. 返回 dict，scheduler 会挂到 Req 上
        return {
            "mm_items": items,
            "input_ids": inputs["input_ids"].tolist()[0],
            "im_token_id": self.IM_TOKEN_ID,
            "im_start_id": self.IM_START_TOKEN_ID,
            "im_end_id": self.IM_END_TOKEN_ID,
        }
```

**关键点**：
- `models` 字段是 **class 引用而不是字符串**，用来让 `ProcessorRegistry` 自动配对。
- 不要把巨大的 tensor 同步传输；必要时用 `CudaIpcTensorTransportProxy`（基类已封装）。
- 音频模型重写 `process_audio_async`；视频模型的处理可以复用图像路径或写 `process_video_async`。

### Step 4 — 配置 Chat Template（可选）

若模型的 `chat_template.jinja` 已随 HF checkpoint 下发，**什么都不用做**。

仅当需要 SGLang 自带 template、或需要一个匹配函数时，在 [`python/sglang/srt/parser/conversation.py`](../../python/sglang/srt/parser/conversation.py) 中添加：

```python
register_conv_template(
    Conversation(
        name="your-model",
        system_template="<|system|>{system_message}<|end|>",
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyle.XXX,
        sep="<|end|>",
        # 对图像 token 的处理规则
        image_token="<image>",
    )
)

@register_conv_template_matching_function
def match_your_model(model_path: str):
    if "your-model" in model_path.lower():
        return "your-model"
```

### Step 5 — 实现视觉 / 音频编码器

多数模型选择把编码器单独建一个类（如 `YourModelVisionTransformer`），放在同一个 `.py` 文件里或拆成 `your_model_vit.py` 单独放。

**用 SGLang 的组件**：

| HF 原生 | SGLang 替换 |
|---|---|
| `nn.MultiheadAttention` / 手写 QKV 的 ViT Attention | `sglang.srt.layers.attention.vision.VisionAttention` |
| `nn.Linear` 大模块 | `ColumnParallelLinear` / `RowParallelLinear` / `ReplicatedLinear` |
| `nn.LayerNorm` | `sglang.srt.layers.layernorm` 里的相应实现（多数 ViT 直接用 torch 原生 LN 也可） |
| 手写的旋转位置编码 | `get_rope()` 或模型自定义（Qwen-VL 的 M-RoPE 是典型例外） |
| `nn.Conv2d` patch embed | 保持 `nn.Conv2d`，或用 `Conv3dLayer`（时间维） |
| `nn.GELU` / `nn.SiLU` | `QuickGELU` 或原生（视觉场景性能差异小） |

**Pixel → Patch → Token** 流程示例（Qwen2-VL 风格）：

```python
class YourModelVisionTransformer(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.patch_embed = Conv3dLayer(
            in_channels=3, out_channels=config.hidden_size,
            kernel_size=(config.temporal_patch_size,
                         config.patch_size, config.patch_size),
            stride=(config.temporal_patch_size,
                    config.patch_size, config.patch_size),
            bias=False,
        )
        self.blocks = nn.ModuleList([
            YourModelVisionBlock(config, quant_config=quant_config,
                                 prefix=add_prefix(f"blocks.{i}", prefix))
            for i in range(config.depth)
        ])
        self.merger = YourModelPatchMerger(...)   # 把 ViT token 投到 LM hidden_size

    def forward(self, pixel_values, grid_thw):
        hidden_states = self.patch_embed(pixel_values)
        hidden_states = self._apply_rotary_pos_emb(hidden_states, grid_thw)
        for blk in self.blocks:
            hidden_states = blk(hidden_states, ...)
        return self.merger(hidden_states)
```

ViT 内部的 Attention：

```python
from sglang.srt.layers.attention.vision import VisionAttention

self.attn = VisionAttention(
    embed_dim=config.hidden_size,
    num_heads=config.num_heads,
    projection_size=config.hidden_size,
    use_qkv_parallel=True,
    quant_config=quant_config,
    prefix=add_prefix("attn", prefix),
)
```

> `VisionAttention` 会根据硬件选择合适的 backend（flash_attn / torch sdpa / triton），无须自己调度。

### Step 6 — 实现顶层 `YourModelForConditionalGeneration`

这是 `EntryClass` 指向的类，它把 ViT、LLM backbone、投影层、Logits 层拼在一起：

```python
class YourModelForConditionalGeneration(nn.Module):
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_prefix={
            # transformers v4.52+ 的新权重命名
            "model.language_model.": "language_model.model.",
            "model.visual.":         "visual.",
            # 兼容老版本
            "lm_head.":              "language_model.lm_head.",
            "model.":                "language_model.model.",
        },
    )

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config

        # 1. ViT
        self.visual = YourModelVisionTransformer(
            config.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("visual", prefix),
        )

        # 2. Language model backbone —— 直接复用已实现的 LLM
        #    e.g. Qwen2Model / LlamaModel / DeepseekV2Model
        self.language_model = YourBackboneLM(config, quant_config, ...)
        self.model = self.language_model.model  # 为兼容 general_mm_embed_routine

        # 3. LM head
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size, config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    # ---- 占位符展开：方法 ④ ----
    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        # 两种常见 pattern：
        #  (a) 单 token 重复：<image><image>...<image>（Qwen / Gemma）
        #  (b) 成对 start/end：<img_start>...<img_end>（LLaVA / InternVL）
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    # ---- 特征提取：方法 ⑤ ----
    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        pixel_values = torch.cat(
            [item.feature for item in items], dim=0
        ).type(self.visual.dtype)
        image_grid_thw = torch.concat(
            [item.image_grid_thw for item in items], dim=0,
        )
        return self.visual(pixel_values, grid_thw=image_grid_thw)

    # 若支持视频
    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        ...

    # 若支持音频
    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        ...

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def should_apply_lora(self, module_name: str) -> bool:
        return not module_name.startswith("visual")   # 跳过视觉塔

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds=None,
        get_embedding: bool = False,
    ):
        # 如果模型用 M-RoPE（3 维位置），替换 positions
        if getattr(self, "is_mrope_enabled", False):
            positions = forward_batch.mrope_positions

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
        )

        if get_embedding:
            return self.pooler(hidden_states, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch,
        )


EntryClass = YourModelForConditionalGeneration
```

**`general_mm_embed_routine` 是核心**——它会：
1. 用 `language_model.get_input_embeddings()` 把 text token 变成 embeddings；
2. 用 `multimodal_model.get_image_feature(items)` / `get_video_feature` / `get_audio_feature` 拿到视觉 / 音频 embeddings；
3. 把视觉 embeddings 按占位符位置 "scatter" 替换进 text embeddings（这一步依赖 `pad_input_ids` 做好的占位）；
4. 把合成的 `input_embeds` 喂给 `language_model` 跑推理。

### Step 7 — `pad_input_ids` 两种 pattern 的选择

SGLang 在 [`managers/mm_utils.py`](../../python/sglang/srt/managers/mm_utils.py) 提供两种现成 pattern：

| Pattern | 使用场景 | 例子 |
|---|---|---|
| `MultiModalityDataPaddingPatternMultimodalTokens` | 模型用**单个 token 重复展开**表示图像/视频/音频 | Qwen2/3-VL (`<image><image>...<image>`)、Gemma |
| `MultiModalityDataPaddingPatternTokenPairs` | 用**起止 token 对**标记一段多模态数据 | LLaVA、InternVL (`<img_start>...<img_end>`) |

若两种都不适用（例如 Phi-4-MM 的音频和图像有复杂交错），写自己的 `pad_input_tokens` 实现；参考 [`models/phi4mm.py`](../../python/sglang/srt/models/phi4mm.py)。

**这一步一定要做对**：展开后的占位 token 数必须**精确等于** ViT 输出的 token 数。少了会错位，多了会产生全零的"空 embedding"——都会让模型输出明显偏差。

### Step 8 — 处理 M-RoPE（多模态位置编码）

Qwen2-VL 及其后续模型使用 3 维 rotary（`[t, h, w]`），在 forward 里用：

```python
if self.is_mrope_enabled:
    positions = forward_batch.mrope_positions   # shape (3, seq_len)
```

`mrope_positions` 在 Processor 里通过 `build_input_ids_with_timestamps` 之类的工具计算，框架已在 `Req` 上准备好；你只需要在 forward 里替换 `positions`。漏替换 → 视觉 token 的位置全错 → 输出完全乱。

### Step 9 — 本地运行验证

#### 9.1 对比 HF Transformers（必做）

```bash
python3 scripts/playground/reference_hf.py \
    --model-path <your_vlm_path> \
    --model-type vlm

python3 -m sglang.bench_one_batch \
    --correct --model <your_vlm_path>
```

应当给出一致的文本输出 + 接近的 prefill logits。

#### 9.2 启动服务 + 图像请求

```bash
python3 -m sglang.launch_server \
    --model-path <path> --port 30000 \
    --enable-multimodal
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"x",
    "messages":[{
      "role":"user",
      "content":[
        {"type":"text","text":"What is in this image?"},
        {"type":"image_url","image_url":{"url":"https://.../cat.jpg"}}
      ]
    }]
  }'
```

#### 9.3 多卡

`--tp-size 2 --tp-size 4` 逐步验证。注意：ViT 的参数通常较小，多数情况下用 `ReplicatedLinear`（每卡完整）而不是 column/row parallel 能避免同步开销；仅 LM backbone 走 TP。

### Step 10 — 加入测试套件

- **生成正确性**：[`test/registered/models/test_generation_models.py`](../../test/registered/models/test_generation_models.py) 的 `ALL_OTHER_MODELS`。
- **VLM 端到端**：[`test/registered/vlm/test_vision_openai_server_{a,b,...}.py`](../../test/registered/vlm/)，挑一个分桶文件加测试。
- **MMMU 精度**：按 [`benchmark/mmmu/README.md`](../../benchmark/mmmu/README.md) 跑 SGLang 与 HF 的对比，结果写进 PR。SGLang 精度应不显著低于 HF。

---

## 2. 音频模型的差异点

| 差异项 | 说明 |
|---|---|
| Processor | 重写 `process_audio_async`，用 `load_audio` 读音频（WAV / MP3），走官方 `WhisperFeatureExtractor` / `Qwen2AudioProcessor` |
| 特殊 token | 多数用 `<|audio_start|>` / `<|audio_end|>` 成对；少数用 `<audio><audio>...<audio>` 重复 |
| 顶层类 | 定义 `get_audio_feature`；音频 backbone 通常是 `WhisperEncoder` 或 Conformer，独立放一个类 |
| LM 侧 | 多数模型用标准 Qwen / Llama backbone；无须额外改 |
| 测试 | 使用 `test/registered/audio/` 下的用例 |

参考：[`models/qwen2_audio.py`](../../python/sglang/srt/models/qwen2_audio.py)、[`models/voxtral.py`](../../python/sglang/srt/models/voxtral.py)、[`models/qwen3_asr.py`](../../python/sglang/srt/models/qwen3_asr.py)。

---

## 3. 视频的差异点

1. Processor 多走一步：**抽帧 + 缩放 + 时间维 patch**。参考 `smart_resize_image` + `preprocess_video`（在 [`multimodal/processors/qwen_vl.py`](../../python/sglang/srt/multimodal/processors/qwen_vl.py) 里）。
2. ViT 需要 3D patch（`Conv3dLayer` 或 `Conv2d` + 时间维循环）。
3. `video_grid_thw` 字段（时间、高、宽）在 `MultimodalDataItem` 上；`get_video_feature` 里按 thw 组织 token。
4. 帧数较多时请务必启用 **CUDA IPC 传输**（基类已封装），否则 pixel_values 的拷贝会成为瓶颈。

---

## 4. 全模态融合（图 + 音 + 视频）

参考 [`models/qwen3_omni.py`](../../python/sglang/srt/models/qwen3_omni.py) 和 [`models/gemma4_mm.py`](../../python/sglang/srt/models/gemma4_mm.py)。要点：

1. Processor 统一接收多种 modality，返回一个混合的 `mm_items` 列表（每个 item 带 `Modality` 枚举标识）。
2. 顶层类同时实现 `get_image_feature` / `get_video_feature` / `get_audio_feature`。
3. `general_mm_embed_routine` 的 `data_embedding_funcs` 参数可显式指定每个 modality 对应的函数：

```python
hidden_states = general_mm_embed_routine(
    input_ids=input_ids,
    forward_batch=forward_batch,
    language_model=self.model,
    multimodal_model=self,
    data_embedding_funcs={
        Modality.IMAGE: self.get_image_feature,
        Modality.VIDEO: self.get_video_feature,
        Modality.AUDIO: self.get_audio_feature,
    },
    positions=positions,
)
```

---

## 5. 外部 VLM 包（不改 SGLang 源码）

SGLang 提供三个环境变量配合，支持外部 VLM 插件：

```bash
export SGLANG_EXTERNAL_MODEL_PACKAGE=custom_vlm           # 含模型 EntryClass 的包
export SGLANG_EXTERNAL_MM_MODEL_ARCH=CustomQwen2VL        # 架构名（告诉 SGLang 这是多模态）
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=custom_vlm    # 含 Processor 的包（通常同上）

python -m sglang.launch_server \
    --model-path /path/to/Qwen2-VL-2B-Instruct \
    --port 8000 \
    --enable-multimodal
```

Processor 写成独立类但放同一个包，通过 `models = [YourCustomModel]` 关联。例子参见 [`docs/supported_models/extending/support_new_models.md`](../../docs/supported_models/extending/support_new_models.md) 的 "Multimodal Model" 章节。

---

## 6. 提交前自检清单

**LLM 侧**（参见 [`01_add_llm.md`](./01_add_llm.md) 清单全部条目）：
- [ ] `EntryClass` 名 == HF `architectures[0]`
- [ ] `forward()` 签名匹配 `(input_ids, positions, forward_batch, input_embeds=None, get_embedding=False)`（VLM 多数不带 `pp_proxy_tensors`；若支持 PP 则补上）
- [ ] LLM backbone 的 `stacked_params_mapping` 完整
- [ ] 无 `from vllm ...`

**VLM 特有**：
- [ ] 架构名加入 `multimodal_model_archs`
- [ ] `Processor` 在 `srt/multimodal/processors/` 下，`models = [YourModel]` 正确关联
- [ ] `pad_input_ids` 展开占位符数 == `get_image_feature` 输出 token 数
- [ ] ViT 多头 Attention 用 `VisionAttention`
- [ ] `get_image_feature` / `get_video_feature` / `get_audio_feature`（按需）完整
- [ ] `should_apply_lora` 跳过视觉塔
- [ ] `hf_to_sglang_mapper`（若权重命名差异）
- [ ] 支持 M-RoPE 的模型在 forward 里用了 `forward_batch.mrope_positions`
- [ ] `test_vision_openai_server_*.py` 里新增用例
- [ ] MMMU 精度实测并在 PR 描述中列出

---

## 7. 常见陷阱

1. **忘了把架构加进 `multimodal_model_archs`** → scheduler 走纯文本路径，图片信息完全丢失。
2. **`pad_input_ids` 展开数和 ViT 输出 token 数不一致** → 输出错位但表面"能跑通"，精度掉几十个点。
3. **`get_image_feature` 返回的 dtype 不对** → 和 LM backbone 拼接时报 dtype mismatch，或静默上下溢。必须 `.type(self.visual.dtype)`。
4. **ViT 用了 FP8 量化但 patch_embed 没量化** → 量化 backbone 期待 FP8 输入，却收到 BF16，kernel 崩。
5. **`hf_to_sglang_mapper` 的顺序搞反** → 前缀替换时把 `model.language_model.` 先替成 `language_model.model.`，再被 `model.` 这条误替一次。写 mapper 时把**更长的 prefix 放前面**。
6. **M-RoPE 模型漏了 `positions = forward_batch.mrope_positions`** → 视觉 token 位置全错。
7. **多图 / 多片视频时忘了 hash 区分** → RadixAttention 把不同图像当同一个 prefix 共享，导致结果互相污染。这一步其实由框架 `multimodal-data-hash` 自动完成，但要求 `pad_input_ids` 用了正确的 pattern class。
8. **Processor 在 CPU 上做 resize 但没分 worker** → TTFT 显著拉长；用基类的 `executor` 或手动 `ProcessPoolExecutor`。
9. **音频 `feature_extractor` 给了错误 sampling rate** → Whisper 系列默认 16 kHz；任何不一致都会直接让识别崩掉。
10. **`chat_template.jinja` 没处理 image_url 字段** → 客户端发了图但 template 不产出占位符，backend 收到的 prompt 里连 `<image>` 都没有，自然没图。

---

## 8. 进一步阅读

- [`01_add_llm.md`](./01_add_llm.md)：LLM 接入基础，本文所有"LLM 侧"项都假定你已读
- [`03_add_dit.md`](./03_add_dit.md)：扩散模型（Text→Image / Video）接入
- [`docs_zh/path.md`](../path.md)：仓库目录导览
- [`python/sglang/srt/managers/mm_utils.py`](../../python/sglang/srt/managers/mm_utils.py)：`general_mm_embed_routine` / `pad_input_tokens` 原理
- [`test/README.md`](../../test/README.md)：VLM 测试编排

---

**一句话总结**：VLM = 把 LLM 接入搞定 + 把**图像/视频/音频 → 占位 token → 视觉 token 替换回来**这一整条管线正确接通。难点在**占位符展开数与 ViT 输出 token 数必须严格一致**，以及**Processor / Pad / Feature 三段的 dtype / shape 对得上**。
