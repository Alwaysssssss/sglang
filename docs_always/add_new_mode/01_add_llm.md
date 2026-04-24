# 在 SGLang 中新增一个 LLM（纯文本大语言模型）

> 目标读者：希望在 `sglang/srt` 中接入一个新的 **Causal LM / Seq2Seq LM / Reward / Embedding** 架构，走标准文本推理链路（Tokenizer → Scheduler → ModelRunner → ForwardBatch → RadixAttention → LogitsProcessor → Detokenizer）。
>
> 相关参考：
> - 源代码目录：[`python/sglang/srt/models/`](../../python/sglang/srt/models/)
> - 注册表：[`python/sglang/srt/models/registry.py`](../../python/sglang/srt/models/registry.py)
> - 官方英文指南：[`docs/supported_models/extending/support_new_models.md`](../../docs/supported_models/extending/support_new_models.md)
>
> 本文档不涉及视觉 / 音频编码器、`pad_input_ids`、`get_image_feature` 等多模态逻辑——请看 [`02_add_vlm.md`](./02_add_vlm.md)；扩散模型请看 [`03_add_dit.md`](./03_add_dit.md)。

---

## 0. 要改哪些东西

SGLang 对 LLM 的接入非常克制，**核心只需要一个文件**：`python/sglang/srt/models/{your_model}.py`，声明一个继承 `nn.Module` 的模型类，并在文件末尾放 `EntryClass = [YourModelForCausalLM]`。其他事情由框架或已有通用组件处理：

| 职责 | 谁来做 | 你需要改吗 |
|------|--------|------------|
| 按 HF `config.json` 的 `architectures` 找到模型类 | `ModelRegistry`（自动扫描 `sglang.srt.models` 下的 `EntryClass`） | **否**，只要文件里正确写了 `EntryClass` |
| 加载权重 | `model_loader/loader.py` → 调用模型的 `load_weights()` | **看情况**，多数模型需要定义 `stacked_params_mapping` |
| 维护 KV Cache（RadixAttention / Paged） | `RadixAttention` 层 | **否**，复用 `sglang.srt.layers.radix_attention.RadixAttention` |
| 采样、logits 后处理 | `LogitsProcessor` | **否**，把 hidden_states 交给 `self.logits_processor(...)` 即可 |
| TP / PP / DP | `ColumnParallelLinear` / `RowParallelLinear` / `VocabParallelEmbedding` / `make_layers` | **多数只需调用**，不需要自己写通信 |
| HF 自定义 Config | `transformers` 或 `sglang/srt/configs/` | 若 HF 没有官方 Config 类，则新建一个 |
| Tokenizer / Chat template | `transformers` + 模型仓库自带 `tokenizer_config.json` / `chat_template.jinja` | **多数情况下不动** |

一句话：**加 LLM 的 95% 工作量是把一个新的网络结构翻译成 SGLang 的层**，剩下 5% 是把权重名对好。

---

## 1. 推荐工作流（九步）

### Step 1 — 找到最接近的参考实现

**永远不要从空白文件开始写**。SGLang 已经实现了约 180 个 LLM，覆盖绝大多数主流架构。按以下顺序找参考：

| 你的新模型类似 | 推荐参考文件 |
|---|---|
| Llama / GQA / RMSNorm / RoPE 标准 decoder-only | [`models/llama.py`](../../python/sglang/srt/models/llama.py) |
| Qwen2 / Qwen2.5（Llama 的变种，带 `attention_bias=True`） | [`models/qwen2.py`](../../python/sglang/srt/models/qwen2.py) |
| Qwen3 系列（含 QK Norm） | [`models/qwen3.py`](../../python/sglang/srt/models/qwen3.py) |
| MoE（Mixtral / Qwen2-MoE / DBRX 结构） | [`models/mixtral.py`](../../python/sglang/srt/models/mixtral.py)、[`models/qwen2_moe.py`](../../python/sglang/srt/models/qwen2_moe.py)、[`models/dbrx.py`](../../python/sglang/srt/models/dbrx.py) |
| DeepSeek V2/V3（MLA + MoE + DeepEP） | [`models/deepseek_v2.py`](../../python/sglang/srt/models/deepseek_v2.py) |
| MTP / NextN（多 token 预测） | [`models/deepseek_nextn.py`](../../python/sglang/srt/models/deepseek_nextn.py)、[`models/qwen3_5_mtp.py`](../../python/sglang/srt/models/qwen3_5_mtp.py) |
| Embedding / Reward / Classification（非生成） | [`models/llama_embedding.py`](../../python/sglang/srt/models/llama_embedding.py)、[`models/llama_reward.py`](../../python/sglang/srt/models/llama_reward.py)、[`models/llama_classification.py`](../../python/sglang/srt/models/llama_classification.py) |
| 混合 / 线性注意力（Mamba / RWKV / Hybrid） | [`models/falcon_h1.py`](../../python/sglang/srt/models/falcon_h1.py)、[`models/nemotron_h.py`](../../python/sglang/srt/models/nemotron_h.py)、[`models/kimi_linear.py`](../../python/sglang/srt/models/kimi_linear.py) |
| vLLM 已实现但 SGLang 还没 | 对比 vLLM `model_executor/models/*.py`，按 §4 checklist 改 |

**研读重点**：参考实现里的 `forward` 签名、`load_weights` 的权重映射、`stacked_params_mapping` 的写法——这三处对接成本最高。

### Step 2 — 评估能否"继承而不是复制"

SGLang 内大量模型直接继承别人。动手前先问：

1. 新模型**只是 Llama 的某些层换了一下**？（比如加 QK Norm、改 RoPE 参数）→ 继承 `LlamaForCausalLM`，重写必要模块。
2. 新模型**只改了采样后处理**？→ 继承已有的 `*ForCausalLM`，重写 `forward()` 里 logits 那一段。
3. 新模型**只加了 Reward Head / Classification Head**？→ 参考 `llama_reward.py` / `llama_classification.py`。

能继承就继承，差异用子类表达。

### Step 3 — 准备 HF Config

启动时通过 `transformers.AutoConfig` 读取 `config.json`。若模型在 HF Transformers 主仓已有 `XxxConfig`，**什么都不用做**；若没有：

1. 在 `python/sglang/srt/configs/{your_model}.py` 新建：

```python
from transformers import PretrainedConfig

class YourModelConfig(PretrainedConfig):
    model_type = "your_model"   # 与 config.json 的 "model_type" 一致

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        intermediate_size: int = 11008,
        max_position_embeddings: int = 8192,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
```

2. 在 [`python/sglang/srt/configs/__init__.py`](../../python/sglang/srt/configs/__init__.py) 中 export：

```python
from sglang.srt.configs.your_model import YourModelConfig

__all__ = [
    # ... existing ...
    "YourModelConfig",
]
```

### Step 4 — 建立骨架

创建 `python/sglang/srt/models/{your_model}.py`：

```python
from typing import Iterable, Optional, Tuple
import torch
from torch import nn

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding,
)
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, make_layers

# 1. Attention / MLP / DecoderLayer / Model
# 2. YourModelForCausalLM(nn.Module)
# 3. EntryClass = [YourModelForCausalLM]
```

### Step 5 — 实现网络组件

五个类是典型结构（下面以 Llama 风格为例）：

#### 5.1 `YourModelMLP`

```python
class YourModelMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act,
                 quant_config=None, prefix=""):
        super().__init__()
        # SwiGLU：gate_proj + up_proj 合并成 gate_up_proj
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2, bias=False,
            quant_config=quant_config, prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False,
            quant_config=quant_config, prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x
```

要点：
- **合并权重**：`gate_proj + up_proj` 合成 `gate_up_proj`（SwiGLU）；`q_proj + k_proj + v_proj` 合成 `qkv_proj`。HF 权重文件里是分开的，因此 `load_weights` 要做 shard 映射（见 §5.5）。
- **TP 切法**：column parallel 切输出维，row parallel 切输入维（内部加 all-reduce）。

#### 5.2 `YourModelAttention`

```python
class YourModelAttention(nn.Module):
    def __init__(self, config, layer_id, quant_config=None, prefix=""):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.hidden_size // self.total_num_heads
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size, self.head_dim,
            self.total_num_heads, self.total_num_kv_heads,
            bias=False, quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim, config.hidden_size,
            bias=False, quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            is_neox_style=True,
        )
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

    def forward(self, positions, hidden_states, forward_batch: ForwardBatch):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.num_heads * self.head_dim,
             self.num_kv_heads * self.head_dim,
             self.num_kv_heads * self.head_dim], dim=-1,
        )
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output
```

**关键点**：
- **`RadixAttention` 必须接收 `layer_id`**：SGLang 用 `layer_id` 作为 KV Cache 的 key，漏传就会崩。
- `is_neox_style`：Llama 风格 = `True`（split-half）；GPT-J 风格 = `False`（interleaved）。**搞错会全输出乱码**。
- `forward_batch` 里包含 Prefill / Decode 模式、序列长度、page 索引，`RadixAttention` 内部会分派到对应 attention backend（FlashAttention / FlashInfer / Triton / CutlassMLA 等）。

#### 5.3 `YourModelDecoderLayer`

```python
class YourModelDecoderLayer(nn.Module):
    def __init__(self, config, layer_id, quant_config=None, prefix=""):
        super().__init__()
        self.self_attn = YourModelAttention(
            config, layer_id, quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = YourModelMLP(
            config.hidden_size, config.intermediate_size, config.hidden_act,
            quant_config, prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, forward_batch, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # RMSNorm 支持带残差的 fused 版本
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
```

> SGLang 的 `RMSNorm` 允许把 `x + residual` 折叠进去（第二个返回值是新 residual）。利用这个能省一次内存往返。

#### 5.4 `YourModelModel`（不带 LM Head 的 backbone）

```python
class YourModelModel(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.pp_group = get_pp_group()
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size, quant_config=quant_config,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: YourModelDecoderLayer(
                config=config, layer_id=idx,
                quant_config=quant_config, prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix="model.layers",
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

    def forward(self, input_ids, positions, forward_batch,
                input_embeds=None, pp_proxy_tensors=None):
        if self.pp_group.is_first_rank:
            hidden_states = (
                input_embeds if input_embeds is not None
                else self.embed_tokens(input_ids)
            )
            residual = None
        else:
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            hidden_states, residual = self.layers[i](
                positions, hidden_states, forward_batch, residual,
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors({
                "hidden_states": hidden_states, "residual": residual,
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
```

**关键点**：
- `make_layers` 根据 pipeline parallel 切层、在不需要的 rank 上放 `PPMissingLayer`（占位，防止 load_weights 报错）。
- 中间 rank 必须返回 `PPProxyTensors(...)`，最后 rank 才返回真正的 hidden_states。

#### 5.5 `YourModelForCausalLM`（暴露给 `EntryClass` 的顶层类）

```python
class YourModelForCausalLM(nn.Module):
    default_bitsandbytes_target_modules = [
        ".gate_proj.", ".up_proj.", ".down_proj.",
        ".q_proj.", ".k_proj.", ".v_proj.", ".o_proj.",
    ]
    column_parallel_weights_modules = [".down_proj.", ".o_proj."]

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()
        self.model = YourModelModel(
            config, quant_config, prefix=add_prefix("model", prefix),
        )

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size, config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        self.logits_processor = LogitsProcessor(config)

        # (SGLang 合并后的参数名, HF 原始子权重名, shard_id)
        self.stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch,
                input_embeds=None, get_embedding=False, pp_proxy_tensors=None):
        hidden_states = self.model(
            input_ids, positions, forward_batch,
            input_embeds, pp_proxy_tensors=pp_proxy_tensors,
        )
        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids, hidden_states, self.lm_head, forward_batch,
                )
            else:
                raise NotImplementedError
        else:
            return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            for (param_name, weight_name, shard_id) in self.stacked_params_mapping:
                if weight_name not in name:
                    continue
                new_name = name.replace(weight_name, param_name)
                if new_name.endswith(".bias") and new_name not in params_dict:
                    continue
                param = params_dict[new_name]
                param.weight_loader(param, loaded, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded)

EntryClass = [YourModelForCausalLM]
```

**`forward` 签名必须严格匹配** `(input_ids, positions, forward_batch, input_embeds=None, get_embedding=False, pp_proxy_tensors=None)`。这是 `ModelRunner.forward_extend` / `forward_decode` 约定好的接口，多一个参数或少一个默认值都会让调度器抛 TypeError。

### Step 6 — 处理特殊情况

| 场景 | 做法 |
|------|------|
| RoPE 是 YaRN / Llama3 scaling / partial rope | `get_rope()` 传 `rope_scaling={...}`，见 [`layers/rotary_embedding.py`](../../python/sglang/srt/layers/rotary_embedding.py) |
| QK Norm（Qwen3、GLM4） | 在 Attention 里加 `self.q_norm = RMSNorm(head_dim, ...)` / `k_norm`，在 rotary 之前或之后归一 |
| Sliding Window Attention（Gemma、Mistral） | `RadixAttention` 构造时传 `sliding_window=xxx` |
| Attention Sink | 参考 [`models/gpt_oss.py`](../../python/sglang/srt/models/gpt_oss.py) |
| GQA 的非整除情况（`num_kv_heads < tp_size`） | 使用 `max(1, ...)` 并让部分 rank 重复 KV；QKVParallelLinear 已处理 |
| 模型带 LoRA 子模块 | 顶层类实现 `should_apply_lora(module_name) -> bool`，对 vision tower 返回 False |
| 权重文件前缀不同（`model.` vs `language_model.model.`） | 写一个 `WeightsMapper`（见 [`models/utils.py`](../../python/sglang/srt/models/utils.py)），或在 `load_weights` 中显式 rename |

### Step 7 — 注册 Chat Template（可选）

若模型自带 `chat_template.jinja`（放在 HF 仓库里），**不需要改 SGLang**——`transformers` 会自动加载。仅当官方没提供 Jinja template、或需要显式匹配时，在 [`python/sglang/srt/parser/conversation.py`](../../python/sglang/srt/parser/conversation.py) 里添加匹配函数。

### Step 8 — 本地验证

#### 8.1 单步对比（与 HF Transformers）

```bash
python3 scripts/playground/reference_hf.py \
    --model-path <hf_or_local_path> \
    --model-type text

python3 -m sglang.bench_one_batch \
    --correct \
    --model <hf_or_local_path>
```

两者应给出**相同的文本**和**非常接近的 prefill logits**（误差 < 1e-2 量级）。若差异明显，优先排查：

1. `stacked_params_mapping` 是否把 QKV / gate_up 的 shard_id 顺序搞反。
2. RoPE 的 `is_neox_style` / `rope_theta` / `rope_scaling`。
3. Norm 的 `eps`。
4. `tie_word_embeddings` 是否正确。

#### 8.2 Server Smoke Test

```bash
python3 -m sglang.launch_server \
    --model-path <path> \
    --port 30000 \
    --trust-remote-code
```

```bash
curl http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"x","messages":[{"role":"user","content":"Hello"}]}'
```

#### 8.3 多 GPU / TP / PP

先把 `--tp-size` 设成 2 验证；再加 `--pp-size` 验证 Pipeline Parallel。多数 bug 在这一步暴露（例如漏掉 `VocabParallelEmbedding`、`PPMissingLayer`）。

### Step 9 — 加入测试套件

在 [`test/registered/models/test_generation_models.py`](../../test/registered/models/test_generation_models.py) 的 `ALL_OTHER_MODELS` 列表里加：

```python
ModelCase("org/your-model-name"),
```

跑一次：

```bash
ONLY_RUN=org/your-model-name python3 -m unittest \
    test_generation_models.TestGenerationModels.test_others
```

**PR 必须附带的 benchmark**：
- [GSM8K](../../benchmark/gsm8k/)：数学题准确率，判断采样路径是否正确
- [MMLU](../../benchmark/mmlu/)：通用知识，判断模型语义能力没掉

---

## 2. 处理 MoE / MLA / 线性注意力等变体

### 2.1 MoE（Mixture-of-Experts）

- 用 `sglang.srt.layers.moe.fused_moe_triton` 的 `FusedMoE` 或 `EPMoE` 封装 experts。
- **权重合并**：gate / up / down 每个 expert 都需走 `stacked_params_mapping` 的扩展形式（参考 `qwen2_moe.py`）。
- 若用 DeepEP（DeepSeek V3 的跨机专家并行），走 [`models/deepseek_v2.py`](../../python/sglang/srt/models/deepseek_v2.py) 的路径。
- Router 的 top-k、scoring、normalize 都要和参考实现一致；router dtype 是否升 fp32 决定数值稳定性。

### 2.2 MLA（Multi-head Latent Attention，DeepSeek V2/V3）

- Q/K 分解方式与标准 MHA 完全不同。不要从 `llama.py` 派生，直接参考 [`models/deepseek_v2.py`](../../python/sglang/srt/models/deepseek_v2.py)。
- `RadixAttention` 内置 MLA 专用路径。

### 2.3 Hybrid / Linear Attention（Mamba、RWKV、Jet、Nemotron-H）

- 线性注意力层**不是** `RadixAttention`；用 `sglang.srt.layers.attention.mamba` 等专用实现。
- KV Cache 结构完全不同（state 而非 key/value），框架需要知道这是线性注意力模型——在 [`models/linear_attn_model_registry.py`](../../python/sglang/srt/models/linear_attn_model_registry.py) 注册。
- 混合模型（Jet = Transformer + Mamba 交替）每层按自己类型分派。

### 2.4 MTP / Speculative

- 主模型文件 + 一个 `*_mtp.py` / `*_nextn.py` 文件。
- `EntryClass` 只注册主模型；MTP 分支通过 `--speculative-algorithm EAGLE / NEXTN` 启用，由框架自动加载。
- 参考 [`models/deepseek_nextn.py`](../../python/sglang/srt/models/deepseek_nextn.py)、[`models/qwen3_5_mtp.py`](../../python/sglang/srt/models/qwen3_5_mtp.py)。

---

## 3. Embedding / Reward / Classification 模型

这些"非生成"模型的外层类**不能直接上 `LogitsProcessor`**，要改成 `Pooler` 或 `ClassificationHead`：

```python
from sglang.srt.layers.pooler import Pooler, PoolingType

self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

def forward(self, ..., get_embedding=False):
    hidden_states = self.model(...)
    return self.pooler(hidden_states, forward_batch)
```

- Reward Model：参考 [`models/llama_reward.py`](../../python/sglang/srt/models/llama_reward.py)
- Classification：参考 [`models/llama_classification.py`](../../python/sglang/srt/models/llama_classification.py)
- `model_config.py` 的 `is_generation_model()` 会根据 `architectures` 是否含 `*ForSequenceClassification` / `*ForRewardModel` 自动识别；新增架构时记得在那里加一条。

---

## 4. 从 vLLM 移植：变更 Checklist

SGLang 与 vLLM 的模型接口高度相似，多数 vLLM 模型可以 1 小时内移植过来。**逐项替换**：

| vLLM 组件 | 替换为 SGLang 组件 | 备注 |
|-----------|---|------|
| `Attention` | `RadixAttention` | 必须传 `layer_id` |
| `LogitsProcessor`（vllm 的） | `LogitsProcessor`（sglang 的） | 签名略不同 |
| `Sampler` | **删掉** | sglang 采样由 scheduler 管 |
| ViT 的多头 `Attention` | `VisionAttention` | 见 `02_add_vlm.md` |
| `RMSNorm` / `SiluAndMul` | sglang 同名类 | 注意 fused 残差版本 |
| `forward(..., kv_caches, attn_metadata)` | `forward(..., forward_batch)` | `forward_batch` 里已经有一切 |
| `weight_loader` 调用方式 | 走 `default_weight_loader` 或 `param.weight_loader` | — |
| 文件末尾的注册 | 加 `EntryClass = [XxxForCausalLM]` | 类名必须 = HF `architectures[0]` |

**确保最终文件里没有 `from vllm ... import ...`**——否则 SGLang 无法在无 vLLM 环境下加载。

---

## 5. 外部模型包（不改 SGLang 源码）

不想把新模型合进主仓，可通过环境变量注册：

```bash
pip install -e .   # 你自己的包，入口含 EntryClass

export SGLANG_EXTERNAL_MODEL_PACKAGE=custom_pkg
python -m sglang.launch_server --model-path /path/to/ckpt --port 8000
```

`ModelRegistry.register(external_pkg, overwrite=True)` 会扫描你的包、收集 `EntryClass`。把 `config.json` 的 `architectures` 改成你的类名即可：

```json
{"architectures": ["LlamaWrapper"]}
```

---

## 6. 提交前自检清单

- [ ] 新模型文件在 `python/sglang/srt/models/{your_model}.py`，末尾有 `EntryClass`
- [ ] `EntryClass` 类名 == HF `config.json` 的 `architectures[0]`
- [ ] `forward()` 签名严格匹配 `(input_ids, positions, forward_batch, input_embeds=None, get_embedding=False, pp_proxy_tensors=None)`
- [ ] 所有 Attention 层都用 `RadixAttention` 且传了正确的 `layer_id`
- [ ] QKV / gate_up 在 `stacked_params_mapping` 中映射
- [ ] `load_weights()` 处理了过滤（`rotary_emb.inv_freq`、`tie_word_embeddings` 下的 `lm_head`）
- [ ] TP 下 `num_heads % tp_size == 0`，GQA 下 `num_kv_heads` 处理了 `max(1, ...)`
- [ ] PP 下 `embed_tokens` / `norm` / `lm_head` 用了 `PPMissingLayer` 包裹
- [ ] 和 HF Transformers 同 prompt 输出一致
- [ ] 加入 `ALL_OTHER_MODELS` 并本地跑通
- [ ] PR 附带 GSM8K / MMLU 结果
- [ ] 文件里没有任何 `from vllm ... import ...`

---

## 7. 常见陷阱

1. **忘了传 `layer_id` 给 `RadixAttention`** → 运行时 KV Cache 索引冲突，显存炸 / 结果乱。
2. **RoPE `is_neox_style` 搞错** → 输出"接近正常"但第几 token 后开始偏。**同 seed、同 prompt、对比中间 Q/K 张量**能精准定位。
3. **合并权重的 shard_id 顺序搞反** → `qkv_proj` 里 Q/K/V 错排，attention 完全乱。
4. **`tie_word_embeddings=True` 时重复加载 `lm_head.weight`** → load_weights 报 KeyError。显式 skip 那个 key。
5. **PP 中间 rank 忘了返回 `PPProxyTensors`** → 下一个 rank 拿不到 `residual`，崩。
6. **量化参数没继承到新层** → 每一层构造时漏传 `quant_config`，加载后全走 fp16。
7. **TP 下 RMSNorm 的权重是 replicated 的**（不用切）。误用 `ColumnParallelLinear` 等价物会 shape 不对。
8. **`trust_remote_code` 模型**在 SGLang 里优先走你提供的 Config，不要依赖 `auto_map`。

---

## 8. 进一步阅读

- [`docs_zh/path.md`](../path.md)：仓库全目录解读
- [`docs/supported_models/`](../../docs/supported_models/)：官方英文文档
- [`test/README.md`](../../test/README.md)：CI 测试编排、新模型如何加入 suite
- [`docs/developer_guide/`](../../docs/developer_guide/)：调度器 / Scheduler / ForwardBatch 深入原理

---

**一句话总结**：在 `srt/models/` 下写一个文件，把网络翻译成 SGLang 的层（`RadixAttention`、`*ParallelLinear`、`RMSNorm`、`LogitsProcessor`），末尾贴 `EntryClass`，对齐权重命名即可。难的不是框架，难的是和 HF 参考实现**对齐到底**。
