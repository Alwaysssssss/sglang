# 在 SGLang Diffusion 中新增一个 DiT / 扩散模型

> 目标读者：希望在 `sglang.multimodal_gen` 中接入一个新的 **扩散模型**（Text→Image / Image→Image / Text→Video / Image→Video / 3D 等），基于 `DiT` / `UNet` + `VAE` + `Text Encoder` + `Scheduler` 的标准扩散架构。
>
> 相关参考：
> - 扩散运行时目录：[`python/sglang/multimodal_gen/`](../../python/sglang/multimodal_gen/)
> - 扩散模型架构文档（中文）：[`docs_zh/multimodal_gen/`](../multimodal_gen/) 系列
> - 扩散 Pipeline 中心注册表：[`python/sglang/multimodal_gen/registry.py`](../../python/sglang/multimodal_gen/registry.py)
> - 官方英文指南：[`docs/diffusion/support_new_models.md`](../../docs/diffusion/support_new_models.md)
> - Skill 完整原始描述：[`python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-add-model/SKILL.md`](../../python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-add-model/SKILL.md)
>
> **本文档与前两篇的关系**：
> - [`01_add_llm.md`](./01_add_llm.md) 覆盖纯 LLM（走 srt 链路，`RadixAttention` / `LogitsProcessor`）
> - [`02_add_vlm.md`](./02_add_vlm.md) 覆盖多模态 LLM（LLM + ViT + Processor）
> - 本文档覆盖**扩散模型**——走的是完全不同的运行时 `sglang.multimodal_gen`，不共享 srt 的调度器、KV Cache、Attention Backend。

---

## 0. 扩散模型接入和 LLM 接入的本质差异

| 维度 | LLM / VLM（`sglang.srt`） | 扩散模型（`sglang.multimodal_gen`） |
|---|---|---|
| 运行时抽象 | `ModelRunner` + `Scheduler` + `ForwardBatch` + `RadixAttention` | `ComposedPipeline` + `PipelineStage` + `Req` + `Executor` |
| 入口注册 | 模型文件里的 `EntryClass`，被 `ModelRegistry` 扫描 | 两层：Pipeline 类通过 `EntryClass` 被发现；SamplingParams + PipelineConfig 通过 `register_configs()` 手动登记 |
| 请求粒度 | 一条 prompt → 多次 decode | 一条 prompt → 一次"多步去噪" |
| KV Cache | 核心 | **不存在** |
| 主要关心 | 并行、吞吐、首 token 延迟 | 输出像素精度、与 Diffusers 的数值对齐、Sampler 调度 |
| 和 HuggingFace 对标 | `transformers` | `diffusers` + `model_index.json` |

**"添加一个扩散模型"的工作 = 把 Diffusers 里对应的 `pipeline_*.py` 搬进 SGLang 的 Stage/Pipeline 抽象里，同时复用 SGLang 的融合 kernel、TP/SP 分布式、kernel-level 优化**。

---

## 1. 关键目录速查

| 职责 | 路径 |
|---|---|
| Pipeline 类（把 Stage 串起来） | [`python/sglang/multimodal_gen/runtime/pipelines/`](../../python/sglang/multimodal_gen/runtime/pipelines/) |
| 模型专属 Stage | [`runtime/pipelines_core/stages/model_specific_stages/`](../../python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/) |
| 标准共享 Stage | [`runtime/pipelines_core/stages/`](../../python/sglang/multimodal_gen/runtime/pipelines_core/stages/) |
| `PipelineStage` 基类 | [`runtime/pipelines_core/stages/base.py`](../../python/sglang/multimodal_gen/runtime/pipelines_core/stages/base.py) |
| `ComposedPipelineBase` | [`runtime/pipelines_core/composed_pipeline_base.py`](../../python/sglang/multimodal_gen/runtime/pipelines_core/composed_pipeline_base.py) |
| DiT 模型实现 | [`runtime/models/dits/`](../../python/sglang/multimodal_gen/runtime/models/dits/) |
| VAE 实现 | [`runtime/models/vaes/`](../../python/sglang/multimodal_gen/runtime/models/vaes/) |
| Text Encoder 实现 | [`runtime/models/encoders/`](../../python/sglang/multimodal_gen/runtime/models/encoders/) |
| Scheduler 实现 | [`runtime/models/schedulers/`](../../python/sglang/multimodal_gen/runtime/models/schedulers/) |
| Pipeline 配置 | [`configs/pipeline_configs/`](../../python/sglang/multimodal_gen/configs/pipeline_configs/) |
| DiT 结构配置 | [`configs/models/dits/`](../../python/sglang/multimodal_gen/configs/models/dits/) |
| VAE 结构配置 | [`configs/models/vaes/`](../../python/sglang/multimodal_gen/configs/models/vaes/) |
| Encoder 结构配置 | [`configs/models/encoders/`](../../python/sglang/multimodal_gen/configs/models/encoders/) |
| 采样参数 | [`configs/sample/`](../../python/sglang/multimodal_gen/configs/sample/) |
| 中央注册表 | [`registry.py`](../../python/sglang/multimodal_gen/registry.py) |

---

## 2. 两种 Pipeline 风格：Hybrid vs Modular

### 2.1 风格 A：Hybrid（**推荐默认**）

三段式：

```
{Model}BeforeDenoisingStage (模型专属)  →  DenoisingStage (标准)  →  DecodingStage (标准)
```

- `{Model}BeforeDenoisingStage`：**一个** 模型专属 Stage，把输入校验、文本/图像编码、latent 准备、timestep 计算全部装进去。
- `DenoisingStage`：跨模型通用的去噪循环（调用 DiT / UNet forward）。
- `DecodingStage`：跨模型通用的 VAE decode。

**优点**：现代扩散模型的前处理差异极大（文本编码器多样、latent packing 方式不同、条件机制复杂）。把差异封装在一个 Stage 内，避免共享 Stage 里充满 `if model_name == ...` 分支，也让开发者可以近乎**照搬 Diffusers 参考 `__call__`**。

### 2.2 风格 B：Modular（标准 Stage 合身时）

用框架现成的细粒度 Stage（`TextEncodingStage` / `LatentPreparationStage` / `TimestepPreparationStage` 等），典型调用：

- `add_standard_t2i_stages()`（文生图）
- `add_standard_ti2i_stages()`（图 + 文到图）
- `add_standard_ti2v_stages()`（图 + 文到视频）

**适用场景**：
- 前处理能 80%+ 复用标准 Stage（标准 CLIP/T5 + 标准 latent）；
- 某个模型专属的优化步骤需要抽成独立 Stage（便于单独 profile / 并行控制 / 多 pipeline 复用）。

### 2.3 抉择表

| 情境 | 推荐 |
|---|---|
| 模型前处理独特/复杂（VLM captioning、AR token 生成、自定义 latent packing…） | **Hybrid** |
| 模型能套进标准 T2I / TI2I / TI2V 模式 | **Modular** |
| 从 Diffusers 移植带多步定制的 pipeline | **Hybrid** |
| 给现有模型加变体、主干逻辑共用 | **Modular**（用 `PipelineConfig` 回调覆盖差异） |
| 某个前处理步骤需要特别的并行 / profiling 隔离 | **Modular** |

**两种风格的共同契约**：进入 `DenoisingStage` 之前，`Req` 上必须备齐标准张量字段：`latents`、`timesteps`、`sigmas`、`prompt_embeds`、`negative_prompt_embeds`、`generator`、`raw_latent_shape`、`num_inference_steps`……（完整清单见 §5）。

---

## 3. 九步实施流程

### Step 1 — 研读参考实现

**写任何代码前**，必须拿到模型的官方参考实现或 Diffusers Pipeline 源码，否则无法和目标输出对齐。信息来源优先级：

1. Diffusers 里的 `pipeline_*.py`（或 HuggingFace 仓库自带的 pipeline）
2. 作者官方 GitHub repo 的 reference 实现
3. 至少拿到 HuggingFace 模型 ID，能查 `model_index.json` 和对应 pipeline 类

**从参考代码识别 6 件事**：

1. `model_index.json` 中必需的模块（`text_encoder` / `vae` / `transformer` / `scheduler` …）
2. 文本 prompt 如何编码
3. latent 如何准备（形状、dtype、scale）
4. timesteps / sigmas 如何计算
5. DiT / UNet 接受哪些条件 kwargs（**这一项最关键**）
6. 去噪循环细节（CFG、guidance_scale…）和 VAE 解码细节（scaling factor、tiling…）

### Step 2 — 评估"复用 vs 新建"

**禁止盲目造新文件**。决策清单：

1. 对比现有 Pipeline（Flux、Wan、Qwen-Image、GLM-Image、HunyuanVideo、LTX、SANA、StableDiffusion3、Z-Image…）：
   - 结构相似 → 在现有 Pipeline 上**加一个 config 变体**，而不是新建 Pipeline 类
   - 复用现有 `BeforeDenoisingStage`，只改参数
   - 若匹配标准模式，直接 `add_standard_t2i_stages()` / `add_standard_ti2i_stages()` / `add_standard_ti2v_stages()`
2. 检查 [`runtime/pipelines_core/stages/`](../../python/sglang/multimodal_gen/runtime/pipelines_core/stages/) 和 `model_specific_stages/`：若已有 Stage 覆盖 80%+ 需求，**扩展它**而不是复制。
3. 检查模型组件：VAE（如 `AutoencoderKL`）、文本编码器（CLIP、T5、LLaMA-3）、scheduler 多数可直接复用。

### Step 3 — 实现模型组件

#### 3.1 DiT / Transformer

位置：`runtime/models/dits/{model_name}.py`

要点：
- **参数命名与 Diffusers 保持一致**，`ComponentLoader` 才能自动加载权重。
- 使用 SGLang 的融合算子：
  - `LayerNormScaleShift` / `RMSNormScaleShift`（合并了 norm + modulation 的两次乘加）
  - `apply_qk_norm`（QKNorm 的融合实现）
  - `get_attn_backend()`（自动挑 Flash / Sage / USP / Ring 等 backend）
- 与 Diffusers 原始实现在**残差位置、AdaLayerNorm 顺序、timestep embedding 方式**上对齐。

**分布式支持（TP / SP）——推荐加**。可先在单卡跑通再增量补，参考：

- [`runtime/models/dits/wanvideo.py`](../../python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py)：完整 TP + SP 参考
  - TP：`ColumnParallelLinear`（Q/K/V 投影）+ `RowParallelLinear`（输出投影），注意力 head 按 `tp_size` 切分
  - SP：序列维度 shard，`get_sp_world_size()` / padding 对齐 / `sequence_model_parallel_all_gather` 聚合
  - Cross-attention 跳过 SP（`skip_sequence_parallel=is_cross_attention`）
- [`runtime/models/dits/qwen_image.py`](../../python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py)：SP + USPAttention 参考
  - `USPAttention`（Ulysses + Ring Attention），通过 `--ulysses-degree` / `--ring-degree` 配置
  - `MergedColumnParallelLinear`（带 Nunchaku 量化支持）/ `ReplicatedLinear`

关键 import：

```python
from sglang.multimodal_gen.runtime.distributed import (
    divide,
    get_sp_group,
    get_sp_world_size,
    get_tp_world_size,
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    ReplicatedLinear,
    MergedColumnParallelLinear,
)
```

#### 3.2 VAE

- 多数扩散模型用 `AutoencoderKL`（Stable Diffusion 系）或其 3D 变体（视频模型）——直接复用 `runtime/models/vaes/` 下的现成实现。
- 只有**非标准 VAE**才新建文件（例如 Wan 的 3D causal VAE、LTX 的 pixel-shuffle VAE）。
- VAE 精度很敏感，`PipelineConfig.vae_precision` 一般设 `"bf16"` 或 `"fp32"`，低精度会导致纹理丢失。

#### 3.3 Text Encoder

- CLIP / T5 / Gemma / LLaMA 等已实现，路径 `runtime/models/encoders/`。
- 新 encoder 才写新文件；注意 HuggingFace 权重命名要对齐。

#### 3.4 Scheduler

- 绝大多数扩散模型用 `FlowMatchEulerDiscreteScheduler` / `DPM++` / `EulerDiscrete` —— 路径 `runtime/models/schedulers/`。
- 只有采样规则与现有不同（如 `DMDScheduler`、某些 Turbo 模型的 bespoke sigmas）才新建。

### Step 4 — 写结构配置（三个 `@dataclass`）

#### 4.1 DiT 配置

`configs/models/dits/{model_name}.py`：

```python
from dataclasses import dataclass, field
from sglang.multimodal_gen.configs.models.dits.base import DiTConfig, DiTArchConfig

@dataclass
class MyModelDiTArchConfig(DiTArchConfig):
    in_channels: int = 16
    out_channels: int = 16
    num_layers: int = 40
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    patch_size: int = 2
    hidden_size: int = 3072
    ...

@dataclass
class MyModelDiTConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=MyModelDiTArchConfig)
    precision: str = "bf16"
```

#### 4.2 VAE 配置

`configs/models/vaes/{model_name}.py`（定义 `vae_scale_factor`、`latents_mean`、`latents_std` 等）。

#### 4.3 采样参数

`configs/sample/{model_name}.py`：

```python
@dataclass
class MyModelSamplingParams(SamplingParams):
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    height: int = 1024
    width: int = 1024
    prompt: Optional[str] = None
    negative_prompt: str = ""
    seed: Optional[int] = None
```

### Step 5 — 写 `PipelineConfig`

`PipelineConfig` 承载**静态配置 + 给标准 Stage 用的回调方法**：

```python
@dataclass
class MyModelPipelineConfig(ImagePipelineConfig):
    task_type: ModelTaskType = ModelTaskType.T2I
    vae_precision: str = "bf16"
    should_use_guidance: bool = True
    vae_tiling: bool = False
    enable_autocast: bool = False

    dit_config: DiTConfig = field(default_factory=MyModelDiTConfig)
    vae_config: VAEConfig = field(default_factory=MyModelVAEConfig)

    # --- DenoisingStage 回调 ---
    def get_freqs_cis(self, batch, device, rotary_emb, dtype):
        """为 DiT 准备 rotary position embedding。"""
        ...

    def prepare_pos_cond_kwargs(self, batch, latent_model_input, t, **kwargs):
        """构造正向条件 kwargs（喂给 DiT.forward）。"""
        return {
            "hidden_states": latent_model_input,
            "encoder_hidden_states": batch.prompt_embeds[0],
            "timestep": t,
        }

    def prepare_neg_cond_kwargs(self, batch, latent_model_input, t, **kwargs):
        """构造负向条件 kwargs（CFG 用）。"""
        return {
            "hidden_states": latent_model_input,
            "encoder_hidden_states": batch.negative_prompt_embeds[0],
            "timestep": t,
        }

    # --- DecodingStage 回调 ---
    def get_decode_scale_and_shift(self):
        """VAE decode 之前用的 (scale, shift) — 对 latent 做反归一化。"""
        return self.vae_config.latents_std, self.vae_config.latents_mean

    def post_denoising_loop(self, latents, batch):
        """去噪循环结束后的可选后处理（dtype 转换等）。"""
        return latents.to(torch.bfloat16)

    def post_decoding(self, frames, server_args):
        """VAE decode 之后的可选后处理。"""
        return frames
```

> **⚠️关键契约**：`prepare_pos_cond_kwargs` / `prepare_neg_cond_kwargs` 返回 dict 的键值**必须与 DiT `forward()` 的签名完全一致**，否则 `DenoisingStage` 会直接 `TypeError` 或给出错误条件。

### Step 6 — 实现 `BeforeDenoisingStage`（Hybrid 的心脏）

位置：`runtime/pipelines_core/stages/model_specific_stages/{model_name}.py`

职责：
- 输入校验（`height` / `width` 合法性、`prompt` 长度…）
- 文本 / 图像编码（调用 text encoder、image encoder）
- Latent 准备（初始化噪声 latent）
- Timestep / sigma 调度

骨架（与 Diffusers 参考 `__call__` 前半段一一对应）：

```python
class MyModelBeforeDenoisingStage(PipelineStage):
    def __init__(self, vae, text_encoder, tokenizer, transformer, scheduler):
        super().__init__()
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.scheduler = scheduler

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()

        # 1. 文本编码
        prompt_embeds, negative_prompt_embeds = self._encode_prompt(...)

        # 2. Latent 初始化（按 sampler shape、height/width、VAE scale factor 计算）
        latents = self._prepare_latents(...)

        # 3. Timestep / sigma
        timesteps, sigmas = self._prepare_timesteps(...)

        # 4. 结果挂到 batch 上
        batch.latents = latents
        batch.timesteps = timesteps
        batch.num_inference_steps = len(timesteps)
        batch.sigmas = sigmas.tolist()   # 必须是 Python list
        batch.prompt_embeds = [prompt_embeds]                # list[Tensor]
        batch.negative_prompt_embeds = [negative_prompt_embeds]
        batch.generator = generator
        batch.raw_latent_shape = latents.shape
        batch.height = server_args.sampling_params.height
        batch.width = server_args.sampling_params.width
        return batch
```

**`DenoisingStage` 期望的 `batch` 字段**（全部必须在 `BeforeDenoisingStage.forward()` 填好）：

| 字段 | 类型 | 含义 |
|---|---|---|
| `batch.latents` | `torch.Tensor` | 初始噪声 latent |
| `batch.timesteps` | `torch.Tensor` | 时间步调度 |
| `batch.num_inference_steps` | `int` | 去噪步数 |
| `batch.sigmas` | `list[float]` | sigma 调度（**必须是 Python list** 不是 numpy / tensor） |
| `batch.prompt_embeds` | `list[torch.Tensor]` | 正向 prompt embedding（**外层 list 包装**，每个 text encoder 一项） |
| `batch.negative_prompt_embeds` | `list[torch.Tensor]` | 负向 prompt embedding（同上包装） |
| `batch.generator` | `torch.Generator` | RNG 用于复现 |
| `batch.raw_latent_shape` | `tuple` | 原始 latent 形状，供 `DecodingStage` 反打包 |
| `batch.height` / `batch.width` | `int` | 输出分辨率 |

### Step 7 — 定义 Pipeline 类

Pipeline 类只负责把 Stage 串起来，非常薄：

#### Hybrid 风格

```python
# runtime/pipelines/my_model.py

from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import DenoisingStage

class MyModelPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "MyModelPipeline"   # 必须 == model_index.json 的 _class_name

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(
            MyModelBeforeDenoisingStage(
                vae=self.get_module("vae"),
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )
        self.add_stage(
            DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )
        self.add_standard_decoding_stage()


EntryClass = [MyModelPipeline]
```

#### Modular 风格

```python
class MyModelPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "MyModelPipeline"
    _required_config_modules = [...]

    def create_pipeline_stages(self, server_args: ServerArgs):
        # 一次性添加：InputValidation + TextEncoding + LatentPrep + TimestepPrep + Denoising + Decoding
        self.add_standard_t2i_stages(
            prepare_extra_timestep_kwargs=[prepare_mu],   # 模型专属 hook
        )


EntryClass = [MyModelPipeline]
```

> `pipeline_name` **必须等于** HuggingFace `model_index.json` 里的 `_class_name` 字段，不然注册发现不了你。

### Step 8 — 注册

编辑 [`python/sglang/multimodal_gen/registry.py`](../../python/sglang/multimodal_gen/registry.py) 的 `_register_configs()` 函数，追加：

```python
register_configs(
    sampling_param_cls=MyModelSamplingParams,
    pipeline_config_cls=MyModelPipelineConfig,
    hf_model_paths=[
        "org/my-model-name",           # HuggingFace 模型 ID；可多个
    ],
    model_detectors=[
        # 可选：按路径子串识别
        lambda path: "my-model" in path.lower(),
    ],
)
```

> **`register_configs` 的实际签名（不含 `model_family` 参数）**：
> ```python
> def register_configs(
>     sampling_param_cls,
>     pipeline_config_cls,
>     hf_model_paths: Optional[List[str]] = None,
>     model_detectors: Optional[List[Callable[[str], bool]]] = None,
> )
> ```
> 一些旧文档里写的 `model_family="..."` 是早期 API，已废除。

Pipeline 类本身由 `_discover_and_register_pipelines()` **自动发现 `EntryClass`**，不需要手动登记。

### Step 9 — 验证输出质量

> **输出是噪声 = 实现错误，不允许凑合上线**。

典型"噪声输出"成因（按排查优先级）：

1. `get_decode_scale_and_shift` 返回了错误的 latent scale / shift
2. timestep / sigma 调度方向、dtype、数值区间不对
3. 条件 kwargs 字段名与 DiT `forward()` 签名不匹配
4. VAE 解码配置错误（`vae_scale_factor` 错、漏反归一化…）
5. 旋转位置编码风格错（`is_neox_style=True`：split-half；`False`：interleaved）
6. Prompt embedding 格式错（漏 list 包装、选错 encoder 输出）

**调试手段**：

- 和 Diffusers 参考 pipeline **同 seed 并排跑**，比对每一步中间张量：
  - `prompt_embeds` 范数 / 分布
  - 初始 `latents` 数值范围（标准正态？ std=1？）
  - `timesteps` / `sigmas` 数组数值
  - 第 0、1、2 步去噪后 `latent` 的 mean / std
- 独立检查每个 Stage 输出的 shape 与数值范围

---

## 4. 任务类型与 Task 模板

SGLang Diffusion 预置了三类标准任务，它们对应 `add_standard_*_stages()`：

| 任务 | 标准 Stage 序列 | 典型模型 |
|---|---|---|
| **T2I** Text → Image | InputValidation + TextEncoding + TimestepPrep + LatentPrep + Denoising + Decoding | Flux、SD3、Qwen-Image、Z-Image、SANA |
| **TI2I** Text+Image → Image | 上述 + ImageEncoding（或 ImageVAEEncoding） | Qwen-Image-Edit、Flux-Kontext |
| **TI2V** Text+Image → Video | 上述 + ImageVAEEncoding + 视频专用 LatentPrep | Wan-I2V、HunyuanVideo、LTX |

**纯视频 T2V**、**3D 模型**、**声音/音频扩散**：暂无"标准"序列，走 Hybrid。

---

## 5. 分布式：TP / SP / USP

### 5.1 Tensor Parallel（TP）

DiT 里每个 Block 的 Q/K/V/Output 线性层切 head：

```python
self.to_q = ColumnParallelLinear(dim, num_heads * head_dim, ...)
self.to_k = ColumnParallelLinear(dim, num_kv_heads * head_dim, ...)
self.to_v = ColumnParallelLinear(dim, num_kv_heads * head_dim, ...)
self.to_out = RowParallelLinear(num_heads * head_dim, dim, ...)

# 注意力 head 按 tp_size 切
self.num_heads_per_rank = divide(total_num_heads, tp_size)
```

### 5.2 Sequence Parallel（SP）

视频 / 高分辨率图像场景，**序列维**很长（Wan 的 latent 序列可达几万），把它切到多卡：

```python
seq_len = hidden_states.shape[1]
sp_world_size = get_sp_world_size()
assert seq_len % sp_world_size == 0, "seq_len 必须能整除 sp_world_size"
hidden_states = hidden_states.chunk(sp_world_size, dim=1)[get_sp_group().rank_in_group]

# forward ...

# 最后 gather 回完整序列
hidden_states = sequence_model_parallel_all_gather(hidden_states, dim=1)
```

**Cross-attention 要跳过 SP**（因为 encoder_hidden_states 不切）：

```python
class MyAttnBlock(nn.Module):
    def __init__(self, is_cross_attention=False, ...):
        self.skip_sequence_parallel = is_cross_attention
```

### 5.3 USP（Ulysses + Ring Attention）

Qwen-Image / Flux 使用 USP 做序列并行 attention，通过 `--ulysses-degree` / `--ring-degree` 启用：

```python
from sglang.multimodal_gen.runtime.layers.attention.usp import USPAttention

self.attn = USPAttention(
    ulysses_degree=get_ulysses_world_size(),
    ring_degree=get_ring_world_size(),
    ...
)
```

**什么时候用哪个**：

| 模型序列长度 | 推荐 |
|---|---|
| < 10k token | 单卡或 TP-only |
| 10k–100k（大图 / 视频） | SP（Wan 风格）或 USP |
| > 100k（长视频） | USP + Ring Attention |

---

## 6. LoRA 支持

如果模型已经继承 `LoRAPipeline`：

```python
class MyModelPipeline(LoRAPipeline, ComposedPipelineBase):
    ...
```

**权重层命名必须与 Diffusers 官方兼容**，否则 LoRA 的 `target_modules` 匹配不上：

```python
# 官方 Diffusers 权重名称形如:
# transformer.transformer_blocks.0.attn.to_q.weight
# 你的 DiT 必须把层命成相同路径
```

---

## 7. 量化 / 融合 Kernel

启用 `sglang.multimodal_gen` 的融合 kernel：

```python
from sglang.multimodal_gen.runtime.layers.norm import (
    LayerNormScaleShift, RMSNormScaleShift,
)

# AdaLayerNorm 合并：norm + (1 + scale) * x + shift
self.norm1 = LayerNormScaleShift(dim, ...)

def forward(self, x, scale_shift_params):
    x = self.norm1(x, scale_shift_params)   # 一次 kernel 完成三步
```

量化：
- **Nunchaku (FP4 / INT4 PTQ)**：Flux / Qwen-Image 已支持，见 `flux_2_nvfp4.py`、`qwen_image.py`
- **BF16 / FP8**：由 `PipelineConfig.vae_precision`、`transformer` 的 `torch_dtype` 控制

---

## 8. 测试与精度校验

### 8.1 测试文件位置

| 用例类别 | 路径 |
|---|---|
| GPU 集成用例 | [`python/sglang/multimodal_gen/test/server/gpu_cases.py`](../../python/sglang/multimodal_gen/test/server/gpu_cases.py) |
| 测试用例 dataclass / 常量 / 阈值 | [`test/server/testcase_configs.py`](../../python/sglang/multimodal_gen/test/server/testcase_configs.py) |
| Suite 选择 / runtime 分片 / standalone 文件 | [`test/run_suite.py`](../../python/sglang/multimodal_gen/test/run_suite.py) |
| 组件精度 hook | [`test/server/accuracy_hooks.py`](../../python/sglang/multimodal_gen/test/server/accuracy_hooks.py) |
| 组件精度 skip 配置 | [`test/server/accuracy_config.py`](../../python/sglang/multimodal_gen/test/server/accuracy_config.py) |

**运行入口**：

```bash
PYTHONPATH=python python3 python/sglang/multimodal_gen/test/run_suite.py --suite unit

PYTHONPATH=python python3 python/sglang/multimodal_gen/test/run_suite.py \
    --suite component-accuracy-1-gpu -k <case_id>

PYTHONPATH=python python3 python/sglang/multimodal_gen/test/run_suite.py \
    --suite 1-gpu --total-partitions 1 --partition-id 0 -k <case_id>
```

### 8.2 组件精度（component-accuracy）

harness 把 SGLang 的组件（DiT / VAE / Encoder）和 Diffusers/HF 的**逐组件**对比——比 pipeline 级推理更严格。新增用例时必须显式选择以下三种处理方式之一：

| 情况 | 处理 |
|---|---|
| 需要 harness 侧 minimal hook（缺 forward 参数、缺 autocast / runtime 上下文、家族特定输入准备） | 在 `accuracy_hooks.py` 写**最小** hook，**禁止**借机改变对比模式或 harness 行为 |
| 已被别的用例以同源组件 + 同拓扑覆盖（LoRA、Cache-DiT、upscaling 等变体常见） | 在 `accuracy_config.py` 加 skip，注明具体原因 |
| HF/Diffusers 参考组件无法忠实加载/比对（布局缺失、checkpoint 不完整、raw 组件契约不支持、对齐后仍有可证明偏差） | 在 `accuracy_config.py` 加 skip，原因必须**具体且技术化**（禁止 "flaky" / "needs investigation" 这种模糊表述） |

### 8.3 性能 / benchmark

- 用 `warmup excluded` 的 latency 行作为 benchmark 基准数字
- prompt、seed、shape、步数、模型路径、backend、GPU 拓扑**必须固定**
- 去噪性能 + profiler trace 走 [`sglang-diffusion-benchmark-profile`](../../python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/) skill
- 服务级 benchmark 走 [`python/sglang/multimodal_gen/benchmarks/bench_serving.py`](../../python/sglang/multimodal_gen/benchmarks/bench_serving.py)

---

## 9. 提交前自检清单

**通用（Hybrid / Modular 都必须）**：

- [ ] Pipeline 文件在 `runtime/pipelines/{model_name}.py`，末尾有 `EntryClass`
- [ ] `pipeline_name` == Diffusers `model_index.json` 的 `_class_name`
- [ ] `_required_config_modules` 覆盖 `model_index.json` 列出的所有模块
- [ ] `PipelineConfig` 在 `configs/pipeline_configs/{model_name}.py`
- [ ] `SamplingParams` 在 `configs/sample/{model_name}.py`
- [ ] DiT 模型在 `runtime/models/dits/{model_name}.py`（或复用已有）
- [ ] DiT 结构配置在 `configs/models/dits/{model_name}.py`
- [ ] VAE 复用已有（如 `AutoencoderKL`）或新建于 `runtime/models/vaes/`
- [ ] VAE 配置复用或新建于 `configs/models/vaes/{model_name}.py`
- [ ] 在 `registry.py` 里通过 `register_configs()` 登记（别忘了加 `_register_configs()` 里）
- [ ] `PipelineConfig` 回调的 kwargs 键值 == DiT `forward()` 签名
- [ ] Latent scale/shift 数值正确（`get_decode_scale_and_shift`）
- [ ] 用了融合 kernel（`LayerNormScaleShift` / `RMSNormScaleShift` / 等）
- [ ] 权重命名与 Diffusers 对齐，能自动加载
- [ ] 评估过 DiT 的 TP/SP 支持（推荐；TP+SP 参考 `wanvideo.py`、USPAttention 参考 `qwen_image.py`）
- [ ] **输出质量已验证**：非噪声，与 Diffusers 参考输出对齐

**Hybrid 风格特有**：

- [ ] `BeforeDenoisingStage` 在 `stages/model_specific_stages/{model_name}.py`
- [ ] `BeforeDenoisingStage.forward()` 填齐 `DenoisingStage` 需要的所有 batch 字段

---

## 10. 常见陷阱

1. **`batch.sigmas` 必须是 Python list**，不是 numpy array 或 tensor。用 `.tolist()` 转。
2. **`batch.prompt_embeds` 是 tensor 的 list**（每个 encoder 一份），不是单个 tensor。用 `[tensor]` 包一层。
3. 别忘 `batch.raw_latent_shape`，`DecodingStage` 要它来反打包 latent。
4. **RoPE 风格**：`is_neox_style=True` = split-half；`False` = interleaved。对照参考模型认真确认——**搞错会全是噪声**。
5. **VAE 精度**：多数 VAE 需要 fp32 / bf16 才数值稳定，`PipelineConfig.vae_precision` 要设对。
6. **不要**把模型专属逻辑硬塞进共享 Stage。不合身就老老实实 Hybrid 写专属 `BeforeDenoisingStage`，而不是在共享 Stage 里加 `if model_name == ...` 分支。
7. **权重加载不上** 90% 是命名不对。让你的 DiT 层名与 Diffusers 一模一样（包括 `transformer_blocks.0.attn.to_q.weight` 这种层级路径）。
8. **CFG 条件字段在 `prepare_pos_cond_kwargs` / `prepare_neg_cond_kwargs` 不一致** → `DenoisingStage` 会在 `should_use_guidance=True` 时抛 KeyError 或给出错误条件。两个方法返回的 dict 必须结构一致，只有 `encoder_hidden_states` 等条件内容不同。
9. **TP 时 num_heads 不能整除 tp_size** → 直接 `assert` 挂。在 DiT 构造时显式检查。
10. **SP 时 sequence 长度不被 sp_world_size 整除** → 要在 `BeforeDenoisingStage` 里做 padding 对齐（参考 `wanvideo.py` 的 `_pad_for_sequence_parallel`）。
11. **`vae_tiling=True` 但 VAE 实现不支持 tiling** → 静默忽略或输出噪声。仅在 VAE 显式实现了 tile_encode/decode 时启用。
12. **`register_configs()` 漏加** → `--model-path` 指向你的模型时，SGLang 找不到 config，报"unknown model family"。

---

## 11. 进一步阅读

- 入门：[`docs_zh/multimodal_gen/README.md`](../multimodal_gen/README.md)
- 架构总览：[`docs_zh/multimodal_gen/01_architecture_overview.md`](../multimodal_gen/01_architecture_overview.md)
- 注册表与配置：[`docs_zh/multimodal_gen/02_registry_and_config.md`](../multimodal_gen/02_registry_and_config.md)
- 运行时执行：[`docs_zh/multimodal_gen/03_runtime_execution.md`](../multimodal_gen/03_runtime_execution.md)
- Pipeline 与 Stage：[`docs_zh/multimodal_gen/04_pipeline_and_stage.md`](../multimodal_gen/04_pipeline_and_stage.md)
- Loader 与模型组件：[`docs_zh/multimodal_gen/05_loader_and_models.md`](../multimodal_gen/05_loader_and_models.md)
- 三段式拆服务（encode / denoise / decode）：[`docs_zh/multimodal_gen/07_disaggregation_and_optimization.md`](../multimodal_gen/07_disaggregation_and_optimization.md)
- Skill 深度解读（8 号）：[`docs_zh/multimodal_gen/08_skill_add_model.md`](../multimodal_gen/08_skill_add_model.md)
- Wan2.1 完整 case study：[`docs_zh/multimodal_gen/09_case_study_wan2_1.md`](../multimodal_gen/09_case_study_wan2_1.md)

---

**一句话总结**：写扩散模型 = **翻译 Diffusers 的 `__call__` 到 BeforeDenoisingStage + 对齐 DiT forward 签名 + 把 scale/shift/sigma/embedding 四个数值点对齐**。难的不是代码结构，难的是和 Diffusers 参考输出**逐张量对齐**——一旦对齐，后续 TP/SP 分布式、融合 kernel、LoRA / 量化都是框架级"开关"问题。
