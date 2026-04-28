# Skill 详解：`sglang-diffusion-add-model`

> 原始 skill 位置：`python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-add-model/`
>
> - `SKILL.md`：主指令，定义「向 SGLang 新增一个扩散模型/Diffusers Pipeline 变体」的完整方法论
> - `references/testing-and-accuracy.md`：落地测试与组件精度校验的引用文档
>
> 该 skill 的触发时机：**当需要向 `sglang.multimodal_gen` 新增一个扩散模型或 Pipeline 变体时使用。**
> 它是一份「Playbook」，而不是一个自动化脚本，核心是规定两种 Pipeline 风格、实现步骤、各种契约字段、易错点和测试要求。

---

## 1. 背景：为什么需要这个 Skill

`sglang.multimodal_gen` 是一个「统一的扩散/多模态生成运行时」（详见 [01_architecture_overview.md](./01_architecture_overview.md)），上层通过 `DiffGenerator` + `registry.py` 自动识别模型，下层通过 `ComposedPipelineBase + PipelineStage + Executor` 组织推理流程。

新增一个模型意味着同时涉及：

- 模型组件实现（DiT / VAE / Encoder / Scheduler）
- 结构 & 采样 & Pipeline 三类配置
- 若干 Stage 的拼装（要么自己写 `BeforeDenoisingStage`，要么复用标准 Stages）
- Registry 注册
- 权重命名契约、CFG 条件契约、latent scale/shift 契约……

如果没有规范，容易出现「能跑但是出噪声」「DiT 的 forward 签名与条件不匹配」「和 Diffusers 精度对不齐」等典型问题。Skill 就是为此而生的防踩坑指南。

---

## 2. 两种 Pipeline 风格

Skill 把添加模型的路径分成两种风格，明确推荐 Hybrid（A 风格）作为默认。

### 2.1 风格 A：Hybrid 单体 Pipeline（**推荐**）

三段式结构：

```
BeforeDenoisingStage (模型专属)  →  DenoisingStage (标准)  →  DecodingStage (标准)
```

- `BeforeDenoisingStage`：**单一的、模型专属的预处理 Stage**，集中承担输入校验、文本/图像编码、latent 准备、timestep/sigma 计算。每个模型一份。
- `DenoisingStage`：框架标准去噪循环（DiT/UNet 前向），跨模型共享。
- `DecodingStage`：框架标准 VAE 解码，跨模型共享。

**为什么推荐？**  现代扩散模型的前处理差异极大（不同文本编码器、不同 latent packing、不同条件机制）。Hybrid 风格把这些差异封装在**一个** Stage 内，避免在共享 Stage 里塞一堆 `if model_name == ...` 分支；也让开发者可以近乎照搬 Diffusers 参考 Pipeline 的 `__call__`。

### 2.2 风格 B：Modular 组合风格

使用框架已有的细粒度 Stage（`TextEncodingStage` / `LatentPreparationStage` / `TimestepPreparationStage` 等）通过组合搭 Pipeline。典型形式是直接调用：

- `add_standard_t2i_stages()`（文生图）
- `add_standard_ti2i_stages()`（图生图）
- `add_standard_ti2v_stages()`（图生视频）

适用条件：

- 新模型的前处理**大部分可以复用标准 Stage**（例：标准 CLIP/T5 文本编码 + 标准 latent 准备，只需少量定制）。
- 某个模型专属的优化步骤**需要抽成独立 Stage**（便于 profile、并行控制、多 Pipeline 复用）。

参考：`QwenImagePipeline`（T2I、TI2I）、`FluxPipeline`、`WanPipeline`。

### 2.3 如何抉择

| 情境 | 推荐风格 |
|------|----------|
| 模型前处理独特/复杂（VLM captioning、AR token 生成、自定义 latent packing…） | **Hybrid** — 统一到 `BeforeDenoisingStage` |
| 模型能套进标准 T2I / TI2I / TI2V 模式 | **Modular** — 用 `add_standard_*_stages()` |
| 从 Diffusers 移植带多步定制的 pipeline | **Hybrid** — 把 `__call__` 逻辑整体搬进 Stage |
| 给现有模型加变体、主干逻辑共用 | **Modular** — 复用 Stage，用 `PipelineConfig` 回调覆盖差异 |
| 某个预处理步骤需要特别的并行/profiling 隔离 | **Modular** — 把该步骤抽成独立 Stage |

**核心契约（两种风格通用）**：走进 `DenoisingStage` 之前，`Req` batch 上的标准张量字段（latents、timesteps、prompt_embeds…）必须齐备。只要这个契约满足，后续 Pipeline 就是可组合的。

---

## 3. 关键文件与目录速查

skill 给出的关键路径，按职责分层：

| 用途 | 路径 |
|------|------|
| Pipeline 类 | `runtime/pipelines/` |
| 模型专属 Stage | `runtime/pipelines_core/stages/model_specific_stages/` |
| `PipelineStage` 基类 | `runtime/pipelines_core/stages/base.py` |
| Pipeline 基类 | `runtime/pipelines_core/composed_pipeline_base.py` |
| 标准 Stage（Denoising / Decoding 等） | `runtime/pipelines_core/stages/` |
| Pipeline 配置 | `configs/pipeline_configs/` |
| 采样参数 | `configs/sample/` |
| DiT 模型实现 | `runtime/models/dits/` |
| VAE 实现 | `runtime/models/vaes/` |
| Encoder 实现 | `runtime/models/encoders/` |
| Scheduler 实现 | `runtime/models/schedulers/` |
| 模型/VAE/DiT 结构配置 | `configs/models/dits/`、`vaes/`、`encoders/` |
| 中央注册表 | `registry.py` |

---

## 4. 九步实施流程（核心）

Skill 把添加一个模型的过程拆成 9 个严格的步骤。下面逐步展开。

### Step 1 — 取得并研读参考实现

> **写任何代码前**，必须先拿到模型的官方参考实现或 Diffusers Pipeline 源码。不准拍脑袋猜架构。

信息来源优先级：

1. Diffusers 里的 `pipeline_*.py` 源码（或 HuggingFace 仓库里带的 pipeline）
2. 作者官方 GitHub 仓库的 reference 实现
3. 至少给出 HuggingFace 模型 ID，便于查 `model_index.json` 和对应 pipeline 类

要从参考代码中识别出来的 6 件事：

1. `model_index.json` 中必需的模块（`text_encoder` / `vae` / `transformer` / `scheduler` …）
2. 文本 prompt 如何编码
3. latent 如何准备（形状、dtype、scale）
4. timesteps / sigmas 如何计算
5. DiT / UNet 接受哪些条件 kwargs
6. 去噪循环细节（CFG、guidance_scale…）与 VAE 解码细节（scaling factor、tiling…）

### Step 2 — 评估「复用现有 Pipeline / Stage」的可能性

> 禁止盲目造新文件。只有当现有实现需要大改、或者找不到架构相似的实现时，才创建新的 Pipeline / Stage。

决策清单：

1. 对比新模型与现有 Pipeline（Flux、Wan、Qwen-Image、GLM-Image、HunyuanVideo、LTX…）。结构类似就优先：
   - 在现有 Pipeline 上加一个 config 变体，而不是新建 Pipeline 类
   - 复用现有 `BeforeDenoisingStage`，只改参数
   - 若匹配标准模式，直接使用 `add_standard_t2i_stages()` / `add_standard_ti2i_stages()` / `add_standard_ti2v_stages()`
2. 检查 `runtime/pipelines_core/stages/` 和 `stages/model_specific_stages/`。若已有 Stage 能覆盖 80%+ 需求，扩展而不是复制。
3. 检查现有模型组件 — VAE（如 `AutoencoderKL`）、文本编码器（CLIP、T5）、scheduler 很多能直接复用。

### Step 3 — 实现模型组件

**DiT / Transformer**：`runtime/models/dits/{model_name}.py`

关键点：

- 使用 SGLang 的融合算子（`LayerNormScaleShift`、`RMSNormScaleShift`）
- 使用 SGLang 的注意力后端选择器（`get_attn_backend`）
- **参数命名与 Diffusers 保持一致**，以便权重自动加载

**分布式支持（TP / SP）—— 推荐加**。可在单卡验证通过后增量补齐。

- `wanvideo.py`：完整 TP + SP 参考
  - TP：`ColumnParallelLinear`（Q/K/V 投影）+ `RowParallelLinear`（输出投影），注意力 head 按 `tp_size` 切分
  - SP：序列维度 shard，`get_sp_world_size()`、padding 对齐、`sequence_model_parallel_all_gather` 聚合
  - Cross-attention 跳过 SP（`skip_sequence_parallel=is_cross_attention`）
- `qwen_image.py`：SP + USPAttention 参考
  - SP：`USPAttention`（Ulysses + Ring Attention），通过 `--ulysses-degree` / `--ring-degree` 配置
  - TP：`MergedColumnParallelLinear` (QKV + Nunchaku 量化) / `ReplicatedLinear`

> 上面两者只是参考。具体要根据本模型的 head 数、序列维度可分性、哪些 Linear 适合 column/row parallel、哪些模块要排除 SP 来决定。

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
)
```

**VAE / Encoders / Schedulers**：只有在模型用了非标准实现时才写新文件，否则复用。

### Step 4 — 写模型结构配置

三个 `@dataclass`：

- **DiT 配置** `configs/models/dits/{model_name}.py`：继承 `DiTConfig`，主要放 `arch_config`（`in_channels` / `num_layers` / `patch_size` …）
- **VAE 配置** `configs/models/vaes/{model_name}.py`：继承 `VAEConfig`，定义 `vae_scale_factor` 等
- **采样参数** `configs/sample/{model_name}.py`：继承 `SamplingParams`，定义 `num_inference_steps` / `guidance_scale` / 默认 `height` / `width`

### Step 5 — 写 `PipelineConfig`

`PipelineConfig` 承载**静态配置** + **给标准 Stage 用的回调方法**。模板骨架：

```python
@dataclass
class MyModelPipelineConfig(ImagePipelineConfig):
    task_type: ModelTaskType = ModelTaskType.T2I
    vae_precision: str = "bf16"
    should_use_guidance: bool = True
    vae_tiling: bool = False
    enable_autocast: bool = False

    dit_config: DiTConfig = field(default_factory=MyModelDitConfig)
    vae_config: VAEConfig = field(default_factory=MyModelVAEConfig)

    # --- DenoisingStage 用 ---
    def get_freqs_cis(self, batch, device, rotary_emb, dtype): ...
    def prepare_pos_cond_kwargs(self, batch, latent_model_input, t, **kwargs): ...
    def prepare_neg_cond_kwargs(self, batch, latent_model_input, t, **kwargs): ...

    # --- DecodingStage 用 ---
    def get_decode_scale_and_shift(self):
        return self.vae_config.latents_std, self.vae_config.latents_mean

    def post_denoising_loop(self, latents, batch):
        return latents.to(torch.bfloat16)

    def post_decoding(self, frames, server_args):
        return frames
```

> **关键契约**：`prepare_pos_cond_kwargs` / `prepare_neg_cond_kwargs` 返回的 dict 键值**必须与 DiT 的 `forward()` 签名完全一致**，否则 DenoisingStage 会直接报 TypeError 或给出错误的条件。

### Step 6 — 实现 `BeforeDenoisingStage`（Hybrid 的心脏）

位置：`runtime/pipelines_core/stages/model_specific_stages/{model_name}.py`。

职责：

- 输入校验
- 文本 / 图像编码
- Latent 准备
- Timestep / sigma 调度

主要流程（与 Diffusers 参考 `__call__` 前半段一一对应）：

1. 取 device / dtype / generator
2. `_encode_prompt(...)` → `prompt_embeds`、`negative_prompt_embeds`
3. `_prepare_latents(...)` → `latents`
4. `_prepare_timesteps(...)` → `timesteps`、`sigmas`
5. **把结果挂到 `batch` 上**，供 `DenoisingStage` 使用

`DenoisingStage` 期望的 `batch` 字段（**全部必须在 `BeforeDenoisingStage.forward()` 中填好**）：

| 字段 | 类型 | 含义 |
|------|------|------|
| `batch.latents` | `torch.Tensor` | 初始噪声 latent |
| `batch.timesteps` | `torch.Tensor` | 时间步调度 |
| `batch.num_inference_steps` | `int` | 去噪步数 |
| `batch.sigmas` | `list[float]` | sigma 调度（**必须是 Python list**） |
| `batch.prompt_embeds` | `list[torch.Tensor]` | 正向 prompt embedding（**外层 list 包装**） |
| `batch.negative_prompt_embeds` | `list[torch.Tensor]` | 负向 prompt embedding（同上） |
| `batch.generator` | `torch.Generator` | 复现用 RNG |
| `batch.raw_latent_shape` | `tuple` | 原始 latent 形状（供 `DecodingStage` 反打包用） |
| `batch.height` / `batch.width` | `int` | 输出分辨率 |

### Step 7 — 定义 Pipeline 类

Pipeline 类只负责把 Stage 串起来，文件极薄。模板：

```python
class MyModelPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "MyModelPipeline"  # 必须与 model_index.json 的 _class_name 一致

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler",
        # ... 对齐 model_index.json 列出的所有模块
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


EntryClass = [MyModelPipeline]   # registry 发现入口
```

### Step 8 — 注册模型

在 `python/sglang/multimodal_gen/registry.py` 中通过 `register_configs` 注册：

```python
register_configs(
    model_family="my_model",
    sampling_param_cls=MyModelSamplingParams,
    pipeline_config_cls=MyModelPipelineConfig,
    hf_model_paths=[
        "org/my-model-name",   # HuggingFace 模型 ID，支持多个
    ],
)
```

Pipeline 类本身由 `_discover_and_register_pipelines()` 自动发现 `EntryClass`，**不需要手动再注册**。

### Step 9 — 验证输出质量

> **输出是噪声 = 实现错误。不允许凑合上线。**

典型噪声成因（按排查优先级）：

1. `get_decode_scale_and_shift` 返回了错误的 latent scale / shift
2. timestep / sigma 调度方向、dtype、数值区间不对
3. 条件 kwargs 字段名与 DiT `forward()` 签名对不上
4. VAE 解码配置错误（`vae_scale_factor` 错、漏反归一化…）
5. 旋转位置编码风格错（`is_neox_style=True`：split-half；`False`：interleaved）
6. prompt embedding 格式错（漏 list 包装、选错 encoder 输出）

调试手段：

- 和 Diffusers 参考 pipeline **同 seed 并排跑**，比对中间张量（latents、prompt_embeds、timesteps）
- 独立检查每个 Stage 输出的 shape 与数值范围

---

## 5. 参考实现一览

### Hybrid 风格（多数新模型建议用这条）

| 模型 | Pipeline | BeforeDenoisingStage | PipelineConfig |
|------|----------|---------------------|----------------|
| GLM-Image | `runtime/pipelines/glm_image.py` | `stages/model_specific_stages/glm_image.py` | `configs/pipeline_configs/glm_image.py` |
| Qwen-Image-Layered | `runtime/pipelines/qwen_image.py` (`QwenImageLayeredPipeline`) | `stages/model_specific_stages/qwen_image_layered.py` | `configs/pipeline_configs/qwen_image.py`（`QwenImageLayeredPipelineConfig`） |

### Modular 风格（标准 Stage 合身时）

| 模型 | Pipeline | 说明 |
|------|----------|------|
| Qwen-Image (T2I) | `runtime/pipelines/qwen_image.py` | `add_standard_t2i_stages()` |
| Qwen-Image-Edit | `runtime/pipelines/qwen_image.py` | `add_standard_ti2i_stages()` |
| Flux | `runtime/pipelines/flux.py` | `add_standard_t2i_stages()` + 自定义 `prepare_mu` |
| Wan | `runtime/pipelines/wan_pipeline.py` | `add_standard_ti2v_stages()` |

---

## 6. 提交前自检清单

**通用（两种风格都要过）**：

- [ ] Pipeline 文件在 `runtime/pipelines/{model_name}.py`，含 `EntryClass`
- [ ] `PipelineConfig` 在 `configs/pipeline_configs/{model_name}.py`
- [ ] `SamplingParams` 在 `configs/sample/{model_name}.py`
- [ ] DiT 模型在 `runtime/models/dits/{model_name}.py`
- [ ] DiT 结构配置在 `configs/models/dits/{model_name}.py`
- [ ] VAE — 复用已有（如 `AutoencoderKL`）或新建于 `runtime/models/vaes/`
- [ ] VAE 配置 — 复用或新建于 `configs/models/vaes/{model_name}.py`
- [ ] 在 `registry.py` 通过 `register_configs()` 注册
- [ ] `pipeline_name` == Diffusers `model_index.json` 的 `_class_name`
- [ ] `_required_config_modules` 覆盖 `model_index.json` 全部模块
- [ ] `PipelineConfig` 回调键值 == DiT `forward()` 签名
- [ ] Latent scale/shift 正确
- [ ] 用了已有的融合 kernel（参照 benchmark/profile skill 的 `existing-fast-paths.md`）
- [ ] 权重名对齐 Diffusers，能自动加载
- [ ] 评估了 DiT 的 TP/SP 支持（推荐；TP+SP 参考 `wanvideo.py`，USPAttention 参考 `qwen_image.py`）
- [ ] **输出质量已验证**：非噪声，与 Diffusers 参考输出对齐

**仅 Hybrid 风格**：

- [ ] `BeforeDenoisingStage` 在 `stages/model_specific_stages/{model_name}.py`
- [ ] `BeforeDenoisingStage.forward()` 把 `DenoisingStage` 需要的所有 batch 字段填齐

---

## 7. 常见陷阱（Common Pitfalls）

1. **`batch.sigmas` 必须是 Python list**，不是 numpy array。用 `.tolist()` 转。
2. **`batch.prompt_embeds` 是 tensor 的 list**（每个 encoder 一份），不是单个 tensor。用 `[tensor]` 包一层。
3. 别忘 `batch.raw_latent_shape`，`DecodingStage` 要它来反打包 latent。
4. 旋转位置编码风格：`is_neox_style=True` = split-half 旋转；`False` = interleaved。对照参考模型认真确认。
5. VAE 精度：很多 VAE 需要 fp32 / bf16 才能数值稳定。`PipelineConfig.vae_precision` 设置好。
6. **不要**把模型专属逻辑硬塞进共享 Stage。不合身就老老实实走 Hybrid 写专属 `BeforeDenoisingStage`，而不是往共享 Stage 里加分支判断。

---

## 8. 实现之后：测试与组件精度（`references/testing-and-accuracy.md`）

模型跑出**非噪声输出**之后，skill 要求进入测试 & 精度阶段。这部分由 `references/testing-and-accuracy.md` 单独管。

### 8.1 测试代码放在哪里

- GPU 集成用例 → `test/server/gpu_cases.py`
- 复用的 dataclass / 常量 / 阈值 / 用例工厂 → `test/server/testcase_configs.py`
- Suite 选择、runtime 分片、standalone 文件 → `test/run_suite.py`（**不要**在别处硬编码 CI 分片列表）
- 新增 standalone 文件加进 suite 时，拿到第一次 CI 实测 runtime 后更新 `STANDALONE_FILE_EST_TIMES`

常用本地入口：

```bash
PYTHONPATH=python python3 python/sglang/multimodal_gen/test/run_suite.py --suite unit
PYTHONPATH=python python3 python/sglang/multimodal_gen/test/run_suite.py --suite component-accuracy-1-gpu -k <case_id>
PYTHONPATH=python python3 python/sglang/multimodal_gen/test/run_suite.py --suite 1-gpu --total-partitions 1 --partition-id 0 -k <case_id>
```

### 8.2 组件精度决策（新增 GPU 用例时必做）

组件精度（component-accuracy）harness 会把 SGLang 的组件和 Diffusers/HF 参考组件**逐组件**对比，比 pipeline 级推理更严格。新增 `ONE_GPU_CASES` / `TWO_GPU_CASES` / B200 专用分组时，必须明确决策（不能扔给 CI 去发现）：

| 情况 | 处理方式 |
|------|----------|
| 1. 该家族需要 harness 侧的最小 hook（缺必需 forward 参数、缺 autocast/runtime 上下文、家族特定输入准备…） | 在 `test/server/accuracy_hooks.py` 写**最小** hook。禁止借机改变对比模式或改 harness 行为去「让测试通过」 |
| 2. 该组件已被别的用例以同源组件、同拓扑覆盖（LoRA、Cache-DiT、upscaling 等变体常见） | 在 `test/server/accuracy_config.py` 加 skip，并写具体原因，例：`Representative VAE accuracy is already covered by ... for the same source component and topology` |
| 3. HF/Diffusers 参考组件无法忠实加载/比对（HF 组件布局缺失或不支持、checkpoint 不完整、raw 组件契约不支持、对齐后仍有可证明的偏差…） | 在 `accuracy_config.py` 加 skip，原因必须**具体且技术化**，禁止 "flaky"、"needs investigation" 这种模糊表述 |

### 8.3 Follow-up 覆盖范围

模型跑通且输出质量验证过之后，按用户要求的深度补测试 & 基准。若用户没说深度，先**提出最小可用验证集**再启动长时 GPU 运行。

建议的测试维度：

- Pipeline 构造与 Stage 接线
- 单卡推理产出非噪声
- 如支持 TP/SP，补多卡推理
- 为新增的数学 / 解析 / 调度 / loader 行为补单测

性能数据规范：

- 使用 `warmup excluded` 的 latency 行作为命令行生成的基准数字
- prompt、seed、shape、步数、模型路径、后端、GPU 拓扑**固定**
- 去噪性能 + profiler trace 走 `sglang-diffusion-benchmark-profile` skill
- 服务级基准走 `python/sglang/multimodal_gen/benchmarks/bench_serving.py`

---

## 9. 与本目录其他文档的关系

- 底层机制/数据流问题 → 先读 [03_runtime_execution.md](./03_runtime_execution.md) 和 [04_pipeline_and_stage.md](./04_pipeline_and_stage.md)
- `registry.py` 的识别流程与 `PipelineConfig` 归一化 → [02_registry_and_config.md](./02_registry_and_config.md)
- `ComponentLoader` / 权重加载与模型组件位置 → [05_loader_and_models.md](./05_loader_and_models.md)
- 三段式（encode / denoise / decode）拆服务与性能优化 → [07_disaggregation_and_optimization.md](./07_disaggregation_and_optimization.md)

简而言之：**这份 skill 把「加模型」从「读源码拼接」降级成「按清单补齐」**，代价是要严格遵守 `DenoisingStage` / `DecodingStage` 的契约字段与回调签名。一旦契约对齐，Pipeline 本身就是一行行 `add_stage(...)` 的积木。
