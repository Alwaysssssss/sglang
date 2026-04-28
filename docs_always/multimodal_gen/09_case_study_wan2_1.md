# 案例实战：以 Wan2.1 为例，向 `multimodal_gen` 新增一个 DiT 模型

> 本文是 [08_skill_add_model.md](./08_skill_add_model.md) 的实战篇。
>
> - 08 讲的是"通用方法论和两种 Pipeline 风格"
> - 本文讲的是"如果以 Wan2.1 为目标，这 9 步分别在哪几个文件、哪几行落地"
>
> 建议两篇配合读：遇到不理解的术语（契约、回调、Stage 等）回到 08 查；需要"我现在要在哪里写代码"时看本篇。

Wan2.1 是一个视频扩散（DiT）模型，官方 HuggingFace 路径如 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`。在 SGLang `multimodal_gen` 中它走的是 **Modular 组合风格**——Pipeline 类非常薄，核心是调用 `add_standard_t2i_stages()`，所有模型差异都塞在 `PipelineConfig` + 结构 / 采样配置 + DiT 实现里。本文以它为范本，手把手讲清楚"新增一个 Wan 家族 DiT 模型"要碰哪些文件。

---

## 1. 落地前先回答的 3 个问题

这是 08 提到的"Step 1：研读参考实现"的 Wan2.1 实例化。

| 问题 | Wan2.1 的答案 |
|------|---------------|
| `model_index.json` 中的 `_class_name`？ | `WanPipeline`（T2V）或 `WanImageToVideoPipeline`（I2V）。**这个名字就是 `pipeline_name` 的唯一真源。** |
| 需要哪些组件？ | `text_encoder` + `tokenizer`（UMT5-XXL）、`transformer`（Wan DiT）、`vae`（Wan3D VAE）、`scheduler`（Wan2.1 自家的 UniPC Multistep）；I2V 变体再加 `image_encoder`（CLIP Vision）+ `image_processor` |
| 走 Hybrid 还是 Modular？ | **Modular**：文本编码、latent 准备、timestep 准备都能被标准 Stage 覆盖。唯一的特殊点是 scheduler 要用 Wan 官方 UniPC 而非 Diffusers 那版——在 `initialize_pipeline` 里手动覆写就够了 |

如果你要做的新模型三件事都和 Wan2.1 相近（同 transformer 家族、同编码器、只是架构数字不同），几乎可以抄 Wan 的 layout；否则请先判断是否要走 Hybrid（另起 `BeforeDenoisingStage`）。

---

## 2. 全景：Wan2.1 在代码里是怎么"散"开的

把"Wan2.1"这个概念拆开看，它在仓库里对应 **7 个层面的文件**，每个层面职责非常单一：

```
                        Wan2.1（概念）
 ┌────────────┬──────────────┬────────────┬────────────┬──────────────┐
 │            │              │            │            │              │
 注册表      Pipeline       PipelineConfig  SamplingParams  DiT 配置+实现  VAE 配置+实现
 registry.py wan_pipeline.py  wan.py         wan.py         wanvideo.py     wanvae.py
```

下表给出每个文件的具体路径和职责，后续步骤会一一对应回到这里：

| 职责 | 路径 | Wan2.1 对应内容 |
|------|------|-----------------|
| 模型识别 | `python/sglang/multimodal_gen/registry.py` | `register_configs(hf_model_paths=["Wan-AI/Wan2.1-T2V-1.3B-Diffusers"], ...)` |
| Pipeline 类（T2V） | `runtime/pipelines/wan_pipeline.py` | `WanPipeline`（`pipeline_name = "WanPipeline"`） |
| Pipeline 类（I2V） | `runtime/pipelines/wan_i2v_pipeline.py` | `WanImageToVideoPipeline` |
| PipelineConfig | `configs/pipeline_configs/wan.py` | `WanT2V480PConfig` / `WanT2V720PConfig` / `WanI2V480PConfig` / `WanI2V720PConfig` / …… |
| SamplingParams | `configs/sample/wan.py` | `WanT2V_1_3B_SamplingParams` / `WanT2V_14B_SamplingParams` / …… |
| DiT 结构配置 | `configs/models/dits/wanvideo.py` | `WanVideoArchConfig`（attention heads / layers / patch…）、`WanVideoConfig`（套壳） |
| DiT 实现 | `runtime/models/dits/wanvideo.py` | `WanVideoTransformer3DModel`（TP + SP 就绪） |
| VAE 结构配置 | `configs/models/vaes/wanvae.py` | `WanVAEArchConfig` / `WanVAEConfig`（含 latent mean/std） |
| VAE 实现 | `runtime/models/vaes/wanvae.py` | `WanVAE3D` |
| Scheduler | `runtime/models/schedulers/scheduling_flow_unipc_multistep.py` | `FlowUniPCMultistepScheduler` |

**一句话记住结构**：`registry` 找得到你 → `Pipeline` 把 Stage 串起来 → `PipelineConfig` 承载静态配置 + 给标准 Stage 的回调 → `DiT / VAE / Scheduler` 负责真正的计算。

---

## 3. 九步落地 Wan2.1（按代码位置对齐）

下面每一步都与 08 的 9 步一一对应，重点是"对应到 Wan2.1 具体是哪个文件的哪段代码"。

### Step 1 — 研读参考实现

Wan2.1 的官方参考实现有两份：

- Diffusers 的 `WanPipeline`（`src/diffusers/pipelines/wan/pipeline_wan.py`）
- 阿里官方 repo `https://github.com/Wan-Video/Wan2.1`

对比之后确认如下事实，这些事实会直接写进我们各个 config：

- DiT patch size = `(1, 2, 2)`、`text_len = 512`、`in/out channels = 16`、`text_dim = 4096`（UMT5-XXL 隐藏维度）
- VAE: 3D VAE，时间步长 4、空间步长 8，`z_dim = 16`，每维都有自己的 `latents_mean` / `latents_std`（详见 `wanvae.py`）
- Scheduler：阿里官方用 UniPC multistep，**不是** Diffusers 默认的那一版
- T2V 1.3B 默认 `flow_shift = 3.0`、`guidance_scale = 3.0`、50 steps；T2V 14B 默认 `flow_shift = 5.0`、`guidance_scale = 5.0`

### Step 2 — 评估是否能复用已有 Pipeline

如果你要加的是一个 **"结构仍然是 Wan"** 的新变体（例如蒸馏版、Fun 版、Turbo 版、新的分辨率档），决策非常清楚：**不要**新写 Pipeline 类，只新写一个 `PipelineConfig` + `SamplingParams`。

Wan 家族现在已经有 8+ 个变体全部复用同一个 `WanPipeline` / `WanImageToVideoPipeline` 类，差异只体现在 config 层（`registry.py` 里约 80 行的连续 `register_configs(...)` 就是例子）：

- `WanT2V480PConfig` / `WanT2V720PConfig`：不同分辨率档
- `TurboWanT2V480PConfig`：蒸馏模型，`flow_shift = 8.0`，多一个 `dmd_denoising_steps`
- `FastWan2_1_T2V_480P_Config`：DMD 加速
- `Wan2_2_TI2V_5B_Config` / `Wan2_2_T2V_A14B_Config` / `Wan2_2_I2V_A14B_Config`：Wan2.2，但 Pipeline 类仍复用

只有当新模型用到了"Wan 根本没有的能力"（不同的 cross-attention 结构、全新的输入模态、不一样的 latent 打包方式）时才值得新建一个 Pipeline 类；多数情况下**扩 config 即可**。

### Step 3 — 实现（或复用）模型组件

Wan2.1 的组件：

- **DiT**：`runtime/models/dits/wanvideo.py::WanVideoTransformer3DModel`
  - 关键：参数名和 Diffusers 的 `WanTransformer3DModel` 保持一致（借由 `param_names_mapping` 正则映射，见 `configs/models/dits/wanvideo.py`），这样权重能直接加载
  - 使用 SGLang 的融合算子：`LayerNormScaleShift` / `RMSNorm` / `ModulateProjection` / `PatchEmbed` / `TimestepEmbedder`
  - 已经实现了 **TP + SP**：
    - TP：QKV 用 `ColumnParallelLinear`，输出用 `RowParallelLinear`，注意力 head 按 `get_tp_world_size()` 切
    - SP：序列维度 shard，cross-attention 用 `skip_sequence_parallel=True` 跳过 SP
    - 如果你做的新 Wan 变体改了 head 数 / head dim，只要保证能被 `tp_size` 整除，上面这套都自动生效
- **VAE**：`runtime/models/vaes/wanvae.py::WanVAE3D`。支持 parallel encode/decode、feature cache
- **Encoder**：UMT5-XXL，直接复用 `runtime/models/encoders/t5.py`。**无需**为 Wan 新写 Encoder
- **Scheduler**：`runtime/models/schedulers/scheduling_flow_unipc_multistep.py::FlowUniPCMultistepScheduler`

> 如果你的新 Wan 变体只换了主干参数（层数、head 数、`rope_max_seq_len` 等），**不要**新建一个 DiT 类，改 `WanVideoArchConfig` 的 dataclass 默认值或在 `PipelineConfig.__post_init__` 里改它即可。

### Step 4 — 写模型结构配置

**DiT arch config** `configs/models/dits/wanvideo.py::WanVideoArchConfig`。已经定义好 40+ 字段，新加变体通常只需要改这几个：

```11:25:sglang/python/sglang/multimodal_gen/configs/models/dits/wanvideo.py
@dataclass
class WanVideoArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_blocks])

    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^patch_embedding\.(.*)$": r"patch_embedding.proj.\1",
            ...
```

> `param_names_mapping` / `reverse_param_names_mapping` / `lora_param_names_mapping` 三张映射表是 Wan 能自动加载 Diffusers 权重与各种 LoRA 的关键。如果你改了 DiT 实现里 Linear 的命名，记得同步更新这三张表。

**VAE arch config** `configs/models/vaes/wanvae.py::WanVAEArchConfig` 里有两组常值必须和参考权重完全一致：

```21:56:sglang/python/sglang/multimodal_gen/configs/models/vaes/wanvae.py
    latents_mean: tuple[float, ...] = (
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715,  0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
    )
    latents_std: tuple[float, ...] = (
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
    )
```

**这组数就是 `DecodingStage` 用的 scale 和 shift。08 文档反复强调"输出是噪声 = 实现错误"，最高频的成因就是这里贴错、贴歪了一个维度。**

**SamplingParams** `configs/sample/wan.py`。T2V 1.3B 默认长这样：

```45:76:sglang/python/sglang/multimodal_gen/configs/sample/wan.py
@dataclass
class WanT2V_1_3B_SamplingParams(SamplingParams):
    height: int = 480
    width: int = 832
    num_frames: int = 81
    fps: int = 16
    guidance_scale: float = 3.0
    negative_prompt: str = (
        "Bright tones, overexposed, static, blurred details, ..."
    )
    num_inference_steps: int = 50
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [(832, 480), (480, 832)],
    )
    teacache_params: TeaCacheParams = field(
        default_factory=lambda: TeaCacheParams(
            teacache_thresh=0.08, use_ret_steps=True, ...
        )
    )
```

**新增变体的做法**：继承已有的 `WanT2V_1_3B_SamplingParams` / `WanT2V_14B_SamplingParams`，只改你需要改的字段（默认分辨率、guidance_scale、teacache 系数等）。

### Step 5 — 写 `PipelineConfig`

**核心文件**：`configs/pipeline_configs/wan.py`。T2V 480P 基类：

```58:93:sglang/python/sglang/multimodal_gen/configs/pipeline_configs/wan.py
@dataclass
class WanT2V480PConfig(PipelineConfig):
    """Base configuration for Wan T2V 1.3B pipeline architecture."""

    task_type: ModelTaskType = ModelTaskType.T2V
    dit_config: DiTConfig = field(default_factory=WanVideoConfig)
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    flow_shift: float | None = 3.0

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(),)
    )
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = (
        field(default_factory=lambda: (t5_postprocess_text,))
    )

    precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp32",))

    def __post_init__(self):
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True
```

几个要点：

1. **`task_type`** 决定了 SamplingParams 的输入校验（比如 I2V 必须传 `image_path`）。Wan 的 T2V 用 `T2V`、I2V 用 `I2V`、Wan2.2 的 5B 模型还出现了 `TI2V`
2. **`text_encoder_configs` + `postprocess_text_funcs`** 是 `TextEncodingStage` 的契约：每个 encoder 一份 config，对应一个后处理函数。Wan 只用 T5 一个编码器，所以都是 1 元组
3. **`postprocess_text_funcs`** 是"encoder 输出 → prompt_embeds tensor"的唯一钩子。Wan 的写法见文件顶部 `t5_postprocess_text`：按 attention_mask 截断、再 pad 到 `text_len = 512`
4. **`flow_shift`** 是 `FlowUniPCMultistepScheduler` 的参数，在 `Pipeline.initialize_pipeline` 里读取
5. **`vae_config.load_encoder / load_decoder`** 决定 VAE 加载哪一半权重。T2V 只需要 decoder，I2V 两边都要——这个细节不处理会白白占 VAE encoder 的显存
6. 若新变体需要 DMD、Turbo、自定义 latent shape，就像这份文件里 `TurboWanT2V480PConfig` / `Wan2_2_TI2V_5B_Config` 那样继承基类、覆写字段或 `prepare_latent_shape`

> Wan 走 Modular 路线，因此**没有**写 `prepare_pos_cond_kwargs` / `prepare_neg_cond_kwargs` 这种 Hybrid 专用回调——它们是共享 `DenoisingStage` 要的，而 Wan 的标准 `DenoisingStage` 已经能覆盖默认情况。如果你的变体需要往 DiT 的 forward 塞额外 kwargs（比如 I2V 的 `encoder_hidden_states_image`），建议看 08 文档和 `stages/denoising.py` 里 `PipelineConfig` 暴露出来的回调签名。

### Step 6 — `BeforeDenoisingStage`？Wan 不需要

因为 Wan 走 Modular：它在 Pipeline 里直接调 `add_standard_t2i_stages()`，所有前处理都由框架内置的标准 Stage 接管：

- `InputValidationStage`：校验 `task_type`、分辨率、`num_frames` 是否合法
- `TextEncodingStage`：用 `text_encoder_configs` + `postprocess_text_funcs` 把 prompt 编码成 `prompt_embeds`
- `LatentPreparationStage`：按 `vae_scale_factor_*` 和 `dit_config.arch_config.in_channels` 生成初始噪声 latent
- `TimestepPreparationStage`：按 scheduler 产出 `timesteps` / `sigmas`
- `DenoisingStage`：跑去噪循环
- `DecodingStage`：VAE 解码

> **只有当你确定"标准 Stage 装不下"时**（比如要调 VLM 做 caption、要 AR 生成 token、要自定义 latent packing），才需要写一个 `BeforeDenoisingStage` 并换成 Hybrid 风格。Wan2.1 到 Wan2.2 加了 `expand_timesteps`、`boundary_ratio` 这些东西，也都是通过 `PipelineConfig` 回调就搞定了，没升级到 Hybrid。

### Step 7 — 定义 Pipeline 类

Wan T2V 的全部实现：

```24:49:sglang/python/sglang/multimodal_gen/runtime/pipelines/wan_pipeline.py
class WanPipeline(LoRAPipeline, ComposedPipelineBase):
    """
    Wan video diffusion pipeline with LoRA support.
    """

    pipeline_name = "WanPipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def initialize_pipeline(self, server_args: ServerArgs):
        # We use UniPCMScheduler from Wan2.1 official repo, not the one in diffusers.
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(
            shift=server_args.pipeline_config.flow_shift
        )

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_standard_t2i_stages()


EntryClass = WanPipeline
```

整个 Pipeline 类只做 3 件事：

1. `pipeline_name = "WanPipeline"`：**必须** 和 HuggingFace `model_index.json` 里的 `_class_name` 一致，这是注册表去找 Pipeline 的唯一凭据
2. `_required_config_modules`：与 `model_index.json` 里组件列表对齐；`ComponentLoader` 就靠这个加载对应模块
3. `initialize_pipeline`：框架默认会按 Diffusers 的类名去实例化 scheduler，但 Wan2.1 官方的 UniPC 和 Diffusers 那一版不兼容，所以在这里手动覆写。这种"覆写 scheduler"是 Pipeline 类常见的一点定制
4. `create_pipeline_stages`：就一行 `add_standard_t2i_stages()`。I2V 版本（`wan_i2v_pipeline.py`）则是 `add_standard_ti2v_stages()`

**`EntryClass = WanPipeline`** 这行是告诉 registry"这是我的 Pipeline 入口"——`_discover_and_register_pipelines()` 扫所有 `runtime/pipelines/*.py`，凡是带 `EntryClass` 属性的就把它的 `pipeline_name` 注册进全局 `_PIPELINE_REGISTRY`。**不用手动再注册 Pipeline 类。**

### Step 8 — 在 `registry.py` 注册 Config

这是唯一一处需要你加 import + 写一段 `register_configs(...)` 的地方。Wan2.1 T2V 1.3B 的注册：

```652:660:sglang/python/sglang/multimodal_gen/registry.py
    # Wan
    register_configs(
        sampling_param_cls=WanT2V_1_3B_SamplingParams,
        pipeline_config_cls=WanT2V480PConfig,
        hf_model_paths=[
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        ],
        model_detectors=[lambda hf_id: "wanpipeline" in hf_id.lower()],
    )
```

三个要点：

1. **`sampling_param_cls`** 必须给——它决定 CLI / SDK 看到什么默认采样参数
2. **`pipeline_config_cls`** 必须给——它决定加载哪份 DiT / VAE / text-encoder 结构
3. **`hf_model_paths` 或 `model_detectors` 至少给一个**：
   - `hf_model_paths`：精确 / 前缀 / 短名 / HF cache 路径都能 hit
   - `model_detectors`：当用户的 HF id 不在 `hf_model_paths` 里，但 `model_index.json` 的 `_class_name` 匹配某个 lambda 时，仍可命中（兜底）

Wan 家族后面还有十几条 `register_configs(...)` 把 14B、I2V 480/720P、Turbo、DMD、Fun、Wan2.2 全挂上同一个 `WanPipeline` 类——就是在复用这套机制。

> 新增变体的 registry 条目**放在哪儿**？看 `registry.py` 的 `_register_configs()`，Wan 家族都在 `# Wan` 注释下方连续写，保持同一家族的所有变体 **紧邻在一起**，别乱插。

### Step 9 — 验证输出质量

跑一下最小 smoke 看是不是噪声：

```bash
PYTHONPATH=python python -m sglang.multimodal_gen.runtime.entrypoints.cli generate \
    --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A curious raccoon peers through a vibrant field of yellow sunflowers." \
    --height 480 --width 832 --num-frames 81 --num-inference-steps 5 \
    --save-output --output-path ./smoke_out/
```

- 输出 5 帧是预期的：`num_inference_steps=5` 是为了抢 smoke 速度，真 sampling 请用默认 50
- 如果得到噪声，按 08 文档第 9 步的"噪声排查清单"从前往后查，Wan 最典型的锅排序是：
  1. `WanVAEArchConfig.latents_mean` / `latents_std` 少贴一个维度或贴反
  2. `flow_shift` 与 `FlowUniPCMultistepScheduler.shift` 不匹配（Turbo 是 8.0，基础版 3.0 / 5.0）
  3. `param_names_mapping` 少写一条，DiT 里某些 Linear 加载时被当成 zero-init
  4. `t5_postprocess_text` 的 pad 长度 `text_len` 与 DiT 的 `text_len` 不一致
  5. VAE 精度：Wan VAE 对 fp32 敏感，`vae_precision` 一定要设成 `"fp32"`（基类已经这么设了，继承时别覆盖成 bf16）

---

## 4. 新增一个 Wan 变体（最常见场景）

假设你要接入一个虚构的 `MyOrg/MyWan-T2V-3B-Diffusers`，它和 Wan 1.3B 结构相同，只是 DiT 层数换成 30、`flow_shift` 要用 4.0、默认分辨率 512x832、10 步即可。全部改动只有 **3 个文件**：

### (a) `configs/sample/wan.py` — 新增一个 SamplingParams

```python
@dataclass
class MyWanT2V_3B_SamplingParams(WanT2V_1_3B_SamplingParams):
    height: int = 512
    width: int = 832
    num_inference_steps: int = 10
    guidance_scale: float = 4.0
```

### (b) `configs/pipeline_configs/wan.py` — 新增一个 PipelineConfig

```python
@dataclass
class MyWanT2V480PConfig(WanT2V480PConfig):
    flow_shift: float | None = 4.0

    def __post_init__(self):
        super().__post_init__()
        # 把 DiT 层数从 40 改成 30
        self.dit_config.arch_config.num_layers = 30
```

> 若需要的是 14B 基类的变体，则从 `WanT2V720PConfig` 继承；若是 I2V，则从 `WanI2V480PConfig` / `WanI2V720PConfig` 继承（注意 I2V 要同时 `load_encoder = True`，这在它们的 `__post_init__` 里已经帮你处理了）。

### (c) `registry.py` — 挂上 hf 路径

在现有"# Wan"那一大段里加一条：

```python
register_configs(
    sampling_param_cls=MyWanT2V_3B_SamplingParams,
    pipeline_config_cls=MyWanT2V480PConfig,
    hf_model_paths=[
        "MyOrg/MyWan-T2V-3B-Diffusers",
    ],
)
```

这样就完成了——**不需要**碰 Pipeline 类、不需要新建 DiT 文件、不需要写 Stage，只要权重名与 Wan 对齐（靠 `param_names_mapping` 保证），`DiffGenerator.from_pretrained("MyOrg/MyWan-T2V-3B-Diffusers")` 就能跑起来。

---

## 5. 什么时候必须写新 Pipeline / 新 DiT / 新 Stage？

Wan 家族给出了清晰边界：

| 情况 | 什么叫 Wan 家族"装不下"？ |
|------|---------------------------|
| 需要新 DiT 文件 | 新模型的 attention 结构、条件注入方式、forward 签名和 Wan 不兼容；改正则映射也救不回来 |
| 需要新 Pipeline 类 | `model_index.json._class_name` 不是 `WanPipeline` / `WanImageToVideoPipeline`（框架按类名找 Pipeline） |
| 需要新的 `BeforeDenoisingStage`（Hybrid） | 前处理需要做 VLM captioning、AR token 生成、跨模态 rope、非标 latent packing 等超出 `add_standard_t2i/ti2v_stages()` 能力的事 |
| 需要新 scheduler | 现有 Flow Match / UniPC / DPM-solver 都不匹配 |
| 其他绝大多数情况 | **增 `PipelineConfig` + `SamplingParams` + `register_configs` 就够** |

---

## 6. Wan2.1 checklist（复用 08 的清单，补 Wan 特有点）

通用项请参照 [08_skill_add_model.md](./08_skill_add_model.md#6-提交前自检清单)。Wan 家族还要额外确认：

- [ ] `pipeline_name == "WanPipeline"`（T2V）或 `"WanImageToVideoPipeline"`（I2V），与 `model_index.json._class_name` 一致
- [ ] `FlowUniPCMultistepScheduler(shift=self.pipeline_config.flow_shift)` 在 `initialize_pipeline` 里手动注入（默认 scheduler 不对）
- [ ] I2V 变体：`vae_config.load_encoder = True`，并提供 `image_encoder_config`（Wan I2V 用 CLIP Vision）
- [ ] `postprocess_text_funcs` 里 pad 到的长度与 `WanVideoArchConfig.text_len`（默认 512）一致
- [ ] VAE 精度固定 fp32；DiT 精度 bf16
- [ ] 若支持 TP/SP：`num_attention_heads` 能被 `tp_size` 整除；sequence 维度可被 `ulysses_degree * ring_degree` 整除
- [ ] 新变体的 `teacache_params` 系数需要自己 profile（Wan 1.3B / 14B 的系数见 `_wan_1_3b_coefficients` / `_wan_14b_coefficients`）
- [ ] Wan2.2 特殊字段 `expand_timesteps` / `boundary_ratio` / `vae_stride` 只在需要时开启

---

## 7. 进一步阅读

- 深入 `ComposedPipelineBase` + 各 Stage 的执行时序：[03_runtime_execution.md](./03_runtime_execution.md)、[04_pipeline_and_stage.md](./04_pipeline_and_stage.md)
- `registry.get_model_info` 内部链路：[02_registry_and_config.md](./02_registry_and_config.md)
- 权重加载、`param_names_mapping` 工作原理、FSDP shard 规则：[05_loader_and_models.md](./05_loader_and_models.md)
- 三段式（encode / denoise / decode）拆服务优化：[07_disaggregation_and_optimization.md](./07_disaggregation_and_optimization.md)
- 通用"新增模型"方法论和 Hybrid 风格：[08_skill_add_model.md](./08_skill_add_model.md)
