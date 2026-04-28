# Wan 模型家族全景：所有模型 + 详细结构

> 本文聚焦 `python/sglang/multimodal_gen` 中 **Wan 系列** 的所有模型与完整结构。
>
> 配套阅读：
> - [09_case_study_wan2_1.md](./09_case_study_wan2_1.md)：如何把 Wan2.1 当作参照"在这套框架里新增一个模型"。
> - [10_wan2_1_end_to_end.md](./10_wan2_1_end_to_end.md)：Wan2.1 CLI → kernel 的端到端数据流。
> - [wan2_1_guide.md](./wan2_1_guide.md)：用户侧运行/部署文档。
>
> 本文不讲"怎么调用"，只讲：**代码里到底有几个 Wan？每个 Wan 的 Transformer / VAE / Encoder / Scheduler 长什么样？T2V、I2V、TI2V、Turbo、DMD、Fun、Causal 之间的结构差异在哪儿？**

---

## 目录

1. [一张图看懂 Wan 系列在代码里的"拆法"](#1-一张图看懂-wan-系列在代码里的拆法)
2. [Wan 家族在仓库中的所有型号清单](#2-wan-家族在仓库中的所有型号清单)
3. [Pipeline 类矩阵（4 个具体类 + 1 个通用类）](#3-pipeline-类矩阵4-个具体类--1-个通用类)
4. [DiT 结构：`WanTransformer3DModel`](#4-dit-结构wantransformer3dmodel)
5. [DiT 变体：`WanTransformerBlock_VSA` / `CausalWanTransformer3DModel`](#5-dit-变体wantransformerblock_vsa--causalwantransformer3dmodel)
6. [VAE 结构：`AutoencoderKLWan`（Wan2.1 / Wan2.2 统一实现）](#6-vae-结构autoencoderklwanwan21--wan22-统一实现)
7. [Scheduler：`FlowUniPCMultistepScheduler` / `FlowMatchEulerDiscreteScheduler`](#7-schedulerflowunipcmultistepscheduler--flowmatcheulerdiscretescheduler)
8. [Text / Image Encoder](#8-text--image-encoder)
9. [各型号结构差异一览表（面试级）](#9-各型号结构差异一览表面试级)
10. [权重命名映射：`param_names_mapping` 做了什么](#10-权重命名映射param_names_mapping-做了什么)
11. [分布式支持：TP / SP 在 Wan 里的切分策略](#11-分布式支持tp--sp-在-wan-里的切分策略)
12. [TeaCache / Cache-DiT / DMD / VSA / SLA 速查](#12-teacache--cache-dit--dmd--vsa--sla-速查)

---

## 1. 一张图看懂 Wan 系列在代码里的"拆法"

Wan 在 `multimodal_gen` 里是个典型的"**一个模型族，多层复用**"的例子。同一条推理通路被 8+ 种变体共用，差异被严格下沉到 config。整体结构如下：

```
 ┌───────────────────────── 用户视角：一堆不同 HF 模型 ID ─────────────────────────┐
 │  Wan2.1-T2V-1.3B / T2V-14B / I2V-14B-480P / I2V-14B-720P                      │
 │  Wan2.2-TI2V-5B / T2V-A14B / I2V-A14B                                         │
 │  TurboWan2.1-* / FastWan2.1-* / FastWan2.2-* / Wan2.1-Fun-1.3B-InP / …        │
 └──────────────────────────────────┬────────────────────────────────────────────┘
                                    │  registry.py: hf_model_paths → *Config 类
                                    ▼
 ┌──────────────────────── PipelineConfig 层（多态） ────────────────────────────┐
 │  WanT2V480PConfig / WanT2V720PConfig                                          │
 │  WanI2V480PConfig / WanI2V720PConfig                                          │
 │  TurboWanT2V480PConfig / TurboWanI2V720Config                                 │
 │  FastWan2_1_T2V_480P_Config                                                   │
 │  Wan2_2_TI2V_5B_Config / Wan2_2_T2V_A14B_Config / Wan2_2_I2V_A14B_Config      │
 │  FastWan2_2_TI2V_5B_Config                                                    │
 │  SelfForcingWanT2V480PConfig                                                  │
 │    ↓  每个 *Config 只决定：flow_shift、task_type、VAE 载半边、DMD 时间步、     │
 │       boundary_ratio、expand_timesteps 这些"静态参数"                         │
 └──────────────────────────────────┬────────────────────────────────────────────┘
                                    │  pipeline_name → Pipeline 类
                                    ▼
 ┌──────────────────────── Pipeline 类层（仅 4 个） ─────────────────────────────┐
 │  WanPipeline (T2V)        ← 绝大多数 T2V/TI2V 变体都复用它                     │
 │  WanImageToVideoPipeline (I2V)                                                │
 │  WanDMDPipeline (T2V + DMD)                                                   │
 │  WanImageToVideoDmdPipeline (I2V + DMD)                                       │
 │  WanCausalDMDPipeline (Causal Self-Forcing 专用)                              │
 │    ↓  Pipeline 只管：scheduler 替换 + Stage 编排                              │
 └──────────────────────────────────┬────────────────────────────────────────────┘
                                    │  5 个核心模块
                                    ▼
 ┌───────────────────────── 计算层（真干活的地方） ──────────────────────────────┐
 │  DiT:        WanTransformer3DModel  (主力)                                    │
 │              WanTransformerBlock_VSA  (VSA 稀疏注意力版，按 backend 切换)     │
 │              CausalWanTransformer3DModel  (Causal Self-Forcing 专用)          │
 │  VAE:        AutoencoderKLWan  (3D Causal VAE，Wan2.1 / Wan2.2 同类)          │
 │  Encoder:    T5EncoderModel (UMT5-XXL) + CLIPVisionModel (仅 I2V)             │
 │  Scheduler:  FlowUniPCMultistepScheduler (默认)                                │
 │              FlowMatchEulerDiscreteScheduler (DMD)                            │
 │              SelfForcingFlowMatch (Causal)                                    │
 └───────────────────────────────────────────────────────────────────────────────┘
```

**一句话**：Wan 系列在代码里只有 **3 个 DiT 实现** + **1 个 VAE 实现** + **5 个 Pipeline 类**，它们被 20+ 份 `PipelineConfig` / `SamplingParams` 组合出各种 HF model 变体。

---

## 2. Wan 家族在仓库中的所有型号清单

下表按 `registry.py` 的注册顺序给出，所有型号都落在 `# Wan` 和 `# Wan2.2` 段。

| # | HF Model ID | PipelineConfig | SamplingParams | Pipeline 类 | DiT 权重规模 | 说明 |
|---|-------------|----------------|-----------------|-------------|-------------|------|
| 1 | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | `WanT2V480PConfig` | `WanT2V_1_3B_SamplingParams` | `WanPipeline` | 1.3B | 官方 Wan2.1 T2V 小模型，480P |
| 2 | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | `WanT2V720PConfig` | `WanT2V_14B_SamplingParams` | `WanPipeline` | 14B | 官方 Wan2.1 T2V 大模型，支持 720P |
| 3 | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` | `WanI2V480PConfig` | `WanI2V_14B_480P_SamplingParam` | `WanImageToVideoPipeline` | 14B | Wan2.1 I2V，480P |
| 4 | `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` | `WanI2V720PConfig` | `WanI2V_14B_720P_SamplingParam` | `WanImageToVideoPipeline` | 14B | Wan2.1 I2V，720P |
| 5 | `IPostYellow/TurboWan2.1-T2V-1.3B-Diffusers` | `TurboWanT2V480PConfig` | `WanT2V_1_3B_SamplingParams` | `WanPipeline` | 1.3B | Turbo 蒸馏版 T2V 1.3B，`flow_shift=8.0`，4 步 |
| 6 | `IPostYellow/TurboWan2.1-T2V-14B-*-Diffusers` | `TurboWanT2V480PConfig` | `WanT2V_14B_SamplingParams` | `WanPipeline` | 14B | Turbo 蒸馏版 T2V 14B |
| 7 | `IPostYellow/TurboWan2.2-I2V-A14B-Diffusers` | `TurboWanI2V720Config` | `Turbo_Wan2_2_I2V_A14B_SamplingParam` | `WanImageToVideoPipeline` | A14B | Turbo + MoE，`boundary_ratio=0.9` |
| 8 | `weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers` | `WanI2V480PConfig` | `Wan2_1_Fun_1_3B_InP_SamplingParams` | `WanImageToVideoPipeline` | 1.3B | Wan2.1 Fun InP 变体，用 I2V 管线但容量更小 |
| 9 | `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` | `FastWan2_1_T2V_480P_Config` | `FastWanT2V480PConfig` | `WanDMDPipeline` | 1.3B | DMD 加速，3 步 |
| 10 | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | `Wan2_2_TI2V_5B_Config` | `Wan2_2_TI2V_5B_SamplingParam` | `WanPipeline` | 5B | Wan2.2 原生 TI2V（同时支持 T2V 与 I2V），`vae_stride=(4,16,16)` |
| 11 | `FastVideo/FastWan2.2-TI2V-5B-*-Diffusers` | `FastWan2_2_TI2V_5B_Config` | `Wan2_2_TI2V_5B_SamplingParam` | `WanDMDPipeline` | 5B | DMD 版 TI2V 5B |
| 12 | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | `Wan2_2_T2V_A14B_Config` | `Wan2_2_T2V_A14B_SamplingParam` | `WanPipeline` | A14B (MoE) | Wan2.2 T2V，高/低噪声两套 expert，`boundary_ratio=0.875` |
| 13 | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | `Wan2_2_I2V_A14B_Config` | `Wan2_2_I2V_A14B_SamplingParam` | `WanImageToVideoPipeline` | A14B (MoE) | Wan2.2 I2V，`boundary_ratio=0.900` |

此外，`SelfForcingWanT2V480PConfig`（`runtime/pipelines/wan_causal_dmd_pipeline.py` + `runtime/models/dits/causal_wanvideo.py`）是**不绑定到 HF ID**的 Causal Self-Forcing 专用配置，需要用户显式传 `--pipeline-config-name SelfForcingWanT2V480PConfig` 才能命中。

**规模分类：**

- 1.3B：Wan2.1 T2V 小模型、Turbo 1.3B、FastWan 1.3B、Wan2.1-Fun 1.3B
- 5B：Wan2.2 TI2V 原生 5B、FastWan2.2 TI2V 5B
- 14B（稠密）：Wan2.1 T2V/I2V 14B、Turbo 14B
- A14B（MoE / 双专家）：Wan2.2 T2V A14B、Wan2.2 I2V A14B、Turbo Wan2.2 I2V A14B

A14B 的实现细节见 [§9 boundary_ratio](#9-各型号结构差异一览表面试级)：同一个 Transformer 加载两份权重（high_noise / low_noise），按当前 timestep 跨 `boundary_ratio` 时切换。

---

## 3. Pipeline 类矩阵（4 个具体类 + 1 个通用类）

| Pipeline 类 | `pipeline_name` | Scheduler | Stage 组合 | I2V? | MoE? |
|-------------|-----------------|-----------|------------|------|------|
| `WanPipeline` | `"WanPipeline"` | `FlowUniPCMultistepScheduler` | `add_standard_t2i_stages()` | 否 | 支持（通过 `boundary_ratio`） |
| `WanImageToVideoPipeline` | `"WanImageToVideoPipeline"` | `FlowUniPCMultistepScheduler` | `add_standard_ti2v_stages()` | 是 | 支持 |
| `WanDMDPipeline` | `"WanDMDPipeline"` | `FlowMatchEulerDiscreteScheduler` | 手动拼 + `DmdDenoisingStage` | 否 | — |
| `WanImageToVideoDmdPipeline` | `"WanImageToVideoDmdPipeline"` | `FlowMatchEulerDiscreteScheduler` | `add_standard_ti2v_stages(denoising_stage_factory=DmdDenoisingStage)` | 是 | 支持（传入 `transformer_2`） |
| `WanCausalDMDPipeline` | `"WanCausalDMDPipeline"` | （由 config 提供） | 手动拼 + `CausalDMDDenoisingStage` | 否 | — |

关键事实：**所有 Pipeline 类都只做 2 件事**——换 scheduler + 串 Stage。**所有模型结构差异都不在 Pipeline 里，而在 config 里**。

`WanPipeline` 的全部代码就 30 行：

```24:49:python/sglang/multimodal_gen/runtime/pipelines/wan_pipeline.py
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

---

## 4. DiT 结构：`WanTransformer3DModel`

这是整个家族的**主力 Transformer**。源码：`python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py`。

### 4.1 顶层输入输出

```python
def forward(
    self,
    hidden_states: torch.Tensor,                                   # [B, C_in, T, H, W]
    encoder_hidden_states: torch.Tensor | list[torch.Tensor],      # T5 prompt embeds [B, L_text, 4096]
    timestep: torch.LongTensor,                                    # [B]  或 [B, seq_len] (wan2.2 ti2v)
    encoder_hidden_states_image: torch.Tensor | None = None,       # I2V: CLIP Vision hidden states [B, 257, D]
    guidance=None,
    **kwargs,
) -> torch.Tensor:                                                 # [B, C_out, T, H, W]
```

- 输入是 **5D video latent**（带时间维）
- 输出是"去噪残差"，shape 与输入相同（`out_channels == in_channels == 16` for Wan2.1）

### 4.2 `WanVideoArchConfig` 里的关键常量（Wan2.1 14B 默认）

```82:114:python/sglang/multimodal_gen/configs/models/dits/wanvideo.py
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_len = 512
    num_attention_heads: int = 40
    attention_head_dim: int = 128
    in_channels: int = 16
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 13824
    num_layers: int = 40
    cross_attn_norm: bool = True
    qk_norm: str = "rms_norm_across_heads"
    eps: float = 1e-6
    image_dim: int | None = None
    added_kv_proj_dim: int | None = None
    rope_max_seq_len: int = 1024
    pos_embed_seq_len: int | None = None
    exclude_lora_layers: list[str] = field(default_factory=lambda: ["embedder"])

    # Wan MoE
    boundary_ratio: float | None = None
```

派生字段：

- `hidden_size = num_attention_heads * attention_head_dim = 40 * 128 = 5120`
- `num_channels_latents = out_channels = 16`（Wan2.1）/ `48`（Wan2.2 TI2V，通过 `vae_stride=(4,16,16)` 实现）

### 4.3 前处理：Patch + 条件嵌入

```
输入 latent [B, 16, T, H, W]
  │
  │ PatchEmbed(in_chans=16, embed_dim=5120, patch_size=(1,2,2))
  │   ├ Conv3d: kernel=(1,2,2) stride=(1,2,2)
  │   └ 不做 flatten（由 forward 自己 flatten）
  │     输出 [B, 5120, T, H/2, W/2]，展平后 [B, T·H/2·W/2, 5120]
  │
  │ WanTimeTextImageEmbedding
  │   ├ time_embedder:  TimestepEmbedder(freq_dim=256) → MLP(silu)  → [B, 5120]
  │   ├ time_modulation: Linear + silu → [B, 5120*6] = [B,6,5120] 共 6 份调制参数
  │   ├ text_embedder:   MLP(gelu_pytorch_tanh, 4096→5120) → [B, L_text, 5120]
  │   └ image_embedder:  仅 I2V 有
  │       FP32LayerNorm → MLP(gelu) → FP32LayerNorm → [B, 257, 5120]
  │
  └→ 下游 N 层 TransformerBlock
```

几个细节：

- **patch_size 默认 `(1,2,2)`**：时间维不做 patchify（每帧独立），空间下采样 2×2
  - Wan2.2 TI2V 5B 例外：`vae_stride=(4,16,16)` 而不是改 patch_size
- **time modulation 一次产生 6 份 scale/shift**：用于 block 内 `shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa`
- **I2V 的 image_embedder**：把 CLIP Vision 第 -2 层的 hidden states（257 tokens = 1 CLS + 16×16 patch）编码到 5120 维，和 text embed 拼接（见 `WanI2VCrossAttention` 里 `context[:, :257]`）

### 4.4 Transformer Block：`WanTransformerBlock`

每层 block 按严格的 "AdaLN-Zero + Self-Attn + Cross-Attn + FFN" 三段式：

```
hidden_states [B, L, 5120]
  │
  │ ─────────── 1. Self-Attention ───────────
  │ norm_hidden = LayerNormScaleShift(hidden_states, shift_msa, scale_msa)   # fp32
  │ q = to_q(norm_hidden)   →  RMSNorm(norm_q) → [B, L, H_local, 128]
  │ k = to_k(norm_hidden)   →  RMSNorm(norm_k) → [B, L, H_local, 128]
  │ v = to_v(norm_hidden)                       → [B, L, H_local, 128]
  │ (q, k) = apply_flashinfer_rope_qk_inplace(q, k, cos_sin)   # 3D RoPE
  │ attn_output = USPAttention(q, k, v)         # SDPA-style, 支持 SP
  │ attn_output = to_out(attn_output)           # RowParallelLinear
  │ norm_hidden, hidden = ScaleResidualLayerNormScaleShift(
  │     hidden_states, attn_output, gate_msa, 0, 0)   # residual + next norm
  │
  │ ─────────── 2. Cross-Attention ───────────
  │ attn2(norm_hidden, context=encoder_hidden_states)
  │   ├ T2V: WanT2VCrossAttention   ← 只对 text
  │   └ I2V: WanI2VCrossAttention   ← 对 (img_tokens[:257], text_tokens[257:]) 各算一次再相加
  │ norm_hidden, hidden = ScaleResidualLayerNormScaleShift(
  │     hidden, attn_output, 1, c_shift_msa, c_scale_msa)
  │
  │ ─────────── 3. Feed-Forward ───────────
  │ ff = MLP(norm_hidden)   (gelu_pytorch_tanh, dim→ffn_dim→dim)
  │ hidden = MulAdd(ff, c_gate_msa, hidden)
  │
  └→ 下一 block
```

- **AdaLN 参数**：`scale_shift_table = Parameter(torch.randn(1, 6, dim) / dim**0.5)`，和 `temb` 相加后 chunk 成 6 份
- **QK Norm**：默认 `rms_norm_across_heads`（跨 head 共享 `RMSNorm(dim)`），实现上配合 `tp_rmsnorm` 用 `tensor_parallel_rms_norm` 做分布式 all-reduce
- **RoPE**：3D RoPE，时间/高度/宽度各占 `d - 4*(d//6)`、`2*(d//6)`、`2*(d//6)` 维，`theta=10000`；CUDA 上用 FlashInfer 的 `apply_flashinfer_rope_qk_inplace`（inplace 原地变换），ROCm 上走 aiter，其他平台 fallback 到纯 PyTorch
- **融合算子**：`LayerNormScaleShift`、`ScaleResidualLayerNormScaleShift`、`MulAdd`——都是 SGLang 内置的 fused kernel，把 "scale + shift + residual + LN" 合成 1 个 kernel，减少 HBM 往返

### 4.5 I2V 的 Cross-Attention：双分支

`WanI2VCrossAttention` 比 T2V 多一条图像分支：

```249:334:python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py
class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self, ...) -> None:
        super().__init__(..., is_cross_attention=True)

        self.add_k_proj = ColumnParallelLinear(dim, dim, gather_output=False, ...)
        self.add_v_proj = ColumnParallelLinear(dim, dim, gather_output=False, ...)
        self.norm_added_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        context_img = context[:, :257]         # 前 257 token：CLIP Vision
        context = context[:, 257:]             # 后面：T5 text

        q, _ = self.to_q(x)
        k, _ = self.to_k(context)
        v, _ = self.to_v(context)
        k_img, _ = self.add_k_proj(context_img)
        v_img, _ = self.add_v_proj(context_img)

        img_x = self.attn(q, k_img, v_img)     # query 对图像
        x     = self.attn(q, k,     v)         # query 对文本
        x = x + img_x                          # 两路相加
        x, _ = self.to_out(x)
        return x
```

**核心设计**：共享同一个 `query`，但图像和文本走**独立的 K/V 投影**（`add_k_proj / add_v_proj` 专给图像用），两路 attention 输出**相加**再过 `to_out`。

### 4.6 输出：Norm + 线性投影 + Unpatchify

```python
# 末端：再一次 AdaLN
shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
hidden = self.norm_out(hidden, shift, scale)
hidden = self.proj_out(hidden)          # [B, L, C_out * p_t * p_h * p_w]

# Unpatchify：[B, T', H', W', p_t, p_h, p_w, C_out] → [B, C_out, T, H, W]
hidden = hidden.reshape(B, T', H', W', p_t, p_h, p_w, -1)
hidden = hidden.permute(0, 7, 1, 4, 2, 5, 3, 6)
output = hidden.flatten(6, 7).flatten(4, 5).flatten(2, 3)
```

注意 `proj_out` 是 `ColumnParallelLinear(gather_output=True)`，输出在 TP 组内 all-gather。

### 4.7 Wan2.2 TI2V 的特殊形状：timestep per-token

Wan2.1 / Wan2.2 14B 的 timestep 是 `[B]`（全局一个 t）；Wan2.2 TI2V 5B 会把 timestep 扩成 `[B, L]`（每个 latent token 一个 t，用于"保留第一帧不去噪"的 first-frame masking）。代码里通过 `timestep.dim()` 自动分支：

```1083:1126:python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py
        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.dim() == 2:
            # ti2v
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image,
                timestep_seq_len=ts_seq_len,
            )
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))
```

对应的 first-frame mask 构造在 `runtime/pipelines_core/stages/model_specific_stages/wan_ti2v.py` 中（见 [§9](#9-各型号结构差异一览表面试级)）。

---

## 5. DiT 变体：`WanTransformerBlock_VSA` / `CausalWanTransformer3DModel`

### 5.1 `WanTransformerBlock_VSA`——Video Sparse Attention 版

当 `attention_backend == "video_sparse_attn"` 时，`WanTransformer3DModel.__init__` 会把 block 类从 `WanTransformerBlock` 切到 `WanTransformerBlock_VSA`：

```899:925:python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py
        attn_backend = get_global_server_args().attention_backend
        transformer_block = (
            WanTransformerBlock_VSA
            if (attn_backend and attn_backend.lower() == "video_sparse_attn")
            else WanTransformerBlock
        )
        self.blocks = nn.ModuleList(
            [
                transformer_block(
                    inner_dim,
                    config.ffn_dim,
                    config.num_attention_heads,
                    ...
                )
                for i in range(config.num_layers)
            ]
        )
```

**VSA 版 block 与默认版的 2 个差异**：

1. **多一个 `to_gate_compress` Linear**：产出 `gate_compress` 送进 `UlyssesAttention_VSA`，用于显式的稀疏度 gating
2. **`ColumnParallelLinear(gather_output=True)`**：不再做 Row/Column 配对的 TP shard，而是所有 Linear 都 gather——因为 VSA kernel 当前实现需要完整 KV

VSA 不开时（默认），Wan 走普通的 USP（Ulysses-Style Parallel）attention，是 TP + SP-ready 的实现。

### 5.2 `CausalWanTransformer3DModel`——Causal Self-Forcing 专用

这是一个**独立的 DiT 类**（不是通过 block 切换，而是另写一份），实现 CausVid 论文的自回归块级 causal masking。

- 文件：`runtime/models/dits/causal_wanvideo.py`
- Pipeline：`WanCausalDMDPipeline`
- 关键区别：
  - 复用 `WanT2VCrossAttention` 和 `WanTimeTextImageEmbedding`（从 `wanvideo.py` 导入）
  - Self-Attn 换成 `CausalWanSelfAttention`——内部用 PyTorch 2.x 的 **FlexAttention + BlockMask**（compile to max-autotune），每个 block 覆盖 `num_frames_per_block` 个 latent 帧
  - 支持**显式 KV-Cache**（推理时不重算历史帧的 K/V，只 append 当前 chunk）
  - 支持**attention sink**：`sink_size` 个"永不 evict"的锚点帧
  - 训练路径 `_forward_train` 和推理路径 `_forward_inference` 分离，前者用 flex_attention 跑 blockwise mask，后者逐 chunk 推并滚动 KV-Cache

配套常量（见 `WanVideoArchConfig` 里 "Causal Wan" 段）：

```104:114:python/sglang/multimodal_gen/configs/models/dits/wanvideo.py
    # Causal Wan
    local_attn_size: int = -1       # -1 = global; 正数 = 滑窗大小（帧数）
    sink_size: int = 0              # attention sink: 最前 N 帧永不滑出
    num_frames_per_block: int = 3   # 每 chunk 几个 latent 帧
    sliding_window_num_frames: int = 21
    attention_type: str = "original"  # 可选 "sla" / "sagesla"
    sla_topk: float = 0.1
```

当 `attention_type == "sla"` / `"sagesla"` 时，**标准** `WanTransformerBlock`（不是 Causal 版）也会把 `attn1` 换成 `MinimalA2AAttnOp` 做稀疏 self-attn——这是另一条独立的稀疏路径，走 `AttentionBackendEnum.SLA_ATTN` / `SAGE_SLA_ATTN`。

**三者关系**：

| 类 | 目的 | 触发方式 |
|----|------|----------|
| `WanTransformerBlock` | 默认 dense self-attn | 绝大多数 Wan 变体 |
| `WanTransformerBlock_VSA` | Video Sparse Attention（稀疏时空注意力） | `attention_backend=video_sparse_attn` |
| `WanTransformerBlock` + `MinimalA2AAttnOp` | SLA / SageSLA 稀疏 | `dit_config.arch_config.attention_type in {"sla", "sagesla"}` |
| `CausalWanTransformer3DModel` | 自回归块级 causal + KV-Cache | `SelfForcingWanT2V480PConfig` / `WanCausalDMDPipeline` |

---

## 6. VAE 结构：`AutoencoderKLWan`（Wan2.1 / Wan2.2 统一实现）

源码：`python/sglang/multimodal_gen/runtime/models/vaes/wanvae.py`

### 6.1 顶层参数（Wan2.1 默认）

```11:73:python/sglang/multimodal_gen/configs/models/vaes/wanvae.py
@dataclass
class WanVAEArchConfig(VAEArchConfig):
    base_dim: int = 96
    decoder_base_dim: int | None = None
    z_dim: int = 16                             # latent channels
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)    # 各层通道倍率
    num_res_blocks: int = 2
    attn_scales: tuple[float, ...] = ()
    temperal_downsample: tuple[bool, ...] = (False, True, True)
    dropout: float = 0.0
    latents_mean: tuple[float, ...] = (-0.7571, -0.7089, ..., -0.2921)      # 16 个
    latents_std: tuple[float, ...]  = ( 2.8184,  1.4541, ...,  1.9160)      # 16 个
    is_residual: bool = False       # Wan2.2 VAE 为 True
    in_channels: int = 3
    out_channels: int = 3
    patch_size: int | None = None
    scale_factor_temporal: int = 4
    scale_factor_spatial: int = 8
    clip_output: bool = True
```

**关键结论**：

- Wan 的 VAE 是**3D Causal VAE**，时间维做 4× 下采样，空间维做 8× 下采样
- latent 通道数 `z_dim=16`（Wan2.1 / Wan2.2 通用）
- **`latents_mean` / `latents_std` 是 16 维向量**，每个 latent 通道一个 scale/shift，不是全局标量
  - `scaling_factor = 1 / std.view(1,16,1,1,1)`
  - `shift_factor   = mean.view(1,16,1,1,1)`
  - DecodingStage 里 `z = z / scaling_factor + shift_factor` 才能把 latent 还原成像素域
- Wan2.1 与 Wan2.2 都用同一个 `AutoencoderKLWan` 类，差异只在 `is_residual`（Wan2.2 是）

### 6.2 编解码结构

```
【Encoder  WanEncoder3d】 输入 [B, 3, T, H, W]
  └ CausalConv3d(3, 96, k=3)
  └ down_blocks  (按 dim_mult=(1,2,4,4) 构造，共 3 次下采样)
       ├ ResidualBlock × num_res_blocks  (2 次)
       ├ Resample(mode=downsample2d|3d)    # 按 temperal_downsample 决定是否带时间下采样
       │   - downsample3d 额外带 time_conv: CausalConv3d(kernel=(3,1,1), stride=(2,1,1))
       └ 逐 stage 通道倍增
  └ mid_block (1 × ResidualBlock + 1 × AttentionBlock + 1 × ResidualBlock)
  └ norm_out (WanRMS_norm) + SiLU
  └ CausalConv3d(384, 32, k=3)   # 32 = z_dim * 2 = 16 * 2（mu + logvar）

【中间】 quant_conv: CausalConv3d(32, 32, k=1)
         → 按通道 split 出 mu[:16] / logvar[16:]  → DiagonalGaussianDistribution

【Decoder  WanDecoder3d】 输入 latent [B, 16, T', H', W']
  └ post_quant_conv: CausalConv3d(16, 16, k=1)
  └ CausalConv3d(16, 384, k=3)
  └ mid_block (1 × Res + 1 × Attn + 1 × Res)
  └ up_blocks  (按 dim_mult 逆序构造，共 3 次上采样)
       ├ ResidualBlock × (num_res_blocks+1)
       └ Resample(mode=upsample2d|3d)
             - upsample3d: 先 2x 时空 nearest-exact 插值 + Conv2d，再 time_conv 扩 T
  └ norm_out + SiLU
  └ CausalConv3d(96, 3, k=3)  → clamp(-1, 1)
```

### 6.3 Causal 3D Conv：保证自回归兼容

`WanCausalConv3d`（`wan_common_utils.py`）继承自 `nn.Conv3d`，但**把时间维的 padding 全挪到左边**：

```109:155:python/sglang/multimodal_gen/runtime/models/vaes/parallel/wan_common_utils.py
class WanCausalConv3d(nn.Conv3d):
    def __init__(self, ...):
        super().__init__(...)
        self._padding = (
            self.padding[2], self.padding[2],     # W: 对称
            self.padding[1], self.padding[1],     # H: 对称
            2 * self.padding[0], 0,               # T: 全挪到左侧（causal）
        )
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            # 用上一 chunk 的 last frames 代替 pad
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)
```

这样每层的输出只依赖"当前及之前的帧"，**允许 chunk-wise 推理**：前一 chunk 的最后 `CACHE_T=2` 帧作为下一 chunk 的左侧 pad，保持数值一致。

### 6.4 Feature Cache 机制：chunk decoding

当 `use_feature_cache=True`（默认），VAE 的 encode / decode 会：

1. 把输入视频按时间切成 chunk（encode：第一个 1 帧，其后每 4 帧一组；decode：每 1 latent 帧一组）
2. 每层 `WanCausalConv3d` 都查 `feat_cache[idx]`（上一 chunk 末尾 `CACHE_T=2` 帧），用它做左 pad 替代传统 padding
3. 写回 `feat_cache[idx] = 当前 chunk 最后 CACHE_T 帧`
4. 最终 cat 所有 chunk 的输出

这样**不管视频多长**（哪怕 81 帧 / 121 帧），VAE 都可以用**恒定显存**完成编解码。

### 6.5 Wan2.2 的差异：`is_residual=True`

Wan2.2 VAE 改用"残差下/上采样块"：

- `WanResidualDownBlock`：增加 `AvgDown3D` 捷径路径，主路是 ResidualBlocks + 可选 Resample
- `WanResidualUpBlock`：对应增加 `DupUp3D` 捷径路径

Wan2.2 TI2V 5B 的 VAE 有**更激进的下采样率**（`vae_stride=(4,16,16)`，空间 16×），所以 latent 的 H/W 只有输入的 1/16，显存极省但质量依赖更大的 base_dim。

### 6.6 并行支持：`WanDist*` + `ParallelTiledVAE`

`wan_dist_utils.py`（680 行）是 Encoder/Decoder 所有 block 的**高度分片版**（`WanDistCausalConv3d` / `WanDistResidualBlock` / ...），由 `use_parallel_encode / use_parallel_decode` 触发：

- **空间 TP + tiling**：沿 H 维切给 `sp_world_size` 张卡，每张卡只处理自己的 tile，最后 `gather_and_trim_height`
- **tiling 与并行正交**：`tile_sample_min_num_frames` 控时间 tile，`use_temporal_tiling=True` 开时间分块
- **`parallel_tiled_decode`**：把多个时间 tile 分发到不同 SP rank 上同时算

这套是 Wan VAE 能处理 `720P × 121 帧` 的关键。

---

## 7. Scheduler：`FlowUniPCMultistepScheduler` / `FlowMatchEulerDiscreteScheduler`

### 7.1 `FlowUniPCMultistepScheduler`（默认 Wan scheduler）

文件：`runtime/models/schedulers/scheduling_flow_unipc_multistep.py`

- 基于 Rectified Flow，用 UniPC (Unified Predictor-Corrector) 多步求解器
- **单参数 `shift`**（即 `flow_shift`）：用于非均匀采样步长
  - `sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)`
  - 直观：`shift` 越大，越把采样点向 t=0（高噪声端）聚
- 每个 Wan 变体的 `flow_shift`：

| Config | `flow_shift` | 推理步数 |
|--------|-------------:|---------:|
| `WanT2V480PConfig`（1.3B） | 3.0 | 50 |
| `WanT2V720PConfig`（14B）、`WanI2V480/720PConfig` | 5.0 | 50 |
| `TurboWanT2V480PConfig`、`FastWan2_1_T2V_480P_Config` | 8.0 | 3-4 |
| `TurboWanI2V720Config` | 8.0 | 4 |
| `Wan2_2_TI2V_5B_Config` | 5.0 | 50 |
| `Wan2_2_T2V_A14B_Config` | 12.0 | 40 |
| `Wan2_2_I2V_A14B_Config` | 5.0 | 40 |

**`WanPipeline` 会在 `initialize_pipeline` 里手动注入这个 scheduler**（不走 diffusers 默认），因为 Diffusers 的 UniPC 实现和 Wan 官方的不一致。

### 7.2 `FlowMatchEulerDiscreteScheduler`（DMD 用）

文件：`runtime/models/schedulers/scheduling_flow_match_euler_discrete.py`

- 给 DMD（Distribution Matching Distillation）加速用
- 和 UniPC 的区别：Euler 是 1 阶求解器，但 DMD 模型已被蒸馏到只需 3-4 步，所以 Euler 够用
- `WanDMDPipeline` / `WanImageToVideoDmdPipeline` 用它

**DMD 的时间步是 config 里硬编码的**：

```python
# TurboWanT2V480PConfig
dmd_denoising_steps = [988, 932, 852, 608]              # 4 步
# FastWan2_1_T2V_480P_Config
dmd_denoising_steps = [1000, 757, 522]                  # 3 步
# FastWan2_2_TI2V_5B_Config
dmd_denoising_steps = [1000, 757, 522]                  # 3 步
# SelfForcingWanT2V480PConfig
dmd_denoising_steps = [1000, 750, 500, 250]             # 4 步
```

### 7.3 Scheduler 切换表

| Config | Scheduler |
|--------|-----------|
| 默认 / 非 DMD 的 `Wan*T2V*` / `Wan*I2V*` | `FlowUniPCMultistepScheduler` |
| `TurboWan*` | 仍然是 `FlowUniPCMultistepScheduler`，但用 `dmd_denoising_steps` 硬编码少数几个 timestep |
| `FastWan2_1_T2V_480P_Config` → `WanDMDPipeline` | `FlowMatchEulerDiscreteScheduler` |
| `FastWan2_2_TI2V_5B_Config` → `WanDMDPipeline` | `FlowMatchEulerDiscreteScheduler` |
| `SelfForcingWanT2V480PConfig` → `WanCausalDMDPipeline` | 由 CausalDMDDenoisingStage 自行提供 |

---

## 8. Text / Image Encoder

### 8.1 Text Encoder：UMT5-XXL

- 配置：`T5Config()`（`configs/models/encoders/t5.py`）
- 实现：`runtime/models/encoders/t5.py::T5EncoderModel`
- 精度：**`text_encoder_precisions = ("fp32",)`**（Wan 基类强制 fp32；VAE 也是 fp32）
- 输出维度：`text_dim=4096` → DiT 里的 `text_embedder`（gelu_pytorch_tanh MLP）映射到 5120
- 后处理：`t5_postprocess_text`

```26:39:python/sglang/multimodal_gen/configs/pipeline_configs/wan.py
def t5_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    mask: torch.Tensor = outputs.attention_mask
    hidden_state: torch.Tensor = outputs.last_hidden_state
    seq_lens = mask.gt(0).sum(dim=1).long()
    assert torch.isnan(hidden_state).sum() == 0
    prompt_embeds = [u[:v] for u, v in zip(hidden_state, seq_lens, strict=True)]
    prompt_embeds_tensor: torch.Tensor = torch.stack(
        [
            torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
        dim=0,
    )
    return prompt_embeds_tensor
```

关键：**按 attention_mask 截断，再 pad 到 `text_len=512`**。`512` 这个魔数必须和 `WanVideoArchConfig.text_len` 一致，否则 DiT 的 cross-attn 会出现 `L_kv` 与训练不符的问题（虽然不会 crash，但会出噪声）。

### 8.2 Image Encoder：CLIP Vision（仅 I2V）

- 仅 `WanI2V*Config` 和 `Wan2_2_I2V_A14B_Config`、`Wan2.1-Fun` 使用
- 配置：`CLIPVisionConfig()`
- 输出取法：**倒数第二层** hidden states（`outputs.hidden_states[-2]`）
- 长度：257 tokens（1 CLS + 16×16 patch tokens）
- 用在 DiT：通过 `encoder_hidden_states_image` 进 `image_embedder`（FP32LN + MLP + FP32LN）映射到 5120，然后和 text embed 拼接成 `[B, 257+512, 5120]` 送进 cross-attn

---

## 9. 各型号结构差异一览表（面试级）

下表汇总了 Wan 家族每个变体在"结构关键点"上的差异。凡是有差异的维度都列出来：

| Config | task_type | DiT 类 | num_layers | hidden_size | MoE? | VAE encode? | `flow_shift` | DMD steps | boundary_ratio | expand_ts |
|---|---|---|---:|---:|:-:|:-:|---:|:-:|---:|:-:|
| `WanT2V480PConfig` (1.3B) | T2V | `WanTransformer3DModel` | 30 | 1536 | — | ✗ | 3.0 | — | — | ✗ |
| `WanT2V720PConfig` (14B) | T2V | 同上 | 40 | 5120 | — | ✗ | 5.0 | — | — | ✗ |
| `WanI2V480PConfig` (14B) | I2V | 同上（`image_dim=1280`） | 40 | 5120 | — | ✓ | 3.0 | — | — | ✗ |
| `WanI2V720PConfig` (14B) | I2V | 同上 | 40 | 5120 | — | ✓ | 5.0 | — | — | ✗ |
| `TurboWanT2V480PConfig` | T2V | 同上 | 30/40 | 1536/5120 | — | ✗ | 8.0 | [988,932,852,608] | — | ✗ |
| `TurboWanI2V720Config` | I2V | 同上 | 40 | 5120 | ✓ (2 transformer) | ✓ | 8.0 | [996,932,852,608] | 0.9 | ✗ |
| `FastWan2_1_T2V_480P_Config` | T2V | 同上 | 30 | 1536 | — | ✗ | 8.0 | [1000,757,522] | — | ✗ |
| `Wan2_2_TI2V_5B_Config` | TI2V | 同上（`in_channels=48`） | 30 | 3072 | — | ✓ | 5.0 | — | — | ✓ |
| `FastWan2_2_TI2V_5B_Config` | TI2V | 同上 | 30 | 3072 | — | ✓ | 5.0 | [1000,757,522] | — | ✓ |
| `Wan2_2_T2V_A14B_Config` | T2V | 同上（×2） | 40 | 5120 | ✓ | ✗ | 12.0 | — | 0.875 | ✗ |
| `Wan2_2_I2V_A14B_Config` | I2V | 同上（×2） | 40 | 5120 | ✓ | ✓ | 5.0 | — | 0.900 | ✗ |
| `SelfForcingWanT2V480PConfig` | T2V(causal) | `CausalWanTransformer3DModel` | 30 | 1536 | — | ✗ | 5.0 | [1000,750,500,250] | — | ✗ |

### 9.1 `boundary_ratio`：Wan2.2 的 MoE 切换机制

Wan2.2 A14B 实际上把**同一份** `WanTransformer3DModel` 加载**两份权重**（`transformer` 和 `transformer_2`），分别叫 high_noise expert 和 low_noise expert：

- 推理开始时 `t` 大（噪声多），用 `transformer`（high_noise expert）
- 当 `t / num_train_timesteps < boundary_ratio` 时切到 `transformer_2`（low_noise expert）
- 两个 expert **同构同参数量**，只是训练目标不同（前者学"大步去噪"，后者学"精修细节"）

Turbo Wan2.2 I2V A14B 的 `boundary_ratio=0.9` 表示"前 10% 噪声区段用 high_noise expert，其余用 low_noise expert"。

### 9.2 `expand_timesteps`：Wan2.2 TI2V 的 first-frame masking

见 `runtime/pipelines_core/stages/model_specific_stages/wan_ti2v.py`：

```134:160:python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/wan_ti2v.py
def expand_wan_ti2v_timestep(
    batch: Req,
    t_device: torch.Tensor,
    target_dtype: torch.dtype,
    seq_len: int,
    reserved_frames_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Expand the timestep tensor for Wan TI2V's first-frame masking semantics."""

    batch_size = batch.raw_latent_shape[0]
    t_device_rounded = t_device.to(target_dtype)

    local_seq_len = seq_len
    if get_sp_world_size() > 1 and getattr(batch, "did_sp_shard_latents", False):
        local_seq_len = seq_len // get_sp_world_size()

    if get_sp_parallel_rank() == 0 and reserved_frames_mask is not None:
        temp_ts = (reserved_frames_mask[0][:, ::2, ::2] * t_device_rounded).flatten()
        temp_ts = torch.cat(
            [
                temp_ts,
                temp_ts.new_ones(local_seq_len - temp_ts.size(0)) * t_device_rounded,
            ]
        )
        return temp_ts.unsqueeze(0).repeat(batch_size, 1)

    return t_device.repeat(batch_size, local_seq_len)
```

核心想法：对"需要保留的第一帧" token，timestep 置 0（不去噪）；其他 token 给正常的 t。这样 DiT 只需要接受 `[B, L]` 形状的 timestep，internal AdaLN 会自动对每个 token 用不同 scale/shift。

### 9.3 `vae_stride` vs `patch_size`：Wan2.2 TI2V 的"偷渡"

Wan2.1 的 latent shape：`[B, 16, T/4, H/8, W/8]`，DiT 的 patch 再切 1×2×2 → `[B, 5120, T/4, H/16, W/16]`。

Wan2.2 TI2V 5B 走 `vae_stride=(4,16,16)`：

```176:191:python/sglang/multimodal_gen/configs/pipeline_configs/wan.py
class Wan2_2_TI2V_5B_Config(WanT2V480PConfig, WanI2VCommonConfig):
    flow_shift: float | None = 5.0
    task_type: ModelTaskType = ModelTaskType.TI2V
    expand_timesteps: bool = True
    # ti2v, 5B
    vae_stride = (4, 16, 16)

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        F = num_frames
        z_dim = self.vae_config.arch_config.z_dim
        vae_stride = self.vae_stride
        oh = batch.height
        ow = batch.width
        shape = (batch_size, z_dim, F, oh // vae_stride[1], ow // vae_stride[2])
        return shape
```

VAE 还是同一个 `AutoencoderKLWan`，但 H/W 维下采样 16×（相当于 spatial 8×2），这样 seq_len 只有 Wan2.1 的 1/4，能在 5B DiT 上跑 720P × 121 帧。

---

## 10. 权重命名映射：`param_names_mapping` 做了什么

Diffusers 原生的 `WanTransformer3DModel` 和 SGLang 实现的参数命名**不完全一致**（后者用了 fused Linear、Column/RowParallel、重命名的 FFN 层）。`param_names_mapping` 用正则把 HF 权重名映射到 SGLang 内部名：

```17:39:python/sglang/multimodal_gen/configs/models/dits/wanvideo.py
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^patch_embedding\.(.*)$": r"patch_embedding.proj.\1",
            r"^condition_embedder\.text_embedder\.linear_1\.(.*)$": r"condition_embedder.text_embedder.fc_in.\1",
            r"^condition_embedder\.text_embedder\.linear_2\.(.*)$": r"condition_embedder.text_embedder.fc_out.\1",
            r"^condition_embedder\.time_embedder\.linear_1\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_in.\1",
            r"^condition_embedder\.time_embedder\.linear_2\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_out.\1",
            r"^condition_embedder\.time_proj\.(.*)$": r"condition_embedder.time_modulation.linear.\1",
            r"^condition_embedder\.image_embedder\.ff\.net\.0\.proj\.(.*)$": r"condition_embedder.image_embedder.ff.fc_in.\1",
            r"^condition_embedder\.image_embedder\.ff\.net\.2\.(.*)$": r"condition_embedder.image_embedder.ff.fc_out.\1",
            r"^blocks\.(\d+)\.attn1\.to_q\.(.*)$": r"blocks.\1.to_q.\2",
            r"^blocks\.(\d+)\.attn1\.to_k\.(.*)$": r"blocks.\1.to_k.\2",
            r"^blocks\.(\d+)\.attn1\.to_v\.(.*)$": r"blocks.\1.to_v.\2",
            r"^blocks\.(\d+)\.attn1\.to_out\.0\.(.*)$": r"blocks.\1.to_out.\2",
            r"^blocks\.(\d+)\.attn1\.norm_q\.(.*)$": r"blocks.\1.norm_q.\2",
            r"^blocks\.(\d+)\.attn1\.norm_k\.(.*)$": r"blocks.\1.norm_k.\2",
            r"^blocks\.(\d+)\.attn1\.attn_op\.local_attn\.proj_l\.(.*)$": r"blocks.\1.attn1.local_attn.proj_l.\2",
            r"^blocks\.(\d+)\.attn2\.to_out\.0\.(.*)$": r"blocks.\1.attn2.to_out.\2",
            r"^blocks\.(\d+)\.ffn\.net\.0\.proj\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
            r"^blocks\.(\d+)\.ffn\.net\.2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
            r"^blocks\.(\d+)\.norm2\.(.*)$": r"blocks.\1.self_attn_residual_norm.norm.\2",
        }
    )
```

几个亮点：

- **HF: `attn1.to_q` → SGLang: `to_q`**（self-attn 的 QKV 被 hoist 到 block 顶层，不再嵌套在 `attn1` 下）
- **HF: `ffn.net.0.proj` → SGLang: `ffn.fc_in`**；**HF: `ffn.net.2` → SGLang: `ffn.fc_out`**（标准 MLP 命名）
- **HF: `norm2` → SGLang: `self_attn_residual_norm.norm`**（SGLang 把"残差后的 LN"融进了 `ScaleResidualLayerNormScaleShift` 模块）
- **`time_proj` → `time_modulation.linear`**（时间调制被重命名成更描述性的名字）

### 10.1 `reverse_param_names_mapping`

是 `param_names_mapping` 的逆映射，用于**保存权重回 HF 格式**（`tools/wan_repack.py`）：

```41:63:python/sglang/multimodal_gen/configs/models/dits/wanvideo.py
    reverse_param_names_mapping: dict = field(...)
```

### 10.2 `lora_param_names_mapping`：社区 LoRA 适配

社区 LoRA（从 Wan 官方训练脚本导出的）用的是**阿里原版命名**（不是 Diffusers 风格），例如 `self_attn.q`、`cross_attn.q`、`ffn.0`。这张表在 `param_names_mapping` 之前应用，统一成 Diffusers 风格后再走正常 mapping：

```67:80:python/sglang/multimodal_gen/configs/models/dits/wanvideo.py
    lora_param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^blocks\.(\d+)\.self_attn\.q\.(.*)$": r"blocks.\1.attn1.to_q.\2",
            r"^blocks\.(\d+)\.self_attn\.k\.(.*)$": r"blocks.\1.attn1.to_k.\2",
            r"^blocks\.(\d+)\.self_attn\.v\.(.*)$": r"blocks.\1.attn1.to_v.\2",
            r"^blocks\.(\d+)\.self_attn\.o\.(.*)$": r"blocks.\1.attn1.to_out.0.\2",
            r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$": r"blocks.\1.attn2.to_q.\2",
            ...
            r"^blocks\.(\d+)\.ffn\.0\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
            r"^blocks\.(\d+)\.ffn\.2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
        }
    )
```

---

## 11. 分布式支持：TP / SP 在 Wan 里的切分策略

### 11.1 TP（Tensor Parallel）

- QKV 投影：`ColumnParallelLinear(gather_output=False)`，每张卡持有 `H_local = num_heads / tp_size` 个 head
- 输出投影 `to_out`：`RowParallelLinear(input_is_parallel=True, reduce_results=True)`，自动 all-reduce
- FFN 同理：`fc_in` 是 Column，`fc_out` 是 Row
- QK RMSNorm：当 `qk_norm=rms_norm_across_heads` 且 `tp_size > 1` 时，用 `tensor_parallel_rms_norm` 做跨 TP 的 RMS 计算（涉及一次 all-reduce 求 `sum(x**2)`）
- 限制：`num_attention_heads % tp_size == 0`

### 11.2 SP（Sequence Parallel）

- 框架：`USPAttention`（Ulysses-Style Parallel）
- 沿 `seq_len` 维切分，每张卡只持有 `L / sp_size` 个 token
- Cross-Attention 跳过 SP（`skip_sequence_parallel=True`）：因为 KV 长度 = `text_len+257`，相对短，直接复制到所有 SP rank 更划算
- RoPE 对应的位置索引要根据 `sp_rank` 计算偏移；见 `WanTransformer3DModel._compute_rope_for_sequence_shard` 用 `@lru_cache(1)` 缓存
- 输出时通过 `sequence_model_parallel_all_gather` 聚合回完整序列
- 序列长度不能整除 SP size 时会自动 pad 零

### 11.3 FSDP 分片条件

```python
def is_blocks(n: str, m) -> bool:
    return "blocks" in n and str.isdigit(n.split(".")[-1])

_fsdp_shard_conditions: list = field(default_factory=lambda: [is_blocks])
```

表示**只按 transformer block 粒度做 FSDP 分片**（每个 block 单独 shard，shared modules 如 patch_embedding / condition_embedder / proj_out 不分）。这是 FSDP 和 TP 混用时的常规做法。

---

## 12. TeaCache / Cache-DiT / DMD / VSA / SLA 速查

### 12.1 TeaCache

- **原理**：根据 modulated input 的 L1 距离判断当前 timestep 是否"足够接近前一次"，接近就直接复用上次的 residual
- **系数**：`_wan_1_3b_coefficients` / `_wan_14b_coefficients`（5 次多项式拟合的 L1 → 相对误差曲线），来自 `ali-vilab/TeaCache` 官方
- **阈值**：`teacache_thresh`（0.08 ~ 0.30），越大跳得越多但可能掉质量
- **CFG 支持**：正负 prompt 分别维护 `previous_residual` / `previous_residual_negative`
- 代码位置：`WanTransformer3DModel.should_skip_forward_for_cached_states` / `retrieve_cached_states` / `maybe_cache_states`

### 12.2 Cache-DiT（实验性）

- 通过 `SGLANG_CACHE_DIT_ENABLED=1` 环境变量启用
- 比 TeaCache 更细粒度（block 级别而非整层 DiT）
- Wan2.2 暂无 TeaCache 系数，推荐 Cache-DiT

### 12.3 DMD（Distribution Matching Distillation）

- 见 `DmdDenoisingStage`：只对 `dmd_denoising_steps` 里的几个 timestep 做 forward，其他步无噪声 add
- 配合 `FlowMatchEulerDiscreteScheduler` 用 Euler 1 阶
- **3-4 步出视频**，适合在线服务

### 12.4 VSA（Video Sparse Attention）

- 触发：`--attention-backend video_sparse_attn`
- 换 block 类到 `WanTransformerBlock_VSA`，多一路 `to_gate_compress`
- 走 `UlyssesAttention_VSA`；kernel 在 `runtime/layers/attention/` 下
- 配置 JSON：`configs/backend/vmoba/wan_1.3B_77_448_832.json`

### 12.5 SLA / SageSLA（Sparse Attention Layer）

- 触发：`dit_config.arch_config.attention_type = "sla"` 或 `"sagesla"`
- 走 `MinimalA2AAttnOp`，topk 控制在 `sla_topk`（默认 0.1 = 保留 top-10% 的 KV）
- backend：`AttentionBackendEnum.SLA_ATTN` / `SAGE_SLA_ATTN`

### 12.6 精度策略

| 组件 | 精度 | 原因 |
|------|------|------|
| DiT 权重 & 激活 | bf16 | 速度 + 数值稳定兼顾 |
| VAE 权重 & 激活 | **fp32** | Wan VAE 对低精度敏感（会出噪点），**绝对不要改** |
| T5 text encoder | fp32 | 同上，prompt 长度 512，代价可接受 |
| CLIP image encoder（I2V） | fp32 | 同上 |
| AdaLN 的 `shift/scale_msa` | fp32（代码 `assert shift_msa.dtype == torch.float32`） | 防止调制参数下溢 |
| TimestepEmbedder 输出 | fp32 | 同上 |
| RoPE `cos/sin` | fp32 | 高精度旋转角度 |
| Attention QKV 内部计算 | 由 attn backend 决定 | FA/FlashInfer 默认 bf16，部分走 fp32 accum |

---

## 附录 A：所有 Wan 文件路径总清单

| 组件 | 路径 |
|---|---|
| DiT 主实现 | `python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py` |
| DiT Causal 实现 | `python/sglang/multimodal_gen/runtime/models/dits/causal_wanvideo.py` |
| DiT 结构 config | `python/sglang/multimodal_gen/configs/models/dits/wanvideo.py` |
| VAE 主实现 | `python/sglang/multimodal_gen/runtime/models/vaes/wanvae.py` |
| VAE 结构 config | `python/sglang/multimodal_gen/configs/models/vaes/wanvae.py` |
| VAE 并行 utils | `python/sglang/multimodal_gen/runtime/models/vaes/parallel/wan_common_utils.py`、`wan_dist_utils.py` |
| Pipeline（T2V） | `python/sglang/multimodal_gen/runtime/pipelines/wan_pipeline.py` |
| Pipeline（I2V） | `python/sglang/multimodal_gen/runtime/pipelines/wan_i2v_pipeline.py` |
| Pipeline（T2V + DMD） | `python/sglang/multimodal_gen/runtime/pipelines/wan_dmd_pipeline.py` |
| Pipeline（I2V + DMD） | `python/sglang/multimodal_gen/runtime/pipelines/wan_i2v_dmd_pipeline.py` |
| Pipeline（Causal DMD） | `python/sglang/multimodal_gen/runtime/pipelines/wan_causal_dmd_pipeline.py` |
| PipelineConfig 全集 | `python/sglang/multimodal_gen/configs/pipeline_configs/wan.py` |
| SamplingParams 全集 | `python/sglang/multimodal_gen/configs/sample/wan.py` |
| TI2V 特化 Stage helper | `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/wan_ti2v.py` |
| Scheduler（默认） | `python/sglang/multimodal_gen/runtime/models/schedulers/scheduling_flow_unipc_multistep.py` |
| Scheduler（DMD） | `python/sglang/multimodal_gen/runtime/models/schedulers/scheduling_flow_match_euler_discrete.py` |
| 权重 repack 工具 | `python/sglang/multimodal_gen/tools/wan_repack.py` |
| Registry 注册段 | `python/sglang/multimodal_gen/registry.py` 第 652-744 行（`# Wan`、`# Wan2.2`） |

---

## 附录 B：快速自检 checklist

如果你在修改 Wan 某个子模块后跑出噪声/崩溃，按顺序查：

1. `param_names_mapping` 有没有漏掉新加的 Linear？→ `ReadLints` + `python -c "from diffusers import WanTransformer3DModel; m=WanTransformer3DModel(); print([n for n,_ in m.named_parameters()])"` 对照
2. `latents_mean` / `latents_std` 是不是贴错维度或少贴？→ 一定是 16 个值
3. `flow_shift` 是否与 scheduler 匹配？→ `FlowUniPCMultistepScheduler(shift=flow_shift)` 是否被 `initialize_pipeline` 正确注入
4. `text_len=512` 是否同时出现在 `t5_postprocess_text` 的 pad 长度 + `WanVideoArchConfig.text_len`？
5. VAE 精度是否被继承覆盖成 bf16？→ 永远保持 `vae_precision="fp32"`
6. TP：`num_attention_heads` 能否被 `tp_size` 整除？
7. SP：seq_len 若不能整除 `sp_size`，代码会自动 pad，但如果你关了 `sequence_shard_enabled` 而忘了 gather，就会形状不对
8. Wan2.2 TI2V：`expand_timesteps=True` 是否透传到 DiT（见 `Wan2_2_TI2V_5B_Config.__post_init__`）
9. Wan2.2 A14B：`boundary_ratio` 是否写进了 `dit_config`（否则 scheduler 不知道切 expert）
10. I2V：`vae_config.load_encoder = True` 是否生效（T2V 默认 False，继承时容易出错）

---

至此，Wan 系列在 `multimodal_gen` 中的**全部模型与结构细节**覆盖完成。想深入 Pipeline/Stage 编排时序，请读 [04_pipeline_and_stage.md](./04_pipeline_and_stage.md)；想定位权重加载链路，请读 [05_loader_and_models.md](./05_loader_and_models.md)。
