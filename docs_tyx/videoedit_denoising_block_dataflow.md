# VideoEdit 降噪阶段与单个 Transformer Block 数据流

本文基于当前 SGLang 仓库中的 `WanVideoEditPipeline` 和
`WanVideoEditTransformer3DModel` 实现，说明以下问题：

1. 一个降噪 timestep 向 Transformer 输入什么。
2. 输入单个 Transformer block 的张量是什么。
3. block 内依次执行哪些算子。
4. Self-Attention 和 Cross-Attention 分别在哪里计算 Q、K、V。
5. 当前 FP8、投影融合和 SageAttention 分别作用在哪一段。
6. block 输出和最终噪声预测在形状及数值含义上发生什么变化。

本文描述的当前模型配置来自
`videoedit_fp8_static/transformer/config.json`：

| 配置 | 当前值 |
|---|---:|
| Transformer blocks | 40 |
| Hidden size | 5120 |
| Attention heads | 40 |
| Head dimension | 128 |
| FFN hidden size | 13824 |
| Transformer 输入通道 | 36 |
| Transformer 输出通道 | 16 |
| Patch size | `(1, 2, 2)` |
| 文本条件长度 | 512 tokens |
| 图像条件长度 | 当前案例为 257 tokens |

相关核心代码：

- 降噪循环：
  [`videoedit_wan.py`](../python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py#L559)
- Transformer 外层：
  [`wanvideo.py`](../python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py#L1187)
- 单个 Transformer block：
  [`wanvideo.py`](../python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py#L486)
- VideoEdit Cross-Attention：
  [`wan_videoedit.py`](../python/sglang/multimodal_gen/runtime/models/dits/wan_videoedit.py#L36)

## 1. 整体降噪流程

一次 VideoEdit 请求会被切分为一个或多个 81 帧窗口。每个窗口先准备文本、
参考图、原视频、mask 和初始噪声，然后执行多个降噪 timestep。

一个 timestep 的主流程是：

```text
当前噪声 latent z_t
    │
    ├── 拼接 packed mask
    ├── 拼接 masked-video latent
    ▼
36 通道 Transformer 输入
    │
    ├── timestep
    ├── prompt embedding
    └── reference image embedding（use_clip=true 时）
    ▼
PatchEmbed
    ▼
40 × WanTransformerBlock
    ▼
输出 LayerNorm + Linear + unpatchify
    ▼
16 通道 noise_pred / flow prediction
    │
    ├── 可选 CFG：uncond + cfg × (cond - uncond)
    ▼
scheduler.step(noise_pred, timestep, z_t)
    ▼
下一步 latent z_(t-1)
```

40 步推理会重复以上过程 40 次。如果 `guidance_scale > 1`，每个 timestep
通常还会分别运行正向提示词和负向提示词两次 Transformer forward，然后做
CFG 合并。

## 2. Transformer 前的输入构造

### 2.1 原视频和 mask 条件

预处理代码位于
[`preprocess.py`](../python/sglang/multimodal_gen/runtime/videoedit/preprocess.py#L544)。

对一个 81 帧窗口：

1. mask 内需要编辑的区域在条件视频中被置零。
2. 第一个窗口的第一帧默认被完整保留，作为参考条件；后续窗口不强制执行这条首帧保留规则。
3. masked video 经过 VAE Encoder，得到 16 通道
   `runtime_cond_latents`。
4. mask 在空间上缩小 8 倍，并按 VAE 的 4 帧时间压缩方式打包为 4 个通道，
   得到 `runtime_cond_masks`。
5. 原始未遮挡视频也经过 VAE Encoder，得到 16 通道
   `runtime_video_latents`，主要用于初始化 latent。

因此每个 timestep 的 Transformer 图像输入是：

```python
latent_model_input = torch.cat(
    [
        latents,                    # 16 channels：当前待去噪 latent
        runtime_cond_masks,         #  4 channels：打包后的空间 mask
        runtime_cond_latents,       # 16 channels：masked-video VAE latent
    ],
    dim=1,
)
```

即：

```text
16 + 4 + 16 = 36 channels
```

对应代码：
[`videoedit_wan.py`](../python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py#L605)。

这里要区分三个值：

| 张量 | 是否随 timestep 变化 | 含义 |
|---|---|---|
| `latents` | 是 | 当前正在逐步去噪的结果 |
| `cond_masks` | 否 | 告诉模型哪些空间位置保留、哪些位置需要编辑 |
| `cond_latents` | 否 | 原视频去除 mask 区域后的内容条件 |

Transformer 不直接接收 RGB 视频，而是接收 VAE latent。

### 2.2 文本条件

提示词先经过文本编码器，得到 `runtime_prompt_embeds`。进入 Transformer 后，
`condition_embedder.text_embedder` 将文本特征从 4096 维映射到 Transformer
使用的 5120 维：

```text
text encoder output
[B, 512, 4096]
    │
    ├── Linear 4096 → 5120
    ├── GELU
    └── Linear 5120 → 5120
    ▼
[B, 512, 5120]
```

这 512 个文本 token 会在每个 block 的 Cross-Attention 中提供文本 K、V。

### 2.3 参考图条件

当 `use_clip=true` 时，窗口第一帧经过 image encoder，再由
`condition_embedder.image_embedder` 映射到 5120 维。当前案例中图像条件为
257 个 token：

```text
[B, 257, 1280]
    │
    ├── LayerNorm
    ├── Linear / GELU / Linear
    └── LayerNorm
    ▼
[B, 257, 5120]
```

Transformer 将图像 token 放在文本 token 前面：

```text
encoder_hidden_states =
    concat([image_tokens, text_tokens], dim=1)

shape = [B, 257 + 512, 5120]
      = [B, 769, 5120]
```

VideoEdit Cross-Attention 在每个 block 内重新把它拆为图像上下文和文本上下文。

### 2.4 timestep 条件

当前 timestep 经过：

```text
timestep
  → sinusoidal embedding
  → timestep MLP
  → time modulation Linear
  → [B, 6, 5120]
```

在每个 block 内，这 6 组向量与 block 自己的
`scale_shift_table` 相加，得到：

```text
shift_msa
scale_msa
gate_msa
c_shift_msa
c_scale_msa
c_gate_msa
```

它们不会作为普通 token 参加 Attention，而是控制 LayerNorm 后的缩放、平移和
残差门控，使相同的 40 个 block 能根据当前噪声时间步改变行为。

## 3. Transformer 外层如何把 36 通道变成 block 输入

Transformer 收到的 `hidden_states` 形状为：

```text
[B, 36, T_latent, H_latent, W_latent]
```

首先执行一个 `Conv3d` PatchEmbed：

```text
Conv3d(
    in_channels=36,
    out_channels=5120,
    kernel_size=(1, 2, 2),
    stride=(1, 2, 2),
)
```

因此：

```text
[B, 36, T, H, W]
    ↓ Conv3d patch embedding
[B, 5120, T, H/2, W/2]
    ↓ flatten + transpose
[B, L, 5120]

L = T × (H/2) × (W/2)
```

`PatchEmbed` 是卷积投影，不是 Attention，也不在 block 的 QKV 计算中。

当前 `profile81` 案例：

```text
对齐后的窗口像素尺寸：592 × 1728
VAE latent 空间尺寸：  74 × 216
VAE latent 时间长度：  21
Patch 后空间尺寸：     37 × 108
全局 token 数：         21 × 37 × 108 = 83,916
SP=2 后每卡 block 输入：41,958 tokens
```

所以当前双卡运行时，单个 block 的主要输入是：

```text
hidden_states:
    [1, 41958, 5120]     # 每卡本地视频 token

encoder_hidden_states:
    [1, 769, 5120]       # 257 image + 512 text

timestep_proj:
    [1, 6, 5120]

freqs_cis:
    与本地视频 token 对应的 3D RoPE cos/sin
```

## 4. 单个 WanTransformerBlock 的完整数据流

一个 block 的代码入口位于
[`WanTransformerBlock.forward`](../python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py#L658)。

简化数据流如下：

```text
block input x0: [B, L, 5120]
    │
    ├── 1. timestep-modulated LayerNorm
    ├── 2. Self QKV Projection
    ├── 3. Q/K RMSNorm + 3D RoPE
    ├── 4. Self-Attention
    ├── 5. Self output projection
    └── 6. gated residual: x1 = x0 + gate_msa × self_out
                              │
                              ├── LayerNorm
                              ├── 7. Cross Q/K/V Projection
                              ├── 8. Text Cross-Attention
                              ├── 9. Image Cross-Attention
                              ├── 10. Cross output projection
                              └── residual: x2 = x1 + cross_out
                                                │
                                                ├── modulated LayerNorm
                                                ├── 11. FFN Linear 5120 → 13824
                                                ├── 12. GELU
                                                ├── 13. FFN Linear 13824 → 5120
                                                └── gated residual:
                                                    x3 = x2 + c_gate_msa × ffn_out

block output x3: [B, L, 5120]
```

block 的输入和输出形状相同。block 改变的是每个视频 token 的 5120 维特征值，
把当前 latent 内容、全局视频关系、提示词、参考图和 timestep 信息融合进去。

## 5. 第一段：Self-Attention

### 5.1 timestep 调制的 LayerNorm

首先计算：

```text
x_norm = LayerNorm(x0) × (1 + scale_msa) + shift_msa
```

当前 CUDA 实现会尽量使用融合的 `LayerNorm + scale + shift` kernel。

形状保持：

```text
[B, L, 5120] → [B, L, 5120]
```

### 5.2 Self-Attention 的 QKV 在哪里计算

Q、K、V 在这里产生：

```python
qkv, _ = self.to_qkv(norm_hidden_states)
query, key, value = qkv.chunk(3, dim=-1)
```

对应代码：
[`wanvideo.py`](../python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py#L691)。

开启当前参数：

```text
--transformer-fp8-fused-projections true
```

后，原本三个独立 Linear：

```text
Q = x_norm × Wq + bq
K = x_norm × Wk + bk
V = x_norm × Wv + bv
```

被合并为一次 packed Linear：

```text
[Q | K | V] = x_norm × [Wq | Wk | Wv] + [bq | bk | bv]

[B, L, 5120]
    ↓ fused QKV Linear
[B, L, 15360]
    ↓ chunk(3)
Q, K, V: 各 [B, L, 5120]
```

这里的“融合”是把三个具有相同输入的投影矩阵拼成一个大矩阵，减少：

- 重复读取同一个输入。
- 重复的激活 FP8 量化。
- GEMM kernel launch 次数。

它不是把 QK Attention 计算也融合进 Linear。QKV Linear 和后面的
Attention kernel 仍然是两个阶段。

### 5.3 Q/K Norm、拆 head 和 RoPE

Q、K 会做 RMSNorm，V 不做：

```text
Q = RMSNorm(Q)
K = RMSNorm(K)
```

随后从 hidden 维拆成 40 个 head：

```text
[B, L, 5120]
    ↓ unflatten
[B, L, 40, 128]
```

再对 Q、K 应用视频 3D RoPE。RoPE 根据 token 对应的时间、纵向和横向位置旋转
Q/K 特征，使 Attention 能感知视频时空位置。

RoPE 不改变张量形状。

### 5.4 Ulysses SP 和 Self-Attention 核心

当前 `SP=2, Ulysses=2`。每卡最初持有一半视频 token 和全部 head：

```text
[B, 41958, 40, 128]
```

Ulysses All-to-All 后，每卡改为持有完整视频序列和一半 head：

```text
[B, 83916, 20, 128]
```

然后执行：

```text
scores = Q × K^T / sqrt(128)
P      = softmax(scores)
O      = P × V
```

当前 Self-Attention backend 是 SageAttention：

```text
--videoedit-self-attention-backend sage_attn
```

当前 kernel 为 `qk_int8_pv_fp8_cuda`：

- 传入 SageAttention 前的接口张量仍然是 BF16。
- kernel 内部对 Q/K 使用 INT8 低精度路径。
- PV 使用 FP8 低精度路径。
- 该低精度 Attention 核心与前面的 FP8 QKV Linear 是两件独立的优化。

Self-Attention 输出经反向 All-to-All 恢复本地序列布局，然后展平 head：

```text
[B, L, 40, 128] → [B, L, 5120]
```

### 5.5 Self 输出投影和残差

Attention 输出再经过：

```text
self_out = Linear(5120 → 5120)
x1 = x0 + gate_msa × self_out
```

随后融合算子同时完成：

1. `gate_msa × self_out`。
2. 加回残差 `x0`。
3. 对结果做 LayerNorm。

Self 分支结束后：

```text
x1:               [B, L, 5120]  # 更新后的 block 主残差
norm_hidden_states:[B, L, 5120]  # 作为 Cross-Attention 的输入
```

## 6. 第二段：VideoEdit Cross-Attention

VideoEdit 使用两组 Cross-Attention：

1. 视频 query 对文本 K/V。
2. 同一个视频 query 对参考图 K/V。

实现位于
[`WanVideoEditCrossAttention.forward`](../python/sglang/multimodal_gen/runtime/models/dits/wan_videoedit.py#L46)。

### 6.1 Cross-Attention 的 QKV 在哪里计算

Cross-Attention 不会从同一个张量同时生成 Q、K、V：

```text
Q 来自视频 hidden states
K/V 来自外部条件 context
```

具体为：

```text
Q_video = Linear(norm_hidden_states)

[K_text | V_text] = fused Linear(text_context)

[K_image | V_image] = fused Linear(image_context)
```

当前形状：

```text
视频输入：
    norm_hidden_states = [B, 41958, 5120]

文本条件：
    text_context = [B, 512, 5120]

图像条件：
    image_context = [B, 257, 5120]
```

投影后：

```text
Q_video:
    [B, 41958, 5120] → [B, 41958, 40, 128]

fused text KV:
    [B, 512, 5120]
        → [B, 512, 10240]
        → K_text, V_text 各 [B, 512, 40, 128]

fused image KV:
    [B, 257, 5120]
        → [B, 257, 10240]
        → K_image, V_image 各 [B, 257, 40, 128]
```

Q、文本 K 和图像 K 会做 RMSNorm，V 不做 RMSNorm。

Cross-Attention 的条件 K/V 在两个 SP rank 上是复制的，因此当前实现跳过 Ulysses All-to-All：每张卡直接使用本地 41,958 个视频 Q、全部 40 个 head，分别关注完整的 512 个文本 K/V 和 257 个图像 K/V。

### 6.2 两次 Cross-Attention

文本分支：

```text
O_text = softmax(Q_video × K_text^T / sqrt(128)) × V_text
```

图像分支：

```text
O_image = softmax(Q_video × K_image^T / sqrt(128)) × V_image
```

随后：

```text
O_cross = O_text + O_image
cross_out = Linear(O_cross, 5120 → 5120)
```

当前 Cross-Attention backend 是 BF16 FlashAttention：

```text
--videoedit-cross-attention-backend fa
```

因此当前 Cross 部分的精度情况是：

| Cross 子操作 | 当前精度/实现 |
|---|---|
| Q Linear | FP8 W8A8 scaled GEMM |
| Text K/V Linear | FP8 W8A8 fused KV GEMM |
| Image K/V Linear | FP8 W8A8 fused KV GEMM |
| `QK^T → softmax → PV` 核心 | BF16 FlashAttention |
| Output Linear | FP8 W8A8 scaled GEMM |

也就是说，Cross-Attention 的投影权重已经 FP8 量化，但核心 Attention 当前没有
使用 SageAttention。

### 6.3 Cross 残差和 FFN 前归一化

Cross 输出直接加回主残差，没有单独的 Cross gate：

```text
x2 = x1 + cross_out
```

然后执行：

```text
x2_norm = LayerNorm(x2) × (1 + c_scale_msa) + c_shift_msa
```

当前 CUDA 路径将 Cross 残差加法、LayerNorm、scale 和 shift 尽量合并到一个
fused kernel。

形状仍为：

```text
[B, L, 5120]
```

## 7. 第三段：FFN

FFN 不是 gated MLP，而是普通两层 MLP：

```text
ff_hidden = Linear(x2_norm, 5120 → 13824)
ff_hidden = GELU_tanh(ff_hidden)
ff_out    = Linear(ff_hidden, 13824 → 5120)
x3        = x2 + c_gate_msa × ff_out
```

维度变化：

```text
[B, L, 5120]
    ↓ fc_in
[B, L, 13824]
    ↓ GELU
[B, L, 13824]
    ↓ fc_out
[B, L, 5120]
    ↓ gated residual
[B, L, 5120]
```

最后的 `c_gate_msa × ff_out + x2` 使用 `MulAdd` 融合 elementwise kernel。

这时一个 block 结束，输出为：

```text
x3: [B, L, 5120]
```

它与 block 输入维度完全一致，因此能直接送入下一个 block。

## 8. 一个 block 中当前的 FP8 Linear 清单

开启 fused projections 后，每个 block 有 8 次 FP8 Linear 调用，但对应 12 个
逻辑权重矩阵：

| 顺序 | Runtime Linear | 逻辑矩阵 | 输入 → 输出 |
|---:|---|---:|---|
| 1 | Self fused QKV | 3 | `5120 → 15360` |
| 2 | Self output | 1 | `5120 → 5120` |
| 3 | Cross Q | 1 | `5120 → 5120` |
| 4 | Cross text fused KV | 2 | `5120 → 10240` |
| 5 | Cross image fused KV | 2 | `5120 → 10240` |
| 6 | Cross output | 1 | `5120 → 5120` |
| 7 | FFN fc_in | 1 | `5120 → 13824` |
| 8 | FFN fc_out | 1 | `13824 → 5120` |

40 个 block 一共是：

```text
8 × 40 = 320 个 block 内 runtime FP8 Linear 模块
```

再加上 timestep、文本、图像条件和最终输出等 8 个 Linear，当前运行时总计
328 个量化 Linear 模块。

### 8.1 一个静态 FP8 Linear 内部发生什么

以 Self fused QKV 为例：

```text
BF16 x_norm
    │
    ├── 使用离线标定得到的静态 input_scale
    ▼
FP8 activation
    │
    ├── FP8 weight（E4M3，per-output-channel weight scale）
    ▼
scaled FP8 GEMM
    │
    ├── 应用 input scale
    ├── 应用 weight scale
    └── bias epilogue
    ▼
BF16 QKV output
```

当前 checkpoint 的配置为：

```json
{
  "quant_method": "fp8",
  "fmt": "e4m3",
  "activation_scheme": "static",
  "weight_scale_granularity": "channel"
}
```

注意：

- 静态激活 scale 表示运行时不再扫描输入 absmax/min-max 来重新计算 scale。
- BF16 到 FP8 的类型转换仍然需要执行，所以静态量化并不等于完全没有激活量化
  kernel。
- 激活量化和 scaled GEMM 当前是相邻的两个阶段，不是一个完全融合的 kernel。
- 反量化不是先把整个 FP8 权重恢复为 BF16 再调用普通 GEMM；scale 会在 scaled
  GEMM 的输出缩放/epilogue 路径中应用。

实现入口：
[`fp8.py`](../python/sglang/multimodal_gen/runtime/layers/quantization/fp8.py#L583)。

## 9. 40 个 block 之后发生什么

最后一个 block 仍输出：

```text
[B, L, 5120]
```

Transformer 外层随后执行：

```text
1. timestep-modulated output LayerNorm
2. proj_out Linear
3. unpatchify
```

因为 patch size 为 `(1, 2, 2)`，每个 token 要还原一个 `2 × 2` 空间 patch，
且每个位置预测 16 个 latent 通道，所以：

```text
proj_out:
5120 → 16 × 1 × 2 × 2 = 64
```

形状变化：

```text
[B, L, 5120]
    ↓ norm_out
[B, L, 5120]
    ↓ proj_out
[B, L, 64]
    ↓ reshape + permute + unpatchify
[B, 16, T_latent, H_latent, W_latent]
```

这个 16 通道输出是当前 timestep 的 `noise_pred`/flow prediction，不是最终 RGB
视频。

如果使用 CFG：

```text
noise_pred =
    noise_uncond
    + guidance_scale × (noise_cond - noise_uncond)
```

最后 FlowMatch scheduler 更新 latent：

```text
z_next = z_t + noise_pred × (sigma_next - sigma_t)
```

实现位于：
[`videoedit_flow_match.py`](../python/sglang/multimodal_gen/runtime/models/schedulers/videoedit_flow_match.py#L72)。

经过所有 timestep 后，最终 16 通道 latent 才送入 VAE Decoder，恢复为视频帧。

## 10. QKV 计算位置总结

### Self-Attention

```text
输入来源：
Q、K、V 全部来自当前视频 hidden states。

当前实现：
一次 fused FP8 Linear 同时计算 [Q | K | V]。

后续：
Q/K RMSNorm → RoPE → SageAttention → output Linear。
```

### Text Cross-Attention

```text
Q 来源：
当前视频 hidden states。

K、V 来源：
文本 encoder hidden states。

当前实现：
Q 是一次 FP8 Linear；
K/V 是一次 fused FP8 KV Linear；
Attention 核心是 BF16 FlashAttention。
```

### Image Cross-Attention

```text
Q 来源：
与文本 Cross 共用同一份视频 Q。

K、V 来源：
参考图 image encoder hidden states。

当前实现：
K/V 是一次 fused FP8 added-KV Linear；
Attention 核心是 BF16 FlashAttention。
```

## 11. 最容易混淆的三个概念

### 11.1 QKV Linear 不等于 Attention 核心

QKV Linear：

```text
X × Wq
X × Wk
X × Wv
```

这是“激活 × 权重”的矩阵乘法，当前使用 W8A8 FP8。

Attention 核心：

```text
Q × K^T
softmax
P × V
```

这是“激活 × 激活”的计算，不包含模型 Linear 权重。当前 Self 使用
SageAttention 低精度 kernel，Cross 使用 BF16 FlashAttention。

### 11.2 投影融合不等于把整个 Attention 融成一个算子

当前融合的是：

```text
Q + K + V 三次 Linear → 一次 packed QKV Linear
K + V 两次 Linear     → 一次 packed KV Linear
```

没有融合的是：

```text
QKV Linear
Q/K RMSNorm
RoPE
Attention core
output Linear
```

这些仍然是多个算子阶段。

### 11.3 block 输出维度不变，但表示内容已经改变

单个 block：

```text
[B, L, 5120] → [B, L, 5120]
```

维度没有变化，不代表没有计算。每个 token 已经依次融合：

- 当前 timestep 的噪声强度信息。
- 全视频 token 之间的时空关系。
- 提示词语义。
- 参考图语义。
- FFN 的非线性特征变换。

40 个 block 持续迭代这种表示，最后再投影成 scheduler 所需的 16 通道
noise/flow prediction。
