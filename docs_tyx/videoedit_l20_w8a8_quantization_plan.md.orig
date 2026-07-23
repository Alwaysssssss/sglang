# VideoEdit DiT 在 NVIDIA L20 上的 W8A8 量化落地计划

## 1. 结论先行

本项目建议采用下面的主线，而不是在“动态量化”和“离线量化”之间二选一：

- **权重：离线量化为 FP8 E4M3FN，按输出通道（per-output-channel）保存 scale。**
- **激活：推理时动态量化为 FP8 E4M3FN，按 token（per-token）计算 scale。**
- **累加、bias、残差、Norm、attention softmax 和 Linear 输出：继续使用 BF16/FP32。**
- 第一版只量化 40 个 transformer block 中的 attention 和 FFN Linear；保留 `condition_embedder`、`proj_out`、`patch_embedding`、Norm 为 BF16/FP32。
- FP8 是 L20 上的第一优先级。INT8 W8A8 作为第二条对照分支，只有 FP8 主线稳定后再接入。
- 离线静态 activation scale 只作为第二阶段实验，不作为首版生产方案。

这里的“离线权重 + 动态激活”仍然是标准的 W8A8：进入 GEMM 的 weight 和 activation 都是 8 bit，只是 weight scale 提前生成，activation scale 在每次推理时生成。

推荐最终产物：

```text
/home/tyx/workspace/difusser-model/step-55000-diffusers-lh/
└── transformer-fp8-e4m3fn-channel-dynamic-act/
    ├── config.json
    ├── diffusion_pytorch_model-00001-of-00004.safetensors
    ├── ...
    ├── diffusion_pytorch_model.safetensors.index.json
    ├── quantization_manifest.json
    └── source_checksums.json
```

不要覆盖原始目录：

```text
/home/tyx/workspace/difusser-model/step-55000-diffusers-lh/transformer
```

## 2. 已确认的模型与仓库现状

本文是 `docs_tyx/videoedit_wan14b_dit_linear_quant_plan.md` 的后续落地版。旧文档主要描述 `quant_config` 透传；相关代码目前已经存在。本文重点补足 L20 backend、offline checkpoint 格式、activation 决策、校准、benchmark 和回退门槛。

当前会话所在机器无法通过 `nvidia-smi` 访问 GPU，仓库目录下也没有可直接使用的 `.venv`，因此本文没有把本机实测结果冒充为 L20 结论。所有 kernel、显存和速度判断都必须在 Phase 0 的目标 L20 环境中复核。

### 2.1 checkpoint 结构

已直接检查指定目录。它是 `WanVideoEditTransformer3DModel`，不是 Wan 1.3B：

- `num_layers=40`
- `num_attention_heads=40`
- `attention_head_dim=128`
- `hidden_size=5120`
- `ffn_dim=13824`
- `in_channels=36`
- `out_channels=16`
- `image_dim=1280`
- `added_kv_proj_dim=5120`
- 原始 tensor 总大小：`32,848,151,168 bytes`，约 `30.59 GiB`
- 其中 BF16 2-D Linear weight 共有 488 个，约 `30.58 GiB`

Linear weight 分布如下：

| 范围 | Linear 数量 | BF16 weight | 占全部 2-D Linear weight |
|---|---:|---:|---:|
| `blocks.*.attn1.*` | 160 | 7.8125 GiB | 25.55% |
| `blocks.*.attn2.*` | 240 | 11.7188 GiB | 38.32% |
| `blocks.*.ffn.*` | 80 | 10.5469 GiB | 34.49% |
| `condition_embedder.*` | 7 | 0.4987 GiB | 1.63% |
| `proj_out` | 1 | 0.0006 GiB | 0.002% |

只量化 480 个 block Linear 就覆盖了 98.37% 的 Linear weight。保留条件投影和输出投影为 BF16，几乎不损失显存收益，但能降低首版质量风险。

按“480 个 block weight FP8 + 每输出通道 FP32 scale + 其余参数保持原精度”估算，离线 transformer tensor 总量约为：

```text
16,711,303,808 bytes ≈ 15.56 GiB
```

相对原 checkpoint，weight 文件体积预计减少约 49%。实际 GPU 峰值还包含 activation、attention workspace、VAE 和其他组件，不能直接按 49% 推导端到端峰值。

### 2.2 当前已经具备的能力

当前代码已经完成 VideoEdit/Wan Linear 的 `quant_config` 透传：

- `condition_embedder`
- `blocks.*.attn1`
- `blocks.*.attn2`，包括 VideoEdit 的 `add_k_proj` / `add_v_proj`
- `blocks.*.ffn`
- `proj_out`

CUDA FP8 路径当前支持：

- BF16 checkpoint 加载后在线量化 weight；
- FP8 serialized checkpoint；
- dynamic activation；
- static activation；
- tensor-wise weight scale；
- block-wise weight scale；
- L20/SM89 且 CUDA 版本不低于 12.4 时的 CUTLASS/`sgl_kernel` FP8 路径。

当前 CLI 已提供一个非常适合做第一阶段可行性验证的参数：

```text
--transformer-quantization fp8_dynamic
```

它会从 BF16 checkpoint 加载，再把 Linear weight 在线转成 FP8；在 CUDA 上 activation 使用动态 per-token scale。它不是最终离线产物，但应作为离线方案的行为和性能基准。

### 2.3 当前必须修复的缺口

现有能力还不能直接产出理想的 L20 离线 W8A8 模型，原因如下。

1. `convert_hf_to_fp8.py` 用字符串排除规则筛层，其中 `"net" not in key` 会跳过所有 `blocks.*.ffn.net.*`。这会漏掉约 10.55 GiB 的 BF16 FFN weight，不能称为完整的 DiT W8A8。
2. 当前 converter 的 `channel` 策略写出 `quant_method="compressed-tensors"`，但 multimodal diffusion 的本地量化注册表没有接通该配置，因此不能把它当成当前可用的离线 per-channel FP8 路径。
3. 当前 `Fp8LinearMethod` 的 serialized 非 block checkpoint 按 per-tensor scale 创建参数，尚不能直接加载 `[out_features, 1]` 的 per-channel scale。
4. block FP8 在 L20/SM89 上会走 Triton fallback；当前代码的 CUTLASS block FP8 分支要求 Hopper 或更新架构。它需要实测，但不应默认认为比 Ada 上的普通 per-channel FP8 kernel 更快。
5. converter 只把量化配置写到输出目录的 `config.json`。如果只传 `--transformer-weights-path`，loader 目前优先使用基础 transformer 的配置，并不保证读取 override 目录的 `config.json`。首版可用 `--transformer-path <量化目录>`；正式方案应补上 override 目录配置或 safetensors metadata 的解析。
6. 当前 diffusion 量化注册表没有 CUDA INT8 W8A8；`ModelSlim` 实现明确面向 Ascend/NPU。仓库 SRT 底层已有 `sgl_kernel.int8_scaled_mm` 和动态 per-token INT8 实现，但需要适配到 multimodal diffusion 的 `LinearBase`、参数类和 loader。
7. 当前转换脚本按整个 shard 加载到 CUDA，且默认多 worker。四个 shard 合计约 30.59 GiB，同时转换可能造成 GPU/host OOM。正式 converter 必须串行处理 shard，并尽量逐 tensor 转换。

因此，不应直接运行当前脚本的 `--strategy channel` 或默认多 worker 后就开始做性能结论。

## 3. 为什么主线选择 FP8，而不是先做 INT8

NVIDIA L20 属于 Ada（compute capability 8.9）。官方 CUDA 文档确认 SM89 Tensor Core 支持 FP8 和 INT8；Ada 还引入了第四代 Tensor Core 的 FP8 支持：

- [CUDA Compute Capability 与 Tensor Core 数据类型表](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/05-appendices/compute-capabilities.html)
- [NVIDIA Ada GPU Architecture Tuning Guide](https://docs.nvidia.com/cuda/archive/12.9.2/ada-tuning-guide/index.html)
- [CUTLASS 的 Ada/SM89 FP8 MMA 说明](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/wmma_programming.html)

本项目先做 FP8 的主要原因不是硬件不支持 INT8，而是软件和数值特性：

- 当前 diffusion runtime 的 FP8 Linear 已经接通，INT8 diffusion adapter 尚未接通；
- FP8 E4M3 有指数位，比对称 INT8 更容易覆盖 diffusion activation 在不同 timestep、CFG 分支和窗口间变化的动态范围；
- 在线 FP8 dynamic 已经可以先验证真实 kernel、质量和上限；
- FP8 与 INT8 都是 1 byte/weight，单看离线 weight 显存没有本质差异；
- INT8 只有在 L20 实测 kernel 更快、同时质量过关时才有切换价值。

所以 INT8 不是被排除，而是放到 FP8 之后做有意义的 A/B，而不是同时维护两套尚未验证的实现。

## 4. 量化格式和首版层范围

### 4.1 FP8 数值格式

首版使用：

```text
Weight:     FP8 E4M3FN, static, per-output-channel, symmetric
Activation: FP8 E4M3FN, dynamic, per-token, symmetric
Accumulator/output: BF16（kernel 内部按实现选择更高精度累加）
Scale:      FP32
Bias:       BF16
Norm:       BF16/FP32，保持现状
```

离线 weight 的基本计算为：

```text
s_w[i] = max(abs(W[i, :])) / 448
q_w[i, :] = cast_fp8_e4m3fn(clamp(W[i, :] / s_w[i], -448, 448))
```

全零通道必须特殊处理，scale 使用非零安全值，禁止产生 NaN/Inf。

动态 activation 的基本计算为：

```text
s_x[token] = max(abs(X[token, :])) / 448
q_x[token, :] = cast_fp8_e4m3fn(clamp(X[token, :] / s_x[token], -448, 448))
```

### 4.2 首版量化 allowlist

只量化真正映射到 SGLang quantized Linear 的 2-D weight，使用显式 allowlist，不再使用宽泛的字符串黑名单：

```text
blocks.<0..39>.attn1.to_q
blocks.<0..39>.attn1.to_k
blocks.<0..39>.attn1.to_v
blocks.<0..39>.attn1.to_out.0

blocks.<0..39>.attn2.to_q
blocks.<0..39>.attn2.to_k
blocks.<0..39>.attn2.to_v
blocks.<0..39>.attn2.to_out.0
blocks.<0..39>.attn2.add_k_proj
blocks.<0..39>.attn2.add_v_proj

blocks.<0..39>.ffn.net.0.proj
blocks.<0..39>.ffn.net.2
```

总数必须严格等于 480。数量不等时 converter 立即失败，不能静默生成部分量化模型。

首版保留为 BF16/FP32：

```text
condition_embedder.time_embedder.linear_1
condition_embedder.time_embedder.linear_2
condition_embedder.time_proj
condition_embedder.text_embedder.linear_1
condition_embedder.text_embedder.linear_2
condition_embedder.image_embedder.ff.net.0.proj
condition_embedder.image_embedder.ff.net.2
proj_out
patch_embedding
所有 norm / scale_shift_table / bias / rotary / attention softmax
```

保留 `condition_embedder` 和 `proj_out` 只增加约 0.5 GiB BF16 weight，却能避免首版把最直接影响条件注入与最终像素预测的投影也量化。

## 5. activation：动态还是静态

### 5.1 推荐：动态 per-token activation

首版明确选择动态 activation，理由如下：

- VideoEdit activation 会随 denoise timestep 变化；
- cond/uncond CFG 分支的分布不同；
- mask 大小、参考图、prompt、分辨率和 crop 会改变分布；
- 多窗口推理的首窗、中间窗、尾窗分布可能不同；
- per-token scale 能局部适应变化，不需要让所有 token 共用一个最坏值；
- 视频 token 数通常较大，scale 归约开销相对大 GEMM 更容易被摊薄，但仍要实测。

动态并不代表没有离线模型。离线 checkpoint 保存 FP8 weight 和 weight scale；activation 仍在每次 GEMM 前转成 FP8。

### 5.2 静态 activation 的正确含义

静态 activation 不是把 activation 本身离线保存，而是通过校准集为每个 Linear 保存一个固定 `input_scale`。推理时仍需把 BF16 activation 转成 FP8，只是不用重新计算 amax。

当前 FP8 runtime 的 static scheme 每层只有一个 scale。它不能天然表达：

- 不同 timestep 一个 scale；
- cond/uncond 各自一个 scale；
- 不同窗口或分辨率一个 scale。

为了覆盖所有场景，一个全局固定 scale 往往会被极端值拉大，导致大多数 activation 的有效精度下降；使用 percentile clipping 又会带来饱和风险。因此它只应作为动态方案之后的性能优化实验。

### 5.3 静态 activation 的启用门槛

只有同时满足以下条件才考虑上线静态 activation：

1. 动态 activation 已经质量和稳定性通过；
2. profile 证明 activation amax/quant kernel 是可见瓶颈；
3. static 相对 dynamic 的 **DiT forward** 额外加速至少 5%；
4. holdout 数据上 saturation rate、逐步 latent 误差和最终视频质量均通过门槛；
5. 生产的 scheduler、step 数、CFG 和输入分布稳定。

如果生产同时使用多种 `num_inference_steps`、scheduler 或分辨率，优先继续使用 dynamic；不要用一个校准 scale 强行覆盖所有 profile。

## 6. 分阶段实施计划

### Phase 0：冻结 BF16 baseline 和目标机环境

目标：在做任何离线转换之前，先确定正确性参考和 L20 实际软件栈。

在目标 L20 上记录：

```text
GPU 型号、显存、compute capability
driver 版本
CUDA runtime/toolkit 版本
torch 版本及 torch.version.cuda
sglang-kernel 版本
Triton 版本
cutlass_fp8_supported() 的返回值
torch.float8_e4m3fn 是否可用
sgl_kernel.fp8_scaled_mm / int8_scaled_mm 是否可导入
```

当前仓库依赖声明是 `torch==2.9.1`、CUDA 12.9 相关包、`sglang-kernel==0.4.1`，但必须以 L20 推理环境实际安装值为准。

冻结一组 BF16 baseline：

- 关闭 TeaCache；
- 关闭 DiT offload；
- 首轮关闭 `torch.compile`；
- 固定 attention backend；
- 固定输入、mask、reference、prompt、negative prompt、seed；
- 固定 `num_frames`、`infer_len`、overlap、scheduler、step 数和 CFG；
- 先跑短 smoke，再跑单窗口 81 帧，最后跑多窗口样本；
- 至少 2 次 warmup、5 次正式计时，报告 median 和 p95，不只报单次耗时。

保存：

```text
BF16 输出视频
perf JSON
峰值/稳态显存
每个 denoise step 的 DiT forward 时间
完整运行日志
环境 manifest
```

通过标准：BF16 可重复完成推理，输出无 NaN/Inf，计时波动可解释。

### Phase 1：先验证在线 FP8 dynamic

目标：不等待 offline converter 改造，先判断 L20 上当前 FP8 kernel 是否真的加速 VideoEdit。

使用原 BF16 transformer，加：

```text
--transformer-quantization fp8_dynamic
```

此阶段重点检查：

- 488 个目标 Linear 是否都得到 `Fp8LinearMethod`；
- `condition_embedder`、`attn1`、VideoEdit `attn2`、FFN、`proj_out` 是否无遗漏；
- weight 加载后是否实际变为 `torch.float8_e4m3fn`；
- activation 是否走 dynamic per-token；
- L20 上是否调用 `fp8_scaled_mm`，而不是 dequant 后 BF16 matmul fallback；
- 是否存在大量 shape 不合适而回退的 Linear；
- load-time 峰值是否因先加载 BF16 weight 而接近或超过 L20 显存上限。

应增加一次模型启动审计，打印并保存：

```text
quant_method -> layer count
weight dtype -> layer count
weight scale shape 分布
ignored/unquantized Linear 列表
选中的 kernel/backend
```

Phase 1 的作用是验证数值行为和速度上限，不是最终交付。在线量化仍依赖 32.85 GB 原权重，并可能有更高的加载瞬时显存和更长启动时间。

进入下一阶段的条件：

- 所有预期 FP8 Linear 都走真实 W8A8 kernel；
- 无 NaN/Inf、shape mismatch 或 dtype mismatch；
- DiT forward 相对 BF16 有明确加速；
- 短样本没有明显质量崩坏。

如果这里没有 DiT 加速，先用 profiler 定位 kernel/shape/fallback，不要立刻投入离线格式开发，因为离线权重只解决启动和存储，不会自动修复低效 kernel。

### Phase 2：实现生产目标——离线 per-channel FP8 weight + dynamic activation

这是本计划的核心实现阶段。

#### 2.1 改造 offline converter

修改 `python/sglang/multimodal_gen/tools/convert_hf_to_fp8.py`，或新增 VideoEdit 专用 wrapper，要求：

- 增加明确的 `wan_videoedit` profile/allowlist；
- 仅量化匹配 allowlist 的 2-D `.weight`；
- 断言 block Linear 数量为 480；
- `--strategy channel` 输出 diffusion runtime 能识别的 FP8 config，而不是未接通的 `compressed-tensors`；
- scale 形状固定为 `[out_features, 1]`，dtype 为 FP32；
- bias、Norm、条件投影、输出投影保持原 dtype；
- 输入 shard 串行处理，默认 `max_workers=1`；
- 最好逐 tensor 转 GPU 量化后搬回 CPU，避免同时保留整 shard 的 BF16 和 FP8 CUDA tensor；
- 输出先写临时目录，所有验证完成后再原子切换为最终目录；
- 不复制旧的、缺少 scale key 的 stale index；重新生成与实际 shard/key 完全一致的 index；
- 为每个 shard 写 SHA256，并记录 source checkpoint SHA256；
- 写 `quantization_manifest.json`。

manifest 至少包含：

```json
{
  "source": ".../step-55000-diffusers-lh/transformer",
  "format": "fp8_e4m3fn",
  "weight_scheme": "channel",
  "activation_scheme": "dynamic_token",
  "quantized_linear_count": 480,
  "skipped_linear_count": 8,
  "quantized_weight_bytes_before": 32296140800,
  "quantized_weight_bytes_after": 16148070400,
  "layers": {},
  "source_checksums": {},
  "output_checksums": {}
}
```

每层记录 shape、原 dtype、目标 dtype、scale shape、最大绝对误差、相对 L2 误差、是否含 NaN/Inf。

#### 2.2 扩展 serialized FP8 per-channel scale

修改 `python/sglang/multimodal_gen/runtime/layers/quantization/fp8.py`：

- 在 `Fp8Config` 中显式表示 `weight_scale_scheme="channel"`；
- serialized channel 模式使用 `ChannelQuantScaleParameter([out_features, 1])`；
- TP column-parallel 时正确切分 output channel scale；
- TP row-parallel 时 scale 与 output channel 对齐，weight 沿 input 维切分；
- `process_weights_after_loading()` 不再把已经是 channel-wise 的 scale 当成 tensor-wise 再扩展；
- 保持现有 tensor/block checkpoint 的向后兼容；
- 在 L20 上确保最终进入 `fp8_scaled_mm` 的 weight layout、scale layout 和 dtype 正确。

#### 2.3 让量化配置与 override 权重一起加载

有两个可接受实现，优先选 A：

- **A：** `--transformer-weights-path` 指向目录时，loader 读取该目录的 `config.json` 量化字段，或读取 safetensors `_quantization_metadata`；结构配置仍以基础 transformer 为准。
- **B：** 首版要求 `--transformer-path "$FP8_TRANSFORMER_DIR"`，让配置和权重来自同一目录。

正式交付建议完成 A，并定义清楚优先级：

```text
显式 override checkpoint metadata/config
> 基础 transformer quantization_config
> 命令行 runtime quantization
```

检测到两个互相冲突的量化来源时必须报错，不能静默覆盖。

#### 2.4 离线 checkpoint 完整性验证

在启动完整模型前先做纯 checkpoint validator：

- safetensors 所有 key 在 index 中且只出现一次；
- index 不引用不存在的 key/file；
- 480 个 FP8 weight；
- 480 个 channel scale；
- 8 个保留的 BF16 Linear weight；
- 无 NaN/Inf scale；
- scale 全部大于 0；
- shape 与源 checkpoint 一致；
- 量化后反量化误差统计在 manifest 中；
- 输出总 tensor 大小接近 15.56 GiB；偏差较大时停止加载排查。

### Phase 3：FP8 质量敏感性与 mixed-precision 回退

先比较以下三组：

```text
A. BF16 baseline
B. online FP8 dynamic（488 个 Linear）
C. offline FP8 channel + dynamic activation（480 个 block Linear）
```

B 与 C 的意义不同：

- B 验证当前 runtime 的完整 FP8 行为；
- C 是生产候选，条件投影和 `proj_out` 为 BF16；
- B 与 C 差异较大时，要区分是 layer scope 还是 offline scale/loader 造成的。

如果 C 仍有明显质量问题，按测得的 layer sensitivity 回退，不要一次恢复整个模型。建议顺序：

1. 先检查是否有 kernel/scale/layout bug；
2. 恢复误差最大的具体层；
3. 优先评估 `attn2.add_k_proj` / `add_v_proj`，它们直接承载参考图条件；
4. 再评估首尾若干 transformer block；
5. 最后才大范围恢复整个 attn2 或 FFN。

离线 converter 应支持 `--exclude-regex` 或 layer list，使 mixed-precision checkpoint 可复现，而不是手工改 tensor。

### Phase 4：静态 activation 校准实验（可选）

#### 4.1 校准数据集

建议 32～64 个有代表性的 VideoEdit 请求，另留至少 20% holdout，不参与 scale 选择。覆盖：

- 生产中的主要分辨率和宽高比；
- 小 mask（<5%）、中 mask（5%～25%）、大 mask（>25%）；
- 静止 mask 和移动 mask；
- 有/无 reference image（如果生产两种都用）；
- 人物、文字、纹理、快速运动、低光等内容；
- 短单窗口、正常 81 帧、长视频多窗口；
- 首窗、中间窗、尾窗和 overlap；
- cond/uncond 两个 CFG 分支；
- 生产使用的所有 timestep；
- 生产实际使用的 scheduler、step 数和 guidance scale。

如果生产存在多套 scheduler 或 step 数，应分别校准 profile；若运行时不能可靠选择 profile，则保留动态 activation。

#### 4.2 采集方式

为每个量化 Linear 的输入流式采集统计，禁止保存全部 activation：

```text
shape / token 数分布
每次调用的 absmax
P99 / P99.9 / P99.99 / P99.999
均值、方差、RMS
按 timestep bucket 的 absmax
cond/uncond 分支
window index
候选 scale 下的 saturation rate
```

第一版候选：

```text
minmax scale
P99.99 clipping scale
P99.999 clipping scale
```

每层只能在 calibration set 上选择候选，在 holdout 上做最终判断。

#### 4.3 静态产物

静态 checkpoint 在 dynamic 版本基础上增加每个量化 Linear 的 FP32 `input_scale`，并在 config 中写：

```json
{
  "quant_method": "fp8",
  "weight_scale_scheme": "channel",
  "activation_scheme": "static"
}
```

同时保存独立的 `activation_calibration.json`，记录数据集版本、命令、scale 算法、每层饱和率和 holdout 结果。

不允许只把 scale 塞进 safetensors 而不保存校准来源。

### Phase 5：CUDA INT8 W8A8 对照分支（FP8 稳定后）

目标格式：

```text
Weight:     INT8, static, per-output-channel, symmetric
Activation: INT8, dynamic, per-token, symmetric
Accumulator: INT32
Output:     BF16
Scale:      FP32
```

量化公式：

```text
s_w[i] = max(abs(W[i, :])) / 127
q_w[i, :] = round(clamp(W[i, :] / s_w[i], -127, 127))

s_x[token] = max(abs(X[token, :])) / 127
q_x[token, :] = round(clamp(X[token, :] / s_x[token], -127, 127))
```

实现建议：

- 参考 SRT 的 `W8A8Int8Config` / `W8A8Int8LinearMethod`；
- 新增 multimodal diffusion 版本，使用本地 `LinearBase`、`ModelWeightParameter`、`ChannelQuantScaleParameter` 和 `UnquantizedLinearMethod`；
- 注册 `quant_method="w8a8_int8"`；
- forward 调用 `sgl_kernel.int8_scaled_mm`；
- offline converter 输出 INT8 weight 和 `[out_features, 1]` FP32 scale；
- 第一版仍只量化 480 个 block Linear；
- 不要把 NPU 的 ModelSlim checkpoint 格式直接复用到 CUDA；
- 先只支持 offline serialized INT8，暂不做 BF16 load 后在线 INT8 导出。

如果 INT8 质量不及 FP8，不要马上做静态 activation。优先分析 activation channel outlier；只有 INT8 有明显速度优势时，才考虑 SmoothQuant 类的离线通道平滑。SmoothQuant 会改权重与上游缩放关系，必须单独作为新 checkpoint 类型验证。

INT8 进入生产候选的必要条件：

- 相对 FP8 dynamic 有稳定、可重复的实际加速；
- weight 显存并不会比 FP8 更低，因此速度必须证明它的维护价值；
- 质量达到与 FP8 相同的门槛；
- `int8_scaled_mm` 无 BF16 fallback。

### Phase 6：组合功能与生产化

量化主线通过后，再逐一验证：

1. `torch.compile`
2. TeaCache / Cache-DiT
3. 多窗口与 overlap
4. layerwise offload
5. TP/SP 多卡
6. serve 并发请求

不要在首轮同时打开 TeaCache、compile、offload 和量化，否则出现质量或性能问题时无法归因。

对于 48 GB L20，FP8 DiT 很可能让 no-offload 更可行，因此优先验证 no-offload。只有实际 OOM 时才加入 layerwise offload；量化与 offload 组合会改变 H2D 带宽、prefetch 和 weight layout，必须单独计时。

## 7. 测试和 benchmark 矩阵

### 7.1 最小矩阵

| ID | Weight | Activation | 用途 | 优先级 |
|---|---|---|---|---|
| BF16 | BF16 | BF16 | 正确性与性能基线 | 必做 |
| FP8-ONLINE-DYN | 在线 per-channel FP8 | dynamic per-token FP8 | 验证当前 kernel 上限 | 必做 |
| FP8-OFFLINE-TENSOR-DYN | 离线 per-tensor FP8 | dynamic per-token FP8 | 兼容性 smoke，不是最终候选 | 可选 |
| FP8-OFFLINE-CHANNEL-DYN | 离线 per-channel FP8 | dynamic per-token FP8 | 生产主候选 | 必做 |
| FP8-OFFLINE-BLOCK-DYN | 离线 128x128 FP8 | dynamic token-group FP8 | Triton 对照 | 可选 |
| FP8-OFFLINE-CHANNEL-STATIC | 离线 per-channel FP8 | calibrated static FP8 | 额外性能实验 | 后做 |
| INT8-OFFLINE-CHANNEL-DYN | 离线 per-channel INT8 | dynamic per-token INT8 | 与 FP8 对照 | 后做 |

在 L20 上，block FP8 当前会走 Triton，而普通 channel FP8 可以走 SM89 CUTLASS/`sgl_kernel`；因此不能只看 block quant 的误差更小就默认选 block，必须同时看 kernel 时间。

### 7.2 每组都要记录的性能指标

```text
checkpoint 体积
模型启动时间
load-time 峰值 GPU/CPU 内存
加载完成后的 steady GPU memory
单个 DiT forward p50/p95
每个 denoise step 时间
完整 denoise 时间
完整请求端到端时间
token shape / GEMM M-N-K 分布
FP8/INT8 kernel 占比
量化 activation kernel/amax 开销
BF16 fallback 次数和层名
GPU utilization / memory bandwidth / power（可取得时）
```

端到端速度必须和 DiT-only 速度同时报告。VAE decode、预处理、视频编解码不会因为 DiT W8A8 自动加速，端到端收益会受 Amdahl 定律限制。

### 7.3 建议的初始 go/no-go 门槛

这些是工程门槛，不是最终产品质量标准；产品侧可再收紧：

- offline FP8 transformer tensor 总量不高于 17 GiB；
- 480 个目标 Linear 的真实 8-bit kernel 覆盖率为 100%；
- steady transformer 参数显存相对 BF16 降低至少 45%；
- DiT forward p50 相对 BF16 加速至少 1.25x；
- 端到端 p50 相对 BF16 加速至少 1.15x；
- 5 次正式运行无 NaN/Inf/OOM；
- 输出帧数、尺寸和编码均正确；
- 同 seed 视频 `SSIM mean >= 0.95`、`SSIM p05 >= 0.90` 作为初筛；
- 人工检查没有新增身份漂移、文字破坏、mask 边缘闪烁、纹理沸腾或窗口接缝；
- static activation 必须比 dynamic activation 再快至少 5%，否则保留 dynamic；
- INT8 必须相对 FP8 有稳定收益，否则不增加第二套生产格式。

SSIM 不能单独决定 diffusion 质量。它只用于快速发现明显偏差，最终还需要 ROI、时序和人工检查。

## 8. 质量验证设计

### 8.1 逐层和逐 denoise step

在少量诊断样本上对 BF16 与量化模型采集：

```text
每层 Linear 输出 cosine similarity
normalized RMSE
每个 block 输出 cosine / relative L2
每个 denoise step 的 DiT 最终输出误差
latent trajectory 误差
NaN/Inf 与饱和率
```

诊断采集只用于少量样本，避免把完整 activation 持久化造成巨大 I/O。

### 8.2 最终视频

使用仓库已有的 `scripts/compare_video_similarity.py` 生成：

```text
SSIM mean/min/p05
PSNR
MAE/MSE
worst frames 和 diff 图
```

VideoEdit 还应分区域检查：

- mask 内修复区域；
- mask 边缘带；
- mask 外区域；
- overlap/窗口接缝；
- 相邻帧的时序稳定性。

若启用 paste-back，mask 外像素可能被后处理直接恢复，因此不能让全帧 SSIM 掩盖 mask 内质量问题。至少额外报告 mask ROI 和边缘带指标。

### 8.3 数据集分层

至少分三层：

```text
Smoke: 2～4 个短请求，验证加载和数值稳定
Regression: 10～20 个固定请求，每次代码改动必跑
Qualification: 32～64 个代表生产分布的请求，用于最终 go/no-go
```

## 9. 单元测试与集成测试清单

### 9.1 converter 单测

- FP8 per-channel scale 和反量化公式；
- 全零 tensor；
- 极小值和极大值；
- shape 不是 128 倍数时仍能处理 channel scheme；
- allowlist 精确得到 480 层；
- `condition_embedder` 和 `proj_out` 精确得到 8 个 BF16 Linear；
- 不把 bias、Norm 名称误放进 ignored Linear 列表；
- index、metadata、manifest 一致；
- 单 shard 和多 shard；
- 中断后不留下看似完整的最终目录。

### 9.2 runtime 单测

- `Fp8Config(weight_scale_scheme="channel")` 创建正确参数；
- ColumnParallel/RowParallel 的 scale shard；
- TP=1 和 TP=2；
- serialized channel FP8 load 后 dtype/layout 正确；
- ignored layer 仍走 `UnquantizedLinearMethod`；
- tensor/block 旧格式无回归；
- config 来源优先级和冲突报错；
- 目标层 `quant_method` 数量审计。

### 9.3 CUDA kernel 测试

对实际 VideoEdit 主要 shape 进行 BF16/FP8 数值与速度测试：

```text
[M, 5120] x [5120, 5120]
[M, 5120] x [5120, 13824]
[M, 13824] x [13824, 5120]
```

`M` 不要只用小值，要从实际单窗口和 bbox crop 运行中采样 token 数分布。确认 profiler 中的 kernel 名称和 fallback。

## 10. 推荐命令流程

下面命令是流程模板。输入视频、mask、完整 base model 路径按实际环境补充。

### 10.1 公共变量

```bash
export SOURCE_TRANSFORMER=/home/tyx/workspace/difusser-model/step-55000-diffusers-lh/transformer
export FP8_TRANSFORMER=/home/tyx/workspace/difusser-model/step-55000-diffusers-lh/transformer-fp8-e4m3fn-channel-dynamic-act
export MODEL_PATH=/path/to/VideoEdit-diffusers-model
export INPUT_VIDEO=/path/to/input.mp4
export INPUT_MASK=/path/to/mask.mp4
export REF_IMAGE=/path/to/reference.png
export OUT_DIR=/path/to/quant-benchmark
```

### 10.2 在线 FP8 可行性测试

```bash
python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$MODEL_PATH" \
  --transformer-path "$SOURCE_TRANSFORMER" \
  --transformer-quantization fp8_dynamic \
  --prompt "一个男人在舞台演讲，背后有两排文字。" \
  --video-input-path "$INPUT_VIDEO" \
  --mask-input-path "$INPUT_MASK" \
  --reference-image-path "$REF_IMAGE" \
  --output-path "$OUT_DIR/fp8_online_dynamic" \
  --output-file-name output.mp4 \
  --perf-dump-path "$OUT_DIR/fp8_online_dynamic/perf.json" \
  --num-frames 80 \
  --infer-len 81 \
  --overlap 5 \
  --num-inference-steps 1 \
  --guidance-scale 5.0 \
  --seed 42 \
  --dtype bf16 \
  --no-enable-teacache \
  --no-dit-cpu-offload \
  --no-dit-layerwise-offload \
  --no-enable-frame-interpolation \
  --no-enable-upscaling \
  --num-gpus 1 \
  --tp-size 1
```

这里沿用当前已知的 80 输入帧 / 81 infer length / 1 step 形态作为 smoke 示例。正式质量和性能结论必须再按生产实际 step 数执行，不能只用 1 step。

### 10.3 离线转换（完成 Phase 2 改造后）

预期接口示例：

```bash
python -m sglang.multimodal_gen.tools.convert_hf_to_fp8 \
  --model-dir "$SOURCE_TRANSFORMER" \
  --save-dir "$FP8_TRANSFORMER" \
  --profile wan_videoedit \
  --strategy channel \
  --activation-scheme dynamic \
  --max-workers 1 \
  --verify
```

注意：这是改造后的目标接口；当前 converter 尚不满足本计划要求，尤其不能直接把当前 `channel` 输出当成可加载的 diffusion FP8 checkpoint。

### 10.4 离线模型推理

在 loader 支持 override 目录量化 config 前，优先用：

```text
--transformer-path "$FP8_TRANSFORMER"
```

完成 override config/metadata 支持后，也可使用：

```text
--transformer-path "$SOURCE_TRANSFORMER"
--transformer-weights-path "$FP8_TRANSFORMER"
```

离线 checkpoint 已经带量化配置时，不再同时传 `--transformer-quantization fp8_dynamic`，否则应触发量化来源冲突错误。

## 11. 交付物

FP8 主线完成时应交付：

1. 可复现的 offline converter；
2. `transformer-fp8-e4m3fn-channel-dynamic-act` checkpoint；
3. source/output checksums；
4. quantization manifest；
5. checkpoint validator；
6. FP8 serialized per-channel runtime 支持；
7. loader 对 override 量化配置的确定性处理；
8. layer/kernel 审计日志；
9. BF16 vs online FP8 vs offline FP8 benchmark 报告；
10. 固定 regression 数据集与质量报告；
11. 一份 L20 单卡运行 runbook；
12. 原始 BF16 checkpoint 完整保留，命令行可一键回退。

建议的后续代码文件范围：

```text
python/sglang/multimodal_gen/tools/convert_hf_to_fp8.py
python/sglang/multimodal_gen/runtime/layers/quantization/fp8.py
python/sglang/multimodal_gen/runtime/loader/transformer_load_utils.py
python/sglang/multimodal_gen/runtime/utils/quantization_utils.py
python/sglang/multimodal_gen/test/unit/test_*fp8*.py
```

INT8 分支另行增加 diffusion `w8a8_int8` config/method 和 converter 测试，不与 FP8 首版混在同一个变更中。

## 12. 决策树

```text
BF16 baseline 稳定？
  ├─ 否 -> 先修 baseline，不进入量化
  └─ 是
      |
      v
online FP8 dynamic 是否走真实 L20 FP8 kernel并加速？
  ├─ 否 -> profile fallback / shape / CUDA / sgl-kernel
  └─ 是
      |
      v
offline per-channel FP8 + dynamic activation 是否通过质量和性能门槛？
  ├─ 否 -> 检查格式/layout，再做按层 BF16 回退
  └─ 是 -> 作为生产主候选
      |
      +--> dynamic activation 开销是否成为瓶颈？
      |      ├─ 否 -> 保持 dynamic
      |      └─ 是 -> 做 static calibration A/B
      |
      +--> 是否还需要更高速度？
             ├─ 否 -> 结束
             └─ 是 -> 接 CUDA INT8 W8A8，与 FP8 做严格 A/B
```

## 13. 最终建议

对当前 VideoEdit 14B checkpoint 和 L20，最稳妥的工程顺序是：

1. 用现成的 online `fp8_dynamic` 验证 L20 真实加速；
2. 实现并导出 offline per-channel FP8 weight；
3. 保持 activation dynamic per-token；
4. 用 sensitivity 驱动 mixed precision，而不是一开始全模型硬量化；
5. static activation 只在 profiler 证明有价值时校准；
6. INT8 只在 FP8 稳定后作为性能对照接入。

这条路线最先解决用户真正需要的两件事：**L20 上真实 W8A8 kernel 加速**和**可独立分发、快速加载的离线量化 transformer**，同时把静态 activation 和 INT8 的额外复杂度放在有数据支持的阶段。
