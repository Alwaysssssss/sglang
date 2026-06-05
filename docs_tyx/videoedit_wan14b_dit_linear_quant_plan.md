# VideoEdit Wan 14B DiT Linear 量化修改计划

## 目标

在 VideoEdit 推理链路中为 `WanVideoEditTransformer3DModel` 的 DiT transformer Linear 接入现有 diffusion 量化能力，目标模型按 Wan 14B 结构执行，不按 Wan 1.3B 结构改形状。

14B 结构以当前仓库配置为准：

- `num_layers=40`
- `num_attention_heads=40`
- `attention_head_dim=128`
- `hidden_size=5120`
- `ffn_dim=13824`
- VideoEdit 输入输出：`in_channels=36`，`out_channels=16`
- VideoEdit 图像条件：`image_dim=1280`，`added_kv_proj_dim=5120`

本计划只量化 transformer DiT 内的 Linear，不量化 VAE、text encoder、scheduler、norm、rotary、patch embedding 的 `Conv3d`。

## 当前代码判断

已有链路：

- `TransformerLoader` 会解析 transformer 目录或 `--transformer-weights-path` 的量化信息，并把 `quant_config` 传给模型构造函数。
- `maybe_load_fsdp_model()` 加载权重后会遍历模块，调用 `quant_method.process_weights_after_loading()`。
- `WanTransformerBlock` 内 self-attention 的 `to_q/to_k/to_v/to_out`、cross-attention、block FFN、`proj_out` 已经使用 `ColumnParallelLinear` / `RowParallelLinear` 并传入 `quant_config`。

主要缺口：

- `WanTimeTextImageEmbedding` 没有接收和下传 `quant_config`，因此 time/text/image 条件投影仍走非量化 Linear 路径。
- `TimestepEmbedder` 和 `ModulateProjection` 没有把 `quant_config` 传到内部 `MLP` / `ColumnParallelLinear`。
- `WanImageEmbedding` 内部 `MLP` 没有量化配置。
- `WanVideoEditTransformer3DModel` 替换每个 block 的 `attn2` 时没有传入 per-block `prefix`，量化跳过列表和量化 metadata 的层名匹配不够精确。

## 最小修改范围

只改以下文件：

- `python/sglang/multimodal_gen/runtime/layers/visual_embedding.py`
- `python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py`
- `python/sglang/multimodal_gen/runtime/models/dits/wan_videoedit.py`
- 新增或补充一个小型 unit test 文件，建议放在 `python/sglang/multimodal_gen/test/unit/`

不新增 serve 参数，不改 loader，不改权重加载器，不改 VAE/text encoder。

## Phase 1：补齐 14B Wan VideoEdit 的 Linear 量化透传

代码要求：

1. 在 `TimestepEmbedder.__init__()` 增加可选 `quant_config=None`，并传给内部 `MLP`。
2. 在 `ModulateProjection.__init__()` 增加可选 `quant_config=None`，并传给内部 `ColumnParallelLinear`。
3. 在 `WanImageEmbedding.__init__()` 增加可选 `prefix=""`、`quant_config=None`，并传给内部 `MLP`。
4. 在 `WanTimeTextImageEmbedding.__init__()` 增加可选 `prefix=""`、`quant_config=None`，并传给：
   - `time_embedder`
   - `time_modulation`
   - `text_embedder`
   - `image_embedder`
5. 在 `WanTransformer3DModel.__init__()` 创建 `condition_embedder` 时传入 `prefix="condition_embedder"` 和 `quant_config=quant_config`。
6. 在 `WanVideoEditTransformer3DModel.__init__()` 替换 `block.attn2` 时改为枚举 block index，并传入 `prefix=f"blocks.{i}.attn2"` 和 `quant_config=quant_config`。

进入下一阶段的要求：

- `quant_config=None` 时模型结构和参数名不变，bf16 主线加载和推理不受影响。
- `quant_config` 非空时，以下模块都能拿到量化方法：
  - `condition_embedder.time_embedder.mlp.fc_in/fc_out`
  - `condition_embedder.time_modulation.linear`
  - `condition_embedder.text_embedder.fc_in/fc_out`
  - `condition_embedder.image_embedder.ff.fc_in/fc_out`
  - `blocks.*.attn1.*`
  - `blocks.*.attn2.*`
  - `blocks.*.ffn.*`
  - `proj_out`
- 不出现新的 missing/unexpected key；原有 diffusers 到 SGLang 参数名映射仍然生效。

## Phase 2：权重加载和量化格式验证

优先验证现有 diffusion 量化实现已经支持的格式：

1. FP8 dynamic / FP8 serialized。
2. ModelSlim，如果量化 checkpoint 带 `quant_model_description.json`。
3. ModelOpt FP4 只作为后续分支，不作为第一阶段验收目标。

验证要求：

- 使用 14B VideoEdit transformer 权重路径，不使用 1.3B 配置或 1.3B checkpoint。
- 启动日志必须能看到 transformer quant config 被解析，且 `transformer_weights_path` 指向量化 transformer。
- 模型加载后检查所有目标 Linear 的 `quant_method` 类型，不允许条件投影仍然落回 `UnquantizedLinearMethod`，除非该层明确在量化配置里被跳过。
- 单窗口 81 帧推理能完成，无 NaN/Inf，无 shape mismatch。

建议先跑最小 smoke：

```bash
export MODEL_PATH=/path/to/VideoEdit-diffusers-model
export QUANT_TRANSFORMER_PATH=/path/to/wan14b-videoedit-quant-transformer

python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$MODEL_PATH" \
  --transformer-weights-path "$QUANT_TRANSFORMER_PATH" \
  --prompt "repair the video" \
  --video-input-path /path/to/input.mp4 \
  --mask-input-path /path/to/mask.mp4 \
  --output-path outputs \
  --output-file-name videoedit_wan14b_quant_smoke.mp4 \
  --num-frames 81 \
  --infer-len 81 \
  --height 720 \
  --width 1280 \
  --no-dit-cpu-offload \
  --no-dit-layerwise-offload
```

进入下一阶段的要求：

- 14B 量化 transformer 可加载。
- 81 帧 smoke 推理完成。
- 显存峰值低于 bf16 transformer baseline。
- 输出视频可解码，基础质量无明显崩坏。

## Phase 3：质量和性能评估

质量评估要求：

- 固定同一输入视频、mask、prompt、seed、窗口参数。
- 和 bf16 主线输出对比。
- 重点看 mask 边缘、身份一致性、纹理稳定性、窗口拼接处闪烁。
- 记录 compare JSON、perf JSON、serve/CLI 日志。

性能评估要求：

- 单卡 14B：先测无 offload。
- 若 OOM，再单独开启 `dit_layerwise_offload`，不要和 Cache-DiT 同时启用。
- SP2/TP2、Cache-DiT、attention backend 组合必须单独重测，不继承单卡结论。

进入下一阶段的要求：

- 量化分支比 bf16 baseline 有明确显存收益。
- 端到端耗时没有明显回退；如果回退，需要定位是量化 kernel、offload 还是 attention backend。
- 质量指标和人工检查都达到可接受范围。

## Phase 4：可选扩展

只有 Phase 1-3 达标后再考虑：

- 更低比特格式，例如 ModelOpt FP4。
- 更细的 skip list，例如保留 `condition_embedder` 或 `proj_out` 为 bf16。
- 量化与 Cache-DiT 的组合。
- 量化与 SP/TP 的组合。

这些扩展必须独立记录结果，不能覆盖 bf16 主线和 FP8 主线结论。

## 风险点

- 条件投影层量化可能比 block FFN/attention 更影响质量。如果 Phase 2 出现明显质量问题，优先尝试只量化 blocks 和 `proj_out`，把 `condition_embedder.*` 加入 skip list。
- `WanVideoEditTransformer3DModel` 替换 `attn2` 后必须保留 per-block prefix，否则量化配置中的层名跳过或 metadata 匹配可能不稳定。
- 14B 的 `hidden_size=5120` 对 TP 切分和 FP8 block size 有整除约束，若使用 block-wise quant，要先确认 checkpoint block shape 和 TP size 兼容。
- 量化和 layerwise offload 都会影响权重驻留方式，二者组合需要单独验证加载后处理和 H2D prefetch。

## 推荐实现顺序

1. 只做 Phase 1 的 `quant_config` 透传和 prefix 修正。
2. 加 unit test，确认 `quant_config=None` 与 `quant_config=Fp8Config()` 两种构造路径。
3. 用 14B bf16 transformer 跑一次回归 smoke。
4. 用 14B 量化 transformer 跑 Phase 2 smoke。
5. 再进入 Phase 3 的质量和性能评估。
