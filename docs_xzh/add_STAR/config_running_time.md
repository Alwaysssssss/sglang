# STAR FP8 路径运行配置与耗时总结

本文档固定记录当前**可复现、已跑通**的 STAR 单卡 `FP8` 路径配置，目标是方便后续分析：

1. 这条路到底用了哪些 `SGLang` 底层加速
2. 哪些加速能力虽然已接入但当前**没有启用**
3. 哪些地方做了 `offload`
4. 还有哪些地方值得继续深挖优化

---

## 1. 当前记录对象

这里记录的是当前已经实际跑通的本地模型目录 `FP8` 命令：

- 模型目录：`/sgl-workspace/sglang/model_artifacts/sglang_star_cogvideox_sr`
- FP8 transformer 目录：
  `/sgl-workspace/sglang/model_artifacts/sglang_star_cogvideox_sr/transformer-fp8-block128`
- 运行产物目录：
  `/sgl-workspace/sglang/outputs/star_fp8_full_localmodel_v1`
- summary：
  [summary.json](/sgl-workspace/sglang/outputs/star_fp8_full_localmodel_v1/summary.json:1)

对应命令：

```bash
python -m sglang.multimodal_gen.test.manual.profile_star_cogvideox_sr \
  --model-path /sgl-workspace/sglang/model_artifacts/sglang_star_cogvideox_sr \
  --transformer-weights-path /sgl-workspace/sglang/model_artifacts/sglang_star_cogvideox_sr/transformer-fp8-block128 \
  --condition-video-path /sgl-workspace/STAR_mg/input/cogvideox_test/lq/023_klingai_reedit.mp4 \
  --prompt-path /sgl-workspace/STAR_mg/input/cogvideox_test/text/023_klingai_reedit.txt \
  --reference-video /sgl-workspace/STAR_mg/cogvideox-based/sat/output/ref_seed1234/0_A_serene_scene_of_a_panda_bear_playing_a_guitar_at_sunset_unfolds_by_a_tranquil_lake._The_panda,_with_its_black-and-whit/000000.mp4 \
  --output-dir /sgl-workspace/sglang/outputs/star_fp8_full_localmodel_v1 \
  --seed 1234 \
  --num-frames 7 \
  --height 480 \
  --width 720 \
  --fps 8 \
  --num-inference-steps 50 \
  --guidance-scale 6.0 \
  --condition-video-num-frames 25 \
  --attention-backend fa \
  --num-gpus 1 \
  --enable-torch-compile \
  --condition-video-vae-peak-memory-mode text_encoder_and_transformer \
  --keep-transformer-gpu-resident-between-requests \
  --warmup-runs 1 \
  --measured-runs 1 \
  --original-star-warm-e2e-s 228.034
```

---

## 2. 这次实测结果

来自 [summary.json](/sgl-workspace/sglang/outputs/star_fp8_full_localmodel_v1/summary.json:1)：

- `avg_wall_clock_s = 162.1279`
- `avg_internal_total_s = 160.7381`
- `warm_e2e_speedup = 1.4065x`
- baseline parity：通过
- `ssim_mean = 0.9353`
- `mse_mean = 32.4037`
- `mae_mean = 3.0599`
- strict parity：未通过

这条路在当前台账里的位置：

- 属于单卡量化路径
- 比当前本地 exact 复现实验 `1.2532x` 更快
- 仍然低于历史单卡 exact 最佳 `1.4314x`
- 也低于全局最快双卡 `1.8628x`

---

## 3. 这条 FP8 路径实际启用了什么底层加速

下面只列**当前这条命令真实启用**的能力，不列“代码里有但这次没开”的能力。

### 3.1 量化

- `FP8` transformer 权重
- 加载目录：
  `/sgl-workspace/sglang/model_artifacts/sglang_star_cogvideox_sr/transformer-fp8-block128`
- summary 中可见：
  `quantization.status = enabled`
  `transformer_weights_path = .../transformer-fp8-block128`

这一条的意义是：

- attention / MLP / time_embed / AdaLN / final layer / text projection 这些线性层热点能进入 `SGLang` 的量化 linear 路径
- 当前 phase7 里已经把单卡 `ReplicatedLinear` 热路径的 `quant_config` 补齐了，所以这次不是“假 FP8”

### 3.2 Attention backend

- `attention_backend = fa`
- 单卡 attention 走的是 `SGLang LocalAttention`
- backend 选择的是 `FlashAttention`

这意味着：

- 不是原生 PyTorch eager attention
- 不是 `torch_sdpa`
- 是走了 `SGLang` attention abstraction + `FA` backend

### 3.3 `torch.compile`

- `enable_torch_compile = true`

这意味着：

- DiT 主计算图是 compile 路线
- 当前这条路径的核心收益仍然主要来自 `compile + FA + FP8`

### 3.4 Fused norm / modulation

当前 STAR transformer 内部仍然会使用 phase7 接入的 fused 路径：

- fused `LayerNorm + scale/shift`
- fused `residual + norm + scale/shift`

也就是说，虽然这条命令没有显式传“fusedln 开关”，但当前代码里的 STAR 热路径本身已经带着这层优化。

### 3.5 Transformer resident 策略

- `keep_transformer_gpu_resident_between_requests = true`

这意味着：

- warmup 后 transformer 默认尽量继续留在 GPU
- 避免每个请求都完整重新装载 transformer

这也是这条 `FP8` 路径能稳定进入热态的重要前提之一。

### 3.6 Condition-video VAE encode 前的峰值显存治理

- `condition_video_vae_peak_memory_mode = text_encoder_and_transformer`

这不是“提升单步算子吞吐”的优化，而是为了让整条单卡 `FP8 + compile` 路径能稳定跑完 measured request。

这条策略的实际行为是：

- 在 `condition video VAE encode` 之前，临时把 `text encoder` 和 `transformer` 从 GPU 挪开
- 降低 VAE encode 的峰值显存
- 避免 warmup 成功但 measured request 在 VAE encode 阶段 OOM

---

## 4. 这条 FP8 路径没有启用什么底层加速

下面这些能力虽然在 `SGLang` / STAR phase7 代码里已经接入或至少接了入口，但**当前这条命令没有用上**。

### 4.1 并行类能力

- 没有启用 `CFG parallel`
- 没有启用 `TP`
- 没有启用 `SP / Ulysses / Ring`
- 没有启用双卡路径

所以这条 `FP8` 路径是纯单卡路径。

### 4.2 Cache 类能力

- 没有启用 `TeaCache`
- 没有启用 `cache-dit`

当前 summary 里虽然显示 `teacache/cache_dit = integrated`，但那表示“代码接入了”，不是“这次实际开了”。

### 4.3 Attention / RoPE 其他 backend

- 没有启用 `FlashInfer RoPE`
- 没有启用 `SAGE_ATTN`
- 没有启用 `SAGE_ATTN_3`
- 没有启用 `AITER`

当前实际使用的仍然只是：

- `FA`

### 4.4 Local enhancer 新路径

- 没有启用 `local_enhancer_mode = fused_5d`
- 当前使用的是 `local_enhancer_mode = legacy`

### 4.5 其他量化路线

- 没有启用 `Nunchaku SVDQuant`
- 没有启用 `AWQ`
- 没有启用混合量化矩阵实验路线

当前真正启用的是：

- `FP8 block128` transformer

### 4.6 常规 server 级 offload

这条命令里明确没有打开：

- `dit_cpu_offload = false`
- `text_encoder_cpu_offload = false`
- `vae_cpu_offload = false`

也就是说，这条 `FP8` 路线不是常规“常驻 CPU offload”方案。

---

## 5. 这条 FP8 路径到底在哪里做了 offload

这部分最容易混淆，所以单独拆开。

### 5.1 没有做的 offload

从 `server` 配置看，没有启用常规长期 offload：

- `DiT CPU offload`：关
- `text encoder CPU offload`：关
- `VAE CPU offload`：关

### 5.2 实际做了的 offload

虽然上面三个长期 offload 都关了，但这条路还是有一个**阶段性、临时性的 offload**：

- 在 `condition video VAE encode` 前
- 按 `text_encoder_and_transformer` 模式
- 临时把 `text encoder` 和 `transformer` 挪到 CPU

这属于：

- `peak-memory management`
- 不是常规 steady-state offload
- 目的是保住 measured request 不 OOM

### 5.3 offload 之后如何继续跑

这条路径又配了：

- `keep_transformer_gpu_resident_between_requests = true`

因此它不是“所有阶段都一直 offload”。

更准确地说：

1. normal path 下尽量保持 transformer resident
2. 但在 `condition video VAE encode` 这个显存最危险的点，临时把 text encoder 和 transformer 挪开
3. 后续再回到正常 denoise 热路径

所以这条 `FP8` 方案本质是：

- `resident fast path`
- 外加一个 `pre-VAE temporary offload guard`

---

## 6. 为什么这条 FP8 路径能到 1.40x 左右

按当前理解，这条路的主要收益来源排序大致是：

1. `FP8` 降低 transformer 线性层开销
2. `FlashAttention` 降低 attention 开销
3. `torch.compile` 优化 denoise 主图
4. phase7 fused norm/modulation 热路径减少一部分额外开销

而它之所以还能稳定完成整条请求，不在 VAE encode 阶段炸掉，主要靠的是：

5. `condition_video_vae_peak_memory_mode = text_encoder_and_transformer`

---

## 7. 从这条配置看，后面还值得挖哪里

如果后面要继续分析还能不能提速，优先看下面几类点。

### 7.1 当前已经确定是热点的部分

- `DenoisingStage`
- `TextEncodingStage`
- `STARConditionVideoVAEEncodingStage`
- `STARCogVideoXSRDecodingStage`

这条 `FP8` 路线里，真正的大头仍然是 denoise。

### 7.2 还没吃到的潜在加速空间

- `TeaCache`
- `cache-dit`
- `FlashInfer RoPE`
- `local enhancer fused_5d`
- `SAGE / AITER`
- 更深层的 conv / local-enhancer kernel 化

### 7.3 需要特别谨慎的点

- `peak-memory offload guard` 虽然保稳定，但也可能带来额外 wall-clock 开销
- transformer resident / temporary offload 之间可能还有切换成本
- 当前单卡 `FP8` 路线虽然 baseline 过了，但 strict parity 没过

---

## 8. 一句话结论

当前这条可复现 `FP8` 单卡路径，本质上是：

- `FP8 + FlashAttention + torch.compile + fused norm/modulation`
- 再加上
- `condition-video VAE encode` 前的临时 `text_encoder + transformer` 显存卸载保护

它**没有**启用：

- 双卡并行
- `CFG parallel`
- `TeaCache`
- `cache-dit`
- `FlashInfer RoPE`
- `fused_5d local enhancer`
- `SAGE / AITER`

所以后续如果还要继续做加速分析，这些“未启用但已接入/可实验”的点，就是最值得优先复盘的对象。***
