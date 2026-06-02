# VideoEdit 接入 Cache-DiT 的修改方案

本文目标是给 SGLang 原生 VideoEdit pipeline 增加 Cache-DiT 加速能力。第一阶段不重新设计缓存算法，先复用 cache-dit 里 Wan 1.3B / Wan 2.1 单 transformer 的版式：`blocks` 作为缓存 block 列表，`ForwardPattern.Pattern_2`，并开启 CFG 正负分支隔离。

## 1. 先看结论

第一阶段建议走最小闭环：

1. 只支持 SGLang 原生 `WanVideoEditPipeline`，不走 Diffusers backend。
2. 复用 cache-dit 已注册的 `Wan` adapter。`WanVideoEditTransformer3DModel` 类名以 `Wan` 开头，且继承 `WanTransformer3DModel`，block 列表也是 `self.blocks`，因此可以先按 Wan 1.3B / Wan 2.1 单 transformer 路径接入。
3. 通过环境变量 `SGLANG_CACHE_DIT_*` 控制，不把 `--cache-dit-config` 作为第一阶段入口。当前 `--cache-dit-config` 主要服务 Diffusers pipeline。
4. Cache-DiT 与 `dit_layerwise_offload` 硬互斥，必须显式使用 `--dit-layerwise-offload false`。`dit_cpu_offload` 与 `torch.compile` 不是天然互斥项，只是第一阶段排障时建议先关闭，等 Cache-DiT 正确性确认后再逐项打开。
5. 重点补齐显式日志、CFG 分支状态顺序、验证脚本和测试，避免“看起来启用了但实际 fallback / 复用错分支”。

## 2. 当前状态

### 2.1 VideoEdit 已经有基础挂载点

`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py`

`VideoEditDenoisingStage.forward()` 里已经在 denoising loop 前调用：

```python
self._manage_device_placement(self.transformer, None, server_args)
self._maybe_enable_cache_dit(params.runtime_effective_num_inference_steps, batch)
```

这说明 VideoEdit 已经复用了 `DenoisingStage._maybe_enable_cache_dit()`，不需要重新写一套 cache 生命周期。

### 2.2 通用 Cache-DiT 路径

`python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`

`_maybe_enable_cache_dit()` 会：

- 读取 `SGLANG_CACHE_DIT_ENABLED`；
- 从环境变量构造 `CacheDitConfig`；
- 生成 SCM mask；
- 单 transformer 调 `enable_cache_on_transformer()`；
- 已启用后，每个新请求调用 `refresh_context_on_transformer()`。

`python/sglang/multimodal_gen/runtime/cache/cache_dit_integration.py`

`enable_cache_on_transformer()` 目前先要求：

```python
BlockAdapterRegister.is_supported(transformer)
```

然后调用：

```python
cache_dit.enable_cache(
    transformer,
    cache_config=cache_config,
    calibrator_config=calibrator_config,
    parallelism_config=None,
)
```

### 2.3 WanVideoEdit 可以先复用 Wan adapter

`python/sglang/multimodal_gen/runtime/models/dits/wan_videoedit.py`

`WanVideoEditTransformer3DModel` 继承 `WanTransformer3DModel`，只是替换了 cross-attention：

```python
class WanVideoEditTransformer3DModel(WanTransformer3DModel):
    ...
    for block in self.blocks:
        block.attn2 = WanVideoEditCrossAttention(...)
```

cache-dit 1.3.0 的 `Wan` adapter 逻辑是：

```python
BlockAdapter(
    transformer=pipe.transformer,
    blocks=pipe.transformer.blocks,
    forward_pattern=ForwardPattern.Pattern_2,
    check_forward_pattern=True,
    has_separate_cfg=True,
)
```

`BlockAdapterRegister.is_supported()` 按 class name prefix 判断，`WanVideoEditTransformer3DModel` 会命中 `Wan`。cache-dit 的 `_relaxed_assert()` 只对 diffusers 模块做严格类型检查，SGLang 自定义模块不会被 diffusers 类型断言挡住。

结论：第一阶段不需要手写 block forward，先复用 Wan 单 transformer adapter。

## 3. 目标行为

启用方式：

```bash
export SGLANG_CACHE_DIT_ENABLED=true
export SGLANG_CACHE_DIT_FN=1
export SGLANG_CACHE_DIT_BN=0
export SGLANG_CACHE_DIT_WARMUP=2
export SGLANG_CACHE_DIT_RDT=0.24
export SGLANG_CACHE_DIT_MC=3
export SGLANG_CACHE_DIT_SCM_PRESET=fast
```

启动 VideoEdit 时：

```bash
--dit-layerwise-offload false
# 可选排障项，不是 Cache-DiT 硬要求：
# --dit-cpu-offload false
# --enable-torch-compile false
```

期望日志：

```text
cache-dit enabled on transformer
Enabling cache-dit on transformer with config: Fn=..., Bn=..., W=..., R=..., MC=...
SCM enabled: ...
```

如果显式加 VideoEdit 日志，期望看到：

```text
VideoEdit cache-dit uses Wan adapter: blocks=..., pattern=Pattern_2, separate_cfg=True
```

## 4. 修改点

### 4.1 显式标记 VideoEdit 使用 Wan adapter

修改：

```text
python/sglang/multimodal_gen/runtime/cache/cache_dit_integration.py
```

建议新增小 helper：

```python
def _is_wan_videoedit_transformer(transformer: torch.nn.Module) -> bool:
    return transformer.__class__.__name__ == "WanVideoEditTransformer3DModel"
```

在 `enable_cache_on_transformer()` 里，`BlockAdapterRegister.is_supported(transformer)` 通过后增加明确日志：

```python
if _is_wan_videoedit_transformer(transformer):
    blocks = getattr(transformer, "blocks", None)
    logger.info(
        "VideoEdit cache-dit uses Wan adapter: blocks=%s, pattern=Pattern_2, separate_cfg=True",
        len(blocks) if blocks is not None else "missing",
    )
```

这样做的目的不是改变逻辑，而是把“VideoEdit 命中了 Wan adapter”从隐式 class prefix 行为变成可观测行为。

### 4.2 增加受控 fallback adapter

同一个文件中预留 fallback。只有在后续 cache-dit 版本修改 `BlockAdapterRegister` 匹配规则，或 `WanVideoEditTransformer3DModel` 改名导致不再命中 `Wan` 时才启用。

伪代码：

```python
if not BlockAdapterRegister.is_supported(transformer):
    if _is_wan_videoedit_transformer(transformer):
        return enable_cache_on_wan_videoedit_transformer(
            transformer,
            config,
            model_name=model_name,
            sp_group=sp_group,
            tp_group=tp_group,
        )
    raise ValueError(...)
```

fallback 的 BlockAdapter 必须复用 Wan 单 transformer 设置：

```python
cache_dit.enable_cache(
    BlockAdapter(
        transformer=transformer,
        blocks=transformer.blocks,
        forward_pattern=ForwardPattern.Pattern_2,
        params_modifiers=ParamsModifier(
            cache_config=cache_config,
            calibrator_config=calibrator_config,
        ),
        check_forward_pattern=True,
        has_separate_cfg=True,
    ),
    parallelism_config=None,
)
```

注意这里使用 `params_modifiers`，不是 `params_modifier`。当前 cache-dit 1.3.0 的 `BlockAdapter` 构造参数名是复数。

### 4.3 调整 CFG 状态设置顺序

修改：

```text
python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py
```

当前顺序是：

```python
self._maybe_enable_cache_dit(...)
batch.do_classifier_free_guidance = bool(params.runtime_do_cfg)
batch.is_cfg_negative = False
```

建议改成：

```python
batch.do_classifier_free_guidance = bool(params.runtime_do_cfg)
batch.is_cfg_negative = False
self._maybe_enable_cache_dit(params.runtime_effective_num_inference_steps, batch)
```

原因：Wan adapter `has_separate_cfg=True`，VideoEdit 的正负 CFG 分支靠 `batch.is_cfg_negative` 区分。虽然实际 forward 前也会设置 `is_cfg_negative`，但在挂载/刷新 cache context 之前先把请求级 CFG 状态放好，更稳，也更容易排查。

denoising loop 内现有逻辑保留：

```python
batch.is_cfg_negative = False
noise_pred = self.transformer(...)

batch.is_cfg_negative = True
noise_uncond = self.transformer(...)
batch.is_cfg_negative = False
```

### 4.4 明确 native VideoEdit 不使用 `--cache-dit-config`

修改或补充文档：

```text
docs_tyx/videoedit_single_gpu_speedup_runbook.md
docs_tyx/videoedit_serve_user_guide.md
```

说明：

- 原生 VideoEdit 第一阶段使用 `SGLANG_CACHE_DIT_*` 环境变量；
- `--cache-dit-config` 保留给 Diffusers backend；
- 如果后续要做 YAML/JSON 配置，再单独把 native path 的 `CacheDitConfig` 从 env 扩展为 `server_args.cache_dit_config`。

### 4.5 offload 约束

已有校验：

```text
python/sglang/multimodal_gen/runtime/server_args.py
```

当 `dit_layerwise_offload` 与 `SGLANG_CACHE_DIT_ENABLED=true` 同时开启时会直接抛错。保留这个行为。

第一阶段建议运行组合：

```text
Cache-DiT + no DiT offload + text/image/VAE offload 可按需开关
```

不支持：

```text
Cache-DiT + dit_layerwise_offload
```

可验证但建议放到第二轮：

```text
Cache-DiT + dit_cpu_offload
Cache-DiT + torch.compile
```

`dit_cpu_offload` 没有硬性互斥。它主要负责整模 DiT CPU/GPU 迁移或 FSDP CPU offload，不会像 layerwise offload 那样把 block 权重释放成 placeholder。为了先定位 Cache-DiT 正确性，第一轮可以关闭；等单卡或双卡 no-offload 跑通后，再单独验证。

### 4.6 torch.compile 顺序

Cache-DiT 与 `torch.compile` 的核心约束是顺序：必须先挂 Cache-DiT，再 compile。

当前 `DenoisingStage.__init__()` 会在初始化时尝试 `_maybe_enable_torch_compile(transformer)`，而 VideoEdit 是在 `forward()` 中 `_maybe_enable_cache_dit()`。因此在现有 VideoEdit 代码结构下，`--enable-torch-compile true` 可能导致 compile 先于 Cache-DiT patch。第一阶段为了避免混淆，建议显式关闭：

```bash
--enable-torch-compile false
```

如果后续必须支持 `Cache-DiT + torch.compile`，再改结构：

1. VideoEdit denoising stage 初始化时不要立即 compile transformer；
2. 第一次真实请求进入 denoising 前先 `_maybe_enable_cache_dit()`；
3. cache 挂载成功后再 compile；
4. warmup request 不启用 Cache-DiT，但可以触发 compile 的策略需要单独设计，避免 compile 捕获未 patch 的 forward。

## 5. 推荐落地顺序

### 阶段 0：确认现状

不改代码，直接用当前环境跑一次 81 帧单窗口：

```bash
export SGLANG_CACHE_DIT_ENABLED=true
export SGLANG_CACHE_DIT_FN=1
export SGLANG_CACHE_DIT_BN=0
export SGLANG_CACHE_DIT_WARMUP=2
export SGLANG_CACHE_DIT_RDT=0.24
export SGLANG_CACHE_DIT_MC=3
export SGLANG_CACHE_DIT_SCM_PRESET=fast
```

启动时明确：

```bash
--dit-layerwise-offload false
# 可选排障项，不是 Cache-DiT 硬要求：
# --dit-cpu-offload false
# --enable-torch-compile false
```

如果日志已经出现 `cache-dit enabled on transformer`，说明当前 class prefix 已经命中 Wan adapter。后续改动以日志、CFG 顺序和测试为主。

如果报：

```text
WanVideoEditTransformer3DModel is not officially supported by cache-dit
```

再做 4.2 的 fallback adapter。

### 阶段 1：显式化支持

实现：

1. 加 `_is_wan_videoedit_transformer()`；
2. 在 `enable_cache_on_transformer()` 里输出 VideoEdit 专用日志；
3. 调整 `VideoEditDenoisingStage.forward()` 中 CFG 状态设置顺序；
4. 保持 `dit_layerwise_offload` 互斥校验不变。

### 阶段 2：小输入正确性

同一组输入、同一 seed，跑：

1. baseline：`SGLANG_CACHE_DIT_ENABLED=false`；
2. conservative：`FN=1, BN=1, WARMUP=4, RDT=0.12, MC=2, SCM_PRESET=medium`；
3. fast：`FN=1, BN=0, WARMUP=2, RDT=0.24, MC=3, SCM_PRESET=fast`。

检查：

- 输出视频帧数一致；
- 无 shape mismatch；
- 正负 CFG 分支都正常执行；
- 画质没有明显局部闪烁或 mask 区域错修；
- perf dump 中 denoising 时间下降。

### 阶段 3：全帧窗口验证

跑当前全帧场景，例如 156 帧、`infer_len=81`、`overlap=0` 或已有 overlap 配置。

重点确认：

- 每个窗口进入 denoising 前都会 refresh cache context；
- 最后一个反射/补齐窗口 shape 仍为 81 帧，不触发动态 shape 问题；
- `use_repaired_context=True` 时，不会复用上一个窗口的旧隐藏状态。

## 6. 验证命令模板

### 6.1 Serve 启动

下面模板把 `dit_cpu_offload` 和 `enable_torch_compile` 也关掉，是为了第一轮排障变量最少；硬要求只有 `--dit-layerwise-offload false`。

```bash
cd /home/tyx/workspace/zhouhao6/sglang
source .venv/bin/activate

export MODEL_PATH=/home/tyx/workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model
export TRANSFORMER_PATH=/home/tyx/workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer
export OUT_DIR=/home/tyx/workspace/zhouhao6/sglang/output_tyx
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_ALLOC_CONF=expandable_segments:True

export SGLANG_CACHE_DIT_ENABLED=true
export SGLANG_CACHE_DIT_FN=1
export SGLANG_CACHE_DIT_BN=0
export SGLANG_CACHE_DIT_WARMUP=2
export SGLANG_CACHE_DIT_RDT=0.24
export SGLANG_CACHE_DIT_MC=3
export SGLANG_CACHE_DIT_SCM_PRESET=fast

sglang serve \
  --model-type diffusion \
  --model-path "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 30000 \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 2 \
  --ring-degree 1 \
  --dit-layerwise-offload false \
  --dit-cpu-offload false \
  --text-encoder-cpu-offload false \
  --image-encoder-cpu-offload false \
  --vae-cpu-offload false \
  --enable-torch-compile false \
  --attention-backend fa \
  --warmup true \
  --warmup-steps 1 \
  --output-path "$OUT_DIR" \
  --input-save-path /tmp/sglang-videoedit-inputs \
  --transformer-path "$TRANSFORMER_PATH"
```

### 6.2 日志检查

```bash
grep -i "cache-dit" serve.log
grep -i "VideoEdit cache-dit uses Wan adapter" serve.log
grep -i "fallback\\|unsupported\\|shape mismatch" serve.log
```

有效启用至少要看到：

```text
cache-dit enabled on transformer
```

如果只看到环境变量，但没有 enable 日志，本轮不能算有效 Cache-DiT 测试。

## 7. 测试计划

### 7.1 单元级

建议增加或扩展测试：

```text
python/sglang/multimodal_gen/test/
```

覆盖点：

1. 构造 class name 为 `WanVideoEditTransformer3DModel` 的轻量 fake module，确认 `BlockAdapterRegister.is_supported(fake)` 为 True。
2. `_is_wan_videoedit_transformer()` 只对 VideoEdit transformer 返回 True。
3. `VideoEditDenoisingStage.forward()` 在 `_maybe_enable_cache_dit()` 前设置 `batch.do_classifier_free_guidance` 和 `batch.is_cfg_negative`。
4. 启用 `SGLANG_CACHE_DIT_ENABLED=true` 且 `dit_layerwise_offload=True` 时仍然抛错。

### 7.2 冒烟测试

最小输入：

- `num_frames=81`；
- `infer_len=81`；
- `num_inference_steps=20`；
- `seed` 固定；
- `overlap=0`。

对比：

1. baseline no-cache；
2. Cache-DiT conservative；
3. Cache-DiT fast。

### 7.3 性能与质量记录

记录：

- 总耗时；
- denoising 阶段耗时；
- 峰值显存；
- 输出帧数；
- SSIM/LPIPS 或人工观察结论；
- mask 区域边界是否有明显闪烁。

## 8. 风险和处理

### 8.1 adapter 隐式命中风险

当前依赖 class name prefix 命中 `Wan`。如果后续类名不再以 `Wan` 开头，或 cache-dit 改了匹配策略，会失效。

处理：保留 4.2 的 fallback adapter，并用日志检查实际命中的路径。

### 8.2 CFG 缓存串分支

VideoEdit 同一步会先跑 positive，再跑 negative。必须使用 `has_separate_cfg=True`，并保证 `batch.is_cfg_negative` 在每次 transformer forward 前正确切换。

处理：复用 Wan adapter；调整 CFG 状态设置顺序；新增测试。

### 8.3 layerwise offload 冲突

Cache-DiT 会跳过或复用 block，layerwise offload 依赖每层严格执行 pre-hook / post-hook 来 prefetch/release 权重，两者同时开可能访问到已经被 release 成 placeholder 的权重。

处理：保持互斥；Cache-DiT 场景显式 `--dit-layerwise-offload false`。

### 8.4 torch.compile 顺序

如果先 compile，再挂 Cache-DiT，patch 可能不会被编译图正确捕获。

处理：第一阶段显式关闭 compile；第二阶段单独改初始化顺序。

### 8.5 多窗口 stale cache

VideoEdit 全帧会拆多个 81 帧窗口。不同窗口的 masked video latents 和 mask 条件不同，不能跨窗口复用旧 context。

处理：每个窗口进入 denoising 前调用 `_maybe_enable_cache_dit()`，已启用时走 `refresh_context_on_transformer()`。验证全帧输出时重点看窗口边界和 mask 区域。

## 9. 完成标准

第一阶段完成条件：

1. `SGLANG_CACHE_DIT_ENABLED=true` 时，VideoEdit serve 能稳定启动；
2. 日志明确显示 VideoEdit 使用 Wan adapter；
3. 81 帧单窗口和 156 帧全帧都能出片；
4. baseline 与 Cache-DiT 输出帧数一致；
5. 没有 `unsupported`、`fallback`、`shape mismatch`、CFG 分支串缓存等问题；
6. denoising 耗时相对 baseline 有可观下降；
7. 文档明确 `dit_layerwise_offload` 和 `torch.compile` 的阶段性限制。

