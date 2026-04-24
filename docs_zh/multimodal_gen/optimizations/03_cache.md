# 03. 缓存（Cache）

> 源码位置：`python/sglang/multimodal_gen/runtime/cache/cache_dit_integration.py`、`runtime/cache/teacache.py`、`runtime/pipelines_core/stages/denoising.py`、`runtime/pipelines/diffusers_pipeline.py`、`configs/sample/teacache.py`、`runtime/models/dits/*`、`envs.py`。

扩散模型的去噪循环里，大量步数相邻的 hidden state 变化很小，因此可以把当前 step 的 transformer 输出/残差缓存下来，给下一步或若干步复用。`multimodal_gen` 实现了**两大类缓存体系**：

- **TeaCache**：全模型级时间步跳过；框架原生 Mixin + 参数化判据。
- **cache-dit**：细粒度块级缓存（Fn/Bn/W/R/MC），底层复用 `cache_dit` 第三方包，上层自己做 **SP/TP 一致性 patch** 与 **Wan2.2 双专家 BlockAdapter**。

此外还有 Diffusers 后端通用路径（`_apply_cache_dit`）与 SCM 步级 mask。

## 1. 方案矩阵

| 方案 | 位置 | 是否跳整块 forward | 控制 | 与 CFG | 与并行 | 与 torch.compile |
|------|------|-------------------|------|--------|--------|-----------------|
| **TeaCache** | `runtime/cache/teacache.py` + 各 DiT 模型 | ✔（需模型实现 `should_skip_forward_for_cached_states`）| `SamplingParams.enable_teacache` + `teacache_params` | 分正/负支路缓存 | 对并行透明 | 可与 compile 共存 |
| **cache-dit**（原生 DiT 路径） | `runtime/cache/cache_dit_integration.py` + `DenoisingStage._maybe_enable_cache_dit` | 由 `cache_dit` 包在 block 级处理 | 环境变量 `SGLANG_CACHE_DIT_*` | 双塔模式 `has_separate_cfg=True` | 需要 `_patch_cache_dit_similarity` 做 all_reduce | **先挂 cache-dit 再 compile** |
| **cache-dit**（Diffusers 路径） | `runtime/pipelines/diffusers_pipeline.py::_apply_cache_dit` | 同上 | `--cache-dit-config` | 由 cache-dit 处理 | — | 自动 `cache_dit.set_compile_configs()` |
| **DBCache / TaylorSeer / SCM** | 作为 cache-dit 的 config 字段 | 块内残差替代 / 阶段性重计算 | `SGLANG_CACHE_DIT_TAYLORSEER`、`SGLANG_CACHE_DIT_SCM_*` | 同 cache-dit | 同上 | 同上 |
| **FB Cache** | 代码库内**无**此字符串；cache-dit 论文里对应 `Fn_compute_blocks` / `Bn_compute_blocks` 字段 | — | — | — | — |

## 2. TeaCache

### 2.1 核心原理

TeaCache 把 modulated timestep 嵌入（Wan 里是 `timestep_proj` 或 `temb`）当作"相似度判据"，相邻 step 若相似就复用上一个 step 的 residual。

判据在 `teacache.py::_compute_l1_and_decide`（约 171–216 行）：

```python
rel_l1 = (diff.abs().mean() / prev_modulated_inp.abs().mean())
```

这是**全局均值比**（不是逐 token 的条件分支，与 cache-dit 的 similarity 逻辑不同）。

### 2.2 多项式 rescale + 累加阈值

`coefficients` 是一个 4 次多项式（`configs/sample/teacache.py` 约 38–40 行，形如 `c[0]*x**4 + ... + c[4]`）：

```python
rescale_func = np.poly1d(coefficients)
accumulated_rel_l1_distance += rescale_func(rel_l1)
```

`teacache_thresh` 是累加器阈值：`accumulated_rel_l1_distance >= thresh` 时**必须计算**并清零；否则认为可走缓存（`should_calc=False`）。

支持 **`coefficients_callback`** 动态切换多项式系数（约 60–63 行），例如按采样器类型自适应。

### 2.3 边界步与 CFG

- `_compute_teacache_decision`（约 237–247 行）：`is_boundary_step` 时**强制计算**；
- `get_skip_boundaries` 在 `do_cfg` 时把 `start/end` 步数乘 2（约 78–80 行）；
- CFG 正/负支路分开缓存：`batch.is_cfg_negative` + `_supports_cfg_cache`（只限前缀 `wan` / `hunyuan` / `zimage`，约 127–153 行）决定使用独立的 `previous_modulated_input{_negative}` / `accumulated_rel_l1_distance{_negative}`；
- Wan 实现里 `maybe_cache_states` / `retrieve_cached_states` 对 `previous_residual{_negative}` 分别处理（`wanvideo.py` 约 1199–1247 行）。

### 2.4 跑通情况

| 模型 | TeaCache 能否真正跳过 forward |
|------|------------------------------|
| Wan / Wan2.2（视频 DiT） | ✔ 完整实现（`wanvideo.py`） |
| HunyuanVideo | ✘ `enable_teacache` 时 `raise NotImplementedError`（`hunyuanvideo.py` 约 693–695 行） |
| ZImage / 其它 CachableDiT | 多数未覆盖 `should_skip_forward_for_cached_states`，默认 False（即开关打开也不跳） |

### 2.5 用户侧配置

```python
generator.generate(sampling_params_kwargs=dict(
    prompt="...",
    enable_teacache=True,
    teacache_params=TeaCacheParams(
        teacache_thresh=0.1,
        coefficients=[...],
        ret_steps=True,
        skip_start=1,
        skip_end=1,
    ),
))
```

`configs/sample/teacache.py` 里是默认系数集合；`SamplingParams.enable_teacache` / `teacache_params` 位于约 163–167 行。

## 3. cache-dit：原生 DiT 路径

### 3.1 框架集成

入口：`runtime/cache/cache_dit_integration.py::enable_cache_on_transformer`（约 224–336 行）。

- 先校验 `BlockAdapterRegister.is_supported(transformer)`，目前支持 **Flux、QwenImage、HunyuanDiT、HunyuanVideo、Wan** 等（约 252–261 行）；
- 构造 `DBCacheConfig`（含 SCM 字段）与可选 `TaylorSeerCalibratorConfig`；
- 调 `cache_dit.enable_cache(transformer, cache_config=..., calibrator_config=..., parallelism_config=None)`（约 315–320 行）。

**block 级 patch 由 `cache_dit` 包内对已注册架构的 `BlockAdapter` 完成**，仓库不直接写 `forward` hook。

### 3.2 双塔 `enable_cache_on_dual_transformer`（Wan2.2）

`runtime/cache/cache_dit_integration.py` 约 339–524 行：

- 使用 `BlockAdapter`，同时传入两个 transformer 的 `blocks` 列表；
- `forward_pattern=ForwardPattern.Pattern_2`；
- `has_separate_cfg=True`（约 491–504 行）——对应双专家 + CFG 的语义；
- `model_name="wan2.2"`。

### 3.3 SP / TP 下的相似度一致性

多卡下每个 rank 看到的激活不同，`similarity` 若各自独立判断会导致 **不同 rank 的 skip 决策不一致**，从而死锁或结果错乱。`_patch_cache_dit_similarity()`（约 41–102 行）：

- patch `CachedContextManager.similarity`，对 residual 的 `mean diff / mean t1` 做 `dist.all_reduce(AVG)`；
- 在 `context_manager` 上设置 `_sglang_sp_group` / `_sglang_tp_group` / `_sglang_tp_sp_group`；
- TP + SP 混用时选择 `get_dit_group()`（即包含两个维度 rank 的大组）。

### 3.4 生命周期：`DenoisingStage._maybe_enable_cache_dit`

`runtime/pipelines_core/stages/denoising.py` 约 246–432 行：

1. 若已启用（`_cache_dit_enabled=True`）：只调用 `refresh_context_on_transformer` 或 `refresh_context_on_dual_transformer`，用 `envs.SGLANG_CACHE_DIT_SCM_PRESET` 等刷新 SCM（约 262–278 行）；
2. 否则需 `envs.SGLANG_CACHE_DIT_ENABLED=True` 且 `not batch.is_warmup`（约 281–283 行）；
3. 解析 sp/tp group（约 285–301 行）；
4. 从环境变量读 DBCache / TaylorSeer / SCM，构造 `CacheDitConfig`；
5. 单塔调 `enable_cache_on_transformer`，双塔调 `enable_cache_on_dual_transformer(..., model_name="wan2.2")`；
6. 置 `_cache_dit_enabled=True`（约 431 行）。

**关键顺序**：首次加载 transformer 时 `_maybe_enable_cache_dit` 必须在 `_maybe_enable_torch_compile` **之前**（约 522–524 行）；否则 cache-dit patch 会被 compile 捕获/破坏。

**LTX2 特例**：`ltx_2_denoising.py` 约 157–164、585 行对 TI2V / 带图条件请求 `_disable_cache_dit_for_request`，跳过 cache-dit 避免陈旧激活。

### 3.5 SCM（Step-wise Cache Mask）

`get_scm_mask`（约 138–178 行）：包装 `cache_dit.steps_mask` 得到 `1=算 / 0=cache` 的步级 mask。

环境变量：
- `SGLANG_CACHE_DIT_SCM_PRESET`
- `SGLANG_CACHE_DIT_SCM_COMPUTE_BINS`
- `SGLANG_CACHE_DIT_SCM_CACHE_BINS`
- `SGLANG_CACHE_DIT_SCM_POLICY`

### 3.6 环境变量一览

`envs.py` 约 259–283 行及 304–343 行：

| 变量 | 作用 |
|------|------|
| `SGLANG_CACHE_DIT_ENABLED` | 总开关 |
| `SGLANG_CACHE_DIT_FN`、`SGLANG_CACHE_DIT_BN` | 前/后计算块数 |
| `SGLANG_CACHE_DIT_WARMUP` | 前几步不缓存 |
| `SGLANG_CACHE_DIT_RDT` | residual diff threshold |
| `SGLANG_CACHE_DIT_MC` | max cache steps |
| `SGLANG_CACHE_DIT_TAYLORSEER` | 启用 TaylorSeer |
| `SGLANG_CACHE_DIT_TS_ORDER` | TaylorSeer 阶数 |
| `SGLANG_CACHE_DIT_SCM_*` | SCM 预设 |
| `SGLANG_CACHE_DIT_SECONDARY_*` | Wan2.2 低噪专家的独立配置 |

**冲突校验**：`dit_layerwise_offload` 与 `SGLANG_CACHE_DIT_ENABLED` 互斥（`server_args.py` 约 1363–1368 行）。

## 4. cache-dit：Diffusers 后端路径

`runtime/pipelines/diffusers_pipeline.py` 的 `_apply_cache_dit`（约 472–475 行前后）对整根 `diffusers pipe` 调 `cache_dit.enable_cache`。配置来自 CLI `--cache-dit-config`（`server_args.py` 约 805–810 行，字段 `cache_dit_config`）。

Diffusers 路径里顺序是 **先 `_apply_cache_dit`，再 `_apply_torch_compile`**（约 472–475 行）；启用 cache-dit 时会尝试 `cache_dit.set_compile_configs()`（约 620–625 行）。

## 5. 与 Forward Context 的联动

`DenoisingStage._predict_noise_with_cfg`（`denoising.py` 约 1599–1630 行）：

```python
batch.is_cfg_negative = False   # cond
...set_forward_context(forward_batch=batch)...
batch.is_cfg_negative = True    # uncond
...set_forward_context(forward_batch=batch)...
```

`TeaCacheMixin._get_teacache_context` 读取同一个 `Req` 对象上的 `is_cfg_negative`，与 cache-dit 的 `has_separate_cfg` 对应同一语义。

此外，因为 cache-dit 会把 transformer `forward` 包成 `functools.partial`，`DenoisingStage.prepare_extra_func_kwargs` 从 `_original_forward` 取真实签名（约 1193–1198 行）。

## 6. 缓存与 `true_cfg_scale` / CFG Zero Star

- `true_cfg_scale`（Qwen-Image 等使用的 guidance distillation 参数）定义在 `SamplingParams`，与缓存**正交**：cache-dit 绑定在 transformer forward 层，不区分 "蒸馏尺度"；
- **CFG Zero Star**（Helios 体系）未与 `runtime/cache` 做专门耦合，使用同一套 `set_forward_context + batch` 路径。

## 7. 关键类/函数速查

| 文件 | 符号 | 作用 |
|------|------|------|
| `runtime/cache/teacache.py` | `TeaCacheMixin`、`TeaCacheContext`、`_compute_l1_and_decide`、`_compute_teacache_decision`、`get_skip_boundaries` | TeaCache 判据与 CFG |
| `runtime/cache/cache_dit_integration.py` | `enable_cache_on_transformer`、`enable_cache_on_dual_transformer`、`_patch_cache_dit_similarity`、`get_scm_mask`、`refresh_context_on_*` | cache-dit 集成 |
| `runtime/pipelines_core/stages/denoising.py` | `DenoisingStage._maybe_enable_cache_dit` | cache-dit 生命周期 |
| `runtime/pipelines/diffusers_pipeline.py` | `_apply_cache_dit` | Diffusers 路径 cache-dit |
| `configs/sample/teacache.py` | `TeaCacheParams` | 用户侧 CLI/python 参数 |
| `configs/sample/sampling_params.py` | `enable_teacache`、`teacache_params` | 请求级控制 |
| `runtime/models/dits/wanvideo.py` | `should_skip_forward_for_cached_states`、`maybe_cache_states`、`retrieve_cached_states` | Wan 视频 DiT 的 TeaCache 落点 |

## 8. 调优建议

- **Wan2.1 / Wan2.2 视频** 首选 TeaCache（需模型原生支持）；threshold 0.05–0.15；CFG 场景记得 `skip_boundaries` 要翻倍；
- **Flux / QwenImage 图像** 首选 cache-dit（DBCache），配合 TaylorSeer 提高跨步重用；
- **长视频 + SP/TP** 一定要保证 `_patch_cache_dit_similarity` 正确配置 dit group；
- **多 batch/dynamic shape** 慎重：cache-dit 的 context manager 会按形状缓存，波动大时命中率下降；
- **启用 Offload 时**：`SGLANG_CACHE_DIT_ENABLED` 与 `dit_layerwise_offload` 互斥，若需要 offload 又想加速，优先 TeaCache；
- **和 torch.compile 合用**：务必先挂 cache-dit，再 compile，否则会 re-trace。
