# VideoEdit 接入 TeaCache 并兼容 Offload 的修改方案

本文给出在 SGLang `videoedit` 推理链路中接入 TeaCache 的代码修改计划。目标是：

- `enable_teacache=false` 时保持现有行为不变。
- `enable_teacache=true` 时，VideoEdit 使用最小 Wan 模型的 TeaCache 判跳标准。
- TeaCache 可以和 `--dit-cpu-offload`、`--dit-layerwise-offload` 一起使用。

这里的“最小 Wan 模型”对应当前仓库中的 `WanT2V_1_3B_SamplingParams`，也就是 `python/sglang/multimodal_gen/configs/sample/wan.py` 里已经写好的 `_wan_1_3b_coefficients` 和默认 TeaCache 参数。

## 1. 当前状态

当前仓库已经具备通用 TeaCache 框架：

```text
python/sglang/multimodal_gen/runtime/cache/teacache.py
python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py
python/sglang/multimodal_gen/configs/sample/wan.py
```

`WanTransformer3DModel` 已经实现了 TeaCache 的核心路径：

- 每次 DiT forward 后缓存 `hidden_states - original_hidden_states`。
- 下一步如果判定可跳过，则用 `hidden_states + previous_residual` 代替 transformer block 计算。
- `use_ret_steps=True` 时使用 `timestep_proj` 做相邻步相似度判断。
- `use_ret_steps=False` 时使用 `temb` 做判断。

VideoEdit 侧也已经暴露了入口参数：

```text
python/sglang/multimodal_gen/runtime/videoedit/cli.py
python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py
python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py
```

CLI 和 API 都已有 `enable_teacache` 字段，并会传入 `WanVideoEditSamplingParams`。

当前缺口：

1. `WanVideoEditSamplingParams` 没有设置 `teacache_params`，所以 `enable_teacache=True` 后 `_get_teacache_context()` 会因为参数为空而直接 no-op。
2. `WanVideoEditConfig.prefix` 是 `"WanVideoEdit"`，当前 `TeaCacheMixin` 的 CFG 分支缓存判断只认精确前缀 `"wan"`，导致 VideoEdit 不会被识别为 Wan 系 CFG 可分离模型。
3. `VideoEditDenoisingStage` 没有在正向/负向 CFG pass 前设置 `batch.is_cfg_negative`，TeaCache 无法区分正负分支缓存。
4. TeaCache reset 逻辑目前用的是 `self.is_cfg_negative` 的旧值，CFG 第一个负向 pass 也可能触发 reset。建议顺手改成使用当前 `forward_batch.is_cfg_negative`。

## 2. 判跳标准

VideoEdit 第一版直接复用 Wan 1.3B 的 TeaCache 标准：

```python
TeaCacheParams(
    teacache_thresh=0.08,
    use_ret_steps=True,
    coefficients_callback=_wan_1_3b_coefficients,
    start_skipping=5,
    end_skipping=1.0,
)
```

判跳流程保持和现有 Wan 一致：

1. 当前步取 `timestep_proj` 作为 modulated input。
2. 计算相邻步相对 L1：

```python
rel_l1 = (modulated_inp - prev_modulated_inp).abs().mean() / prev_modulated_inp.abs().mean()
```

3. 用 `_wan_1_3b_coefficients` 的四阶多项式重标定 `rel_l1`。
4. 将重标定后的值累加到 `accumulated_rel_l1_distance`。
5. 如果累计值达到 `teacache_thresh=0.08`，本步正常计算并清零累计值。
6. 如果累计值小于阈值，本步跳过 transformer blocks，复用上一轮 residual。
7. `start_skipping=5` 表示前 5 个 denoising step 强制计算；有 CFG 时按现有实现折算成前 10 个 forward pass。

## 3. 代码修改点

### 3.1 为 VideoEdit SamplingParams 加默认 TeaCache 参数

文件：

```text
python/sglang/multimodal_gen/configs/sample/videoedit_wan.py
```

推荐最小改动：

```python
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams
from sglang.multimodal_gen.configs.sample.wan import _wan_1_3b_coefficients
```

在 `WanVideoEditSamplingParams` 增加：

```python
teacache_params: TeaCacheParams = field(
    default_factory=lambda: TeaCacheParams(
        teacache_thresh=0.08,
        use_ret_steps=True,
        coefficients_callback=_wan_1_3b_coefficients,
        start_skipping=5,
        end_skipping=1.0,
    )
)
```

更干净的长期做法是把 `_wan_1_3b_coefficients`、`_wan_14b_coefficients` 抽到新文件：

```text
python/sglang/multimodal_gen/configs/sample/wan_teacache.py
```

然后 `wan.py` 和 `videoedit_wan.py` 都从这个文件 import，避免 VideoEdit 依赖 `wan.py` 的私有函数。

### 3.2 让 TeaCache 识别 WanVideoEdit 的 CFG 分支缓存

文件：

```text
python/sglang/multimodal_gen/runtime/cache/teacache.py
```

当前逻辑是精确匹配：

```python
self._supports_cfg_cache = (
    self.config.prefix.lower() in self._CFG_SUPPORTED_PREFIXES
)
```

建议改成前缀匹配：

```python
prefix = self.config.prefix.lower()
self._supports_cfg_cache = any(
    prefix.startswith(supported)
    for supported in self._CFG_SUPPORTED_PREFIXES
)
```

这样 `Wan`、`WanVideoEdit`、后续其他 Wan 派生 DiT 都能复用 Wan 的正负 CFG 分支缓存。

同时建议把 reset 判断从旧状态改成当前 batch 状态：

```python
is_cfg_negative = forward_batch.is_cfg_negative

if current_timestep == 0 and not is_cfg_negative:
    self.reset_teacache_state()
```

这样第一个负向 CFG pass 不会误清掉第一个正向 pass 已经建立的 cache 状态。

### 3.3 在 VideoEdit denoising 中标记 CFG 正负分支

文件：

```text
python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py
```

在进入 denoising loop 前，设置 TeaCache 看到的 CFG 总体状态：

```python
batch.do_classifier_free_guidance = bool(params.runtime_do_cfg)
batch.is_cfg_negative = False
```

每步正向 prompt pass 前：

```python
batch.is_cfg_negative = False
with set_forward_context(...):
    noise_pred = self.transformer(...)
```

每步负向 prompt pass 前：

```python
batch.is_cfg_negative = True
with set_forward_context(...):
    noise_uncond = self.transformer(...)
```

负向 pass 结束后恢复：

```python
batch.is_cfg_negative = False
```

注意：VideoEdit 有 dynamic CFG，`do_cfg` 是每步动态值；但 TeaCache 的 CFG 边界折算建议继续使用 `params.runtime_do_cfg` 这个静态值。原因是 Wan 的 TeaCache 计数器 `self.cnt` 按 forward pass 累加，正负 CFG 分支共用同一个计数器；如果中途把 `batch.do_classifier_free_guidance` 从 true 切到 false，会让 `start_skipping/end_skipping` 边界突然从双 pass 语义切回单 pass 语义，容易出现不可解释的跳步窗口变化。

### 3.4 确认 offload 兼容路径

相关文件：

```text
python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py
python/sglang/multimodal_gen/runtime/utils/layerwise_offload.py
python/sglang/multimodal_gen/runtime/managers/gpu_worker.py
```

`--dit-cpu-offload`：

- VideoEdit denoising 里已经调用 `_manage_device_placement(self.transformer, None, server_args)`。
- DiT 在 denoising loop 前被加载到 GPU，TeaCache 的 residual tensor 也保留在当前 GPU 上。
- 每个窗口从 `current_timestep=0` 开始会 reset TeaCache state，不会跨窗口复用旧 residual。

`--dit-layerwise-offload`：

- layerwise offload 只管理 `WanTransformer3DModel.layer_names = ["blocks"]` 中的 transformer blocks。
- TeaCache skip 时会绕过 `for block in self.blocks`，因此不会触发 block 的 pre/post hooks，也不会加载 blocks 权重。
- 下一次需要真实计算时，`blocks[0]` 的 pre-hook 会调用 `prepare_for_next_req()`，重新保证 layer 0/prefetch 层在 GPU 上。
- `norm_out`、`proj_out`、`patch_embedding`、`condition_embedder` 不在 layerwise offload 管理范围内，skip 后仍能完成输出投影。

因此不需要给 TeaCache 和 layerwise offload 加互斥限制。现有互斥只针对 `SGLANG_CACHE_DIT_ENABLED`，不要把 TeaCache 误归到 Cache-DiT 的限制里。

建议加一个防御性检查：

```python
def retrieve_cached_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
    residual = (
        self.previous_residual_negative
        if self.is_cfg_negative
        else self.previous_residual
    )
    if residual is None:
        return hidden_states
    return hidden_states + residual.to(device=hidden_states.device, dtype=hidden_states.dtype)
```

正常情况下不会出现 `residual is None` 的 skip；这个检查主要是避免 offload、窗口 reset 或异常中断后的坏状态直接报错。

## 4. 验证方案

### 4.1 单元级检查

建议补轻量测试：

1. `WanVideoEditSamplingParams(enable_teacache=True).teacache_params` 非空。
2. `get_coefficients()` 返回 Wan 1.3B 的系数。
3. `WanVideoEditConfig(prefix="WanVideoEdit")` 初始化的 DiT 满足 `_supports_cfg_cache=True`。
4. CFG 场景下正负 pass 分别更新 `previous_residual` 和 `previous_residual_negative`。

### 4.2 CLI 冒烟测试

分别跑四组：

```bash
# baseline
python -m sglang.multimodal_gen.runtime.videoedit.cli repair ... \
  --no-enable-teacache \
  --no-dit-cpu-offload \
  --no-dit-layerwise-offload

# TeaCache only
python -m sglang.multimodal_gen.runtime.videoedit.cli repair ... \
  --enable-teacache \
  --no-dit-cpu-offload \
  --no-dit-layerwise-offload

# TeaCache + DiT CPU offload
python -m sglang.multimodal_gen.runtime.videoedit.cli repair ... \
  --enable-teacache \
  --dit-cpu-offload true \
  --no-dit-layerwise-offload

# TeaCache + layerwise offload
python -m sglang.multimodal_gen.runtime.videoedit.cli repair ... \
  --enable-teacache \
  --dit-layerwise-offload true \
  --dit-offload-prefetch-size 1 \
  --no-dit-cpu-offload
```

关注点：

- 四组都能完成输出。
- `VideoEditDenoisingStage` latency 在 TeaCache 组下降。
- `--dit-layerwise-offload` 组没有 placeholder shape mismatch、device mismatch、None residual。
- 同 seed 输出没有明显闪烁或局部破坏；必要时用现有 video similarity 脚本和 baseline 做 SSIM/LPIPS 对比。

### 4.3 API 冒烟测试

`/v1/videos/repairs` 请求体里设置：

```json
{
  "enable_teacache": true,
  "num_inference_steps": 20,
  "guidance_scale": 5.0
}
```

服务端启动分别覆盖：

```bash
--dit-cpu-offload true --dit-layerwise-offload false
--dit-cpu-offload false --dit-layerwise-offload true --dit-offload-prefetch-size 1
```

如果云端默认打开 text/image/vae offload，也要一起跑一组，确认 TeaCache 只影响 DiT denoising，不破坏其他组件的 offload。

## 5. 风险和处理

1. 质量风险：VideoEdit 是编辑模型，不一定和 T2V 1.3B 的最优阈值完全一致。第一版按需求复用最小 Wan 标准，后续如果局部修复质量下降，再单独标定 `teacache_thresh`。
2. Dynamic CFG 风险：后几步 `current_cfg` 可能等于 1，不再跑负向 pass。建议 TeaCache 的 CFG 边界仍用静态 `params.runtime_do_cfg`，避免边界随 step 动态跳变。
3. 多窗口风险：VideoEdit 会逐窗口推理。必须依赖 `current_timestep=0` reset TeaCache state，不能跨窗口复用 residual。
4. Layerwise offload 风险：skip 连续发生时不会触发 block hooks。下一次真实计算必须从 `blocks[0]` pre-hook 重新 prepare，现有 hook 已满足；验证时重点看连续 skip 后第一步真实计算。
5. Cache-DiT 混用风险：TeaCache 和 Cache-DiT 都会改 DiT forward 复用逻辑，第一版不建议同时打开 `enable_teacache` 和 `SGLANG_CACHE_DIT_ENABLED=1`。即使代码没有强制互斥，测试矩阵也先排除这个组合。

## 6. 推荐落地顺序

1. 先加 `WanVideoEditSamplingParams.teacache_params`，用 Wan 1.3B 参数。
2. 改 `TeaCacheMixin` 的 CFG 支持判断和 reset 条件。
3. 改 `VideoEditDenoisingStage` 的 `batch.is_cfg_negative` 标记。
4. 加 residual 的 `None/device/dtype` 防御。
5. 跑 no-offload、DiT CPU offload、layerwise offload 三组冒烟。
6. 再补轻量单元测试和 API 测试。

