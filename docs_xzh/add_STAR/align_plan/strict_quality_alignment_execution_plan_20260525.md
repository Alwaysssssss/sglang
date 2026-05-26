# STAR strict 质量对齐实施文档（2026-05-25）

## 1. 文档目的

这份文档用于指导下一阶段的 STAR 画面对齐工作。当前主目标不是继续追求更高加速比，而是在**相同推理参数**下，把集成到 `sglang` 的 STAR 结果尽可能对齐到原版 `STAR_mg`，并优先达到 `phase_5_decoding_parity_and_acceptance.md` 中定义的 `strict` 阈值。

这份文档只关注：

1. 如何建立稳定、可解释的 strict 对齐基线
2. 如何把“编码器差异 / 脚本差异 / 模型差异”分开排查
3. 先做哪些工作，后做哪些工作
4. 哪些方向当前不要继续投入

本轮后续工作的硬约束：

1. **不改动、不回退当前已经完成的 `sglang` STAR 加速工作**
2. **后续修改只允许围绕推理语义本身做对齐**
3. **`sglang` 自己的 compile / FlashAttention / fused / runtime 加速实现，当前视为固定前提，而不是待回退对象**

---

## 2. 当前基线与关键结论

### 2.1 当前最重要的现状

当前 `sglang` 侧已经可以在本地模型目录下稳定跑通 STAR exact 和 FP8 路线。  
其中，当前 exact compile 主线已经通过 `strict 0.95` release gate，但距离 `raw 0.97 / 0.98` 的高标准对齐目标仍有差距。

当前最应该围绕的 exact 基线为：

1. 单卡
2. exact
3. `attention_backend = fa`
4. `fps = 8`
5. 固定 reference case
6. 不引入 quantization / cache / 多卡 / 新 backend 变量

这里需要明确：

1. 当前主线基线不是“最接近原版代码结构的慢路径”
2. 当前主线基线是“**已经通过 strict release gate 的 `sglang` exact compile 路径**”
3. 后续要继续对齐的是这条固定加速路径上的**推理语义**

### 2.2 当前 exact 本地模型目录结果

现有 exact 本地模型目录产物：

1. 输出目录：`/sgl-workspace/sglang/outputs/star_repro_single_fa_compile_fusedln_v2_fps8_localmodel`
2. 结果摘要：`summary.json`

关键指标：

1. `ssim_mean = 0.9364771154`
2. `ssim_min = 0.9304712817`
3. `mse_max = 33.1698989868`
4. `mae_mean = 2.9407699871`
5. `strict` 下 `num_failed_frames = 25`
6. `strict` 下 `failed_frame_ratio = 1.0`

重要结论：

1. 当前 strict 失败的主要瓶颈是 `SSIM`
2. `MSE / MAE` 已经明显落在 strict 阈值之内
3. 这更像是**轻微但系统性的结构偏差**，而不是尺度错、帧数错、通道错这种粗错误

### 2.3 当前已通过的对齐验收结果

在统一 `mp4` 编码质量并补齐 `raw frame` 对比后，当前 exact compile 主线已经通过 release gate。

当前推荐参考这组最新产物：

1. 输出目录：`/sgl-workspace/sglang/outputs/star_align/sglang_exact_compile_case023_q10`
2. reference 目录：`/sgl-workspace/sglang/outputs/star_align/reference_case023_q10_raw`

关键结果：

1. `reference raw` vs `candidate raw`
   - `ssim_mean = 0.9601536133`
   - `ssim_min = 0.9575826519`
   - `num_failed_frames = 0`
2. `reference mp4` vs `candidate mp4`
   - `ssim_mean = 0.9529690934`
   - `ssim_min = 0.9507997807`
   - `num_failed_frames = 0`
3. `reference raw -> reference mp4` 编码损失
   - `ssim_mean = 0.9885351781`
4. `candidate raw -> candidate mp4` 编码损失
   - `ssim_mean = 0.9877945149`

重要结论：

1. 当前 `strict 0.95` release gate 已完成
2. 编码器质量差异已被验证为此前 strict 失败的重要来源之一
3. 但 `raw` 还只有 `0.960 / 0.958` 左右，因此距离 `0.97 / 0.98` 的高标准目标仍有实质差距

### 2.4 当前 FP8 结果的定位

现有 FP8 本地模型目录结果同样 baseline 通过、strict 失败，但当前不是主线。

FP8 当前只保留为：

1. 后续回归参考
2. strict 过线后再复验的副线

当前不要继续围绕 FP8 做主攻排查。

### 2.5 当前新增语义实验结论

在固定 compile + FA 主线下，已经额外完成两组高价值语义 A/B：

1. `condition_video_vae_sample_rng_mode = global_seed`
2. `vae_tiling = false`

结果：

1. `global_seed` 路线显著劣化
   - `raw_ssim_mean` 下降到约 `0.8827`
   - 连 baseline 都无法通过
   - 说明这条路径不是原版 STAR 的正确参考语义
2. `no_vae_tiling` 与当前主线结果几乎完全一致
   - `raw_ssim_mean` 仍约为 `0.9602`
   - `mp4 strict` 指标也与当前主线一致
   - 说明 `vae_tiling` 不是当前 `0.960x` 差距的主要来源

额外 trace 对比结论：

1. `condition_video preprocess` 已与 reference 完全一致
2. `condition latent` 的统计量与 reference 非常接近
3. `decode 前 final latents` 与 `decode 输入 latents` 的统计量也都已非常接近

当前最重要的含义是：

1. 低层次的 `condition preprocess / VAE encode 粗统计 / decode scale-shift` 已不是主要矛盾
2. 剩余差距更像是 **denoise 过程内部的 tensor 级语义差异**
3. 下一步不应继续围绕 `global_seed` 或 `vae_tiling` 反复试验

---

## 3. 验收口径与术语

### 3.1 `raw` 和 `mp4` 分别是什么

后续讨论里，`raw` 和 `mp4` 需要严格区分：

1. `raw`
   - 指 VAE decode 和后处理完成后、进入视频编码器之前的逐帧原始图像
   - 当前通常以 `frames/frame_XXXX.png` 的形式落盘
   - 这个口径最接近“模型本身到底有没有对齐”
   - 如果要把 SSIM 继续抬到 `0.97 / 0.98`，主要应该看这个口径
2. `mp4`
   - 指经过 `imageio/ffmpeg` 等视频编码器压缩后的最终交付文件
   - 这个口径更接近用户实际看到的成品
   - 但它会混入 `codec`、`quality`、像素格式等编码器变量
   - 因此它适合做 release gate，不适合单独拿来判断“模型是否已经完全复刻”

简化理解：

1. `raw` 主要回答：模型本身是否对齐
2. `mp4` 主要回答：最终交付视频是否对齐

### 3.2 当前 release gate

以 `phase_5_decoding_parity_and_acceptance.md` 为准，当前 release gate 仍保留为：

```text
min_ssim = 0.95
max_mse = 60.0
max_mae = 5.0
allow_frame_count_delta = 0
max_failed_frame_ratio = 0.0
```

这里必须注意两点：

1. 这条 `strict 0.95` 口径仍然有效，适合作为当前主线 release gate
2. 这条口径可以同时用于 `raw` 和 `mp4`，但对 `mp4` 的解释必须考虑编码器差异

### 3.3 新的高标准对齐目标

由于当前项目目标是尽可能完整复刻原版 STAR，因此在 `strict 0.95` 之上，再引入一层更高标准的对齐目标：

#### High-Bar 对齐目标

优先使用 `raw vs raw` 比较，目标建议定为：

```text
raw_ssim_mean >= 0.97
raw_ssim_min  >= 0.97
raw_mse_max   <= 60.0
raw_mae_mean  <= 5.0
raw_failed_frame_ratio = 0.0
```

含义：

1. `0.97` 级别的目标主要用来判断“模型级对齐是否已经足够完善”
2. 这条目标比当前 `strict 0.95` 明显更严格
3. 这条目标应优先建立在 `raw` 上，而不是优先建立在 `mp4` 上

#### Stretch Goal

`0.98` 可以保留为 stretch goal，但当前不建议直接定义成主验收线：

```text
raw_ssim_mean >= 0.98
raw_ssim_min  >= 0.98
```

原因：

1. `0.98` 更适合作为“接近 reference implementation 无明显残差”的长期目标
2. 对当前阶段而言，`0.98` 不应替代 `0.95` release gate
3. 在没有多 case、没有更完整逐步 trace 的前提下，不建议把 `0.98` 直接写成必须满足的交付门槛

### 3.4 为什么高标准优先看 `raw`

如果把目标抬到 `0.97 / 0.98`，优先看 `raw` 而不是 `mp4` 的原因很直接：

1. `raw` 更接近模型本体，不会混入视频编码器变量
2. `mp4` 结果天然会受到编码质量和 codec 路径影响
3. 如果 `raw` 还没到 `0.97`，就不应要求 `mp4` 先到 `0.97 / 0.98`
4. 正确顺序应是：
   - 先把 `raw` 推到更高 SSIM
   - 再验证 `mp4` 交付结果是否同步稳定提升

---

## 4. 已确认的参考语义

这部分用于避免后续排查时“凭感觉修逻辑”。

### 4.1 原版 `STAR_mg` 当前参考链路的关键语义

来自 `STAR_mg/cogvideox-based/sat/sample_sr.py`、`diffusion_video.py`、`data_video.py`、`sgm/modules/diffusionmodules/sampling.py`：

1. condition video 来自 `PairedCaptionDataset`，当前 reference case 直接取 `lq[:25]`
2. `lq` 预处理语义是：
   - `width > 720` 时：`bilinear` 缩放后 `center_crop(480, 720)`
   - `width < 720` 时：`bicubic` 放大到宽度 720
3. condition video 归一化是 `(video / 255) * 2 - 1`
4. `sample_sr` 初始噪声形状是 `[B, T, C, H, W]`
5. `lq` encode 后回到 `[B, T, C, H, W]`，再为 CFG 做 `torch.cat((lq, lq), dim=0)`
6. decode 语义是第一段 `0:3`，后续每次滚动 `2` 帧，最后一段 `clear_fake_cp_cache = True`
7. color fix 当前是关闭的
8. 原版 `mp4` 保存使用 `imageio.get_writer(..., quality=10)`

### 4.2 当前 `sglang` 已对齐的关键语义

来自 `python/sglang/multimodal_gen/runtime` 当前实现：

1. STAR 已使用 modular pipeline，而不是混合大 stage
2. decode window 已按 STAR 专用 stage 实现
3. STAR scheduler 不是纯 ODE 简化版，仍然保留了每步 stochastic noise 注入
4. initial noise 已拆出 STAR 专用 CPU generator 以贴近原版初始噪声路径
5. `StarCogVideoXSRVAEConfig.encode_sample_mode()` 已设置为 `sample`

### 4.3 当前不要误判为 bug 的点

以下行为看起来反直觉，但当前应先视为**参考语义**而不是“顺手修掉”的 bug：

1. 原版 `DynamicCFG` 的步长调度不是简单的 `step_index = 0..49`
2. `sglang` 当前 dynamic CFG 公式应先以“是否和原版 trace 一致”为准
3. 不要因为公式看起来奇怪，就在没有对照 trace 的前提下直接改动

---

## 5. 当前最值得优先怀疑的差异源

下面按 ROI 排序给出优先排查项。

### 5.1 视频编码器差异

这是当前必须先拆掉的变量。

已确认：

1. 原版 `sample_sr.py` 写 `mp4` 时使用 `quality=10`
2. `sglang` 通用输出路径当前默认 `output_compression = 50`，落到 `imageio.mimsave(..., quality=5)`

这意味着：

1. 当前 `reference.mp4` 和 `candidate.mp4` 的编码器质量参数并不一致
2. strict 主要卡在 `SSIM`，而 `SSIM` 对视频编码损失很敏感
3. 如果不先把 raw frame parity 和编码器 parity 分开，后续很容易误把编码损失当成模型偏差

### 5.2 参考侧缺少 raw frame 产物

当前 `compare_star_sglang_outputs.py` 已支持对 frame dir 做比较，`sglang` 侧也能用 `--save-frame-pngs` 输出原始 PNG 帧。

但当前 reference 侧默认只有 `mp4`，缺少 decode 后、写盘前的 raw frame 目录。

这会导致：

1. 无法判断 strict 失败来自模型还是编码器
2. 无法建立真正的 raw-to-raw gate

### 5.3 condition-video VAE posterior 采样 RNG 路径

这是模型侧最值得优先验证的差异源之一。

现状：

1. 原版 `encode_first_stage()` 没有显式传 generator 给 posterior sample
2. `sglang` 的 `STARConditionVideoVAEEncodingStage` 当前通过 `retrieve_latents(..., batch.generator, sample_mode="sample")` 取 latent

这意味着：

1. 两侧即便都“可复现”，也可能不是同一条 posterior 采样轨迹
2. 这种差异会直接影响 condition latent，全程影响 denoise 结果
3. 这种差异非常符合“所有帧都略偏一点，但不是粗错”的当前表现

当前状态更新：

1. 已完成 `global_seed` A/B
2. 结果显著劣化，因此这条分支当前应视为**已排除方向**
3. 当前默认 `generator` 模式保留

### 5.4 VAE tiling 行为

当前 `sglang` STAR pipeline config 默认 `vae_tiling = true`。

需要明确：

1. 原版 reference 路径是否在 encode / decode 上使用相同 tiling 语义
2. 如果原版是 full-frame，而 `sglang` 使用 tiling，可能带来轻微但稳定的结构差异

当前状态更新：

1. 已完成 `vae_tiling on/off` A/B
2. 结果与当前主线几乎完全一致
3. 因此 `vae_tiling` 当前不再视为高优先级差异源

### 5.5 condition video 预处理细节

虽然当前 `STARConditionVideoLoadingStage` 已尽量贴近 `PairedCaptionDataset`，但仍要逐项核对：

1. resize 插值模式
2. center crop 位置
3. 帧选择顺序
4. 归一化区间
5. `[B, T, C, H, W]` 与 `[B, C, T, H, W]` 转换点

### 5.6 denoise 热路径中的轻微数值差异

只有在前几项都排除之后，再集中检查：

1. CFG 合并路径
2. latent concat 顺序
3. qk layernorm / fused layernorm / modulation 热路径
4. rope / local enhancer / attention backend 的细微数值漂移

当前状态更新：

1. 在 `condition preprocess`、`condition latent`、`decode 前后粗统计` 都已经接近 reference 的情况下
2. 剩余的主要矛盾已收敛到 **denoise 过程内部的 tensor 级差异**
3. 下一步优先顺序应调整为：
   - 初始噪声与 scheduler 噪声 RNG 轨迹
   - batched CFG combine 的 tensor 级差异
   - selected denoise steps 的 latent / noise_pred 对照

---

## 6. 实施原则

### 6.1 先拆变量，再调模型

严格按以下顺序：

1. 先拆 `raw frame` 和 `mp4` 编码差异
2. 再补 reference trace
3. 再做中间量二分
4. 最后才动数值实现

这里的“动数值实现”只指：

1. 修正与 reference 不一致的推理语义
2. 修正 condition latent / scheduler / decode / CFG 这些流程级偏差

不包括：

1. 回退 `sglang` 现有 compile 路径
2. 回退 `sglang` attention / fused / runtime 加速实现
3. 为了更像原版而改回逐算子执行

### 6.2 单变量实验

在 strict 对齐阶段，每次只改一个变量。

不要同时改：

1. compile
2. attention backend
3. quantization
4. multi-GPU
5. cache / teacache / cache-dit
6. 新的 fused 热路径

### 6.3 固定加速主线，不把“关加速”当成方案

后续语义对齐的主线必须固定在当前已经通过验收的加速配置上：

1. `enable_torch_compile = true`
2. `attention_backend = fa`
3. 保留当前 `sglang` fused / runtime 路径
4. 不改回原版逐算子实现

原因：

1. 用户当前要求是“不要对原有加速工作做任何变动”
2. 因此“先关 compile 看会不会更像原版”不再是主线方案
3. 后续所有对齐都应该回答：
   - 在**保持当前 `sglang` 加速后端不变**时，推理语义还有哪里与原版 STAR 不一致

允许存在的唯一例外：

1. 个别临时 A/B 可以把关某个选项作为定位手段
2. 但这种 A/B 只能作为证据采集
3. 不能作为最终修复方案合入主线

### 6.4 不要凭“看起来更合理”修改参考语义

所有修改都必须基于：

1. reference trace
2. candidate trace
3. raw frame parity
4. selected-step latent / noise_pred 对照

---

## 7. 建议的实施阶段

## 7.1 Phase A：冻结基线与产物命名

先冻结这条主线，不再换 case，不再换口径：

1. reference case 继续使用 `023_klingai_reedit`
2. exact 单卡作为主 debug 路线
3. `attention_backend = fa`
4. `fps = 8`
5. `num_inference_steps = 50`
6. `guidance_scale = 6.0`
7. `condition_video_num_frames = 25`
8. `enable_color_fix = false`
9. `enable_torch_compile = true`
10. 保留当前 `sglang` fused / runtime 加速路径

建议固定两条运行入口：

1. 原版 reference：沿用 `infer_STAR.md` 中原版命令
2. `sglang` debug candidate：沿用 `infer_STAR.md` 中 exact smoke / exact profile 命令

建议新增一套统一目录命名：

1. `outputs/star_align/reference_raw_case023`
2. `outputs/star_align/reference_mp4_case023`
3. `outputs/star_align/sglang_exact_raw_case023`
4. `outputs/star_align/sglang_exact_mp4_case023`
5. `outputs/star_align/trace_case023`

## 7.2 Phase B：先把编码器因素拆开

这一阶段的目标不是修模型，而是回答一个问题：

`strict` 到底是卡在模型，还是卡在 `mp4` 编码器。

需要完成的工作：

1. 在原版 `sample_sr.py` 增加“写 `mp4` 前保存 raw PNG 帧”的开关
2. 在 `sglang` 手工脚本中固定输出 raw PNG 帧
3. 在 `sglang` 手工脚本中显式暴露 `output_quality` 或 `output_compression`
4. 建立下面四组比较结果：
   - `reference_raw` vs `candidate_raw`
   - `reference_raw` vs `reference_mp4_decoded`
   - `candidate_raw` vs `candidate_mp4_decoded`
   - `reference_mp4` vs `candidate_mp4`

这一阶段的判定规则：

1. 如果 `raw vs raw` 已经达到 `strict 0.95`，而 `mp4 vs mp4` 不通过，优先修输出编码一致性，不要先动模型
2. 如果 `raw vs raw` 连 `strict 0.95` 都不过，再进入模型侧排查
3. 如果 `raw vs raw` 已过 `0.95`，但还没有到 `0.97`，说明 release gate 已满足，但“高标准对齐目标”还没有完成

## 7.3 Phase C：补齐 reference / candidate trace

当前缺少足够的中间量对照。下一阶段应优先补 trace，而不是盲改 runtime。

建议最少保存以下字段：

1. `prompt`
2. `negative_prompt`
3. `seed`
4. `condition_video_indices`
5. `condition_video_fps`
6. `condition_video_preprocess_summary`
7. `prompt_embeds` 的 shape / mean / std
8. `image_latent` 的 shape / mean / std
9. 初始 `randn` 的 shape / mean / std 与前若干元素快照
10. `timesteps`
11. `alphas_cumprod_sqrt`
12. 每步 CFG scale 序列
13. 第 `0 / N/2 / N-1` 步 `noise_pred` 的 shape / mean / std
14. 第 `0 / N/2 / N-1` 步 latent 的 shape / mean / std
15. decode 前最终 latent 的 shape / mean / std
16. raw frame 前 3 帧 PNG

建议 trace 产物格式：

1. `trace_manifest.json`
2. `selected_tensors.pt`
3. `frames/`

建议 reference 侧落点：

1. `STAR_mg/cogvideox-based/sat/sample_sr.py`
2. `STAR_mg/cogvideox-based/sat/diffusion_video.py`

建议 `sglang` 侧落点：

1. `python/sglang/multimodal_gen/test/manual/run_star_cogvideox_sr_smoke.py`
2. `python/sglang/multimodal_gen/test/manual/profile_star_cogvideox_sr.py`

## 7.4 Phase D：在固定加速后端下做语义二分定位

这一阶段的根约束：

1. 固定当前 `sglang` compile exact 主线
2. 固定 `attention_backend = fa`
3. 固定现有 fused / runtime 实现
4. 只检查“推理语义是否与原版 STAR 一致”

阶段顺序必须固定，不要跳步。

### D1. condition video preprocess parity

先对齐：

1. 取帧顺序
2. resize / crop
3. 归一化
4. 输出张量 layout

如果这一层不齐，不要继续查 scheduler。

主要代码位置：

1. `STAR_mg/cogvideox-based/sat/data_video.py`
2. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/video_condition_loading.py`

### D2. text encode parity

确认：

1. prompt 文本完全一致
2. negative prompt 为空字符串
3. unconditional text embeddings 的 zero 语义与原版一致
4. tokenizer 长度、attention mask、embedding 统计一致

主要代码位置：

1. `STAR_mg/cogvideox-based/sat/sample_sr.py`
2. `STAR_mg/cogvideox-based/sat/sgm/modules/diffusionmodules/guiders.py`
3. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/text_encoding.py`
4. `python/sglang/multimodal_gen/configs/pipeline_configs/star_cogvideox_sr.py`

### D3. VAE encode parity

这是当前最值得优先挖的模型侧差异源。

重点核对：

1. posterior 是 `sample` 还是 `mode`
2. posterior sample 的 RNG 来源
3. `scale_factor = 0.7` 是否完全同语义
4. encode 时是否存在 tiling 差异
5. `image_latent` 数值分布是否与原版对齐

如果 `image_latent` 还没有对齐，不要继续查 attention，也不要试图通过关闭 `sglang` 加速路径来规避问题。

主要代码位置：

1. `STAR_mg/cogvideox-based/sat/diffusion_video.py`
2. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/video_condition_vae_encoding.py`
3. `python/sglang/multimodal_gen/configs/models/vaes/star_cogvideox_vae.py`
4. `python/sglang/multimodal_gen/runtime/models/vaes/star_cogvideox_vae.py`

### D4. initial noise / scheduler parity

确认以下项：

1. initial noise 的形状与 layout
2. initial noise 的 CPU / CUDA RNG 路径
3. `timesteps` 与 `alphas_cumprod_sqrt`
4. 每步 stochastic noise 注入是否一致
5. CFG scale 序列是否一致

注意：

1. `sglang` 当前 scheduler 不是纯 deterministic ODE
2. 不要把 scheduler 的 stochastic path 误认为已经被删掉
3. dynamic CFG 语义应先以 trace 为准，不要凭主观修改
4. 这一层要做的是确认“当前 `sglang` scheduler 语义是否与原版一致”，而不是把 scheduler 改写成原版逐行实现

主要代码位置：

1. `STAR_mg/cogvideox-based/sat/sgm/modules/diffusionmodules/sampling.py`
2. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/star_latent_preparation.py`
3. `python/sglang/multimodal_gen/runtime/models/schedulers/star_vpsde_dpmpp2m.py`
4. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/timestep_preparation.py`

### D5. denoise parity

只有在 `image_latent`、initial noise、scheduler trace 都对齐后，再查 denoise。

重点核对：

1. batched CFG 路径
2. `latent_model_input` 与 `image_latent` 的 concat 顺序
3. 第 `0 / N/2 / N-1` 步 `noise_pred`
4. 第 `0 / N/2 / N-1` 步 latent
5. compile 主线下的语义结果是否与 reference 一致

这里要特别注意：

1. 原版本身就是 cond/uncond batched forward，不要先入为主地把 `enable_batched_cfg` 当成错误
2. 当前主线不是“关闭 compile 看会不会更像原版”
3. 如果怀疑 compile 下存在语义偏差，应优先查输入语义、CFG scale、latent concat、condition latent 是否一致
4. 只有在证据非常明确时，才允许把 compile/no-compile A/B 当成一次性定位工具，而不是最终方案

建议复用工具：

1. `python/sglang/multimodal_gen/tools/compare_diffusion_trajectory_similarity.py`

主要代码位置：

1. `STAR_mg/cogvideox-based/sat/sgm/modules/diffusionmodules/guiders.py`
2. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`
3. `python/sglang/multimodal_gen/runtime/models/dits/star_cogvideox_sr.py`

### D6. decode parity

如果最终 latent 已经接近，但 raw frame 仍不通过 strict，就回到 decode。

重点核对：

1. decode 前反 scale / shift
2. temporal window 切分
3. `clear_fake_cp_cache` 时机
4. `vae_tiling` 是否造成细微差异
5. `(frames / 2 + 0.5).clamp(0, 1)` 与原版后处理是否完全一致

主要代码位置：

1. `STAR_mg/cogvideox-based/sat/sample_sr.py`
2. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/star_cogvideox_sr_decoding.py`
3. `python/sglang/multimodal_gen/configs/pipeline_configs/star_cogvideox_sr.py`

## 7.5 Phase E：固定加速主线下的高标准语义对齐收口

由于当前 compile exact 主线已经通过 `strict 0.95`，后续阶段不再围绕“是否回到 compile 主线”展开，而是直接在 compile exact 主线上继续提高 `raw` 对齐度。

后续收口目标：

1. 在当前 compile exact 主线上稳定保持 `strict 0.95`
2. 在不改动现有加速工作前提下，把 `raw` 对齐继续从 `0.960 / 0.958` 往 `0.97+` 推进
3. 所有修正都必须证明自己是在修复推理语义，而不是在牺牲加速实现

---

## 8. 建议的首批具体任务

按优先级排序，建议下一位 Codex 直接从这里开始。

### P0

1. 为原版 `sample_sr.py` 增加 raw PNG 帧导出能力
2. 为 `run_star_cogvideox_sr_smoke.py` / `profile_star_cogvideox_sr.py` 增加显式 `output_quality` 或 `output_compression` 控制
3. 用统一编码质量重新生成一组 `mp4` 对比

### P1

1. 为 original / `sglang` 两边补 `trace_manifest.json`
2. 至少导出 `image_latent`、initial noise、timesteps、selected-step latents、decode 前 latent

### P2

1. 先做 `reference_raw` vs `candidate_raw` strict
2. 如果 raw strict 已过，先完成编码器口径统一
3. 如果 raw strict 已过但仍只有 `0.96x`，则进入“高标准语义对齐”阶段，而不是继续改输出编码器

### P3

1. 如果目标是把 `raw` 从 `0.96x` 继续提高，优先查 condition-video VAE posterior RNG 语义
2. 然后查 `vae_tiling`
3. 再查 condition-video preprocess
4. 以上修改必须保持当前 compile / FA / fused 路径不变

### P4

1. 只有在上面都排掉之后，再去查 latent concat、CFG combine、decode scale/shift 这类推理语义问题
2. 不把 fused norm / rope / attention / local enhancer 改回原版算子当成方案

---

## 9. 决策树

### 情况 A

`reference_raw` vs `candidate_raw` strict 通过，但 `reference_mp4` vs `candidate_mp4` strict 不通过。

处理方式：

1. 不要继续改模型
2. 统一编码质量、编码器参数、输出写盘路径
3. 把 strict release gate 的输入改为统一编码后的产物

### 情况 B

`image_latent` 已明显偏离 reference。

处理方式：

1. 停在 VAE encode 层
2. 优先查 RNG、sample_mode、tiling、preprocess
3. 不要先查 denoise
4. 不要通过关 compile、换 backend、回退 fused 路径来绕过问题

### 情况 C

`image_latent` 接近，但第 1 步或中间步 latent 很快漂移。

处理方式：

1. 查 initial noise
2. 查 scheduler timesteps / per-step noise
3. 查 CFG combine 路径
4. 保持当前 `sglang` 加速后端不变，只修正语义

### 情况 D

最终 latent 接近，但 raw frame 仍不过 strict。

处理方式：

1. 查 decode scale / shift
2. 查 temporal windows
3. 查 VAE tiling
4. 不把“回退到原版逐算子 decode 路径”作为默认解法

---

## 10. 完成标准

这一轮质量对齐工作建议按分层 gate 收口：

### Gate 1：Release Gate

这层 gate 代表“当前版本可以作为 release 候选”。

要求：

1. `reference raw` vs `candidate raw` 达到 `strict 0.95`
2. `reference mp4` vs `candidate mp4` 达到 `strict 0.95`
3. 当前主线 exact acceptance path 复验通过
4. 上述通过建立在**不改动当前加速主线**的前提下

### Gate 2：High-Bar 对齐目标

这层 gate 代表“模型级对齐已经比较完善”。

要求：

1. `reference raw` vs `candidate raw`
2. `raw_ssim_mean >= 0.97`
3. `raw_ssim_min >= 0.97`
4. `raw_failed_frame_ratio = 0.0`
5. 其余 `MSE / MAE / frame_count` 仍满足 `strict 0.95` 的限制
6. 保持当前 compile / FA / fused 加速实现不变

### Gate 3：Stretch Goal

这层 gate 不作为当前硬性 release 条件，但可作为长期追求目标。

要求：

1. `reference raw` vs `candidate raw`
2. `raw_ssim_mean >= 0.98`
3. `raw_ssim_min >= 0.98`
4. `raw_failed_frame_ratio = 0.0`

建议解释方式：

1. Gate 1 通过：可以说“当前主线验收通过”
2. Gate 2 通过：可以说“在不回退加速实现的前提下，高标准语义对齐完成度较高”
3. Gate 3 通过：可以说“非常接近 reference implementation”

---

## 11. 当前明确不要主攻的方向

下一阶段不要把主要时间继续投入在以下方向：

1. FP8
2. AWQ
3. Nunchaku / SVDQuant
4. 双卡 cfg-parallel
5. cache / teacache / cache-dit 继续提速
6. FlashInfer RoPE
7. 为了贴近原版而回退 `sglang` 现有 compile / FA / fused 加速路径
8. 把运行时改成原版逐算子实现

这些方向都应等 strict 质量主线完成后再回头处理。

---

## 12. 最后一条执行建议

后续所有高标准语义对齐工作，建议都遵循一句话：

**先证明差异来自推理语义，再去修语义；不要为了更像原版而撤掉 `sglang` 的加速实现。**

如果 raw frame、trace、encoder 参数还没有拆清楚，就不要急着改 denoise 热路径，也不要急着追求更高速度。
