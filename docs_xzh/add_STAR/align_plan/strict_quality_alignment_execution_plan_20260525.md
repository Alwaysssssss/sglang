# STAR strict 质量对齐实施文档（2026-05-25）

## 1. 文档目的

这份文档用于指导下一阶段的 STAR 画面对齐工作。当前主目标不是继续追求更高加速比，而是在**相同推理参数**下，把集成到 `sglang` 的 STAR 结果尽可能对齐到原版 `STAR_mg`，并优先达到 `phase_5_decoding_parity_and_acceptance.md` 中定义的 `strict` 阈值。

这份文档只关注：

1. 如何建立稳定、可解释的 strict 对齐基线
2. 如何把“编码器差异 / 脚本差异 / 模型差异”分开排查
3. 先做哪些工作，后做哪些工作
4. 哪些方向当前不要继续投入

---

## 2. 当前基线与关键结论

### 2.1 当前最重要的现状

当前 `sglang` 侧已经可以在本地模型目录下稳定跑通 STAR exact 和 FP8 路线，但主观质量与原版 STAR 仍未对齐到 strict。

当前最应该围绕的 exact 基线为：

1. 单卡
2. exact
3. `attention_backend = fa`
4. `fps = 8`
5. 固定 reference case
6. 不引入 quantization / cache / 多卡 / 新 backend 变量

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

### 2.3 当前 FP8 结果的定位

现有 FP8 本地模型目录结果同样 baseline 通过、strict 失败，但当前不是主线。

FP8 当前只保留为：

1. 后续回归参考
2. strict 过线后再复验的副线

当前不要继续围绕 FP8 做主攻排查。

---

## 3. strict 验收口径

以 `phase_5_decoding_parity_and_acceptance.md` 为准，当前目标口径为：

```text
min_ssim = 0.95
max_mse = 60.0
max_mae = 5.0
allow_frame_count_delta = 0
max_failed_frame_ratio = 0.0
```

这里必须注意两点：

1. `strict` 适用于同一台机器、同一 backend、同一视频编码器、同一 dtype 配置
2. 当前 `mp4` 比较里混入了视频编码器差异，因此**不能直接把 strict 失败全部解释为模型本身失败**

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

### 5.4 VAE tiling 行为

当前 `sglang` STAR pipeline config 默认 `vae_tiling = true`。

需要明确：

1. 原版 reference 路径是否在 encode / decode 上使用相同 tiling 语义
2. 如果原版是 full-frame，而 `sglang` 使用 tiling，可能带来轻微但稳定的结构差异

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

---

## 6. 实施原则

### 6.1 先拆变量，再调模型

严格按以下顺序：

1. 先拆 `raw frame` 和 `mp4` 编码差异
2. 再补 reference trace
3. 再做中间量二分
4. 最后才动数值实现

### 6.2 单变量实验

在 strict 对齐阶段，每次只改一个变量。

不要同时改：

1. compile
2. attention backend
3. quantization
4. multi-GPU
5. cache / teacache / cache-dit
6. 新的 fused 热路径

### 6.3 先 no-compile exact，再 compile exact

质量对齐时建议先建立一个**最少变量**的 exact debug 基线，再回归 compile 路径。

原因：

1. compile 是性能变量，不应在第一轮就和质量变量绑死
2. 如果 no-compile exact 已 strict 通过，而 compile exact 失败，问题范围会立即缩小

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

1. 如果 `raw vs raw` 已经 strict 通过，而 `mp4 vs mp4` 不通过，优先修输出编码一致性，不要先动模型
2. 如果 `raw vs raw` 本身也不通过，再进入模型侧排查

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

## 7.4 Phase D：按阶段做二分定位

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

如果 `image_latent` 还没有对齐，不要继续查 attention。

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
5. compile / no-compile 是否引入额外漂移

这里要特别注意：

1. 原版本身就是 cond/uncond batched forward，不要先入为主地把 `enable_batched_cfg` 当成错误
2. 若怀疑 compile 引入漂移，先做 `sglang vs sglang` 的 no-compile / compile 对照，而不是直接改 reference 对比脚本

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

## 7.5 Phase E：回归 compile exact acceptance

当 no-compile exact 已经通过 raw strict 与 mp4 strict 后，再回归当前 compile exact 主线：

1. 使用 `infer_STAR.md` 中的 exact profile 命令复验
2. 对比 no-compile exact 与 compile exact 的 raw frame parity
3. 若 compile 退化，再用 `compare_diffusion_trajectory_similarity.py` 做 `sglang vs sglang` 轨迹对照

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
2. 如果 raw strict 已过，停止模型侧调参，优先统一视频写盘路径

### P3

1. 如果 raw strict 未过，优先查 condition-video VAE posterior RNG
2. 然后查 `vae_tiling`
3. 再查 condition-video preprocess

### P4

1. 只有在上面都排掉之后，再去查 fused norm / rope / attention / local enhancer

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

### 情况 C

`image_latent` 接近，但第 1 步或中间步 latent 很快漂移。

处理方式：

1. 查 initial noise
2. 查 scheduler timesteps / per-step noise
3. 查 CFG combine 路径

### 情况 D

最终 latent 接近，但 raw frame 仍不过 strict。

处理方式：

1. 查 decode scale / shift
2. 查 temporal windows
3. 查 VAE tiling

---

## 10. 完成标准

这一轮质量对齐工作建议按三层 gate 收口：

### Gate 1：raw frame strict

在 reference raw frame 与 `sglang` raw frame 上达到：

1. `ssim_min >= 0.95`
2. `mse_max <= 60.0`
3. `mae_mean <= 5.0`
4. `failed_frame_ratio = 0.0`

### Gate 2：mp4 strict

在统一编码参数后，reference `mp4` 与 candidate `mp4` 达到同样 strict 阈值。

### Gate 3：current exact acceptance path strict

在当前主线 exact 命令上复验 strict 通过。

只有 Gate 1、Gate 2、Gate 3 都通过，才算当前质量对齐主目标完成。

---

## 11. 当前明确不要主攻的方向

下一阶段不要把主要时间继续投入在以下方向：

1. FP8
2. AWQ
3. Nunchaku / SVDQuant
4. 双卡 cfg-parallel
5. cache / teacache / cache-dit 继续提速
6. FlashInfer RoPE

这些方向都应等 strict 质量主线完成后再回头处理。

---

## 12. 最后一条执行建议

后续所有 strict 对齐工作，建议都遵循一句话：

**先证明差异来自模型，再去改模型。**

如果 raw frame、trace、encoder 参数还没有拆清楚，就不要急着改 denoise 热路径，也不要急着追求更高速度。
