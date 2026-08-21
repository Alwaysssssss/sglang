# VideoEdit-diffusers 与 SGLang VideoEdit 推理对齐决策

## 1. 文档目的

本文记录对以下两套 VideoEdit 推理实现逐项审查、讨论后的最终决策：

- 算法基线：`/mnt/shanhai-ai/liuh/VideoEdit-diffusers/infer.py`
- SGLang 实现：`python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py`

算法仓以 tracked 的主入口 `infer.py` 为准，不使用未跟踪 batch/legacy 脚本的默认值覆盖主入口语义。

本文是实现与验收依据，不是代码完成状态报告。标记为“必须对齐”的项目仍需按本文实现和验证；标记为“明确保留”的差异不应在本轮被顺手修改。

数值验收以两端 bbox crop-only 输出为主。crop 是模型实际生成并参与 paste-back 的区域，能够直接暴露算法差异；完整 paste-back 视频含有大量未编辑背景，只作为输出几何、回贴和编码链路的辅助检查，不能替代 crop 主验收。

本轮实施必须分为两个串行阶段：

1. **阶段一：单窗口对齐。** 先消除同一窗口内的输入预处理、随机噪声、conditioning、DiT、scheduler、VAE 和 decode 数值差异。
2. **阶段二：多窗口对齐。** 只有阶段一通过验收后，才实现和验证 overlap 传播、long/short 双 pass、bridge、backward 方向及 global-index 归并。

禁止在单窗口 numerical baseline 尚未通过时同时调试多窗口时序；否则无法区分差异来自窗口内部还是窗口调度。

## 2. 决策摘要

目标实现只保留一套有参考图的 VideoEdit 主时序：

```text
必填 edited reference
        |
        v
reference out-of-band + ref_frame_idx（默认 0）
        |
        v
选择较长方向 long pass
        |
        v
long 生成结果构造 reversed bridge
        |
        v
short pass
        |
        v
按原视频 global index 归并并排序
        |
        v
输出严格对应原视频 0..N-1
```

窗口内部统一采用算法仓语义：

```text
stride = infer_len - overlap
完整 overlap 生成结果回填下一窗
overlap mask 全黑
后窗跳过 overlap 输出
tail 使用 reverse-mirror
```

不再为 reference prepend、drop-first、weighted overlap、reflect tail 等旧语义保留第二套 VideoEdit 主路径。

### 2.1 分阶段对齐原则

阶段一固定使用最小但完整的单窗口场景：

```text
source video frames = 48
edited reference frames = 1
ref_frame_idx = 0
infer_len = 49
overlap = 0
仅执行一个 forward window
最终输出 = 48 个源视频索引对应的生成帧
```

阶段一只验证窗口内部算法，不以多窗口功能是否完成作为阻塞条件。以下内容在阶段一必须对齐：

- bbox、crop、resize 和 reference 预处理；
- `diffuser` / `diffsynth` CLIP 预处理选择及对应 image embeddings；
- mask dilation、scale、空间下采样和时间 packing；
- masked-video condition latent；
- CPU float32 初始噪声；
- timestep、dynamic CFG 和每一步 DiT 输出；
- VAE decode、crop-only 结果及 paste-back 前的生成帧。

阶段一明确不调试以下多窗口变量：

- 后续窗口起点和 stride；
- overlap 完整帧传播；
- overlap mask 清黑和 skip commit；
- long/short 两个 pass；
- bridge 构造和缩减；
- backward pass 与 global-index 跨 pass 归并；
- tail padding。

阶段二复用阶段一已经验证的单窗口执行内核，只增加序列规划和窗口间状态传递，不重新引入另一套 preprocessing、latent 或 decode 路径。

## 3. 请求契约

### 3.1 必须提供参考图

- SGLang VideoEdit 请求必须提供 `reference_image`。
- 无参考图请求直接报错。
- 不保留 legacy 无参考图推理分支，避免维护两套时序实现。

### 3.2 `ref_frame_idx`

- 新增整数请求字段 `ref_frame_idx`。
- 未传时默认 `0`。
- 在解码、同步截断 video/mask 后校验：

  ```text
  0 <= ref_frame_idx < 有效原视频帧数
  ```

- `ref_frame_idx` 表示 edited reference 对应的原视频帧位置，不是 prepend 后时间线的位置。

### 3.3 `bridge_overlap`

- 新增整数请求字段 `bridge_overlap`，默认 `5`。
- 请求值必须满足：

  ```text
  bridge_overlap >= 1
  (bridge_overlap - 1) % 4 == 0
  ```

  即只接受 `1、5、9、13...`。
- 非法请求直接报错，不静默取整。
- long pass 可用生成帧不足时，自动缩小为不超过可用数量的最大合法值。
- short pass 为空时不构造 bridge。

### 3.4 删除或固定的请求能力

以下字段不再保留为 VideoEdit 算法选择项：

| 字段 | 决策 |
|---|---|
| `chunks` | 完全排除，不进入正式契约 |
| `drop_reference_frame` | 删除；reference 天然不进入 global output |
| `keep_intermediate_windows` | 删除；当前无消费者，也不属于算法契约 |
| `overlap_commit_mode` | 删除；固定为 native skip |
| `tail_padding_mode` | 删除；固定为 reverse-mirror |
| `mask_downsample_mode` | 删除；固定为 `nearest` |
| `use_repaired_context` | 不再参与窗口语义；使用完整 overlap 直接传播 |
| `init_latent_mode` | 删除模式选择；固定纯噪声初始化 |
| `strength` | 固定为 `1.0` |
| `vary_seed_by_window` | 删除；每窗重置为同一请求 seed |

`save_crop_only` 所在输出模块保持当前实现；生产请求默认关闭 crop sidecar，
对齐测试命令显式开启。

## 4. 任意参考帧双 pass

设原视频共 `N` 帧，参考位置为 `k=ref_frame_idx`。

### 4.1 Reference out-of-band

- 全局预处理不得把 edited reference prepend 到原视频。
- 原视频全程保持原生 global index `0..N-1`。
- edited reference 单独保存，只在 pass 序列组装时加入。
- 最终输出不再依靠删除首帧修正时间线。

### 4.2 Long/short 方向规划

右侧长度大于或等于左侧时：

```text
long_idx  = [k, k+1, ..., N-1]
short_idx = [k-1, k-2, ..., 0]
```

左侧更长时：

```text
long_idx  = [k, k-1, ..., 0]
short_idx = [k+1, k+2, ..., N-1]
```

- 两侧相等时固定选择右侧作为 long pass。
- backward pass 通过反序输入实现，不切换模型。
- 每个 pass 和窗口继续使用局部连续时间位置，RoPE 从局部 0 开始；不引入 global `temporal_ids`。

### 4.3 Long pass 的 frame `k`

Long pass 开头必须为：

```text
frames = [edited_reference, source_frame_k, ...]
masks  = [全黑,             source_mask_k,  ...]
index  = [None,             k,              ...]
```

- 只有 edited reference 的 mask 清黑。
- 原视频 frame `k` 保留真实 mask并正常生成。
- 最终 index `k` 使用 source frame `k` 的生成结果，不直接输出 edited reference。

### 4.4 Bridge 构造

令实际 bridge 长度为 `b`：

```text
bridge = long_output[1 : 1+b][::-1]
```

- 从 local output 1 开始，排除 local 0 edited reference。
- 必须使用 long pass 生成结果，不使用原视频帧或 weighted accumulator。
- 取连续 `b` 帧后反序，使其方向与 short pass 一致。
- bridge mask 全黑。
- bridge global index 全为 `None`，不写入最终视频。

### 4.5 输出索引

- edited reference 和 bridge 的 `seq_idx=None`。
- 每个真实源视频帧携带其原始 global index。
- backward pass 生成结果仍按原始 index 写回。
- 使用 `generated_by_index[global_index]` 归并。
- 最终按 index 排序，输出严格为原视频 `0..N-1`，总帧数与有效输入帧数一致。

## 5. 滑窗实现

### 5.1 窗口起点

Long 和 short pass 使用同一规则：

```text
stride = infer_len - overlap
starts = 0, stride, 2*stride, ...
```

- 不再使用 `infer_len-overlap-1`。
- 不额外插入 `start-1` anchor。

### 5.2 完整 overlap 传播

相邻窗口之间：

```text
previous_output[stride : stride+overlap]
    -> next_window[0 : overlap]
```

- 下一窗口前 `overlap` 帧全部直接替换为上一窗口生成结果。
- 这 `overlap` 帧的 mask 全部清黑。
- 不只替换 local 0。
- 不从原视频或累计加权结果重新构造传播上下文。

### 5.3 输出提交

```text
首窗口：保留 [0 : valid_len]
后续窗：保留 [overlap : valid_len]
```

- 不对重叠帧做 weighted blending。
- 最终重复区域采用前一窗口版本。

### 5.4 Tail padding

固定使用 reverse-mirror：

```text
原尾部：..., N-2, N-1
padding：     N-1, N-2, N-3, ...
```

- 第一帧 padding 重复末帧。
- padding 帧只补足模型窗口，不进入最终输出。

### 5.5 明确保留的 `overlap=0` 行为

SGLang 多窗口时允许 `overlap=0` 的当前行为不修改。该项不按算法仓“多窗口必须有 overlap”的校验收紧，属于明确保留的工程差异。

## 6. 图像与 mask 预处理

### 6.1 Bbox

外部请求继续使用 SGLang 的每边扩张比例 `bbox_expand_scale=s`，内部转换为：

```text
algorithm_multiplier = 2*s + 1
```

内部 bbox 几何必须对齐算法仓：

- 根据 multiplier 计算目标宽、高和目标面积。
- bbox 碰到图像边界时优先平移，而不是直接损失尺寸。
- 某一维受全图边界限制时，扩大另一维以尽量保持目标面积。
- 最终再 clamp 到图像范围。

默认 `bbox_expand_scale` 明确保留为 `0.3`，对应内部 multiplier `1.6`；不改为算法主脚本默认的 multiplier `2.0`。

Union bbox、短边不足 480 时的扩张、16 对齐等当前已基本一致的逻辑保持不变。

### 6.2 CLIP 预处理

- VideoEdit 支持通过 `clip_preprocess` 选择 `diffuser` 或 `diffsynth` 两种预处理方式。
- `diffuser` 使用 checkpoint 对应的 Hugging Face `CLIPImageProcessor`，并与算法仓默认值对齐。
- `diffsynth` 使用与 DiffSynth 链路一致的手工预处理：归一化、bicubic 缩放到 224×224，再应用 CLIP mean/std 标准化。
- `clip_preprocess` 默认值为 `diffuser`；其他值在请求校验阶段直接报错。
- 继续取 CLIP `hidden_states[-2]`。
- CLI、API 和采样参数需要完整透传 `clip_preprocess`。
- `use_clip=false` 能力保持；专用模型类已支持 0 image token + 512 text token。

### 6.3 Mask 下采样

- 固定使用 `mode="nearest"`。
- 不允许请求切换为 `nearest-exact`。
- mask temporal packing、反转阈值和 `[B,4,F,H/8,W/8]` 布局保持现有已对齐实现。

## 7. 输入长度与默认值

### 7.1 Video/mask 长度

- video 和 mask 实际解码帧数必须完全相同，否则请求失败。
- 禁止使用 `min(video_len, mask_len)` 自动容错。
- 若用户显式指定正整数 `num_frames`，先验证原始输入等长，再对二者同步截断。
- 截断后校验 `ref_frame_idx`。

### 7.2 `num_frames`

- HTTP、CLI 和直调入口统一默认处理完整视频。
- 使用 `-1` 或 `None` 表示完整视频。
- 只有显式正整数才限制处理帧数。

### 7.3 `infer_len`

接受任意满足以下结构约束的值：

```text
infer_len >= 1
(infer_len - 1) % 4 == 0
```

- 不再只允许 `{49, 81}`。
- 服务可额外做资源上限校验，但不得把 49/81 描述为模型结构限制。

### 7.4 默认采样参数

| 参数 | 最终默认值 | 说明 |
|---|---:|---|
| `num_frames` | `-1` / `None` | 默认处理完整视频；`env.md` 显式使用 48 帧基线 |
| `ref_frame_idx` | `0` | 服务决策；不同于算法演示脚本默认 70 |
| `bridge_overlap` | `5` | 与算法一致 |
| `infer_len` | `49` | 与算法主入口一致 |
| `overlap` | `5` | 与算法主入口一致 |
| `num_inference_steps` | `40` | 已一致 |
| `guidance_scale` | `5.0` | 已一致 |
| `seed` | `42` | 已一致 |
| `dynamic_cfg` | `true` | 已一致 |
| `bbox_expand_scale` | `0.3` | 明确保留 SGLang 默认；内部倍率 1.6 |
| `dilate_px` | `8` | 与对齐命令一致 |
| `feather_px` | `8` | 与对齐命令一致 |
| `adain_boundary_dilate` | `0` | 默认 no-op |
| `save_crop_only` | `false` | 生产默认只产出主视频；对齐测试显式开启 crop sidecar |
| `decode_mode` | `stream` | 生产默认降低输入帧主机内存，并启用预取 |
| `enable_teacache` | `true` | 常规推理默认加速；对齐基线显式关闭 |

## 8. 随机数、latent 与缓存

### 8.1 初始 latent

严格按算法固定：

```text
strength = 1.0
video_latents = None
initial_latents = pure noise
```

- 不保留 `add_noise` 分支。
- 原视频 VAE latent 只作为 conditioning，不作为 diffusion 初始状态。

### 8.2 随机数

- 使用 CPU float32 generator。
- 每个窗口重新执行同一个请求 seed 的 `manual_seed(seed)`。
- 不按 pass 或 window index 改变 seed。

### 8.3 TeaCache

- VideoEdit 默认开启 TeaCache。
- Golden test 和严格算法对齐模式必须通过请求参数显式关闭。
- 开启 TeaCache 的常规推理属于非严格对齐模式。

### 8.4 Cache-DiT

Cache-DiT 及其他 SGLang 基础设施缓存保持当前服务契约，本轮不修改，也不纳入 VideoEdit 算法对齐范围。

## 9. AdaIN 与输出

### 9.1 Boundary AdaIN

当 `adain_boundary_dilate > 0` 时完整对齐算法仓：

- 在 edit mask interior 统计 generated RGB mean/std。
- 在 `dilate(mask)-mask` 外侧 ring 统计 original RGB mean/std。
- 按通道进行 AdaIN 校正。
- 使用模糊 boundary weight，只在边界附近混入校正结果。
- eager 与 stream/paste-back 必须复用同一实现。
- 参数为 `0` 时严格 no-op。

不引入另一套 color-correction 算法。

### 9.2 输出 FPS

- 编码时保留源视频浮点 FPS，不四舍五入为整数。
- 输出 metadata 与实际编码 FPS 使用相同值。

### 9.3 本轮不修改的输出模块

`save_crop_only` 当前实际行为是额外保存 crop sidecar而不是只保存 crop。写盘实现不变，
但生产默认关闭；需要 crop 验收对象的测试命令必须显式开启。

## 10. 模型、scheduler 与 VAE

### 10.1 模型类选择已验证通过

当前实际 transformer：

```text
/mnt/shanhai-ai/shanhai-workspace/zhouhao6/
video_diffusers/pretrain_models/VideoEdit-diffusers-model/
step_46500/transformer
```

其 `config.json` 已设置：

```json
"_class_name": "WanVideoEditTransformer3DModel"
```

已完成只读验证：

- SGLang registry 实际解析到 `WanVideoEditTransformer3DModel`。
- 40 个 block 的 `attn2` 全部替换为 `WanVideoEditCrossAttention`。
- checkpoint 与 SGLang 模型映射后均为 1303 个权重键。
- missing、extra、shape mismatch 均为 0。
- `use_clip=true` 时拆分为 257 image + 512 text。
- `use_clip=false` 时拆分为 0 image + 512 text。

因此模型类选择不再列为当前未对齐项。

运行时必须显式把 transformer component 指向上述 `step_46500/transformer`；基础模型目录当前没有可用的 `transformer/`，不能依靠默认拼接路径。

### 10.2 量化路径

当前对齐基线是 BF16。按层量化存在的 cross-attention prefix 条件性风险只作为独立备注，不纳入本轮实施待办。

### 10.3 Scheduler

当前双方都使用 shift 5、sigma min 0、extra one step 的 FlowMatch Euler 语义，当前默认已对齐，不需要修改。

### 10.4 VAE

静态契约已基本对齐：

- posterior `mode()`；
- 16 通道 mean/std normalize；
- 时间压缩 4、空间压缩 8；
- decode 范围转换；
- tiling tile 256、stride 192。

两边实现载体和 tiled feature-cache 行为不同，源码审查不足以证明逐像素一致。因此：

- 当前不把 VAE 判定为已确认差异。
- 当前不修改 VAE。
- 将 VAE 列为 numerical golden test 验证项。

## 11. 明确保留或排除的差异

| 项目 | 最终决策 |
|---|---|
| 无参考图 VideoEdit | 不保留；reference 必填 |
| `chunks` | 完全不考虑 |
| 多窗口 `overlap=0` | 保持 SGLang 当前行为 |
| bbox 外部参数单位 | 保留 SGLang 每边比例，内部换算 |
| bbox 默认值 | 保留 `0.3` |
| TeaCache | 常规推理默认开；strict/golden 显式关闭 |
| Cache-DiT | 保持当前契约，不处理 |
| 量化路径 | 不纳入本轮 |
| `save_crop_only` 输出模块 | 写盘实现不修改；生产默认关闭，测试显式开启 |
| VAE native/diffusers | 先做数值验证，不预判修改 |
| 输出 codec/bitrate/sidecar 命名 | 工程差异，不纳入算法对齐 |
| camelCase alias 覆盖范围 | 接口工程问题，不纳入本轮算法实现 |

## 12. 已确认对齐、无需修改的内核

以下项目源码契约已对齐：

- DiT `in_channels=36`、`out_channels=16`。
- 输入拼接顺序 `[noise16, cond_mask4, cond_latent16]`。
- mask temporal packing 和 latent 帧数公式。
- 默认 CPU float32 noise 基础路径。
- FlowMatch scheduler 数学与当前参数。
- dynamic CFG 公式和 cond/uncond 组合。
- T5 清理、最大长度 512 和负向 prompt主路径。
- CLIP 使用 `hidden_states[-2]`；只需修改 pixel preprocessing。
- VAE posterior mode及基础 normalize/decode 契约。
- 当前 BF16 VideoEdit 专用 transformer 类和权重映射。
- 当前局部连续 RoPE 行为；不传 `temporal_ids` 与算法一致。

## 13. 验收要求

### 13.1 阶段一：单窗口验收

阶段一使用第 2.1 节固定的 48 个源视频帧加 1 个 edited reference，必须先建立可重复运行的单窗口 comparison loop。两端锁定：

```text
同一 step_46500 BF16 checkpoint
同一输入 video/mask/reference/prompt
ref_frame_idx=0
source_frames=48
infer_len=49
overlap=0
num_inference_steps、CFG、seed 完全相同
TeaCache=false
其他 DiT cache=false
CPU float32 generator
每个窗口使用同一请求 seed
纯噪声初始化，strength=1
mask downsample=nearest
同一种 clip_preprocess
同一种 VAE tiling 设置
```

单窗口对齐不能只比较完整 paste-back MP4。必须按以下顺序比较边界产物，并在第一个不一致处停止向后归因：

| 顺序 | 对比边界 | 最低检查内容 |
|---:|---|---|
| 1 | 全局几何 | bbox 坐标、crop 宽高、aligned 宽高必须完全相同 |
| 2 | 像素输入 | resized reference、window video、window mask 的 shape、dtype、min/max/mean 和误差 |
| 3 | CLIP 输入与输出 | `pixel_values`、`hidden_states[-2]` |
| 4 | mask condition | `mask_video_tensor`、packed `cond_masks` |
| 5 | VAE condition | normalized `cond_latents` |
| 6 | 初始状态 | CPU noise、initial latent、timesteps |
| 7 | DiT | 第 0 步 cond/uncond `noise_pred`，随后再检查全部 step |
| 8 | 最终 latent | denoise 完成后的 latent |
| 9 | decode | VAE float output、uint8 crop frames |
| 10 | 输出 | crop-only 是主验收；paste-back 和完整编码 MP4 只作次级回归 |

阶段一当前已知、必须先消除的差异：

1. SGLang 必须真实执行请求选择的 `clip_preprocess`，不能在 stage 内固定为 DiffSynth 手工预处理。
2. 算法仓使用 `nearest` 时，SGLang 不得使用 `nearest-exact`。
3. bbox 参数换算后，两端必须得到完全相同的 bbox 坐标和 crop 尺寸；不能接受 1 像素偏差。
4. 两端必须确认加载同一 Transformer、VAE、text encoder、image encoder 和 tokenizer 权重，而不只比较模型目录名称。
5. CPU generator、noise dtype、noise shape 和 seed 重置位置必须一致。

当前红灯基线记录于：

```text
outputs/case0008_compare_report.json
```

已观察到的早期红灯基线应先看 crop：

```text
crop-only：SSIM mean 约 0.815，MAE mean 约 34.70
算法仓 crop-only：1778x747
SGLang crop-only：1778x748
完整视频（辅助）：SSIM mean 约 0.898，MAE mean 约 22.36
```

crop-only 差异大于 paste-back 后的完整视频差异，说明未编辑背景会稀释真实误差，当前首要问题位于模型输入或窗口内部推理，而不是 paste-back。该结论作为阶段一定位起点；每次修正一个变量后都应重新运行同一 crop comparison loop，禁止一次同时修改 CLIP、mask 和 bbox 后仅观察完整 MP4。

阶段一通过条件：

- 上述 1–8 级中间边界达到预先记录的数值容差；离散 mask 和索引必须完全相同。
- crop-only 不再出现当前 `1778×747` 与 `1778×748` 的几何差异。
- 48 个输出帧数量和顺序一致。
- crop-only comparison 通过约定阈值；这是算法 numerical golden 的主门禁。
- 完整 paste-back 视频仍需记录几何和指标，但只作为辅助回归，不用其较高分数覆盖 crop 失败。
- 同一命令至少连续运行两次，结论稳定。

阶段一未通过时，不开始实现或调试阶段二。

### 13.2 阶段二：多窗口验收

阶段二必须直接复用阶段一已经通过的窗口内核。先按从简单到复杂的顺序扩展：

1. `ref_frame_idx=0`、两个窗口、正向单 pass，验证完整 overlap 传播。
2. `ref_frame_idx=0`、三个及以上窗口，验证重复传播和 tail padding。
3. 任意参考帧但 short 为空或极短，验证 long pass。
4. long/short 均非空且正向 long pass，验证 bridge 和 global index。
5. backward long pass，验证逆序输入和正向索引恢复。
6. 首帧、中间帧、末帧以及不同长度视频的完整矩阵。

多窗口必须检查：

- 窗口起点严格为 `0, stride, 2*stride, ...`。
- 下一窗口 `0:overlap` 与上一窗口输出 `stride:stride+overlap` 像素完全相同。
- overlap mask 全黑。
- 后续窗口只提交 `[overlap:valid_len]`。
- bridge 只来自 long pass 连续生成结果，顺序反转正确。
- reference、bridge 和 padding 的 global index 为 `None`，不得进入输出。
- 最终 `generated_by_index` 覆盖所有有效源视频索引，且没有重复或缺失。
- 每个窗口内部的中间 tensor 仍满足阶段一容差，不能因多窗口封装而回归。
- 最终 numerical golden 仍以按 global index 排序后的 crop-only 48 帧为主；完整 paste-back 视频只作辅助检查。

### 13.3 单元与契约测试

至少覆盖：

1. `ref_frame_idx` 为 0、中间帧、末帧，以及越界错误。
2. 左侧更长、右侧更长、两侧相等时的 long/short 规划。
3. `bridge_overlap` 为 1/5/9、非法值、可用帧不足和 short 为空。
4. edited reference、bridge、source frame `k` 的 mask 与 global index。
5. backward pass 结果按 global index 恢复为正向输出。
6. 多窗口完整 overlap 回填、mask 清黑和后窗 skip。
7. reverse-mirror 第一帧 padding 重复末帧。
8. bbox 位于中心、四边和四角时的平移与面积补偿。
9. video/mask 不等长请求失败。
10. 任意合法 `infer_len=4n+1` 及非法窗口长度。
11. reference 缺失请求失败。
12. 默认完整视频与显式 `num_frames` 同步截断。
13. 输出帧数等于有效原视频帧数，输出 FPS 保留浮点值。

### 13.4 算法 golden test

严格对比时必须显式锁定：

```text
同一 step_46500 BF16 checkpoint
同一输入 video/mask/reference/prompt
阶段一固定 ref_frame_idx=0、overlap=0；阶段二再锁定相同 ref_frame_idx 和 bridge_overlap
同一 infer_len、overlap、steps、CFG、seed
TeaCache=false
其他 DiT cache=false
CLIPImageProcessor diffuser preprocessing
CPU float32 generator，每窗相同 seed
纯噪声初始化，strength=1
nearest mask downsample
主比较对象为两端 crop-only 视频；完整 paste-back 视频仅作辅助
```

由于两端 bbox 参数单位不同，测试必须显式换算：

```text
algorithm_multiplier = 2*sglang_bbox_expand_scale + 1
```

### 13.5 VAE numerical test

使用相同输入 tensor/latent和相同 BF16 VAE checkpoint，分别比较：

- condition encode latent 的 max/mean absolute error；
- 非 tiled decode误差；
- tiled decode误差；
- 转换为 uint8 后的像素差分布；
- 最终单窗口、单步或少步 pipeline 的 crop 输出差异；paste-back 输出只作辅助。

在得到数值证据前，不因 native/diffusers 实现载体不同而直接要求重写 VAE。

## 14. 实施优先级

### 14.1 阶段一：先完成单窗口对齐

按第一个数值分叉点驱动实施：

1. 固定 48 源帧加 1 reference 的单窗口 comparison loop，并保存当前红灯基线。
2. 对齐 bbox 算法和参数换算，确保 bbox、crop、aligned geometry 完全一致。
3. 增加并透传 `clip_preprocess`，分别支持 `diffuser` 和 `diffsynth`；golden test 每次只选择同一种模式。
4. 固定对齐基线的 mask downsample 为 `nearest`，比较 packed `cond_masks`。
5. 比较 reference/window pixel tensor 和 CLIP `pixel_values`、image embeddings。
6. 比较 VAE condition latents，必要时再细分 tiling、dtype 和 normalize 边界。
7. 固定 CPU float32 noise、纯噪声初始化、timesteps 和 dynamic CFG。
8. 比较第 0 步及后续每步 DiT 输出，定位模型执行层的首个数值分叉。
9. 比较最终 latent、VAE decode 和 crop-only，以 paste-back 作为次级回归。
10. 运行阶段一 golden test，连续两次稳定通过后冻结窗口内核。

阶段一不实现 long/short、bridge、完整 overlap 传播或 weighted/native 迁移，除非它们是移除单窗口 legacy prepend/drop-first 所必需的最小数据结构调整。

### 14.2 阶段二：再完成多窗口对齐

阶段一通过后按以下顺序实施：

1. reference 必填及 out-of-band 全局数据结构。
2. native stride、完整 overlap 传播、mask 清黑和 skip commit。
3. reverse-mirror tail padding。
4. long/short 规划和 source frame `k` 的真实 mask。
5. bridge 构造、合法长度缩减和 conditioning-only 标记。
6. backward pass 及 global-index 输出归并。
7. 输入长度校验、任意 `4n+1` infer length和最终默认参数。
8. 多窗口进度、stream decode、offload 和缓存状态隔离回归。
9. 多窗口端到端 golden test。

### 14.3 后置工程项

以下工作不能阻塞阶段一窗口内核对齐，可在阶段二稳定后处理：

1. AdaIN eager/stream 共用实现。
2. TeaCache 显式 opt-in 回归。
3. Cache-DiT、量化和性能优化回归。
4. 输出 codec、bitrate 和 sidecar 命名等工程差异。
