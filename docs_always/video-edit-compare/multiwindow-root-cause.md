# 多窗口数值验收根因调查（2026-09-24）

本轮为诊断，不修改生产实现、默认配置或现有环境；诊断脚本及产物均在 sglang 仓库内。

## 复现与控制变量

- case0008，110 帧，2 步，参考帧 20，infer_len=49、overlap=5、bridge=5、seed=42，tight bbox，无 TeaCache。
- 冻结原始仓源码到 `outputs/videoedit-root-cause-20260924/reference_snapshot`，避免外部继续更新源码影响对照。
- 源码归档 SHA256：`c240a656b23d0759179d1eb0ab259aea418ac9ed3b7148eb2fa7f9ee71a8b5c4`。
- 原始运行时：Torch 2.6.0+cu124 / cuDNN 90100；指定 VE_PACKAGES 运行时：Torch 2.9.1+cu128 / cuDNN 91002。没有安装、升级依赖。
- 冻结后重新运行，crop 指标精确复现此前失败：SSIM mean=0.9701314948、min=0.9427945426；62 帧失败，恰为 0..19、68..109。MSE/MAE 均达标。
- 原先 48 帧单窗口测试是 40 步/ref=0，不是多窗口 2 步/ref=20 的严格消融对照；本次改为在同一次 110 帧运行内比较三个窗口。

## 已证实的差异来源

### 1. 编码器跨运行时数值差异

第一窗口的 RGB、mask、masked video tensor、cond_masks、文本 embedding、初始噪声、timestep 均逐元素相同。
进入 DiT 前，CLIP image embedding MAE=0.00993383117；VAE condition latent MAE=0.004989063367。

CLIP/VAE 权重文件及 CLIP processor 配置经 `cmp` 确认字节一致。
固定同一输入、同一份原始模型代码、同一权重，仅用两个已有运行时分别执行：

| 重放结果 | 原始 Torch 2.6 边界 | sglang Torch 2.9 边界 |
|---|---|---|
| 原始 CLIP / Torch 2.6 | 完全一致 | 不同 |
| 原始 CLIP / Torch 2.9 | 不同 | 完全一致 |
| 原始 VAE / Torch 2.6 | 完全一致 | 不同 |
| 原始 VAE / Torch 2.9 | 不同 | 完全一致 |

CLIP processor pixel_values 在两个运行时也完全一致。CLIP 比较时将 FP32 hidden states 转成实际送入 DiT 的 BF16。
因此这部分不是 decode、resize、mask 或权重不一致，而是运行时数值实现差异。
尚不能只凭此实验将责任归到 Torch、cuDNN 或某个 CUDA kernel 的具体版本。

### 2. 同运行时仍存在 DiT 实现差异

将冻结的原始算法放到指定现有 Torch 2.9 运行时，不改算法、不更换权重：

- 第一窗口首次 DiT 的全部张量输入逐元素一致。
- 首次 DiT 输出仍不同：MAE=0.00379958353、max_abs=0.078125。
- 完整 crop MP4 仍有 42 帧失败（68..109），SSIM mean=0.9719893927、min=0.9539877859。

所以“统一运行环境”不是充分修复。已有 `strict_videoedit_math` 仅在诊断进程中开启时，首次 DiT 输出 MAE 降到 0.00331963529，但仍不相等，也不是一键解决方案。

#### 进一步定位：融合 LayerNorm 提前 BF16 截断

第一层内部采样（每个大张量取前 262144 个元素）显示：patch embedding、time/text/image condition embedding 均相同；第一层 self-attention 的 Q/K/V 已不同。

原始 `models/transformer_wan.py:508`：

```python
(norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
```

sglang 默认 `strict_videoedit_math=False`，走 `WanTransformerBlock.forward` → `LayerNormScaleShift.forward_cuda` → `fused_norm_scale_shift`。
`python/sglang/jit_kernel/diffusion/cutedsl/common/norm_fusion.py` 的 `apply_layernorm_cta` 将归一化结果先写回输入同类型的寄存器张量（BF16），然后外层 kernel 才计算 scale/shift。

因此两者实际上是：

```text
原始：FP32 LayerNorm → FP32 scale/shift → BF16
融合：FP32 归约 → LayerNorm 结果转 BF16 → scale/shift → BF16
```

用捕获的原始 norm 输出、同一 timestep modulation 和 checkpoint scale_shift_table 重放前 50 个完整 token，共 256000 个元素：

| 重放运算 | 相对 sglang 实际 norm 输出的不同元素 | MAE | max_abs |
|---|---:|---:|---:|
| 原始 FP32 顺序 | 83509 / 256000 | 0.0006663823 | 0.0625 |
| 只加入提前 BF16 截断 | **0 / 256000** | **0** | **0** |

这是已经精确验证的一处代码级差异，不是泛泛猜测“浮点误差”。它不是全部差异的唯一来源：同运行时 strict 完整多窗口实验仍有 42 帧失败，SSIM mean=0.9748239013、min=0.9561944277。不能宣称只打开 strict 就已修好；其它 DiT 数值差异仍需逐项定位。

探针运行在第一层完成后故意抛出 `Diagnostic first-block capture complete (intentional stop)`，其退出码 1 是诊断预期，不是新的推理故障。

### 3. carry/bridge 将已有生成误差变成下一窗口输入差异

帧号为原视频 0-based，下表不包含末尾镜像填充的输出：

| 窗口 | 输入 | 提交帧 | 两端输入 RGB 不同的局部位置 |
|---|---|---|---|
| W0 正向首窗 | 编辑参考图 + 原视频 20..67 | 20..67 | 无 |
| W1 正向续窗 | W0 回贴后 63..67 + 原视频 68..109 | 68..109 | 0..4，仅 carry |
| W2 反向首窗 | W0 回贴后 24..20 + 原视频 19..0 | 19..0 | 0..4 及 45..48，仅 bridge 与其镜像 |

依赖关系为 W0→W1、W0→W2，而不是 W0→W1→W2。carry/bridge 在内存中传递，没有 MP4 编解码回灌。
W1 补 2 帧；W2 补 24 帧，其中末 4 帧镜像到了 bridge。其它原视频输入帧完全一致，支持差异由生成结果反馈，而不是错帧或读错窗口。

跨运行时对照的编码前 generated RGB（只统计提交帧）：

| 窗口 | SSIM mean | SSIM min | <0.97 的帧数 |
|---|---:|---:|---:|
| W0 | 0.992739 | 0.991163 | 0/48 |
| W1 | 0.972486 | 0.970612 | 0/42 |
| W2 | 0.968945 | 0.964303 | 11/20 |

单窗口“通过”只说明差异低于阈值，不是输出相同。后续窗口把这些不同的生成帧重新作为条件，数值误差对后续输出的影响增大。

### 4. 最终编码差异进一步消耗 SSIM 余量

原始冻结实现使用 crop 面积比例缩放目标码率，以及 two-pass/qcomp=1.0；sglang 当前最终合并使用未按 crop 面积缩放的源码率与不同编码流程。

控制实验：取完全相同的 sglang generated RGB，构造相同的两个无损中间片段，分别通过原始与 sglang 最终合并/编码路径：

- 110 帧均通过；SSIM mean=0.9863200067、min=0.9834312019。
- 说明编码差异本身不是本例全部失败的充分原因，但不能当作像素一致处理。
- 实际跨运行时对照中，W1 编码前 42 帧全通过，编码后的 42 帧全失败；W2 编码前 11 帧失败，编码后 20 帧失败。
- SSIM 不可线性相加，不应把这几个差值直接相加进行归因。

## 证据与复现入口

产物根目录：`outputs/videoedit-root-cause-20260924/`。

- `cross-runtime-crop.json`：冻结源码后原始对照复现。
- `same-runtime-crop.json`：同运行时控制实验。
- `boundaries-cross-runtime.log`、`boundaries-same-runtime.log`：张量边界。
- `pixels-cross-runtime.jsonl`、`pixels-same-runtime.jsonl`：编码前 RGB / carry 位置。
- `encoding-control.log`：相同像素的两套编码控制实验。
- `encoders_torch26/`、`encoders_torch29/`：隔离编码器的结果。
- `reference29_probe/`、`sglang_probe/`：第一层内部采样（模块同名不保证语义相同，尤其 self-attention、norm1；须按运算边界映射）。
- `norm-rounding-control.jsonl`：提前截断的精确重放；`strict-crop.json`：strict 完整多窗口结果。

诊断入口（GPU 号需先确认空闲）：

```bash
VE_GPU=6 /opt/conda/bin/python -B scripts/videoedit_diagnose_run.py reference
VE_GPU=2 /opt/conda/bin/python -B scripts/videoedit_diagnose_run.py sglang
VE_GPU=6 /opt/conda/bin/python -B scripts/videoedit_diagnose_run.py reference29
VE_GPU=2 /opt/conda/bin/python -B scripts/videoedit_diagnose_run.py sglang_strict
VE_GPU=6 /opt/conda/bin/python -B scripts/videoedit_diagnose_run.py reference29_probe
VE_GPU=7 /opt/conda/bin/python -B scripts/videoedit_diagnose_run.py sglang_probe
PYTHONPATH=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/lib/python3.11/site-packages \
  /opt/conda/bin/python -B scripts/videoedit_diagnose_norm.py
```

诊断脚本读取验证脚本到内存后替换路径，不改原始仓代码或环境；strict 通过诊断进程 hook 开启，生产默认值不变。

## 修复方向（本轮未实施）

先以相同输入的 DiT 边界建立数值回归，定位并对齐实际运算顺序/精度，再验证 carry/bridge 反馈后的多窗口；编码前 RGB 与最终 MP4 分开验收。
不能通过放宽阈值、只检查全尺寸回贴结果或只看第一窗口来声明多窗口算法对齐。
