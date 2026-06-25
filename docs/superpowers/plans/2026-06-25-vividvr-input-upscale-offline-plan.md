# VividVR Offline Input Upscale 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 仅在本地 offline / benchmark 路径中，把原版 Vivid-VR 的 `upscale` 输入预缩放语义原生接回 `sglang`，并要求功能与 `/home/zhiheng/Vivid-VR` 中的原版 `upscale` 实现完全对齐，同时保持当前 `Phase C` / `Phase D` 已验收基线在默认配置下不回归。

**架构：** 在 `VividVRSamplingParams` 中增加 Vivid-VR 专属 `upscale` 字段，默认值固定为 `1.0`；在 `runtime.vividvr.preprocess` 中按原版规则对控制视频做输入 resize，再沿用现有 padding、`reference_video`、decode/postprocess 链路。只把该参数暴露给本地 runner 与 benchmark 工具，不进入 `serve` / FlowCut / callback 契约。

**技术栈：** Python、PyTorch、Pydantic dataclass-style sampling params、pytest、现有 Vivid-VR preprocess / pipeline / benchmark tools。

---

## 范围与约束

- 本计划只接入原版 `upscale`，不处理 `enable_upscaling` / `upscaling_scale`。
- `sglang` 中新接入的 `upscale` 必须作为原版能力迁移处理，不接受“名字一致但行为近似”的替代实现。
- `upscale` 语义必须与原版 `/home/zhiheng/Vivid-VR/VRDiT/inference.py` 对齐：
  - `0.0`：输入视频短边缩放到 `1024`
  - `1.0`：不缩放
  - 其他 `> 0`：按倍率做 bicubic resize
- 对齐要求不仅包括参数值含义，还包括生效阶段与后续尺寸语义：
  - 生效阶段必须是推理前输入控制视频预处理
  - 必须作用于 `reference_video`、`original_height`、`original_width` 这条后续链路
  - 不能替换成推理后输出超分
- 默认值必须是 `1.0`，以保护当前 `960x720` Phase C / D / E 基线。
- 本轮不改 `serve`、FlowCut、MinIO、callback、progress，也不改 `run_vividvr_phase_c_single.py` 的固定基线命令。

## 文件结构

- 修改 `python/sglang/multimodal_gen/configs/sample/vividvr.py`
  - 增加 `upscale` 字段与参数校验。
- 修改 `python/sglang/multimodal_gen/runtime/vividvr/preprocess.py`
  - 新增输入 resize helper，并在 `load_control_video(...)` 中接入。
- 修改 `python/sglang/multimodal_gen/tools/run_vividvr_inference.py`
  - 增加 CLI `--upscale`，传入 `VividVRSamplingParams`。
- 修改 `python/sglang/multimodal_gen/tools/run_vividvr_phase_d_long_video.py`
  - 增加 CLI `--upscale`，透传到 benchmark request。
- 修改 `python/sglang/multimodal_gen/test/unit/test_sampling_params.py`
  - 覆盖 `upscale` 默认值与非法值校验。
- 创建 `python/sglang/multimodal_gen/test/unit/test_vividvr_preprocess.py`
  - 覆盖 `upscale=0/1/2` 的输入 resize 语义。
- 修改 `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py`
  - 覆盖 `--upscale` CLI 解析和 dry-run/request 透传。

---

## 任务 1：补齐 VividVR 参数契约

**文件：**
- `python/sglang/multimodal_gen/configs/sample/vividvr.py`
- `python/sglang/multimodal_gen/test/unit/test_sampling_params.py`

- [ ] **步骤 1：新增 `upscale` 字段并校验**

实现要求：

```python
upscale: float = 1.0
```

校验规则：

- 允许 `0.0`
- 禁止负数
- 禁止 `nan` / `inf`
- 保持 `bool` 非法

建议补到 `_validate_vividvr()`：

```python
if (
    isinstance(self.upscale, bool)
    or not isinstance(self.upscale, (int, float))
    or not math.isfinite(float(self.upscale))
    or float(self.upscale) < 0.0
):
    raise ValueError(f"upscale must be a finite number >= 0, got {self.upscale!r}")
```

- [ ] **步骤 2：补单元测试**

测试点：

- `VividVRSamplingParams().upscale == 1.0`
- `upscale=0.0` 合法
- `upscale=2.0` 合法
- `upscale=-1.0` / `math.nan` / `math.inf` / `True` 非法

测试命令：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_sampling_params.py -q
```

---

## 任务 2：在 preprocess 接入原版输入预缩放

**文件：**
- `python/sglang/multimodal_gen/runtime/vividvr/preprocess.py`
- `python/sglang/multimodal_gen/test/unit/test_vividvr_preprocess.py`

- [ ] **步骤 1：新增 resize helper**

在 `preprocess.py` 中新增只处理控制视频 tensor 的 helper，例如：

```python
def _resize_control_video(video: torch.Tensor, upscale: float) -> torch.Tensor:
    if upscale == 1.0:
        return video
    height, width = int(video.shape[-2]), int(video.shape[-1])
    if upscale == 0.0:
        scale = 1024.0 / float(min(height, width))
    else:
        scale = float(upscale)
    resized = torch.nn.functional.interpolate(
        video,
        scale_factor=scale,
        mode="bicubic",
        align_corners=False,
    )
    return resized.clamp_(0.0, 1.0)
```

实现时要点：

- `video` 维度当前是 `(T, C, H, W)`，插值前需要视需要 reshape 成 `(T, C, H, W)` 可直接喂给 `F.interpolate`
- `0.0` 模式按短边到 `1024` 计算倍率
- `reference_video` 必须取 resize 后、padding 前的视频
- `original_height` / `original_width` 必须记录 resize 后尺寸
- `original_num_frames` / `num_padding_frames` 语义保持不变

- [ ] **步骤 2：让 `load_control_video(...)` 接受 `upscale`**

建议把签名改成：

```python
def load_control_video(video_path: str, *, upscale: float = 1.0) -> dict[str, object]:
```

接入顺序：

1. 解码得到原始帧
2. 转成 `(T, C, H, W)` float tensor
3. 按 `upscale` 做输入 resize
4. 保存 `reference_video`
5. 再做尾帧 padding

- [ ] **步骤 3：补 preprocess 单测**

新增 `test_vividvr_preprocess.py`，覆盖：

- `upscale=1.0` 时高宽不变
- `upscale=2.0` 时 `original_height` / `original_width` 翻倍
- `upscale=0.0` 时短边变为 `1024`
- `reference_video.shape[0] == original_num_frames`
- `video.shape[0] == reference_video.shape[0] + num_padding_frames`

测试可直接 monkeypatch `load_control_video_frames(...)` 返回两帧 `PIL.Image`，避免依赖真实视频文件。

测试命令：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_preprocess.py -q
```

---

## 任务 3：只把参数暴露给 offline / benchmark 工具

**文件：**
- `python/sglang/multimodal_gen/tools/run_vividvr_inference.py`
- `python/sglang/multimodal_gen/tools/run_vividvr_phase_d_long_video.py`
- `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py`

- [ ] **步骤 1：更新本地通用 runner**

在 `run_vividvr_inference.py`：

- `parse_args()` 新增：

```python
parser.add_argument(
    "--upscale",
    type=float,
    default=1.0,
    help="Original Vivid-VR input resize factor. 0.0 means short side to 1024.",
)
```

- `build_request(...)` 的 `request_kwargs` 新增：

```python
"upscale": args.upscale,
```

- 如果 `build_dry_run_payload(...)` 或 `build_runtime_config_snapshot(...)` 当前会回显请求摘要，则把 `upscale` 一并纳入，避免 benchmark 记录里丢参。

- [ ] **步骤 2：更新 Phase D benchmark runner**

在 `run_vividvr_phase_d_long_video.py`：

- 增加 `--upscale`，默认 `1.0`
- `make_request(...)` 的 `request_kwargs` 增加 `"upscale": args.upscale`

这一步只做透传，不改 preset；默认 preset 仍靠 `1.0` 保持现状。

- [ ] **步骤 3：补工具层测试**

在 `test_stage_e_vividvr_inference_tool.py` 增加：

- CLI 解析 `--upscale 0`
- `build_request(...)` 产出的 `params.upscale == 0.0`
- dry-run / runtime snapshot 中包含 `upscale` 时，断言字段值正确

测试命令：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py -q
```

---

## 任务 4：做最小有效验证并守住基线

**文件：**
- 不新增实现文件；只跑验证

- [ ] **步骤 1：跑针对性单测**

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_sampling_params.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_preprocess.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py -q
```

- [ ] **步骤 2：跑轻量 dry-run 回归**

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python \
  python/sglang/multimodal_gen/tools/run_vividvr_inference.py \
  --input-video /home/zhiheng/Vivid-VR/input/720p/test_video_960x720.mp4 \
  --prompt-file /home/zhiheng/Vivid-VR/input/720p/prompt.txt \
  --dry-run \
  --upscale 1.0
```

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python \
  python/sglang/multimodal_gen/tools/run_vividvr_inference.py \
  --input-video /home/zhiheng/Vivid-VR/input/720p/test_video_960x720.mp4 \
  --prompt-file /home/zhiheng/Vivid-VR/input/720p/prompt.txt \
  --dry-run \
  --upscale 0.0
```

预期：

- `upscale=1.0` 不改变当前默认请求形态
- `upscale=0.0` 能正常构造请求，不触发参数校验错误

- [ ] **步骤 3：决定是否追加重型验收**

本轮默认不直接跑重型 GPU 推理验收；只有当单测或 dry-run 暴露出尺寸链路风险，才再开 `tmux` 做 Phase C 单 clip 实跑。若需要实跑，必须用仓库标准 tmux 规则执行。
