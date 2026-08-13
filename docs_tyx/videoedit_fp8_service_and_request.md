# VideoEdit FP8 服务启动与请求命令

## 1. 路径和服务地址

| 项目 | 位置 |
|---|---|
| SGLang 代码目录 | `/sgl-workspace/sglang` |
| 完整模型目录 | `/mnt/nas/models/DifusserEdit/pretrain_models/VideoEdit-diffusers-model` |
| BF16 Transformer 目录 | `/mnt/nas/models/DifusserEdit/pretrain_models/VideoEdit-diffusers-model/transformer` |
| 输入视频 | `/root/VideoEdit/test/1080.mp4` |
| 输入 Mask | `/root/VideoEdit/test/mask_1080_merged.mp4` |
| 参考图片 | `/root/VideoEdit/test/local.png` |
| 服务监听地址 | `0.0.0.0:30000` |
| 服务器本机请求地址 | `http://127.0.0.1:30000` |
| 局域网请求地址 | `http://10.51.28.123:30000` |

`0.0.0.0` 表示服务监听服务器的所有网卡。请求在服务器本机执行时使用
`127.0.0.1`；从其他机器访问时使用 `10.51.28.123`，并确保安全组和防火墙允许
TCP `30000` 端口。

## 2. 启动双卡 FP8 量化服务

下面使用 BF16 Transformer 启动，并在模型加载阶段在线生成 FP8 权重。运行时 Linear
使用动态 FP8 W8A8 Triton GEMM，Self-Attention 使用 SageAttention。

```bash
cd /sgl-workspace/sglang

export MODEL_PATH=/mnt/nas/models/DifusserEdit/pretrain_models/VideoEdit-diffusers-model
export TRANSFORMER_PATH="$MODEL_PATH/transformer"
export OUT_DIR=/root/VideoEdit/test/sglang_fp8_server_outputs
export INPUT_DIR=/root/VideoEdit/test/sglang_fp8_server_inputs
export CACHE_DIR=/root/VideoEdit/test/sglang_cache

mkdir -p \
  "$OUT_DIR" \
  "$INPUT_DIR" \
  "$CACHE_DIR/triton" \
  "$CACHE_DIR/torchinductor" \
  "$CACHE_DIR/xdg" \
  "$CACHE_DIR/tmp"

export TRITON_CACHE_DIR="$CACHE_DIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torchinductor"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"
export TMPDIR="$CACHE_DIR/tmp"

CUDA_VISIBLE_DEVICES=0,1 sglang serve \
  --model-type diffusion \
  --model-path "$MODEL_PATH" \
  --transformer-path "$TRANSFORMER_PATH" \
  --host 0.0.0.0 \
  --port 30000 \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 2 \
  --ring-degree 1 \
  --dit-cpu-offload false \
  --dit-layerwise-offload true \
  --text-encoder-cpu-offload true \
  --image-encoder-cpu-offload true \
  --vae-cpu-offload true \
  --warmup true \
  --warmup-steps 1 \
  --output-path "$OUT_DIR" \
  --input-save-path "$INPUT_DIR" \
  --transformer-quantization fp8_dynamic \
  --transformer-fp8-gemm-backend triton \
  --transformer-fp8-fused-projections true \
  --videoedit-self-attention-backend sage_attn \
  --videoedit-cross-attention-backend fa
```

看到以下日志后再发送请求：

```text
Application startup complete.
Uvicorn running on http://0.0.0.0:30000
```

可以在另一个终端检查服务：

```bash
curl --noproxy '*' --fail http://127.0.0.1:30000/health
curl --noproxy '*' --fail http://127.0.0.1:30000/v1/models
```

## 3. 发送视频修复请求

该请求处理完整视频，使用 81 帧窗口、10 帧 overlap 和 40 个去噪 step。Mask 不膨胀、
不缩放、不羽化。

```bash
TASK_ID="videoedit-normal-fp8-$(date +%Y%m%d-%H%M%S)"

curl --noproxy '*' -sS \
  -X POST http://127.0.0.1:30000/v1/videos/repairs \
  -H 'Content-Type: application/json' \
  -d @- <<JSON | python3 -m json.tool
{
  "task_id": "$TASK_ID",
  "model": "videoedit-normal",
  "timeout": -1,
  "prompt": "一个男人站在舞台中央演讲，背后有两排巨大的立体文字。",
  "video_input_path": "/root/VideoEdit/test/1080.mp4",
  "mask_input_path": "/root/VideoEdit/test/mask_1080_merged.mp4",
  "reference_image_url": "/root/VideoEdit/test/local.png",
  "output_storage": "local",
  "output_path": "/root/VideoEdit/test/${TASK_ID}.mp4",
  "num_frames": -1,
  "infer_len": 81,
  "overlap": 10,
  "num_inference_steps": 40,
  "guidance_scale": 5.0,
  "dynamic_cfg": true,
  "dynamic_cfg_max_step": 15,
  "dynamic_cfg_min": 1.0,
  "seed": 42,
  "dtype": "bf16",
  "decode_mode": "stream",
  "enable_paste_back": true,
  "drop_reference_frame": true,
  "dilate_px": 0,
  "mask_scale": 1.0,
  "feather_px": 0,
  "enable_teacache": true
}
JSON
```

查询任务进度：

```bash
curl --noproxy '*' -sS \
  "http://127.0.0.1:30000/v1/videos/$TASK_ID/progress" \
  | python3 -m json.tool
```

从其他机器发送请求时，只需把 URL 中的 `127.0.0.1` 改为 `10.51.28.123`。

## 4. 参数说明

- `--transformer-quantization fp8_dynamic`：加载 BF16 权重后在线量化为 FP8。
- `--transformer-fp8-gemm-backend triton`：Linear 使用 Triton FP8 Scaled GEMM。
- `--transformer-fp8-fused-projections true`：启用 QKV/KV 投影融合。
- `--videoedit-self-attention-backend sage_attn`：Self-Attention 使用低精度 SageAttention。
- 请求中的 `dtype=bf16` 不会关闭 W8A8；它表示未量化算子和 Linear 输入仍以 BF16 进入量化路径。
- 请求中的 `model=videoedit-normal` 只是任务结果中的模型标签，不会切换服务实际加载的模型。
- `mask_scale=1.0` 表示保持原始 Mask 尺寸，不应设置为 `0`。
- 当前命令显式启用了 TeaCache。若要单独测量量化收益，应把
  `enable_teacache` 改为 `false`，并确保 BF16/FP8 两边使用完全相同的请求。

## 5. 离线 FP8 权重说明

如果服务器已经复制了离线 FP8 Transformer，可以把：

```bash
export TRANSFORMER_PATH=/sgl-workspace/sglang/videoedit_fp8_offline/transformer
```

并删除启动命令中的：

```text
--transformer-quantization fp8_dynamic
```

其余 Triton、融合投影和 Attention 参数保持不变。离线权重与在线量化使用相同的 forward
计算路径，主要区别是离线版本不需要在每次启动时重新量化权重。
