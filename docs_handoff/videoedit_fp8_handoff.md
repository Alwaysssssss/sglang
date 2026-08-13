# VideoEdit FP8 交接说明

## 基本信息

| 项目 | 值 |
|---|---|
| docker容器 | `tyx-codex` |
| 代码目录 | `/sgl-workspace/sglang` |
| 模型目录 | `/mnt/nas/models/DifusserEdit/pretrain_models/VideoEdit-diffusers-model` |
| Transformer | `/mnt/nas/models/DifusserEdit/pretrain_models/VideoEdit-diffusers-model/transformer` |
| 服务器 IP | `10.51.28.123` |
| 端口 | `30000` |
| 本机 API | `http://127.0.0.1:30000` |
| 远程 API | `http://10.51.28.123:30000` |

## 启动量化服务

```bash
cd /sgl-workspace/sglang

MODEL_PATH=/mnt/nas/models/DifusserEdit/pretrain_models/VideoEdit-diffusers-model
TRANSFORMER_PATH="$MODEL_PATH/transformer"
OUT_DIR=/root/VideoEdit/test/fp8_server_outputs
INPUT_DIR=/root/VideoEdit/test/fp8_server_inputs

mkdir -p "$OUT_DIR" "$INPUT_DIR"

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
  --output-path "$OUT_DIR" \
  --input-save-path "$INPUT_DIR" \
  --transformer-quantization fp8_dynamic \
  --transformer-fp8-gemm-backend triton \
  --transformer-fp8-fused-projections true \
  --videoedit-self-attention-backend sage_attn \
  --videoedit-cross-attention-backend fa
```

量化相关参数是：

```text
--transformer-quantization fp8_dynamic
--transformer-fp8-gemm-backend triton
--transformer-fp8-fused-projections true
--videoedit-self-attention-backend sage_attn
--videoedit-cross-attention-backend fa
```

## HTTP 请求

服务启动完成后，在另一个终端执行：

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
  "seed": 42,
  "dtype": "bf16",
  "decode_mode": "stream",
  "enable_paste_back": true,
  "drop_reference_frame": true,
  "dilate_px": 0,
  "mask_scale": 1.0,
  "feather_px": 0
}
JSON
```

查询进度：

```bash
curl --noproxy '*' -sS \
  "http://127.0.0.1:30000/v1/videos/$TASK_ID/progress" \
  | python3 -m json.tool
```

远程请求时，将 URL 中的 `127.0.0.1` 改为 `10.51.28.123`。

## CLI 直接运行

CLI 不需要先启动服务，适合本地功能和在线 FP8 量化链路验证：

```bash
cd /sgl-workspace/sglang

USE_TRITON_W8A8_FP8_KERNEL=1 CUDA_VISIBLE_DEVICES=0,1 \
python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path /mnt/nas/models/DifusserEdit/pretrain_models/VideoEdit-diffusers-model \
  --transformer-path /mnt/nas/models/DifusserEdit/pretrain_models/VideoEdit-diffusers-model/transformer \
  --transformer-quantization fp8_dynamic \
  --prompt "一个男人站在舞台中央演讲，背后有两排巨大的立体文字。" \
  --video-input-path /root/VideoEdit/test/1080.mp4 \
  --mask-input-path /root/VideoEdit/test/mask_1080_merged.mp4 \
  --reference-image-path /root/VideoEdit/test/local.png \
  --output-path /root/VideoEdit/test \
  --output-file-name output_normal_fp8_cli.mp4 \
  --num-frames -1 \
  --infer-len 81 \
  --overlap 10 \
  --num-inference-steps 40 \
  --guidance-scale 5.0 \
  --dynamic-cfg \
  --dynamic-cfg-max-step 15 \
  --seed 42 \
  --dtype bf16 \
  --decode-mode stream \
  --enable-paste-back \
  --drop-reference-frame \
  --dilate-px 0 \
  --mask-scale 1.0 \
  --feather-px 0 \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 2 \
  --ring-degree 1 \
  --no-dit-cpu-offload \
  --dit-layerwise-offload \
  --text-encoder-cpu-offload \
  --image-encoder-cpu-offload \
  --vae-cpu-offload
```

正式性能测试使用“量化服务 + HTTP 请求”。当前 CLI 未暴露融合投影和按 Self/Cross
分别指定 Attention backend 的参数，因此 CLI 结果不能直接与服务侧最终配置比较。

## A100 机器模型路径

| 项目 | 模型路径 |
|---|---|
| 完整模型 | `/home/tyx/workspace/video_diffusers` |
| 普通 40 步 Transformer | `/home/tyx/workspace/difusser-model/step-55000-diffusers-lh` |
| 蒸馏 4 步 Transformer | `/root/VideoEdit/model/DifusserEdit/merged_dit_lightx2v_lora_scale_1p0` |
