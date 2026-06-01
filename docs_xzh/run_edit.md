export MODEL_PATH=/mnt/nas/models/DifusserEdit/pretrain_models/VideoEdit-diffusers-model
export TRANSFORMER_PATH=$MODEL_PATH/transformer
export INPUT_SAVE_DIR=/tmp/sglang-videoedit-cloud-inputs
export CACHE_DIR=/tmp/sglang-cache

export CUDA_VISIBLE_DEVICES=0,1
export VIDEOEDIT_QUEUE_CAPACITY=1
export FLASHINFER_WORKSPACE_BASE=$CACHE_DIR/flashinfer
export XDG_CACHE_HOME=$CACHE_DIR/xdg
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$INPUT_SAVE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$XDG_CACHE_HOME" /tmp/videoedit-cloud-perf

ls -lh "$MODEL_PATH/model_index.json" "$TRANSFORMER_PATH/config.json"

export PYTHONPATH=/mnt/nas/xzh/project/VideoEdit/sglang/python:$PYTHONPATH
sglang serve \
  --model-type diffusion \
  --model-path "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 30000 \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 2 \
  --ring-degree 1 \
  --dit-cpu-offload true \
  --dit-layerwise-offload true \
  --text-encoder-cpu-offload true \
  --image-encoder-cpu-offload true \
  --vae-cpu-offload true \
  --warmup true \
  --warmup-steps 1 \
  --output-path "" \
  --input-save-path "$INPUT_SAVE_DIR" \
  --transformer-path "$TRANSFORMER_PATH"

curl --noproxy '*' -sS -X POST http://127.0.0.1:30042/v1/videos/repairs \
    -H 'Content-Type: application/json' \
    -d '{
      "video_input_path": "/mnt/nas/models/DifusserEdit/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4",
      "mask_input_path": "/mnt/nas/models/DifusserEdit/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4",
      "callback_url": "http://127.0.0.1:18080/videoedit/callback",
      "output_storage": "local",
      "output_path": "/mnt/nas/xzh/project/VideoEdit/sglang/output_results/15108907_3840_2160_50fps_api_sp1_no_offload_fa_156f_all_gpu0.mp4",
      "num_frames": 156,
      "infer_len": 81,
      "overlap": 0,
      "num_inference_steps": 20,
      "guidance_scale": 5.0,
      "dynamic_cfg": true,
      "dynamic_cfg_max_step": 15,
      "seed": 42,
      "dtype": "bf16",
      "decode_mode": "eager",
      "enable_paste_back": true,
      "drop_reference_frame": false,
      "perf_dump_path": "/mnt/nas/xzh/project/VideoEdit/sglang/output_results/videoedit_perf_api_sp1_no_offload_fa_156f_all_gpu0.json"
    }'
