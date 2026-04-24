


docker exec -it sglang-dev zsh

source /opt/venv/bin/activate

sglang generate \
    --model-path /root/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b \
    --text-encoder-cpu-offload \
    --pin-cpu-memory \
    --prompt "A curious raccoon walking through an autumn forest, cinematic lighting" \
    --height 480 \
    --width 832 \
    --num-frames 81 \
    --fps 16 \
    --num-inference-steps 40 \
    --guidance-scale 4.5 \
    --seed 2024 \
    --save-output \
    --output-path outputs \
    --output-file-name "wan21_sp1.mp4"

[04-23 03:36:29] Running pipeline stages: ['InputValidationStage', 'TextEncodingStage', 'LatentPreparationStage', 'TimestepPreparationStage', 'DenoisingStage', 'DecodingStage']
[04-23 03:36:29] [InputValidationStage] started...
[04-23 03:36:29] [InputValidationStage] finished in 0.0000 seconds
[04-23 03:36:29] [TextEncodingStage] started...
[04-23 03:36:32] [TextEncodingStage] finished in 2.3317 seconds
[04-23 03:36:32] [LatentPreparationStage] started...
[04-23 03:36:32] [LatentPreparationStage] finished in 0.0010 seconds
[04-23 03:36:32] [TimestepPreparationStage] started...
[04-23 03:36:32] [TimestepPreparationStage] finished in 0.0002 seconds
[04-23 03:36:32] [DenoisingStage] started...
100%|██████████████████████████████████████████████████████████████████████| 40/40 [03:31<00:00,  5.28s/it]
[04-23 03:40:03] [DenoisingStage] average time per step: 5.2824 seconds
[04-23 03:40:03] [DenoisingStage] finished in 211.2998 seconds
[04-23 03:40:03] [DecodingStage] started...
[04-23 03:40:17] [DecodingStage] finished in 14.0261 seconds
[04-23 03:40:19] Output saved to outputs/wan21_sp1.mp4
[04-23 03:40:19] Pixel data generated successfully in 229.70 seconds
[04-23 03:40:19] Completed batch processing. Generated 1 outputs in 229.70 seconds


sglang generate \
    --model-path /root/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b \
    --num-gpus 2 \
    --ulysses-degree 2 \
    --text-encoder-cpu-offload \
    --pin-cpu-memory \
    --prompt "A curious raccoon walking through an autumn forest, cinematic lighting" \
    --height 480 \
    --width 832 \
    --num-frames 81 \
    --fps 16 \
    --num-inference-steps 40 \
    --guidance-scale 4.5 \
    --seed 2024 \
    --save-output \
    --output-path outputs \
    --output-file-name "wan21_sp2.mp4"

[04-23 03:24:47] Running pipeline stages: ['InputValidationStage', 'TextEncodingStage', 'LatentPreparationStage', 'TimestepPreparationStage', 'DenoisingStage', 'DecodingStage']
[04-23 03:24:47] [InputValidationStage] started...
[04-23 03:24:47] [InputValidationStage] finished in 0.0001 seconds
[04-23 03:24:47] [TextEncodingStage] started...
[04-23 03:24:50] [TextEncodingStage] finished in 3.4621 seconds
[04-23 03:24:50] [LatentPreparationStage] started...
[04-23 03:24:50] [LatentPreparationStage] finished in 0.0013 seconds
[04-23 03:24:50] [TimestepPreparationStage] started...
[04-23 03:24:50] [TimestepPreparationStage] finished in 0.0004 seconds
[04-23 03:24:50] [DenoisingStage] started...
100%|███████████████████████████████████████████████████| 40/40 [02:01<00:00,  3.03s/it]
[04-23 03:26:51] [DenoisingStage] average time per step: 3.0259 seconds
[04-23 03:26:51] [DenoisingStage] finished in 121.0393 seconds
[04-23 03:26:51] [DecodingStage] started...
[04-23 03:26:59] [DecodingStage] finished in 7.9133 seconds
[04-23 03:27:00] Output saved to outputs/wan21_sp2.mp4
[04-23 03:27:01] Pixel data generated successfully in 133.86 seconds
[04-23 03:27:01] Completed batch processing. Generated 1 outputs in 133.86 seconds


cd /root/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b


github.com-personal

git clone git@github.com-personal.com:shanhai-mgtv/VideoEdit-diffusers.git

使用@python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-add-model技能，将该视频编辑模型@../VideoEdit-diffusers新增到仓库中，先给出方案，方案写到@docs_zh/add_new_mode/add_videoedit_diffusers中