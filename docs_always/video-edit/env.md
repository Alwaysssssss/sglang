https://github.com/shanhai-mgtv/VideoEdit-diffusers
/mnt/shanhai-ai/shanhai-workspace/fanruidi/projects/VideoEdit-new/DiffSynth-Studio/test_videos/pexel_test_data_0410
/mnt/shanhai-ai/shanhai-workspace/fanruidi/projects/VideoEdit-new/DiffSynth-Studio/test_videos/pexel_test_data_0410/prompt.txt prompt直接在这个文件内拿匹配的就行

    g.add_argument(
        "--model_path",
        default="/mnt/shanhai-ai/shanhai-workspace/fanruidi/projects/VideoEdit-new/"
                "VideoEdit-diffusers/pretrain_models/Wan2.1-I2V-14B-480P-Diffusers/",
        help="Base Wan2.1 diffusers model directory",
    )
    g.add_argument(
        "--transformer_path",
        default="/mnt/shanhai-ai/shanhai-workspace/fanruidi/projects/VideoEdit-new/"
                "VideoEdit-diffusers/utils/wan_converted_step_9500/",
        help="Fine-tuned transformer checkpoint directory",
    )

python3 infer.py --video_path /mnt/shanhai-ai/shanhai-workspace/fanruidi/projects/VideoEdit-new/DiffSynth-Studio/test_videos/pexel_test_data_0410/videos/1144932-hd_1920_1080_30fps_short.mp4 --mask_path /mnt/shanhai-ai/shanhai-workspace/fanruidi/projects/VideoEdit-new/DiffSynth-Studio/test_videos/pexel_test_data_0410/masks/1144932-hd_1920_1080_30fps_No_bbox_mask.mp4 --output_name 1144932-hd_1920_1080_30fps_No_bbox_mask.mp4 --prompt "A vibrant pink flower with a yellow center remains the focal point against green foliage throughout the video." --num_frames 81