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


python3 infer.py --video_path /mnt/shanhai-ai/shanhai-workspace/fanruidi/projects/VideoEdit-new/DiffSynth-Studio/test_videos/pexel_test_data_0410/videos/20729655-hd_1920_1080_25fps_short.mp4 --mask_path /mnt/shanhai-ai/shanhai-workspace/fanruidi/projects/VideoEdit-new/DiffSynth-Studio/test_videos/pexel_test_data_0410/masks/20729655-hd_1920_1080_25fps_No_bbox_mask.mp4 --output_name 20729655-hd_1920_1080_25fps_No_bbox_mask.mp4 --prompt "A person stands on a bridge by a river, facing city buildings, spreading arms wide against a cloudy sky." --num_frames 81

export HTTP_PROXY="http://localhost:10909"
export HTTPS_PROXY="http://localhost:10909"
export ALL_PROXY="http://localhost:10909"
export PATH="/opt/node-v22/bin:$PATH" && codex --full-auto --add-dir /root/zhouhao6/video_diffusers/pretrain_models --add-dir /root/zhouhao6/VideoEdit-diffusers

UV_HTTP_TIMEOUT=1800 uv pip install \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://pypi.org/simple \
  -e "python[diffusion]" --prerelease=allow

  UV_HTTP_TIMEOUT=1800 uv pip install   --index-url https://pypi.tuna.tsinghua.edu.cn/simple   -e "python[diffusion]" --prerelease=allow



  FlowMatchScheduler是否是一样的？

  根据方案 @sglang/docs_always/add_new_mode/add_videoedit_diffusers/README.md，以及原始算法仓库 @/root/zhouhao6/VideoEdit-diffusers，完善/root/zhouhao6/video_diffusers/
  pretrain_models/Wan2.1-I2V-14B-480P-Diffusers，要求不新增文件，不要删除文件