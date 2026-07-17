

拷贝/mnt/shanhai-ai/shanhai-workspace/zhouhao6中所有内容到/home/tyx/workspace, 并且把所有权限都给用户 tyx

sudo cp -r /mnt/shanhai-ai/shanhai-workspace/zhouhao6/ /home/tyx/workspace/
sudo chown -R tyx:tyx /home/tyx/workspace
sudo chmod -R 777 /home/tyx/workspace

cp -r /mnt/shanhai-ai/shanhai-workspace/jyutong/posecontrol/train5/models/wan_erase_wan2.2_ckpts/checkpoint-step00105000/ /home/tyx/workspace/difusser-model/wan_erase/


cp -r /mnt/shanhai-ai/shanhai-workspace/fanruidi/projects/VideoEdit-new/VideoEdit_diffusers/output_videos_jyt/merged_dit_lightx2v_lora_scale_1p0/ /home/tyx/workspace/difusser-model/