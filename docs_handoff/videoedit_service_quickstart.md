# VideoEdit 服务使用说明

> 当前项目分支：`cos`（提交 `7cf82d27b`）

## 1. 重建容器脚本用法

```bash
RECREATE=1 bash /root/VideoEdit/sglang/scripts/start_videoedit_container.sh
```

- `RECREATE=1`：删除旧容器并重建。修改代码后必须用这个，确保加载新代码。
- 只重启、不重建现有容器：

```bash
RESTART_EXISTING=1 bash /root/VideoEdit/sglang/scripts/start_videoedit_container.sh
```

查看服务日志：

```bash
docker logs -f videoedit_reset
```

启动后确认服务就绪：

```bash
curl --noproxy '*' -sS http://127.0.0.1:30000/health
docker exec videoedit_reset bash scripts/videoedit_dual_service/status.sh
```

## 2. 本地案例

输入输出都是本机路径，`output_storage` 用 `local`：

```bash
curl --noproxy '*' -sS \
  -X POST http://127.0.0.1:30000/v1/videos/repairs \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id": "videoedit-normal-test-002",
    "model": "videoedit-normal",
    "timeout": -1,
    "prompt": "一个男人站在舞台中央演讲，背后有两排巨大的立体文字。",
    "video_input_path": "/root/VideoEdit/test/1080.mp4",
    "mask_input_path": "/root/VideoEdit/test/mask_1080_merged.mp4",
    "reference_image_url": "/root/VideoEdit/test/local.png",
    "output_storage": "local",
    "output_path": "/root/VideoEdit/test/output_normal_001.mp4",
    "num_frames": -1,
    "infer_len": 49,
    "overlap": 5,
    "num_inference_steps": 20,
    "guidance_scale": 5.0,
    "dynamic_cfg": true,
    "dynamic_cfg_max_step": 15,
    "seed": 42,
    "dtype": "bf16",
    "decode_mode": "stream",
    "enable_paste_back": true,
    "drop_reference_frame": true
  }' | python3 -m json.tool
```

返回 `"code": 0, "status": "queued"` 即受理成功，处理完成后视频写到 `output_path`。

## 3. 远程案例

输入用 URL，输出上传到 S3/MinIO。需要带 `minio_config`，`output_storage` 用 `s3`：

```bash
curl --noproxy '*' -sS \
  -X POST http://127.0.0.1:30000/v1/videos/repairs \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id": "videoedit-normal-remote-001",
    "model": "videoedit-normal",
    "timeout": -1,
    "prompt": "一个男人站在舞台中央演讲，背后有两排巨大的立体文字。",
    "video_url": "http://minio.example.com:9000/flowcut/input/1080.mp4",
    "mask_url": "http://minio.example.com:9000/flowcut/input/mask_1080.mp4",
    "reference_image_url": "http://minio.example.com:9000/flowcut/input/local.png",
    "minio_config": {
      "endpoint": "minio.example.com:9000",
      "bucket_name": "flowcut",
      "access_key": "admin",
      "secret_key": "your-secret-key",
      "secure": false,
      "region": "us-east-1"
    },
    "output_storage": "s3",
    "output_bucket": "flowcut",
    "output_object_key": "test/output/remote_001.mp4",
    "num_frames": -1,
    "infer_len": 49,
    "overlap": 5,
    "num_inference_steps": 20,
    "guidance_scale": 5.0,
    "dynamic_cfg": true,
    "dynamic_cfg_max_step": 15,
    "seed": 42,
    "dtype": "bf16",
    "decode_mode": "stream",
    "enable_paste_back": true,
    "drop_reference_frame": true
  }' | python3 -m json.tool
```

要点：

- `output_storage=s3` 时 `minio_config` 必填。
- `output_object_key` 指定输出在 bucket 中的路径；不传时自动生成 `YYYY/MM/DD/HHMMSS_{task_id}.mp4`。
- 接口也兼容驼峰写法（`videoUrl`、`maskUrl`、`minioConfig`、`outputObjectKey` 等）。

## 4. 更换模型路径

配置文件：`scripts/videoedit_dual_service/config.env`，修改以下三个字段：

| 字段 | 含义 |
| --- | --- |
| `BASE_MODEL` | 基础模型目录 |
| `NORMAL_TRANSFORMER` | normal 后端 transformer 目录 |
| `DMD_TRANSFORMER` | DMD 后端 transformer 目录 |

改完后用 `RECREATE=1` 重建容器生效。注意路径是容器内路径，宿主 `/root/VideoEdit` 已挂载进容器，模型放在 `/root/VideoEdit/model/...` 下即可。

## 5. 查看日志与请求详细信息

查看服务日志：

```bash
docker logs -f videoedit_reset
```

查看 normal 后端任务日志：

```bash
docker exec videoedit_reset tail -f /root/VideoEdit/tmp/sglang-videoedit-dual/logs/normal.log
```

### 开启请求详细信息（本地/远程请求审计）

默认关闭，需要给后端 `serve` 进程加上审计参数。修改 `scripts/videoedit_dual_service/start.sh`，在 `start_backend()` 里的 `sglang serve` 命令中追加：

```
--videoedit-request-log-dir /root/VideoEdit/tmp/sglang-videoedit-request-logs
```

可选：默认会对 minio 密钥等敏感字段脱敏（显示为 `***`）；需要记录明文再加：

```
--videoedit-request-log-sensitive-values
```

改完用 `RECREATE=1` 重建容器。之后每个请求都会在 `/root/VideoEdit/tmp/sglang-videoedit-request-logs` 下生成一个 `*.request.json` 审计文件，包含客户端 IP、完整请求体、校验后的请求以及状态（received / validated / rejected_invalid 等）。

查看已生成的请求审计文件：

```bash
docker exec videoedit_reset ls -lt /root/VideoEdit/tmp/sglang-videoedit-request-logs
```
