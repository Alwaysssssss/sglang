# VideoEdit 指定 S3/MinIO 输出路径修改方案

本文给出一个代码修改方案，使 `/v1/videos/repairs` 支持把输出视频上传到指定的 S3/MinIO bucket 路径，例如：

```text
localminio/flowcut/test/output/cloud_sp1_offload_81f.mp4
```

目标是让推理完成后，云端对象存储的指定 prefix 下出现输出视频，而不是只能上传到 bucket 根目录：

```text
localminio/flowcut/cloud_sp1_offload_81f.mp4
```

## 1. 当前行为和限制

当前代码已经支持 S3-compatible 上传，但上传 object key 被固定成输出文件 basename。

关键代码在 `python/sglang/multimodal_gen/runtime/entrypoints/openai/storage.py`：

```python
async def upload_and_cleanup(self, file_path: str) -> Optional[str]:
    ...
    key = os.path.basename(file_path)
    url = await self.upload_file(file_path, key)
```

因此即使你希望上传到：

```text
flowcut/test/output/cloud_sp1_offload_81f.mp4
```

当前也只会上传到：

```text
flowcut/cloud_sp1_offload_81f.mp4
```

`VideoRepairRequest` 里已经有字段：

```python
output_storage: str = "local"
output_path: Optional[str] = None
output_bucket: Optional[str] = None
output_object_key: Optional[str] = None
```

但当前 `/v1/videos/repairs` 没有把 `output_bucket` / `output_object_key` 传给 `cloud_storage.upload_and_cleanup()`。

## 2. 需要达到的效果

请求中指定：

```json
{
  "output_storage": "s3",
  "output_bucket": "flowcut",
  "output_object_key": "test/output/cloud_sp1_offload_81f.mp4"
}
```

推理完成后：

```bash
mc stat localminio/flowcut/test/output/cloud_sp1_offload_81f.mp4
```

能够查到对象。

`/progress` 返回：

```json
{
  "id": "cloud_sp1_offload_81f",
  "status": "completed",
  "progress": 100,
  "file_path": null,
  "url": "http://127.0.0.1:19000/flowcut/test/output/cloud_sp1_offload_81f.mp4",
  "error": null
}
```

注意：这里的 `test/output/` 是 S3/MinIO object key prefix。它在 MinIO Console、`mc ls`、S3 API 中会显示成目录层级，但它不是 Mac Finder 里的普通文件夹。对象存储的正确访问方式仍然是 `mc`、MinIO Console、HTTP URL 或 S3 SDK。

如果需求是“Mac 普通文件系统某个目录下直接出现 mp4 文件”，那不是 S3 上传语义，需要另做本地文件复制、NFS/共享目录、scp/rsync、或者一个专门的文件落盘服务。本文方案解决的是“指定 S3/MinIO bucket 内的对象路径”。

## 3. 推荐接口

推荐使用已有字段，不新增协议字段：

```json
{
  "output_storage": "s3",
  "output_bucket": "flowcut",
  "output_object_key": "test/output/cloud_sp1_offload_81f.mp4"
}
```

字段含义：

- `output_storage`: 设为 `s3` 时表示输出必须上传到 S3-compatible object storage。
- `output_bucket`: 可选。为空时使用环境变量 `SGLANG_S3_BUCKET_NAME`。
- `output_object_key`: 云端对象 key。支持包含 `/`，例如 `test/output/a.mp4`。

建议不要复用普通本地 `output_path` 来表达云端路径，避免和当前本地输出路径语义冲突。

可选兼容增强：后续也可以支持：

```json
{
  "output_storage": "s3",
  "output_path": "s3://flowcut/test/output/cloud_sp1_offload_81f.mp4"
}
```

但第一版推荐先接通 `output_bucket` / `output_object_key`，改动更小，语义更清晰。

## 4. 代码修改点

### 4.1 修改 `storage.py`

文件：

```text
python/sglang/multimodal_gen/runtime/entrypoints/openai/storage.py
```

目标：

- `upload_file()` 支持传入可选 `bucket_name`。
- `upload_and_cleanup()` 支持传入可选 `destination_key` 和 `bucket_name`。
- 如果没有传 `destination_key`，保持当前行为：使用 `os.path.basename(file_path)`。

建议修改成类似下面的结构：

```python
def _normalize_object_key(key: str) -> str:
    key = key.strip().lstrip("/")
    if not key:
        raise ValueError("S3 object key must not be empty")
    if any(part == ".." for part in key.split("/")):
        raise ValueError(f"S3 object key must not contain '..': {key}")
    return key


async def upload_file(
    self,
    local_path: str,
    destination_key: str,
    bucket_name: Optional[str] = None,
) -> Optional[str]:
    if not self.is_enabled():
        return None

    destination_key = _normalize_object_key(destination_key)
    target_bucket = bucket_name or self.bucket_name
    if not target_bucket:
        logger.error("Upload failed: S3 bucket is not configured")
        return None

    def _sync_upload():
        ext = os.path.splitext(local_path)[1].lower()
        content_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".mp4": "video/mp4",
            ".glb": "model/gltf-binary",
            ".obj": "text/plain",
        }.get(ext, "application/octet-stream")

        self.client.upload_file(
            local_path,
            target_bucket,
            destination_key,
            ExtraArgs={"ContentType": content_type},
        )

    try:
        await asyncio.get_running_loop().run_in_executor(None, _sync_upload)
    except Exception as e:
        logger.error(f"Upload failed for {target_bucket}/{destination_key}: {e}")
        return None

    if self.endpoint_url:
        url = f"{self.endpoint_url.rstrip('/')}/{target_bucket}/{destination_key}"
    else:
        region = self.region_name or "us-east-1"
        url = f"https://{target_bucket}.s3.{region}.amazonaws.com/{destination_key}"

    logger.info(f"Uploaded {local_path} to {url}")
    return url


async def upload_and_cleanup(
    self,
    file_path: str,
    destination_key: Optional[str] = None,
    bucket_name: Optional[str] = None,
) -> Optional[str]:
    if not self.is_enabled():
        return None

    key = destination_key or os.path.basename(file_path)
    url = await self.upload_file(file_path, key, bucket_name=bucket_name)

    if url:
        try:
            os.remove(file_path)
        except OSError as e:
            logger.warning(f"Failed to remove temporary file {file_path}: {e}")
    return url
```

### 4.2 修改 `video_api.py` 的 dispatch 参数

文件：

```text
python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py
```

给 `_dispatch_job_async()` 增加云端输出参数：

```python
async def _dispatch_job_async(
    job_id: str,
    batch: Req,
    *,
    temp_dirs: list[str] | None = None,
    output_persistent: bool = True,
    cloud_output_bucket: str | None = None,
    cloud_output_key: str | None = None,
    cloud_output_required: bool = False,
) -> None:
```

上传处改为：

```python
cloud_url = await cloud_storage.upload_and_cleanup(
    save_file_path,
    destination_key=cloud_output_key,
    bucket_name=cloud_output_bucket,
)

if cloud_output_required and not cloud_url:
    raise RuntimeError(
        f"Failed to upload output video to S3: "
        f"bucket={cloud_output_bucket or '<default>'}, key={cloud_output_key}"
    )
```

这样当请求明确要求 `output_storage=s3` 时，上传失败不会被误报成 `completed`。

同时给 `_dispatch_video_repair_job_async()` 透传这些参数：

```python
async def _dispatch_video_repair_job_async(
    job_id: str,
    batch: Req,
    *,
    temp_dirs: list[str] | None = None,
    output_persistent: bool = True,
    cloud_output_bucket: str | None = None,
    cloud_output_key: str | None = None,
    cloud_output_required: bool = False,
) -> None:
    try:
        await VIDEO_STORE.update_fields(job_id, {"status": "running", "progress": 1})
        await _dispatch_job_async(
            job_id,
            batch,
            temp_dirs=None,
            output_persistent=output_persistent,
            cloud_output_bucket=cloud_output_bucket,
            cloud_output_key=cloud_output_key,
            cloud_output_required=cloud_output_required,
        )
    finally:
        _VIDEOEDIT_SEMAPHORE.release()
        for td in temp_dirs or []:
            shutil.rmtree(td, ignore_errors=True)
```

### 4.3 在 `create_video_repair()` 里解析云端输出目标

在构造 `sampling_params` 前，解析 `req.output_storage`：

```python
output_storage = (req.output_storage or "local").lower()
cloud_output_required = output_storage == "s3"
cloud_output_bucket = req.output_bucket if cloud_output_required else None
cloud_output_key = req.output_object_key if cloud_output_required else None
```

如果用户指定 `output_storage=s3` 但没有指定 `output_object_key`，可以使用默认 key：

```python
if cloud_output_required and not cloud_output_key:
    cloud_output_key = f"{request_id}.mp4"
```

当使用云端输出时，建议不要依赖本地持久化路径。可以继续使用临时目录：

```python
local_output_path = None if cloud_output_required else req.output_path

output_dir, output_file_name = _split_output_path(
    local_output_path, request_id, server_args.output_path
)

if cloud_output_required and cloud_output_key:
    output_file_name = os.path.basename(cloud_output_key.rstrip("/")) or f"{request_id}.mp4"

output_persistent = output_dir is not None
if output_dir is None:
    output_dir = tempfile.mkdtemp(prefix="sglang_videoedit_output_")
    temp_dirs.append(output_dir)
    output_persistent = False
```

最后创建后台任务时透传：

```python
asyncio.create_task(
    _dispatch_video_repair_job_async(
        request_id,
        batch,
        temp_dirs=temp_dirs or None,
        output_persistent=output_persistent,
        cloud_output_bucket=cloud_output_bucket,
        cloud_output_key=cloud_output_key,
        cloud_output_required=cloud_output_required,
    )
)
```

## 5. 修改后如何使用

启动 serve 前仍然设置 S3/MinIO 环境变量：

```bash
export SGLANG_CLOUD_STORAGE_TYPE=s3
export SGLANG_S3_BUCKET_NAME=flowcut
export SGLANG_S3_ENDPOINT_URL='http://127.0.0.1:19000'
export SGLANG_S3_REGION_NAME=us-east-1
export SGLANG_S3_ACCESS_KEY_ID='你的 MinIO access key'
export SGLANG_S3_SECRET_ACCESS_KEY='你的 MinIO secret key'
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
```

建议启动前先跑上传探针：

```bash
python - <<'PY'
import os
import tempfile
import boto3

endpoint = os.environ["SGLANG_S3_ENDPOINT_URL"]
bucket = os.environ["SGLANG_S3_BUCKET_NAME"]
key = "test/output/sglang_s3_upload_probe.txt"

client = boto3.client(
    "s3",
    endpoint_url=endpoint,
    region_name=os.environ.get("SGLANG_S3_REGION_NAME") or "us-east-1",
    aws_access_key_id=os.environ["SGLANG_S3_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["SGLANG_S3_SECRET_ACCESS_KEY"],
)

path = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
with open(path, "w") as f:
    f.write("sglang s3 upload probe\n")

client.upload_file(path, bucket, key, ExtraArgs={"ContentType": "text/plain"})
print("upload probe ok:", f"{endpoint.rstrip('/')}/{bucket}/{key}")
PY
```

请求示例：

```python
import json
import os
import urllib.request

task_id = "cloud_sp1_offload_81f"
payload = {
    "task_id": task_id,
    "prompt": "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video.",
    "video_url": "http://127.0.0.1:19000/flowcut/test/video/15108907_3840_2160_50fps_short.mp4",
    "mask_url": "http://127.0.0.1:19000/flowcut/test/mask/15108907_3840_2160_50fps_No_bbox_mask.mp4",
    "output_storage": "s3",
    "output_bucket": "flowcut",
    "output_object_key": "test/output/cloud_sp1_offload_81f.mp4",
    "num_frames": 81,
    "infer_len": 81,
    "overlap": 0,
    "num_inference_steps": 20,
    "guidance_scale": 5.0,
    "dynamic_cfg": True,
    "dynamic_cfg_max_step": 15,
    "seed": 42,
    "dtype": "bf16",
    "enable_paste_back": True,
    "drop_reference_frame": True,
    "perf_dump_path": "/tmp/videoedit_perf_api_cloud_sp1_offload_81f.json",
}

req = urllib.request.Request(
    "http://127.0.0.1:30000/v1/videos/repairs",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=60) as resp:
    print(resp.status)
    print(resp.read().decode("utf-8"))
```

查询进度：

```bash
curl --noproxy '*' -s http://127.0.0.1:30000/v1/videos/cloud_sp1_offload_81f/progress
```

预期完成结果：

```json
{
  "id": "cloud_sp1_offload_81f",
  "status": "completed",
  "progress": 100,
  "file_path": null,
  "url": "http://127.0.0.1:19000/flowcut/test/output/cloud_sp1_offload_81f.mp4",
  "error": null
}
```

验证云端对象：

```bash
mc stat localminio/flowcut/test/output/cloud_sp1_offload_81f.mp4
mc cp localminio/flowcut/test/output/cloud_sp1_offload_81f.mp4 ./cloud_sp1_offload_81f.mp4
```

如果打开 MinIO Console，也应该能在 `flowcut` bucket 下看到 `test/output/cloud_sp1_offload_81f.mp4` 这个对象。

## 6. 效果说明

修改后可以实现：

```text
GPU 机器推理完成
  -> 临时生成 /tmp/sglang_videoedit_output_xxx/cloud_sp1_offload_81f.mp4
  -> 通过 S3 API 上传到 flowcut/test/output/cloud_sp1_offload_81f.mp4
  -> 上传成功后删除临时本地 mp4
  -> /progress 返回云端 URL
```

也就是说，云端 MinIO/S3 的指定 object key 会出现输出视频：

```text
localminio/flowcut/test/output/cloud_sp1_offload_81f.mp4
```

但它仍然是对象存储里的对象，不是 Mac 普通文件系统目录里的裸 mp4 文件。你能通过以下方式访问：

- `mc ls` / `mc stat` / `mc cp`
- MinIO Console
- HTTP URL，如果 bucket 或对象权限允许下载
- S3 SDK 或云厂商控制台

如果你要求的是 Mac 上某个普通目录直接出现：

```text
/Users/xxx/output/cloud_sp1_offload_81f.mp4
```

那不属于 S3/MinIO object key 语义，需要额外做文件系统写入方案。可选路线包括：

- 推理完成后由 Mac 端 `mc mirror` / `mc cp` 拉取到普通目录。
- 在 Mac 上写一个 watcher，监听 MinIO bucket 新对象，然后复制到普通目录。
- 不走 S3，新增一个 HTTP 文件上传接口，把 mp4 以 multipart 形式发给 Mac 端服务并由它写入本地目录。
- 使用 NFS/SMB/SSHFS 把 Mac 目录挂载到 GPU 机器，再把 `output_path` 指向挂载目录。

## 7. 测试建议

最小测试：

1. 启动 serve 前跑 S3 上传探针，确认 `test/output/sglang_s3_upload_probe.txt` 能上传。
2. 启动 serve，提交带 `output_storage=s3`、`output_bucket=flowcut`、`output_object_key=test/output/cloud_sp1_offload_81f.mp4` 的请求。
3. 等 `/progress` 为 `completed`。
4. 验证：

```bash
mc stat localminio/flowcut/test/output/cloud_sp1_offload_81f.mp4
```

5. 下载检查：

```bash
mc cp localminio/flowcut/test/output/cloud_sp1_offload_81f.mp4 /tmp/cloud_sp1_offload_81f.mp4
python - <<'PY'
import cv2
path = "/tmp/cloud_sp1_offload_81f.mp4"
cap = cv2.VideoCapture(path)
print({
    "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    "fps": cap.get(cv2.CAP_PROP_FPS),
})
cap.release()
PY
```

单元测试建议：

- `storage.py`：验证 `upload_and_cleanup(file_path, destination_key="test/output/a.mp4", bucket_name="flowcut")` 调用底层 client 时 bucket/key 正确。
- `storage.py`：验证不传 `destination_key` 时仍保持旧行为，使用 basename。
- `video_api.py`：mock `cloud_storage.upload_and_cleanup()`，验证 `VideoRepairRequest.output_object_key` 被传入。
- 错误路径：当 `output_storage=s3` 且上传失败时，任务应进入 `failed`，不能静默 `completed` 且 `url=null`。
