# VSR Server API

## 服务方式

复用视频编辑的原生 SGLang HTTP server、scheduler、视频任务存储、结果下载和终态回调。
新增 `POST /v1/videos/restorations`；加载 `WanVSRPipeline`，模型常驻，一次仅接受一个 VSR 任务，忙时立即返回 code=2。
不加载额外的独立推理模型，也不为每个请求重启 Python。

VSR pipeline 返回已生成的 `OutputBatch.output_file_paths`，服务层不会二次编码。
默认保留全部源音轨；音频合并结束后才返回最终文件、完成任务并触发上传／成功回调。音轨不兼容时转 AAC，其他合并错误使任务失败。依赖及同步限制见 [音频保留](audio.md)。
每个 tile 开始前检查取消标记／超时；GPU 推理结束或报错后才释放接单权限，避免前一个请求还在运行时并发访问因果缓存。

## 单卡启动

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=8 TORCHINDUCTOR_COMPILE_THREADS=4 \
PYTHONPATH=python output_results/vsr/migration_env210/bin/python \
  docs_always/add_new_mode/add_vsr/scripts/serve_vsr.py \
  --checkpoint-dir /mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/outputs/best/checkpoint-1300 \
  --wan-root /mnt/shanhai-ai/liuh/Wan2.2-TI2V-5B-Diffusers \
  --host 127.0.0.1 --port 30176 \
  --output-dir output_results/vsr/server_api
```

环境为 torch 2.10/cu126。脚本启用 encoder/decoder compile、cuDNN benchmark、channels_last_3d、decoder implicit padding、GPU 后处理；不启用双卡、空间 batch 或固定条件缓存。
`NO_PROXY` 用于保证本地 HTTP 测试及回调不经过环境代理；其他 URL 仍遵循现有视频接口的代理配置。

## 提交

```bash
curl --noproxy '*' http://127.0.0.1:30176/v1/videos/restorations \
  -H 'Content-Type: application/json' \
  -d '{
    "taskId": "vsr-demo-001",
    "video_input_path": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/vsr/input/input.mp4",
    "target_resolution": "3840x2160"
  }'
```

`target_resolution` 为 **高度×宽度**；上例输出宽2160、高3840。
返回 `{"code":0,"message":"Task submitted","id":"vsr-demo-001"}`。code=1 表示校验／提交失败，code=2 表示忙碌。与视频编辑一样，业务拒绝通过 JSON code 表达。

### 完整请求参数

输入视频必须且只能提供本地路径或远程地址，其余参数均可选。表中的处理默认值来自 `WanVSRPipelineConfig`，服务启动配置可以覆盖它们；请求省略时继承服务配置。`preserve_audio` 和 `timeout` 的默认值由请求模型直接设置。

| 参数 | 类型 | 默认值与约束 |
| --- | --- | --- |
| `taskId / task_id` | string | 可选，自动生成；仅允许字母、数字、下划线、连字符，长度 1～128；不能重复 |
| `video_input_path` | string | 服务端可读的本地路径；Docker 部署填写容器路径 |
| `videoUrl / video_url` | string | 远程视频地址或既有下载器支持的源；与 video_input_path 必须且只能提供一个 |
| `target_resolution` | string | 默认 null；格式为高×宽（HxW），高、宽必须为正偶数；横向 1920×1080 填 1080x1920 |
| `long_edge` | int | 默认 3840；正整数，按最长边等比缩放；同时指定时优先使用 target_resolution；两者省略时先使用服务固定分辨率，否则使用服务最长边 |
| `tile_t` | int | 默认 33；时间分块帧数，必须为正整数且满足 4n+1 |
| `tile_h` | int | 默认 320；空间分块高度，必须为正的 32 倍数 |
| `tile_w` | int | 默认 640；空间分块宽度，必须为正的 32 倍数 |
| `temporal_overlap` | int | 默认 5；时间重叠帧数，≥0 且小于 tile_t |
| `spatial_overlap` | int | 默认 32；空间重叠像素，≥0 且小于 min(tile_h, tile_w) |
| `color_ref` | string | 默认 global；允许 global（全局参考）、chunk（分块参考）、none（不使用颜色参考） |
| `color_ref_samples` | int | 默认 64；全局颜色参考采样数，≥0 |
| `crf` | int | 默认 5；编码质量参数，范围 0～51，越小质量越高 |
| `preserve_audio` | bool | 默认 true；保留全部源音轨，false 输出无声视频 |
| `read_queue` | int | 默认 2；读取队列容量，范围 1～16 |
| `write_queue` | int | 默认 4；写入队列容量，范围 1～16 |
| `gpu_postprocess` | bool | 省略时继承启动配置；代码默认 false，本文启动脚本启用此项 |
| `timeout` | int | 默认 -1 不设期限；正数为处理期限秒数，在 tile 边界响应，媒体探测／音频合并期间每 0.2 秒检查 |
| `callbackUrl / callback_url` | string | 可选 HTTP(S) 终态回调地址，须从服务端／容器内可访问 |
| `minioConfig / minio_config` | object | 可选对象存储配置，字段见下表 |
| `outputObjectKey / output_object_key` | string | 必须配合 minioConfig；省略时使用 <任务 ID>.mp4 |
| `output_bucket` | string | 可选，覆盖对象存储配置中的输出桶 |

### 补全处理参数的 curl 示例

以下使用容器内测试视频；每次提交使用新的任务 ID。示例显式设置 `gpu_postprocess` 为 `false`，如需继承服务启动配置可删除该字段。

```bash
curl --noproxy '*' -fsS http://127.0.0.1:30176/v1/videos/restorations \
  -H 'Content-Type: application/json' \
  -d '{
    "taskId": "vsr-demo-001",
    "video_input_path": "/input/smoke-generated.mp4",
    "target_resolution": "256x256",
    "timeout": -1,
    "tile_t": 33,
    "tile_h": 320,
    "tile_w": 640,
    "temporal_overlap": 5,
    "spatial_overlap": 32,
    "color_ref": "global",
    "color_ref_samples": 64,
    "crf": 5,
    "preserve_audio": true,
    "read_queue": 2,
    "write_queue": 4,
    "gpu_postprocess": false
  }'
```

按最长边等比缩放时，将 `"target_resolution": "256x256"` 替换为 `"long_edge": 3840`。使用远程输入时，将 `video_input_path` 替换为 `videoUrl`，不要同时提供二者。

### 可选回调与对象存储配置

需要回调或上传时，将相应字段合并到上述请求 JSON，替换示例地址和凭据；不使用时省略对应字段。

```json
{
  "callbackUrl": "http://your-callback-server/vsr/callback",
  "minioConfig": {
    "endpoint": "minio.example.com:9000",
    "bucket_name": "videos",
    "access_key": "YOUR_ACCESS_KEY",
    "secret_key": "YOUR_SECRET_KEY",
    "secure": false,
    "region": "us-east-1",
    "provider": "minio",
    "prefix": "vsr"
  },
  "outputObjectKey": "vsr-demo-001.mp4",
  "output_bucket": "videos"
}
```

`minioConfig` 完整字段：

| 字段 | 类型 | 必填／默认值与说明 |
| --- | --- | --- |
| `endpoint` | string | 必填，存储端点，可包含 HTTP(S) 协议 |
| `bucket_name` | string | 必填，默认桶；也支持 `bucket` / `bucketName` |
| `access_key` | string | 必填，访问密钥；也支持 `accessKey` / `rootUser` |
| `secret_key` | string | 必填，私有密钥；也支持 `secretKey` / `rootPass` |
| `secure` | bool | 默认 false；端点未写协议时决定使用 HTTP 或 HTTPS；也支持 `useSSL` |
| `region` | string | 默认 us-east-1 |
| `provider` | string | 可选；cos 启用 COS 处理，minio 使用通用 S3 兼容处理；省略时依据端点识别 COS |
| `prefix` | string | 可选，输出对象键前缀 |

容器中的 `127.0.0.1` 指容器自身，远程输入、回调和存储端点须从容器内可访问。对象存储复用既有实现，真实外部上传尚未验收。

输出路径由服务控制，保存在 `--output-dir/vsr/<id>.mp4`。模型精度和编译选项在启动时设定，不逐请求改变。接口拒绝未定义的顶层字段，如 `prompt`、`seed`、`dtype`、`output_path`。
FPS、帧数取自源视频；通用采样元数据要求整数 FPS，但 VSR 实际编码会重新读取并保留源视频真实 FPS。

## 状态、下载、取消

- `GET /health`：服务存活。
- `GET /v1/videos/<id>`：queued/running/completed/failed，包含结果路径、URL和推理耗时。
- `GET /v1/videos/<id>/progress`：状态和终态回调执行结果；当前进度为 queued=0、running=1、completed=100，不是逐帧百分比。
- `GET /v1/videos/<id>/content`：完成后下载 MP4；云存储输出使用返回的 URL。
- `DELETE /v1/videos/<id>`：复用视频接口取消；当前沿用框架超时错误表示取消，等待后端停止后返回 failed。

回调复用通用视频 payload：`id/status/progress/file_path/url/error/reason/inference_time_s` 等，失败重试沿用现有视频接口。
任务状态在内存中，服务重启不会恢复历史任务；本地成功输出文件仍保留。

## 单卡验收

```bash
python docs_always/add_new_mode/add_vsr/scripts/test_server_api.py
```

测试先执行3次53帧、320×640真实输入请求，共6个33帧 tile，首次编译和这3次请求不计性能；随后执行两次53帧4K请求。
覆盖本地输入、HTTP URL、终态回调、状态轮询、下载、忙碌拒绝、重复 ID、无效路径、超时与后续恢复，以及两次完整输出哈希一致。
质量单独检查，阈值 SSIM≥0.985、MSE≤36、MAE≤6，不把首次请求耗时作为稳态性能。
MinIO 复用了现有实现，本轮不连接外部对象存储；没有宣称完成真实云存储验收。

## 已完成的实测

仅使用 GPU6。3次预热请求共6个真实 tile 完成后，53帧、H3840×W2160：

| 请求 | 服务端耗时 | HTTP提交至轮询发现完成 |
|---|---:|---:|
| 第一次 | 106.036 s | 106.260 s |
| 第二次 | 106.105 s | 108.256 s |
| 平均 | **106.071 s** | 107.258 s |

HTTP客户端每2秒轮询，因此该列包含轮询等待，不能把与服务端耗时的差全部解释为服务开销。首个编译请求39.966秒，后两个小分辨率预热请求3.287/3.279秒，均不参与稳态统计。

两次服务输出 SHA256 均为 `d35a8a66d26f39695b1c4b5c502f87528caf7501f6679ffa9a3bcaad6d8fa15c`，与此前已测的同配置单卡 `parallel_round1/0_single.mp4` 完全一致。因此复用该精确文件的逐帧 MP4 指标，不重复执行相同质量计算：53/53通过，最低SSIM=0.9889910038，最大MSE=2.6709003，最大MAE=1.1163722。当前验收阈值0.985/36/6。没有另行声明服务原始RGB dump验收。

已完成真实 HTTP 验证：本地输入、localhost URL下载、5次终态回调、状态查询、内容下载、忙碌拒绝、重复ID拒绝、无效路径、超时失败后恢复、DELETE取消。MinIO实际上传仍未测试。
26项测试通过，含原生pipeline返回已有输出文件、padding/有序tile/重复请求回归、tile中断以及取消时唤醒满队列上的reader。

记录位于 `output_results/vsr/server_api_test/`：`results.json`、`summary.json`、`quality.json`、`cancellation.json`、`server_info.json`、`tests.log`、`server.log`、`http_test.log`。
本轮测试服务完成后关闭，释放GPU6；按上面的命令可重新启动常驻服务。

## 双卡服务

使用相同 HTTP API 与模型优化选项，启动时追加 `--tile-devices cuda:0 cuda:1`，并设置 `CUDA_VISIBLE_DEVICES=6,7`：

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=8 TORCHINDUCTOR_COMPILE_THREADS=4 \
PYTHONPATH=python output_results/vsr/migration_env210/bin/python \
  docs_always/add_new_mode/add_vsr/scripts/serve_vsr.py \
  --checkpoint-dir /mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/outputs/best/checkpoint-1300 \
  --wan-root /mnt/shanhai-ai/liuh/Wan2.2-TI2V-5B-Diffusers \
  --tile-devices cuda:0 cuda:1 \
  --output-dir output_results/vsr/server_api_dual
```

这里 `num_gpus=1` 表示只启动一个框架 scheduler；实际使用两张物理 GPU，由 `pipeline_config.tile_devices` 管理完整模型副本。不要同时增加框架 scheduler 数量。
VSR 双卡 scheduler 使用非 daemon 进程，允许建立 tile worker 子进程；其他默认模型的进程方式不变。启动脚本退出时清理整个服务进程树。
请求在批间超时或关闭 iterator 时保留 worker 池，后续请求复用已加载的两份模型。通信或模型运行异常仍关闭失效的池。

双卡预热测试命令：

```bash
python docs_always/add_new_mode/add_vsr/scripts/test_server_api.py \
  --warmup-resolution 320x1216 \
  --output-dir output_results/vsr/server_api_dual_test
```

320×1216 在每个时间窗口产生2个空间 tile，分配到两卡。3次53帧预热请求合计12个 tile，即每卡6次；320×640只有1个空间tile，不适合双卡预热。超时后的恢复请求也使用双tile尺寸，检查两张卡均可继续服务。

### 双卡实测结果

同一53帧4K输入，GPU6+GPU7，每卡预热6次后：

| 请求 | 服务端耗时 | HTTP提交至发现完成（2秒轮询） |
|---|---:|---:|
| 第一次 | 60.417 s | 62.220 s |
| 第二次 | 60.943 s | 62.230 s |
| 平均 | **60.680 s** | 62.225 s |

相对上一轮同配置单卡服务106.071秒，双卡约1.748倍，耗时减少42.79%。这是先后两轮服务测试的比较，不是同一进程单/双卡交错测试。首个双卡编译请求82.346秒、后两次小尺寸预热3.773/3.693秒，均不计入以上结果。

两次4K视频SHA256相同：`d77dd43b31325152bec53fbc1f19d22a10198aa7ee6bc0e6b4640e1dad04252c`。
真实HTTP本地/URL输入、5次终态回调、下载、忙碌拒绝、重复ID、超时与后续双tile恢复全部通过。29项相关单元测试通过。
记录：`output_results/vsr/server_api_dual_test/results.json`、`summary.json`、`performance_summary.json`、`quality.json`。
测试服务已经关闭，退出时同时清理scheduler及tile worker进程。

双卡服务输出与原始 SwiftVR、上一轮单卡服务分别进行了新的逐帧MP4比较（均53帧，结构检查通过）：

| 对照 | 最低SSIM | 最大MSE | 最大MAE | 失败帧 |
|---|---:|---:|---:|---|
| 原始SwiftVR | 0.9890014453 | 2.6671414 | 1.1159816 | 无 |
| 单卡服务 | 0.9894649453 | 2.5499482 | 1.0791202 | 无 |

满足用户阈值SSIM≥0.985、MSE≤36、MAE≤6。此次未另行导出双卡服务原始RGB，精度结论对应最终MP4。
