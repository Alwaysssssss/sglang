# Vivid-VR 正式主机起服务命令

本文档用于在正式主机上长期驻留 Vivid-VR FlowCut 服务，等待真实用户请求。

当前目标口径：

- 主服务走 Vivid-VR 自己的 FlowCut 路由：`/v1/videos/repairs/flowcut`
- 默认正式配置使用双卡 `dual_gpu_fa_eager_compile`
- 默认不保存用户输入视频
- 默认不保留本地推理结果
- 默认不配置持久 caption 目录，让 bridge 生成的 caption 跟随 request 临时 workdir 生命周期

这份文档只覆盖“正式常驻服务”的启动方式，不覆盖 mock 验收。手动验收可参考 [mock_test.md](/home/zhiheng/sglang/docs_xzh/run_command/mock_test.md:1)。

## 1. 正式部署前提

- 主推理环境固定使用 `/home/zhiheng/sglang/.venv`
- caption sidecar 环境固定使用 `/home/zhiheng/sglang/.venv-vividvr-caption`
- 正式双卡默认语义固定为：
  - `--attention-backend fa`
  - `--sp-degree 2`
  - `SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global`
  - control pooling 已删除，`eager_global` 固定恢复 full global control context
  - `--enable-torch-compile`
- 正式请求必须传 `callbackUrl`
- 如果希望“默认不保留本地结果”真正成立，请求方必须传 `minioConfig`

最后这一点要特别说明：

- 服务端可以通过 `--input-save-path ""` 和 `--output-path ""` 关闭本地持久目录。
- 但如果请求方既不传 `minioConfig`，也不传 `outputPath`，当前实现没有外部化结果的出口，任务完成后可能仍会保留临时产物。
- 所以正式部署约束应当是：
  - 服务端默认关闭本地持久目录
  - 请求端默认必须传 `minioConfig`

## 2. 推荐端口

- caption sidecar：`31200`
- 双卡主服务：`31191`
- `torchrun / dist` master port：`30191`
- scheduler port：`56191`

如果正式主机已有端口占用，允许整体平移，但要保持所有关联端口一致。

## 3. 首次部署时准备 caption sidecar 环境

通常只需要执行一次：

```bash
cd /home/zhiheng/sglang
bash python/sglang/multimodal_gen/tools/setup_vividvr_caption_env.sh
```

准备完成后，caption 环境应位于：

```bash
/home/zhiheng/sglang/.venv-vividvr-caption
```

## 4. 首次部署前检查 `torch.compile` 的 Python 头文件

双卡正式默认配置包含 `--enable-torch-compile`。在新服务器上，如果系统缺少与
`/home/zhiheng/sglang/.venv/bin/python` 对应版本的 CPython 开发头文件，首次
compile 可能在 Triton/C 扩展编译阶段报 `Python.h: No such file or directory`。

这不是 Vivid-VR 代码问题，而是主机缺少 Python dev headers。

### 4.1 先确认 `.venv` 里的 Python 次版本

```bash
/home/zhiheng/sglang/.venv/bin/python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
```

当前仓库默认环境通常是 `3.10`，所以新服务器上优先安装匹配的开发包。

### 4.2 推荐解法：直接安装匹配版本的 Python 开发包

Ubuntu / Debian：

```bash
sudo apt-get update
sudo apt-get install -y python3.10-dev
```

如果发行版没有拆成 `python3.10-dev`，也可以安装：

```bash
sudo apt-get install -y libpython3.10-dev
```

安装完成后，可用下面的命令确认头文件已就位：

```bash
/home/zhiheng/sglang/.venv/bin/python - <<'PY'
import sysconfig
print(sysconfig.get_config_var("INCLUDEPY"))
PY
```

再检查：

```bash
ls "$(/home/zhiheng/sglang/.venv/bin/python - <<'PY'
import sysconfig
print(sysconfig.get_config_var("INCLUDEPY"))
PY
)/Python.h"
```

只要 `Python.h` 存在，通常就不需要额外设置 `CPATH`。

### 4.3 兜底解法：无法安装系统包时，手动提供匹配版本头文件

如果正式主机不能直接 `apt install`，至少要把与 `.venv` 解释器版本完全一致的
CPython 头文件准备到某个固定目录，然后在启动服务前导出：

```bash
export CPATH=/path/to/python3.10/include/python3.10:/path/to/python3.10/include${CPATH:+:$CPATH}
export C_INCLUDE_PATH=/path/to/python3.10/include/python3.10:/path/to/python3.10/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}
```

本次验收机上的临时兜底方式就是这样处理的。核心要求只有两个：

- 头文件版本必须与 `/home/zhiheng/sglang/.venv/bin/python` 的 minor version 一致
- `Python.h` 必须能被 compile 阶段的编译器直接找到

如果版本不一致，例如 `.venv` 是 `3.10`，却给了 `3.11` 的 headers，后续仍然可能
出现编译失败或更隐蔽的 ABI 问题，不建议这样混用。

### 4.4 正式部署建议

正式长期部署优先采用 `4.2` 的系统包方案，不要把临时解压目录当成长期依赖。

推荐顺序：

1. 先确认 `.venv` Python 次版本
2. 安装同版本 `pythonX.Y-dev` / `libpythonX.Y-dev`
3. 用一次 `sglang serve --enable-torch-compile` 的冷启动验证 compile 能正常开始
4. 只有在系统包无法安装时，才退回到 `CPATH` / `C_INCLUDE_PATH` 兜底方案

## 5. 启动 caption sidecar

先起 sidecar，再起主服务。

```bash
tmux new-session -d -s vividvr_caption_sidecar \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && CUDA_VISIBLE_DEVICES=0,1 /home/zhiheng/sglang/.venv-vividvr-caption/bin/python python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py --host 127.0.0.1 --port 31200 --parallel-workers 2 --worker-devices cuda:0,cuda:1 --cogvlm2-ckpt-path /home/zhiheng/ckpts/cogvlm2-llama3-caption 2>&1 | tee Vivid_Acceptance/logs/vividvr_caption_sidecar_$(date -u +%Y%m%dT%H%M%SZ).log'
```

查看 sidecar：

```bash
tmux attach -r -t vividvr_caption_sidecar
```

健康检查：

```bash
curl --noproxy '*' --silent --show-error --fail http://127.0.0.1:31200/health
```

预期结果：

```json
{"status":"ok"}
```

## 6. 启动正式双卡主服务

这条命令是当前推荐的正式常驻服务命令。

关键点：

- `--input-save-path ""`：不保留用户输入视频缓存
- `--output-path ""`：不保留本地持久输出目录
- 不传 `--vividvr-caption-work-dir`：bridge caption 默认跟随 request 临时 workdir
- `--host 0.0.0.0`：允许外部请求接入当前主机
- 如果新主机还没安装匹配版本的 Python dev headers，可临时补 `CPATH` 和 `C_INCLUDE_PATH`

```bash
tmux new-session -d -s vividvr_serve_dual_formal \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && export PYTHONUNBUFFERED=1 && export PYTHONPATH=python && export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global && CUDA_VISIBLE_DEVICES=0,1 /home/zhiheng/sglang/.venv/bin/sglang serve \
    --model-path /home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B \
    --model-id VividVR \
    --pipeline-class-name CogVideoXVividVRControlNetPipeline \
    --component-paths.vividvr /home/zhiheng/Vivid-VR/ckpts/Vivid-VR \
    --attention-backend fa \
    --num-gpus 2 \
    --tp-size 1 \
    --sp-degree 2 \
    --ulysses-degree 2 \
    --ring-degree 1 \
    --enable-torch-compile \
    --dist-timeout 3600 \
    --host 0.0.0.0 \
    --port 31191 \
    --master-port 30191 \
    --scheduler-port 56191 \
    --strict-ports \
    --input-save-path "" \
    --output-path "" \
    --vividvr-caption-bridge \
    --vividvr-caption-sidecar-url http://127.0.0.1:31200 \
    --vividvr-caption-sidecar-timeout 1800 \
    2>&1 | tee Vivid_Acceptance/logs/vividvr_serve_dual_formal_$(date -u +%Y%m%dT%H%M%SZ).log'
```

查看主服务：

```bash
tmux attach -r -t vividvr_serve_dual_formal
```

健康检查：

```bash
curl --noproxy '*' --silent --show-error --fail http://127.0.0.1:31191/health
```

预期结果：

```json
{"status":"ok"}
```

当前正式 bridge 服务不再要求 `--prompt-file-path`。如果 caption sidecar 成功产出 caption 文件，主服务会直接以该文件为 prompt 来源。

如果需要临时给 compile 补头文件，可把上面的命令改成下面这种写法：

```bash
tmux new-session -d -s vividvr_serve_dual_formal \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && export PYTHONUNBUFFERED=1 && export PYTHONPATH=python && export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global && export CPATH=/path/to/python3.10/include/python3.10:/path/to/python3.10/include${CPATH:+:$CPATH} && export C_INCLUDE_PATH=/path/to/python3.10/include/python3.10:/path/to/python3.10/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH} && CUDA_VISIBLE_DEVICES=0,1 /home/zhiheng/sglang/.venv/bin/sglang serve \
    --model-path /home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B \
    --model-id VividVR \
    --pipeline-class-name CogVideoXVividVRControlNetPipeline \
    --component-paths.vividvr /home/zhiheng/Vivid-VR/ckpts/Vivid-VR \
    --attention-backend fa \
    --num-gpus 2 \
    --tp-size 1 \
    --sp-degree 2 \
    --ulysses-degree 2 \
    --ring-degree 1 \
    --enable-torch-compile \
    --dist-timeout 3600 \
    --host 0.0.0.0 \
    --port 31191 \
    --master-port 30191 \
    --scheduler-port 56191 \
    --strict-ports \
    --input-save-path "" \
    --output-path "" \
    --vividvr-caption-bridge \
    --vividvr-caption-sidecar-url http://127.0.0.1:31200 \
    --vividvr-caption-sidecar-timeout 1800 \
    2>&1 | tee Vivid_Acceptance/logs/vividvr_serve_dual_formal_$(date -u +%Y%m%dT%H%M%SZ).log'
```

这只是过渡方案。正式机器稳定后，还是应回到 `4.2` 的系统包安装方式。

## 7. 可选：正式单卡主服务

如果临时只启单卡，可用下面这条。其余部署约束不变。

```bash
tmux new-session -d -s vividvr_serve_single_formal \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && export PYTHONUNBUFFERED=1 && export PYTHONPATH=python && CUDA_VISIBLE_DEVICES=0 /home/zhiheng/sglang/.venv/bin/sglang serve \
    --model-path /home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B \
    --model-id VividVR \
    --pipeline-class-name CogVideoXVividVRControlNetPipeline \
    --component-paths.vividvr /home/zhiheng/Vivid-VR/ckpts/Vivid-VR \
    --attention-backend fa \
    --num-gpus 1 \
    --tp-size 1 \
    --sp-degree 1 \
    --ulysses-degree 1 \
    --ring-degree 1 \
    --enable-torch-compile \
    --host 0.0.0.0 \
    --port 31190 \
    --master-port 30190 \
    --scheduler-port 56190 \
    --strict-ports \
    --input-save-path "" \
    --output-path "" \
    --vividvr-caption-bridge \
    --vividvr-caption-sidecar-url http://127.0.0.1:31200 \
    --vividvr-caption-sidecar-timeout 1800 \
    2>&1 | tee Vivid_Acceptance/logs/vividvr_serve_single_formal_$(date -u +%Y%m%dT%H%M%SZ).log'
```

## 8. 正式请求约束

正式主机虽然默认不保留本地输入和本地结果，但这依赖请求方遵守下面的约束。

### 7.1 必填项

- `taskId`
- `callbackUrl`
- `videoUrl` 或 `video_input_path` 二选一

### 7.2 强制建议项

- `minioConfig`
- `outputObjectKey`
- `outputBucket`

推荐原因：

- `minioConfig` 用于把结果外部化到对象存储
- `outputObjectKey` 用于固定对象路径
- `outputBucket` 用于显式指定结果 bucket

### 7.3 当前推荐请求示例

```bash
curl --fail-with-body -X POST "http://127.0.0.1:31191/v1/videos/repairs/flowcut" \
  -H "Content-Type: application/json" \
  -d '{
    "taskId":"prod-demo-001",
    "callbackUrl":"http://callback.example.com/flowcut",
    "timeout":-1,
    "videoUrl":"https://example.com/input/newspaper.mov",
    "minioConfig":{
      "endpoint":"s3.example.com",
      "bucketName":"flowcut-input",
      "accessKey":"***",
      "secretKey":"***",
      "secure":true,
      "region":"us-east-1"
    },
    "outputBucket":"flowcut-output",
    "outputObjectKey":"prod/vividvr/prod-demo-001",
    "numInferenceSteps":20,
    "numTemporalProcessFrames":121,
    "seed":42,
    "upscale":1.0
  }'
```

补充说明：

- `minioConfig.endpoint` 当前传的是 `host:port`，不要带 `http://` 或 `https://`
- `outputObjectKey` 如果不带后缀，服务会自动补成输入视频的扩展名
- `upscale` 是原版 Vivid-VR 的输入预缩放语义，不是后处理超分开关

## 9. 正式运行时的预期行为

在本文件这套正式命令下，默认预期如下：

- 请求输入视频会先落到 request 临时 workdir
- 如果请求方传了 `minioConfig`，成功结果会上传到对象存储
- 上传成功后，本地结果文件会删除
- request 临时 workdir 会在任务结束后清理
- bridge 生成的 caption sidecar 会跟随 request workdir 一起清理
- 服务不会删除调用方原始本地 `video_input_path`

如果请求方没有传 `minioConfig`，则不应再宣称“默认不保留本地结果”；这时结果可能保留在临时工作目录中，行为不满足本文件的正式部署目标。

## 10. 常用查询与取消

提交后可用下面几条接口：

查询详情：

```bash
curl --fail "http://127.0.0.1:31191/v1/videos/repairs/flowcut/<task_id>"
```

查询进度：

```bash
curl --fail "http://127.0.0.1:31191/v1/videos/repairs/flowcut/<task_id>/progress"
```

取消任务：

```bash
curl --fail -X DELETE "http://127.0.0.1:31191/v1/videos/repairs/flowcut/<task_id>"
```

取消后的当前对外语义与 `online_videoedit` 对齐：

- 任务状态会变成 `failed`
- `reason` 会是 `Request timed out.`

## 10. 日志与 tmux

默认日志目录：

```bash
/home/zhiheng/sglang/Vivid_Acceptance/logs
```

只读查看：

```bash
tmux attach -r -t vividvr_caption_sidecar
tmux attach -r -t vividvr_serve_dual_formal
```

停止服务：

```bash
tmux kill-session -t vividvr_serve_dual_formal
tmux kill-session -t vividvr_caption_sidecar
```

## 11. 推荐上线检查

按顺序检查：

1. sidecar `health` 返回 `ok`
2. 主服务 `health` 返回 `ok`
3. 主服务日志中没有端口冲突、caption bridge 初始化失败、compile 初始化失败
4. 调用方已确认会传 `callbackUrl`
5. 调用方已确认会传 `minioConfig`
6. 如需外网访问，防火墙和反向代理已放通主服务端口

满足以上条件后，再让正式请求流量接入。
