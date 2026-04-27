# 服务器网络配置(服务器走本地的代理)

1. **vscode 打开服务器配置文件**

   ```text
   Host 10.xx.xx.xx
     HostName 10.xx.xx.xx
     User root
     Port xxxx
     RemoteForward 10808 localhost:7897
   ```

> **localhost:7897 通过 clash 查看，也可能是 7890 等**
> **RemoteForward:10808 选择一个不被占用的端口即可**

2. 在服务器的 `~/.bashrc` 添加以下内容

```
export http_proxy=http://127.0.0.1:10808
export https_proxy=http://127.0.0.1:10808
```

**然后执行：**

```text
source ~/.bashrc
```

3. IDE 中按下`Cmd/Ctrl + Shift + P`，输入`Open Remote Settings`，进入远程服务器的设置面板，搜索「proxy」，将： `http-proxy` 设置为 `http://127.0.0.1:10808`，并将Http:Proxy Strict SSL取消选择

# Docker 环境及网络配置

1. **安装 Dev contianer 插件**

2. **在服务器的.bachrc 中添加:**

```
export MODEL_ROOT=/home/model/models
export CODEX_PACKAGE_DIR=/usr/lib/node_modules/@openai/codex
export CLAUDE_CODE_PACKAGE_DIR=$HOME/.npm-global/lib/node_modules/@anthropic-ai/claude-code

# 如果 devcontainer 使用 --network host，就用 127.0.0.1,和服务器走相同的代理
export CONTAINER_PROXY=http://127.0.0.1:10808		
```

3. shift+command+p:在Dev container中选择打开sglang文件夹

> 权重下载到MODEL_ROOT=/home/model/models，在mounts中映射到了容器的models文件夹中

# wan2.1测试

```
 sglang generate \
    --model-path models/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "a cat is playing piano" \
    --num-gpus 1 \
    --height 480 \
    --width 832 \
    --num-frames 81 \
    --fps 16 \
    --num-inference-steps 50 \
    --guidance-scale 3.0 \
    --save-output \
    --output-path outputs \
    --output-file-name wan21_t2v.mp4
```





