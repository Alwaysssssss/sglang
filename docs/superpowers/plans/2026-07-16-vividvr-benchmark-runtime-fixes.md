# Vivid-VR Benchmark 运行时修复实现计划

> **面向 AI 代理的工作者：** 在当前会话中内联执行；用户已明确不使用子代理。步骤使用复选框跟踪进度。

**目标：** 让 benchmark 能认证下载 Moto 私有结果，并让所有 torch.compile 服务在启动前获得现有 Python 3.10 开发头文件。

**架构：** 保持现有 benchmark 生命周期不变，只替换结果下载传输方式，并扩展 compile scheme 的服务环境。下载逻辑从结果 URL 推导 S3 地址；头文件逻辑使用轻量本地探测，不导入会触发运行时编译的推理模块。

**技术栈：** Python 3.10、boto3/botocore、pytest、GCC、Moto S3。

---

### 任务 1：认证 S3 结果下载

**文件：**
- 修改：`python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py`

- [ ] 编写失败测试：伪造 boto3 客户端，断言 `_download_result` 使用 URL 中的 endpoint、bucket、key 以及测试凭据，并生成目标文件。
- [ ] 运行该测试，确认旧的匿名 httpx 实现不能满足 boto3 调用断言。
- [ ] 用 boto3 `download_file` 实现最小认证下载，显式设置 `Config(proxies={})`，保留 `.partial` 原子替换和失败清理。
- [ ] 重跑该测试并确认通过。

### 任务 2：compile 服务头文件环境

**文件：**
- 修改：`python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py`

- [ ] 编写失败测试：创建临时 `Python.h` 和 multiarch `pyconfig.h`，通过 `SGLANG_PYTHON_DEV_INCLUDE` 指定目录，断言 compile scheme 环境注入 `CPATH/C_INCLUDE_PATH` 并保留已有路径。
- [ ] 运行该测试，确认当前 `build_service_environment` 缺少这两个变量。
- [ ] 实现轻量头文件探测与环境路径合并；compile scheme 未找到头文件时提前抛出 `BenchmarkConfigError`。
- [ ] 重跑该测试及既有服务环境测试并确认通过。

### 任务 3：回归与真实依赖 smoke test

**文件：**
- 验证：`python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py`
- 验证：`python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py`

- [ ] 运行 benchmark 单元测试文件，确认零失败。
- [ ] 使用探测到的本机头文件运行 `gcc -E`，确认 `Python.h` 及 multiarch 配置可解析。
- [ ] 在 tmux 中启动临时 Moto 服务，创建私有对象并调用 `_download_result`，确认下载内容一致后停止自有 session。
- [ ] 检查 `git diff --check`、相关差异和工作区状态，仅保留本任务修改。
