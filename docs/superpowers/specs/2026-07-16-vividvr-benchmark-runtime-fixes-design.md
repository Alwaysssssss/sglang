# Vivid-VR Benchmark 运行时修复设计

## 目标

修复加速 benchmark 中两个已定位的运行时问题：Moto S3 私有对象被匿名下载导致 `403`，以及 `sglang serve` 未继承现有 Python 3.10 开发头文件路径导致 torch.compile 编译失败。

## 设计

- 结果下载继续使用服务返回的 URL 作为对象位置来源，但从 URL 解析 endpoint、bucket 和 key，改用带 benchmark 测试凭据的 boto3 客户端下载。客户端显式禁用代理，并先写入 `.partial` 文件再原子替换目标文件。
- 对启用 torch.compile 的实验，在启动服务进程前探测 Python 开发头文件。优先使用 `SGLANG_PYTHON_DEV_INCLUDE`，然后检查 Python `sysconfig` 路径及仓库历史使用的本地系统包解压路径。找到后，将主 include 目录及 multiarch 父目录注入 `CPATH` 和 `C_INCLUDE_PATH`。
- eager 实验不强制要求 Python 开发头文件；compile 实验找不到头文件时在服务启动前明确失败，避免长时间加载后才暴露 GCC 错误。

## 错误处理

- S3 URL 缺少协议、bucket 或 object key 时抛出 `BenchmarkDataError`。
- boto3 下载异常原样保留，且无论成功失败都清理 `.partial` 文件。
- compile 实验找不到 `Python.h` 时抛出 `BenchmarkConfigError`，错误信息列出检查方向。

## 验证

- 单元测试验证认证下载使用正确 endpoint、bucket、key、凭据和无代理配置，并验证原子产物。
- 单元测试验证 compile 服务环境包含正确的 `CPATH/C_INCLUDE_PATH`，且保留已有环境路径。
- 使用本机现有 headers 运行 GCC 预处理 smoke test。
- 使用临时 Moto 服务执行私有对象上传和认证下载 smoke test，确认不再出现匿名 `403`。
