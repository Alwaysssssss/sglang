
# 机器账户创建与环境配置指南

## 1. 新建账户信息

为本机创建以下两个用户，并配置 SSH 公钥、远程和研发相关开发环境。

---

### 账户一
- 用户名：`zhiheng`
- 登录密码：`123321`
- SSH 公钥：
  ```
  ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFjcoPVzY/eqZN1c9KPH3TF4SHMKmx/ku3PPYN4gGFlj heng@HengdeMacBook-Pro.local
  ```

---

### 账户二
- 用户名：`tyx`
- 登录密码：`111tyx`
- SSH 公钥：
  ```
  ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDDJCYGWzMcBpoSp2nsDOmspkRJZsXx2gczVyGzZcWGD taoyuxuan2002@qq.com
  ```

---

## 2. 环境配置要求

### 2.1 Python/Conda/UV 环境

- 系统需支持多版本 Python 环境，可用 `conda` 或 `pyenv` 管理。
- 安装 [uv](https://github.com/astral-sh/uv) 以替代 pip 加速依赖安装，支持 `uv pip install ...` 使用。
- 为两个账号分别设置合适的工作目录与虚拟环境。

### 2.2 远程开发支持

- 确保 SSH 服务开启，添加上述公钥到各自用户的 `~/.ssh/authorized_keys`，允许免密登录。
- 检查防火墙设置，保证 22 端口可用。
- 安装/升级 Remote-SSH 支持组件，确保可通过 VS Code 的 Remote SSH 功能顺利远程连接和开发。

### 2.3 配置 Codex 与 Claude Code 插件

- 预装或为每个用户在 VS Code 中配置 Codex、Claude Code 等智能辅助插件（如未能直接装在服务器端，可写入推荐配置/README 供开发者本地 VS Code 侧安装）。
- 若有必要，设置相关代理环境变量（如 HTTP_PROXY、HTTPS_PROXY、ALL_PROXY）。

### 2.4 账户安全与管理建议

- 初始密码登录后，建议提示用户尽快修改为强密码。
- 审查用户目录和权限，避免越权访问。

---

## 3. 操作示例（以 Ubuntu/Linux 环境为例）

```bash
# 1. 创建用户及设置密码
sudo useradd -m zhiheng
sudo passwd zhiheng
sudo useradd -m tyx
sudo passwd tyx

# 2. 添加 SSH 公钥
sudo mkdir -p /home/zhiheng/.ssh && sudo tee /home/zhiheng/.ssh/authorized_keys
sudo mkdir -p /home/tyx/.ssh && sudo tee /home/tyx/.ssh/authorized_keys

# 3. (可选) 配置 Python/conda/uv 环境
# 下载安装 Miniconda、uv, pyenv 按需配置...

# 4. 检查 SSH 服务状态
sudo systemctl status sshd

# 5. 测试 VS Code 远程连接及环境
```

---

## 4. 常见问题提示

- SSH 配置错误无法连接时，可查阅 `/var/log/auth.log` 或 `journalctl -u sshd` 获取详细报错信息。
- 代理或网络环境特殊时，记得设置相关代理变量，见 `bashrc/zshrc` 示例。
- 为保障新环境安全，建议每个新用户第一次用初始密码登陆后自行改密，定期检查账户安全。

如有其他特殊环境需求，请补充说明。