# `python/sglang/srt/server_args_config_parser.py` 模块分析

## 定位

该模块负责把 YAML 配置文件合并进 CLI 参数列表，让命令行应用支持 `--config xxx.yaml`。它不理解 `ServerArgs` 语义，只在 argparse 层做“配置字典 -> 参数数组”的转换和优先级合并。

## 关键类

- `ConfigArgumentMerger`：主类，接收 `argparse.ArgumentParser` 或 legacy `boolean_actions`。
- `merge_config_with_args(cli_args)`：查找 `--config`，读取 YAML，转换成参数列表，并与 CLI 参数合并。
- `_extract_config_file_path`：确保最多一个 `--config` 且后面有路径。
- `_parse_yaml_config` / `_validate_yaml_file`：只接受 `.yaml` / `.yml`，根节点必须是 dict。
- `_convert_config_to_args`：把 bool/list/scalar 分别转换成 CLI 形式。

## 合并语义

配置参数会和命令行参数拼接，使优先级保持“CLI > Config > Defaults”。bool 处理分两类：对于 argparse 的 `store_true` action，配置值为 true 才加入 flag，false 则跳过；普通 bool 会转换为 `--key true/false`。list 会转换为 `--key item1 item2 ...`。

## 设计要点和风险

- 当前代码只支持 `_StoreTrueAction` 和 `_StoreAction`，其他 argparse action 会被记录为 unsupported，出现在配置文件中会报错。
- YAML key 会保留原始连字符形式生成 `--key`，但校验时用 `key.replace("-", "_")` 匹配 argparse dest。
- 该模块不校验业务合法性；配置值最终仍要交给 `ServerArgs` / argparse 做类型和语义检查。
