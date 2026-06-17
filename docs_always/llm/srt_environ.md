# `python/sglang/srt/environ.py` 模块分析

## 定位

`environ.py` 是 SRT 的环境变量声明中心。它用描述符把 `SGLANG_*` / `SGL_*` 环境变量包装成类型化字段，避免在代码各处直接 `os.getenv` 后手写解析。

## 关键类

- `EnvField`：环境变量描述符基类，保存默认值、显式 None 标记、`get/set/override/clear` 等方法。
- `EnvBool`、`EnvInt`、`EnvFloat`、`EnvStr`、`EnvTuple`：具体类型解析器。
- `ToolStrictLevel`：tool call 解析的严格程度枚举。
- `Envs`：所有环境变量声明集合，覆盖模型下载、日志、CI、grammar、debug/test、profile、scheduler、attention、cache、通信等开关。
- `envs`：模块级实例，代码中通过 `envs.SGLANG_XXX.get()` 读取。
- `temp_set_env`：临时设置非 SGLang 环境变量的 context manager，默认拒绝 `SGLANG_` / `SGL_` key，防止绕过集中声明。

## 运行机制

`EnvField.__set_name__` 把类属性名作为环境变量名。调用 `.get()` 时，如果环境变量不存在则返回 default；如果存在则按字段类型解析；解析失败会 warning 并回退默认值。`__bool__` 和 `__len__` 被显式禁用，强制调用方写 `.get()`，减少“描述符对象本身被当成 bool”的误用。

## 依赖关系

`server_args`、`scheduler`、`model_executor`、`layers`、`mem_cache`、`observability`、`utils` 等模块都读取 `envs`。它是运行时特性开关的底层来源，但不应该承载复杂业务逻辑。

## 设计要点和风险

- 新增 SGLang 环境变量应加到 `Envs`，不要在业务代码直接 `os.getenv("SGLANG_...")`。
- 默认值是行为契约，尤其是性能和 debug 开关；修改默认值可能改变线上吞吐、日志量或调度语义。
- `EnvField.set(None)` 使用字符串 `"None"` 加内部标记区分显式 None，这要求同一个进程内不要绕过 `EnvField` 改写该变量。
