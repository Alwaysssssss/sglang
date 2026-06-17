# `python/sglang/srt/connector` 模块分析

## 定位

`connector` 抽象 SRT 与外部存储/远端实例的连接方式，覆盖 KV/文件拉取、Redis、S3 和 remote instance。它主要服务于模型文件获取、远端权重/实例连接和一些扩展存储场景。

## 关键文件

- `__init__.py`：`ConnectorType`、`create_remote_connector`、`get_connector_type`。
- `base_connector.py`：`BaseConnector`、`BaseKVConnector`、`BaseFileConnector` 抽象，包含 `pull_files()` / `weight_iterator()` 一类文件和权重访问接口。
- `redis.py`：`RedisConnector`，KV 连接器实现。
- `s3.py`：`S3Connector`、`list_files` 和 allow/ignore 过滤。
- `remote_instance.py`：`RemoteInstanceConnector`。
- `utils.py`：URL/model name 解析和从 DB 拉文件。
- `serde/`：序列化接口，包含 `Serializer`、`Deserializer`、`SafeSerializer`、`SafeDeserializer` 和 `create_serde`。

## 运行流程

调用方根据 URL 或配置创建 connector。KV connector 提供 get/set 等键值能力，file connector 负责列文件和拉取文件，remote instance connector 则用于连接其他 SGLang 实例或远端权重服务；instance 模式还会创建自定义分布式组。serde 子包为连接器传输内容提供可替换序列化。

## 依赖关系

`server_args` 使用 `ConnectorType` 和 URL 解析；`model_loader`、remote instance 权重加载、RunAI/object storage 相关路径可能间接依赖连接器。S3/Redis 实现依赖对应外部客户端库和网络环境。

## 设计要点和风险

- connector 是边界模块，应保持接口小而清晰，避免把业务逻辑下沉到存储适配层。
- URL scheme 到 connector type 的解析是用户可见行为，新增 scheme 时要同步 `ServerArgs` 校验和文档。
- S3 文件过滤影响加载文件集合，错误过滤可能导致缺权重或加载额外大文件。
- S3 list、Redis 拉取和 allow/ignore 过滤要重点验证；分页、过滤未生效或大目录会直接影响模型加载。
- 某些路径会改全局 signal handler 或偏 CUDA 设备校验，嵌入式使用时需要注意副作用。
- serde 的“safe”并不等于可信任远端；跨进程/跨网络输入仍需限制来源。
