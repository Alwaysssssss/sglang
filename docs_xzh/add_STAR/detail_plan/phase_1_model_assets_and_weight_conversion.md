# Phase 1：模型资产与权重转换

## 1. 阶段目标

本阶段的目标是先把模型资产边界钉死，确保后续实现从第一天起就不依赖 `STAR_mg` 原仓库运行。

阶段完成后，应满足：

1. 已定义 SGLang 侧的目标模型目录结构
2. 已明确 STAR 原始权重输入来源
3. 已设计并实现离线转换脚本方案
4. 已明确 transformer / VAE / text encoder / tokenizer / scheduler 的资产归属
5. 已有最小 smoke test 能验证“加载目标目录而不 import STAR_mg”

---

## 2. 本阶段范围

### 本阶段处理

1. 模型目录布局设计
2. checkpoint / tokenizer / scheduler 元信息整理
3. 权重转换脚本设计
4. key remap 规则设计
5. 资产清单与版本记录

### 本阶段不处理

1. pipeline 类实现
2. `SamplingParams` / `Req` 扩展
3. stage 编排
4. Denoising 主循环接线
5. parity 结果验证

---

## 3. 为什么第一阶段先做这个

如果第一阶段不先把资产层做干净，后面所有代码都容易走向下面这种坏形态：

1. 运行时临时读取 STAR 的 YAML
2. 运行时从 STAR checkpoint 里按训练框架逻辑加载模块
3. loader、pipeline、测试脚本都依赖 STAR 环境变量和目录结构

这会直接破坏本项目最核心的目标：**松耦合接入**。

因此本阶段的原则是：

1. **权重转换复杂可以接受**
2. **运行时耦合复杂不可接受**

---

## 4. 计划涉及的代码文件

### 4.1 新增文件

建议新增：

1. `python/sglang/multimodal_gen/tools/convert_star_cogvideox_sr.py`
2. `python/sglang/multimodal_gen/tools/star_cogvideox_keymap.py`
3. `python/sglang/multimodal_gen/tools/star_cogvideox_manifest.py`

### 4.2 可能新增的辅助文档或模板

可选新增：

1. `docs_xzh/add_STAR/detail_plan/assets_layout_example.md`
2. `docs_xzh/add_STAR/detail_plan/key_mapping_checklist.md`

如果不单独建文档，也应把这些内容沉淀在转换脚本的帮助信息或旁边注释中。

---

## 5. 源资产清单

需要明确 STAR 原始推理所依赖的资产至少包括：

1. Transformer 权重
2. 3D VAE 权重
3. T5 text encoder 权重
4. tokenizer 目录
5. scheduler 参数
6. 任何与模型结构绑定的 config 元信息

建议在转换脚本中统一通过 CLI 参数传入，不允许依赖环境变量。

### 推荐 CLI 参数

```bash
python -m sglang.multimodal_gen.tools.convert_star_cogvideox_sr \
  --src-transformer /path/to/star/transformer.ckpt \
  --src-vae /path/to/star/vae.ckpt \
  --src-text-encoder /path/to/star/t5_dir \
  --src-tokenizer /path/to/star/t5_dir \
  --src-config /path/to/star/config.yaml \
  --output-dir /path/to/converted_model
```

如果 `src-config` 不是强依赖，也应作为可选项，仅用于抽取结构参数，不可成为运行时依赖。

---

## 6. 目标模型目录结构

建议转换后目录结构如下：

```text
<model_root>/
  model_index.json
  star_integration_config.json
  transformer/
    config.json
    model.safetensors
  vae/
    config.json
    model.safetensors
  text_encoder/
    config.json
    model.safetensors
  tokenizer/
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    ...
  scheduler/
    scheduler_config.json
  manifests/
    source_assets.json
    conversion_report.json
    key_mapping_report.json
```

### 必须满足的要求

1. 目录本身足以支撑 SGLang native loader 运行
2. 不需要 STAR repo 在旁边存在
3. 所有结构参数都能从本目录读取
4. 模型版本与转换来源可追溯

---

## 7. 转换脚本的职责拆分

建议不要把所有逻辑都塞在一个脚本里，建议拆成三层：

### 7.1 入口脚本

文件：

1. `convert_star_cogvideox_sr.py`

职责：

1. 解析 CLI 参数
2. 校验输入路径
3. 驱动转换流程
4. 输出报告和退出码

### 7.2 key mapping 规则层

文件：

1. `star_cogvideox_keymap.py`

职责：

1. 定义原始权重 key 到目标权重 key 的映射规则
2. 维护显式字典映射
3. 维护 regex / prefix 批量替换规则

### 7.3 manifest / 报告层

文件：

1. `star_cogvideox_manifest.py`

职责：

1. 记录输入资产路径和哈希
2. 记录缺失 key / 多余 key
3. 记录转换后目录结构
4. 为后续复现实验提供来源证明

---

## 8. key mapping 的实施方式

## 8.1 不要直接在运行时做 key 兼容

不建议：

1. 在 `runtime/models/dits/star_cogvideox_sr.py` 里写大量 `if old_key in state_dict`
2. 在 loader 里为 STAR 开特殊分支补丁

推荐做法：

1. 所有 key 兼容在离线转换阶段完成
2. 运行时目标 state dict 应该尽量与目标类一一对应

## 8.2 mapping 规则建议结构

建议在 `star_cogvideox_keymap.py` 中维护如下三类规则：

1. **显式一对一映射**
   用于结构名变化较大的关键层
2. **前缀替换映射**
   用于大段模块整体搬迁
3. **忽略规则**
   用于训练态参数、优化器残留或推理无关权重

推荐数据结构：

```python
EXACT_KEY_MAP = {
    "old_key_a": "new_key_a",
}

PREFIX_RULES = [
    ("model.transformer.", "transformer."),
]

IGNORE_PATTERNS = [
    "optimizer",
    "loss",
]
```

## 8.3 转换时的校验输出

转换脚本必须输出：

1. 未匹配源 key 列表
2. 目标缺失 key 列表
3. dtype 与 tensor shape 摘要
4. 参数总量统计

如果缺失 key 非空，不应默认成功；需要：

1. 明确区分“可忽略缺失”和“结构错误缺失”
2. 非可忽略缺失时转换脚本直接失败

---

## 9. 组件级转换策略

## 9.1 Transformer

目标：

1. 产出可被 `runtime/models/dits/star_cogvideox_sr.py` 直接加载的 state dict

建议策略：

1. 先按原始模块前缀拆出 transformer 权重
2. 对 patch embedding、AdaLN、本地增强模块等重点层做显式映射
3. 对通用 block 层做规则化 rename
4. 输出 `transformer/config.json`

`config.json` 至少应包含：

1. latent channels
2. hidden size
3. layer 数
4. attention head 数
5. patch size
6. spatial/temporal latent 尺寸约束
7. 文本长度约束

## 9.2 VAE

目标：

1. 产出可被 `runtime/models/vaes/star_cogvideox_vae.py` 直接加载的 state dict

建议策略：

1. 只保留推理所需 encoder / decoder / regularizer 参数
2. 不迁移 loss、训练辅助模块
3. 输出 `vae/config.json`

`config.json` 至少应包含：

1. latent channels
2. spatial compression ratio
3. temporal compression ratio
4. scaling factor / shift factor
5. encode/decode 的推理模式参数

## 9.3 Text Encoder 和 Tokenizer

优先目标：

1. 尽量复用现成 T5 资产目录
2. 避免把 STAR conditioner 结构一并迁移进来

如果原资产已经是标准 HuggingFace 目录，则：

1. 直接复制
2. 在 manifest 中记录来源

如果不是，则：

1. 转换为标准 `from_pretrained()` 可读取的目录结构

## 9.4 Scheduler

如果 scheduler 主要依赖配置而不是大权重：

1. 以 `scheduler_config.json` 的方式导出
2. 记录推理步数、sigma/timestep 相关参数

---

## 10. 转换脚本实施步骤

建议按以下顺序实现：

1. 先实现 `--dry-run`
   只读取源资产并打印摘要，不写文件
2. 再实现 `--export-config-only`
   只生成目标 config 和 manifest
3. 再实现完整权重导出
4. 最后补强校验和失败处理

### 推荐主流程伪代码

```python
def main():
    args = parse_args()
    src = inspect_source_assets(args)
    target_layout = build_target_layout(args.output_dir)

    if args.dry_run:
        print_source_summary(src)
        return

    export_text_encoder_and_tokenizer(src, target_layout)
    export_scheduler_config(src, target_layout)

    transformer_sd = load_transformer_state_dict(src)
    mapped_transformer_sd, transformer_report = remap_transformer_keys(transformer_sd)
    save_transformer(mapped_transformer_sd, target_layout)

    vae_sd = load_vae_state_dict(src)
    mapped_vae_sd, vae_report = remap_vae_keys(vae_sd)
    save_vae(mapped_vae_sd, target_layout)

    write_model_index(target_layout)
    write_manifests(target_layout, src, transformer_report, vae_report)
    run_post_export_smoke_checks(target_layout)
```

---

## 11. 阶段内测试计划

## 11.1 建议新增的测试或脚本

建议新增：

1. `python/sglang/multimodal_gen/test/unit/manual/test_star_weight_conversion.py`
2. `python/sglang/multimodal_gen/test/unit/test_vae_loader.py` 中增加 STAR 模型目录加载用例

## 11.2 必跑检查

至少执行：

1. 转换脚本 `--dry-run`
2. 转换脚本完整导出
3. 用独立 Python 进程读取导出的：
   - transformer
   - vae
   - tokenizer
   - text_encoder
4. 确认过程中不需要 `PYTHONPATH` 指向 `STAR_mg`

### 推荐命令

```bash
python -m sglang.multimodal_gen.tools.convert_star_cogvideox_sr --dry-run ...
python -m sglang.multimodal_gen.tools.convert_star_cogvideox_sr ...
pytest python/sglang/multimodal_gen/test/unit/test_vae_loader.py -q
```

---

## 12. 阶段验收标准

本阶段完成的标准是：

1. 已有稳定的转换后模型目录结构
2. 已有转换脚本和 manifest 机制
3. 已确认运行时不需要访问 `STAR_mg`
4. 已确认导出资产能被后续 SGLang 代码按本地目录加载
5. 所有缺失 key / unexpected key 已被解释或清零

---

## 13. 失败信号与止损点

出现以下情况时，不要进入下一阶段：

1. 仍需要 STAR YAML 才能决定结构参数
2. 仍需要 STAR repo import 才能加载模型
3. key mapping 依赖大量运行时兼容逻辑
4. transformer / VAE 的目标 config 仍不稳定

如果出现这些情况，优先回到本阶段继续收敛，而不是在后续 pipeline 里打补丁。
