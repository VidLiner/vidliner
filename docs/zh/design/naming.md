# VidLiner —— 命名与词汇

[English](../../design/naming.md)

本文档的存在，是为了让代码库对"这个东西叫什么"只有一个答案。下面每个术语都是为 VidLiner 依据自身需求选定的。没有任何一项继承自其他项目的词汇、DSL、命令体系、包命名空间或目录结构。

## 产品词汇

| 术语 | 含义 |
| --- | --- |
| **Recipe** | 声明式 YAML 文档，描述一次增强 job 的意图。 |
| **Operator** | 一个领域操作，带声明的输入、输出、配置 schema 与能力需求。 |
| **Capability** | 流水线需要的一项带版本的具名能力，如 `vision.object_detection.v1`。 |
| **Backend** | 提供一项或多项能力的具体实现。 |
| **Runtime profile** | 说明这台机器上哪项能力由哪个 backend 提供的文档。 |
| **Binding** | runtime profile 中"能力 → backend 名"的映射条目。 |
| **Job** | 一次针对数据集的、已编译 recipe 的执行。 |
| **Node** | job 的 operation graph 中一个 operator 实例。 |
| **Artifact** | 节点产出的任何摘要寻址数据。 |
| **Candidate** | 某个目标对象的一个生成替换变体。 |
| **Scene context** | 关于目标对象及其周围环境的实测事实。 |
| **Gate** | 针对某个指标的一条验收阈值。 |
| **Acceptance policy** | 候选进入数据集必须通过的门槛集合。 |
| **Decision** | gate 阶段的结果：accepted、rejected 或 needs review。 |
| **Assessment** | 某个评估器关于候选或样本的单次实测证据。 |
| **Evidence** | 为某个决策提供依据的执行事实记录。 |
| **Provenance** | 从输出样本回溯到源、recipe、种子与 backend 的链接。 |
| **Lineage** | 源样本与其增强后代之间的父子关系。 |

## 模块布局（每个名字都由本项目拥有）

```
vidliner/
  domain/        带类型对象、reason、策略模型
  core/          图引擎、身份、规范序列化、结果、错误、随机性
  pipeline/      recipe 加载、编译器、规划门面、job 编排
  operators/     每个领域操作一个模块
  capabilities/  协议定义（抽象端口）
  backends/      具体适配器（唯一允许出现厂商代码的地方）
  runtime/       profile 加载、注册表、引擎、容量、估算
  quality/       指标、门槛、验收策略引擎
  annotations/   内部 IR 转换器（COCO、YOLO）
  storage/       工作区、内容寻址产物存储、SQLite 状态
  control/       数据集发现、split、去重、报告
  cli/           Typer 应用
```

刻意避免：任何 `augforge` 标识（需求中的占位名不是产品名）、任何从其他项目借来的缩写模块名、以及与"并非本项目"的项目一一对应的目录结构。

## 标识符约定

| 类别 | 约定 | 示例 |
| --- | --- | --- |
| 模块 | `snake_case` | `scene_analysis` 风格：`scene.py` |
| 类 | `PascalCase` | `ReplacementPlanner` |
| 公开函数 | 动词开头 `snake_case` | `assemble_operation_graph` |
| 能力名 | 点分、带版本 | `quality.background_preservation.v1` |
| Backend 名 | `snake_case` | `heuristic_detector` |
| Operator 名 | 点分动词 | `generate.replacement` |
| 派生 ID 前缀 | 短、有文档 | `j_`、`s_`、`o_`、`c_`、`n_` |
| Reason code | `SCREAMING_SNAKE` | `BACKGROUND_CHANGED` |
| 环境变量 | `VIDLINER_` 前缀 | `VIDLINER_WORKSPACE` |

ID 前缀：`j_` job、`s_` sample、`o_` object、`c_` candidate、`n_` node、`r_` run。
产物本身以摘要寻址，不使用别名前缀。
