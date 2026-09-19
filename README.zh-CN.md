# VidLiner

<p align="center">
  <img src="./brand/assets/vidliner-logo-dark.svg" alt="VidLiner — 可证明的合成数据" width="420" />
</p>

<p align="center">
  <strong>生成。验证。重建标签。导出证据。</strong><br />
  <sub>早期 Alpha · 图像流水线 MVP · 视频能力规划中</sub>
</p>

[English](./README.md)

**以对象为中心、强制验证的合成训练数据生产流水线。**

VidLiner 把你已有的图片变成**经过验证的**训练样本：定位目标对象 → 替换它 → **在生成图上重新检测并分割该对象** → 证明画面其余部分未被改动 → 用真实存在的像素重建标注 → 按显式验收策略评估 → 只导出通过验收的样本。

它交付的不是"更漂亮的图片"，而是：

```
训练图像  +  正确标签  +  生成溯源  +  质量证据
```

任何一个候选样本如果无法安全地成为训练数据，就会被拒绝，并且拒绝原因是机器可读的。

> **项目状态：** VidLiner 仍处于早期 Alpha。内置 profile 为演示和测试提供确定性、无外部依赖的全流程；在真实训练数据集使用前，请绑定生产级 backend。品牌资产见[品牌工具包](./brand/README.zh-CN.md)，首发文案见[发布素材](./brand/launch/README.zh-CN.md)。

---

## 它解决什么问题

合成数据增强有三种典型危害，VidLiner 在结构上逐一封堵：

| 危害 | VidLiner 的做法 |
| --- | --- |
| **标签错误。** 对象的尺寸、形状或类别变了，却沿用原来的框。 | 被替换对象的标注**重新计算**（基于生成后的像素），标签绝不跨替换复制。 |
| **背景漂移。** 生成模型悄悄重绘了场景，模型学到错误的背景不变性。 | 在目标 mask（含羽化带）之外测量 SSIM、像素变化比例与感知差异；存在大面积背景变化时，硬性门槛直接拒绝。 |
| **数据集泄漏与近重复泛滥。** 验证集里出现训练图的近副本，或被近似变体灌满。 | 每个样本都带 lineage（血缘），增强只能作用于白名单内的 split，导出时用感知哈希做去重。 |

---

## 安装

需要 Python 3.11 或更高版本。**无需下载模型、无需付费 API**，开箱即可跑通全流程：

```bash
pip install -e ".[dev]"
vidliner --help
```

如果 `pip install` 的构建隔离阶段因为网络失败，可用环境里已有的 setuptools 直接构建：

```bash
pip install -e . --no-build-isolation
```

可选依赖：

* `pip install -e ".[cv]"` 安装 OpenCV，供需要更快图像运算的 backend 使用；
* `pip install -e ".[http]"` 安装 `httpx`，通用 HTTP 替换 backend 需要它。

> 也可以不安装直接运行示例：`python examples/car-swap/demo.py`（脚本会自行引导导入路径）。

---

## 快速开始

```bash
# 1. 创建工作区：目录、状态数据库、起始 runtime profile
vidliner init ./workspace

# 2. 先看清楚数据，再动手
vidliner inspect ./workspace/data/source --format coco-instance

# 3. 写 recipe（或从带注释的模板开始）
vidliner recipe init ./workspace/car-swap.yaml

# 4. 校验，并查看它将要做什么——不执行任何生成
vidliner recipe validate ./workspace/car-swap.yaml
vidliner plan ./workspace/car-swap.yaml --workspace ./workspace

# 5. 执行：通过的样本被导出，被拒绝的保留诊断证据
vidliner run ./workspace/car-swap.yaml --workspace ./workspace

# 6. 复核、重评分、导出
vidliner jobs --workspace ./workspace
vidliner qa latest --workspace ./workspace --show 5
vidliner report review latest --workspace ./workspace
vidliner export latest --workspace ./workspace --format coco-instance
```

产物：

```
workspace/data/output/
  images/             仅通过验收的样本
  annotations/        重建后的标签（指定格式）
  provenance/         每个通过样本一条 JSON 生成溯源记录
  manifest.json       数据集体检清单：种子、backend、评分
  dataset-report.json 类别 / 尺寸 / 长宽比分布，以及各类别通过率
```

---

## 架构一页速览

```
Recipe (YAML)                     用户想要什么
     │
     ├── 由编译器产出 ────────────────────────────────────────────┐
     ▼                                                           │
Operation Graph (DAG)             ingest → detect → select → segment → scene → plan
     │                                  → generate → refine → verify → evaluate → annotate → export
     │
     ▼  由引擎执行
Engine（asyncio、有界并发、可缓存、可恢复）
     │  每个节点经由此派发
     ▼
Operator  ──需要──▶  Capability  ──由 profile 绑定到──▶  Backend
（领域操作）          （抽象能力）                        （唯一允许出现厂商代码的地方）
     │
     ▼
Artifacts（内容寻址） + State（SQLite） + Reports（文件）
```

五个支点：

1. **Recipe 只声明意图，编译器产出带类型的图。** 这让 `plan` 与 `--dry-run` 完全免费，也让局部重跑成为可能。
2. **Operator 只声明能力，永不指名厂商。** 由 runtime profile 决定谁来提供 `vision.object_detection.v1`，因此同一份 recipe 既能跑本地模型，也能跑本地推理服务或云端服务，无需修改。
3. **一切内容寻址、一切可缓存。** 节点缓存键是 `(operator, version, implementation, config, 输入摘要, seed)`，所以重跑只复用仍然有效的工作，恢复也只恢复该恢复的部分。
4. **质量是一组门槛，不是一个分数。** 硬性门槛失败即拒绝，与总体分多高无关——一个错误标签比丢掉一个样本更糟。
5. **系统失败与样本不合格是两件事。** 全部候选都被拒绝的 job 依然是"成功"的，计数器会如实说明。

完整设计见 `ARCHITECTURE.zh-CN.md`，决策理由见 `docs/zh/design/decisions.md`。

---

## Recipe 示例

```yaml
recipe: car-swap

dataset:
  input: ./data/source
  format: coco-instance
  splits: { mode: directory, names: [train, val, test] }
  augment_splits: [train]          # 除非显式允许，val/test 的增强会被拒绝

target:
  classes: [car]
  min_score: 0.35
  min_area_px: 1024
  max_per_sample: 2

replacement:
  mode: strict                     # strict | creative（creative 默认不进入正式数据集）
  strategy: category
  values: [sedan, suv, pickup]
  candidates_per_object: 3
  seed: 20240517

preserve:
  background: strict
  geometry: true
  lighting: true

quality:
  minimum_overall: 0.82
  hard_gates:
    semantic_match: 0.90
    background_preservation: 0.93
    annotation_consistency: 0.95
  maximum_artifact_score: 0.15
  minimum_target_presence: 0.60
  review_band: 0.03

export:
  format: coco-instance
  path: ./data/output
```

`vidliner recipe schema` 输出完整 JSON Schema；逐字段说明见 `docs/zh/recipe.md`。

---

## 命令行

| 命令 | 用途 |
| --- | --- |
| `vidliner init` | 创建工作区、状态数据库与起始 runtime profile。 |
| `vidliner inspect <dir>` | 报告媒体类型、尺寸、split 与标注覆盖率。 |
| `vidliner recipe validate\|schema\|init\|show` | 处理 recipe。 |
| `vidliner plan <recipe>` | 编译并估算开销。**不会调用任何生成 backend。** |
| `vidliner run <recipe>` | 执行、评估、导出。`--dry-run` 在生成前停止。 |
| `vidliner jobs` | 列出 job 及其计数器与通过率。 |
| `vidliner job show\|resume\|cancel` | 查看、恢复或取消 job。 |
| `vidliner qa <job>` | 用存储的质量证据重新套用验收策略。**不重算任何像素。** |
| `vidliner export <job>` | 把通过的样本导出为数据集。 |
| `vidliner backend list\|check` | 查看能力清单并探测 backend（绝不发起生成调用）。 |
| `vidliner report review\|summary\|events` | 重新生成报告、读取结构化事件日志。 |

退出码：`0` 成功；`1` 结果是"否"（能力未绑定、无样本通过）；`2` 命令无法执行（recipe 非法、路径无效）。

---

## 支持的格式

**输入** —— `jpg`、`jpeg`、`png`、`webp`；`mp4` 会被识别并以明确的 Phase 2 错误拒绝。标注：COCO detection / instance-segmentation、YOLO detection / segmentation，或完全没有标注。

**输出** —— COCO detection、COCO instance segmentation、YOLO detection、YOLO segmentation。

新增格式只需在 `vidliner/annotations/` 增加一个转换器；因为一切都流经同一套内部标注 IR，流水线本身无需改动。

---

## 质量门槛

`vidliner run` 从不问"生成模型是否返回了图片"，而是问：这个结果能否成为训练数据。

| 指标 | 方向 | 拦截的问题 |
| --- | --- | --- |
| `semantic_match` | 越大越好 | 替换结果不是要求的类别 |
| `target_presence` | 越大越好 | 在生成图中找不到该对象 |
| `background_preservation` | 越大越好 | mask 之外的场景被改动 |
| `mask_boundary` | 越大越好 | 可见接缝、光晕或"切穿" |
| `geometry` | 越大越好 | 对象位移、缩放、离地 |
| `artifact_free` | 越大越好 | 重复对象、悬浮物、文字水印、形变 |
| `annotation_consistency` | 越大越好 | 重建标签为空、越界或类别不符 |

硬性门槛失败即拒绝，与总体分无关；落在复核带内的边缘样本标为 `NEEDS_REVIEW`，由静态 HTML 报告交给人工判定。详见 `docs/zh/quality.md`。

---

## 演示 backend 不允许导出

新克隆仓库拿到的 profile 绑定的是显著性启发式、合成生成器与颜色评估器，因此整条流水线无需下载模型、无需 API Key 就能跑。它们每一项都是**占位实现**，用它们产出的数据集"长得像"训练数据，却不是训练数据。

因此 job 会跑，但导出会被拒绝：

```bash
vidliner plan examples/car-swap/car-swap.yaml
# DEMONSTRATION stack: the export would be refused
#   vision.object_detection.v1       -> builtin_detector (demonstration stand-in)
#   vision.instance_segmentation.v1  -> builtin_segmenter (demonstration stand-in)
#   generation.object_replacement.v1 -> builtin_replacement (demonstration stand-in)
#   quality.semantic_match.v1        -> builtin_metrics (demonstration stand-in)
```

这四项能力产出的正是最终进入数据集的数据；planner、refiner、artifact evaluator 出现占位实现，只改变样本是**怎么做出来的**，不阻塞任何东西。为这四项绑定真实 backend，拒绝就会消失。如果这份数据集本来就是用于查看，请在 recipe 里写明 `acceptance.allow_demo_backends: true`——它会连同当时使用的绑定一起被记进 manifest，而此后修改任何 profile 都无法改变那次判定。详见 `docs/zh/backends.md`。

---

## 数据溯源

每个通过验收的样本都带有一条溯源记录：源样本 ID 与摘要、recipe 哈希、job ID、目标对象、替换意图、种子树、operator 与 backend 标识、生成参数、全部质量分、带 reason code 的决策、输出摘要与时间戳。

凭据只以**引用**形式存在（`env:`、`file:`），绝不会出现在 recipe、manifest、日志行或溯源记录中——这一点有端到端测试保证。

---

## 可复现性

一个 job 只有一个根种子，每个节点按身份路径派生自己的种子：
`job_seed → sample_seed → target_seed → candidate_seed`。
因此新增一个候选值不会改变已有候选的种子。节点身份也按同一路径派生，这使得重跑能命中缓存、中断能接着跑。

---

## 文档

| 文档 | 内容 |
| --- | --- |
| `ARCHITECTURE.zh-CN.md` | 完整设计：分层、数据流、扩展点。 |
| `CONTRIBUTING.zh-CN.md` | 如何参与开发，以及保持项目可信的规则。 |
| `docs/zh/getting-started.md` | 逐步走通第一次运行。 |
| `docs/zh/recipe.md` | recipe 全字段与校验规则。 |
| `docs/zh/runtime.md` | runtime profile、绑定、凭据与预检。 |
| `docs/zh/backends.md` | 如何编写 backend；能力契约。 |
| `docs/zh/quality.md` | 指标、门槛、决策与 reason code。 |
| `docs/zh/datasets.md` | split、lineage、去重与数据集报告。 |
| `docs/zh/testing.md` | 测试布局与合成 fixture。 |
| `docs/zh/design/` | 架构决策记录、领域模型、DAG 设计、各类 schema、里程碑。 |
| `examples/README.zh-CN.md` | 可运行的端到端示例说明。 |

英文文档为各文件的同名版本（`README.md`、`ARCHITECTURE.md`、`docs/*.md`）。

---

## 当前状态

MVP 已端到端实现图片流水线：ingest、detection、segmentation、scene analysis、planning、candidate generation、refinement、**生成后重新检测与重新分割**、quality evaluation、annotation regeneration、acceptance、dataset export、job state、resume、cache、CLI 与报告。验证记录见 `docs/zh/design/milestones.md`。

**刻意尚未实现**（但扩展点已就位）：视频替换与跟踪（`ObjectTrack`、`FrameRef`、`temporal_consistency` 已在数据模型中）、分布式 worker、对象存储、Web 复核界面。任何未实现的功能都会 `raise NotImplementedError` 或抛出明确的能力错误，而不是假装成功。

## 许可

MIT，见 `LICENSE`。
