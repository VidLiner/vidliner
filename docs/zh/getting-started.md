# 快速上手

这份走查大约五分钟，不需要下载模型、不需要 API Key、不需要联网。最终你会得到一个带标注、带溯源、带质量报告的数据集。

---

## 1. 安装

```bash
pip install -e ".[dev]"
vidliner --help
```

如果构建隔离阶段因为网络失败：

```bash
pip install -e . --no-build-isolation
```

## 2. 创建工作区

工作区是 VidLiner 自己拥有的目录：产物存储、job 状态与运行目录都在里面。源图片建议放在其中（也可以放在 profile 允许的其他位置）。

```bash
mkdir -p playground
vidliner init ./playground
```

得到：

```
playground/
  runtime.yaml     哪个 backend 提供哪项能力
  state.db         job、node、candidate 与缓存状态
  artifacts/       内容寻址的输出
  runs/            每个 job 一个目录
  datasets/        默认导出位置
```

起始 profile 把每项能力都绑到内置本地 backend，因此流水线立刻可跑。用
`vidliner backend check --workspace ./playground` 查看绑定情况。

## 3. 准备数据

把图片放进 `playground/data/source/`。如果有标注，按你的格式要求的布局放置：

```
playground/data/source/
  train/            # split 目录（也可用文件名前缀或 manifest）
    img_001.png
    img_002.png
  annotations/
    instances_train.json      # COCO
```

也可以完全没有标注：ingest 与 detection 都不需要它。标注只影响"未改动对象的标签继承"。

动手之前先看清楚：

```bash
vidliner inspect ./playground/data/source --format coco-instance
```

```
dataset   playground/data/source
samples   42
splits    train=34, val=8
augment   train
images    42   dimensions 640x480, 1280x720
annotated 42/42 samples, 96 objects
classes   car=96
```

## 4. 写 recipe

```bash
vidliner recipe init ./playground/car-swap.yaml
```

至少调整 `dataset.input`、`target.classes`、`replacement.values`，然后校验：

```bash
vidliner recipe validate ./playground/car-swap.yaml
```

校验器是刻意严格的：未知指标名、没有占位符的 `prompt_template`、要求增强 `test`，全都是**错误**而不是警告——因为每一条都会产出"看起来没问题、其实有问题"的数据集。

## 5. 花时间之前先规划

```bash
vidliner plan ./playground/car-swap.yaml --workspace ./playground
```

```
job        j_1f0c9a2b7d4e5f60
graph      462 nodes, hash 9c1d2e3f4a5b
stages     ingest=42, detect=42, select=42, segment=42, scene=42, plan=42,
           generate=126, refine=126, evaluate=126, annotate=126, export=126
samples    42  candidates 126
external   0 call(s)  accelerator ops 0
estimate   6.30s, cost n/a
  vision.object_detection.v1                   -> builtin_detector
  planning.replacement.v1                      -> builtin_planner
  generation.object_replacement.v1             -> builtin_replacement
  ...
```

规划只解析绑定并统计节点，**不会调用任何生成 backend**——这就是它可以在你还没做任何决定前安全运行的原因。

`plan` 同时写出 `runs/<job>/plan.json`：完整图、每个节点的 lineage 与配置、以及估算。`vidliner run --dry-run` 产出同样的文档后停止。

## 6. 执行

```bash
vidliner run ./playground/car-swap.yaml --workspace ./playground
```

```
job        j_1f0c9a2b7d4e5f60  [succeeded]
candidates 126  accepted 118  rejected 8  review 0  acceptance 94%
nodes      462 ok, 0 failed, 0 cached, 0 resumed
dataset    playground/data/output (coco-instance)
  written  118 image(s), 118 accepted sample(s), 0 duplicate(s) excluded
evidence   playground/runs/j_1f0c9a2b7d4e5f60
```

每个被拒绝的候选都保留了诊断证据。看看原因：

```bash
vidliner qa latest --workspace ./playground --show 5
```

```
evaluated  126 candidate(s); 0 changed decision
counts     accepted=118, rejected=8
  c_04a1...  rejected  0.612  suv          BACKGROUND_CHANGED
  c_1b77...  rejected  0.588  pickup       GEOMETRY_SCALE_CHANGE,GEOMETRY_VIOLATION
  ...
```

## 7. 复核需要人工判断的样本

落在复核带内的候选会被标为 `NEEDS_REVIEW`，而不是被悄悄接受或拒绝。它们出现在一个可直接双击打开的静态 HTML 文件里：

```bash
vidliner report review latest --workspace ./playground
open playground/runs/j_1f0c9a2b7d4e5f60/review/index.html
```

每张卡片显示：原始帧、目标 mask、生成候选、放大后的差异图、重建标注、每个指标及其门槛、以及 reason code。决策记录在报告同目录的 `decisions.json` 中——这正是未来 Web UI 要实现的接口。

## 8. 不重新生成也能重新评估

质量阈值是数据。如果你决定把 `background_preservation` 调严，不需要重新生成任何东西：

```bash
# 修改 recipe 里的 quality.hard_gates.background_preservation，然后：
vidliner qa latest --workspace ./playground --recipe ./playground/car-swap.yaml
vidliner export latest --workspace ./playground
```

`qa` 只把策略重新套用到已存的质量证据上，不重算像素、不调用任何 backend。

## 9. 你会得到什么

```
playground/data/output/
  images/                  118 张 PNG —— 只含通过验收的样本
  annotations/
    instances_train.json   重建后的标签，全数据集统一类别词表
  provenance/
    c_04a1....json         源摘要、recipe 哈希、种子、backend、评分、决策
  manifest.json            数据集体检清单：计数、类别、种子树、被排除项
  dataset-report.json      类别 / 尺寸 / 长宽比分布；各类别通过率
```

检查一条溯源记录：

```bash
python -m json.tool playground/data/output/provenance/$(ls playground/data/output/provenance | head -1)
```

它无需任何其他文件即可回答：这个样本来自哪个源样本、由哪份 recipe 和哪个种子产出、跑了哪些 operator 与 backend、模型被要求做什么、每个指标得了几分、最终决定是什么以及为什么、以及何时发生。

## 10. 下一步

* **接入你自己的模型。** 在 `runtime.yaml` 中绑定真实 backend —— 见 `docs/zh/runtime.md` 与 `docs/zh/backends.md`。recipe 与流水线无需改动。
* **理解门槛。** `docs/zh/quality.md` 解释每个指标，以及如何在不削弱门槛的前提下调参。
* **保护你的 split。** `docs/zh/datasets.md` 覆盖 lineage、泄漏与去重。
* **中断与恢复。** 取消一次长跑后用 `vidliner job resume <job-id>` 接回；已完成节点会被复用，不会重算。
* **跑测试。** `pytest -q` 用合成数据跑完整套件，不访问网络。

## 11. 常见问题

**`ModuleNotFoundError: No module named 'vidliner'`**
包没有装进当前环境。在仓库根目录执行 `pip install -e .`（若网络受限则加 `--no-build-isolation`）。直接运行示例脚本也可以，脚本会自行引导导入路径。

**`vidliner backend check` 列出"capabilities with no backend"**
这说明有 recipe 需要但 profile 未绑定的能力。`generation.video_replacement.v1`、`vision.object_tracking.v1`、`vision.depth_estimation.v1` 属于 Phase 2，尚未实现；`quality.vision_evaluation.v1` 是可选的 VLM 评估器。`quality.background_preservation.v1` 属于内置测量，会显示为 `builtin`。

**通过了 0 个样本，但 job 是 succeeded**
这是正常的、且有信息量的结果：系统没有失败，只是样本不合格。用 `vidliner qa <job> --show 10` 看 reason code。全部是 `SEMANTIC_MISMATCH` 说明生成模型不行；`BACKGROUND_CHANGED` 说明它在重绘场景；`GEOMETRY_SCALE_CHANGE` 说明 mask 扩张或对象尺度不对。
