# 数据集、split 与完整性

数据增强流水线会以三种特定方式把数据集变差。本文说明 VidLiner 如何逐一防止。

---

## 1. 泄漏（Leakage）

模型在"训练样本的近副本"上被评估，会得到它并不配得到的高分。增强恰恰制造这种风险。

**每个样本都带 lineage：**

```python
SampleLineage(root_digest, ancestors, augmentation_depth, parent_sample_id)
```

**增强只能作用于白名单。** `dataset.augment_splits` 默认 `[train]`。写入 `test` 或 `validation` 时，若不同时设置 `allow_test_augmentation: true` 会被拒绝，而在 `replacement.mode: strict` 下一律拒绝。

**导出时校验整张图。** `lineage_edges` 中记录的每一条血缘边都会被校验，任一端点 split 不一致即令导出失败：

```
ValidationFailure [SPLIT_LEAKAGE]: 3 augmentation descendant(s) would cross a split boundary:
  a_19f0... (train) <- s_04a1... (validation); ...
```

该检查针对**整个通过集合**运行，而不是逐样本，因为一条不匹配的边就足以让数据集失效。

**split 的发现方式：**

```yaml
splits:
  mode: directory          # train/ val/ test/ 子目录
  names: [train, val, test]

splits:
  mode: filename           # train_img_001.png

splits:
  mode: manifest           # 显式的 相对路径 → split 文档
  manifest: ./splits.json

splits:
  mode: none               # 全部视为 train
```

`write_split_manifest` 会把发现结果落盘，使后续运行能精确复现同一分配。

---

## 2. 近重复泛滥

被近似变体灌满的数据集教给模型的是记忆，而不是概念。

去重在导出时执行，包含两项互补检查：

| 检查 | 可用性 | 能发现 |
| --- | --- | --- |
| 感知哈希 | 始终可用 | 重压缩、缩放、轻微色彩偏移 |
| embedding 相似度 | 绑定 `quality.embedding.v1` 时 | 哈希看不见的语义重复 |

```yaml
duplicates:
  enabled: true
  perceptual_hash: phash        # phash | ahash | dhash | none
  hamming_threshold: 6
  compare_against_source: true
  embedding_threshold: 0.98
```

三种哈希算法仅用 numpy 实现，因此在没有 embedding 模型的部署里去重依然可用：`phash`（32 点 DCT 低频哈希，默认）、`ahash`、`dhash`。

每一次剔除都会把参照对象与距离记入 `duplicates` 表以及数据集 manifest 的 `excluded` 段，因此数据集报告可以解释到底删掉了什么、为什么删。

---

## 3. 分布漂移

增强的意义不是把数据集变大，而是**有意识地改变它的分布**。那就必须被测量。

每次导出都会写出 `dataset-report.json`：

```json
{
  "format": "vidliner.dataset-report@1",
  "counts": {
    "accepted": 118, "rejected": 8, "review": 0, "duplicates": 0, "objects": 118
  },
  "class_distribution": { "sedan": 41, "suv": 39, "pickup": 38 },
  "object_size_distribution": {
    "small(<96^2)": 22, "medium(<224^2)": 71, "large(>=224^2)": 25
  },
  "aspect_ratio_distribution": {
    "portrait(0.5-0.9)": 12, "square(0.9-1.1)": 40, "landscape(1.1-2.0)": 66
  },
  "replacement_distribution": { "sedan": 41, "suv": 39, "pickup": 38 },
  "acceptance_by_category": {
    "sedan": { "accepted": 41, "rejected": 3, "review": 0, "acceptance_rate": 0.93 }
  },
  "reason_code_histogram": { "BACKGROUND_CHANGED": 5, "GEOMETRY_SCALE_CHANGE": 3 },
  "quality_averages": { "semantic_match": 0.94, "background_preservation": 0.98 },
  "duplicate_count": 0
}
```

开始训练前先读它。某个类别的通过率骤降，说明生成模型在该类别上不行，而 reason 直方图会告诉你被哪道门槛拦下。

---

## 导出的数据集

```
output/
  images/                  仅通过验收的样本，按 candidate id 命名
  annotations/
    instances_train.json   COCO；YOLO 则为 labels/ + classes.txt + data.yaml
  provenance/
    c_04a1....json         每个通过样本一条记录
  manifest.json            计数、类别、种子树、被排除项、逐样本摘要
  dataset-report.json      分布报告
```

只有 `ACCEPTED` 候选会被写入。`NEEDS_REVIEW` 除非 recipe 设置 `acceptance.export_review: true`，否则不导出；`REJECTED` 永远不会出现在 `images/` 中——它的诊断证据留在 `runs/<job>/samples/`。

图像文件名与标注条目来自同一份映射（在 `_write_images` 中构建），因此"标注指向的文件名"和"实际写出的文件名"不可能不一致。

---

## 溯源（Provenance）

Manifest 的设计目标是：审计者不需要任何其他文件。每条通过样本的记录包含：

| 分组 | 字段 |
| --- | --- |
| 来源 | `source_sample_id`、`source_digest`、`source_path`、`split`、`lineage_root`、`ancestors`、`augmentation_depth` |
| 意图 | `recipe_name`、`recipe_hash`、`job_id`、`job_seed`、`target_object_id`、`target_class`、`replacement_category`、`replacement_description`、`replacement_mode`、`intent` |
| 种子 | `job_seed`、`sample_seed`、`target_seed`、`candidate_seed` |
| 代码 | `operator_versions`、`backend_ids`、`model_ids`、`generation_parameters` |
| 产出 | `output_digest`、`output_path`、`output_shape`、`annotation_path`、`annotation_format`、`annotation_object_count` |
| 质量 | `quality_scores`、`overall_score`、`decision`、`reason_codes`、`policy_hash`、`duplicate_of`、`perceptual_hash` |
| 时间 | `created_at`、`vidliner_version` |

凭据永不出现。基于字段名的检查（`find_secret_keys`）会拒绝任何含疑似凭据字段的记录，并有测试端到端断言这一性质。

---

## 复现某个样本

所需信息都在记录与 manifest 中：

```
job_seed  →  sample_seed  →  target_seed  →  candidate_seed
```

Manifest 保存样本级种子树，溯源保存候选种子。对同一份源字节重跑同一 recipe 会复现全部标识，因此每个确定性步骤都命中节点缓存，并向 backend 请求同一个生成种子。

如果源文件发生变化，ingest 会以 `ARTIFACT_DIGEST_MISMATCH` 拒绝继续，而不是去增强与记录不再匹配的字节。

---

## 标注正确性

产品赖以成立的规则：**标注永不跨替换复制。**

* 被替换对象的框与多边形由重建后的 mask 重新计算（`source: "regenerated"`）；
* 其他对象的标签在确认其像素仍然可见后才被继承——被替换区域完全覆盖的对象是合理消失，只要还有任意像素在替换区域之外就视为保留（`source: "inherited"`）；
* 重建后的 bundle 在导出前会被校验：框在图像内、面积非零、mask 与帧尺寸一致、多边形顶点存在、类别词表自洽。

结构性问题会抛出 `ANNOTATION_INVALID` 或 `ANNOTATION_TOO_SMALL`，而不是留下一个静默损坏的标签。

导出时所有样本的类别 id 会被**重映射到全数据集统一的类别词表**（`_rebase_categories`）：每个样本自身的 bundle 有自己的编号历史，而导出的数据集必须对全部样本使用同一套编号，否则同一个类别 id 在不同文件里含义不同。
