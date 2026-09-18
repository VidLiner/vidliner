# Recipe（数据增强配方）

Recipe 是一份 YAML 文档，只表达**你想要什么**。它从不指名模型、服务或凭据，也不包含任何会解析到工作区之外的路径。

校验：

```bash
vidliner recipe validate car-swap.yaml
```

输出机器可读的 schema：`vidliner recipe schema`。

---

## 顶层

```yaml
recipe: car-swap            # 必填，不含空白的 slug
description: swap cars      # 可选，出现在报告中

dataset: {...}              # 必填
target: {...}               # 必填
replacement: {...}          # 必填
preserve: {...}             # 可选，默认值见下
refine: {...}               # 可选
quality: {...}              # 可选
acceptance: {...}           # 可选
duplicates: {...}           # 可选
export: {...}               # 可选
limits: {...}               # 可选
estimates: {...}            # 可选，不计入 recipe 哈希
```

---

## `dataset`

| 字段 | 默认 | 含义 |
| --- | --- | --- |
| `input` | — | 数据集目录。相对路径相对工作区解析。 |
| `format` | `null` | 源标注格式：`coco-detection`、`coco-instance`、`yolo-detection`、`yolo-segmentation`；纯图片目录可省略。 |
| `splits.mode` | `none` | `none`、`directory`、`filename`、`manifest`。 |
| `splits.names` | `[train, val, test]` | 用于识别各 split 的目录名或文件名前缀。 |
| `splits.manifest` | `null` | `相对路径 → split` 的 JSON 文档，`mode: manifest` 时必填。 |
| `augment_splits` | `[train]` | 允许被增强的 split。 |
| `allow_test_augmentation` | `false` | 显式允许增强 `test` / `validation`。 |
| `recursive` | `true` | 是否递归子目录。 |
| `max_samples` | `0` | 发现样本数上限，`0` 表示不限。 |

名为 `annotations`、`labels`、`provenance`、`review`、`reports` 的子目录永远不会被当作样本。

**Split 安全。** 在 `augment_splits` 中写 `test` 或 `validation`，若不同时设置 `allow_test_augmentation: true` 会被拒绝；而在 `replacement.mode: strict` 下**一律拒绝**——严格模式下防泄漏不可被选择性关闭。导出时还会再次校验每一条 lineage 边，端点 split 不一致即以 `SPLIT_LEAKAGE` 失败。

---

## `target`

| 字段 | 默认 | 含义 |
| --- | --- | --- |
| `classes` | — | 待替换的类别列表，非空。会被去空白、去重、排序。 |
| `min_score` | `0.35` | 检测置信度下限。 |
| `min_area_px` | `1024` | 忽略更小的对象。 |
| `max_per_sample` | `2` | 每张图目标数上限，同时也是图预留的对象分支数。 |
| `top_k_by` | `score` | `score` 或 `area`；同分按 object id 排序，保证选择确定性。 |
| `strategy` | `all` | `all`、`largest`、`first`、`explicit`。 |
| `object_ids` | `[]` | `strategy: explicit` 时必填。 |

> 内置显著性检测器只报告它认识的类别，**并且**这些类别必须真的出现在源标注里。请求数据集从未标注过的类别会得到 0 个目标——那是一次成功但无事可做的 job，而不是伪造出的检测结果。

---

## `replacement`

| 字段 | 默认 | 含义 |
| --- | --- | --- |
| `mode` | `strict` | 生产训练数据用 `strict`；探索用 `creative`。 |
| `strategy` | `category` | `category`、`description`、`reference`。 |
| `values` | `[]` | `strategy: category` 时必填。 |
| `description` | `""` | `strategy: description` 时必填。 |
| `references` | `[]` | `strategy: reference` 时必填。 |
| `candidates_per_object` | `3` | 取值 1–16。每个候选对应一个图节点。 |
| `seed` | `null` | 根种子。命令行 `--seed` 会覆盖它。 |
| `mask_expansion_px` | `0` | 生成前扩张 mask（阴影、接地边缘）。 |
| `context_padding_px` | `32` | 传给生成 backend 的上下文裁剪边距。 |
| `prompt_template` | 见下 | 必须包含 `{category}` 或 `{description}`。 |
| `negative_constraints` | 见下 | 传给生成 backend 的负向约束。 |

默认 prompt 模板：

```
Replace the {category} in the masked region with {description}. Keep the background, camera
viewpoint, lighting direction, shadows, and ground contact unchanged.
```

另外还可用 `{source}`（源类别）、`{lighting}`、`{background}`，它们由测量得到的 scene context 填充。

**strict 与 creative。** 在 `strict` 模式下，计划会保持对象的姿态、尺度、位置、光照与遮挡，几何预算很紧。在 `creative` 模式下几何预算被放宽、强度降低，且除非显式设置 `acceptance.allow_creative_in_dataset: true`，creative 替换永远不会进入正式数据集。

---

## `preserve`（必须保持不变的东西）

| 字段 | 默认 | 含义 |
| --- | --- | --- |
| `background` | `strict` | `strict`、`balanced`、`loose`。 |
| `geometry` | `true` | 约束质心、面积、长宽比与接地。 |
| `lighting` | `true` | 要求尊重测量到的光照。 |
| `pose`、`scale`、`position`、`occlusion` | `true` | 额外传给 planner 的保持开关。 |
| `geometry_tolerance_centroid` | `0.08` | 质心位移上限（相对图像对角线）。 |
| `geometry_tolerance_area_min` / `_max` | `0.70` / `1.45` | 允许的面积比区间。 |
| `geometry_tolerance_aspect` | `0.30` | 长宽比变化上限。 |
| `geometry_tolerance_ground_px` | `24` | 接地行位移上限（像素）。 |

---

## `refine`（后处理链）

精修与生成是分开的阶段，因此换更好的 harmonizer 不需要重新生成。

```yaml
refine:
  steps:
    - operator: refine.mask_edges
      config: { feather_px: 3, close_px: 5, dilate_px: 0 }
    - operator: refine.alpha_blend
    - operator: refine.harmonize
      config: { strength: 0.45 }
      enabled: true
```

每个 `operator` 必须是已注册的精修名；未知名称会在加载 recipe 时就校验失败。合成器会把它在羽化 mask 之外触碰到的任何像素恢复为原值，因此混合不可能"漏"到背景上。

---

## `quality`

| 字段 | 默认 | 含义 |
| --- | --- | --- |
| `minimum_overall` | `0.82` | 加权总分软门槛。 |
| `hard_gates` | `{semantic_match: 0.90, background_preservation: 0.93, annotation_consistency: 0.95}` | 任一失败即拒绝，与总分无关。 |
| `warn_gates` | `{}` | 失败只记警告。 |
| `maximum_artifact_score` | `0.15` | 伪影证据上限；会转成 `artifact_free` 门槛。 |
| `minimum_target_presence` | `0.60` | 替换区域必须真的存在对象。 |
| `review_band` | `0.03` | 距硬门槛这个裕度内 → `NEEDS_REVIEW`。 |
| `background_change_ceiling` | `0.06` | 允许变化的非目标像素比例上限。 |
| `weights` | `{}` | 可选，覆盖总分中各指标权重。 |
| `evaluators` | `[]` | 可选，显式指定评估器 backend 名。 |
| `allow_missing_metrics` | `true` | 为 `false` 时，没有测量值的指标直接拒绝。 |
| `temporal` | 见下 | 视频专用门槛，图片输入下会被记录但忽略。 |

门槛里的指标名必须存在于指标注册表，否则校验失败；同一指标不能同时出现在 `hard_gates` 与 `warn_gates`。各指标含义见 `docs/zh/quality.md`。

---

## `acceptance`

| 字段 | 默认 | 含义 |
| --- | --- | --- |
| `on_quality_reject` | `keep_diagnostics` | 保留被拒证据，或 `discard`。 |
| `export_review` | `false` | 是否把 `NEEDS_REVIEW` 也导出。 |
| `allow_creative_in_dataset` | `false` | 是否允许 creative 替换通过验收。 |

---

## `duplicates`

| 字段 | 默认 | 含义 |
| --- | --- | --- |
| `enabled` | `true` | 导出时执行去重。 |
| `perceptual_hash` | `phash` | `phash`、`ahash`、`dhash`、`none`。 |
| `hamming_threshold` | `6` | 低于该指纹距离视为近重复。 |
| `compare_against_source` | `true` | 除已通过输出外，也比对源数据集。 |
| `embedding_backend` | `null` | 可选的 embedding backend，用于语义级去重。 |
| `embedding_threshold` | `0.98` | embedding 余弦相似度高于该值视为重复。 |

---

## `export`

| 字段 | 默认 | 含义 |
| --- | --- | --- |
| `format` | `coco-instance` | 输出标注格式。 |
| `path` | `dataset` | 输出目录，在工作区内解析。 |
| `splits` | `{}` | 输出 split 命名。 |
| `copy_images` | `true` | 是否把通过的图像复制进数据集。 |
| `image_format` | `same` | `same`、`png`、`jpg`。 |
| `include_rejected` | `false` | 是否也导出被拒候选（仅诊断用）。 |
| `include_provenance` | `true` | 每个通过样本写一条溯源记录。 |
| `include_review_report` | `true` | 写静态 HTML 复核报告。 |
| `min_annotation_area_px` | `4` | 小于该面积的标注被拒绝。 |

---

## `limits`

| 字段 | 默认 | 含义 |
| --- | --- | --- |
| `max_samples` | `0` | 处理样本数上限。 |
| `max_candidates_total` | `0` | 候选总数上限。 |
| `per_node_timeout_s` | `600` | 单次节点尝试超时。 |
| `job_timeout_s` | `0` | 整个 job 的时间预算，`0` 表示不限。 |
| `fail_fast` | `false` | 遇到第一个节点失败即停止。 |
| `resume` | `true` | 复用上一次尝试中已完成的节点。 |
| `cache` | `true` | 使用节点结果缓存。 |
| `workers` | `4` | 最大并发节点数。 |

---

## `estimates`（仅用于规划）

只影响规划展示。这些值**不计入 recipe 哈希**，因为改一个成本估算并不改变流水线的产出。

```yaml
estimates:
  currency: USD
  generation_unit_cost: 0.04
  generation_unit_seconds: 12
```

---

## 哈希与身份

```
recipe_hash = sha256(canonical_json(不含 estimates 的 recipe))
job_id      = sha256(recipe_hash, seed, 排序后的样本 id)
```

因此同一份 recipe 对同一批样本的两次运行会共用同一个 job id，互相恢复。要强制全新一次尝试，用 `vidliner run --new` 或显式 `--job-id`。
