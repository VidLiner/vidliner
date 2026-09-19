# 质量与验收

[English](../quality.md)

质量阶段只回答一个问题：**这个生成样本能否安全地成为训练数据？**

它从不问"生成模型是否返回了图片"。返回成功并不构成任何证据。

---

## 五个步骤

```
Generate  →  Re-detect  →  Verify  →  Re-annotate  →  Validate  →  Accept
生成        重新检测       验证        重建标注         校验          验收
```

每一步都是图中的一个阶段，各有 operator，其证据都会被记录：

1. **Generate** —— 每个 `(样本, 对象, 候选)` 产出一张候选图。
2. **Re-detect** —— `verify.redetect` 在**生成图**中找到该对象并重新分割；bbox、polygon 与面积都由这个 mask 推导。
3. **Verify** —— evaluate 阶段针对证据测量 7 个指标。
4. **Re-annotate** —— annotate 阶段用**重新检测得到的** mask 重建标签。
5. **Validate** —— 验收策略把测量值变成决策。
6. **Accept** —— 只有 `ACCEPTED` 候选进入数据集。

### 两个 mask，两种含义

本阶段最重要的一处区分：

| mask | 是什么 | 用来做什么 |
| --- | --- | --- |
| **输入 mask** | 交给生成器的区域 | **唯一**被授权发生改变的区域，因此它是 `background_preservation` 所排除的部分 |
| **重建 mask** | 检测器在输出中找到的对象 | `target_presence`、`mask_boundary`、`geometry`、`annotation_consistency`，以及导出的标签 |

用输入 mask 去承担第二列的职责，正是让流水线"不闭合"的错误：一个忽略 mask、挪走对象或原图返回的生成器会通过所有检查，因为被比较的 mask 就是它拿到的那一个。`verify.redetect` 的存在就是为了让这件事不可能发生，而 `tests/pipeline/test_verification_loop.py` 用一个"删掉对象"的 backend 端到端地证明了这一点。

---

## 指标

每个指标都是 `[0, 1]` 内的数值，且一律表达为**越大越好**，包括 `artifact_free`（它表示"无伪影的程度"）。整个词汇表统一方向，可以消除门槛运算中的一整类符号错误。

| 指标 | 拦截的问题 |
| --- | --- |
| `semantic_match` | 替换结果不是要求的类别 |
| `target_presence` | 替换区域内没有检测器能看到的对象 |
| `background_preservation` | 目标之外的场景被改动 |
| `mask_boundary` | mask 边缘出现接缝、光晕或切穿 |
| `geometry` | 对象位移、缩放、长宽比变化或失去接地 |
| `artifact_free` | 重复、悬浮、文字、形变 |
| `annotation_consistency` | 重建标签为空、越界、过小或类别不符 |

视频另有 `temporal_consistency`（身份稳定性、mask 稳定性、位置连续性、闪烁）；该指标已存在于词汇表中，图片输入下会被忽略。

### `background_preservation`

在目标 mask 之外由三项测量合成，并排除羽化带，避免把正常的混合接缝算成背景变化：

```
ssim_outside       非目标区域的结构相似度
changed_ratio      差值超过 0.10 的非目标像素比例
perceptual_delta   1 - (平均感知差异 / 0.25)，先做轻度平滑

background_preservation = 0.5*ssim_outside + 0.3*(1 - changed_ratio) + 0.2*perceptual_delta
```

若 `changed_ratio` 超过 `quality.background_change_ceiling`（默认 `0.06`），会**立即**以 `BACKGROUND_CHANGED` 拒绝，甚至不看加权后的分数。这条"大面积背景变化"规则刻意做得很钝。

### `geometry`

```
geometry = 1 - max(
    质心位移 / centroid_shift_max,
    |log(面积比)| / |log(边界)|,
    长宽比变化 / aspect_ratio_delta_max,
    接地行位移(px) / ground_contact_max_px
)
```

裁剪到 `[0, 1]`。每个超出容差的子项还会发出具体 reason code（`GEOMETRY_CENTROID_SHIFT`、`GEOMETRY_SCALE_CHANGE`、`GEOMETRY_ASPECT_CHANGE`、`GEOMETRY_GROUND_CONTACT`）。

### `artifact_free`

来自本地测量（mask 连通分量碎裂、悬浮对象、接缝杂乱、类文字纹理）以及——若绑定了 `quality.vision_evaluation.v1`——其结构化报告。两者按 70/30 融合，权重偏向评估器：VLM 在语义判断上更强，在可复现性上更弱。

### `annotation_consistency`

评分依据是流水线**即将写出**的标注：框在图像内、mask 面积高于下限、多边形至少 3 个点、类别与意图一致。

---

## 总体分

```
overall = Σ 权重[m] · 归一化(m) / Σ 权重[m]
```

求和范围是**实际被测量**的指标。默认权重为
`semantic_match 0.24, target_presence 0.16, background_preservation 0.22, mask_boundary 0.10, geometry 0.12, artifact_free 0.10, annotation_consistency 0.06`，并按实际存在的指标重新归一化。因此某个评估器缺失会降低总分的可信度，而不会静默算作失败——除非设 `allow_missing_metrics: false`，那才是生产环境的正确设置：静默的评估器不能被当成通过。

---

## 决策

评估顺序是固定的：

1. **硬性门槛。** 任一失败即拒绝，并列入 `failed_hard_gates`，与总分无关。
2. **`minimum_overall`。**
3. **复核带。** 与硬门槛的差距不超过 `review_band`（默认 `0.03`）**且**总分达到最低要求时，标为 `NEEDS_REVIEW` 而非 `REJECTED`。

最终结果三选一：

| 决策 | 含义 |
| --- | --- |
| `ACCEPTED` | 被导出 |
| `REJECTED` | 不导出；保留诊断证据 |
| `NEEDS_REVIEW` | 交人工判断；除非 `acceptance.export_review`，否则不导出 |

---

## Reason code

每个决策都带机器可读的 reason code，每条规则只发出固定的若干种：

```
SEMANTIC_MISMATCH  WRONG_CLASS  OBJECT_NOT_FOUND  BACKGROUND_CHANGED
BACKGROUND_NOISE  MASK_INVALID  MASK_EMPTY  MASK_BOUNDARY_ARTIFACT
GEOMETRY_VIOLATION  GEOMETRY_CENTROID_SHIFT  GEOMETRY_SCALE_CHANGE  GEOMETRY_ASPECT_CHANGE
GEOMETRY_GROUND_CONTACT  ARTIFACT_DETECTED  ARTIFACT_BOUNDARY_BREAK  ARTIFACT_DUPLICATE_OBJECT
ARTIFACT_FLOATING_OBJECT  ARTIFACT_WATERMARK  ARTIFACT_TEXT  OBJECT_DISAPPEARED
DEFORMATION  ANNOTATION_OUT_OF_BOUNDS  ANNOTATION_TOO_SMALL  ANNOTATION_MISMATCH
ANNOTATION_EMPTY  TEMPORAL_FLICKER  TEMPORAL_IDENTITY_DRIFT  TEMPORAL_DISCONTINUITY
TEMPORAL_TRACK_BREAK  BELOW_MINIMUM_OVERALL  REVIEW_BORDERLINE  DEMO_BACKEND_NOT_ALLOWED
DUPLICATE_NEAR  DUPLICATE_EXACT  SPLIT_LEAKAGE  CREATIVE_MODE_NOT_ALLOWED
SOURCE_SPLIT_NOT_ALLOWED  EVALUATOR_FAILED  BACKEND_UNAVAILABLE
```

每个码都有归属类别（`quality`、`annotation`、`data`、`execution`）与文档化含义，定义在 `vidliner/domain/reasons.py`。这些码是公开契约的一部分，未经迁移说明不得改名。

---

## 如何调参而不放水

两条规则防止"改阈值"变成"放宽质量"：

* **改 recipe，然后重评分。** `vidliner qa <job>` 把策略重新套用到已存证据上。它免费、可审计，并会在候选的事件历史中保留原始决策。
* **绝不通过删除硬门槛来提升通过率。** 要么有意识地降低某个阈值，要么把该指标移到 `warn_gates` 并接受"现在依赖总分"这一事实。数据集报告会给出各替换类别的通过率，因此让通过率翻倍的策略改动是藏不住的。

用内置合成生成器做第一次运行时，只有那些"测得的语义确实成立"的候选会被接受。这是预期行为：流水线宁可让自己过不了门槛，也不产出一个看起来完整、实际有问题的数据集。

---

## 被拒绝不等于失败

一个所有候选都被拒绝的 job 状态是 `SUCCEEDED`。计数器会把它们区分开：

```
samples 10  targets 10  generated 30  accepted 0  rejected 30  review 0  failed 0
```

用 `vidliner qa <job> --show 10` 排查：它会打印每个候选的状态、总分与 reason code。如果所有候选都是 `SEMANTIC_MISMATCH`，那是生成模型的问题，不是数据的问题；如果是 `BACKGROUND_CHANGED`，说明模型在重绘场景；如果是 `GEOMETRY_SCALE_CHANGE`，检查 mask 扩张或模型输出的对象尺度。
