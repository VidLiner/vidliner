# VidLiner —— 质量指标与验收

[English](../../design/quality-metrics.md)

质量阶段回答一个问题：*这个生成样本能否安全地成为训练数据？*
它从不问"生成模型是否返回了图片"。

## 指标

每个指标都是 `[0, 1]` 内的实数，并有明确方向，用 `Comparison` 表达：`at_least`（越大越好）或 `at_most`（越小越好）。伪影类指标使用 `at_most`，其余使用 `at_least`。

| 指标 | 方向 | 含义 | 默认评估器 |
| --- | --- | --- | --- |
| `semantic_match` | at_least | 生成对象符合请求的类别/描述。 | `quality.semantic_match.v1` |
| `target_presence` | at_least | 替换后目标位置确实存在新对象。 | 重新检测 |
| `background_preservation` | at_least | 非目标像素在容差内未被改动。 | 内置（`quality.background_preservation.v1`） |
| `mask_boundary` | at_least | mask 边缘质量：无光晕、无裁切、无切穿。 | 内置 |
| `geometry` | at_least | 质心、面积、长宽比与接地都在容差内。 | 内置 |
| `artifact_free` | at_least | 1 − 伪影严重度（边界断裂、重复、悬浮、形变、水印文字、意外消失）。 | `quality.artifact_detection.v1` |
| `annotation_consistency` | at_least | 重建标注非空、在图内、面积合理、类别匹配。 | 内置 |
| `temporal_consistency` | at_least | 视频专用：身份稳定性、mask 稳定性、位置连续性、闪烁。 | 预留（Phase 2） |

### 复合子指标

`background_preservation` 由三项测量合成，并在 `detail` 中报告：

```
ssim_outside      非目标区域的结构相似度（已施加羽化）
changed_ratio     差值超过感知阈值的非目标像素比例
perceptual_delta  1 - 归一化后的 mask 外平均 Lab 差异
background_preservation = w1*ssim_outside + w2*(1 - changed_ratio) + w3*perceptual_delta
```

默认 `w1=0.5, w2=0.3, w3=0.2`。若 `changed_ratio` 高于
`quality.background_change_ceiling`（默认 `0.06`），会**先于**加权值计算以 `BACKGROUND_CHANGED` 直接拒绝——这就是需求中"大面积背景变化"那条规则。

### geometry

```
geometry = 1 - max(
    质心位移 / centroid_shift_max,
    |log(面积比)| / |log(面积比边界)|,
    长宽比变化 / aspect_ratio_delta_max,
    接地行位移px / ground_contact_max_px
)
```

裁剪到 `[0, 1]`。每个超出容差的子项还会发出专门的 reason code
（`GEOMETRY_CENTROID_SHIFT`、`GEOMETRY_SCALE_CHANGE`、`GEOMETRY_ASPECT_CHANGE`、`GEOMETRY_GROUND_CONTACT`）。

### 总体分

```
overall = Σ weight[m] * normalise(m) / Σ weight[m]
```

求和范围是实际被评估的指标，其中 `normalise` 把 `at_most` 指标映射为 `1 - value`，使总体分始终保持"越大越好"的含义。默认权重为
`semantic_match 0.24, target_presence 0.16, background_preservation 0.22, mask_boundary 0.10, geometry 0.12, artifact_free 0.10, annotation_consistency 0.06`，并按实际存在的指标重新归一化。`temporal_consistency` 仅对视频样本参与。

## 门槛与验收策略

`AcceptancePolicy` 由 recipe 的 `quality` 段构建：

* `minimum_overall` —— 软门槛；未达到即以 `BELOW_MINIMUM_OVERALL` 拒绝。
* `hard_gates` —— 指标 → 阈值映射。失败即拒绝，**与总分无关**，失败项列入 `AcceptanceDecision.failed_hard_gates`。
* `warn_gates` —— 记为警告；仅当总分同时落在 `review_band` 内时才促成 `NEEDS_REVIEW`。
* `maximum_artifact_score` —— `artifact_free` 的反向硬门槛。

评估顺序：先硬门槛，再 `minimum_overall`，最后复核带。第一条决定性规则生效，其 reason code 即为 manifest 记录的内容。

### 复核带

当硬门槛的差距不超过 `review_band`（默认 `0.03`）*且*总分不低于 `minimum_overall` 时，决策为 `NEEDS_REVIEW` 而非 `REJECTED`。复核项除非 `acceptance.export_review` 否则不导出，并且总会出现在静态 HTML 复核报告中。

## Reason code

封闭枚举位于 `vidliner/domain/reasons`。每个码都带 `ReasonCategory`
（`quality`、`annotation`、`data`、`execution`），并由唯一一条规则发出。

```
SEMANTIC_MISMATCH          WRONG_CLASS                OBJECT_NOT_FOUND
BACKGROUND_CHANGED         BACKGROUND_NOISE           MASK_INVALID
MASK_EMPTY                 MASK_BOUNDARY_ARTIFACT     GEOMETRY_VIOLATION
GEOMETRY_CENTROID_SHIFT    GEOMETRY_SCALE_CHANGE      GEOMETRY_ASPECT_CHANGE
GEOMETRY_GROUND_CONTACT    ARTIFACT_DETECTED          ARTIFACT_BOUNDARY_BREAK
ARTIFACT_DUPLICATE_OBJECT  ARTIFACT_FLOATING_OBJECT   ARTIFACT_WATERMARK
ARTIFACT_TEXT              OBJECT_DISAPPEARED         DEFORMATION
ANNOTATION_OUT_OF_BOUNDS   ANNOTATION_TOO_SMALL       ANNOTATION_MISMATCH
ANNOTATION_EMPTY           TEMPORAL_FLICKER           TEMPORAL_IDENTITY_DRIFT
TEMPORAL_DISCONTINUITY     TEMPORAL_TRACK_BREAK       BELOW_MINIMUM_OVERALL
REVIEW_BORDERLINE          DEMO_BACKEND_NOT_ALLOWED   DUPLICATE_NEAR
DUPLICATE_EXACT            SPLIT_LEAKAGE              CREATIVE_MODE_NOT_ALLOWED
SOURCE_SPLIT_NOT_ALLOWED   EVALUATOR_FAILED           BACKEND_UNAVAILABLE
```

## 质量阶段**不**做的事

* 它不判断 job 是否成功。全部候选都被拒的 job 依然可以是 `SUCCEEDED`；计数器会说
  `accepted=0, rejected=N`。
* 它不静默丢弃证据。被拒候选保留产物、指标行与 HTML 复核条目，因此可以在不重新生成的前提下用
  `vidliner qa <job>` 重新评估以调参。
* 它不信任生成器自身的元数据。backend 的 `raw_metadata` 会被记录以供审计，但永远不会成为门槛的输入。
