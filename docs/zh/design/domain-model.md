# VidLiner —— 领域模型

[English](../../design/domain-model.md)

下列对象都是 `vidliner.domain` 中的 Pydantic v2 模型。除非特别说明，值都是不可变的（frozen）：流水线产出新对象，而不是修改共享对象。

---

## 1. 媒体与样本

| 对象 | 用途 | 关键字段 |
| --- | --- | --- |
| `MediaAsset` | 一个不可变的源文件，永不修改。 | `path`（工作区相对路径）、`kind`（`image`/`video`）、`digest`（字节的 SHA-256）、`size_bytes`、`media_type`、`shape`、`video`、`exif` |
| `VideoSpec` | 视频容器事实，为 Phase 2 预留。 | `fps`、`frame_count`、`duration_s`、`codec` |
| `FrameRef` | 定位媒体资产中的某一帧。 | `asset_digest`、`frame_index`、`timestamp_s` |
| `DatasetSample` | 一个训练单元：媒体资产 + 标注 + lineage。 | `sample_id`、`asset`、`in_split`、`annotation`、`lineage`、`origin` |
| `SampleLineage` | 防泄漏。 | `root_digest`、`ancestors`、`augmentation_depth`、`parent_sample_id` |
| `DatasetRef` | 已发现的数据集目录及其 manifest。 | `root`、`samples`、`format`、`splits` |

`sample_id` 是确定性派生的：`sha256(root_digest + ":" + 相对路径)[:16]`，前缀 `s_`。同一文件在不同路径下副本：资产相同（摘要相同）但样本不同；lineage 校验使用摘要，因此泄漏依然会被抓到。

---

## 2. 帧内对象

| 对象 | 用途 | 关键字段 |
| --- | --- | --- |
| `BoundingBox` | 像素空间框 `xyxy`，裁剪到图像，附 `normalized` 辅助方法。 | `x_min`、`y_min`、`x_max`、`y_max` |
| `PolygonMask` | 一个或多个闭合环（像素坐标）。 | `rings: list[list[Point]]` |
| `BinaryMask` | 紧凑的 mask 描述 + `MaskRef`。 | `mask`、`shape`、`area_px`、`coverage` |
| `MaskRef` | 指向已存 mask 产物的摘要引用。 | `artifact`、`encoding`（`png-l8`/`npy-bool`/`rle`） |
| `DepthMap` | 预留；摘要寻址的深度产物 + 量程。 | `depth`、`min_depth`、`max_depth`、`metric` |
| `ObjectInstance` | 某一帧内的一个已检测/已分割对象。 | `object_id`、`class_name`、`bbox`、`score`、`mask_ref`、`polygon`、`attributes`、`source_frame`、`track_id`、`area_px` |
| `ObjectItem` | 一个目标对象 + 它来自的那次选择的上下文。 | `instance`、`sample_id`、`selection_key`、`classes_requested`、`neighbours`、`ordinal` |
| `ObjectTrack` | 单个实例的跨帧关联，为 Phase 2 预留。 | `track_id`、`class_name`、`observations`、`first_frame`、`last_frame`、`visibility`、`reentry_count` |
| `TrackObservation` | 某一帧对该 track 的证据。 | `frame_index`、`bbox`、`mask_ref`、`visibility`、`score` |
| `SceneContext` | 供 planner 使用的环境事实。 | `object_category`、`orientation_deg`、`scale_class`、`lighting`、`shadow`、`depth_order`、`occlusions`、`background_summary`、`ground_contact` |

`object_id` 为 `sha256(资产摘要 + 帧序号 + 类名 + 量化后的框 + 量化后的分数)[:12]`，前缀 `o_`。因此在同一张图上重跑检测会得到相同的对象身份，进而得到相同的节点身份（ADR-016）。

框与分数在哈希**之前**被量化：检测器一像素的抖动或 1% 的置信度波动，不应该为"显然是同一个对象"创造全新身份及全新的缓存血缘。这是刻意且有文档记录的行为。

---

## 3. 替换

| 对象 | 用途 | 关键字段 |
| --- | --- | --- |
| `ReplacementIntent` | *用户想要什么*，与任何模型无关。 | `target_object`、`replacement_category`、`replacement_description`、`preserve_pose/scale/position/lighting/occlusion`、`allowed_geometry_change`、`mode`（`strict`/`creative`）、`seed` |
| `ReplacementPlan` | Planner 针对某个目标对象的决策。 | `intent`、`mask_expansion_px`、`depth_expansion_px`、`positive_prompt`、`negative_constraints`、`expected_category`、`geometry_budget`、`candidate_specs`、`plan_seed`、`planner_id` |
| `CandidateSpec` | 一个待生成的变体。 | `candidate_key`、`description`、`category`、`seed`、`reference_artifact` |
| `GenerationRequest` | 每个生成 backend 都必须接受的统一请求。 | `source_image`、`target_mask`、`context_crop`、`positive_prompt`、`negative_constraints`、`candidate_key`、`seed`、`reference_image`、`expected_category`、`parameters` |
| `GenerationOutcome` | 统一回复。 | `image`、`backend_id`、`model_id`、`seed`、`duration_ms`、`cost_estimate`、`raw_metadata` |
| `ReplacementCandidate` | 一个具体候选及其生命周期。 | `candidate_id`、`sample_id`、`target_object_id`、`spec`、`state`、`generated`、`refined`、`quality`、`annotation`、`decision`、`artifacts` |
| `RefinementRecipe` / `RefinementStep` | 后处理的声明式描述。 | `steps`，每步指名一个精修 operator 及其配置 |
| `AugmentedSample` | 已通过（或待复核）的输出样本。 | `sample_id`、`source_sample_id`、`candidate_id`、`image`、`annotation`、`lineage`、`quality_summary`、`provenance` |

`candidate_id` 为 `sha256(sample_id + target_object_id + candidate_key)[:16]`，前缀 `c_`。

**重要：** plan 的 `expected_category` 是"整体期望"，而每个候选在自己的 `CandidateSpec.category` 中携带具体取值（例如 recipe 的 `values: [sedan, suv, pickup]` 会产出三个类别各一个候选，而不是三个 sedan）。标注重建读的是**候选自己的类别**——标注必须描述真实存在的图像，而不是计划里的第一个建议。

---

## 4. 标注（内部 IR）

| 对象 | 用途 |
| --- | --- |
| `AnnotationBundle` | 某个样本的格式无关标注集合：`image_shape`、`categories`、`objects`、`source_format`、`attributes` |
| `AnnotationObject` | 一个带标签的对象：`object_id`、`category_id`、`bbox`、`polygon`、`mask_ref`、`score`、`iscrowd`、`track_id`、`attributes`、`source`（`regenerated`/`inherited`/`manual`） |
| `CategorySpec` | `category_id`、`name`、`supercategory` |
| `ExportProfile` | 请求的输出格式与选项：`format`、`path`、`copy_images`、`min_area_px`、`include_score` |

IR 是 operator 唯一读写的东西。COCO 与 YOLO 支持都是围绕它实现的转换器（`vidliner.annotations`），因此新增格式永远不会触碰流水线。

---

## 5. Job、证据与溯源

| 对象 | 用途 | 关键字段 |
| --- | --- | --- |
| `JobManifest` | 一次 job 尝试的不可变记录。 | `job_id`、`recipe_name/hash/snapshot`、`runtime_snapshot`、`seed`、`created_at`、`finished_at`、`state`、`counters`、`node_count`、`backend_ids`、`operator_versions` |
| `OperatorEvidence` | 某个节点执行的证据。 | `node_id`、`operator`、`operator_version`、`backend_id`、`input_digests`、`output_digests`、`started_at`、`duration_ms`、`status`、`attempt`、`cache_hit`、`error_code` |
| `QualityReport` | 逐候选的质量证据。 | `candidate_id`、`metrics`、`outcome`、`evaluators`、`evaluated_at`、`auxiliary` |
| `MetricOutcome` | 单个指标的实测值与门槛比较。 | `metric`、`value`、`threshold`、`comparison`、`severity`、`status`、`reason_codes`、`evaluator_id`、`detail` |
| `AcceptanceDecision` | `state`（`accepted`/`rejected`/`needs_review`）、`reason_codes`、`policy_hash`、`failed_hard_gates` |
| `ProvenanceRecord` | 每个通过样本写出的跨产物链接。 | 源样本/摘要、recipe 哈希、job id、目标对象、意图、种子、operator 版本、backend/model 标识、生成参数、质量分、决策、输出摘要、时间戳 |
| `ReasonCode` | 机器可读拒绝/复核原因的封闭枚举。 | 见 `vidliner/domain/reasons.py` |

---

## 6. 运行时配置对象

| 对象 | 用途 |
| --- | --- |
| `RuntimeProfile` | `name`、`device`、`backends`、`bindings`、`concurrency`、`storage`、`estimates` |
| `BackendSpec` | `use`（导入路径）、`options`、`credentials`（引用）、`capacity` |
| `CredentialRef` | `source`（`env`/`file`/`value`）、`name`；解析只发生在 backend 层 |
| `CapacityPolicy` | `limit`、可选 `period_s`（速率预算）、`weight` |
| `Estimate` | `unit_cost`、`unit_seconds`、`currency`、`external`；供 `plan`/`--dry-run` 使用 |

---

## 7. 身份与确定性规则

1. 每个摘要都是原始字节的小写十六进制 SHA-256。
2. 每个派生对象 ID 都是 `前缀 + sha256(规范化 JSON(payload))[:n]`。
3. 规范化 JSON：键排序、无空白、`None` 省略、浮点用最短往返表示、`-0.0` 归一为 `0.0`。
4. `LineageKey` 用冒号连接且从不重排：`sample:<id>`、`object:<id>`、`candidate:<key>`。
5. 种子派生：`seed(parent, label) = int(sha256(f"{parent}|{label}").hexdigest()[:16], 16)`。Job 种子来自 recipe 或命令行；除此之外没有任何东西引入熵。
