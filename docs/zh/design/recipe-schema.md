# VidLiner —— Recipe Schema

[English](../../design/recipe-schema.md)

Recipe 是一份 YAML 文档，由 `vidliner.domain.recipe.Recipe` 校验。`vidliner recipe schema` 输出生成的 JSON Schema。

```yaml
# 身份
recipe: car-swap            # 必填，slug
description: swap cars      # 可选

dataset:
  input: ./data/source      # 必填，工作区相对或绝对路径
  format: coco-instance     # 可选：coco-detection | coco-instance | yolo-detection | yolo-segmentation
  splits:                   # 可选：如何读取 split 信息
    mode: directory         # directory | filename | manifest | none
    names: [train, val, test]
  augment_splits: [train]   # 默认 [train]；默认拒绝增强 test/val

target:                     # 必填
  classes: [car]            # 必填，非空
  min_score: 0.35           # 检测置信度下限
  min_area_px: 1024         # 忽略过小对象
  max_per_sample: 2         # 每图目标数上限，同时也是图预留的对象分支数
  top_k_by: score           # score | area
  strategy: all             # all | largest | first | explicit
  object_ids: []            # strategy == explicit 时使用

replacement:                # 必填
  mode: strict              # strict | creative
  strategy: category        # category | description | reference
  values:                   # strategy == category 时使用
    - sedan
    - suv
    - pickup
  description: ""           # strategy == description 时使用
  references: []            # strategy == reference 时使用：参考图路径
  candidates_per_object: 3
  seed: 20240517            # 可选；命令行 --seed 会覆盖

preserve:                   # 必须保持不变的东西
  background: strict        # strict | balanced | loose
  geometry: true
  lighting: true
  pose: true
  scale: true
  position: true
  occlusion: true
  geometry_tolerance:
    centroid_shift_max: 0.08      # 相对图像对角线的比例
    area_ratio_min: 0.7
    area_ratio_max: 1.45
    aspect_ratio_delta_max: 0.30
    ground_contact_max_px: 24

refine:                     # 后处理链，有序
  steps:
    - operator: refine.mask_edges
      config: { feather_px: 3, close_px: 5 }
    - operator: refine.alpha_blend
    - operator: refine.harmonize
      config: { strength: 0.45 }

quality:
  minimum_overall: 0.82
  hard_gates:               # 任一项失败即拒绝，无论总分多高
    semantic_match: 0.90
    background_preservation: 0.93
    annotation_consistency: 0.95
  warn_gates:
    artifact_free: 0.85
  maximum_artifact_score: 0.15    # 反向指标：越小越好
  minimum_target_presence: 0.60
  review_band: 0.03               # 距硬门槛该裕度内 → NEEDS_REVIEW
  temporal:                       # 预留；图片输入下被忽略
    minimum_identity_stability: 0.9
    maximum_flicker: 0.1
  weights: {}                     # 可选，覆盖总分权重
  evaluators: []                  # 可选，显式指定评估器 backend 名

acceptance:
  on_quality_reject: keep_diagnostics   # keep_diagnostics | discard
  export_review: false                  # 是否把 NEEDS_REVIEW 也导出
  allow_creative_in_dataset: false      # creative 默认不进入正式数据
  allow_demo_backends: false            # 关键能力为占位实现时是否仍允许导出

duplicates:
  enabled: true
  perceptual_hash: phash          # phash | ahash | dhash | none
  hamming_threshold: 6
  compare_against_source: true

export:
  format: coco-instance           # coco-detection | coco-instance | yolo-detection | yolo-segmentation
  path: ./data/output
  splits: { train: train }
  copy_images: true
  image_format: same              # same | png | jpg
  include_rejected: false
  include_provenance: true
  include_review_report: true

limits:
  max_samples: 0                  # 0 = 全部
  max_candidates_total: 0
  per_node_timeout_s: 600
  job_timeout_s: 0
  fail_fast: false
  resume: true
  cache: true

estimates:                        # 可选，供 plan --dry-run 使用的用户估算
  currency: USD
  generation_unit_cost: 0.04
  generation_unit_seconds: 12
```

## 校验规则

1. `strategy: category` 时 `replacement.values` 必填且非空；取值会被去空白、去重。
2. `strategy: reference` 至少需要一个存在的文件。
3. `candidates_per_object` 取值 1–16。
4. 除非 recipe 设置 `allow_test_augmentation: true`，`augment_splits` 不得包含 `test` / `validation`；而在 `mode: strict` 下即使显式允许也会被拒绝——严格模式下防泄漏不可关闭。
5. 硬门槛指标名必须存在于指标注册表。
6. `mode: creative` 要求 `acceptance.allow_creative_in_dataset` 显式为 `true`，并在 manifest 中记为数据完整性警示。
7. 任何 `refine.steps[].operator` 都必须是已注册的 operator 名；未知名称在**加载 recipe 时**失败，而不是运行中途。
8. 数值做区间校验（分数 `0..1`，像素数 `>0`）。
9. `allow_demo_backends` 是"绑定中含演示占位实现"的 job 唯一能被导出的途径。否则会在预检与导出两次以 `DEMO_BACKEND_NOT_ALLOWED` 拒绝，manifest 会同时记录该开关与涉事能力。详见 `docs/zh/backends.md`。

## 哈希

`recipe_hash = sha256(canonical_json(recipe.model_dump(exclude={'estimates'})))`。
估算被排除，因为它们是标注辅助而非实验的一部分；但包含估算的完整快照仍会存进 job manifest 以供审计。
