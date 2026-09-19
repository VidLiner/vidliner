# VidLiner — Recipe Schema

A recipe is a single YAML document validated by `vidliner.domain.recipe.Recipe`.
`vidliner recipe schema` prints the generated JSON Schema.

```yaml
# identity
recipe: car-swap            # required, slug
description: swap cars      # optional

dataset:
  input: ./data/source      # required, workspace-relative or absolute
  format: coco-instance     # optional: coco-detection | coco-instance | yolo-detection | yolo-segmentation
  splits:                   # optional: how to read split information
    mode: directory         # directory | filename | manifest | none
    names: [train, val, test]
  augment_splits: [train]   # default [train]; test/val augmentation is refused by default

target:                     # required
  classes: [car]            # required, non-empty
  min_score: 0.35           # detection confidence floor
  min_area_px: 1024         # ignore tiny objects
  max_per_sample: 2         # cap targets per image
  top_k_by: score           # score | area
  strategy: all             # all | largest | first | explicit
  object_ids: []            # used when strategy == explicit

replacement:                # required
  mode: strict              # strict | creative
  strategy: category        # category | description | reference
  values:                   # for category strategy
    - sedan
    - suv
    - pickup
  description: ""           # for description strategy
  references: []            # for reference strategy: paths to reference images
  candidates_per_object: 3
  seed: 20240517            # optional; CLI --seed overrides

preserve:                   # what must not change
  background: strict        # strict | balanced | loose
  geometry: true
  lighting: true
  pose: true
  scale: true
  position: true
  occlusion: true
  geometry_tolerance:
    centroid_shift_max: 0.08      # fraction of image diagonal
    area_ratio_min: 0.7
    area_ratio_max: 1.45
    aspect_ratio_delta_max: 0.30
    ground_contact_max_px: 24

refine:                     # post-processing pipeline, ordered
  steps:
    - operator: refine.mask_edges
      config: { feather_px: 3, close_px: 5 }
    - operator: refine.alpha_blend
    - operator: refine.harmonize
      config: { strength: 0.45 }

quality:
  minimum_overall: 0.82
  hard_gates:               # failing any of these rejects, whatever the overall score is
    semantic_match: 0.90
    background_preservation: 0.93
    annotation_consistency: 0.95
  warn_gates:
    artifact_free: 0.85
  maximum_artifact_score: 0.15    # inverted metric: lower is better
  minimum_target_presence: 0.60
  review_band: 0.03               # within this margin of a hard gate -> NEEDS_REVIEW
  temporal:                       # reserved; ignored for image inputs
    minimum_identity_stability: 0.9
    maximum_flicker: 0.1
  weights: {}                     # optional overrides for the overall score
  evaluators: []                  # optional explicit evaluator backend names

acceptance:
  on_quality_reject: keep_diagnostics   # keep_diagnostics | discard
  export_review: false                  # include NEEDS_REVIEW in the dataset output
  allow_creative_in_dataset: false      # creative mode never enters accepted data by default
  allow_demo_backends: false            # permit export when a critical capability is a stand-in

duplicates:
  enabled: true
  perceptual_hash: phash          # phash | ahash | none
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
  max_samples: 0                  # 0 = all
  max_candidates_total: 0
  per_node_timeout_s: 600
  job_timeout_s: 0
  fail_fast: false
  resume: true
  cache: true

estimates:                        # optional user-supplied overrides for plan --dry-run
  currency: USD
  generation_unit_cost: 0.04
  generation_unit_seconds: 12
```

## Validation rules

1. `replacement.values` is required and non-empty for `strategy: category`; each value is trimmed
   and de-duplicated.
2. `strategy: reference` requires at least one existing file in `references`.
3. `candidates_per_object` ≥ 1 and ≤ 16.
4. `augment_splits` may not contain `test` or `validation` unless the recipe sets
   `allow_test_augmentation: true`, which is refused when `mode: strict` — leakage protection is
   not opt-out inside strict mode.
5. Hard-gate metric names must exist in the metric registry.
6. `mode: creative` forces `acceptance.allow_creative_in_dataset` to be explicitly `true` and is
   recorded in the manifest as a dataset-integrity caveat.
7. Any `refine.steps[].operator` must be a registered operator name; unknown names fail validation
   at recipe-load time, not at run time.
8. Numbers are range-checked (`0..1` for scores, `>0` for pixel counts).
9. `allow_demo_backends` is the only way a job whose bindings include a demonstration stand-in may
   be exported. It is refused otherwise with `DEMO_BACKEND_NOT_ALLOWED`, in pre-flight and again at
   export, and the manifest records both the flag and the offending capabilities. See
   `docs/backends.md`.

## Hashing

`recipe_hash = sha256(canonical_json(recipe.model_dump(exclude={'estimates'})))`.
Estimates are excluded because they are labelling aids, not part of the experiment. The unhashed
snapshot including estimates is still stored in the job manifest for audit.
