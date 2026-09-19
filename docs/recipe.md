# Recipes

[简体中文](./zh/recipe.md)

A recipe is a single YAML document that says *what you want*. It never names a model, a service, or a
credential, and it never contains a path that resolves outside the workspace.

Validate one with:

```bash
vidliner recipe validate car-swap.yaml
```

Print the machine-readable schema with `vidliner recipe schema`.

---

## Top level

```yaml
recipe: car-swap            # required, a slug with no whitespace
description: swap cars      # optional, shown in reports

dataset: {...}              # required
target: {...}               # required
replacement: {...}          # required
preserve: {...}             # optional, defaults shown below
refine: {...}               # optional, defaults shown below
quality: {...}              # optional, defaults shown below
acceptance: {...}           # optional
duplicates: {...}           # optional
export: {...}               # optional
limits: {...}               # optional
estimates: {...}            # optional, excluded from the recipe hash
```

---

## `dataset`

| Field | Default | Meaning |
| --- | --- | --- |
| `input` | — | Dataset directory. Relative paths resolve against the workspace. |
| `format` | `null` | Source annotation format: `coco-detection`, `coco-instance`, `yolo-detection`, `yolo-segmentation`, or omitted for images only. |
| `splits.mode` | `none` | `none`, `directory`, `filename`, or `manifest`. |
| `splits.names` | `[train, val, test]` | Directory names or filename prefixes that identify each split. |
| `splits.manifest` | `null` | `relative_path → split` JSON document, required when `mode: manifest`. |
| `augment_splits` | `[train]` | Splits the job may augment. |
| `allow_test_augmentation` | `false` | Explicit opt-in to augmenting `test` or `validation`. |
| `recursive` | `true` | Walk subdirectories. |
| `max_samples` | `0` | Cap on discovered samples; `0` means all. |

Subdirectories named `annotations`, `labels`, `provenance`, `review`, or `reports` are never treated
as samples.

**Split safety.** Naming `test` or `validation` in `augment_splits` is refused unless
`allow_test_augmentation: true` is also set, and it is refused *entirely* in
`replacement.mode: strict` — leakage protection is not opt-out inside strict mode. At export time
every lineage edge is checked again, and an edge whose endpoints disagree about their split fails the
export with `SPLIT_LEAKAGE`.

---

## `target`

| Field | Default | Meaning |
| --- | --- | --- |
| `classes` | — | Non-empty list of classes to replace. Trimmed, de-duplicated, sorted. |
| `min_score` | `0.35` | Detection confidence floor. |
| `min_area_px` | `1024` | Ignore smaller objects. |
| `max_per_sample` | `2` | Cap on targets per image. Also the number of object branches the graph reserves. |
| `top_k_by` | `score` | `score` or `area`; ties break on object id, so selection is deterministic. |
| `strategy` | `all` | `all`, `largest`, `first`, or `explicit`. |
| `object_ids` | `[]` | Required when `strategy: explicit`. |

> The built-in saliency detector only reports classes it knows *and* that the source annotation
> actually contains. Asking for a class the dataset never labels produces zero targets — a
> successful job with nothing to do, not a fabricated detection.

---

## `replacement`

| Field | Default | Meaning |
| --- | --- | --- |
| `mode` | `strict` | `strict` for dataset production; `creative` for exploration. |
| `strategy` | `category` | `category`, `description`, or `reference`. |
| `values` | `[]` | Required for `strategy: category`. |
| `description` | `""` | Required for `strategy: description`. |
| `references` | `[]` | Required for `strategy: reference`. |
| `candidates_per_object` | `3` | 1–16. One graph node per candidate. |
| `seed` | `null` | Root seed. The CLI `--seed` overrides it. |
| `mask_expansion_px` | `0` | Grow the mask before generation (shadows, contact edges). |
| `context_padding_px` | `32` | Context crop padding handed to the generator. |
| `prompt_template` | see below | Must contain `{category}` or `{description}`. |
| `negative_constraints` | see below | Constraints passed to the generator. |

Default prompt template:

```
Replace the {category} in the masked region with {description}. Keep the background, camera
viewpoint, lighting direction, shadows, and ground contact unchanged.
```

`{source}` (the source class), `{lighting}`, and `{background}` are also available and are filled from
the measured scene context.

**Strict versus creative.** In `strict` mode the plan keeps the object's pose, scale, position,
lighting, and occlusion, and the geometry budget is tight. In `creative` mode the geometry budget is
widened and the intensity is reduced, and creative replacements never enter the accepted dataset
unless `acceptance.allow_creative_in_dataset: true` is set explicitly.

---

## `preserve`

| Field | Default | Meaning |
| --- | --- | --- |
| `background` | `strict` | `strict`, `balanced`, or `loose`. |
| `geometry` | `true` | Constrain centroid, area, aspect, and ground contact. |
| `lighting` | `true` | Require the measured illumination to be respected. |
| `pose`, `scale`, `position`, `occlusion` | `true` | Additional preservation flags passed to the planner. |
| `geometry_tolerance_centroid` | `0.08` | Max centroid movement as a fraction of the image diagonal. |
| `geometry_tolerance_area_min` / `_max` | `0.70` / `1.45` | Allowed area ratio. |
| `geometry_tolerance_aspect` | `0.30` | Max change in aspect ratio. |
| `geometry_tolerance_ground_px` | `24` | Max movement of the ground-contact row, in pixels. |

---

## `refine`

Refinement is a separate stage from generation, so a better harmoniser never requires regenerating.

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

Each `operator` must be a registered refinement name; an unknown one fails validation at recipe-load
time. The compositor restores any pixel it touches outside the feathered mask, so a blend cannot leak
into the background.

---

## `quality`

| Field | Default | Meaning |
| --- | --- | --- |
| `minimum_overall` | `0.82` | Soft gate on the weighted overall score. |
| `hard_gates` | `{semantic_match: 0.90, background_preservation: 0.93, annotation_consistency: 0.95}` | Failing any of these rejects, whatever the overall score is. |
| `warn_gates` | `{}` | Failures recorded as warnings. |
| `maximum_artifact_score` | `0.15` | Ceiling on artifact evidence; becomes an `artifact_free` gate. |
| `minimum_target_presence` | `0.60` | The replaced region must contain an object. |
| `review_band` | `0.03` | Within this margin of a hard gate → `NEEDS_REVIEW`. |
| `background_change_ceiling` | `0.06` | Fraction of non-target pixels that may change at all. |
| `weights` | `{}` | Optional per-metric weights for the overall score. |
| `evaluators` | `[]` | Optional explicit evaluator backend names. |
| `allow_missing_metrics` | `true` | When `false`, a metric with no measurement rejects. |
| `temporal` | see below | Video-only gates, recorded and ignored for images. |

Gate metric names must exist in the metric registry; an unknown name fails validation. A metric may
not appear in both `hard_gates` and `warn_gates`. See `docs/quality.md` for what each metric means.

---

## `acceptance`

| Field | Default | Meaning |
| --- | --- | --- |
| `on_quality_reject` | `keep_diagnostics` | Keep rejected evidence, or `discard` it. |
| `export_review` | `false` | Include `NEEDS_REVIEW` candidates in the exported dataset. |
| `allow_creative_in_dataset` | `false` | Permit creative replacements to be accepted. |
| `allow_demo_backends` | `false` | Permit export when a production-critical capability is served by a demonstration backend. |

`allow_demo_backends` is the one place a demonstration dataset is allowed to exist, and it has to be
written down. With the shipped profile — a saliency detector, a synthetic generator, and a colour
heuristic — the pipeline will run a job end to end and then refuse to export it, because the result
is shaped like training data and is not training data. Setting this to `true` says *I know, this is
inspection output*, and records that fact in the job manifest next to the bindings that were used.
`vidliner plan` reports the same condition without generating anything. See
[Writing a backend](./backends.md#demonstration-backends-and-the-production-guard).

The same acknowledgement can be made for a single command with `vidliner run --allow-demo` or
`vidliner export --allow-demo`; the recipe field is the durable form.

---

## `duplicates`

| Field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Run duplicate control at export. |
| `perceptual_hash` | `phash` | `phash`, `ahash`, `dhash`, or `none`. |
| `hamming_threshold` | `6` | Fingerprint distance below which two images count as near-duplicates. |
| `compare_against_source` | `true` | Also compare against the source set, not only accepted outputs. |
| `embedding_backend` | `null` | Optional embedding backend for semantic duplicates. |
| `embedding_threshold` | `0.98` | Cosine similarity above which embeddings count as duplicates. |

---

## `export`

| Field | Default | Meaning |
| --- | --- | --- |
| `format` | `coco-instance` | Output annotation format. |
| `path` | `dataset` | Output directory, resolved inside the workspace. |
| `splits` | `{}` | Split naming for the output. |
| `copy_images` | `true` | Copy accepted images into the dataset. |
| `image_format` | `same` | `same`, `png`, or `jpg`. |
| `include_rejected` | `false` | Export rejected candidates too (diagnostics only). |
| `include_provenance` | `true` | Write one provenance record per accepted sample. |
| `include_review_report` | `true` | Write the static HTML review report. |
| `min_annotation_area_px` | `4` | Reject an annotation smaller than this. |

---

## `limits`

| Field | Default | Meaning |
| --- | --- | --- |
| `max_samples` | `0` | Cap on samples processed. |
| `max_candidates_total` | `0` | Cap on candidates. |
| `per_node_timeout_s` | `600` | Per-attempt node timeout. |
| `job_timeout_s` | `0` | Whole-job budget; `0` means none. |
| `fail_fast` | `false` | Stop at the first node failure. |
| `resume` | `true` | Reuse finished nodes from a previous attempt. |
| `cache` | `true` | Use the node result cache. |
| `workers` | `4` | Maximum concurrent nodes. |

---

## `estimates`

Planning aids only. These values are excluded from the recipe hash, because changing a cost estimate
does not change what the pipeline produces.

```yaml
estimates:
  currency: USD
  generation_unit_cost: 0.04
  generation_unit_seconds: 12
```

---

## Hashing and identity

```
recipe_hash = sha256(canonical_json(recipe without `estimates`))
job_id      = sha256(recipe_hash, seed, sorted sample ids)
```

Two runs of the same recipe against the same samples therefore share a job id and resume each other.
Pass `vidliner run --new` (or an explicit `--job-id`) to force a fresh attempt.
