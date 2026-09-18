# Datasets, splits, and integrity

[简体中文](./zh/datasets.md)

An augmentation pipeline can make a dataset worse in three specific ways. This document is about how
VidLiner prevents each one.

---

## 1. Leakage

A model evaluated on a near-copy of something it trained on reports a score it has not earned.
Augmentation creates exactly that risk.

**Every sample carries lineage:**

```python
SampleLineage(root_digest, ancestors, augmentation_depth, parent_sample_id)
```

**Augmentation is restricted to an allow-list.** `dataset.augment_splits` defaults to `[train]`.
Naming `test` or `validation` is refused unless `allow_test_augmentation: true` is also set, and is
refused entirely in `replacement.mode: strict`.

**At export, the whole graph is checked.** Every lineage edge recorded in `lineage_edges` is
validated, and any edge whose endpoints disagree about their split fails the export:

```
ValidationFailure [SPLIT_LEAKAGE]: 3 augmentation descendant(s) would cross a split boundary:
  a_19f0... (train) <- s_04a1... (validation); ...
```

The check runs across the *whole accepted set*, not sample by sample, because a single mismatched
edge is enough to invalidate the dataset.

**How splits are discovered:**

```yaml
splits:
  mode: directory          # train/ val/ test/ subdirectories
  names: [train, val, test]

splits:
  mode: filename           # train_img_001.png

splits:
  mode: manifest           # an explicit relative_path → split document
  manifest: ./splits.json

splits:
  mode: none               # everything is train
```

`write_split_manifest` (used by the tests and available as a helper) records a discovered assignment
so a later run reproduces it exactly.

---

## 2. Duplicate flooding

A dataset filled with near-identical variants teaches memorisation, not the concept.

Duplicate control runs at export, with two complementary checks:

| Check | Availability | Catches |
| --- | --- | --- |
| perceptual hash | always | recompression, scaling, mild colour shifts |
| embedding similarity | when `quality.embedding.v1` is bound | semantic duplicates a hash cannot see |

```yaml
duplicates:
  enabled: true
  perceptual_hash: phash        # phash | ahash | dhash | none
  hamming_threshold: 6
  compare_against_source: true
  embedding_threshold: 0.98
```

Three hash algorithms are implemented with numpy alone, so duplicate control works in a deployment
with no embedding model: `phash` (a 32-point DCT low-frequency hash, the default), `ahash`, and
`dhash`.

Every drop is recorded with its reference and distance in the `duplicates` table and in the dataset
manifest's `excluded` section, so the dataset report can explain exactly what was removed and why.

---

## 3. Distribution drift

The point of augmentation is not to make a dataset bigger; it is to change its distribution
deliberately. That has to be measured.

Every export writes `dataset-report.json`:

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
  "aspect_ratio_distribution": { "portrait(0.5-0.9)": 12, "square(0.9-1.1)": 40, "landscape(1.1-2.0)": 66 },
  "replacement_distribution": { "sedan": 41, "suv": 39, "pickup": 38 },
  "acceptance_by_category": {
    "sedan": { "accepted": 41, "rejected": 3, "review": 0, "acceptance_rate": 0.93 }
  },
  "reason_code_histogram": { "BACKGROUND_CHANGED": 5, "GEOMETRY_SCALE_CHANGE": 3 },
  "quality_averages": { "semantic_match": 0.94, "background_preservation": 0.98 },
  "duplicate_count": 0
}
```

Read it before you train. An acceptance rate that collapses for one category means the generator is
bad at that category, and the reason histogram tells you which gate caught it.

---

## The exported dataset

```
output/
  images/                   accepted samples only, named by candidate id
  annotations/
    instances_train.json    COCO, or labels/ + classes.txt + data.yaml for YOLO
  provenance/
    c_04a1....json          one record per accepted sample
  manifest.json             counts, categories, seed tree, exclusions, per-sample summary
  dataset-report.json       the distribution report
```

Only `ACCEPTED` candidates are written. `NEEDS_REVIEW` is excluded unless the recipe sets
`acceptance.export_review: true`, and `REJECTED` never appears in `images/` — its diagnostics stay in
`runs/<job>/samples/`.

---

## Provenance

The manifest is designed so an auditor needs no other file. Each accepted sample's record contains:

| Group | Fields |
| --- | --- |
| source | `source_sample_id`, `source_digest`, `source_path`, `split`, `lineage_root`, `ancestors`, `augmentation_depth` |
| intent | `recipe_name`, `recipe_hash`, `job_id`, `job_seed`, `target_object_id`, `target_class`, `replacement_category`, `replacement_description`, `replacement_mode`, `intent` |
| seeds | `job_seed`, `sample_seed`, `target_seed`, `candidate_seed` |
| code | `operator_versions`, `backend_ids`, `model_ids`, `generation_parameters` |
| output | `output_digest`, `output_path`, `output_shape`, `annotation_path`, `annotation_format`, `annotation_object_count` |
| quality | `quality_scores`, `overall_score`, `decision`, `reason_codes`, `policy_hash`, `duplicate_of`, `perceptual_hash` |
| time | `created_at`, `vidliner_version` |

Credentials never appear. A name-based check (`find_secret_keys`) rejects any record containing a
credential-like field, and a test asserts the property end to end.

---

## Reproducing a sample

Everything needed is in the record and the manifest:

```
job_seed  →  sample_seed  →  target_seed  →  candidate_seed
```

The manifest stores the sample-level seed tree, and provenance stores the candidate seed. Re-running
the same recipe against the same source bytes reproduces the identifiers, and therefore hits the node
cache for every deterministic step and re-requests the same generation seed from the backend.

If the source file changes, ingest refuses to continue with `ARTIFACT_DIGEST_MISMATCH` rather than
augmenting bytes that no longer match what was recorded.

---

## Annotation correctness

The rule the product depends on: **an annotation is never copied across a replacement.**

* the replaced object's box and polygon are recomputed from the regenerated mask
  (`source: "regenerated"`);
* every other object's label is inherited after a check that its pixels are still visible — an object
  fully covered by the replacement is legitimately gone, one with any pixel outside is preserved
  (`source: "inherited"`);
* the rebuilt bundle is validated before export: boxes inside the image, non-zero area, masks
  matching the frame, polygon vertices present, category vocabulary consistent.

A structural problem raises `ANNOTATION_INVALID` or `ANNOTATION_TOO_SMALL`, not a silently broken
label.
