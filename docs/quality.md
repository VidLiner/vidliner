# Quality and acceptance

[简体中文](./zh/quality.md)

The quality stage answers one question: **can this generated sample safely become training data?**

It never asks "did the generator return an image". A generator returning success is not evidence that
anything usable came back.

---

## The five steps

```
Generate  →  Re-detect  →  Verify  →  Re-annotate  →  Validate  →  Accept
```

Each step is a stage in the graph, and each has an operator whose evidence is recorded:

1. **Generate** — a candidate image per `(sample, object, candidate)`.
2. **Re-detect** — `verify.redetect` finds the object **in the generated image** and re-segments it.
   The bbox, polygon, and area come from that mask.
3. **Verify** — the evaluate stage measures seven metrics against evidence.
4. **Re-annotate** — the annotate stage rebuilds the label from the re-detected mask.
5. **Validate** — the acceptance policy turns measurements into a decision.
6. **Accept** — only `ACCEPTED` candidates reach the dataset.

### Two masks, two meanings

The single most important distinction in this stage:

| Mask | What it is | What it is used for |
| --- | --- | --- |
| **input mask** | the region handed to the generator | the *only* area where change was authorised, so it is what `background_preservation` excludes |
| **regenerated mask** | the object the detector found in the output | `target_presence`, `mask_boundary`, `geometry`, `annotation_consistency`, and the exported label |

Using the input mask for the second column is the mistake that makes a pipeline open: a generator
that ignores the mask, moves the object, or returns the frame untouched then passes every check,
because the mask being compared is the one it was given. `verify.redetect` exists so that this cannot
happen, and `tests/pipeline/test_verification_loop.py` proves it end to end with a backend that
deletes the object.

---

## Metrics

Every metric is a number in `[0, 1]` and is expressed so that **higher is better**, including
`artifact_free` (which is *freedom from* artifacts). One direction for the whole vocabulary removes a
class of sign errors from gate arithmetic.

| Metric | Catch |
| --- | --- |
| `semantic_match` | the replacement is not the requested category |
| `target_presence` | nothing the detector can see occupies the replaced region |
| `background_preservation` | the scene changed outside the target |
| `mask_boundary` | a seam, a halo, or a cut-through at the mask edge |
| `geometry` | the object moved, resized, changed aspect, or lost ground contact |
| `artifact_free` | duplication, floating objects, text, deformation |
| `annotation_consistency` | the rebuilt label is empty, out of bounds, too small, or mismatched |

Video adds `temporal_consistency` (identity stability, mask stability, position continuity, flicker);
the metric exists in the vocabulary and is ignored for image inputs.

### `background_preservation`

Computed from three measurements outside the target mask, with a feather band excluded so a
legitimate blend seam is not counted as a background change:

```
ssim_outside       structural similarity of the non-target region
changed_ratio      fraction of non-target pixels whose delta exceeds 0.10
perceptual_delta   1 - (mean Lab-ish delta / 0.25), after light smoothing

background_preservation = 0.5*ssim_outside + 0.3*(1 - changed_ratio) + 0.2*perceptual_delta
```

A `changed_ratio` above `quality.background_change_ceiling` (default `0.06`) rejects **immediately**
with `BACKGROUND_CHANGED`, before the weighted value is considered. This is the "large-area
background change" rule, and it is deliberately blunt.

### `geometry`

```
geometry = 1 - max(
    centroid_shift / centroid_shift_max,
    |log(area_ratio)| / |log(bound)|,
    aspect_delta / aspect_delta_max,
    ground_contact_shift_px / ground_contact_max_px
)
```

clamped to `[0, 1]`. Each component that exceeds its tolerance also emits a specific reason code
(`GEOMETRY_CENTROID_SHIFT`, `GEOMETRY_SCALE_CHANGE`, `GEOMETRY_ASPECT_CHANGE`,
`GEOMETRY_GROUND_CONTACT`).

### `artifact_free`

From a local measurement (fragmented mask components, floating objects, seam clutter, text-like
texture) and, when a `quality.vision_evaluation.v1` backend is bound, from its structured report. The
two are blended 70/30 in favour of the evaluator: a vision-language model is a better judge of
semantics and a worse one of reproducibility.

### `annotation_consistency`

Scored from the annotation the pipeline is *about to write*: bounding box inside the image, mask
area above the minimum, a polygon with at least three points, and the category matching the intent.

---

## The overall score

```
overall = Σ weight[m] · normalise(m) / Σ weight[m]
```

over the metrics that were actually measured. Default weights are
`semantic_match 0.24, target_presence 0.16, background_preservation 0.22, mask_boundary 0.10,
geometry 0.12, artifact_free 0.10, annotation_consistency 0.06`, renormalised over the metrics
present. A missing evaluator therefore lowers confidence in the score rather than silently counting
as a failure — unless `allow_missing_metrics: false`, which is the right setting for a production run
where a silent evaluator must not look like a pass.

---

## Decisions

The order of evaluation is fixed:

1. **Hard gates.** Any failure rejects and is listed in `failed_hard_gates`, whatever the overall
   score is.
2. **`minimum_overall`.**
3. **The review band.** A hard-gate miss of no more than `review_band` (default `0.03`) *and* an
   overall score at or above the minimum becomes `NEEDS_REVIEW` instead of `REJECTED`.

The outcome is one of:

| Decision | Meaning |
| --- | --- |
| `ACCEPTED` | exported |
| `REJECTED` | not exported; diagnostics kept |
| `NEEDS_REVIEW` | a human decides; not exported unless `acceptance.export_review` is set |

---

## Reason codes

Every decision carries machine-readable codes emitted by exactly one rule:

```
SEMANTIC_MISMATCH  WRONG_CLASS  OBJECT_NOT_FOUND  BACKGROUND_CHANGED  BACKGROUND_NOISE
MASK_INVALID  MASK_EMPTY  MASK_BOUNDARY_ARTIFACT  GEOMETRY_VIOLATION
GEOMETRY_CENTROID_SHIFT  GEOMETRY_SCALE_CHANGE  GEOMETRY_ASPECT_CHANGE  GEOMETRY_GROUND_CONTACT
ARTIFACT_DETECTED  ARTIFACT_BOUNDARY_BREAK  ARTIFACT_DUPLICATE_OBJECT  ARTIFACT_FLOATING_OBJECT
ARTIFACT_WATERMARK  ARTIFACT_TEXT  OBJECT_DISAPPEARED  DEFORMATION
ANNOTATION_OUT_OF_BOUNDS  ANNOTATION_TOO_SMALL  ANNOTATION_MISMATCH  ANNOTATION_EMPTY
TEMPORAL_FLICKER  TEMPORAL_IDENTITY_DRIFT  TEMPORAL_DISCONTINUITY  TEMPORAL_TRACK_BREAK
BELOW_MINIMUM_OVERALL  REVIEW_BORDERLINE  DUPLICATE_NEAR  DUPLICATE_EXACT  SPLIT_LEAKAGE
CREATIVE_MODE_NOT_ALLOWED  SOURCE_SPLIT_NOT_ALLOWED  EVALUATOR_FAILED  BACKEND_UNAVAILABLE
```

Each code has a category (`quality`, `annotation`, `data`, `execution`) and a documented meaning in
`vidliner/domain/reasons.py`. Codes are part of the public contract and are never renamed without a
migration note.

---

## Tuning without weakening

Two rules keep threshold changes from becoming an excuse to ship bad data:

* **Change the recipe, then re-score.** `vidliner qa <job>` re-applies a policy to stored evidence.
  It is free, it is auditable, and it leaves the original decision in the candidate's event history.
* **Never remove a hard gate to raise the acceptance rate.** Lower a threshold deliberately, or move
  the metric to `warn_gates` and accept that you are now relying on the overall score. The dataset
  report shows the acceptance rate per replacement category, so a policy change that doubles the rate
  is visible.

A healthy first run with the built-in synthetic generator accepts only the candidates whose measured
semantics genuinely hold. That is the intended behaviour: the pipeline fails its own gate rather than
producing a dataset that looks complete and is not.

---

## Rejections are not failures

A job in which every candidate was rejected `SUCCEEDED`. The counters distinguish:

```
samples 10  targets 10  generated 30  accepted 0  rejected 30  review 0  failed 0
```

Investigate with `vidliner qa <job> --show 10`, which prints each candidate's state, overall score,
and reason codes. If the reason is `SEMANTIC_MISMATCH` for every candidate, the generator is wrong,
not the data. If it is `BACKGROUND_CHANGED`, the generator is repainting the scene. If it is
`GEOMETRY_SCALE_CHANGE`, the mask expansion or the model's object scale is off.
