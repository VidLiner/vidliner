# VidLiner — Quality Metrics and Acceptance

The quality stage answers one question: *can this generated sample safely become training data?*
It never asks "did the generator return an image".

## Metrics

Every metric is a real number in `[0, 1]` with a declared direction. Direction is expressed as a
`Comparison`: `at_least` (higher is better) or `at_most` (lower is better). The artifact metrics
use `at_most`; everything else uses `at_least`.

| Metric | Direction | Meaning | Default evaluator |
| --- | --- | --- | --- |
| `semantic_match` | at_least | Generated object matches the requested category/description. | `quality.semantic_match.v1` |
| `target_presence` | at_least | A new object actually exists at the target location after replacement. | re-detection |
| `background_preservation` | at_least | Non-target pixels are unchanged within tolerance. | `quality.background_preservation.v1` |
| `mask_boundary` | at_least | Mask edge quality: no halo, no clipped object, no cut-through. | built-in |
| `geometry` | at_least | Centroid, area, aspect ratio, and ground contact stay inside tolerance. | built-in |
| `artifact_free` | at_most | 1 − (severity of boundary breaks, duplicates, floaters, deformation, watermark/text, unintended disappearance). | `quality.artifact_detection.v1` |
| `annotation_consistency` | at_least | Rebuilt annotation is non-empty, inside the image, area-sane, and class-matched. | built-in |
| `temporal_consistency` | at_least | Video-only: identity stability, mask stability, position continuity, flicker. | reserved (Phase 2) |

### Composite sub-metrics

`background_preservation` is computed from three measurements and reported in `detail`:

```
ssim_outside      structural similarity of the non-target region, feather applied
changed_ratio     fraction of non-target pixels with |Δ| above the perceptual threshold
perceptual_delta  1 - normalised mean absolute Lab delta outside the mask
background_preservation = w1*ssim_outside + w2*(1 - changed_ratio) + w3*perceptual_delta
```

Defaults `w1=0.5, w2=0.3, w3=0.2`. A `changed_ratio` above
`quality.background_change_ceiling` (default `0.06`) rejects immediately with
`BACKGROUND_CHANGED`, before the weighted value is even considered — this is the "large-area
background change" rule from the requirements.

### Geometry

```
geometry = 1 - max(
    centroid_shift / centroid_shift_max,
    |log(area_ratio)| / |log(area_ratio_bound)|,
    aspect_delta / aspect_ratio_delta_max,
    ground_contact_shift_px / ground_contact_max_px
)
```
clamped to `[0, 1]`. Each component that exceeds its tolerance also emits a specific reason code
(`GEOMETRY_CENTROID_SHIFT`, `GEOMETRY_SCALE_CHANGE`, `GEOMETRY_ASPECT_CHANGE`,
`GEOMETRY_GROUND_CONTACT`).

### Overall score

```
overall = Σ weight[m] * normalise(m) / Σ weight[m]
```
over the metrics that were evaluated, where `normalise` maps `at_most` metrics to `1 - value` so
that the overall score keeps the meaning "higher is better". Weights default to
`semantic_match 0.24, target_presence 0.16, background_preservation 0.22, mask_boundary 0.10,
geometry 0.12, artifact_free 0.10, annotation_consistency 0.06` and are renormalised over the
metrics actually present. `temporal_consistency` participates only for video samples.

## Gates and the acceptance policy

`AcceptancePolicy` is built from the recipe's `quality` block:

* `minimum_overall` — soft gate; the candidate must reach it or it is rejected with
  `BELOW_MINIMUM_OVERALL`.
* `hard_gates` — map of metric → threshold. Failing one rejects *regardless* of the overall score
  and the failing metrics are listed in `AcceptanceDecision.failed_hard_gates`.
* `warn_gates` — recorded as warnings, contribute to `NEEDS_REVIEW` only when the overall score is
  also inside `review_band`.
* `maximum_artifact_score` — inverted hard gate for `artifact_free`.

Order of evaluation: hard gates first, then `minimum_overall`, then the review band. The first
decisive rule wins and its reason codes are what the manifest records.

### Review band

When a hard gate is missed by at most `review_band` (default `0.03`) *and* the overall score is at
or above `minimum_overall`, the decision is `NEEDS_REVIEW` rather than `REJECTED`. Review items
are not exported unless `acceptance.export_review` is set, and they are always listed in the static
HTML review report.

## Reason codes

Closed enum in `vidliner.domain.reasons`. Each code carries a `ReasonCategory`
(`quality`, `annotation`, `data`, `execution`) and is emitted by exactly one rule.

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

## What the quality stage is *not*

* It does not decide whether the job succeeded. A job in which every candidate is rejected can
  still be `SUCCEEDED`; the counters say `accepted=0, rejected=N`.
* It does not silently drop evidence. Rejected candidates keep their artifacts, their metric rows,
  and their HTML review entry, so a threshold can be retuned and the candidates re-evaluated with
  `vidliner qa <job>` without regenerating anything.
* It does not trust the generator's own metadata. Backend `raw_metadata` is recorded for audit but
  is never an input to a gate.
