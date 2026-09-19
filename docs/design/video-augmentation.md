# Video augmentation contract

VidLiner treats video generation as two separate responsibilities:

1. an operator or external generator produces a variant;
2. the pipeline proves how the labels and the time axis moved, then verifies the result before export.

The first responsibility is represented by `VideoAugmentSpec` and `VideoVariant` in
`vidliner.domain.video`. The second is represented by `LabelTransform`, `ObjectTrack`, temporal
quality gates, and the existing redetection/resegmentation stages.

## The invariant

Every operation that changes pixels or the clock declares a transform. Transforms compose in
source-to-output order:

```python
from vidliner.domain.video import LabelTransform, TimeTransform, project_box

variant_transform = LabelTransform(time=TimeTransform(scale=0.5, offset=0.0))
new_box = project_box(variant_transform, source_box)
```

The transform is deliberately renderer-neutral. A local ffmpeg operator, a Hypit authoring
fragment, and a future video model must all emit the same evidence shape. This keeps generation
and verification interchangeable without silently reusing stale labels.

The current DAG entry point is `video.plan_variants`. It is backend-free and cacheable: the same
spec and seed produce the same variant IDs and pair relations, whether planning is invoked from
the CLI or from a resumed job.

## Sampling and contrast pairs

`VideoAugmentSpec.strategy` supports three deterministic modes:

* `full-factorial` enumerates the declared parameter grid;
* `paired` emits an identity reference and one-axis siblings;
* `random` derives every choice from the recipe seed and the operator axis.

Variant IDs are content-derived, not list-position-derived. Adding another domain value therefore
does not renumber existing variants or invalidate their cache lineage.

The `fingerprint` on each variant identifies the changed axes. Pair construction can use it to
create typed relationships such as `format-invariance`, `style-invariance`, and
`capture-invariance`, while the production gate still decides whether a generated video is safe
to export.

## Production boundary

Local transforms may be deterministic and offline, but they are not a substitute for verifying a
generated result. Hypit-backed operations must return new semantic anchors when they edit identity
or text. A planned or imported variant with stale anchors is not a preserving sample. It must pass
redetection, resegmentation, and temporal gates before entering a production dataset.
