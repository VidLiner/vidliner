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

## Local rendering

Install `ffmpeg` and `ffprobe` on PATH, then render real MP4 files from a source video:

```sh
vidliner video probe source.mp4 --json
vidliner video generate examples/video/demo-spec.json --input source.mp4 --output rendered
vidliner video render examples/video/demo-spec.json --input source.mp4 --variant 1 --output square.mp4
vidliner video preview rendered/manifest.json --output rendered/review.html
```

`generate` means deterministic local variant generation, not an AI video model. Supported operations
are `format.reframe` (letterbox, never crop), `appearance.grade`, `temporal.speed`, `temporal.trim`,
and `audio.mute`. Generative operations fail closed and require a model backend. Only one reframe
per variant is supported. Existing outputs are protected unless `--force` is explicit, and the
source cannot be overwritten.

`manifest.json` records source/output digests, sampled operations, measured video properties, and
the normalized viewport and source-to-output clock mapping as `label_transform`. Labels still
need projection, clipping, and verification before dataset export. Muting audio is marked as an
edited semantic result, not a claim that speech labels remain valid. Rendering does not authorize
production export and is independent of the image DAG's acceptance policy.

Open `review.html` to compare source and variants with native video controls and inspect each
variant's evidence. Media stays in its original location; the page uses relative file paths, so
keep the source, rendered outputs, and HTML at the same relative locations. If serving over HTTP,
serve their common parent directory. Playback clocks are independent because variants may run at
different speeds. Time mappings are continuous mappings; frame sampling adds up to one frame of
quantization and still requires temporal verification.
