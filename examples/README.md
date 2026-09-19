# Examples

[简体中文](./README.zh-CN.md)

## `car-swap` — the whole pipeline, no downloads, no paid API

```bash
python examples/car-swap/demo.py [output-directory]
```

The demo:

1. creates a workspace with the default local profile;
2. renders ten synthetic annotated road scenes;
3. runs `plan` (free), `run` (generation, verification, re-annotation, gating, export), `qa`
   (re-score without regenerating), and reads the dataset report;
4. prints one provenance record.

Everything it prints comes from the library API — `Session.plan`, `Session.run`, `Session.re_evaluate`
— which is exactly what the CLI calls. If you would rather use the CLI:

```bash
WS=/tmp/vidliner-cli
vidliner init $WS
python - <<'PY'
import sys; sys.path.insert(0, "examples/car-swap")
from demo import _write_coco, _write_scene_dataset
from pathlib import Path
root = Path("/tmp/vidliner-cli/data/source")
_write_scene_dataset(root / "train", count=10)
_write_coco(root, root / "train")
PY
vidliner inspect $WS/data/source --format coco-instance
vidliner plan    examples/car-swap/car-swap.yaml --workspace $WS
vidliner run     examples/car-swap/car-swap.yaml --workspace $WS
vidliner qa      latest --workspace $WS --show 5
vidliner export  latest --workspace $WS --format yolo-segmentation --path ./data/yolo
vidliner report  review latest --workspace $WS
```

### What to look at

| Output | Why it matters |
| --- | --- |
| `plan` node count and bindings | the graph is built and costed before anything generative runs |
| `run` counters | accepted, rejected, and review are counted separately from failures |
| `qa` output | thresholds can be re-applied to stored evidence without regenerating |
| `dataset-report.json` | the replacement distribution is even across `sedan`, `suv`, and `pickup` |
| `manifest.json` | counts, categories, seed tree, and per-sample scores |
| `provenance/*.json` | source digest, recipe hash, seeds, backends, scores, decision |
| `review/index.html` | the human-review surface, for candidates inside the review band |

### What the verification stage does to the example

The example now runs the closed loop, so its acceptance rate is a real measurement rather than a
formality. `verify.redetect` finds the object in each generated image and re-segments it; a candidate
whose replacement landed smaller than the region it replaced, or whose synthetic appearance does not
match the requested category, is rejected with `GEOMETRY_SCALE_CHANGE` or `SEMANTIC_MISMATCH` instead
of passing a comparison with the mask it was handed.

The built-in generator paints a flat, seed-derived colour, so a few candidates legitimately fall
outside the detector's saliency range and are rejected with `OBJECT_NOT_FOUND`. That is the pipeline
working as intended, not a bug: the acceptance rate you see is what the built-in demo stack can
actually produce.

### Why some candidates are rejected

The built-in generator produces deterministic synthetic imagery rather than photoreal edits, and the
built-in evaluators are heuristics. So the interesting behaviour is not that it accepts — it is that
it *rejects* candidates whose measured semantics, background preservation, or geometry do not hold.
Tighten `quality.hard_gates` in `car-swap.yaml` and re-run `vidliner qa`: the same candidates are
re-decided from stored evidence, with the reason codes explaining each one.

This is the property the whole project is built around: the pipeline fails its own gate rather than
producing a dataset that looks complete and is not.

### Using real models

Nothing in the recipe changes. Point the four data-defining capabilities at real backends in the
workspace's `runtime.yaml`:

```yaml
backends:
  my_detector:  { use: my_project.detector:DetectorBackend, options: { model_path: /models/yolo.onnx } }
  my_segmenter: { use: my_project.segmenter:SegmenterBackend, options: { model_path: /models/sam.onnx } }
  my_editor:    { use: my_project.editor:EditorBackend, credentials: { api_key: { source: env, name: EDIT_KEY } } }
  my_judge:     { use: my_project.judge:SemanticJudgeBackend }

bindings:
  vision.object_detection.v1: my_detector
  vision.instance_segmentation.v1: my_segmenter
  generation.object_replacement.v1: my_editor
  quality.semantic_match.v1: my_judge
```

Then remove `acceptance.allow_demo_backends` from the recipe: with these bindings the export is no
longer refused, and the acknowledgement would be a lie about the data.

The example recipe sets `acceptance.allow_demo_backends: true` because it runs on the built-in
demonstration stack, and the pipeline refuses to export a dataset produced that way unless the recipe
says it knows. `vidliner plan` prints the same warning before anything is generated. See
`docs/backends.md` for the contract and `docs/runtime.md` for the profile.
