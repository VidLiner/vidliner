# Getting started

[简体中文](./zh/getting-started.md)

This walkthrough takes about five minutes and needs no model download, no API key, and no network
access. It ends with an exported dataset that has annotations, provenance, and a quality report.

---

## 1. Install

```bash
pip install -e ".[dev]"
vidliner --help
```

---

## 2. Create a workspace

A workspace is a directory VidLiner owns: it holds the artifact store, the job state, and the run
directories. Your source images can live inside it (recommended) or anywhere the profile allows.

```bash
mkdir -p playground
vidliner init ./playground
```

You now have:

```
playground/
  runtime.yaml      which backend serves which capability
  state.db          job, node, candidate, and cache state
  artifacts/        content-addressed outputs
  runs/             one directory per job
  datasets/         default export location
```

The starter profile binds every capability to a built-in local backend, so the pipeline runs
immediately. Those backends are **demonstration stand-ins**, and the pipeline will say so before it
lets you export a dataset built on them — step 5 shows what that looks like.
`vidliner backend check --workspace ./playground` shows what is bound.

---

## 3. Provide data

Put images in `playground/data/source/`. If you have labels, keep them in the layout your format
expects:

```
playground/data/source/
  train/            # split directories (or use filename prefixes, or a manifest)
    img_001.png
    img_002.png
  annotations/
    instances_train.json      # COCO
```

You can also start with no annotations at all: ingestion and detection do not need them. The
annotation-carrying splits matter only for *inheriting* labels of untouched objects.

Inspect before you commit to anything:

```bash
vidliner inspect ./playground/data/source --format coco-instance
```

```
dataset   playground/data/source
samples   42
splits    train=34, val=8
augment   train
images    42   dimensions 640x480, 1280x720
annotated 42/42 samples, 96 objects
classes   car=96
```

---

## 4. Write a recipe

```bash
vidliner recipe init ./playground/car-swap.yaml
```

Open it and adjust at least `dataset.input`, `target.classes`, and `replacement.values`. Validate:

```bash
vidliner recipe validate ./playground/car-swap.yaml
```

The validator is strict on purpose: an unknown metric name, a `prompt_template` with no placeholder,
or a request to augment `test` are all errors, not warnings, because each would produce a dataset
that looks fine and is not.

---

## 5. Plan before you spend

```bash
vidliner plan ./playground/car-swap.yaml --workspace ./playground
```

```
job        j_1f0c9a2b7d4e5f60
graph      462 nodes, hash 9c1d2e3f4a5b
stages     ingest=42, detect=42, select=42, segment=42, scene=42, plan=42,
           generate=126, refine=126, evaluate=126, annotate=126, export=126
samples    42  candidates 126
external   0 call(s)  accelerator ops 0
estimate   6.30s, cost n/a
  vision.object_detection.v1                   -> builtin_detector
  planning.replacement.v1                      -> builtin_planner
  generation.object_replacement.v1             -> builtin_replacement
  ...
```

Planning resolves bindings and counts nodes. It calls **no generative backend**, which is what makes
this safe to run before you have decided anything.

It also tells you what the stack is:

```
DEMONSTRATION stack: the export would be refused
  vision.object_detection.v1                   -> builtin_detector (demonstration stand-in)
  vision.instance_segmentation.v1              -> builtin_segmenter (demonstration stand-in)
  generation.object_replacement.v1             -> builtin_replacement (demonstration stand-in)
  quality.semantic_match.v1                    -> builtin_metrics (demonstration stand-in)
  bind production backends for these capabilities, or set acceptance.allow_demo_backends: true to say you know
```

Those four capabilities produce the data that ends up in the dataset, and every one of them is served
by a stand-in: a saliency heuristic, a synthetic generator, and a colour heuristic. The job will run
— the pipeline is exercisable end to end on any checkout — but the result is shaped like training
data and is not training data, so the export is refused unless you say otherwise. A recipe says so
with `acceptance.allow_demo_backends: true`, which is what `examples/car-swap/car-swap.yaml` does and
what the rest of this walkthrough assumes. See
[Writing a backend](./backends.md#demonstration-backends-and-the-production-guard).

`plan` also writes `runs/<job>/plan.json`: the exact graph, every node's lineage and config, and the
estimate. `vidliner run --dry-run` produces the same document and then stops.

---

## 6. Run

```bash
vidliner run ./playground/car-swap.yaml --workspace ./playground
```

```
job        j_1f0c9a2b7d4e5f60  [succeeded]
candidates 126  accepted 118  rejected 8  review 0  acceptance 94%
nodes      462 ok, 0 failed, 0 cached, 0 resumed
dataset    playground/data/output (coco-instance)
  written  118 image(s), 118 accepted sample(s), 0 duplicate(s) excluded
evidence   playground/runs/j_1f0c9a2b7d4e5f60
```

Every rejected candidate kept its diagnostics. Look at why:

```bash
vidliner qa latest --workspace ./playground --show 5
```

```
evaluated  126 candidate(s); 0 changed decision
counts     accepted=118, rejected=8
  c_04a1...  rejected  0.612  suv          BACKGROUND_CHANGED
  c_1b77...  rejected  0.588  pickup       GEOMETRY_SCALE_CHANGE,GEOMETRY_VIOLATION
  ...
```

---

## 7. Review what needs a human

Candidates inside the review band are marked `NEEDS_REVIEW` rather than silently accepted or
rejected. They appear in a static HTML file that opens from disk:

```bash
vidliner report review latest --workspace ./playground
open playground/runs/j_1f0c9a2b7d4e5f60/review/index.html
```

Each card shows the source frame, the target mask, the generated candidate, an amplified difference
map, the rebuilt annotation, every metric with its gate, and the reason codes. Decisions are recorded
in `decisions.json` next to the report, which is the interface a web UI would implement.

---

## 8. Re-evaluate without regenerating

Quality thresholds are data. If you decide `background_preservation` should be stricter, you do not
need to regenerate anything:

```bash
# edit quality.hard_gates.background_preservation in the recipe, then:
vidliner qa latest --workspace ./playground --recipe ./playground/car-swap.yaml
vidliner export latest --workspace ./playground
```

`qa` re-applies the policy to the stored metric evidence. No pixels are recomputed and no backend is
called.

---

## 9. What you get

```
playground/data/output/
  images/                  118 PNGs — accepted samples only
  annotations/
    instances_train.json   rebuilt labels, one category vocabulary for the set
  provenance/
    c_04a1....json         source digest, recipe hash, seeds, backends, scores, decision
  manifest.json            dataset manifest: counts, categories, seed tree, exclusions
  dataset-report.json      class, size, and aspect distributions; acceptance by category
```

Check the provenance record:

```bash
python -m json.tool playground/data/output/provenance/$(ls playground/data/output/provenance | head -1)
```

It answers, without any other file: what source sample this came from, which recipe and seed produced
it, which operators and backends ran, what the model was asked for, what every metric scored, what
was decided and why, and when.

---

## 10. Next steps

* **Use your own models.** Bind real backends in `runtime.yaml` — see `docs/runtime.md` and
  `docs/backends.md`. No recipe or pipeline change is involved.
* **Understand the gates.** `docs/quality.md` explains every metric and how to tune the policy
  without weakening it.
* **Protect your splits.** `docs/datasets.md` covers lineage, leakage, and duplicate control.
* **Interrupt and resume.** Cancel a long run and bring it back with
  `vidliner job resume <job-id>`; finished nodes are reused, not recomputed.
* **Run the tests.** `pytest -q` runs the whole suite against synthetic data in well under a minute
  per shard, with no network access.
