# VidLiner

<p align="center">
  <img src="./brand/assets/vidliner-logo-dark.svg" alt="VidLiner — Synthetic data you can prove" width="420" />
</p>

<p align="center">
  <strong>Generate. Verify. Rebuild labels. Export evidence.</strong><br />
  <sub>Early alpha · Image pipeline MVP · Video variant planning is available</sub>
</p>

**Object-centric synthetic data augmentation with mandatory verification.**

[简体中文](./README.zh-CN.md)

VidLiner turns images you already have into *verified* training samples. It finds a target object,
replaces it, **re-detects and re-segments the object in the generated image**, proves the rest of the
scene is unchanged, rebuilds the annotation from the pixels that are actually there, evaluates the
result against an explicit acceptance policy, and exports only what passed.

The product is not a prettier picture. The product is:

```
training image  +  correct label  +  provenance  +  quality evidence
```

If a candidate cannot become safe training data, it is rejected, and the reason is recorded in a
machine-readable form.

> **Project status:** VidLiner is early-alpha software. The built-in profile is deterministic and
> dependency-free for demos and testing; bind production-grade backends before using output in a
> real training dataset. See the [brand kit](./brand/README.md) and [launch copy](./brand/launch/README.md).

---

## What it solves

Three failure modes make synthetic augmentation harmful rather than helpful, and VidLiner addresses
each one by construction:

| Failure | How VidLiner prevents it |
| --- | --- |
| **Wrong labels.** The size, shape, or class of the object changed, but the old box was reused. | The object is re-detected and re-segmented **in the generated image**, and the annotation is rebuilt from *that* mask. Labels are never copied across a replacement, and the mask handed to the generator is never trusted as a description of the result. |
| **Background drift.** The generator quietly repainted the scene, so the model learns the wrong invariance. | Outside the (feathered) target mask, SSIM, changed-pixel ratio, and perceptual delta are measured, and a hard ceiling rejects any candidate with large-area background change. |
| **Dataset leakage and duplicate flooding.** Validation ends up containing a near-copy of a training image, and the set fills with near-identical variants. | Every sample carries lineage, augmentation is restricted to an allow-list of splits, and perceptual-hash duplicate control runs at export. |

---

## Installation

VidLiner needs Python 3.11 or newer. It runs the full pipeline with **no downloaded model and no
paid API** out of the box:

```bash
pip install -e ".[dev]"
vidliner --help
```

Optional extras:

* `pip install -e ".[cv]"` installs OpenCV for backends that want faster image operations;
* `pip install -e ".[http]"` installs `httpx`, which the generic HTTP replacement backend needs.

---

## Quickstart

```bash
# 1. Create a workspace: directories, a state database, and a starter runtime profile.
vidliner init ./workspace

# 2. Look at your data before touching it.
vidliner inspect ./workspace/data/source --format coco-instance

# 3. Write a recipe (or start from the documented template).
vidliner recipe init ./workspace/car-swap.yaml

# 4. Validate it and see what it would cost, without running anything.
vidliner recipe validate ./workspace/car-swap.yaml
vidliner plan ./workspace/car-swap.yaml --workspace ./workspace

# 5. Run it. Accepted samples are exported; rejected ones keep their diagnostics.
vidliner run ./workspace/car-swap.yaml --workspace ./workspace

# 6. Review, re-score, and export.
vidliner jobs --workspace ./workspace
vidliner qa latest --workspace ./workspace --show 5
vidliner report review latest --workspace ./workspace
vidliner export latest --workspace ./workspace --format coco-instance
```

The result:

```
workspace/data/output/
  images/             only accepted samples
  annotations/        rebuilt labels in the requested format
  provenance/         one JSON record per accepted sample
  manifest.json       dataset manifest with seeds, backends, and scores
  dataset-report.json class, size, and aspect distributions plus acceptance rates
```

---

## Architecture in one page

```
Recipe (YAML)                     what the user wants
     │
     ├── compiled by ─────────────────────────────────────────────┐
     ▼                                                            │
Operation Graph (DAG)             ingest → detect → select → segment → scene → plan
     │                                   → generate → refine → verify → evaluate → annotate → export
     │
     ▼  executed by
Engine (asyncio, bounded, cached, resumable)
     │  dispatches each node through
     ▼
Operator  ──needs──▶  Capability  ──bound by──▶  Backend
(domain work)         (abstract ability)         (the only place a vendor lives)
     │
     ▼
Artifacts (content-addressed) + State (SQLite) + Reports (files)
```

Five ideas carry the design:

1. **A recipe declares intent; a compiler builds a typed graph.** That is what makes `plan` and
   `--dry-run` free, and what allows partial re-execution.
2. **Operators declare capabilities, never vendors.** The runtime profile decides which backend
   serves `vision.object_detection.v1`, so the same recipe runs against local models, a local
   inference server, or a hosted service without edits.
3. **Everything is content-addressed and cacheable.** A node's cache key is
   `(operator, version, implementation, config, input digests, seed)`, so a re-run reuses exactly the
   work that is still valid and resumes exactly the work that is not.
4. **Quality is a set of gates, not a score.** A hard gate rejects regardless of how good the
   overall score looks, because one corrupt label is worse than one lost sample.
5. **System failure and sample rejection are different things.** A job in which every candidate was
   rejected still *succeeded*; the counters say so.

Read `ARCHITECTURE.md` for the full picture and `docs/design/decisions.md` for why each decision was
made.

---

## Example recipe

```yaml
recipe: car-swap

dataset:
  input: ./data/source
  format: coco-instance
  splits: { mode: directory, names: [train, val, test] }
  augment_splits: [train]          # augmenting val/test is refused unless explicitly allowed

target:
  classes: [car]
  min_score: 0.35
  min_area_px: 1024
  max_per_sample: 2

replacement:
  mode: strict                     # strict | creative (creative never enters accepted data by default)
  strategy: category
  values: [sedan, suv, pickup]
  candidates_per_object: 3
  seed: 20240517

preserve:
  background: strict
  geometry: true
  lighting: true

quality:
  minimum_overall: 0.82
  hard_gates:
    semantic_match: 0.90
    background_preservation: 0.93
    annotation_consistency: 0.95
  maximum_artifact_score: 0.15
  minimum_target_presence: 0.60
  review_band: 0.03

export:
  format: coco-instance
  path: ./data/output
```

`vidliner recipe schema` prints the full JSON Schema; `docs/recipe.md` documents every field.

---

## CLI

| Command | Purpose |
| --- | --- |
| `vidliner init` | Create a workspace, a state database, and a starter runtime profile. |
| `vidliner inspect <dir>` | Report media, dimensions, splits, and annotation coverage. |
| `vidliner recipe validate\|schema\|init\|show` | Work with recipes. |
| `vidliner plan <recipe>` | Compile and cost the job. Calls no generative backend. |
| `vidliner run <recipe>` | Execute, evaluate, and export. `--dry-run` stops before generation. |
| `vidliner jobs` | List jobs with their counters and acceptance rates. |
| `vidliner job show\|resume\|cancel` | Inspect, resume, or cancel a job. |
| `vidliner qa <job>` | Re-apply the acceptance policy to stored evidence. No pixels are recomputed. |
| `vidliner export <job>` | Write accepted samples as a dataset. |
| `vidliner backend list\|check` | Inspect capabilities and probe backends (never with a generative call). |
| `vidliner report review\|summary\|events` | Regenerate reports and read the structured event log. |

Exit codes: `0` success, `1` the answer was "no" (unmet capability, nothing accepted), `2` the
command could not run (bad recipe, invalid path).

---

## Supported formats

**Input** — `jpg`, `jpeg`, `png`, `webp`; `mp4` is recognised and refused with an explicit phase-2
error. Annotations: COCO detection/instance segmentation, YOLO detection/segmentation, or none.

**Output** — COCO detection, COCO instance segmentation, YOLO detection, YOLO segmentation.

Adding a format means adding a converter in `vidliner/annotations/`; the pipeline is untouched
because everything flows through one internal annotation IR.

---

## The quality gate

`vidliner run` never asks "did the generator return an image". It asks whether the result can become
training data:

| Metric | Direction | What it catches |
| --- | --- | --- |
| `semantic_match` | higher is better | the replacement is not the requested thing |
| `target_presence` | higher is better | the object is not found in the generated image |
| `background_preservation` | higher is better | the scene changed outside the mask |
| `mask_boundary` | higher is better | a visible seam, a halo, or a cut-through |
| `geometry` | higher is better | the object moved, resized, or stopped touching the ground |
| `artifact_free` | higher is better | duplication, floating objects, text, deformation |
| `annotation_consistency` | higher is better | the rebuilt label is empty, out of bounds, or mismatched |

A hard gate failure rejects regardless of the overall score. A near miss inside the review band
becomes `NEEDS_REVIEW`, which the static HTML report presents for a human decision. Full detail is
in `docs/quality.md`.

---

## Demonstration backends are refused at export

The profile you get on a fresh checkout binds saliency heuristics, a synthetic generator, and a colour
evaluator, so the whole pipeline runs with no model download and no API key. Every one of those is a
*stand-in*, and a dataset built on them is shaped like training data without being training data.

So the job runs, and the export is refused:

```bash
vidliner plan examples/car-swap/car-swap.yaml
# DEMONSTRATION stack: the export would be refused
#   vision.object_detection.v1       -> builtin_detector (demonstration stand-in)
#   vision.instance_segmentation.v1  -> builtin_segmenter (demonstration stand-in)
#   generation.object_replacement.v1 -> builtin_replacement (demonstration stand-in)
#   quality.semantic_match.v1        -> builtin_metrics (demonstration stand-in)
```

Those four capabilities produce the data that ends up in the dataset; a stand-in for a planner, a
refiner, or an artifact evaluator changes only how a sample was *made*, and does not block anything.
Bind real backends for the four and the refusal disappears. If the dataset is deliberately for
inspection, say so in the recipe with `acceptance.allow_demo_backends: true` — it is recorded in the
manifest next to the bindings that were used, and no profile edited afterwards can change that
verdict. See `docs/backends.md`.

---

## Dataset provenance

Every accepted sample carries a provenance record containing its source sample and digest, the
recipe hash, the job id, the target object, the replacement intent, the seed tree, operator and
backend identifiers, the generation parameters, every quality score, the decision with its reason
codes, the output digest, and the timestamp.

Credentials are resolved by *reference* (`env:`, `file:`) and never appear in a recipe, a manifest,
a log line, or a provenance record. A test asserts this end to end.

---

## Reproducibility

A job has one root seed. Every node derives its own seed along the identity path
`job_seed → sample_seed → target_seed → candidate_seed`, so adding a candidate does not shift the
seeds of the existing ones. Node identity is derived the same way, which is what makes a re-run reuse
its cache and an interrupted run resume.

---

## Documentation

| Document | Contents |
| --- | --- |
| `ARCHITECTURE.md` | The full design: layers, data flow, extension points. |
| `CONTRIBUTING.md` | How to work on the code, and the rules that keep it honest. |
| `docs/getting-started.md` | A guided first run. |
| `docs/recipe.md` | Every recipe field, with validation rules. |
| `docs/runtime.md` | Runtime profiles, bindings, credentials, and pre-flight. |
| `docs/backends.md` | Writing a backend; the capability contract. |
| `docs/quality.md` | Metrics, gates, decisions, and reason codes. |
| `docs/datasets.md` | Splits, lineage, duplicates, and the dataset report. |
| `docs/testing.md` | Test layout and the synthetic fixtures. |
| `docs/design/` | Architecture Decision Record, domain model, DAG design, schemas, milestones. |

---

## Status

The MVP implements the image pipeline end to end: ingest, detection, segmentation, scene analysis,
planning, candidate generation, refinement, **post-generation re-detection and re-segmentation**,
quality evaluation, annotation regeneration, acceptance, dataset export, job state, resume, cache,
CLI, and reports.

Deliberately **not** implemented yet, with the extension points already in place: video replacement
and tracking (`ObjectTrack`, `FrameRef`, and `temporal_consistency` exist in the data model),
distributed workers, object storage, and a web review UI. Anything unimplemented raises
`NotImplementedError` or an explicit capability error rather than pretending to succeed.

## License

MIT. See `LICENSE`.
