# VidLiner — Architecture Decision Record

Status: accepted for the MVP (image-only) milestone.
Scope: this document records *why* the system is shaped the way it is. It is written from
functional requirements only. No reference implementation was copied; only general software
architecture ideas (dependency graphs, ports and adapters, capability binding, content-addressed
storage) were used, which are common industry practice.

---

## Context

Training-data augmentation for object-centric computer vision is not an image-editing problem.
The deliverable is `image + correct annotation + provenance + quality evidence`. A generative
model that returns a plausible picture is *not* a finished data point: the label space must be
rebuilt for the object that changed, and the untouched parts of the scene must still be provably
untouched.

VidLiner therefore treats augmentation as a **verified data-production pipeline**, not as a
generation call.

The pipeline the system must implement:

```
Input Dataset → Sample Analysis → Target Object Selection → Segmentation/Tracking
→ Replacement Planning → Candidate Generation → Compositing/Refinement
→ Quality Evaluation → Annotation Regeneration → Acceptance Gate → Dataset Export
```

---

## ADR-001 — Clean-room functional reimplementation

**Decision.** VidLiner is an independent project. Its package name, module tree, class names,
configuration format, CLI, error taxonomy, and documentation are original designs derived from
the functional requirements. Two reference codebases that exist in the surrounding workspace were
studied only for *general* architecture ideas and are not represented in this repository in any
form: no source, file content, identifier, DSL, command vocabulary, comment, directory layout,
manifest shape, license text, brand asset, or git history.

**Rationale.** The requirement is a clean-room functional reimplementation. Mixing a derivative
of an existing implementation into the core would make provenance and licensing unclear, and
would import design constraints that do not fit computer-vision data production.

**Consequences.**
* Every identifier in this repository was chosen for this project (see `docs/design/naming.md`).
* Where a capability could only be obtained by copying, it was re-derived from the requirement
  and re-implemented; where no implementation exists yet, the code raises `NotImplementedError`
  instead of pretending to succeed.
* The repository is its own git repository with a single, fresh history and no remotes.

---

## ADR-002 — Declarative recipe on top of a typed DAG, not an imperative script

**Decision.** Users describe *intent* in a YAML recipe (what to replace, what to preserve, how
strict the QA gate is, how to export). A compiler turns the recipe plus dataset inspection into
an explicit `OperationGraph` of typed nodes. An engine executes that graph.

**Rationale.** A declarative recipe makes three requirements achievable at once:
1. **Planning without spending.** The graph can be built and costed without calling a paid model
   (`vidliner plan`, `--dry-run`).
2. **Partial re-execution.** Because nodes carry identity and input digests, QA can be re-run, a
   failed node can be retried, and an interrupted job can resume.
3. **Binding independence.** The recipe never names a vendor. Which implementation satisfies a
   capability is a runtime decision.

**Alternatives rejected.**
* *Imperative pipeline script (Python API as the primary interface).* Not inspectable, not
  resumable, no natural plan step, no cost estimation. A Python API still exists, but as a
  secondary surface over the same graph.
* *Workflow engine dependency (Airflow/Prefect/Dagster).* Heavyweight operationally, and their
  unit of work is a task, not an artifact-typed port. We need artifact typing and content-hash
  caching at the edge level.

---

## ADR-003 — Capability + Binding instead of a hard-coded model

**Decision.** An operator declares the *capabilities* it needs (for example
`vision.object_detection.v1`). A Runtime Profile decides which concrete backend serves each
capability. Operators never import or name a backend.

**Rationale.** The same recipe must run on a laptop with local models, on a box with a local HTTP
inference server, and against a hosted generative service, without edits. This is also what makes
the MVP testable end-to-end with fake backends and no paid API.

**Consequences.**
* Core (domain, core, pipeline, operators, quality) has **zero vendor imports**. Enforced by an
  architecture test.
* Adding a model means adding a backend package, not touching the pipeline.

---

## ADR-004 — Backends are the only place vendor SDKs may live

**Decision.** Every backend lives under `vidliner/backends/` (built-in) or an importable plugin
module named in the runtime profile. Vendor SDK imports are confined there. A test asserts that
no module outside `backends/` imports a vendor SDK.

**Rationale.** Keeps the core dependency-inverted and independently testable, and makes the
"which suppliers do we depend on" question answerable by reading one directory.

---

## ADR-005 — Content-addressed artifacts and cache by (operator, version, inputs, config)

**Decision.** Every artifact is written under `artifacts/<sha256[:2]>/<sha256>.<ext>` and referred
to by digest. A node's cache key is
`H(operator name, operator version, sorted input digests, canonical config, implementation
identity, seed policy)`. A cache hit reuses the recorded artifact *only if the file still exists
and still hashes to the recorded digest*.

**Rationale.** Reproducibility and resumability both reduce to "can I prove these inputs produce
this output". Digest identity also makes provenance a hyperlink rather than a log line. Verifying
the bytes on read prevents a corrupt cache from silently poisoning a training set.

**Consequences.** Re-running an identical recipe is nearly free; re-running with one changed
threshold reuses everything upstream of the change.

---

## ADR-006 — Quality is a first-class stage with hard gates, not a score

**Decision.** Acceptance is decided by a policy of independent metric gates. A metric can be a
*warn* or a *hard* gate. A hard gate failure rejects the candidate regardless of the overall
score. The QualityReport carries one entry per metric with the value, threshold, and the machine
readable reason codes it contributed.

**Rationale.** A single weighted score lets a very good segmentation hide a broken background.
For training data, the failure that matters is the one that corrupts a label or teaches the model
the wrong background invariance.

**Consequences.**
* Rejection is a normal, expected outcome and is recorded as `QualityReject` — a *sample* outcome,
  never a *system* error (ADR-010).
* The gate set is data, so `vidliner qa <job>` can re-evaluate existing candidates under a new
  policy without regenerating anything.

---

## ADR-007 — Annotation is regenerated, never inherited blindly

**Decision.** After replacement, the target object's annotation is recomputed by re-running
detection/segmentation on the generated image. Surviving annotations of untouched objects are
carried over but pass a preservation validation against the background-difference evidence. All
annotation data flows through one internal IR; COCO and YOLO are importers/exporters of that IR.

**Rationale.** Requirement priority #1 is annotation correctness. A bounding box carried over from
the source image is *wrong by construction* when the object changed shape or category. Making the
IR the hub keeps format support additive and makes roundtrip testing possible.

---

## ADR-008 — Lineage-aware splits and duplicate control are part of the pipeline, not an add-on

**Decision.** Every sample carries a `split` and a `lineage` (root source digest + ancestors).
Augmentation is only permitted for splits in the recipe's allow-list (default: `train`). The
exporter validates that no augmentation descendant and its source land in different splits, and
computes perceptual hashes to reject near-duplicates against both the source set and already
accepted outputs.

**Rationale.** Dataset leakage and near-duplicate flooding are the two failure modes that make an
augmentation pipeline *harmful*. They cannot be detected after the fact if lineage is not
recorded, so the data model must carry it from ingest.

---

## ADR-009 — Seed tree instead of ambient randomness

**Decision.** A job has a root seed. Every node derives its own seed from
`job_seed → sample_seed → target_seed → candidate_seed` using a stable hash of the identity path.
Backends receive an explicit seed. No module calls module-level `random` or `numpy.random` global
state.

**Rationale.** Reproducibility is requirement priority #3, and ambient randomness makes a
generation result unrecoverable from the manifest. The derived-seed tree also means adding a
candidate does not shift the seeds of existing candidates.

---

## ADR-010 — System failure and sample rejection are different things

**Decision.** The exception taxonomy separates `OperatorFailure`, `BackendFailure`,
`ValidationFailure`, `QualityReject`, `InfrastructureFailure`, and `Cancellation`. A `QualityReject`
never fails a job; it produces a `REJECTED` candidate and diagnostics. Job summaries report
processed/accepted/rejected/review/failed as distinct counters.

**Rationale.** Otherwise a pipeline that produces 5% usable data looks "successful" while a
gateway timeout on one sample looks like "bad data".

---

## ADR-011 — Asynchronous execution, no threads in core logic

**Decision.** The engine is `asyncio`-based. Backends are async-first; a backend that wraps
blocking CPU work declares that it is blocking and the engine dispatches it through
`asyncio.to_thread`. Concurrency is bounded per backend, not globally per process.

**Rationale.** External generative backends are latency-bound and benefit from concurrency;
local GPU work must be serialized. One async engine with declared capacity handles both without a
worker/cluster design that the MVP does not need (ADR-014).

---

## ADR-012 — SQLite for state, filesystem for bulk data

**Decision.** Job, node, candidate, and cache state live in a single `state.db` (SQLite, WAL).
Images, masks, and reports live on the filesystem, content-addressed. Reports that are meant for
humans (HTML review, dataset report) are written as files and linked from the database.

**Rationale.** State must survive a restart and be queryable (`vidliner jobs`, resume). Bulk
pixels do not belong in a database. SQLite is enough for single-host orchestration.

---

## ADR-013 — Credentials never touch recipe, manifest, or git

**Decision.** Backends receive credentials by *reference* (`env:NAME`, `file:/path`, or a literal
`value:` that is rejected inside the repository boundary for recipe files). Manifests record which
reference was used, never the resolved value. A redaction filter runs over all structured logs and
over every serialized manifest.

**Rationale.** Provenance files are shared with datasets; leaking a key there is unrecoverable.
A shared dataset repository that contains secrets cannot be distributed.

---

## ADR-014 — MVP is single-host, image-only, fake-backend-testable; the model leaves room for video

**Decision.** The MVP does not implement distributed workers, queues, cloud storage, a web UI, or
video replacement. However the domain model already contains `ObjectTrack`, `FrameRef`, temporal
quality metrics, and an `AcceptancePolicy` that can express temporal gates, and the engine already
supports stage fan-out by `(sample, target, candidate)`.

**Rationale.** Requirement: "the API and data model need to leave room for the video extension,
but do not over-engineer the MVP". Concretely this means the *interfaces* are temporal-ready while
the MVP *operators* are image-only and raise `NotImplementedError` for video inputs.

---

## ADR-015 — Explicit failure over fabricated success

**Decision.** Unimplemented behaviour raises `NotImplementedError`. There are no placeholder
returns, no `pass`-bodied operators, and no catch-and-continue blocks. A backend that cannot do
something says so through `BackendProbe`.

**Rationale.** A pipeline whose components can silently no-op produces a dataset that looks
complete and is not. Since the product promise is "verified training data", a silent no-op is the
most dangerous possible bug.

---

## ADR-016 — Deterministic identity for every node

**Decision.** A node ID is derived from `(operator name, lineage key, ordinal)` where the lineage
key is a stable digest of the sampled/object/candidate identity. Operations fan out by explicit
keys, not by list position, so inserting one candidate does not renumber the others.

**Rationale.** Cache hits, resume, and provenance comparison across runs all depend on node
identity being stable under unrelated edits.

---

## ADR-018 — The pipeline is closed by re-detecting the object in the generated image

**Decision.** A `verify.redetect` stage runs between refinement and evaluation. It detects the object
in the *generated* image, matches the detection to the region that was replaced, re-segments it, and
derives the bbox, polygon, and area from that mask. Evaluation, annotation, and export all describe
the regenerated object; the mask that was handed to the generator is used for exactly one purpose —
defining the region where change was authorised, which is what `background_preservation` excludes.

**Rationale.** Without this stage several checks are tautological. `target_presence` and `geometry`
compared the input mask with itself, so a generator that ignored the mask, moved the object, or
returned the frame untouched scored perfectly. `annotation_consistency` measured a mask that was a
statement about the *request* rather than about the result, and the exported label inherited that
mask. On the built-in synthetic generator this is invisible; on a real backend it is the difference
between a verified dataset and a plausible-looking one.

**Consequences.**

* The exported label is derived from pixels that exist. An object that is not in the image cannot
  receive a label.
* Not finding the object is an **outcome**: the operator reports `found=False` with an empty mask, the
  quality stage records `OBJECT_NOT_FOUND`, and the gate rejects the candidate. No node fails, and a
  job whose samples are all unusable still reports `SUCCEEDED` with `accepted=0`.
* Re-detection looks for the class the object **had**, not the replacement category. A detector's
  label set is fixed and does not contain sub-categories such as `sedan`; asking for one finds
  nothing even when the replacement succeeded. Whether the replacement belongs to the requested
  category is the semantic evaluator's judgement, because that is a question about appearance.
* The stage needs real detection and segmentation backends. The built-in heuristic pair is enough for
  the demo and for tests, and is not enough for production data — which is what makes the demo-only
  restriction on the default profile necessary rather than cosmetic.
* `annotate` tolerates a missing object by emitting a structurally valid **empty** bundle tagged
  `not_found`, so a rejection does not surface as a node failure in the job's failure counters. A
  job is not broken because a generated sample is unusable; that is exactly the distinction ADR-010
  draws.
