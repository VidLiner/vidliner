# Architecture

[简体中文](./ARCHITECTURE.zh-CN.md)

This document explains how VidLiner is put together and why. It is written so that a new contributor
can find the right place for a change without reading the whole codebase.

For the reasoning behind individual decisions, see `docs/design/decisions.md`.

---

## 1. The shape of the problem

Training-data production has a different success criterion from image generation:

* the **unit of work** is a tracked object instance, not a frame;
* the **deliverable** is an image plus a correct label plus evidence;
* the **failure that matters** is a plausible-looking sample with a wrong label or a changed
  background, because it teaches the model something false.

VidLiner is therefore built as a *verification pipeline* with a generative step inside it, not as a
generation call with checks bolted on.

---

## 2. Layers

The package is layered, and the dependency direction is enforced by a test
(`tests/unit/test_architecture.py`): a lower layer never imports a higher one, and nothing outside
`backends/` imports a vendor SDK.

```
cli/                 user interface (Typer)
  │
pipeline/            orchestration: recipe → graph → job → export
  │
operators/           the domain operations (one module per stage)
  │
capabilities/        the abstract ports: names + protocols
  │
domain/  core/       typed data     and   primitives (identity, errors, graph, results)
  │
storage/  control/   content-addressed artifacts + SQLite state     dataset discovery, splits,
  │                  and reporting
runtime/             profile loading, backend registry, engine, logs
  │
backends/            the only place a vendor SDK or a model runtime may appear
```

| Layer | Belongs there | Does **not** belong there |
| --- | --- | --- |
| `domain` | Pydantic models, enums, validation rules | I/O, model calls, filesystem access |
| `core` | Canonical JSON, identity, seeds, errors, graph, results | domain-specific knowledge |
| `capabilities` | capability names, backend protocols, `PipelineContext` | any concrete backend |
| `operators` | one domain operation with declared ports and config | vendor imports, direct filesystem access |
| `backends` | adapters: heuristics, fakes, local models, HTTP services | pipeline policy |
| `runtime` | profile, registry, capacity, event log, **engine** | domain logic |
| `pipeline` | recipe compilation, graph assembly, job runner, exporter, policy | operator internals |
| `storage` | workspace layout, artifact store, SQLite state | domain decisions |
| `control` | discovery, splits, duplicates, distribution, review HTML | execution |
| `cli` | argument parsing, rendering, exit codes | business logic |

---

## 3. The graph is the centre of the design

A recipe is compiled into an `OperationGraph` of `OperationNode`s. Each node declares:

* **inputs** — typed ports, bound to another node's output, to a sample value, to a job artifact, or
  to a constant;
* **outputs** — typed ports, which the graph validator checks against every consumer;
* **config** — validated by the operator's Pydantic model at *plan* time, so a config error is never
  discovered mid-job;
* **capabilities** — what the runtime must bind for it to run;
* **determinism**, **cacheability**, **retry policy**, **timeout**, **parallelism**;
* **lineage** — the identity path (`sample → object → candidate`) used for seeds and reporting.

The MVP shape, with `*` marking fan-out:

```
per sample S
  ingest:S ──▶ detect:S ──▶ select:S
                              │
                              ├─▶ per object O:  segment ─▶ scene ─▶ plan
                              │                          │
                              │                          └─▶ per candidate C:
                              │                                generate ─▶ refine ─▶ verify ─▶ evaluate ─▶ annotate ─▶ export
                              └───────────────────────────────────────────────────────────────────────────────────────┘
```

Fan-out is keyed by explicit identity (`object:<id>`, `candidate:<key>`), never by list position, so
adding a candidate value to a recipe cannot renumber the existing ones — which is what makes
cross-run cache reuse correct.

`verify` is the stage that closes the loop: it re-detects the object in the **generated** image and
re-segments it, so the geometry, the boundary metric, and the exported annotation all describe the
object that actually exists rather than the mask the generator was handed. See ADR-018.

**Branches that do not exist are skipped, not failed.** The assembler reserves
`target.max_per_sample` object branches so node identity is stable; a sample with one object simply
has nothing to do for the other branches, and every node downstream of a skip is skipped too.

---

## 4. Execution

`runtime/engine.py` is a single-process `asyncio` scheduler. It owns six concerns and delegates
everything else:

| Concern | Mechanism |
| --- | --- |
| Scheduling | Kahn-style in-degree counting; a node is dispatched the moment its dependencies finish |
| Parallelism | a global worker budget per dispatch round, plus a per-node semaphore |
| Retry | the node's policy, further restricted by whether the backend declared its call safe to retry |
| Timeout | per attempt |
| Cancellation | one `asyncio.Event`, honoured before dispatch and while waiting |
| Cache and resume | a resolved node's port values are rehydrated **before** its dependents are dispatched |

The engine never talks to a backend. It asks a `NodeRunner` to execute one node; in production that
is `pipeline/executor.py`, which:

1. resolves the node's input bindings, applying per-branch selection;
2. builds an `ExecutionContext` with the node, its seed, artifact I/O, the sample values, a backend
   resolver, and the cancellation event;
3. calls the operator;
4. encodes what the operator returned into a `NodeResult` whose payload survives a restart.

Quality rejection is not a failure. The gate is data: `pipeline/policy.py` turns a recipe into an
`AcceptancePolicy` and applies it to the metric evidence, producing an `AcceptanceDecision` with
reason codes. `vidliner qa` runs the same function over stored evidence, which is why changing a
threshold costs nothing.

---

## 5. Capabilities, backends, and profiles

Three documents decide what actually runs:

* the **recipe** says what is wanted, and never names a vendor;
* the **runtime profile** says which backend serves each capability on this machine;
* the **backend contract** says what a backend must be able to do.

```yaml
# runtime.yaml
bindings:
  vision.object_detection.v1: local_detector
  generation.object_replacement.v1: http_replacement
```

Resolution rules, in order and never by guessing: an explicit binding wins; otherwise a backend whose
declared capabilities contain the capability is used, provided exactly one matches; otherwise the
registry instantiates enabled backends and asks them. Zero or several candidates is a pre-flight
failure with a precise message.

A backend receives `PipelineContext` — job id, node id, seed, device, config, and `ArtifactIO`. It
cannot resolve workspace paths, open the state database, or read the recipe.

One capability in the vocabulary is deliberately *not* served by a backend:
`quality.background_preservation.v1` is a deterministic pixel comparison the pipeline performs
itself, so it stays in the vocabulary (recipes gate on it) but is never bound in a profile.

Blocking backends (local models, decoders) declare `blocking = True` and expose their synchronous
work through `self._sync`; the runtime runs that on a worker thread. Native-async backends (HTTP
clients) are awaited directly with a separate capacity budget.

---

## 6. Storage

```
<workspace>/
  runtime.yaml
  state.db                      SQLite: jobs, nodes, candidates, metrics, cache, lineage, duplicates
  artifacts/<aa>/<digest>.<ext> content-addressed artifact store
  runs/<job_id>/
    plan.json  events.jsonl  job-summary.json
    samples/<sample_id>/        source.png refined.png annotation.json quality.json decision.json
    review/index.html           static review report
    dataset/                    exported dataset (created by `vidliner export`)
```

Pixels live on the filesystem keyed by SHA-256; state lives in SQLite. A cached or resumed result is
used only after its artifacts have been verified to exist, so a corrupt cache can never be mistaken
for a valid result.

---

## 7. Extension points

| To add… | Do this |
| --- | --- |
| a detection, segmentation, or generation model | write a backend implementing the protocol, then bind the capability in `runtime.yaml` |
| a new export format | add a converter in `annotations/` and an entry in `ExportFormat` |
| a new quality metric | add it to `MetricName`, compute it in the quality stage, and reference it in a recipe gate |
| a new refinement step | add a capability name, an operator, and a recipe step — the assembler picks it up |
| video support | implement the tracking and video-replacement capabilities; the data model, engine fan-out, and temporal metrics already exist |
| a different execution backend | implement `NodeRunner`; the engine is agnostic |

---

## 8. What the MVP deliberately does not do

No distributed workers, queues, GPU scheduler, object storage, web UI, or multi-user authentication.
The extension points above are the seams where those would attach, and
`docs/design/milestones.md` records the phase plan.
