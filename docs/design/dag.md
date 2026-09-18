# VidLiner — DAG Design

## 1. Why a graph

The pipeline is not a straight line. One image can contain several target objects; one object
produces several candidates; each candidate is refined, evaluated, re-annotated, and gated
independently. Expressing that as nested loops produces a program that cannot be planned, cached,
resumed, or costed. Expressing it as a graph makes all four properties fall out of the same
structure.

## 2. Node contract

```python
@dataclass(frozen=True)
class OperationNode:
    node_id: str  # deterministic identity (ADR-016)
    operator: str  # registered operator name
    operator_version: str
    stage: PipelineStage  # ingest | detect | segment | scene | plan | generate |
    # refine | evaluate | annotate | gate | export
    inputs: Mapping[str, PortBinding]  # port name -> binding
    outputs: Mapping[str, PortType]  # declared output ports
    config: Mapping[str, Any]  # validated, canonical-serializable
    needs: tuple[str, ...]  # capability names required at runtime
    retry: RetryPolicy
    timeout_s: float | None
    determinism: Determinism  # deterministic | seeded | nondeterministic
    cacheable: bool
    max_parallelism: int
    lineage: tuple[str, ...]  # identity path used for seed derivation & display
```

A `PortBinding` is either `NodeOutput(node_id, port)`, `SampleSource(sample_ref)`,
`ArtifactSource(role)`, or `Constant(value)`. The graph validator rejects any reference to a
node/port that does not exist or whose `PortType` does not match the consumer's declared input
type. Port types are `image`, `image_ref`, `mask_ref`, `instances`, `scene_context`, `plan`,
`candidate_ref`, `quality_report`, `annotation`, `dataset_slice`, `metrics`, `flag`, `any`.

## 3. MVP graph

The assembler (`vidliner.pipeline.assemble`) builds this shape. `*` marks a fan-out point.

```
per sample S:
  ingest:S            (media_asset)                ── no capability
  detect:S            (instances)                  ── needs vision.object_detection.v1
  select:S            (instances filtered)         ── pure
  segment:S           (instances + masks)          ── needs vision.instance_segmentation.v1

  per selected object O: *
    scene:S:O         (scene_context)              ── needs vision.scene_analysis.v1
    plan:S:O          (plan)                       ── needs planning.replacement.v1

    per candidate C of plan: **
      generate:S:O:C      (candidate_ref)          ── needs generation.object_replacement.v1
      refine:S:O:C        (image + mask)           ── needs generation.mask_refinement.v1,
                                                      generation.image_compositing.v1,
                                                      generation.image_harmonization.v1
      verify:S:O:C        (item + mask + found)    ── needs vision.object_detection.v1
                                                      AND vision.instance_segmentation.v1
      evaluate:S:O:C      (quality_report)         ── needs quality.* evaluator capabilities
      annotate:S:O:C      (annotation)             ── pure; rebuilds from verify's mask
      export:S:O:C        (per-sample evidence)    ── pure; materialises the run's evidence

  The dataset-level export (one COCO/YOLO document, the manifest, the report) runs once per job
  after the graph completes. The per-sample `export:S:O:C` node above materialises the evidence that
  export consumes.
```

Fan-out keys, not positions: `O` is `object:<object_id>`, `C` is `candidate:<candidate_key>`.
Adding a candidate value to the recipe therefore never changes the node id of an existing
candidate, which is what makes cross-run cache reuse correct.

The `source → S` edges carry the original `MediaAsset`; a parallel `image_ref` is derived at
`ingest` so that quality metrics can compare pre- and post-replacement pixels without reloading
the file.

## 4. Engine responsibilities

| Concern | Mechanism |
| --- | --- |
| Validation | `OperationGraph.validate()` — unique ids, acyclic, port/type agreement, capability availability checked against the runtime profile before execution |
| Scheduling | Kahn-style ready queue with per-node in-degree. Nodes whose dependencies are satisfied are dispatched immediately. |
| Parallelism | `asyncio`; a global worker semaphore plus a per-backend semaphore. Blocking backends run in `asyncio.to_thread`. |
| Retry | `RetryPolicy(attempts, backoff_s, jitter, retry_on)`. Backends declare `safe_to_retry`; generative operations default to `attempts=1` unless the backend declares idempotence. |
| Timeout | `asyncio.wait_for` around each attempt, per node. |
| Cancellation | One `asyncio.Event`; checked before dispatch and awaited during long operator awaits. Cancelled nodes persist as `cancelled`, not `failed`. |
| Resume | Completed nodes are loaded from `state.db`; a node is skipped only when its recorded output digests still exist and verify. |
| Cache | Cache key from ADR-005; lookup before dispatch; skipped nodes are recorded with `cache_hit=1`. |
| Failure isolation | A node failure fails only its descendants. Sibling candidates continue. A job fails when a *required* node fails (no candidate could be produced for any target) or when `--fail-fast` is set. |
| QualityReject | Never a node failure. `gate` completes normally and emits a `rejected` decision. |
| Logging | Structured JSON events, one line per state transition, plus `events.jsonl` per run and a row per node in `state.db`. |

## 5. Job and candidate state machines

Job:

```
CREATED → PLANNED → RUNNING → (WAITING) → FINALIZING → SUCCEEDED
                        └──────────────────────────────→ FAILED
                        └──────────────────────────────→ CANCELLED
```

`WAITING` is entered when every runnable node is blocked on an external asynchronous backend
(`submit`/`poll`); it returns to `RUNNING` on the next successful poll. `FINALIZING` covers export
and report writing.

Candidate:

```
GENERATED → EVALUATING → ACCEPTED
                       → REJECTED
                       → REVIEW        (needs human decision; not exported until decided)
```

Persisted transitions are append-only in `candidate_events`; the current state is a column for
query speed. The UI never infers state (requirement §15).

## 6. Dry run

`vidliner plan --dry-run` builds the graph, resolves bindings, and prints:

* sample count, target-object count (from ingestion-time metadata or the detection estimate),
* candidate count, selected backends per capability,
* estimated GPU operations (nodes whose bound backend declares device `cpu`/`cuda`),
* estimated external API operations and estimated cost from `Estimate` in the profile,
* the topological stage order with per-stage node counts.

No generation backend is ever called during `plan` — capability *resolution* is checked, but the
generative capability is not invoked. The `plan` command also writes the plan to
`runs/<job>/plan.json` so `run --plan` can execute the exact reviewed graph.
