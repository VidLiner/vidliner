# VidLiner — MVP Milestones

Each milestone is a working increment with a verification command. A milestone is done when its
checks pass, not when its files exist.

## M0 — Skeleton and design (this document set)

* Deliverable: design documents, repository layout, toolchain (ruff, mypy, pytest) wired up.
* Verify: `pytest -q` runs, `ruff check .` is clean, `vidliner --help` prints the command list.

## M1 — Domain and core primitives

* `vidliner.domain`: media, objects, tracks, replacement, annotations, quality, job models.
* `vidliner.core`: canonical JSON, digest/id derivation, seed tree, error taxonomy, results,
  graph, state enums.
* Verify: `pytest tests/unit/test_canonical.py tests/unit/test_identity.py
  tests/unit/test_seedtree.py tests/unit/test_graph.py`.

## M2 — Storage

* Content-addressed artifact store (write/verify/lookup, MIME and dimension validation, size cap).
* SQLite state store with the schema from `storage-schema.md`, plus migrations.
* Workspace layout creation.
* Verify: `pytest tests/unit/test_artifact_store.py tests/unit/test_state_store.py`.

## M3 — Capabilities and backends

* Protocol definitions for every capability.
* Backend registry with lazy import, binding resolution, and probe-based pre-flight.
* Fake backends (deterministic, no external calls) and the built-in local backends:
  heuristic detector/segmenter/scene/planner/evaluator, PIL compositor/harmonizer, phash.
* One generic HTTP replacement backend.
* Verify: `pytest tests/contract tests/unit/test_registry.py`; `vidliner backend check`.

## M4 — Operators and quality

* Every operator in the MVP graph, each with typed ports and a config schema.
* Metric evaluators, acceptance policy engine, reason codes.
* Verify: `pytest tests/unit/test_policy.py tests/unit/test_operators_*.py`.

## M5 — Annotations

* Internal IR converters: COCO detection/instance-segmentation, YOLO detection/segmentation.
* Roundtrip tests including edge cases (empty, out of bounds, crowd, missing image entry).
* Verify: `pytest tests/unit/test_annotations.py`.

## M6 — Engine and job orchestration

* Graph validation, topological scheduling, bounded parallelism, retry, timeout, cancellation,
  resume, node cache, structured JSON logging, job summaries.
* Verify: `pytest tests/pipeline` including resume, cancellation, cache, and restart tests.

## M7 — CLI and end-to-end flow

* `init`, `inspect`, `recipe validate|schema|init`, `plan`, `run`, `jobs`, `job show|cancel|resume`,
  `qa`, `export`, `backend list|check`, `report review`, `dashboard report`.
* Dry run prints estimates without calling a generative backend.
* Static HTML review report.
* Verify: `pytest tests/pipeline/test_end_to_end.py` and a manual CLI walkthrough from
  `docs/getting-started.md`.

## M8 — Dataset integrity

* Lineage-aware split validation, duplicate control (phash + embedding hook), distribution report.
* Verify: `pytest tests/unit/test_splits.py tests/unit/test_duplicates.py
  tests/pipeline/test_export.py`.

## M10 — Closed verification loop (post-MVP, roadmap item 1)

* `verify.redetect` runs between refinement and evaluation: it re-detects the replaced object in the
  generated image and re-segments it, and everything downstream describes that object.
* The exported annotation is rebuilt from the regenerated mask; the mask handed to the generator is
  used only to define the authorised change region.
* Not finding the object is an outcome (`OBJECT_NOT_FOUND`), not a node failure.
* Verify: `pytest tests/unit/test_verification.py tests/pipeline/test_verification_loop.py`, the
  latter end to end against a backend that deletes the object.

## M9 — Documentation and Definition of Done

* README, ARCHITECTURE, CONTRIBUTING and the `docs/` set, plus a runnable example.
* Verify the Definition of Done checklist below on a 10-image synthetic dataset with fake backends.

## Verification record

The MVP is complete. At the time of writing:

| Check | Command | Result |
| --- | --- | --- |
| Format | `ruff format --check .` | 137 files already formatted |
| Lint | `ruff check .` | clean |
| Types | `ty check vidliner` | clean |
| Install | `pip install -e .` then `vidliner --help` | works in a clean environment |
| Tests | `pytest -q` | 367 passed |
| Demo | `python examples/car-swap/demo.py` | runs end to end with no network access |

Size: ~23.4k lines of package code across 105 modules, ~4.9k lines of tests.
Documentation: ~3.0k lines of Chinese text mirroring the English set, plus the design record.

## Definition of Done (from the product requirement §38)

| Requirement | Where it is proven |
| --- | --- |
| Read a 10-image dataset | `tests/pipeline/test_end_to_end.py` |
| Detect target objects | M4 + e2e |
| Build masks | M4 + e2e |
| Generate replacement candidates per object | M4 + e2e |
| Persist provenance | `tests/unit/test_provenance.py` |
| Automatic QA and rejection of low quality | `tests/unit/test_policy.py`, e2e |
| Recompute bbox/mask | M5 + e2e |
| Export COCO | M5, `tests/pipeline/test_export.py` |
| Export YOLO | M5, `tests/pipeline/test_export.py` |
| Dataset report | M8 |
| Resume an interrupted job | `tests/pipeline/test_resume.py` |
| Cache reuse on re-run | `tests/pipeline/test_cache.py` |
| Whole flow with no paid API | all e2e tests use fake/local backends |
| pytest / type check / lint clean | M9 verification |
| Core decoupled from vendor SDKs | `tests/unit/test_architecture.py` |

## Explicitly out of scope for the MVP

Video replacement, tracking execution, distributed workers, queues, cloud object storage, a web
UI, multi-user authentication, and a plugin marketplace. The domain model and engine already
carry the extension points (`ObjectTrack`, `FrameRef`, `temporal_consistency`, per-sample fan-out),
and `docs/design/roadmap.md` records the Phase 2/3 plan.
