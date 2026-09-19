# Testing

[简体中文](./zh/testing.md)

The suite is the project's main safety net, and it has one non-negotiable property: **it never
touches the network, a paid API, or a downloaded model.** Everything runs against synthetic fixtures
generated in-process.

```bash
pytest -q                       # everything
pytest tests/unit -q            # primitives and models
pytest tests/contract -q        # backend protocols
pytest tests/pipeline -q        # end-to-end jobs
pytest -q -m "not slow"         # skip the few slow cases
ruff check .                    # lint
ty check vidliner               # type check
```

---

## Layout

```
tests/
  conftest.py            fixtures: workspace, session, synthetic dataset, recipe builder
  fixtures/              backends that exist only for a test: one that erases the object,
                         and declarative markers for the production guard
  unit/                  canonical + identity, domain models, engine, storage, quality,
                         annotations, verification, production guard, docs consistency,
                         architecture rules
  contract/              every backend through its protocol
  pipeline/              end to end, export, resume + cache, verification loop, CLI
```

---

## Fixtures

`vidliner/fixtures.py` renders synthetic scenes with numpy: a textured background, a soft vertical
gradient (so lighting measurements have something to measure), and one or more objects with shading
and a darker bottom edge. `build_dataset` writes them as a dataset in any supported layout.

The fixtures in `tests/conftest.py` compose those into what a test actually wants:

| Fixture | Gives you |
| --- | --- |
| `workspace` | an initialised workspace with the default local profile |
| `session` | an open `Session` over that workspace |
| `dataset_root` | ten synthetic images plus a COCO annotation file |
| `synthetic_dataset` | the dataset object, when a test needs to inspect what was drawn |

`make_recipe(...)` builds a valid recipe for that dataset. Its default gates are deliberately
permissive: a test that wants to exercise *rejection* tightens them explicitly, so a general
pipeline test is never gated on a heuristic quality score it is not testing.

---

## The backend test doubles

`RecordingRunner` (in `tests/unit/test_engine.py`) is a `NodeRunner` that records calls and can be
told to fail, to block, or to raise a specific error. It is how the scheduler is tested in isolation:
a failure there means the *engine* is wrong, not an operator.

`tests/contract/test_backends.py` exercises each backend through its protocol with a real
`ArtifactStore` and a real `PipelineContext`. The same assertions apply to a heuristic, the
deterministic stand-in, and the HTTP adapter.

---

## What is deliberately covered

| Area | Representative tests |
| --- | --- |
| identity and determinism | quantised object ids, seed-tree stability under a new candidate, recipe hashing that ignores estimates |
| canonical serialization | sorted keys, `-0.0`, enums, UTC timestamps, refusal of NaN and unknown types |
| graph validation | duplicate ids, unknown dependencies, unknown ports, cycles, port-type agreement |
| scheduling | linear graphs, fan-out, retry policy, no-retry-by-default, unsafe-backend refusal, timeouts, cancellation, descendant skipping, sibling survival, fail-fast |
| cache and resume | cache hit only with identical inputs, resume of a cancelled job, cache dropped when artifacts vanish |
| storage | digest verification, size and dimension guards, path traversal refusal, candidate lifecycle, lineage, leakage detection |
| quality | gate ordering, hard gates beating a high overall score, review band, warn gates, missing-metric policies |
| annotations | COCO and YOLO roundtrips, malformed lines, missing class names, empty bundles |
| the pipeline | ten-image end-to-end run, COCO and YOLO export, provenance completeness, secret absence, split safety, multi-target fan-out, skip semantics |
| the verification loop | an object that is not in the generated image yields `found=False` and `OBJECT_NOT_FOUND` rather than a node failure; the exported label comes from the regenerated mask |
| the production guard | a demonstration stack is refused at pre-flight and again at export, all four critical capabilities are named at once, and the manifest is the source of truth |
| documentation | the reason-code listing and the capability table match the code in both languages |
| architecture | no vendor SDK outside `backends/`, no upward layer imports, no silent exception handling, no undocumented public API |

---

## The specific edge cases the requirements call out

| Case | Test |
| --- | --- |
| mask empty | `test_fake_generator_refuses_an_empty_mask`, `test_segmentation_needs_a_prompt` |
| bbox outside image | `test_annotation_validation_reports_out_of_bounds` |
| object missing | `test_class_filter_produces_no_targets_and_still_succeeds` |
| wrong replacement class | `test_semantic_evaluator_flags_a_mismatch` |
| background changed | `test_background_measurements_detect_a_recoloured_background`, `test_quality_rejection_is_not_a_job_failure` |
| duplicate generation | `test_duplicate_index_detects_an_exact_copy`, `test_near_duplicate_outputs_are_excluded` |
| video frame missing | `test_detector_refuses_video_input` |
| backend timeout | `test_engine_enforces_a_node_timeout` |
| process restart | `test_resume_after_an_interrupted_job` (state is reloaded from SQLite) |
| job cancellation | `test_engine_honours_cancellation`, `test_job_cancel_marks_the_job` |
| invalid annotation | `test_export_with_a_tiny_minimum_area_drops_samples` |
| dataset leakage | `test_exporter_refuses_a_leaking_dataset`, `test_assert_no_leakage_reports_every_offending_edge` |
| dangling capability binding | `test_default_profile_has_no_dangling_binding` |

---

## Adding a test

1. Prefer a synthetic fixture over a fixture file. A binary blob in the repository is a fixture
   nobody can review.
2. Assert the *contract*, not the implementation. A test that asserts an internal helper was called
   breaks when the helper is renamed and never catches a real bug.
3. For a new backend, add a probe test, a happy-path test, and one test per legitimate refusal.
4. For a new gate, add both a passing and a failing case, and assert the reason code.
5. If the change is architectural — a new layer, a new vendor dependency — add the rule to
   `tests/unit/test_architecture.py` first. The rule is cheaper than the review comment.
6. If you add a reason code or a capability, `tests/unit/test_docs_consistency.py` fails until both
   language versions list it. Documentation that mirrors a closed enum is generated from it or
   checked against it; here it is checked, because prose around the list is worth keeping.
