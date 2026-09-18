# Runtime profiles

[简体中文](./zh/runtime.md)

A recipe says *what* you want. A runtime profile says *which backend does it on this machine*. That
separation is what lets the same recipe run against built-in heuristics today, a local model
tomorrow, and a hosted service next week without touching the recipe.

---

## Where a profile is found

1. `--runtime <path>` on the command line;
2. `$VIDLINER_RUNTIME`;
3. `runtime.yaml` in the workspace root;
4. otherwise the built-in local profile.

`vidliner init` writes the local profile, which is why a fresh checkout runs the whole pipeline with
no downloads.

---

## Shape

```yaml
profile: workstation
device: cpu                       # cpu | cuda | mps | auto

storage:
  root: .
  keep_rejected: true
  max_artifact_mb: 256
  state_filename: state.db

concurrency:
  workers: 4                      # maximum concurrent nodes
  per_backend: 2                  # default per-backend bound
  external_requests: 3            # reserved for network-bound backends

backends:
  local_detector:
    use: vidliner.backends.heuristic.detector:HeuristicDetectorBackend
    options: { score_floor: 0.30 }
    capacity: { limit: 2 }
    estimates: { unit_seconds: 0.4 }

  http_replacement:
    use: vidliner.backends.http_replacement:HttpReplacementBackend
    options:
      endpoint: http://127.0.0.1:8080/v1/edit
      request_timeout_s: 120
      max_response_mb: 32
      allowed_media_types: [image/png, image/jpeg, image/webp]
    credentials:
      api_key: { source: env, name: VIDLINER_EDIT_API_KEY }
    capacity: { limit: 1, period_s: 6 }
    estimates: { unit_cost: 0.02, unit_seconds: 9, external: true }

bindings:
  vision.object_detection.v1: local_detector
  generation.object_replacement.v1: http_replacement
```

---

## `use`

An import path, either `package.module:ClassName` or `package.module.ClassName`. The class must
expose `backend_id`, `backend_version`, `capabilities`, and `probe()`, and must implement at least
one capability protocol.

A backend may declare its capabilities as a class attribute (`declared_capabilities`) or through the
spec's `options.declared_capabilities`. Declaring them means `vidliner plan` can resolve bindings
**without importing the backend module**, which is what keeps planning fast and dependency-free.

---

## Binding resolution

In order, and never by guessing:

1. an explicit `bindings[capability]` entry wins;
2. otherwise a backend whose declared capabilities contain the capability is used, provided exactly
   one matches;
3. otherwise enabled backends are instantiated and asked what they serve;
4. zero candidates is an unmet capability; several is an ambiguity error naming the candidates.

`vidliner plan` reports unmet capabilities and exits `1`. `vidliner run` refuses to start.

---

## Capacity

`capacity.limit` bounds concurrent calls into a backend. `capacity.period_s` adds a replenishing rate
budget: at most `limit` calls may *start* in any window of that many seconds, which is how a hosted
service's requests-per-minute policy is respected without serialising the client.

`blocking: true` backends (local models, decoders, subprocesses) run their work on a worker thread so
the job's event loop keeps scheduling. Network-bound backends are awaited directly and drawn from a
separate budget.

---

## Credentials

Credentials are **references**, never values, inside a repository:

```yaml
credentials:
  api_key: { source: env,  name: VIDLINER_EDIT_API_KEY }
  token:   { source: file, name: /run/secrets/edit-token }
```

`source: value` (an inline literal) is refused when the profile lives inside a directory with a VCS
marker, and `vidliner backend check` flags it otherwise.

Resolution happens only in `vidliner/runtime/secrets.py`. Resolved values are held in memory for the
run, redacted from every log line and every serialized profile, and never written to a recipe, a
manifest, a plan, a log, or a provenance record. A test asserts this end to end.

---

## Estimates

`estimates` feed `vidliner plan` only:

| Field | Meaning |
| --- | --- |
| `unit_cost` / `currency` | Money per invocation. |
| `unit_seconds` | Wall-clock per invocation. |
| `external` | Counts as an external call in the estimate. |
| `accelerator` | Counts as accelerator work in the estimate. |

When a backend declares nothing, the stage is counted but contributes zero cost and zero time, and
the plan says so in its notes rather than inventing an average.

---

## Pre-flight

```bash
vidliner backend list    --workspace ./playground     # what is configured
vidliner backend check   --workspace ./playground     # probe and report coverage
vidliner backend list --capabilities                  # the capability vocabulary
```

Capabilities the pipeline measures itself (currently `quality.background_preservation.v1`) are
reported as satisfied by `builtin`; they need no backend and must not appear in `bindings`.

`check` calls `probe()` only. A probe must be cheap and must never perform a paid or generative call:
it checks credential presence, endpoint reachability, model files, and device availability. The
report lists each backend's `backend_id`, `version`, health (`ready`, `degraded`, `unavailable`),
advertised capabilities, determinism, `safe_to_retry`, and device, then lists any capability with no
backend.

---

## Determinism and retry

A backend declares how repeatable it is (`deterministic`, `seeded`, `nondeterministic`) and whether
retrying the same request is safe. The engine uses both:

* a node is cached only when it is deterministic or seeded;
* a generative operation is not retried unless its backend declares `safe_to_retry`, because a retry
  may produce a different image and consume another paid call.

---

## Replacing the built-in backends

The fastest path to production accuracy is to keep the pipeline and swap three capabilities:

```yaml
backends:
  my_detector:
    use: my_project.detector:DetectorBackend
    options: { model_path: /models/yolo.onnx, device: cuda }
    estimates: { unit_seconds: 0.05, accelerator: true }
  my_segmenter:
    use: my_project.segmenter:SegmenterBackend
    options: { model_path: /models/sam.onnx }
  my_editor:
    use: my_project.editor:EditorBackend
    credentials: { api_key: { source: env, name: EDIT_KEY } }

bindings:
  vision.object_detection.v1: my_detector
  vision.instance_segmentation.v1: my_segmenter
  generation.object_replacement.v1: my_editor
```

Nothing else changes. See `docs/backends.md` for the contracts.
