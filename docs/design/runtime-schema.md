# VidLiner — Runtime Profile Schema

A runtime profile answers one question: *on this machine, which backend serves each capability?*
It is separate from the recipe so that the same recipe runs against local models, a local HTTP
inference server, or a hosted generative service without edits.

File: `runtime.yaml`, discovered in the workspace root, overridable with `--runtime <path>` or the
`VIDLINER_RUNTIME` environment variable.

```yaml
profile: workstation          # required, slug
device: cpu                   # cpu | cuda | mps | auto  (advisory; a backend may override)

storage:
  root: .                     # workspace root; relative paths resolve against the profile file
  keep_rejected: true         # retain rejected artifacts for diagnostics
  max_artifact_mb: 256        # refuse to ingest an artifact larger than this

concurrency:
  workers: 4                  # global bounded parallelism
  per_backend: 2              # default per-backend bound
  external_requests: 3        # separate bound for backends that make network calls

backends:
  local_detector:
    use: vidliner.backends.heuristic.detector:HeuristicDetectorBackend
    options:
      score_floor: 0.3
    capacity: { limit: 2 }
    estimates: { unit_seconds: 0.4 }
    demo_only: true             # a saliency heuristic, not a detector

  local_segmenter:
    use: vidliner.backends.heuristic.segmentation:HeuristicSegmentationBackend
    options: { dilate_px: 2 }
    estimates: { unit_seconds: 0.6 }
    demo_only: true

  local_scene:
    use: vidliner.backends.heuristic.scene:HeuristicSceneBackend
    demo_only: true

  local_planner:
    use: vidliner.backends.heuristic.planner:RuleBasedPlannerBackend
    demo_only: true

  http_replacement:
    use: vidliner.backends.http_replacement:HttpReplacementBackend
    options:
      endpoint: http://127.0.0.1:8080/v1/edit
      request_timeout_s: 120
      max_response_mb: 32
      allowed_media_types: [image/png, image/jpeg, image/webp]
      response_image_field: image_base64
    credentials:
      api_key: { source: env, name: VIDLINER_EDIT_API_KEY }
    capacity: { limit: 1, period_s: 6 }
    estimates: { unit_cost: 0.02, unit_seconds: 9, external: true }

  local_evaluator:
    use: vidliner.backends.local.evaluator:LocalMetricEvaluatorBackend
    demo_only: true

bindings:
  vision.object_detection.v1: local_detector
  vision.instance_segmentation.v1: local_segmenter
  vision.scene_analysis.v1: local_scene
  planning.replacement.v1: local_planner
  generation.object_replacement.v1: http_replacement
  quality.semantic_match.v1: local_evaluator
  quality.artifact_detection.v1: local_evaluator
```

## Rules

1. `use` is an import path `module:Attribute` or `module.Attribute`. A backend class must
   implement at least one capability protocol and expose `backend_id`, `backend_version`,
   `capabilities`, and `probe()`.
2. A capability that is required by the compiled graph and has no binding is a **pre-flight
   validation failure**. `vidliner plan` reports it; `vidliner run` refuses to start.
3. When exactly one configured backend advertises a capability, the binding is optional. When
   several do, an explicit binding is required — the resolver never guesses.
4. `credentials` entries are references, never values, in a profile stored inside a repository.
   `{source: value, name: "..."}` is accepted only for profiles outside the workspace boundary and
   is flagged in `backend check`.
5. `capacity.limit` bounds concurrent calls into that backend; `period_s` adds a replenishing rate
   budget (used to respect hosted-service rate limits).
6. `estimates` feed `plan`/`--dry-run` only. They never affect execution.
7. `device` is advisory: a backend may declare a device requirement, and a profile whose `device`
   cannot satisfy it fails pre-flight with `DEVICE_UNAVAILABLE`.
8. `demo_only` marks a backend as a stand-in. It changes no behaviour except one: when a capability
   in `PRODUCTION_CAPABILITIES` resolves to a marked backend, the dataset may not be exported unless
   the recipe sets `acceptance.allow_demo_backends`. A backend class may also declare the marker
   itself, and that declaration is authoritative. The check reads the marker only — it never
   imports, instantiates, or probes a backend, so it costs nothing at plan time.
9. Capabilities the pipeline measures itself (`BUILTIN_CAPABILITIES`, currently
   `quality.background_preservation.v1`) **must not** appear in `bindings`: they need no backend.

## Redaction

Any serialized profile (in a job manifest, a plan artifact, or a log event) passes through
`vidliner.runtime.secrets.redact_profile`, which replaces resolved credential values with the
literal string `"<redacted>"`. Test `test_provenance_has_no_secrets` asserts this end to end.

## Pre-flight (`vidliner backend check`)

For each configured backend the CLI prints: `backend_id`, `backend_version`, advertised
capabilities, probe result (`ready` / `degraded` / `unavailable` with a reason), declared
determinism, `safe_to_retry`, device, and whether an estimate exists. Pre-flight never performs a
generative call: it calls `probe()` only.
