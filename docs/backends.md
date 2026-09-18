# Writing a backend

[简体中文](./zh/backends.md)

A **capability** is an ability the pipeline needs. A **backend** is a concrete implementation of one
or more capabilities. The pipeline names only capabilities, so replacing a model is a configuration
change.

---

## The vocabulary

| Capability | Contract | Used by |
| --- | --- | --- |
| `vision.object_detection.v1` | `detect` | the detect stage **and** post-generation verification |
| `vision.instance_segmentation.v1` | `segment` | the segment stage **and** post-generation verification |
| `vision.scene_analysis.v1` | `analyse` | the scene stage |
| `vision.object_tracking.v1` | `track` | reserved (video) |
| `vision.depth_estimation.v1` | `estimate_depth` | reserved (video) |
| `planning.replacement.v1` | `plan` | the plan stage |
| `generation.object_replacement.v1` | `replace` | the generate stage |
| `generation.image_compositing.v1` | `composite` | the refine stage |
| `generation.image_harmonization.v1` | `harmonize` | the refine stage |
| `generation.mask_refinement.v1` | `refine_mask` | the refine stage |
| `generation.video_replacement.v1` | — | reserved (video) |
| `quality.semantic_match.v1` | `assess_semantics` | the evaluate stage |
| `quality.background_preservation.v1` | — | **built-in**: measured in-house, never bound |
| `quality.artifact_detection.v1` | `evaluate` | the evaluate stage |
| `quality.vision_evaluation.v1` | `evaluate` | the evaluate stage |
| `quality.embedding.v1` | `embed` | duplicate control |

`vidliner backend list --capabilities` prints this list with a one-line description of each.

Most capabilities are served by a backend. One is not: `quality.background_preservation.v1` is
measured by the pipeline itself in `vidliner/quality/metrics.py`, because "did the non-target pixels
change" is a deterministic pixel comparison rather than a model judgement. It stays in the vocabulary
because recipes reference it in acceptance gates, but a runtime profile must not bind it, and
`vidliner backend check` reports it as satisfied by `builtin` instead of uncovered.

---

## The minimum contract

```python
class MyDetector:
    backend_id = "my_detector"  # stable id, used in manifests and cache keys
    backend_version = "1.0.0"  # version of the adapter, not of the model
    capabilities = (CAP_OBJECT_DETECTION,)
    determinism = Determinism.DETERMINISTIC
    safe_to_retry = True
    blocking = False  # True for local CPU/GPU work

    def __init__(self, spec, options, credentials=None):
        self.options = options

    async def probe(self) -> BackendProbe:
        return BackendProbe(
            backend_id=self.backend_id,
            version=self.backend_version,
            health="ready",
            capabilities=self.capabilities,
            determinism=self.determinism,
            message="loaded yolo11n.onnx",
        )

    async def detect(self, request: DetectionRequest, context: PipelineContext) -> DetectionResult:
        image = context.require_io().image(request.image)  # a Pillow image
        ...  # your inference
        return DetectionResult(instances=instances, backend_id=self.backend_id)
```

A backend may satisfy several capabilities; declare them all in `capabilities` and implement the
matching methods.

---

## Two shapes of backend

**Native async** (an HTTP client, a remote service). Write `async def` methods and `await` inside
them. The runtime awaits them directly.

**Blocking** (a local model, a decoder, a subprocess). Set `blocking = True` and put the synchronous
implementations on `self._sync` under the protocol method names:

```python
class LocalDetector:
    blocking = True

    def __init__(self, spec, options, credentials=None):
        self._sync = _Sync(self)

    async def detect(self, request, context):
        return self._sync.detect(request, context)  # the runtime runs this on a worker thread


class _Sync:
    def __init__(self, backend):
        self._backend = backend

    def detect(self, request, context): ...
```

The runtime calls `self._sync.detect` on a thread. Never block the event loop inside an `async def`
that the runtime awaits directly.

---

## What a backend may access

`PipelineContext` carries:

| Field | Use |
| --- | --- |
| `job_id`, `node_id` | logging and diagnostics |
| `seed` | the derived seed for this node — **use it instead of any global randomness** |
| `device` | device hint from the profile |
| `config` | the node's validated configuration |
| `metadata` | profile options and estimates |
| `io` | an `ArtifactIO`: `load`, `load_json`, `image`, `load_mask`, `save_image`, `save_mask`, `save_json` |

A backend cannot resolve workspace paths, open the state database, read the recipe, or reach the
network except through its own client. That is deliberate: it is what makes a backend replaceable and
testable.

**Determinism is a requirement, not a nicety.** Two runs of the same node with the same seed must
produce the same output, or caching, resume, and provenance all become meaningless.

---

## Requests and results

Requests carry **artifact references**, not pixels, so they are cacheable and serialize cleanly. The
exception is `SceneAnalysisRequest`, which carries the already-decoded mask and luminance because the
operator that builds it had to decode them anyway.

Result objects carry the identity of what produced them:

```python
DetectionResult(instances=..., backend_id=..., model_id=..., duration_ms=..., raw={...})
GenerationOutcome(image=..., backend_id=..., model_id=..., seed=..., cost_estimate=..., raw_metadata={...})
```

`raw_metadata` is recorded for audit. It is **never** an input to a quality gate — a generator's own
opinion of its output is not evidence.

---

## Errors

Raise the structured errors from `vidliner.core.errors`:

| Error | Meaning |
| --- | --- |
| `BackendFailure(..., safe_to_retry=...)` | unreachable, timed out, rate limited, returned something unusable |
| `OperatorFailure` | the request itself cannot be satisfied (for example a text prompt this backend does not support) |
| `ValidationFailure` | the input violates a contract |

Set `safe_to_retry=True` only when retrying the identical request is idempotent. A generative call
normally is not.

---

## The built-in backends

They exist so the pipeline runs on a fresh checkout, and they are honest about what they are:

| Backend | Capabilities | Notes |
| --- | --- | --- |
| `heuristic_detector` | detection | saliency-based; reports only classes it knows *and* the dataset labels |
| `heuristic_segmenter` | segmentation | grows a mask from a box or point prompt; refuses a text-only prompt |
| `heuristic_scene` | scene analysis | measures orientation, lighting, ground contact, shadow, occlusion |
| `rule_based_planner` | planning | deterministic; makes no model calls, ever |
| `fake_replacement` | replacement | deterministic synthetic imagery, marked `synthetic: true` |
| `local_metric_evaluator` | semantic, artifact, embedding | colour-appearance heuristic plus artifact measurements |
| `local_refiner` | mask, blend, harmonise | deterministic numpy operations |
| `perceptual_hash` | embedding | 64-bit pHash exposed as a vector |
| `http_replacement` | replacement | generic JSON + base64 adapter for any editing service |

Each built-in probe reports `degraded` where it is a baseline rather than a trained model, so
`vidliner backend check` never overstates readiness.

---

## The HTTP adapter

`HttpReplacementBackend` talks to any service that accepts a JSON body with a base64 image and a
prompt and returns a JSON body with a base64 image:

```yaml
http_replacement:
  use: vidliner.backends.http_replacement:HttpReplacementBackend
  options:
    endpoint: http://127.0.0.1:8080/v1/edit
    request_image_field: image_base64
    request_mask_field: mask_base64
    request_prompt_field: prompt
    request_negative_field: negative_prompt
    response_image_field: image_base64
    response_image_encoding: base64
    allowed_media_types: [image/png, image/jpeg, image/webp]
    max_response_mb: 32
    request_timeout_s: 120
    extra_fields: { model: my-editor-v2 }
  credentials:
    api_key: { source: env, name: VIDLINER_EDIT_API_KEY }
```

The response is treated as untrusted input: size is capped, the declared media type must match the
decoded bytes and be in `allowed_media_types`, dimensions are validated against the store's guard,
and the image is written through the artifact store, which validates again.

---

## Contract tests

`tests/contract/test_backends.py` exercises every built-in backend through its protocol with
synthetic fixtures. A new backend should get the same treatment: a probe test, a happy-path test, and
a test for each way it can legitimately refuse.

Two properties are worth asserting for any backend:

* **determinism** — the same request twice yields the same artifact digest (for `seeded` and
  `deterministic` backends);
* **containment** — a generative backend changes only the region it was asked to change.
