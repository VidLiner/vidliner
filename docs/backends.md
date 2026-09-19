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

If your adapter is a **stand-in** — a heuristic, a stub, a baseline that produces plausible-looking
output without measuring what the capability claims to measure — say so:

```python
class MyPlaceholderDetector:
    demo_only = True  # the runtime profile may repeat this, but the class is authoritative
```

This is not a comment. It is what stops the pipeline from exporting a dataset built on it. See
[Demonstration backends and the production guard](#demonstration-backends-and-the-production-guard).

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

| Backend | Capabilities | Demo? | Notes |
| --- | --- | --- | --- |
| `heuristic_detector` | detection | **yes** | saliency-based; reports only classes it knows *and* the dataset labels |
| `heuristic_segmenter` | segmentation | **yes** | grows a mask from a box or point prompt; refuses a text-only prompt |
| `heuristic_scene` | scene analysis | **yes** | measures orientation, lighting, ground contact, shadow, occlusion |
| `rule_based_planner` | planning | **yes** | deterministic; makes no model calls, ever |
| `fake_replacement` | replacement | **yes** | deterministic synthetic imagery, marked `synthetic: true` |
| `local_metric_evaluator` | semantic, artifact, embedding | **yes** | colour-appearance heuristic plus artifact measurements |
| `local_refiner` | mask, blend, harmonise | no | deterministic numpy operations |
| `perceptual_hash` | embedding | **yes** | 64-bit pHash exposed as a vector |
| `http_replacement` | replacement | no | generic JSON + base64 adapter for any editing service |

Each built-in probe reports `degraded` where it is a baseline rather than a trained model, so
`vidliner backend check` never overstates readiness.

---

## Demonstration backends and the production guard

The default profile is a **demonstration stack**: it is what makes the project runnable and testable
with no model download and no API key. Every one of its perception, generation, and evaluation
backends is a stand-in.

They are useful precisely because they are not real, and they are dangerous for the same reason. A
saliency detector reports "something is here" rather than a class. A synthetic generator paints a
flat colour rather than a car. A colour heuristic cannot judge whether an SUV looks like an SUV. A
dataset built on them is *shaped* exactly like training data — same files, same schema, same
provenance — and it is not training data, because nothing in the loop ever measured the thing the
label asserts.

So the pipeline separates two questions that a single "did it work" flag would conflate:

| Question | Answer |
| --- | --- |
| May the job run? | Yes, with any bound backend. The demo, the tests, and a first look at a new dataset all depend on it. |
| May the result be exported? | Only when the capabilities whose output *becomes* the data are served by production backends, or when the recipe says explicitly that it knows what it is asking for. |

The four capabilities are listed in `PRODUCTION_CAPABILITIES`
(`vidliner/capabilities/names.py`) and are the ones where a stand-in corrupts the data itself rather
than the process that produced it:

| Capability | Why it is critical |
| --- | --- |
| `vision.object_detection.v1` | what is detected is what gets replaced and re-labelled |
| `vision.instance_segmentation.v1` | what is segmented is the exported mask, bbox, polygon, and area |
| `generation.object_replacement.v1` | what is generated is what the model trains on |
| `quality.semantic_match.v1` | what is judged is what is allowed through |

Everything else — the planner, the refiner, the scene analyser, the artifact evaluator, the embedder
— is **not** critical: a stand-in there changes *how* a sample was made, not whether its label
describes the picture. Refusing those would make the guard unusable and would be false precision.

**How to declare a backend trustworthy.** There is no flag for it. Write an adapter that declares the
capability and does the work, and bind it:

```yaml
# runtime.yaml
backends:
  prod_detector:
    use: my_project.backends:YoloDetector
    options: { weights: checkpoints/yolo11n.onnx }
bindings:
  vision.object_detection.v1: prod_detector
  vision.instance_segmentation.v1: prod_segmenter
  generation.object_replacement.v1: prod_editor
  quality.semantic_match.v1: prod_judge
```

The guard reads the `demo_only` marker (from the profile entry and from the class itself) and
nothing else. It does not probe, import, or construct a backend, and it does not test whether a
backend is *good* — only whether it claims to be a stand-in. Judging quality is what the quality
gates are for; judging honesty is what this marker is for.

**When the dataset is deliberately a demonstration.** Set it in the recipe, where it is recorded and
reviewable:

```yaml
acceptance:
  allow_demo_backends: true   # inspection data, not training data — the manifest will say so
```

or pass `--allow-demo` to `vidliner run` / `vidliner export` for a one-off. The manifest then records
`demo_backends` and `demo_backends_allowed`, and `vidliner plan` prints the same warning *before*
anything is generated. A profile edited after a run cannot retroactively make that run
production-grade: the exported decision is based on what the manifest says the job actually used.

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
