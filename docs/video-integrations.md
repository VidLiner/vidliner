# Video and canvas integration contracts

These are library and CLI integration boundaries for an authenticated UI host. The standalone
HTML editor remains offline: no browser-supplied endpoint, credential, or execute button can start
a provider call. A host can add transport routes without creating a second graph or scheduler.

## Video models

`generation.video_generation.v1` binds through the same `RuntimeProfile`, `BackendRegistry`,
`CapabilityBroker`, `CredentialResolver` and capacity limiter used by image backends. A task adapter
implements the existing minimum `Backend` contract plus `VideoGenerationBackend`:

| Method | Input | Result / meaning |
| --- | --- | --- |
| `submit(request, context)` | `VideoGenerationRequest` | `VideoTask`; queued is not finished or accepted. |
| `status(task, context)` | Persisted `VideoTask` | Current status and output URLs; never creates a task. |
| `cancel(task, context)` | Persisted `VideoTask` | Cancel request; provider may still finish. |
| `probe()` | No generation input | Configuration/credential readiness, no paid call. |

`HttpVideoGenerationBackend` supports Runway text/image generation and fal's queue API. The
[example runtime](../examples/video/runtime-ai.yaml) includes Runway Gen-4.5, Kling, Wan and Veo
configurations. Provider endpoints and model parameters were checked against official docs;
integration tests use HTTP fixtures. They are not a claim of live generation with every model,
account entitlement, model quality or production acceptance.

| Route | Configuration | Official protocol/schema |
| --- | --- | --- |
| Runway native API | `provider: runway`, model ID, enabled modes | [Task lifecycle](https://docs.dev.runwayml.com/ai-context.md), [OpenAPI](https://docs.dev.runwayml.com/openapi.json) |
| Kling via fal | `provider: fal`, Kling endpoint, image mode | [Kling schema](https://fal.ai/models/fal-ai/kling-video/v2.5-turbo/pro/image-to-video/api) |
| Wan via fal | `provider: fal`, Wan endpoint, text mode | [Wan schema](https://fal.ai/models/fal-ai/wan/v2.2-a14b/text-to-video/api) |
| Veo via fal | `provider: fal`, Veo endpoint, text mode | [Veo schema](https://fal.ai/models/fal-ai/veo3.1/api) |
| Other fal video endpoints | Model path, mode, field mapping | [Queue lifecycle](https://docs.fal.ai/model-apis/model-endpoints/queue) and the selected model's API page |
| Additional native services / local inference | New adapter at `runtime.backends.<name>.use` | Implement the same lifecycle protocol; no operator or engine fork. |

Models do not share a universal wire format. Keep duration, resolution, audio and reference
parameters in `VideoGenerationRequest.parameters` in the provider's own types. Runway maps prompt
and image to `promptText/promptImage`; fal defaults to `prompt/image_url/video_url`, configurable
through `prompt_field/image_field/video_field/seed_field`. `result_field` selects a dotted JSON
path (default `video.url`). Set `seed_field: null` for models that lack a seed. Unsupported modes
and attempts to override core fields fail before submission. Local image artifacts are loaded
through existing digest-verified `ArtifactIO` and sent as capped inline data URIs; URL inputs must
already be accessible to the provider. No automatic media upload service is created.

Do not automatically retry `submit`: an HTTP timeout may mean a remote task exists even though its
ID was not received. Save returned task handles immediately; query or cancel those handles rather
than re-running the submitting workflow. fal server retries are disabled on submission. Control
URLs must match the configured provider origin, model route and task ID; credentials are never
sent to returned media URLs. Responses are streamed with a byte cap and redirects are refused.
Output URLs may expire and may contain signed access parameters: treat task handles as private
runtime state. A future asset-import host must separately download, validate and store media.

`VideoTask.verified` is always false, and its label transform defaults to `synthesized`. An output
URL is not a digest-addressed artifact or verified training sample. The existing
`generation.video_replacement.v1` remains separate: ordinary text/image generation does not prove
object replacement, source-label preservation or temporal consistency.

CLI discovery and palette schemas:

```sh
vidliner video request-schema
vidliner video task-schema
vidliner workflow catalog
vidliner workflow edit-schema
vidliner workflow execution-schema
```

The catalogue includes `video.submit`, `video.status` and `video.cancel`. A status node checks once;
the host owns bounded polling/backoff and live progress. Engine cancellation stops local scheduling;
it does not implicitly cancel remote tasks. Use the explicit remote cancel hook on saved handles.

## Canvas structural edits

`WorkflowEditRequest` (`vidliner.workflow-edit/v1`) is an atomic batch guarded by
`expected_digest`. It supports adding/removing nodes, configuring/moving nodes, connecting and
disconnecting edges, and binding/unbinding external inputs. Use `new_workflow_node()` to derive
stage, ports, capabilities, determinism and execution defaults from the existing catalogue.
The final document must pass the same operator, port, config and cycle validation as imported
workflows. Temporary incomplete states are allowed inside a batch, never in the written postimage.
Removing a node removes its incident edges; downstream required inputs must be rewired in that
same batch. Semantic edits clear the old recipe hash and stale binding hints; layout changes keep
execution semantics. Invalid or stale edits leave the original document untouched.

```python
from vidliner.domain.workflow_edit import MoveNode, WorkflowEditRequest
from vidliner.domain.workflow import Position
from vidliner.pipeline.workflow_edit import apply_workflow_edits, workflow_digest

request = WorkflowEditRequest(
    expected_digest=workflow_digest(document),
    edits=(MoveNode(node_id="plan", position=Position(x=160, y=80)),),
)
edited = apply_workflow_edits(document, request)
```

```sh
vidliner workflow digest workflow.json
vidliner workflow patch workflow.json edits.json --output edited.json
vidliner workflow validate edited.json
```

The host must atomically compare and replace its stored document using `expected_digest` to prevent
concurrent writers losing edits. The library computes/validates immutable postimages; it does not
claim a database transaction or multi-process file CAS. The offline editor currently edits layout
and config; a graphical node palette and connection gestures can consume these typed edit requests.

## Canvas execution host

`WorkflowExecutionRequest` (`vidliner.workflow-execution/v1`) names a job, seed and reviewed full
document digest. `build_workflow_engine()` revalidates the document, runtime bindings, backend
readiness, supplied artifact roles and source contexts, then returns the existing `OperationEngine`.
It ignores canvas binding hints and obtains actual backends from the host's runtime profile.

```python
from vidliner.pipeline.backends import CapabilityBroker
from vidliner.pipeline.workflow_execution import WorkflowExecutionRequest, build_workflow_engine
from vidliner.pipeline.workflow_edit import workflow_digest

request = WorkflowExecutionRequest(
    job_id="host-assigned-job", seed=42, expected_digest=workflow_digest(document),
    allow_external=True,  # set by the trusted host after its authorization and cost policy
)
engine = await build_workflow_engine(
    document, request, store=session.store, broker=CapabilityBroker(session.registry),
    state=durable_state, artifact_inputs=bound_inputs, sample_for=source_context_for_node,
    on_event=send_progress_to_ui,
)
report = await engine.execute()
# Host status: engine.report; local cancellation: engine.cancel().
```

`allow_external` is a host policy assertion, not browser authentication. Authenticate and authorize
the caller before constructing it. Assign and atomically claim `job_id` plus the reviewed digest
and bound-input identity before executing; duplicate dispatch must return the existing job. Persist
submission intent before provider calls and reconcile unknown outcomes without automatic resubmission.
The factory does not implement distributed exactly-once submission, webhooks, a database job claim
or recovery. Durable `RunState` and task storage belong to the host. Node cache/resume and retries
are disabled by this interface to avoid silently reusing stale status or repeating paid calls.

Canvas execution returns node evidence. Export-stage nodes are refused; training data still goes
through the recipe service's acceptance, production-backend and export policy. Source-context
injection reuses `SampleContext`; job-level inputs reuse `ArtifactSource` and the artifact executor.
No HTTP server, direct browser execution, native Sora/Vertex/DashScope adapter, result downloader or
graph-gesture UI is included in this slice; each can connect at these public boundaries.
