# Video and canvas integration contracts

Library and CLI integration contracts also power a loopback web application. `workflow preview`
remains offline; `workflow serve` adds saved structural edits, execution and live video tasks,
using the same graph, scheduler and backend registry. Credentials remain in the host runtime.

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

`HttpVideoGenerationBackend` supports Runway text/image generation, fal's queue API, and the
[Bifrost](https://github.com/maximhq/bifrost) OpenAI-compatible video gateway. The
[example runtime](../examples/video/runtime-ai.yaml) includes Runway Gen-4.5, Kling, Wan and Veo
configurations. Provider endpoints and model parameters were checked against official docs;
integration tests use HTTP fixtures. They are not a claim of live generation with every model,
account entitlement, model quality or production acceptance.

| Route | Configuration | Official protocol/schema |
| --- | --- | --- |
| Runway native API | `provider: runway`, model ID, enabled modes | [Task lifecycle](https://docs.dev.runwayml.com/ai-context.md), [OpenAPI](https://docs.dev.runwayml.com/openapi.json) |
| Bifrost video gateway | `provider: bifrost`, gateway origin, model ID, enabled modes | [`POST /v1/videos`](https://github.com/maximhq/bifrost), `GET/DELETE /v1/videos/{id}` |
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

Bifrost is an API gateway, not a local model runtime. Configure it with the
`runtime-bifrost.yaml` example and keep its gateway key in `BIFROST_API_KEY`; the gateway then
owns provider routing, fallbacks and provider credentials. VidLiner only sends the provider-neutral
`/v1/videos` request and persists the returned Bifrost video ID for later retrieval or cancellation.

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
and config. The hosted editor consumes these transactions for its node palette, port dragging,
keyboard connections and edge removal; SQLite implements draft compare-and-replace.

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
Native Sora/Vertex/DashScope adapters and result downloading remain extension points.

## Local web application

```sh
vidliner workflow serve examples/video/workflow-canvas.json --workspace ./canvas-workspace
```

Open `http://127.0.0.1:8767`. `Execute` runs the initial provider-free planning node. Expand
`Add an operator`, select a registered operator and supply its config. Drag output circles onto input
circles, or focus an output and press Enter, then focus an input and press Enter. Click a connection
and use `Remove connection`; selected nodes support `Remove node`. Unwired required inputs are
saved as artifact-role placeholders, so the graph remains structurally valid but execution fails
preflight until real inputs are supplied or connected. The local host does not inject dataset sample
contexts or external artifacts; use an embedding host for these inputs. Editing config/layout saves
automatically; `Save JSON` exports a portable copy. The source JSON is never overwritten.

The service binds only `127.0.0.1`. API requests require its per-process token and matching Host/Origin;
tokens stay in the page and credentials are never exposed. No CORS access is granted. External
execution is off by default and can only be enabled by the CLI host. It is a local, single-process
application, not a multi-user deployment or public authentication service. Use one server per
workspace/draft state. The one-MiB JSON API cap bounds commands. Private SQLite state is saved under
the workspace cache directory with owner-only permissions; preserve it to retain handles and deduplication.

Execution IDs are uniquely claimed before running. Repeating the same intent returns its saved job;
reusing an ID with changed draft/runtime/seed is refused. Nodes and remote task handles are persisted.
Restarting the server marks incomplete jobs interrupted and never resubmits them. The UI shows the
latest saved job; `Refresh task` queries a saved handle, and `Cancel task` requests remote cancellation.
`Cancel job` stops local scheduling and requests cancellation of its saved remote handles. Cancellation
acknowledgement does not prove the provider stopped. A crash/timeout before receiving a handle may leave
an unknown remote submission; never regenerate automatically to reconcile that uncertainty.

The host polls pending tasks at five-second intervals for at most ten minutes. Multiple lifecycle
nodes sharing a handle are polled once per task. After timeout, the job waits for explicit refresh/cancel.
Generation results appear as HTML video players using provider URLs, without forwarding credentials
or downloading media. Signed URLs may expire; results remain unverified training media.

## Runway live verification

Install `vidliner[http]`, provide `RUNWAYML_API_SECRET` to the host process, or configure the existing
file credential source in a private copy of [runtime-runway.yaml](../examples/video/runtime-runway.yaml).
The example uses a five-second Gen-4.5 text-to-video request. Generation is a real provider operation
and uses account credits; loading a profile or inspecting the canvas does not submit it.

```sh
vidliner workflow serve examples/video/workflow-runway.json \
  --workspace ./runway-workspace --runtime examples/video/runtime-runway.yaml --allow-external

# Or submit one live verification with persisted evidence and bounded polling:
vidliner video verify examples/video/request-runway.json \
  --workspace ./runway-verification --runtime examples/video/runtime-runway.yaml \
  --job-id runway-smoke-001 --timeout 600
```

`video verify` uses the same host, request and adapter as the web canvas. A repeated `--job-id`
only refreshes its saved task, never resubmits it. Exit 0 means provider success; exit 2 means
validation failure, generation failure, interruption or pending state. Default output gives an
evidence database path; `--json` includes private task handles and result URLs. The command proves
provider/task connectivity and completion, not visual quality or training acceptance. Automated
tests use unbilled HTTP fixtures. A live success may only be claimed after an actual terminal result.
