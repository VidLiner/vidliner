"""Canvas execution host interface over the existing engine, broker and artifact executor.

The host owns durable RunState, source contexts, cancellation and UI transport. This factory
preflights a reviewed draft and returns the ordinary OperationEngine; it never exports datasets.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

from pydantic import Field

from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.core.graph import ArtifactSource, OperationNode, SampleSource
from vidliner.domain.enums import StageName
from vidliner.domain.workflow import WorkflowDocument, WorkflowModel
from vidliner.operators.base import SampleContext
from vidliner.pipeline.backends import CapabilityBroker
from vidliner.pipeline.executor import ArtifactExecutor, seed_for_node
from vidliner.pipeline.workflow import graph_from_workflow
from vidliner.pipeline.workflow_edit import workflow_digest
from vidliner.runtime.engine import EngineOptions, OperationEngine, RunState
from vidliner.storage.workspace import ArtifactStore


class WorkflowExecutionRequest(WorkflowModel):
    """Identify a reviewed draft and trusted host's execution authorization."""

    schema_version: Literal["vidliner.workflow-execution/v1"] = "vidliner.workflow-execution/v1"
    expected_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    job_id: str = Field(min_length=1)
    seed: int = Field(default=0, ge=0)
    allow_external: bool = False


async def build_workflow_engine(
    document: WorkflowDocument,
    request: WorkflowExecutionRequest,
    *,
    store: ArtifactStore,
    broker: CapabilityBroker,
    state: RunState,
    artifact_inputs: Mapping[str, Any] | None = None,
    sample_for: Callable[[OperationNode], SampleContext | None] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    workers: int = 4,
) -> OperationEngine:
    """Revalidate ports, source bindings and runtime capabilities before any node executes.

    The returned engine's ``execute/cancel/report`` are the host's execution/status/cancel hooks.
    Supply a durable state adapter for production hosts. No implicit retry, cache or resume is used
    here: the reviewed intent must not resubmit paid generations after an unknown outcome.
    """
    if workers < 1:
        raise ValidationFailure("workers must be positive", code=ErrorCode.GRAPH_INVALID)
    if workflow_digest(document) != request.expected_digest:
        raise ValidationFailure(
            "execution draft differs from the reviewed digest", code=ErrorCode.GRAPH_INVALID
        )
    graph = graph_from_workflow(document)
    if any(node.stage == StageName.EXPORT for node in graph.nodes):
        raise ValidationFailure(
            "canvas execution cannot export training data; use the recipe acceptance/export service",
            code=ErrorCode.EXPORT_INVALID,
        )
    artifacts = dict(artifact_inputs or {})
    samples = {node.node_id: sample_for(node) if sample_for else None for node in graph.nodes}
    for node in graph.nodes:
        for binding in node.inputs.values():
            if isinstance(binding, ArtifactSource) and binding.role not in artifacts:
                raise ValidationFailure(
                    f"missing external artifact role {binding.role}", code=ErrorCode.PORT_UNBOUND
                )
            if isinstance(binding, SampleSource):
                sample = samples[node.node_id]
                if sample is None or binding.role not in sample.as_mapping():
                    raise ValidationFailure(
                        f"missing sample role {binding.role}", code=ErrorCode.PORT_UNBOUND
                    )
        if node.retry.attempts != 1:
            raise ValidationFailure(
                "canvas execution requires single-attempt nodes", code=ErrorCode.GRAPH_INVALID
            )
    resolution = broker.resolve(
        tuple(sorted({capability for node in graph.nodes for capability in node.needs}))
    )
    if not resolution.is_complete:
        raise ValidationFailure(
            "workflow has unbound runtime capabilities", code=ErrorCode.CAPABILITY_UNBOUND
        )
    for binding in resolution.bindings.values():
        backend = broker.registry.backend(binding.backend)
        probe = await backend.probe()
        if not probe.is_ready:
            raise ValidationFailure(
                f"workflow backend {binding.backend} is unavailable", code=ErrorCode.BACKEND_UNAVAILABLE
            )
        if (probe.external or bool(getattr(backend, "external", True))) and not request.allow_external:
            raise ValidationFailure(
                "external workflow execution requires explicit host authorization",
                code=ErrorCode.BACKEND_REQUEST_FAILED,
            )
    executor = ArtifactExecutor(
        job_id=request.job_id,
        store=store,
        broker=broker,
        sample_for=lambda node: samples[node.node_id],
        artifact_inputs=artifacts,
        on_event=(
            lambda kind, node, detail: on_event({"event": kind, "node_id": node.node_id, "detail": detail})
        )
        if on_event
        else None,
    )
    return OperationEngine(
        graph,
        executor,
        state,
        options=EngineOptions(workers=workers, use_cache=False, resume=False, fail_fast=True),
        seed_for_node=lambda node: seed_for_node(request.seed, node),
        on_event=on_event,
    )
