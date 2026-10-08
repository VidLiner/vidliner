"""Lossless interchange between a canvas document and the existing operation DAG."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.core.graph import (
    ArtifactSource,
    ConstantValue,
    NodeOutput,
    OperationGraph,
    OperationNode,
    PortBinding,
    RetryPolicy,
    SampleSource,
)
from vidliner.domain.workflow import (
    ArtifactInput,
    ConstantInput,
    Position,
    SampleInput,
    WorkflowDocument,
    WorkflowEdge,
    WorkflowNode,
    WorkflowRetry,
)
from vidliner.operators.registry import OperatorRegistry, default_registry


def workflow_from_graph(
    graph: OperationGraph,
    *,
    name: str,
    recipe_hash: str | None = None,
    capability_bindings: dict[str, str] | None = None,
    unmet_capabilities: tuple[str, ...] = (),
) -> WorkflowDocument:
    """Export a validated DAG with deterministic dependency-column layout."""
    graph.validate()
    depths: dict[str, int] = {}
    lanes: dict[int, int] = {}
    nodes: list[WorkflowNode] = []
    edges: list[WorkflowEdge] = []
    for node in graph.topological_order():
        depth = max((depths[dep] + 1 for dep in node.dependency_ids), default=0)
        lane = lanes.get(depth, 0)
        depths[node.node_id] = depth
        lanes[depth] = lane + 1
        inputs: dict[str, SampleInput | ArtifactInput | ConstantInput] = {}
        for port, binding in node.inputs.items():
            if isinstance(binding, NodeOutput):
                edges.append(
                    WorkflowEdge(
                        source=binding.node_id,
                        source_port=binding.port,
                        target=node.node_id,
                        target_port=port,
                    )
                )
            elif isinstance(binding, SampleSource):
                inputs[port] = SampleInput(role=binding.role)
            elif isinstance(binding, ArtifactSource):
                inputs[port] = ArtifactInput(role=binding.role)
            else:
                inputs[port] = ConstantInput(value=binding.value)
        nodes.append(
            WorkflowNode(
                id=node.node_id,
                operator=node.operator,
                operator_version=node.operator_version,
                stage=node.stage,
                name=node.name,
                description=node.description,
                position=Position(x=depth * 320, y=lane * 144),
                inputs=inputs,
                outputs=node.outputs,
                config=node.config,
                needs=node.needs,
                determinism=node.determinism,
                cacheable=node.cacheable,
                retry=WorkflowRetry(
                    attempts=node.retry.attempts,
                    backoff_s=node.retry.backoff_s,
                    multiplier=node.retry.multiplier,
                    jitter=node.retry.jitter,
                    retry_on=tuple(code.value for code in node.retry.retry_on),
                ),
                timeout_s=node.timeout_s,
                max_parallelism=node.max_parallelism,
                lineage=node.lineage,
                selects=node.selects,
            )
        )
    return WorkflowDocument(
        name=name,
        job_lineage=graph.job_lineage,
        nodes=tuple(nodes),
        edges=tuple(edges),
        recipe_hash=recipe_hash,
        capability_bindings=capability_bindings or {},
        unmet_capabilities=unmet_capabilities,
    )


def graph_from_workflow(
    document: WorkflowDocument,
    *,
    registry: OperatorRegistry | None = None,
) -> OperationGraph:
    """Rebuild and validate the same DAG; never import a backend or execute a node.

    Registered contracts are authoritative for outputs, stages and capability requirements.
    Documents cannot relabel a generator as a deterministic/cacheable local operator.
    """
    catalogue = registry if registry is not None else default_registry()
    wired: dict[str, dict[str, PortBinding]] = {node.id: {} for node in document.nodes}
    for edge in document.edges:
        wired[edge.target][edge.target_port] = NodeOutput(edge.source, edge.source_port)
    nodes: list[OperationNode] = []
    try:
        for node in document.nodes:
            spec = catalogue.describe(node.operator, node.operator_version)
            outputs = {port: output.port_type for port, output in spec.outputs.items()}
            if (
                node.outputs != outputs
                or node.stage != spec.stage
                or node.needs != spec.capabilities
                or node.determinism != spec.determinism
                or (node.cacheable and not spec.cacheable)
            ):
                raise ValidationFailure(
                    f"node {node.id} disagrees with the registered contract for {spec.label}",
                    code=ErrorCode.GRAPH_INVALID,
                )
            inputs = wired[node.id]
            for port, binding in node.inputs.items():
                if isinstance(binding, SampleInput):
                    inputs[port] = SampleSource(binding.role)
                elif isinstance(binding, ArtifactInput):
                    inputs[port] = ArtifactSource(binding.role)
                else:
                    inputs[port] = ConstantValue(binding.value)
            unknown = inputs.keys() - spec.inputs.keys()
            if unknown:
                raise ValidationFailure(
                    f"node {node.id} has undeclared input ports: {', '.join(sorted(unknown))}",
                    code=ErrorCode.PORT_UNBOUND,
                )
            selections = dict(node.selects)
            if len(selections) != len(node.selects) or selections.keys() - inputs.keys():
                raise ValidationFailure(
                    f"node {node.id} has invalid selections", code=ErrorCode.GRAPH_INVALID
                )
            nodes.append(
                OperationNode(
                    node_id=node.id,
                    operator=node.operator,
                    operator_version=node.operator_version,
                    stage=node.stage,
                    inputs=inputs,
                    outputs=node.outputs,
                    config=node.config,
                    needs=node.needs,
                    name=node.name,
                    determinism=node.determinism,
                    cacheable=node.cacheable,
                    retry=RetryPolicy(
                        attempts=node.retry.attempts,
                        backoff_s=node.retry.backoff_s,
                        multiplier=node.retry.multiplier,
                        jitter=node.retry.jitter,
                        retry_on=tuple(ErrorCode(code) for code in node.retry.retry_on),
                    ),
                    timeout_s=node.timeout_s,
                    max_parallelism=node.max_parallelism,
                    lineage=node.lineage,
                    selects=node.selects,
                    description=node.description,
                )
            )
        graph = OperationGraph(tuple(nodes), job_lineage=document.job_lineage)
        graph.validate()
        graph.validate_against_operators(catalogue.describe)
    except (ValueError, ValidationError) as exc:
        raise ValidationFailure(f"invalid workflow: {exc}", code=ErrorCode.GRAPH_INVALID) from exc
    return graph


def load_workflow(path: Path) -> WorkflowDocument:
    """Read the versioned JSON boundary with consistent CLI errors."""
    try:
        document = WorkflowDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValidationFailure(f"cannot load workflow {path}: {exc}", code=ErrorCode.GRAPH_INVALID) from exc
    graph_from_workflow(document)
    return document


def operator_catalogue(registry: OperatorRegistry | None = None) -> list[dict[str, object]]:
    """Expose typed ports and config schemas for a node palette, without backend imports."""
    catalogue = registry if registry is not None else default_registry()
    return [
        {
            "operator": spec.name,
            "version": spec.version,
            "stage": spec.stage.value,
            "summary": spec.summary,
            "inputs": {
                name: {
                    "type": port.port_type.value,
                    "required": port.required,
                    "description": port.description,
                }
                for name, port in spec.inputs.items()
            },
            "outputs": {
                name: {"type": port.port_type.value, "description": port.description}
                for name, port in spec.outputs.items()
            },
            "config_schema": spec.config_model.model_json_schema(),
            "capabilities": list(spec.capabilities),
            "determinism": spec.determinism.value,
            "cacheable": spec.cacheable,
            "max_parallelism": spec.max_parallelism,
        }
        for spec in catalogue.specs()
    ]
