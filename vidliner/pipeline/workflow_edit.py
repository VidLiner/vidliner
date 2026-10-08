"""Immutable canvas transactions. Registered operator contracts remain authoritative."""

from pydantic import JsonValue

from vidliner.core.canonical import digest_json
from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.domain.workflow import WorkflowDocument, WorkflowNode, WorkflowRetry
from vidliner.domain.workflow_edit import WorkflowEditRequest
from vidliner.operators.registry import default_registry
from vidliner.pipeline.workflow import graph_from_workflow


def workflow_digest(document: WorkflowDocument) -> str:
    """Identity of the full reviewed draft, including presentation and execution changes."""
    return digest_json(document.model_dump(mode="json"))


def new_workflow_node(
    operator: str, node_id: str, *, config: dict[str, JsonValue] | None = None
) -> WorkflowNode:
    """Build a draft node from authoritative catalogue fields; wire it in an edit transaction."""
    spec = default_registry().describe(operator)
    validated = spec.config_model.model_validate(config or {})
    return WorkflowNode(
        id=node_id,
        operator=spec.name,
        operator_version=spec.version,
        stage=spec.stage,
        config=validated.model_dump(mode="json"),
        outputs={key: value.port_type for key, value in spec.outputs.items()},
        needs=spec.capabilities,
        determinism=spec.determinism,
        cacheable=spec.cacheable,
        retry=WorkflowRetry(attempts=spec.retry_attempts),
        timeout_s=spec.timeout_s,
        max_parallelism=spec.max_parallelism,
        description=spec.summary,
    )


def apply_workflow_edits(document: WorkflowDocument, request: WorkflowEditRequest) -> WorkflowDocument:
    """Validate the entire postimage once; invalid batches leave the original untouched."""
    if workflow_digest(document) != request.expected_digest:
        raise ValidationFailure(
            "workflow changed since the edit request was prepared", code=ErrorCode.GRAPH_INVALID
        )
    payload = document.model_dump(mode="json")
    nodes = {node["id"]: node for node in payload["nodes"]}
    edges = payload["edges"]
    semantic_change = False
    for edit in request.edits:
        data = edit.model_dump(mode="json")
        op = edit.op
        if op == "add_node":
            node = data["node"]
            if node["id"] in nodes:
                raise ValidationFailure("node already exists", code=ErrorCode.GRAPH_INVALID)
            nodes[node["id"]] = node
        elif op in {"connect", "disconnect"}:
            edge = data["edge"]
            if op == "connect":
                edges.append(edge)
            elif edge not in edges:
                raise ValidationFailure("cannot disconnect an absent edge", code=ErrorCode.GRAPH_INVALID)
            else:
                edges.remove(edge)
        else:
            node_id = data["node_id"]
            if node_id not in nodes:
                raise ValidationFailure("edit references an unknown node", code=ErrorCode.GRAPH_INVALID)
            node = nodes[node_id]
            if op == "remove_node":
                del nodes[node_id]
                edges = [edge for edge in edges if edge["source"] != node_id and edge["target"] != node_id]
            elif op == "configure_node":
                node["config"] = data["config"]
            elif op == "move_node":
                node["position"] = data["position"]
            elif op == "bind_input":
                node["inputs"][data["port"]] = data["binding"]
            elif op == "unbind_input":
                if data["port"] not in node["inputs"]:
                    raise ValidationFailure("cannot unbind an absent input", code=ErrorCode.GRAPH_INVALID)
                del node["inputs"][data["port"]]
                node["selects"] = [selection for selection in node["selects"] if selection[0] != data["port"]]
        semantic_change |= op != "move_node"
    payload["nodes"] = list(nodes.values())
    payload["edges"] = edges
    if semantic_change:
        payload["recipe_hash"] = None
        payload["capability_bindings"] = {}
        payload["unmet_capabilities"] = ()
    try:
        edited = WorkflowDocument.model_validate(payload)
    except ValueError as exc:
        raise ValidationFailure(f"invalid canvas transaction: {exc}", code=ErrorCode.GRAPH_INVALID) from exc
    graph_from_workflow(edited)
    return edited
