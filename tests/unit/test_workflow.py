"""Workflow exchange must retain execution semantics and reject misleading contracts."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from tests.conftest import make_recipe
from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.core.graph import ArtifactSource, ConstantValue, OperationGraph, RetryPolicy
from vidliner.domain.recipe import Recipe
from vidliner.domain.workflow import WorkflowDocument
from vidliner.pipeline.assemble import assemble_graph
from vidliner.pipeline.workflow import graph_from_workflow, operator_catalogue, workflow_from_graph
from vidliner.reports.workflow import render_workflow


@pytest.fixture
def graph() -> OperationGraph:
    recipe = Recipe.model_validate(make_recipe("data/source", candidates=1))
    return assemble_graph(recipe, ("sample-1",)).graph


def test_compiled_recipe_roundtrip_preserves_every_execution_field(graph: OperationGraph) -> None:
    # Exercise fields that ordinary recipes leave at their defaults.
    first = replace(
        graph.nodes[0],
        retry=RetryPolicy(3, 0.7, 1.5, 0.2, (ErrorCode.NODE_TIMEOUT,)),
        timeout_s=45,
        max_parallelism=2,
    )
    original = OperationGraph((first, *graph.nodes[1:]), job_lineage=("job", "fixture"))
    document = workflow_from_graph(original, name="roundtrip")
    restored = graph_from_workflow(WorkflowDocument.model_validate_json(document.model_dump_json()))
    assert restored.index() == original.index()
    assert restored.job_lineage == original.job_lineage
    assert workflow_from_graph(original, name="roundtrip") == document


def test_external_inputs_retain_artifact_roles_and_null_constants(graph: OperationGraph) -> None:
    first = graph.nodes[0]
    inputs = dict(first.inputs)
    ports = list(inputs)
    assert len(ports) >= 2
    inputs[ports[0]] = ArtifactSource("reference")
    inputs[ports[1]] = ConstantValue({"payload": None, "flags": [False, 0, ""]})
    original = OperationGraph((replace(first, inputs=inputs), *graph.nodes[1:]))
    doc = workflow_from_graph(original, name="external")
    restored = graph_from_workflow(WorkflowDocument.model_validate_json(doc.model_dump_json()))
    assert restored.index() == original.index()


def test_canvas_positions_do_not_change_the_graph(graph: OperationGraph) -> None:
    document = workflow_from_graph(graph, name="layout").model_dump(mode="json")
    for node in document["nodes"]:
        node["position"] = {"x": -100, "y": 250}
    assert graph_from_workflow(WorkflowDocument.model_validate(document)).index() == graph.index()


@pytest.mark.parametrize(
    "mutation", ["duplicate", "unknown_source", "unknown_port", "double_binding", "version"]
)
def test_schema_rejects_invalid_wiring(graph: OperationGraph, mutation: str) -> None:
    doc = workflow_from_graph(graph, name="invalid").model_dump(mode="json")
    if mutation == "duplicate":
        doc["nodes"].append(doc["nodes"][0])
    elif mutation == "unknown_source":
        doc["edges"][0]["source"] = "missing"
    elif mutation == "unknown_port":
        doc["edges"][0]["source_port"] = "missing"
    elif mutation == "double_binding":
        doc["edges"].append(doc["edges"][0])
    else:
        doc["schema_version"] = "future/v99"
    with pytest.raises(ValidationError):
        WorkflowDocument.model_validate(doc)


def test_cycle_is_rejected_before_catalogue_port_validation(graph: OperationGraph) -> None:
    doc = workflow_from_graph(graph, name="cycle").model_dump(mode="json")
    root = doc["nodes"][0]
    leaf = doc["nodes"][-1]
    port = next(iter(root["inputs"]))
    root["inputs"].pop(port)
    doc["edges"].append(
        {
            "source": leaf["id"],
            "source_port": next(iter(leaf["outputs"])),
            "target": root["id"],
            "target_port": port,
        }
    )
    with pytest.raises(ValidationFailure) as failure:
        graph_from_workflow(WorkflowDocument.model_validate(doc))
    assert failure.value.code is ErrorCode.GRAPH_CYCLE


@pytest.mark.parametrize(
    "field,value",
    [
        ("operator", "unknown.operator"),
        ("operator_version", "99.0.0"),
        ("outputs", {"fabricated": "any"}),
        ("needs", ["invented.capability"]),
        ("stage", "export"),
        ("config", {"unknown_option": True}),
    ],
)
def test_registered_operator_contract_is_authoritative(
    graph: OperationGraph, field: str, value: object
) -> None:
    doc = workflow_from_graph(graph, name="contract").model_dump(mode="json")
    if field == "outputs":
        doc["nodes"][0][field].update(value)
    else:
        doc["nodes"][0][field] = value
    with pytest.raises(ValidationFailure):
        graph_from_workflow(WorkflowDocument.model_validate(doc))


def test_missing_and_undeclared_input_ports_are_rejected(graph: OperationGraph) -> None:
    doc = workflow_from_graph(graph, name="ports").model_dump(mode="json")
    doc["nodes"][0]["inputs"] = {}
    with pytest.raises(ValidationFailure):
        graph_from_workflow(WorkflowDocument.model_validate(doc))
    doc["nodes"][0]["inputs"] = {"invented": {"kind": "constant", "value": None}}
    with pytest.raises(ValidationFailure):
        graph_from_workflow(WorkflowDocument.model_validate(doc))


def test_port_type_mismatch_is_rejected(graph: OperationGraph) -> None:
    doc = workflow_from_graph(graph, name="type").model_dump(mode="json")
    edge = doc["edges"][0]
    # Connect a real declared output with a different type to the same required port.
    source = next(n for n in doc["nodes"] if n["id"] == edge["source"])
    wrong = next(
        (n, port)
        for n in doc["nodes"]
        for port, typ in n["outputs"].items()
        if typ not in source["outputs"].values() and n["id"] != edge["target"]
    )
    edge["source"], edge["source_port"] = wrong[0]["id"], wrong[1]
    with pytest.raises(ValidationFailure):
        graph_from_workflow(WorkflowDocument.model_validate(doc))


def test_secrets_and_nonfinite_positions_are_rejected(graph: OperationGraph) -> None:
    doc = workflow_from_graph(graph, name="private").model_dump(mode="json")
    doc["nodes"][0]["config"]["api_key"] = "must-not-leave-runtime"
    with pytest.raises(ValidationError, match="credential-like"):
        WorkflowDocument.model_validate(doc)
    doc["nodes"][0]["config"].pop("api_key")
    doc["nodes"][0]["position"]["x"] = float("inf")
    with pytest.raises(ValidationError):
        WorkflowDocument.model_validate(doc)


def test_preview_keeps_document_strings_inert(graph: OperationGraph) -> None:
    malicious = '</script><img src=x onerror="alert(1)">'
    document = workflow_from_graph(graph, name=malicious)
    html = render_workflow(document)
    assert malicious not in html
    payload = html.split('<script type="application/json" id="workflow">')[1].split("</script>")[0]
    assert json.loads(payload)["name"] == malicious
    assert "textContent" in html
    assert "https://" not in html


def test_preview_embeds_complete_localization_runtime(graph: OperationGraph) -> None:
    html = render_workflow(workflow_from_graph(graph, name="locale"))
    assert '<script type="application/json" id="locales">' in html
    assert all(f'"{locale}"' in html for locale in ("en-US", "zh-CN", "ja-JP", "ko-KR", "es-ES"))
    assert "VidLinerI18n" in html
    assert 'id="localeSelect"' in html
    assert all(f'value="{locale}"' in html for locale in ("en-US", "zh-CN", "ja-JP", "ko-KR", "es-ES"))


def test_catalogue_includes_video_and_verification_schemas() -> None:
    catalogue = {spec["operator"]: spec for spec in operator_catalogue()}
    assert "video.plan_variants" in catalogue
    assert "verify.redetect" in catalogue
    assert catalogue["verify.redetect"]["inputs"]
    schema = catalogue["video.plan_variants"]["config_schema"]
    assert isinstance(schema, dict)
    assert schema["type"] == "object"
