"""Atomic graph edits and canvas-host execution share the existing runtime."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from vidliner.backends.base import LocalBackend
from vidliner.capabilities.names import CAP_VIDEO_GENERATION
from vidliner.cli.main import app
from vidliner.core.errors import ValidationFailure
from vidliner.domain.enums import NodeStatus
from vidliner.domain.video_generation import VideoTask
from vidliner.domain.workflow import ArtifactInput, WorkflowDocument, WorkflowEdge
from vidliner.domain.workflow_edit import (
    AddNode,
    BindInput,
    ConfigureNode,
    Connect,
    Disconnect,
    MoveNode,
    Position,
    RemoveNode,
    WorkflowEditRequest,
)
from vidliner.operators.values import decode_value, encode_value
from vidliner.pipeline.backends import CapabilityBroker
from vidliner.pipeline.service import Session
from vidliner.pipeline.workflow_edit import apply_workflow_edits, new_workflow_node, workflow_digest
from vidliner.pipeline.workflow_execution import WorkflowExecutionRequest, build_workflow_engine
from vidliner.runtime.profile import BackendSpec, RuntimeProfile
from vidliner.runtime.registry import BackendRegistry
from vidliner.storage.state import InMemoryRunState


def planner():
    return new_workflow_node(
        "video.plan_variants",
        "plan",
        config={
            "spec": {
                "id": "test",
                "seed": 3,
                "operators": [{"name": "format.reframe", "axis": "format", "domains": {"aspect": ["1:1"]}}],
            }
        },
    )


def document():
    return WorkflowDocument(name="plan", nodes=(planner(),), recipe_hash="a" * 64)


def patch(doc, *edits):
    return WorkflowEditRequest(expected_digest=workflow_digest(doc), edits=edits)


def test_semantic_edits_clear_recipe_identity_but_layout_preserves_it():
    doc = document()
    moved = apply_workflow_edits(doc, patch(doc, MoveNode(node_id="plan", position=Position(x=10, y=20))))
    assert moved.recipe_hash == doc.recipe_hash
    config = doc.nodes[0].model_dump(mode="json")["config"]
    config["spec"]["seed"] = 4
    edited = apply_workflow_edits(doc, patch(doc, ConfigureNode(node_id="plan", config=config)))
    assert edited.recipe_hash is None and doc.nodes[0].config["spec"]["seed"] == 3


def test_atomic_add_connect_disconnect_rebind_and_remove():
    doc = document()
    submit = new_workflow_node("video.submit", "submit", config={"request": {"prompt": "a scene"}})
    status = new_workflow_node("video.status", "status")
    edge = WorkflowEdge(source="submit", source_port="task", target="status", target_port="task")
    edited = apply_workflow_edits(
        doc, patch(doc, AddNode(node=submit), AddNode(node=status), Connect(edge=edge))
    )
    assert len(edited.nodes) == 3 and len(doc.nodes) == 1
    rebound = apply_workflow_edits(
        edited,
        patch(
            edited,
            Disconnect(edge=edge),
            BindInput(node_id="status", port="task", binding=ArtifactInput(role="task")),
            RemoveNode(node_id="submit"),
        ),
    )
    assert len(rebound.nodes) == 2 and not rebound.edges


def test_invalid_edit_batch_and_stale_digest_leave_original_untouched():
    doc = document()
    with pytest.raises(ValidationFailure):
        apply_workflow_edits(doc, patch(doc, AddNode(node=new_workflow_node("video.status", "missing-task"))))
    assert len(doc.nodes) == 1
    with pytest.raises(ValidationFailure, match="changed"):
        apply_workflow_edits(
            doc,
            WorkflowEditRequest(
                expected_digest="b" * 64, edits=(MoveNode(node_id="plan", position=Position()),)
            ),
        )


@pytest.mark.asyncio
async def test_planner_runs_through_canvas_host_and_restores_typed_outputs(session: Session):
    doc = document()
    request = WorkflowExecutionRequest(expected_digest=workflow_digest(doc), job_id="canvas")
    state = InMemoryRunState()
    engine = await build_workflow_engine(
        doc, request, store=session.store, broker=CapabilityBroker(session.registry), state=state
    )
    report = await engine.execute()
    assert report.succeeded() and report.count(NodeStatus.SUCCEEDED) == 1
    result = report.nodes["plan"]
    restored = decode_value(result.payload["values"]["variants"])
    assert len(restored) == 2 and restored[0].ops == ()
    assert encode_value(restored)["tag"] == "video-variants"


class FakeVideoBackend(LocalBackend):
    backend_id = "fake-video"
    capabilities = (CAP_VIDEO_GENERATION,)
    declared_capabilities = capabilities
    blocking = False
    external = True
    safe_to_retry = False
    submissions = 0

    async def submit(self, request, context):
        type(self).submissions += 1
        return VideoTask(backend_id=self.backend_id, model_id="fake", task_id="provider-1")

    async def status(self, task, context):
        return task.model_copy(update={"status": "running"})


def video_broker():
    spec = BackendSpec(use="tests.unit.test_workflow_interfaces:FakeVideoBackend")
    return CapabilityBroker(
        BackendRegistry(RuntimeProfile(backends={"video": spec}, bindings={CAP_VIDEO_GENERATION: "video"}))
    )


@pytest.mark.asyncio
async def test_external_canvas_execution_requires_host_authorization(session: Session):
    submit = new_workflow_node("video.submit", "submit", config={"request": {"prompt": "scene"}})
    status = new_workflow_node("video.status", "status")
    doc = WorkflowDocument(
        name="video",
        nodes=(submit, status),
        edges=(WorkflowEdge(source="submit", source_port="task", target="status", target_port="task"),),
    )
    request = WorkflowExecutionRequest(expected_digest=workflow_digest(doc), job_id="video")
    FakeVideoBackend.submissions = 0
    with pytest.raises(ValidationFailure, match="authorization"):
        await build_workflow_engine(
            doc, request, store=session.store, broker=video_broker(), state=InMemoryRunState()
        )
    assert FakeVideoBackend.submissions == 0
    events = []
    engine = await build_workflow_engine(
        doc,
        request.model_copy(update={"allow_external": True}),
        store=session.store,
        broker=video_broker(),
        state=InMemoryRunState(),
        on_event=events.append,
    )
    report = await engine.execute()
    assert report.succeeded(), report
    assert FakeVideoBackend.submissions == 1
    task = decode_value(report.nodes["status"].payload["values"]["task"])
    assert task.status == "running" and not task.verified
    assert any(event["event"] == "operator.video.submitted" for event in events)


@pytest.mark.asyncio
async def test_execution_rejects_stale_draft_invalid_workers_and_missing_capability(session: Session):
    doc = document()
    request = WorkflowExecutionRequest(expected_digest=workflow_digest(doc), job_id="preflight")
    broker = CapabilityBroker(session.registry)
    with pytest.raises(ValidationFailure, match="reviewed digest"):
        await build_workflow_engine(
            doc,
            request.model_copy(update={"expected_digest": "b" * 64}),
            store=session.store,
            broker=broker,
            state=InMemoryRunState(),
        )
    with pytest.raises(ValidationFailure, match="workers"):
        await build_workflow_engine(
            doc,
            request,
            store=session.store,
            broker=broker,
            state=InMemoryRunState(),
            workers=0,
        )
    video = WorkflowDocument(
        name="video",
        nodes=(new_workflow_node("video.submit", "submit", config={"request": {"prompt": "x"}}),),
    )
    with pytest.raises(ValidationFailure, match="unbound"):
        await build_workflow_engine(
            video,
            request.model_copy(update={"expected_digest": workflow_digest(video), "allow_external": True}),
            store=session.store,
            broker=broker,
            state=InMemoryRunState(),
        )


def test_canvas_cycle_is_rejected_atomically():
    doc = document()
    first = new_workflow_node("video.status", "first")
    second = new_workflow_node("video.status", "second")
    with pytest.raises(ValidationFailure):
        apply_workflow_edits(
            doc,
            patch(
                doc,
                AddNode(node=first),
                AddNode(node=second),
                Connect(
                    edge=WorkflowEdge(source="first", source_port="task", target="second", target_port="task")
                ),
                Connect(
                    edge=WorkflowEdge(source="second", source_port="task", target="first", target_port="task")
                ),
            ),
        )
    assert len(doc.nodes) == 1 and not doc.edges


@pytest.mark.asyncio
async def test_task_artifact_role_is_injected_and_missing_role_fails_preflight(session: Session):
    status = new_workflow_node("video.status", "status").model_copy(
        update={"inputs": {"task": ArtifactInput(role="existing-task")}}
    )
    doc = WorkflowDocument(name="status", nodes=(status,))
    request = WorkflowExecutionRequest(
        expected_digest=workflow_digest(doc), job_id="poll", allow_external=True
    )
    with pytest.raises(ValidationFailure, match="missing external"):
        await build_workflow_engine(
            doc, request, store=session.store, broker=video_broker(), state=InMemoryRunState()
        )
    task = VideoTask(backend_id="fake-video", model_id="fake", task_id="provider-1")
    engine = await build_workflow_engine(
        doc,
        request,
        store=session.store,
        broker=video_broker(),
        state=InMemoryRunState(),
        artifact_inputs={"existing-task": task},
    )
    report = await engine.execute()
    assert report.succeeded()
    assert decode_value(report.nodes["status"].payload["values"]["task"]).status == "running"


def test_patch_cli_and_schemas(tmp_path: Path):
    runner = CliRunner()
    doc = document()
    source = tmp_path / "workflow.json"
    source.write_text(doc.model_dump_json())
    change = tmp_path / "edits.json"
    change.write_text(patch(doc, MoveNode(node_id="plan", position=Position(x=20))).model_dump_json())
    out = tmp_path / "edited.json"
    result = runner.invoke(app, ["workflow", "patch", str(source), str(change), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert WorkflowDocument.model_validate_json(out.read_text()).nodes[0].position.x == 20
    for command in ("edit-schema", "execution-schema"):
        assert runner.invoke(app, ["workflow", command]).exit_code == 0
    for command in ("request-schema", "task-schema"):
        assert runner.invoke(app, ["video", command]).exit_code == 0
