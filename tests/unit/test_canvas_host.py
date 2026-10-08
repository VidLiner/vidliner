"""The local host persists intent and graph edits without a second execution engine."""

import asyncio
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.unit.test_workflow_interfaces import FakeVideoBackend, document, video_broker
from vidliner.cli.main import app
from vidliner.core.errors import ValidationFailure
from vidliner.domain.workflow import WorkflowDocument
from vidliner.domain.workflow_edit import MoveNode, Position, WorkflowEditRequest
from vidliner.pipeline.canvas_host import CanvasHost
from vidliner.pipeline.workflow_edit import new_workflow_node, workflow_digest


@pytest.mark.asyncio
async def test_host_executes_once_persists_and_rejects_stale_edits(session, tmp_path):
    host = CanvasHost(session, document(), tmp_path / "canvas.db")
    request = {"job_id": "once", "expected_digest": host.draft()["digest"]}
    first = await host.execute(request)
    await host.runners["once"]
    assert host.store.job("once")["status"] == "succeeded"
    assert first["nodes"]["plan"]["status"] == "succeeded"
    assert (await host.execute(request))["status"] == "succeeded"
    patch = WorkflowEditRequest(
        expected_digest=request["expected_digest"],
        edits=(MoveNode(node_id="plan", position=Position(x=100)),),
    )
    host.edit(patch.model_dump(mode="json"))
    with pytest.raises(ValidationFailure, match="changed"):
        host.edit(patch.model_dump(mode="json"))
    with pytest.raises(ValidationFailure, match="another intent"):
        await host.execute({**request, "seed": 42})
    await host.close()


@pytest.mark.asyncio
async def test_interrupted_generation_never_resubmits_and_handle_survives(session, tmp_path):
    doc = WorkflowDocument(
        name="video",
        nodes=(new_workflow_node("video.submit", "submit", config={"request": {"prompt": "scene"}}),),
    )
    path = tmp_path / "canvas.db"
    host = CanvasHost(session, doc, path, allow_external=True, poll_interval_s=0.01, poll_timeout_s=0.01)
    host.broker = video_broker()
    FakeVideoBackend.submissions = 0
    request = {"job_id": "video", "expected_digest": workflow_digest(doc)}
    await host.execute(request)
    await host.runners["video"]
    assert host.store.job("video")["status"] == "waiting"
    assert FakeVideoBackend.submissions == 1
    await host.close()
    reopened = CanvasHost(session, doc, path, allow_external=True)
    reopened.broker = video_broker()
    assert (await reopened.execute(request))["status"] == "waiting"
    assert FakeVideoBackend.submissions == 1
    updated = await reopened.task_action("video", "submit", "status")
    assert updated["tasks"]["submit"]["status"] == "running"
    await reopened.close()


def test_live_verify_cli_uses_runway_task_lifecycle_with_http_fixtures(monkeypatch, tmp_path):
    """Exercise the real adapter via CLI while keeping this regression test unbilled."""
    httpx = pytest.importorskip("httpx")

    from tests.contract.test_http_video import mock_http

    calls = []

    def handle(request):
        calls.append((request.method, request.url.path))
        if request.method == "POST":
            return httpx.Response(200, json={"id": "fixture-task"})
        return httpx.Response(
            200, json={"status": "SUCCEEDED", "output": ["https://fixture.example/video.mp4"]}
        )

    mock_http(monkeypatch, handle)
    monkeypatch.setenv("RUNWAYML_API_SECRET", "fixture-only")
    root = Path(__file__).parents[2]
    command = [
        "video",
        "verify",
        str(root / "examples/video/request-runway.json"),
        "--runtime",
        str(root / "examples/video/runtime-runway.yaml"),
        "--workspace",
        str(tmp_path / "workspace"),
        "--job-id",
        "verify-once",
    ]
    runner = CliRunner()
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    assert "succeeded" in result.output and "fixture-only" not in result.output
    assert calls == [("POST", "/v1/text_to_video"), ("GET", "/v1/tasks/fixture-task")]
    assert runner.invoke(app, command).exit_code == 0
    assert sum(method == "POST" for method, _ in calls) == 1


@pytest.mark.asyncio
async def test_remote_cancel_is_deduplicated_and_completion_is_not_assumed(session, tmp_path):
    doc = WorkflowDocument(
        name="video",
        nodes=(new_workflow_node("video.submit", "submit", config={"request": {"prompt": "scene"}}),),
    )
    host = CanvasHost(
        session, doc, tmp_path / "canvas.db", allow_external=True, poll_interval_s=0.01, poll_timeout_s=0.1
    )
    host.broker = video_broker()
    adapter = host.broker.handle("generation.video_generation.v1").backend
    cancel_calls = []

    async def cancel(task, context):
        cancel_calls.append(task.task_id)
        return task.model_copy(update={"status": "running"})

    adapter.cancel = cancel
    await host.execute({"job_id": "cancel", "expected_digest": workflow_digest(doc)})
    for _ in range(100):
        if host.store.job("cancel")["tasks"]:
            break
        await asyncio.sleep(0.001)
    await host.cancel("cancel")
    await host.cancel("cancel")
    assert cancel_calls == ["provider-1"]
    assert host.store.job("cancel")["tasks"]["submit"]["status"] == "running"
    await host.close()
