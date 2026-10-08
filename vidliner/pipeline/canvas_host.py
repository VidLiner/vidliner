"""Local UI host over the existing workflow executor and video lifecycle capability."""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any

from vidliner.capabilities.backend import PipelineContext
from vidliner.capabilities.names import CAP_VIDEO_GENERATION
from vidliner.core.canonical import digest_json
from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.domain.video_generation import VideoTask
from vidliner.domain.workflow import ArtifactInput, WorkflowDocument
from vidliner.domain.workflow_edit import WorkflowEditRequest
from vidliner.operators.values import decode_value
from vidliner.pipeline.backends import CapabilityBroker
from vidliner.pipeline.service import Session
from vidliner.pipeline.workflow_edit import new_workflow_node, workflow_digest
from vidliner.pipeline.workflow_execution import WorkflowExecutionRequest, build_workflow_engine
from vidliner.runtime.engine import OperationEngine
from vidliner.storage.canvas import CanvasStore
from vidliner.storage.state import InMemoryRunState


class _RunState(InMemoryRunState):
    def __init__(self, host: CanvasHost, snapshot: dict[str, Any]) -> None:
        super().__init__()
        self.host, self.snapshot = host, snapshot

    def record_node(self, result, node, *, seed, config_hash) -> None:
        """Persist completed graph evidence and provider handles before UI polling."""
        super().record_node(result, node, seed=seed, config_hash=config_hash)
        self.snapshot["nodes"][node.node_id] = result.model_dump(mode="json")
        for value in result.payload.get("values", {}).values():
            if isinstance(value, dict) and value.get("tag") == "video-task":
                task = decode_value(value)
                self.snapshot["tasks"][node.node_id] = task.model_dump(mode="json")
        self.host.store.save_job(self.snapshot)


class CanvasHost:
    """One event-loop owner for saved drafts, bounded jobs and persisted remote handles.

    A server restart never resubmits interrupted intents. Saved handles can be queried or cancelled
    explicitly. The HTTP binding supplies authentication; external authorization is constructor-only.
    """

    def __init__(
        self,
        session: Session,
        document: WorkflowDocument,
        state_path: Path,
        *,
        allow_external: bool = False,
        poll_interval_s: float = 5,
        poll_timeout_s: float = 600,
    ) -> None:
        self.session = session
        self.broker = CapabilityBroker(session.registry)
        self.store = CanvasStore(state_path, document)
        self.allow_external = allow_external
        self.poll_interval_s, self.poll_timeout_s = poll_interval_s, poll_timeout_s
        self.engines: dict[str, OperationEngine] = {}
        self.runners: dict[str, asyncio.Task] = {}
        self.active: dict[str, dict[str, Any]] = {}
        self.task_locks: dict[tuple[str, str], asyncio.Lock] = {}
        for job in self.store.jobs():
            if job["status"] in {"running", "polling", "cancelling"}:
                job["status"] = "interrupted"
                job["error"] = "Host stopped; submission outcome may be unknown. No automatic resubmission."
                self.store.save_job(job)

    def draft(self) -> dict[str, Any]:
        """Return the current draft, identity and host execution policy."""
        document = self.store.document()
        return {
            "document": document.model_dump(mode="json"),
            "digest": workflow_digest(document),
            "allow_external": self.allow_external,
        }

    def edit(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply the shared atomic edit contract; running jobs retain their own snapshots."""
        request = WorkflowEditRequest.model_validate(payload)
        if len(request.edits) > 200:
            raise ValidationFailure("too many edits", code=ErrorCode.GRAPH_INVALID)
        self.store.edit(request)
        return self.draft()

    def node(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create authoritative node fields, using artifact roles for unwired inputs."""
        from vidliner.operators.registry import default_registry

        node = new_workflow_node(payload["operator"], payload["id"], config=payload.get("config"))
        spec = default_registry().describe(node.operator)
        node = node.model_copy(
            update={
                "inputs": {
                    port: ArtifactInput(role=f"{node.id}.{port}")
                    for port, binding in spec.inputs.items()
                    if binding.required
                }
            }
        )
        return node.model_dump(mode="json")

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Preflight and claim a unique intent before scheduling any provider call."""
        request = WorkflowExecutionRequest.model_validate(payload)
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", request.job_id):
            raise ValidationFailure("invalid job id", code=ErrorCode.GRAPH_INVALID)
        request = request.model_copy(update={"allow_external": self.allow_external})
        identity = digest_json(
            {
                "request": request.model_dump(mode="json"),
                "runtime": self.session.profile.model_dump(mode="json"),
            }
        )
        try:
            old = self.store.job(request.job_id)
        except ValidationFailure:
            old = None
        if old is not None:
            return self.store.claim(request.job_id, identity, {}) or old
        if len(self.active) >= 4:
            raise ValidationFailure("canvas host is at capacity", code=ErrorCode.BACKEND_RATE_LIMITED)
        document = self.store.document()
        snapshot = {
            "job_id": request.job_id,
            "digest": request.expected_digest,
            "seed": request.seed,
            "document": document.model_dump(mode="json"),
            "status": "running",
            "nodes": {},
            "tasks": {},
            "events": [],
            "error": None,
            "cancel_requested": False,
            "cancel_requests": [],
        }
        engine = await build_workflow_engine(
            document,
            request,
            store=self.session.store,
            broker=self.broker,
            state=_RunState(self, snapshot),
            on_event=lambda event: self._event(snapshot, event),
            workers=self.session.profile.concurrency.workers,
        )
        if len(self.active) >= 4:
            raise ValidationFailure("canvas host is at capacity", code=ErrorCode.BACKEND_RATE_LIMITED)
        existing = self.store.claim(request.job_id, identity, snapshot)
        if existing is not None:
            return existing
        self.engines[request.job_id] = engine
        self.active[request.job_id] = snapshot
        self.runners[request.job_id] = asyncio.create_task(self._run(snapshot, engine))
        return snapshot

    def _event(self, snapshot: dict[str, Any], event: dict[str, Any]) -> None:
        snapshot["events"] = (snapshot["events"] + [event])[-200:]
        self.store.save_job(snapshot)

    async def _run(self, snapshot: dict[str, Any], engine: OperationEngine) -> None:
        try:
            report = await engine.execute()
            if not report.succeeded():
                snapshot["status"] = "cancelled" if report.cancelled else "failed"
                snapshot["error"] = "Graph execution did not complete; inspect node results."
            if snapshot["tasks"]:
                if snapshot.get("cancel_requested"):
                    await self.cancel(snapshot["job_id"])
                snapshot["status"] = "polling"
                deadline = time.monotonic() + self.poll_timeout_s
                while True:
                    # One request per remote task even when multiple lifecycle nodes share its handle.
                    pending = {
                        (task["backend_id"], task["task_id"])
                        for task in snapshot["tasks"].values()
                        if not VideoTask.model_validate(task).terminal
                    }
                    if not pending:
                        states = {task["status"] for task in snapshot["tasks"].values()}
                        snapshot["status"] = (
                            "failed"
                            if "failed" in states or (not report.succeeded() and not report.cancelled)
                            else "cancelled"
                            if "cancelled" in states or report.cancelled
                            else "succeeded"
                        )
                        break
                    if time.monotonic() >= deadline:
                        snapshot["status"] = "waiting"
                        snapshot["error"] = (
                            "Polling limit reached; saved handles remain available for refresh/cancel."
                        )
                        break
                    for backend_id, task_id in pending:
                        node_id = next(
                            key
                            for key, task in snapshot["tasks"].items()
                            if (task["backend_id"], task["task_id"]) == (backend_id, task_id)
                        )
                        await self.task_action(snapshot["job_id"], node_id, "status")
                    self.store.save_job(snapshot)
                    if any(
                        not VideoTask.model_validate(task).terminal for task in snapshot["tasks"].values()
                    ):
                        await asyncio.sleep(self.poll_interval_s)
            elif report.succeeded():
                snapshot["status"] = "succeeded"
        except asyncio.CancelledError:
            snapshot["status"] = "interrupted"
            snapshot["error"] = "Host stopped; saved tasks can be refreshed. No automatic resubmission."
            raise
        except Exception:
            snapshot["status"] = "interrupted"
            snapshot["error"] = (
                "Execution or task query failed; no automatic resubmission. Refresh saved handles."
            )
        finally:
            self.store.save_job(snapshot)
            self.active.pop(snapshot["job_id"], None)
            self.engines.pop(snapshot["job_id"], None)
            self.runners.pop(snapshot["job_id"], None)

    async def task_action(self, job_id: str, node_id: str, method: str) -> dict[str, Any]:
        """Query or cancel one persisted handle through the existing capacity limiter."""
        if method not in {"status", "cancel"}:
            raise ValidationFailure("invalid task action", code=ErrorCode.GRAPH_INVALID)
        snapshot = self.active.get(job_id) or self.store.job(job_id)
        if node_id not in snapshot["tasks"]:
            raise ValidationFailure("unknown task node", code=ErrorCode.GRAPH_INVALID)
        task = VideoTask.model_validate(snapshot["tasks"][node_id])
        if not self.allow_external:
            raise ValidationFailure(
                "external task actions are disabled", code=ErrorCode.BACKEND_REQUEST_FAILED
            )
        handle = self.broker.handle(CAP_VIDEO_GENERATION)
        lock = self.task_locks.setdefault((job_id, task.task_id), asyncio.Lock())
        async with lock:
            snapshot = self.active.get(job_id) or self.store.job(job_id)
            task = VideoTask.model_validate(snapshot["tasks"][node_id])
            if method == "cancel" and task.task_id in snapshot.get("cancel_requests", []):
                return snapshot
            updated = await handle.call(
                method, task, PipelineContext(job_id=job_id, node_id=node_id, seed=snapshot["seed"])
            )
            updated = VideoTask.model_validate(updated)
            if method == "cancel":
                snapshot.setdefault("cancel_requests", []).append(task.task_id)
            for key, value in snapshot["tasks"].items():
                if (value["backend_id"], value["task_id"]) == (task.backend_id, task.task_id):
                    snapshot["tasks"][key] = updated.model_dump(mode="json")
            if job_id not in self.active and all(
                VideoTask.model_validate(value).terminal for value in snapshot["tasks"].values()
            ):
                states = {value["status"] for value in snapshot["tasks"].values()}
                failed = "failed" in states or any(
                    node["status"] == "failed" for node in snapshot["nodes"].values()
                )
                snapshot["status"] = (
                    "failed" if failed else "cancelled" if "cancelled" in states else "succeeded"
                )
                if snapshot["status"] == "succeeded":
                    snapshot["error"] = None
            self.store.save_job(snapshot)
        return snapshot

    async def cancel(self, job_id: str) -> dict[str, Any]:
        """Stop local scheduling and explicitly request cancellation of saved remote tasks."""
        engine = self.engines.get(job_id)
        if engine is not None:
            engine.cancel()
        snapshot = self.active.get(job_id) or self.store.job(job_id)
        snapshot["cancel_requested"] = True
        self.store.save_job(snapshot)
        for node_id in list(snapshot["tasks"]):
            if not VideoTask.model_validate(snapshot["tasks"][node_id]).terminal:
                await self.task_action(job_id, node_id, "cancel")
        return self.active.get(job_id) or self.store.job(job_id)

    async def close(self) -> None:
        """Stop local jobs and persist interrupted intents without remote resubmission."""
        for engine in self.engines.values():
            engine.cancel()
        runners = list(self.runners.values())
        for runner in runners:
            runner.cancel()
        await asyncio.gather(*runners, return_exceptions=True)
        self.store.close()
        self.session.close()
