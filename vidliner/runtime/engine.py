"""The operation engine.

A single-process, `asyncio`-based scheduler over an :class:`~vidliner.core.graph.OperationGraph`.
It owns six concerns and nothing else:

* **scheduling** — Kahn-style in-degree counting; a node is dispatched the moment its dependencies
  have completed;
* **bounds** — a global worker semaphore plus a per-node semaphore, so a GPU-bound node serialises
  while a network-bound node overlaps;
* **retry** — the node's own policy, further restricted by whether the backend declared its call
  safe to retry;
* **timeout** — per attempt;
* **cancellation** — one event, honoured before dispatch and while waiting;
* **resume and cache** — completed nodes are reloaded, cacheable nodes are served from the cache.

The engine never talks to a backend directly. It asks a :class:`NodeRunner` to execute one node,
which keeps capability binding, operator dispatch, and artifact I/O outside the scheduler.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from vidliner.core.cache import cache_key, cacheable
from vidliner.core.errors import (
    BackendFailure,
    Cancellation,
    ErrorCode,
    FailureClass,
    VidlinerError,
)
from vidliner.core.graph import OperationGraph, OperationNode
from vidliner.core.results import NodeResult, NodeTelemetry, PortValue, utc_now
from vidliner.domain.enums import NodeStatus

__all__ = [
    "EngineOptions",
    "ExecutionReport",
    "NodeRunner",
    "OperationEngine",
    "RunState",
    "format_report",
]


class RunState(Protocol):
    """What the engine needs from a job's persisted state, and nothing more.

    Both :class:`~vidliner.runtime.state.InMemoryRunState` (tests) and the SQLite-backed store
    (production) satisfy this protocol.
    """

    def completed_node(self, node_id: str, /) -> NodeResult | None:
        """Return the recorded result of a finished node, or ``None``."""
        ...

    def record_node(self, result: NodeResult, node: OperationNode, /, *, seed: int, config_hash: str) -> None:
        """Persist one node result and its evidence."""
        ...

    def cached_result(self, key: str, /) -> NodeResult | None:
        """Return a cached node result for ``key``, verifying artifacts still exist.

        The parameter is positional-only so an implementation may name it whatever reads best at the
        call site (the SQLite-backed recorder calls it ``cache_key``).
        """
        ...

    def store_cache(self, key: str, result: NodeResult, node: OperationNode, /) -> None:
        """Store a node result under ``key``.

        Positional-only for the same reason as :meth:`cached_result`: the implementation is free to
        name the parameters for its own reader.
        """
        ...

    def artifacts_present(self, result: NodeResult, /) -> bool:
        """Whether every artifact referenced by ``result`` is still readable."""
        ...


class NodeRunner(Protocol):
    """Executes exactly one node."""

    async def __call__(
        self,
        node: OperationNode,
        *,
        seed: int,
        attempt: int,
        cancel: asyncio.Event,
    ) -> NodeResult:
        """Run ``node`` once and return its result."""
        ...

    def implementation_identity(self, node: OperationNode) -> str:
        """Identity of the code that will execute ``node``, used in the cache key."""
        ...


@dataclass(frozen=True, slots=True)
class EngineOptions:
    """Tunables for one engine run."""

    workers: int = 4
    per_node_parallelism: int = 4
    job_timeout_s: float | None = None
    fail_fast: bool = False
    use_cache: bool = True
    resume: bool = True
    prefer_cache: bool = False
    """Consult the node cache before the resume record.

    Both paths exist for the same reason (do not redo finished work), but they are reported
    differently: a *resumed* node is one this job already ran, while a *cached* node may come from an
    earlier job with identical inputs. Resume wins by default so a job's own evidence is preferred;
    ``vidliner run --cache-first`` turns it around for cross-job reuse."""
    default_node_timeout_s: float = 600.0
    log_events: bool = True
    on_resolved: Callable[[str, NodeResult], None] | None = None
    """Called with ``(node_id, result)`` when a node is served without executing.

    The runner uses this to rehydrate the node's port values *before* its dependents are dispatched;
    restoring them afterwards would be too late, because a dependent that cannot read its input is
    legitimately skipped.
    """


@dataclass
class ExecutionReport:
    """The engine's account of one run."""

    nodes: dict[str, NodeResult] = field(default_factory=dict)
    started_at: datetime = field(default_factory=utc_now)
    finished_at: datetime | None = None
    cancelled: bool = False
    timed_out: bool = False
    failed_nodes: tuple[str, ...] = ()
    skipped_nodes: tuple[str, ...] = ()
    cache_hits: tuple[str, ...] = ()
    resumed_nodes: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        """Wall-clock duration of the run."""
        if self.finished_at is None:
            return 0.0
        return (self.finished_at - self.started_at).total_seconds()

    def result(self, node_id: str) -> NodeResult | None:
        """Recorded result for one node."""
        return self.nodes.get(node_id)

    def succeeded(self) -> bool:
        """True when no node failed and the run was not cancelled."""
        return not self.failed_nodes and not self.cancelled

    def count(self, status: NodeStatus) -> int:
        """How many nodes ended in ``status``."""
        return sum(1 for result in self.nodes.values() if result.status is status)

    def payload(self, node_id: str, key: str, default: object = None) -> object:
        """Read one key out of a node's payload, or ``default`` when unavailable."""
        result = self.nodes.get(node_id)
        if result is None:
            return default
        return result.payload.get(key, default)

    def output(self, node_id: str, port: str) -> PortValue | None:
        """Read one output port of a node, or ``None``."""
        result = self.nodes.get(node_id)
        if result is None:
            return None
        return result.outputs.get(port)

    def output_artifact(self, node_id: str, port: str):
        """Read the artifact on one output port, or ``None``."""
        value = self.output(node_id, port)
        return value.artifact if value else None

    def nodes_by_stage(self, stage: str) -> tuple[str, ...]:
        """Node ids that ended in ``stage``, sorted."""
        return tuple(
            sorted(result.node_id for result in self.nodes.values() if result.operator.startswith(stage))
        )


class OperationEngine:
    """Schedules and executes an operation graph."""

    def __init__(
        self,
        graph: OperationGraph,
        runner: NodeRunner,
        state: RunState,
        *,
        options: EngineOptions | None = None,
        seed_for_node: Callable[[OperationNode], int] | int | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._graph = graph
        self._runner = runner
        self._state = state
        self._options = options or EngineOptions()
        self._seed_lookup = seed_for_node
        self._on_event: Callable[[dict[str, Any]], None] | None = on_event
        self._index = graph.index()
        self._dependents = graph.dependents()
        self._cancel = asyncio.Event()
        self._in_flight: set[str] = set()
        self._report = ExecutionReport()
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._speculative: dict[str, NodeResult] = {}

    # -- public API -------------------------------------------------------- #

    @property
    def report(self) -> ExecutionReport:
        """The live report; valid after :meth:`execute` returns."""
        return self._report

    def cancel(self) -> None:
        """Request cancellation. In-flight node tasks are cancelled at the next await point."""
        self._cancel.set()

    def provide(self, node_id: str, result: NodeResult) -> None:
        """Pre-seed a node result, used to inject sample inputs supplied by the caller."""
        self._speculative[node_id] = result

    async def execute(self) -> ExecutionReport:
        """Run the graph to completion, or until the first fatal failure when ``fail_fast``.

        Returns:
            The execution report. Failures are recorded in the report rather than raised, so a
            caller can summarize a partially successful run; only :class:`Cancellation` propagates
            when the job itself is cancelled by the caller.
        """
        self._report = ExecutionReport()
        started = time.perf_counter()
        in_degree = {node_id: len(node.dependency_ids) for node_id, node in self._index.items()}
        ready: deque[str] = deque(sorted(node_id for node_id, degree in in_degree.items() if degree == 0))
        completed: set[str] = set()
        tasks: dict[asyncio.Task[NodeResult], str] = {}
        job_deadline = (
            time.perf_counter() + self._options.job_timeout_s if self._options.job_timeout_s else None
        )

        while ready or tasks:
            if self._cancel.is_set():
                await self._drain(tasks)
                self._report.cancelled = True
                break
            if job_deadline is not None and time.perf_counter() > job_deadline:
                await self._drain(tasks)
                self._report.timed_out = True
                self._report.notes.append(f"job exceeded its {self._options.job_timeout_s}s budget")
                break
            # Dispatch everything that is ready and not blocked, respecting the worker budget.
            while ready and len(tasks) < self._options.workers:
                node_id = ready.popleft()
                node = self._index[node_id]
                precomputed = self._resolve_ready_node(node)
                if precomputed is not None:
                    self._report.nodes[node_id] = precomputed
                    completed.add(node_id)
                    self._release(node_id, in_degree, ready)
                    continue
                tasks[asyncio.create_task(self._run_node(node))] = node_id
            if not tasks:
                if not ready:
                    break
                continue
            done, _pending = await asyncio.wait(
                set(tasks),
                timeout=self._next_timeout(job_deadline),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                continue
            for task in done:
                node_id = tasks.pop(task)
                try:
                    result = task.result()
                except Cancellation:
                    self._report.cancelled = True
                    result = self._cancelled_result(node_id)
                except Exception as exc:  # defensive: _run_node should never raise
                    result = self._failure_result(node_id, exc)
                self._report.nodes[node_id] = result
                if result.status is NodeStatus.SUCCEEDED:
                    completed.add(node_id)
                    self._release(node_id, in_degree, ready)
                elif result.status is NodeStatus.CANCELLED:
                    self._report.cancelled = True
                    self._propagate_skip(node_id, in_degree, completed)
                elif result.status is NodeStatus.SKIPPED:
                    self._report.skipped_nodes = (*self._report.skipped_nodes, node_id)
                    self._propagate_skip(node_id, in_degree, completed)
                else:
                    self._report.failed_nodes = (*self._report.failed_nodes, node_id)
                    if self._options.fail_fast:
                        await self._drain(tasks)
                        self._report.notes.append("stopped after the first failure (fail_fast)")
                        break
                    self._propagate_skip(node_id, in_degree, completed)
            if self._options.fail_fast and self._report.failed_nodes:
                break

        self._report.finished_at = utc_now()
        self._report.notes.append(
            f"engine ran {len(self._report.nodes)} node(s) in {time.perf_counter() - started:.3f}s"
        )
        return self._report

    # -- internals --------------------------------------------------------- #

    def _next_timeout(self, job_deadline: float | None) -> float | None:
        if job_deadline is None:
            return 0.5
        return max(0.01, min(0.5, job_deadline - time.perf_counter()))

    async def _drain(self, tasks: dict[asyncio.Task[NodeResult], str]) -> None:
        """Cancel and await every in-flight task, recording cancellations."""
        for task in tasks:
            task.cancel()
        for task in list(tasks):
            try:
                result = await task
            except (Cancellation, asyncio.CancelledError, Exception):
                node_id = tasks[task]
                result = self._cancelled_result(node_id)
            self._report.nodes[result.node_id] = result
        tasks.clear()

    def _resolve_ready_node(self, node: OperationNode) -> NodeResult | None:
        """Return a result for a node that need not execute: injected, resumed, or cached."""
        injected = self._speculative.pop(node.node_id, None)
        if injected is not None:
            self._report.notes.append(f"node {node.node_id} was supplied by the caller")
            self._notify_resolved(node.node_id, injected)
            return injected

        seed = self._seed_for(node)
        if self._options.prefer_cache:
            cached_first = self._cached_result(node, seed)
            if cached_first is not None:
                return cached_first
        if self._options.resume:
            recorded = self._state.completed_node(node.node_id)
            if (
                recorded is not None
                and recorded.status is NodeStatus.SUCCEEDED
                and self._state.artifacts_present(recorded)
            ):
                self._report.resumed_nodes = (*self._report.resumed_nodes, node.node_id)
                self._state.record_node(recorded, node, seed=seed, config_hash=node.config_hash)
                resumed = recorded.model_copy(update={"resumed": True})
                self._notify_resolved(node.node_id, resumed)
                return resumed

        if not self._options.prefer_cache:
            return self._cached_result(node, seed)
        return None

    def _cached_result(self, node: OperationNode, seed: int) -> NodeResult | None:
        """Return a cached result for a node, when caching applies to it."""
        if not self._options.use_cache or not cacheable(node):
            return None
        key = self._cache_key_for(node, seed)
        if key is None:
            return None
        cached = self._state.cached_result(key)
        if cached is None:
            return None
        self._report.cache_hits = (*self._report.cache_hits, node.node_id)
        hit = cached.model_copy(update={"cache_hit": True})
        self._notify_resolved(node.node_id, hit)
        return hit

    def _cache_key_for(self, node: OperationNode, seed: int) -> str | None:
        input_digests: dict[str, str] = {}
        for port, binding in node.inputs.items():
            from vidliner.core.graph import NodeOutput

            if not isinstance(binding, NodeOutput):
                continue
            upstream = self._report.nodes.get(binding.node_id)
            if upstream is None:
                return None
            value = upstream.outputs.get(binding.port)
            if value is not None and value.artifact is not None:
                input_digests[port] = value.artifact.digest
        return cache_key(
            node,
            implementation=self._runner.implementation_identity(node),
            input_digests=input_digests,
            seed=seed,
        )

    def _notify_resolved(self, node_id: str, result: NodeResult) -> None:
        """Tell the runner that a node was resolved without executing."""
        callback = self._options.on_resolved
        if callable(callback):
            callback(node_id, result)

    def _seed_for(self, node: OperationNode) -> int:
        lookup = self._seed_lookup
        if lookup is None:
            return 0
        if isinstance(lookup, int):
            return lookup
        if callable(lookup):
            return int(lookup(node))
        return 0  # pragma: no cover - guarded by the constructor's signature

    def _semaphore_for(self, node: OperationNode) -> asyncio.Semaphore:
        semaphore = self._semaphores.get(node.node_id)
        if semaphore is None:
            semaphore = asyncio.Semaphore(max(1, node.max_parallelism))
            self._semaphores[node.node_id] = semaphore
        return semaphore

    async def _run_node(self, node: OperationNode) -> NodeResult:
        seed = self._seed_for(node)
        timeout = node.timeout_s or self._options.default_node_timeout_s
        policy = node.retry
        last_error: BaseException | None = None
        for attempt in range(1, policy.attempts + 1):
            if self._cancel.is_set():
                return self._cancelled_result(node.node_id)
            if attempt > 1:
                delay = policy.delay_for(attempt, seed)
                if delay > 0:
                    # Waiting out the backoff *is* the intended path when the timeout expires; the
                    # cancellation branch is the exceptional one, and it returns early.
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(self._cancel.wait(), timeout=delay)
                        return self._cancelled_result(node.node_id)
            try:
                async with self._semaphore_for(node):
                    result = await asyncio.wait_for(
                        self._runner(node, seed=seed, attempt=attempt, cancel=self._cancel),
                        timeout=timeout,
                    )
            except TimeoutError:
                last_error = VidlinerError(
                    f"node {node.node_id} ({node.operator}) exceeded its {timeout}s timeout",
                    code=ErrorCode.NODE_TIMEOUT,
                )
            except asyncio.CancelledError:
                return self._cancelled_result(node.node_id)
            except VidlinerError as exc:
                last_error = exc
                if not self._retryable(exc, policy, attempt):
                    break
            except Exception as exc:
                last_error = exc
                if attempt >= policy.attempts:
                    break
            else:
                if result.attempt == 0:
                    result = result.model_copy(update={"attempt": attempt})
                self._record(result, node, seed)
                return result
        result = self._failure_result(node.node_id, last_error, attempt=policy.attempts, node=node)
        self._record(result, node, seed)
        return result

    def _retryable(self, error: VidlinerError, policy: object, attempt: int) -> bool:
        attempts = getattr(policy, "attempts", 1)
        if attempt >= attempts:
            return False
        if isinstance(error, Cancellation):
            return False
        if isinstance(error, BackendFailure) and not error.safe_to_retry:
            return False
        allows = getattr(policy, "allows", None)
        if callable(allows):
            return bool(allows(error.code))
        return True

    def _record(self, result: NodeResult, node: OperationNode, seed: int) -> None:
        if result.status is NodeStatus.SKIPPED:
            # A skipped node belongs to a branch that does not exist for this sample; it is recorded
            # as a decision, never as a failure, and never as a cache entry.
            self._emit(node, result)
            return
        self._state.record_node(result, node, seed=seed, config_hash=node.config_hash)
        if result.status is NodeStatus.SUCCEEDED and self._options.use_cache and cacheable(node):
            key = self._cache_key_for(node, seed)
            if key is not None:
                self._state.store_cache(key, result, node)
        self._emit(node, result)

    def _emit(self, node: OperationNode, result: NodeResult) -> None:
        if not self._options.log_events or not callable(self._on_event):
            return
        self._on_event(
            {
                "event": "node.finished",
                "node_id": node.node_id,
                "operator": node.operator,
                "stage": node.stage.value,
                "lineage": node.lineage_label,
                "status": result.status.value,
                "duration_ms": result.duration_ms,
                "backend": result.backend_id,
                "cache_hit": result.cache_hit,
                "resumed": result.resumed,
                "attempt": result.attempt,
                "error_code": result.error_code,
                "outputs": {
                    port: value.artifact.digest if value.artifact else None
                    for port, value in result.outputs.items()
                },
            }
        )

    def _release(
        self,
        node_id: str,
        in_degree: dict[str, int],
        ready: deque[str],
    ) -> None:
        """Decrement the in-degree of every dependent and enqueue the newly runnable ones."""
        for dependent in self._dependents.get(node_id, []):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                ready.append(dependent)

    def _propagate_skip(
        self,
        failed_node: str,
        in_degree: dict[str, int],
        completed: set[str],
    ) -> None:
        """Mark every descendant of a failed node as skipped and remove it from the schedule."""
        queue = deque(self._dependents.get(failed_node, []))
        while queue:
            node_id = queue.popleft()
            if node_id in completed or node_id in self._report.nodes:
                continue
            node = self._index[node_id]
            self._report.skipped_nodes = (*self._report.skipped_nodes, node_id)
            skipped = NodeResult(
                node_id=node_id,
                operator=node.operator,
                operator_version=node.operator_version,
                status=NodeStatus.SKIPPED,
                started_at=utc_now(),
                finished_at=utc_now(),
                duration_ms=0,
                error_class=VidlinerError.__name__,
                error_code=ErrorCode.DEPENDENCY_FAILED.value,
                error_message=f"dependency {failed_node} did not succeed",
            )
            self._report.nodes[node_id] = skipped
            self._record(skipped, node, self._seed_for(node))
            completed.add(node_id)
            for dependent in self._dependents.get(node_id, []):
                if dependent in in_degree:
                    in_degree[dependent] = max(0, in_degree[dependent] - 1)
                queue.append(dependent)

    def _cancelled_result(self, node_id: str) -> NodeResult:
        node = self._index[node_id]
        now = utc_now()
        return NodeResult(
            node_id=node_id,
            operator=node.operator,
            operator_version=node.operator_version,
            status=NodeStatus.CANCELLED,
            started_at=now,
            finished_at=now,
            duration_ms=0,
            error_class=Cancellation.__name__,
            error_code=ErrorCode.JOB_CANCELLED.value,
            error_message="job was cancelled",
        )

    def _failure_result(
        self,
        node_id: str,
        error: BaseException | None,
        *,
        attempt: int | None = None,
        node: OperationNode | None = None,
    ) -> NodeResult:
        target = node or self._index[node_id]
        now = utc_now()
        code = (
            error.code.value if isinstance(error, VidlinerError) else ErrorCode.OPERATOR_CONFIG_INVALID.value
        )
        failure_class = error.failure_class if isinstance(error, VidlinerError) else FailureClass.OPERATOR
        message = error.message if isinstance(error, VidlinerError) else f"{type(error).__name__}: {error}"
        return NodeResult(
            node_id=node_id,
            operator=target.operator,
            operator_version=target.operator_version,
            status=NodeStatus.FAILED,
            started_at=now,
            finished_at=now,
            duration_ms=0,
            backend_id=getattr(error, "backend_id", None),
            attempt=attempt if attempt is not None else 0,
            determinism=target.determinism,
            telemetry=NodeTelemetry(),
            error_class=type(error).__name__ if error else "UnknownError",
            error_code=code,
            error_message=f"[{failure_class.value}] {message}",
        )


def format_report(report: ExecutionReport, *, limit: int = 20) -> str:
    """Render a compact human-readable summary of an execution report."""
    lines = [
        f"nodes: {len(report.nodes)}  succeeded: {report.count(NodeStatus.SUCCEEDED)}  "
        f"failed: {len(report.failed_nodes)}  skipped: {len(report.skipped_nodes)}  "
        f"cached: {len(report.cache_hits)}  resumed: {len(report.resumed_nodes)}",
        f"duration: {report.duration_s:.2f}s",
    ]
    if report.cancelled:
        lines.append("state: cancelled")
    if report.timed_out:
        lines.append("state: timed out")
    for node_id in report.failed_nodes[:limit]:
        result = report.nodes[node_id]
        lines.append(f"  FAILED {node_id} {result.operator}: {result.error_message}")
    for note in report.notes[-3:]:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


# Determinism is re-exported for convenience of engine users.
