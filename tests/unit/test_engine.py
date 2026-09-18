"""Operation graph and engine tests.

These exercise the scheduler in isolation with fake nodes and a fake runner, so a failure here means
the *engine* is wrong, not an operator or a backend.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from vidliner.core.errors import BackendFailure, ErrorCode, VidlinerError
from vidliner.core.graph import (
    ConstantValue,
    NodeOutput,
    OperationGraph,
    OperationNode,
    RetryPolicy,
    SampleSource,
    port_types_compatible,
)
from vidliner.core.results import NodeResult, PortValue, utc_now
from vidliner.domain.enums import ArtifactKind, Determinism, NodeStatus, PortType, StageName
from vidliner.runtime.engine import EngineOptions, OperationEngine, format_report
from vidliner.storage.state import InMemoryRunState


def make_node(
    name: str,
    *,
    inputs: dict[str, object] | None = None,
    outputs: dict[str, PortType] | None = None,
    stage: StageName = StageName.DETECT,
    retry: RetryPolicy | None = None,
    timeout_s: float | None = None,
    cacheable: bool = True,
    determinism: Determinism = Determinism.DETERMINISTIC,
) -> OperationNode:
    """Build a test node with sensible defaults."""
    return OperationNode(
        node_id=f"n_{name}",
        operator=f"test.{name}",
        operator_version="1.0.0",
        stage=stage,
        inputs=dict(inputs or {}),
        outputs=dict(outputs or {"out": PortType.ANY}),
        config={"name": name},
        retry=retry or RetryPolicy(),
        timeout_s=timeout_s,
        cacheable=cacheable,
        determinism=determinism,
    )


def result_for(
    node: OperationNode,
    *,
    status: NodeStatus = NodeStatus.SUCCEEDED,
    payload: dict[str, object] | None = None,
) -> NodeResult:
    """Build a node result for a test node."""
    now = utc_now()
    return NodeResult(
        node_id=node.node_id,
        operator=node.operator,
        operator_version=node.operator_version,
        status=status,
        outputs={"out": PortValue.of_payload(payload or {"value": node.node_id})},
        payload=payload or {},
        started_at=now,
        finished_at=now,
        duration_ms=1,
    )


class RecordingRunner:
    """A NodeRunner that records calls and can be told to fail or block."""

    def __init__(
        self,
        *,
        failures: dict[str, int] | None = None,
        block_for: dict[str, float] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[str, int]] = []
        self._failures = dict(failures or {})
        self._block_for = dict(block_for or {})
        self._error = error

    def implementation_identity(self, node: OperationNode) -> str:
        return f"test:{node.operator}"

    async def __call__(
        self, node: OperationNode, *, seed: int, attempt: int, cancel: asyncio.Event
    ) -> NodeResult:
        self.calls.append((node.node_id, attempt))
        delay = self._block_for.get(node.node_id, 0.0)
        if delay:
            await asyncio.sleep(delay)
        remaining = self._failures.get(node.node_id, 0)
        if remaining > 0:
            self._failures[node.node_id] = remaining - 1
            raise self._error or VidlinerError("boom", code=ErrorCode.OPERATOR_CONFIG_INVALID)
        return result_for(node, payload={"value": node.node_id})


# --------------------------------------------------------------------------- #
# Graph validation
# --------------------------------------------------------------------------- #


def test_graph_rejects_duplicate_node_ids() -> None:
    node = make_node("a")
    graph = OperationGraph(nodes=(node, node))
    with pytest.raises(VidlinerError, match="duplicate node ids"):
        graph.validate()


def test_graph_rejects_unknown_dependency() -> None:
    node = make_node("a", inputs={"in": NodeOutput("n_missing", "out")})
    with pytest.raises(VidlinerError, match="references unknown node"):
        OperationGraph(nodes=(node,)).validate()


def test_graph_rejects_unknown_port_on_a_known_node() -> None:
    producer = make_node("p")
    consumer = make_node("c", inputs={"in": NodeOutput(producer.node_id, "nope")})
    with pytest.raises(VidlinerError, match="does not produce"):
        OperationGraph(nodes=(producer, consumer)).validate()


def test_graph_detects_cycles() -> None:
    first = make_node("a")
    second = make_node("b", inputs={"in": NodeOutput("n_a", "out")})
    first_with_cycle = OperationNode(
        node_id="n_a",
        operator="test.a",
        operator_version="1.0.0",
        stage=StageName.DETECT,
        inputs={"in": NodeOutput("n_b", "out")},
        outputs={"out": PortType.ANY},
    )
    graph = OperationGraph(nodes=(first_with_cycle, second))
    with pytest.raises(VidlinerError, match="cycle"):
        graph.validate()
    del first


def test_graph_topological_order_is_stable() -> None:
    producer = make_node("p", stage=StageName.INGEST)
    consumer = make_node("c", inputs={"in": NodeOutput(producer.node_id, "out")}, stage=StageName.DETECT)
    graph = OperationGraph(nodes=(consumer, producer))
    order = [node.node_id for node in graph.topological_order()]
    assert order == [producer.node_id, consumer.node_id]


def test_graph_reports_capabilities_and_stages() -> None:
    first = OperationNode(
        node_id="n_a",
        operator="test.a",
        operator_version="1",
        stage=StageName.INGEST,
        outputs={"out": PortType.ANY},
        needs=("vision.object_detection.v1",),
    )
    second = OperationNode(
        node_id="n_b",
        operator="test.b",
        operator_version="1",
        stage=StageName.DETECT,
        outputs={"out": PortType.ANY},
        needs=("vision.instance_segmentation.v1",),
    )
    graph = OperationGraph(nodes=(first, second))
    assert graph.capabilities() == ("vision.instance_segmentation.v1", "vision.object_detection.v1")
    assert graph.describe_stages() == [(StageName.INGEST, 1), (StageName.DETECT, 1)]


def test_port_type_compatibility() -> None:
    assert port_types_compatible(PortType.ANY, PortType.IMAGE_REF)
    assert port_types_compatible(PortType.MASK_REF, PortType.MASK_REF)
    assert not port_types_compatible(PortType.MASK_REF, PortType.IMAGE_REF)


def test_retry_policy_backoff_is_deterministic_and_grows() -> None:
    policy = RetryPolicy(attempts=4, backoff_s=0.5, multiplier=2.0)
    assert policy.delay_for(1, 1) == 0.0
    assert policy.delay_for(2, 1) == pytest.approx(0.5)
    assert policy.delay_for(3, 1) == pytest.approx(1.0)
    assert policy.delay_for(3, 1) == policy.delay_for(3, 1)


def test_retry_policy_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(attempts=0)
    with pytest.raises(ValueError):
        RetryPolicy(jitter=2.0)


def test_retry_policy_allows_filter() -> None:
    policy = RetryPolicy(attempts=2, retry_on=(ErrorCode.NODE_TIMEOUT,))
    assert policy.allows(ErrorCode.NODE_TIMEOUT)
    assert not policy.allows(ErrorCode.OPERATOR_CONFIG_INVALID)


def test_node_rejects_missing_outputs() -> None:
    with pytest.raises(ValueError, match="at least one output port"):
        OperationNode(
            node_id="n",
            operator="test",
            operator_version="1",
            stage=StageName.DETECT,
            outputs={},
        )


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


async def test_engine_runs_a_linear_graph() -> None:
    first = make_node("a")
    second = make_node("b", inputs={"in": NodeOutput(first.node_id, "out")})
    graph = OperationGraph(nodes=(first, second))
    runner = RecordingRunner()
    engine = OperationEngine(graph, runner, InMemoryRunState())
    report = await engine.execute()
    assert report.succeeded()
    assert report.count(NodeStatus.SUCCEEDED) == 2
    assert [node_id for node_id, _ in runner.calls] == [first.node_id, second.node_id]


async def test_engine_fans_out_and_runs_independent_branches() -> None:
    root = make_node("root", stage=StageName.INGEST)
    branches = [make_node(f"b{index}", inputs={"in": NodeOutput(root.node_id, "out")}) for index in range(4)]
    graph = OperationGraph(nodes=(root, *branches))
    runner = RecordingRunner()
    engine = OperationEngine(graph, runner, InMemoryRunState(), options=EngineOptions(workers=4))
    report = await engine.execute()
    assert report.count(NodeStatus.SUCCEEDED) == 5
    assert len(runner.calls) == 5


async def test_engine_retries_per_policy_and_reports_the_attempt() -> None:
    node = make_node("flaky", retry=RetryPolicy(attempts=3, backoff_s=0.0))
    graph = OperationGraph(nodes=(node,))
    runner = RecordingRunner(failures={node.node_id: 2})
    engine = OperationEngine(graph, runner, InMemoryRunState())
    report = await engine.execute()
    assert report.succeeded()
    assert [attempt for _, attempt in runner.calls] == [1, 2, 3]


async def test_engine_does_not_retry_by_default() -> None:
    node = make_node("once")
    graph = OperationGraph(nodes=(node,))
    runner = RecordingRunner(failures={node.node_id: 5})
    engine = OperationEngine(graph, runner, InMemoryRunState())
    report = await engine.execute()
    assert len(runner.calls) == 1
    assert report.failed_nodes == (node.node_id,)


async def test_engine_never_retries_a_backend_that_declares_itself_unsafe() -> None:
    node = make_node("unsafe", retry=RetryPolicy(attempts=3, backoff_s=0.0))
    graph = OperationGraph(nodes=(node,))
    runner = RecordingRunner(
        failures={node.node_id: 5},
        error=BackendFailure("nope", backend_id="x", safe_to_retry=False),
    )
    engine = OperationEngine(graph, runner, InMemoryRunState())
    report = await engine.execute()
    assert len(runner.calls) == 1
    assert report.failed_nodes == (node.node_id,)


async def test_engine_skips_descendants_of_a_failed_node() -> None:
    first = make_node("a")
    second = make_node("b", inputs={"in": NodeOutput(first.node_id, "out")})
    third = make_node("c", inputs={"in": NodeOutput(second.node_id, "out")})
    graph = OperationGraph(nodes=(first, second, third))
    runner = RecordingRunner(failures={first.node_id: 9})
    engine = OperationEngine(graph, runner, InMemoryRunState())
    report = await engine.execute()
    assert report.failed_nodes == (first.node_id,)
    assert set(report.skipped_nodes) == {second.node_id, third.node_id}
    assert report.nodes[third.node_id].status is NodeStatus.SKIPPED
    assert report.nodes[third.node_id].error_code == ErrorCode.DEPENDENCY_FAILED.value


async def test_engine_keeps_sibling_branches_alive_after_a_failure() -> None:
    root = make_node("root", stage=StageName.INGEST)
    failing = make_node("bad", inputs={"in": NodeOutput(root.node_id, "out")})
    sibling = make_node("good", inputs={"in": NodeOutput(root.node_id, "out")})
    graph = OperationGraph(nodes=(root, failing, sibling))
    runner = RecordingRunner(failures={failing.node_id: 9})
    engine = OperationEngine(graph, runner, InMemoryRunState())
    report = await engine.execute()
    assert report.nodes[sibling.node_id].status is NodeStatus.SUCCEEDED
    assert report.failed_nodes == (failing.node_id,)


async def test_engine_fail_fast_stops_early() -> None:
    root = make_node("root", stage=StageName.INGEST)
    failing = make_node("bad", inputs={"in": NodeOutput(root.node_id, "out")})
    other = make_node("other", stage=StageName.INGEST)
    graph = OperationGraph(nodes=(root, failing, other))
    runner = RecordingRunner(failures={failing.node_id: 9})
    engine = OperationEngine(
        graph, runner, InMemoryRunState(), options=EngineOptions(fail_fast=True, workers=1)
    )
    report = await engine.execute()
    assert report.failed_nodes == (failing.node_id,)
    assert "fail_fast" in " ".join(report.notes)


async def test_engine_enforces_a_node_timeout() -> None:
    node = make_node("slow", timeout_s=0.05)
    graph = OperationGraph(nodes=(node,))
    runner = RecordingRunner(block_for={node.node_id: 1.0})
    engine = OperationEngine(graph, runner, InMemoryRunState())
    report = await engine.execute()
    assert report.failed_nodes == (node.node_id,)
    assert report.nodes[node.node_id].error_code == ErrorCode.NODE_TIMEOUT.value


async def test_engine_honours_cancellation() -> None:
    node = make_node("slow")
    graph = OperationGraph(nodes=(node,))
    runner = RecordingRunner(block_for={node.node_id: 5.0})
    engine = OperationEngine(graph, runner, InMemoryRunState())

    async def cancel_soon() -> None:
        await asyncio.sleep(0.05)
        engine.cancel()

    await asyncio.gather(engine.execute(), cancel_soon())
    assert engine.report.cancelled


async def test_engine_uses_the_node_cache() -> None:
    node = make_node("cached")
    graph = OperationGraph(nodes=(node,))
    state = InMemoryRunState()
    runner = RecordingRunner()
    first = await OperationEngine(graph, runner, state).execute()
    assert first.count(NodeStatus.SUCCEEDED) == 1
    calls = len(runner.calls)
    second = await OperationEngine(graph, runner, state, options=EngineOptions(prefer_cache=True)).execute()
    assert len(runner.calls) == calls
    assert second.cache_hits == (node.node_id,)
    assert second.nodes[node.node_id].cache_hit


async def test_engine_does_not_cache_nondeterministic_nodes() -> None:
    node = make_node("random", determinism=Determinism.NONDETERMINISTIC, cacheable=True)
    graph = OperationGraph(nodes=(node,))
    state = InMemoryRunState()
    runner = RecordingRunner()
    first = await OperationEngine(graph, runner, state).execute()
    assert first.count(NodeStatus.SUCCEEDED) == 1
    assert not first.cache_hits


async def test_engine_cache_key_changes_with_config() -> None:
    from vidliner.core.cache import cache_key

    first = make_node("a")
    second = OperationNode(
        node_id=first.node_id,
        operator=first.operator,
        operator_version=first.operator_version,
        stage=first.stage,
        outputs=first.outputs,
        config={"name": "different"},
    )
    key_a = cache_key(first, implementation="test", input_digests={}, seed=1)
    key_b = cache_key(second, implementation="test", input_digests={}, seed=1)
    key_c = cache_key(first, implementation="test", input_digests={}, seed=2)
    assert key_a != key_b
    assert key_a != key_c


async def test_engine_resumes_completed_nodes() -> None:
    first = make_node("a")
    second = make_node("b", inputs={"in": NodeOutput(first.node_id, "out")})
    graph = OperationGraph(nodes=(first, second))
    state = InMemoryRunState()
    runner = RecordingRunner()
    await OperationEngine(graph, runner, state).execute()
    calls_after_first = list(runner.calls)
    report = await OperationEngine(graph, runner, state).execute()
    assert report.succeeded()
    # Every node was already complete, so nothing new ran.
    assert runner.calls == calls_after_first
    assert set(report.resumed_nodes) == {first.node_id, second.node_id}


async def test_engine_can_be_told_to_ignore_the_cache_and_resume() -> None:
    node = make_node("a")
    graph = OperationGraph(nodes=(node,))
    state = InMemoryRunState()
    runner = RecordingRunner()
    await OperationEngine(graph, runner, state).execute()
    await OperationEngine(
        graph, runner, state, options=EngineOptions(use_cache=False, resume=False)
    ).execute()
    assert len(runner.calls) == 2


async def test_engine_accepts_a_preseeded_node_result() -> None:
    node = make_node("injected")
    graph = OperationGraph(nodes=(node,))
    runner = RecordingRunner()
    engine = OperationEngine(graph, runner, InMemoryRunState())
    engine.provide(node.node_id, result_for(node, payload={"value": "supplied"}))
    report = await engine.execute()
    assert report.nodes[node.node_id].payload["value"] == "supplied"
    assert runner.calls == []


def test_format_report_mentions_failures() -> None:
    from vidliner.runtime.engine import ExecutionReport

    report = ExecutionReport()
    report.failed_nodes = ("n_1",)
    report.nodes["n_1"] = NodeResult(
        node_id="n_1",
        operator="test",
        operator_version="1",
        status=NodeStatus.FAILED,
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        duration_ms=1,
        error_message="[operator] boom",
    )
    text = format_report(report)
    assert "FAILED n_1" in text


def test_sample_source_binding_encodes() -> None:
    assert SampleSource("source_asset").encode() == "sample:source_asset"
    assert ConstantValue({"a": 1}).encode().startswith("const:")
    assert NodeOutput("n_1", "out").encode() == "node:n_1#out"


def test_artifact_kind_is_available_for_results() -> None:
    assert ArtifactKind.MASK.value == "mask"


def test_graph_validation_is_idempotent(tmp_path: Path) -> None:
    node = make_node("a")
    graph = OperationGraph(nodes=(node,))
    graph.validate()
    graph.validate()
    del tmp_path
