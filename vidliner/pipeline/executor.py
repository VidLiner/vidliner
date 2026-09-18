"""Execution of one graph node.

The executor is the only component that knows how a node's declared ports map onto real values. It
resolves bindings, derives the node's seed, calls the operator with an
:class:`~vidliner.operators.base.ExecutionContext`, and converts what the operator returns into a
:class:`~vidliner.core.results.NodeResult` whose payload survives a restart.

It owns one policy decision that the engine delegates to it: a node belonging to a branch that does
not exist for this sample (a sample with fewer target objects than the plan reserved) is
**skipped**, not failed, and every node that depends on a skipped node is skipped too. That is what
lets one graph shape serve every sample regardless of how many objects it contains.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from vidliner.core.errors import ErrorCode, InfrastructureFailure, OperatorFailure, VidlinerError
from vidliner.core.graph import NodeOutput, OperationNode, SampleSource
from vidliner.core.results import ArtifactRef, NodeResult, PortValue, utc_now
from vidliner.core.seedtree import derive_seed
from vidliner.domain.enums import NodeStatus
from vidliner.operators.base import ExecutionContext, OperatorEvent, SampleContext
from vidliner.operators.registry import create_operator
from vidliner.operators.values import decode_value, encode_value
from vidliner.pipeline.backends import CapabilityBroker
from vidliner.storage.workspace import ArtifactStore, BackendIO

__all__ = ["ArtifactExecutor", "ExecutionOutcome", "SkippedNode", "seed_for_node"]

_SKIP_MARKER = "__skipped__"
_SAMPLE_PORT = "__sample__"


class SkippedNode(VidlinerError):
    """Raised when a node belongs to a branch that does not exist for this sample.

    This is not a failure: a sample with one target object legitimately has nothing to do for the
    second branch the assembler reserved.
    """

    default_code = ErrorCode.DEPENDENCY_FAILED

    def __init__(self, message: str, *, node_id: str, reason: str = "branch absent") -> None:
        super().__init__(
            message, code=ErrorCode.DEPENDENCY_FAILED, detail={"node_id": node_id, "reason": reason}
        )


def seed_for_node(job_seed: int, node: OperationNode) -> int:
    """Derive a node's seed from the job seed and the node's lineage path.

    The path is ``sample → object → candidate``, so nodes that share a candidate derive related
    seeds, and adding a candidate value to a recipe does not shift the seeds of the others.
    """
    material = "/".join(node.lineage) if node.lineage else node.node_id
    return derive_seed(job_seed, material, node.node_id)


@dataclass
class ExecutionOutcome:
    """Everything one node execution produced."""

    result: NodeResult
    values: dict[str, Any] = field(default_factory=dict)
    backend_id: str | None = None


@dataclass
class PortState:
    """The live values produced by each node, keyed by node id and port name."""

    values: dict[str, dict[str, Any]] = field(default_factory=dict)

    def put(self, node_id: str, ports: Mapping[str, Any]) -> None:
        """Record the values produced by one node."""
        self.values[node_id] = dict(ports)

    def get(self, node_id: str, port: str) -> Any:
        """Read one port value, or ``None``."""
        return self.values.get(node_id, {}).get(port)

    def has(self, node_id: str, port: str) -> bool:
        """Whether a port has a value."""
        return port in self.values.get(node_id, {})

    def mark_skipped(self, node_id: str) -> None:
        """Record that a node was skipped because its branch does not exist."""
        self.values[node_id] = {_SKIP_MARKER: True}

    def is_skipped(self, node_id: str) -> bool:
        """Whether a node was skipped."""
        return _SKIP_MARKER in self.values.get(node_id, {})

    def forget(self, node_id: str) -> None:
        """Drop a node's values, used when a re-execution invalidates them."""
        self.values.pop(node_id, None)


class ArtifactExecutor:
    """Executes one node of an operation graph."""

    def __init__(
        self,
        *,
        job_id: str,
        store: ArtifactStore,
        broker: CapabilityBroker,
        sample_for: Callable[[OperationNode], SampleContext | None] | None = None,
        on_event: Callable[[str, OperationNode, dict[str, Any]], None] | None = None,
        ports: PortState | None = None,
    ) -> None:
        self._job_id = job_id
        self._store = store
        self._broker = broker
        self._sample_for = sample_for
        self._on_event = on_event
        self._ports = ports or PortState()

    @property
    def ports(self) -> PortState:
        """The live port values."""
        return self._ports

    # -- NodeRunner protocol ----------------------------------------------- #

    def implementation_identity(self, node: OperationNode) -> str:
        """Identity of the code that will run a node, used in the cache key."""
        capabilities = ",".join(sorted(node.needs))
        return f"operator:{node.operator}@{node.operator_version};caps={capabilities}"

    async def __call__(
        self,
        node: OperationNode,
        *,
        seed: int,
        attempt: int,
        cancel: asyncio.Event,
    ) -> NodeResult:
        """Execute ``node`` once and return its result (the engine's ``NodeRunner`` contract)."""
        outcome = await self.execute(node, seed=seed, attempt=attempt, cancel=cancel)
        return outcome.result

    # -- core -------------------------------------------------------------- #

    async def execute(
        self,
        node: OperationNode,
        *,
        seed: int,
        attempt: int,
        cancel: asyncio.Event,
    ) -> ExecutionOutcome:
        """Execute a node, resolving its inputs and recording its outputs."""
        started = utc_now()
        clock = time.perf_counter()
        sample = self._sample_for(node) if self._sample_for is not None else None
        try:
            inputs = self.resolve_inputs(node)
        except SkippedNode as skipped:
            self._ports.mark_skipped(node.node_id)
            return self._skipped(node, started, clock, skipped.message)
        if sample is not None:
            inputs[_SAMPLE_PORT] = sample

        operator = create_operator(node.operator, node.config)
        context = ExecutionContext(
            job_id=self._job_id,
            node=node,
            seed=seed,
            io=BackendIO(self._store, request_id=node.node_id),
            sample=sample,
            backends=self._broker.handle,
            cancel=cancel,
            emit=self._emitter(node),
            values=self._ports.get,
        )
        produced = dict(await operator.run(inputs, context))
        finished = utc_now()
        # Ports are installed *before* the completion callback runs, because the callback (the job
        # runner) reads the node's produced values to record candidates and evidence.
        self._ports.put(node.node_id, produced)
        ports, artifacts = self._encode_outputs(node, produced, sample)
        payload = {
            "ports": {name: encode_value(value) for name, value in ports.items()},
            "values": {name: _encode_value(value) for name, value in produced.items()},
            "backend_id": self._broker.last_backend_id,
        }
        result = NodeResult(
            node_id=node.node_id,
            operator=node.operator,
            operator_version=node.operator_version,
            status=NodeStatus.SUCCEEDED,
            outputs=ports,
            artifacts=artifacts,
            payload=payload,
            started_at=started,
            finished_at=finished,
            duration_ms=int((time.perf_counter() - clock) * 1000),
            backend_id=self._broker.last_backend_id,
            attempt=attempt,
            determinism=node.determinism,
        )
        return ExecutionOutcome(result=result, values=dict(produced), backend_id=self._broker.last_backend_id)

    def resolve_inputs(self, node: OperationNode) -> dict[str, Any]:
        """Resolve every input binding to a live value.

        Raises:
            SkippedNode: when a bound producer has no value for the requested port, which means this
                node belongs to a branch that does not exist for this sample.
        """
        resolved: dict[str, Any] = {}
        for port, binding in node.inputs.items():
            if isinstance(binding, NodeOutput):
                if self._ports.is_skipped(binding.node_id):
                    raise SkippedNode(
                        f"input {port!r} comes from a skipped branch ({binding.node_id})",
                        node_id=node.node_id,
                    )
                if not self._ports.has(binding.node_id, binding.port):
                    raise SkippedNode(
                        f"input {port!r} has no value from {binding.node_id}",
                        node_id=node.node_id,
                    )
                resolved[port] = self._select(node, port, self._ports.get(binding.node_id, binding.port))
            elif isinstance(binding, SampleSource):
                continue  # sample values are injected through the execution context
            else:
                resolved[port] = getattr(binding, "value", None)
        return resolved

    def _select(self, node: OperationNode, port: str, value: Any) -> Any:
        """Pick one element out of a tuple-valued port when the node declares a selection.

        A per-target node reads one element of the selection's ``items`` tuple. A branch with no
        corresponding object is *skipped*, not failed: a sample with a single target object
        legitimately has nothing to do for the branches the assembler reserved.
        """
        index = node.selection.get(port)
        if index is None:
            return value
        if isinstance(value, dict) and "__skipped__" in value:
            raise SkippedNode(f"port {port!r} comes from a skipped branch", node_id=node.node_id)
        if isinstance(value, (list, tuple)):
            if index >= len(value):
                raise SkippedNode(
                    f"port {port!r} has {len(value)} element(s) but index {index} was requested",
                    node_id=node.node_id,
                    reason="selection index beyond the available objects",
                )
            return value[index]
        if index == 0:
            return value
        raise SkippedNode(
            f"port {port!r} carries a single value but index {index} was requested",
            node_id=node.node_id,
            reason="selection index beyond the available objects",
        )

    def adopt(self, node_id: str, values: Mapping[str, Any]) -> None:
        """Install port values for a node whose result was restored from a previous run."""
        self._ports.put(node_id, values)

    def restore_from_result(self, result: NodeResult) -> bool:
        """Rehydrate a node's port values from a persisted result.

        Returns:
            ``True`` when the values were restored; ``False`` when the payload carried nothing
            decodable, in which case the caller should re-execute rather than guess.
        """
        payload = result.payload.get("values")
        if not isinstance(payload, dict):
            return False
        restored: dict[str, Any] = {}
        for port, encoded in payload.items():
            restored[port] = _decode_value(encoded)
        self._ports.put(result.node_id, restored)
        return True

    # -- helpers ----------------------------------------------------------- #

    def _encode_outputs(
        self,
        node: OperationNode,
        produced: Mapping[str, Any],
        sample: SampleContext | None,
    ) -> tuple[dict[str, PortValue], list[ArtifactRef]]:
        """Turn an operator's return value into port values and a flat artifact list."""
        ports: dict[str, PortValue] = {}
        artifacts: list[ArtifactRef] = []
        for port, value in produced.items():
            ports[port] = value if isinstance(value, PortValue) else self._as_port_value(value)
            artifact = ports[port].artifact
            if artifact is not None:
                artifacts.append(artifact)
        declared = set(node.outputs)
        missing = declared - set(ports)
        if missing:
            raise OperatorFailure(
                f"operator {node.operator} did not produce declared output port(s) {sorted(missing)}",
                code=ErrorCode.PORT_UNBOUND,
                detail={"node_id": node.node_id, "missing": sorted(missing)},
            )
        if self._on_event is not None:
            self._on_event(
                "node.values",
                node,
                {
                    "ports": list(ports),
                    "artifacts": [artifact.digest for artifact in artifacts],
                    "sample_id": sample.sample_id if sample else None,
                },
            )
        return ports, artifacts

    def _as_port_value(self, value: Any) -> PortValue:
        if isinstance(value, ArtifactRef):
            return PortValue.of_artifact(value)
        if isinstance(value, BaseModel):
            return PortValue.of_payload(value.model_dump(mode="json"))
        if isinstance(value, (list, tuple)):
            return PortValue.of_payload(
                {
                    "items": [
                        item.model_dump(mode="json") if isinstance(item, BaseModel) else item
                        for item in value
                    ]
                }
            )
        if isinstance(value, dict):
            return PortValue.of_payload(value)
        return PortValue.of_payload({"value": value})

    def _skipped(self, node: OperationNode, started: datetime, clock: float, reason: str) -> ExecutionOutcome:
        finished = utc_now()
        result = NodeResult(
            node_id=node.node_id,
            operator=node.operator,
            operator_version=node.operator_version,
            status=NodeStatus.SKIPPED,
            started_at=started,
            finished_at=finished,
            duration_ms=int((time.perf_counter() - clock) * 1000),
            error_class=SkippedNode.__name__,
            error_code=ErrorCode.DEPENDENCY_FAILED.value,
            error_message=f"skipped: {reason}",
        )
        return ExecutionOutcome(result=result)

    def _emitter(self, node: OperationNode) -> Callable[[OperatorEvent], None]:
        def emit(event: OperatorEvent) -> None:
            if self._on_event is not None:
                self._on_event(f"operator.{event.kind}", node, dict(event.detail))

        return emit


def _encode_value(value: Any) -> dict[str, Any]:
    try:
        return encode_value(value)
    except VidlinerError:
        return {"tag": "mapping", "payload": {"repr": repr(value)[:512]}}


def _decode_value(encoded: Any) -> Any:
    """Decode one persisted port value.

    A failed decode raises: turning it into ``None`` would look like an absent upstream port and
    would silently skip a branch, which is exactly the kind of quiet corruption the pipeline must
    never produce.
    """
    if not isinstance(encoded, dict) or "tag" not in encoded:
        raise InfrastructureFailure(
            "persisted port value has no type tag",
            code=ErrorCode.PORT_TYPE_MISMATCH,
            detail={"value": str(encoded)[:120]},
        )
    return decode_value(encoded)
