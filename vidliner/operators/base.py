"""The operator framework: declared ports, capability needs, and execution context.

An operator is a small, cohesive unit of domain work with a declared contract:

* **inputs** — typed ports it consumes;
* **outputs** — typed ports it produces;
* **config** — a Pydantic model that validates the node's configuration at plan time;
* **capabilities** — the abilities it needs from the runtime, which it requests through injected
  backend factories rather than by naming a vendor;
* **determinism** — whether repeating it with the same seed reproduces its output.

The context object is deliberately explicit. An operator that needs to store an image asks the
context for a :class:`~vidliner.storage.workspace.BackendIO`; it cannot reach into the workspace,
the database, or the network on its own. That is what keeps operators testable and vendor-free.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from vidliner.core.graph import OperationNode
from vidliner.domain.enums import Determinism, PortType, StageName
from vidliner.storage.workspace import BackendIO

__all__ = [
    "ConfigT",
    "ExecutionContext",
    "InputSpec",
    "Operator",
    "OperatorEvent",
    "OperatorSpec",
    "OutputSpec",
    "SampleContext",
]

ConfigT = TypeVar("ConfigT", bound=BaseModel)

_EMPTY_CONFIG = BaseModel


@dataclass(frozen=True, slots=True)
class InputSpec:
    """One declared input port."""

    port_type: PortType
    description: str = ""
    required: bool = True


@dataclass(frozen=True, slots=True)
class OutputSpec:
    """One declared output port."""

    port_type: PortType
    description: str = ""


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    """Static description of an operator."""

    name: str
    version: str
    stage: StageName
    summary: str
    inputs: Mapping[str, InputSpec]
    outputs: Mapping[str, OutputSpec]
    config_model: type[BaseModel] = _EMPTY_CONFIG
    capabilities: tuple[str, ...] = ()
    determinism: Determinism = Determinism.DETERMINISTIC
    cacheable: bool = True
    retry_attempts: int = 1
    timeout_s: float | None = None
    max_parallelism: int = 4

    @property
    def label(self) -> str:
        """``name@version`` for display and evidence."""
        return f"{self.name}@{self.version}"

    @property
    def input_types(self) -> dict[str, PortType]:
        """Required input port name to declared type."""
        return {name: spec.port_type for name, spec in self.inputs.items() if spec.required}


@dataclass(frozen=True, slots=True)
class SampleContext:
    """The per-sample values the job injects into the graph (the ``SampleSource`` bindings)."""

    sample_id: str
    source_root: str
    source_path: str
    relative_path: str
    split: str
    media_kind: str
    root_digest: str
    source_digest: str
    image_artifact: object | None = None
    """Artifact of the decoded source frame, produced by the ingest node."""
    source_annotation: dict[str, Any] | None = None
    """Annotation in internal-IR JSON form, when the dataset supplied one."""
    source_instances: tuple[dict[str, Any], ...] = ()
    """Objects known from the source annotation, in internal-IR JSON form."""
    perceptual_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_mapping(self) -> dict[str, Any]:
        """Port-role mapping used to resolve ``SampleSource`` bindings."""
        return {
            "sample_id": self.sample_id,
            "source_root": self.source_root,
            "source_path": self.source_path,
            "relative_path": self.relative_path,
            "split": self.split,
            "media_kind": self.media_kind,
            "root_digest": self.root_digest,
            "source_digest": self.source_digest,
            "image_artifact": self.image_artifact,
            "source_annotation": self.source_annotation,
            "source_instances": self.source_instances,
            "perceptual_hash": self.perceptual_hash,
            **self.metadata,
        }


@dataclass(frozen=True, slots=True)
class OperatorEvent:
    """A structured event an operator can emit for the run log."""

    kind: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionContext:
    """Everything an operator is given at run time.

    Attributes:
        job_id: the running job.
        node: the node being executed, including its config and lineage.
        seed: the derived seed for this node.
        io: storage access scoped to this node.
        sample: the sample-level values, when the job injected any.
        backends: factory that builds a bound backend for a capability on demand.
        cancel: set when the job has been asked to stop.
        emit: publish a structured event.
        values: resolve the runtime value produced on an upstream node's output port.
    """

    job_id: str
    node: OperationNode
    seed: int
    io: BackendIO
    sample: SampleContext | None = None
    backends: Callable[[str], Any] | None = None
    cancel: asyncio.Event | None = None
    emit: Callable[[OperatorEvent], None] | None = None
    values: Callable[[str, str], Any] | None = None

    @property
    def config(self) -> dict[str, Any]:
        """The node's validated configuration."""
        return self.node.config

    def option(self, name: str, default: Any = None) -> Any:
        """Read a configuration key with a default."""
        return self.node.config.get(name, default)

    def check_cancelled(self) -> None:
        """Raise :class:`Cancellation` when the job has been asked to stop."""
        if self.cancel is not None and self.cancel.is_set():
            from vidliner.core.errors import Cancellation

            raise Cancellation(f"node {self.node.node_id} stopped before doing work")

    def input_value(self, port: str) -> Any:
        """Read the runtime value produced on an upstream node's port.

        Raises:
            KeyError: when the port was not bound or its producer has not run.
        """
        if self.values is None:
            raise KeyError(f"no value resolver is available for port {port!r}")
        binding = self.node.inputs.get(port)
        if binding is None:
            raise KeyError(f"node {self.node.node_id} has no binding for input {port!r}")
        reference = getattr(binding, "referenced_node", None)
        if reference is None:
            raise KeyError(f"input {port!r} is not produced by another node")
        port_name = getattr(binding, "port", port)
        return self.values(reference, port_name)

    def publish(self, kind: str, **detail: Any) -> None:
        """Emit a structured event, when the engine attached an emitter."""
        if self.emit is not None:
            self.emit(OperatorEvent(kind=kind, detail=detail))


class Operator(ABC):
    """Base class for every operator.

    An operator is constructed with no arguments, declares its contract through :attr:`spec`, and
    returns exactly one value per declared output port from :meth:`run`.
    """

    @property
    @abstractmethod
    def spec(self) -> OperatorSpec:
        """Static description of this operator."""

    @abstractmethod
    async def run(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        """Execute and return one produced value per declared output port.

        ``inputs`` is keyed by declared input port name and ``dict`` rather than a read-only mapping
        because a node's inputs are its own: an operator may record on them (for example attaching the
        sample context) without affecting the producer.
        """

    def validated_config(self, config: Mapping[str, Any]) -> BaseModel:
        """Validate a node configuration against this operator's declared model."""
        return self.spec.config_model.model_validate(dict(config))

    async def probe(self) -> tuple[bool, str]:
        """Report whether the operator can run in this environment.

        The default implementation reports ready; an operator that depends on an optional library
        overrides this so a pre-flight check can warn before a job starts.
        """
        return True, "ready"
