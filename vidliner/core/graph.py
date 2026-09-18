"""The operation graph: the compiled form of a recipe.

The graph is a pure data structure. It knows nothing about execution, backends, or files. It can be
validated, serialized, diffed between runs, and costed without executing anything, which is what
makes ``plan`` and ``--dry-run`` possible.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Protocol, runtime_checkable

from vidliner.core.canonical import digest_json
from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.domain.enums import Determinism, PortType, StageName

__all__ = [
    "ArtifactSource",
    "ConstantValue",
    "NodeOutput",
    "OperationGraph",
    "OperationNode",
    "OperatorDescriptor",
    "PortBinding",
    "RetryPolicy",
    "SampleSource",
    "port_types_compatible",
]


@runtime_checkable
class OperatorDescriptor(Protocol):
    """The part of an operator's declaration the graph validator needs.

    Declared here rather than imported from :mod:`vidliner.operators` so that ``core`` keeps no
    dependency on the operator layer: the pipeline supplies the catalogue, the graph only consumes
    the shape it validates against.
    """

    @property
    def inputs(self) -> Mapping[str, Any]:
        """Declared input ports, keyed by port name."""
        ...

    @property
    def config_model(self) -> type[Any]:
        """The Pydantic model an operator validates its node config with."""
        ...


DEFAULT_MAX_ATTEMPTS: Final = 1
DEFAULT_BACKOFF_S: Final = 0.5


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How many times a node may be attempted and how long to wait between attempts.

    Attributes:
        attempts: total attempts including the first. ``1`` means "no retry", which is the default
            for generative operations because a retry may produce a different image and cost money.
        backoff_s: delay before the second attempt.
        multiplier: exponential growth factor for subsequent delays.
        jitter: fraction of the delay randomised with the node's seed, to avoid synchronised
            retries against an external service.
        retry_on: error codes that are retryable. An empty tuple means "retry anything the backend
            declared safe".
    """

    attempts: int = DEFAULT_MAX_ATTEMPTS
    backoff_s: float = DEFAULT_BACKOFF_S
    multiplier: float = 2.0
    jitter: float = 0.0
    retry_on: tuple[ErrorCode, ...] = ()

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("retry attempts must be at least 1")
        if self.backoff_s < 0:
            raise ValueError("retry backoff must not be negative")
        if self.multiplier < 1:
            raise ValueError("retry multiplier must be at least 1")
        if not 0.0 <= self.jitter <= 1.0:
            raise ValueError("retry jitter must be between 0 and 1")

    @property
    def retries(self) -> int:
        """Number of retries after the first attempt."""
        return self.attempts - 1

    def delay_for(self, attempt: int, seed: int) -> float:
        """Delay in seconds before ``attempt`` (1-based), deterministically jittered."""
        if attempt <= 1:
            return 0.0
        base = self.backoff_s * (self.multiplier ** (attempt - 2))
        if self.jitter:
            from vidliner.core.seedtree import derive_seed

            unit = derive_seed(seed, "retry", attempt) / float(2**63 - 1)
            base *= 1.0 + self.jitter * (unit * 2.0 - 1.0)
        return max(0.0, base)

    def allows(self, code: ErrorCode) -> bool:
        """Whether the policy permits retrying an error with ``code``."""
        return not self.retry_on or code in self.retry_on


@dataclass(frozen=True, slots=True)
class NodeOutput:
    """A binding to one output port of another node."""

    node_id: str
    port: str

    def encode(self) -> str:
        """A stable string form, used in the plan document and in diagnostics."""
        return f"node:{self.node_id}#{self.port}"

    @property
    def referenced_node(self) -> str:
        """The node this binding reads from."""
        return self.node_id


@dataclass(frozen=True, slots=True)
class SampleSource:
    """A binding to a value the job injects for one dataset sample."""

    role: str

    def encode(self) -> str:
        """A stable string form, used in the plan document and in diagnostics."""
        return f"sample:{self.role}"

    @property
    def referenced_node(self) -> None:
        """Sample sources depend on nothing inside the graph."""
        return None


@dataclass(frozen=True, slots=True)
class ArtifactSource:
    """A binding to a job-level artifact supplied on the command line (for example a reference image)."""

    role: str

    def encode(self) -> str:
        """A stable string form, used in the plan document and in diagnostics."""
        return f"artifact:{self.role}"

    @property
    def referenced_node(self) -> None:
        """Job-level artifacts depend on nothing inside the graph."""
        return None


@dataclass(frozen=True, slots=True)
class ConstantValue:
    """A binding to a literal validated against the consumer's declared port type."""

    value: Any

    def encode(self) -> str:
        """A stable string form: a digest of the literal, so two equal constants read alike."""
        return f"const:{digest_json(self.value)[:12]}"

    @property
    def referenced_node(self) -> None:
        """Constants depend on nothing."""
        return None


PortBinding = NodeOutput | SampleSource | ArtifactSource | ConstantValue


@dataclass(frozen=True, slots=True)
class OperationNode:
    """One operator instance in the graph.

    Attributes:
        node_id: deterministic identity (see :func:`vidliner.core.identity.node_id`).
        operator: registered operator name, e.g. ``"generate.replacement"``.
        operator_version: semantic version of the operator implementation.
        stage: pipeline stage, used for reporting and for ordering diagnostics.
        inputs: port name to binding.
        outputs: port name to declared type.
        config: validated configuration; must be canonically serializable.
        needs: capability names the runtime must bind for this node.
        name: operator class name, used for display.
        determinism: how repeatable the node is.
        cacheable: whether a successful result may be reused by cache key.
        retry: retry policy.
        timeout_s: per-attempt timeout, or ``None`` for the engine default.
        max_parallelism: how many instances of this node may run at once (usually 1).
        lineage: identity path used for seed derivation and human-readable reports.
        description: one-line explanation shown by ``plan``.
    """

    node_id: str
    operator: str
    operator_version: str
    stage: StageName
    inputs: dict[str, PortBinding] = field(default_factory=dict)
    outputs: dict[str, PortType] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    needs: tuple[str, ...] = ()
    name: str = ""
    determinism: Determinism = Determinism.DETERMINISTIC
    cacheable: bool = True
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    timeout_s: float | None = None
    max_parallelism: int = 1
    lineage: tuple[str, ...] = ()
    selects: tuple[tuple[str, int], ...] = ()
    """Ports whose value must be picked out of a producer's tuple, with the index to pick.

    A per-target node consumes one element of the selection's ``items`` tuple. The index is part of
    the node's declared inputs rather than an implicit position, so it appears in the plan document
    and a plan can be replayed from it.
    """
    description: str = ""

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("operation node requires an id")
        if not self.operator:
            raise ValueError(f"operation node {self.node_id} requires an operator name")
        if self.max_parallelism < 1:
            raise ValueError(f"operation node {self.node_id} requires max_parallelism >= 1")
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError(f"operation node {self.node_id} requires a positive timeout")
        if not self.outputs:
            raise ValueError(f"operation node {self.node_id} must declare at least one output port")

    @property
    def selection(self) -> dict[str, int]:
        """Port name to index for every port that selects out of a tuple."""
        return dict(self.selects)

    @property
    def lineage_label(self) -> str:
        """The lineage path joined for display."""
        return "/".join(self.lineage) if self.lineage else "<job>"

    @property
    def config_hash(self) -> str:
        """Digest of the node configuration, used in the cache key and in evidence records."""
        return digest_json(self.config)

    @property
    def dependency_ids(self) -> tuple[str, ...]:
        """Node ids this node reads from, de-duplicated and ordered."""
        seen: dict[str, None] = {}
        for binding in self.inputs.values():
            node = binding.referenced_node
            if node is not None:
                seen.setdefault(node, None)
        return tuple(seen)

    def input_bindings(self) -> dict[str, PortBinding]:
        """Return a copy of the input bindings."""
        return dict(self.inputs)


@dataclass(frozen=True, slots=True)
class OperationGraph:
    """A validated-able collection of operation nodes for one job."""

    nodes: tuple[OperationNode, ...]
    job_lineage: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("operation graph must contain at least one node")

    def node(self, node_id: str) -> OperationNode:
        """Look up a node by id.

        Raises:
            KeyError: if no such node exists.
        """
        for candidate in self.nodes:
            if candidate.node_id == node_id:
                return candidate
        raise KeyError(f"unknown node {node_id}")

    def index(self) -> dict[str, OperationNode]:
        """Return a node-id keyed mapping."""
        return {node.node_id: node for node in self.nodes}

    def by_stage(self) -> dict[StageName, list[OperationNode]]:
        """Group nodes by pipeline stage, preserving insertion order."""
        grouped: dict[StageName, list[OperationNode]] = {}
        for node in self.nodes:
            grouped.setdefault(node.stage, []).append(node)
        return grouped

    def capabilities(self) -> tuple[str, ...]:
        """Every capability the graph requires, sorted and de-duplicated."""
        return tuple(sorted({capability for node in self.nodes for capability in node.needs}))

    def dependents(self) -> dict[str, list[str]]:
        """Map each node id to the ids of nodes that consume its outputs."""
        mapping: dict[str, list[str]] = {node.node_id: [] for node in self.nodes}
        for node in self.nodes:
            for dependency in node.dependency_ids:
                if dependency in mapping:
                    mapping[dependency].append(node.node_id)
        return mapping

    def validate(self) -> None:
        """Check structural integrity.

        Raises:
            ValidationFailure: on duplicate ids, unknown dependencies, cycles, port type
                mismatches, or missing input bindings for a declared requirement.
        """
        index = self.index()
        if len(index) != len(self.nodes):
            duplicates = sorted(
                {
                    node.node_id
                    for node in self.nodes
                    if sum(1 for n in self.nodes if n.node_id == node.node_id) > 1
                }
            )
            raise ValidationFailure(
                f"operation graph contains duplicate node ids: {', '.join(duplicates)}",
                code=ErrorCode.GRAPH_INVALID,
            )

        for node in self.nodes:
            for port, binding in node.inputs.items():
                dependency = binding.referenced_node
                if dependency is None:
                    continue
                if dependency not in index:
                    raise ValidationFailure(
                        f"node {node.node_id} input {port!r} references unknown node {dependency}",
                        code=ErrorCode.GRAPH_INVALID,
                    )
                produced = index[dependency].outputs
                if isinstance(binding, NodeOutput) and binding.port not in produced:
                    raise ValidationFailure(
                        f"node {node.node_id} input {port!r} references port {binding.port!r} "
                        f"which node {dependency} does not produce",
                        code=ErrorCode.PORT_UNBOUND,
                    )

        self._validate_acyclic(index)

    def _validate_acyclic(self, index: dict[str, OperationNode]) -> None:
        in_degree = {node_id: len(node.dependency_ids) for node_id, node in index.items()}
        ready = deque(sorted(node_id for node_id, degree in in_degree.items() if degree == 0))
        visited = 0
        dependents = self.dependents()
        while ready:
            current = ready.popleft()
            visited += 1
            for dependent in dependents[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    ready.append(dependent)
        if visited != len(index):
            remaining = sorted(node_id for node_id, degree in in_degree.items() if degree > 0)
            raise ValidationFailure(
                f"operation graph contains a cycle involving: {', '.join(remaining)}",
                code=ErrorCode.GRAPH_CYCLE,
            )

    def validate_against_operators(self, describe: Callable[[str, str | None], OperatorDescriptor]) -> None:
        """Validate port types and node configs against the operator catalogue.

        This is a *pipeline-level* check, deliberately not part of :meth:`validate`: the graph
        structure must be checkable without importing the operator catalogue, while a job's
        pre-flight must check that every node's config and port wiring match the operator that will
        actually run it.

        Args:
            describe: a callable ``(operator_name, version) -> OperatorSpec``. It is injected rather
                than imported so that ``core`` keeps no dependency on ``operators`` — the pipeline
                layer supplies the catalogue.

        Raises:
            ValidationFailure: on a missing input, a port type mismatch, an unknown operator, or an
                invalid node config.
        """
        if not callable(describe):  # pragma: no cover - guarded by the caller's signature
            raise ValidationFailure(
                "graph validation needs a callable that resolves operator descriptors",
                code=ErrorCode.OPERATOR_UNKNOWN,
            )
        index = self.index()
        for node in self.nodes:
            descriptor = describe(node.operator, node.operator_version)
            for port, spec in descriptor.inputs.items():
                binding = node.inputs.get(port)
                if binding is None:
                    if spec.required:
                        raise ValidationFailure(
                            f"node {node.node_id} ({node.operator}) is missing required input {port!r}",
                            code=ErrorCode.PORT_UNBOUND,
                        )
                    continue
                if isinstance(binding, NodeOutput):
                    source = index[binding.node_id]
                    produced = source.outputs.get(binding.port, PortType.ANY)
                    if not port_types_compatible(spec.port_type, produced):
                        raise ValidationFailure(
                            f"node {node.node_id} input {port!r} expects {spec.port_type.value} but "
                            f"receives {produced.value} from {binding.node_id}",
                            code=ErrorCode.PORT_TYPE_MISMATCH,
                        )
            descriptor.config_model.model_validate(node.config)

    def topological_order(self) -> list[OperationNode]:
        """Return nodes in a deterministic topological order (stage order, then node id)."""
        index = self.index()
        in_degree = {node_id: len(node.dependency_ids) for node_id, node in index.items()}
        dependents = self.dependents()
        ready = [node_id for node_id, degree in in_degree.items() if degree == 0]
        order: list[OperationNode] = []
        while ready:
            ready.sort(key=lambda node_id: (_stage_rank(index[node_id].stage), node_id))
            current = ready.pop(0)
            order.append(index[current])
            for dependent in sorted(dependents[current]):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    ready.append(dependent)
        if len(order) != len(index):
            raise ValidationFailure("operation graph could not be ordered", code=ErrorCode.GRAPH_CYCLE)
        return order

    def describe_stages(self) -> list[tuple[StageName, int]]:
        """Stage name and node count, in stage order."""
        grouped = self.by_stage()
        ordered = sorted(grouped.items(), key=lambda item: _stage_rank(item[0]))
        return [(stage, len(nodes)) for stage, nodes in ordered]


_STAGE_ORDER: Final[tuple[StageName, ...]] = (
    StageName.INGEST,
    StageName.DETECT,
    StageName.SELECT,
    StageName.SEGMENT,
    StageName.SCENE,
    StageName.PLAN,
    StageName.GENERATE,
    StageName.REFINE,
    StageName.VERIFY,
    StageName.EVALUATE,
    StageName.ANNOTATE,
    StageName.GATE,
    StageName.EXPORT,
)


def _stage_rank(stage: StageName) -> int:
    return _STAGE_ORDER.index(stage)


def port_types_compatible(expected: PortType, actual: PortType) -> bool:
    """Whether a produced port type may feed an expected port type.

    ``ANY`` matches everything in both directions; identical types match. No implicit conversions
    exist, because an implicit conversion between pipeline stages is exactly the kind of thing that
    silently corrupts a label.
    """
    if expected is PortType.ANY or actual is PortType.ANY:
        return True
    return expected is actual
