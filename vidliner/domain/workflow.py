"""Versioned canvas documents. Execution remains the operation graph's responsibility."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from vidliner.domain.enums import Determinism, PortType, StageName
from vidliner.domain.provenance import assert_no_secrets

NonEmpty = Annotated[str, Field(min_length=1, pattern=r"\S")]


class WorkflowModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False, hide_input_in_errors=True)


class Position(WorkflowModel):
    """Finite presentation coordinates independent of execution identity."""

    x: float = 0
    y: float = 0


class SampleInput(WorkflowModel):
    """A named value injected for a dataset sample."""

    kind: Literal["sample"] = "sample"
    role: NonEmpty


class ArtifactInput(WorkflowModel):
    """A named job-level artifact supplied by the embedding application."""

    kind: Literal["artifact"] = "artifact"
    role: NonEmpty


class ConstantInput(WorkflowModel):
    """An explicit JSON literal, including null."""

    kind: Literal["constant"] = "constant"
    value: JsonValue


ExternalInput = Annotated[SampleInput | ArtifactInput | ConstantInput, Field(discriminator="kind")]


class WorkflowRetry(WorkflowModel):
    """Serializable attempt count, delays and allowed failure codes."""

    attempts: int = Field(default=1, ge=1)
    backoff_s: float = Field(default=0.5, ge=0)
    multiplier: float = Field(default=2, ge=1)
    jitter: float = Field(default=0, ge=0, le=1)
    retry_on: tuple[str, ...] = ()


class WorkflowNode(WorkflowModel):
    """One registered operator's execution fields and canvas position."""

    id: NonEmpty
    operator: NonEmpty
    operator_version: NonEmpty
    stage: StageName
    name: str = ""
    description: str = ""
    position: Position = Field(default_factory=Position)
    inputs: dict[NonEmpty, ExternalInput] = Field(default_factory=dict)
    outputs: dict[NonEmpty, PortType] = Field(min_length=1)
    config: dict[str, JsonValue] = Field(default_factory=dict)
    needs: tuple[str, ...] = ()
    determinism: Determinism = Determinism.DETERMINISTIC
    cacheable: bool = True
    retry: WorkflowRetry = Field(default_factory=WorkflowRetry)
    timeout_s: float | None = Field(default=None, gt=0)
    max_parallelism: int = Field(default=1, ge=1)
    lineage: tuple[str, ...] = ()
    selects: tuple[tuple[NonEmpty, Annotated[int, Field(ge=0)]], ...] = ()


class WorkflowEdge(WorkflowModel):
    """One producer output connected to one consumer input."""

    source: NonEmpty
    source_port: NonEmpty
    target: NonEmpty
    target_port: NonEmpty


class WorkflowDocument(WorkflowModel):
    """A plan snapshot for a canvas, with external inputs separate from connected ports.

    Backend bindings are display hints, never credentials or proof of production readiness.
    A canvas can move nodes without changing the graph or creating a second execution model.
    """

    schema_version: Literal["vidliner.workflow/v1"] = "vidliner.workflow/v1"
    name: NonEmpty
    job_lineage: tuple[str, ...] = ()
    nodes: tuple[WorkflowNode, ...] = Field(min_length=1)
    edges: tuple[WorkflowEdge, ...] = ()
    recipe_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    capability_bindings: dict[str, str] = Field(default_factory=dict)
    unmet_capabilities: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_wiring(self) -> Self:
        index = {node.id: node for node in self.nodes}
        if len(index) != len(self.nodes):
            raise ValueError("workflow contains duplicate node ids")
        bound = {(node.id, port) for node in self.nodes for port in node.inputs}
        for edge in self.edges:
            if edge.source not in index or edge.target not in index:
                raise ValueError("workflow edge references an unknown node")
            if edge.source_port not in index[edge.source].outputs:
                raise ValueError(f"unknown output port {edge.source_port!r} on {edge.source}")
            target = (edge.target, edge.target_port)
            if target in bound:
                raise ValueError(f"input {edge.target}:{edge.target_port} is bound more than once")
            bound.add(target)
        assert_no_secrets(self.model_dump(mode="json"), context="workflow")
        return self
