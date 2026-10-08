"""Versioned, atomic canvas edit requests for browser, CLI and server hosts."""

from typing import Annotated, Literal

from pydantic import Field, JsonValue

from vidliner.domain.workflow import ExternalInput, Position, WorkflowEdge, WorkflowModel, WorkflowNode


class AddNode(WorkflowModel):
    """Add a catalogue-derived node to the transaction's final graph."""

    op: Literal["add_node"] = "add_node"
    node: WorkflowNode


class RemoveNode(WorkflowModel):
    """Remove a node and its incident edges; rewire required inputs in the same batch."""

    op: Literal["remove_node"] = "remove_node"
    node_id: str


class ConfigureNode(WorkflowModel):
    """Replace one node's operator configuration."""

    op: Literal["configure_node"] = "configure_node"
    node_id: str
    config: dict[str, JsonValue]


class MoveNode(WorkflowModel):
    """Set presentation coordinates without changing execution semantics."""

    op: Literal["move_node"] = "move_node"
    node_id: str
    position: Position


class Connect(WorkflowModel):
    """Connect a declared source output to a target input."""

    op: Literal["connect"] = "connect"
    edge: WorkflowEdge


class Disconnect(WorkflowModel):
    """Remove an existing connection."""

    op: Literal["disconnect"] = "disconnect"
    edge: WorkflowEdge


class BindInput(WorkflowModel):
    """Bind one input to an external role or JSON constant."""

    op: Literal["bind_input"] = "bind_input"
    node_id: str
    port: str
    binding: ExternalInput


class UnbindInput(WorkflowModel):
    """Remove an external input binding and its selection policy."""

    op: Literal["unbind_input"] = "unbind_input"
    node_id: str
    port: str


Edit = Annotated[
    AddNode | RemoveNode | ConfigureNode | MoveNode | Connect | Disconnect | BindInput | UnbindInput,
    Field(discriminator="op"),
]


class WorkflowEditRequest(WorkflowModel):
    """Apply a complete transaction against the precise draft the caller reviewed."""

    schema_version: Literal["vidliner.workflow-edit/v1"] = "vidliner.workflow-edit/v1"
    expected_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    edits: tuple[Edit, ...] = Field(min_length=1)
