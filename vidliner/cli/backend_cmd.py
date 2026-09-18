"""``vidliner backend``: inspect and check the runtime profile's backends."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import typer

from vidliner.capabilities.names import (
    BUILTIN_CAPABILITIES,
    CAPABILITY_GROUPS,
    KNOWN_CAPABILITIES,
    describe_capability,
    is_builtin_capability,
)
from vidliner.cli.common import Output, fail, resolve_workspace
from vidliner.core.errors import ValidationFailure
from vidliner.pipeline.service import Session

__all__ = ["app"]

app = typer.Typer(no_args_is_help=True)


@app.command("list")
def list_command(
    workspace: Path | None = typer.Option(None, "--workspace"),
    runtime: Path | None = typer.Option(None, "--runtime"),
    capabilities: bool = typer.Option(
        False, "--capabilities", help="List the capability vocabulary instead."
    ),
    quiet: bool = typer.Option(False, "--quiet"),
    json_mode: bool = typer.Option(False, "--json"),
) -> None:
    """List configured backends, or the capabilities VidLiner knows about."""
    output = Output(json_mode=json_mode, quiet=quiet)
    if capabilities:
        payload = {
            group: [{"capability": name, "description": describe_capability(name)} for name in names]
            for group, names in CAPABILITY_GROUPS.items()
        }
        output.payload(payload)
        if json_mode:
            return
        for group, names in CAPABILITY_GROUPS.items():
            output.line(f"{group}:")
            for name in names:
                output.line(f"  {name:44s} {describe_capability(name)}")
        return

    try:
        session = Session.open(resolve_workspace(workspace), runtime_path=runtime)
    except ValidationFailure as exc:
        raise typer.Exit(code=fail(exc, output=output)) from exc
    profile = session.profile
    registry = session.registry
    rows: list[dict[str, Any]] = []
    for name in profile.backend_names():
        spec = profile.backends[name]
        declared = registry.declared_capabilities(name)
        bindings = [capability for capability, backend in profile.bindings.items() if backend == name]
        rows.append(
            {
                "backend": name,
                "use": spec.use,
                "enabled": spec.enabled,
                "declared": list(declared),
                "bound_to": bindings,
                "capacity": spec.capacity.limit,
                "external": spec.estimates.external,
                "unit_cost": spec.estimates.unit_cost,
                "credentials": sorted(spec.credentials),
                "device": profile.device_for(name),
            }
        )
    session.close()
    output.payload({"profile": profile.profile, "backends": rows})
    if json_mode:
        return
    output.line(f"profile {profile.profile}  device {profile.device}")
    output.table(
        ["backend", "device", "capacity", "external", "cost", "bound capabilities"],
        [
            [
                row["backend"],
                str(row["device"]),
                str(row["capacity"]),
                "yes" if row["external"] else "no",
                f"{row['unit_cost']:.4f}" if row["unit_cost"] else "-",
                ", ".join(row["bound_to"]) or "-",
            ]
            for row in rows
        ],
        fallback="no backends configured",
    )


@app.command("check")
def check_command(
    workspace: Path | None = typer.Option(None, "--workspace"),
    runtime: Path | None = typer.Option(None, "--runtime"),
    probe: bool = typer.Option(
        True, "--probe/--no-probe", help="Call each backend's probe (never a generative call)."
    ),
    quiet: bool = typer.Option(False, "--quiet"),
    json_mode: bool = typer.Option(False, "--json"),
) -> None:
    """Probe every configured backend and report which capabilities are covered."""
    output = Output(json_mode=json_mode, quiet=quiet)
    try:
        session = Session.open(resolve_workspace(workspace), runtime_path=runtime)
    except ValidationFailure as exc:
        raise typer.Exit(code=fail(exc, output=output)) from exc
    registry = session.registry
    probes: dict[str, Any] = asyncio.run(registry.probe_all()) if probe else {}
    index = registry.capability_index(instantiate=False) if not probe else _index_from_probes(probes)
    # A capability the pipeline measures itself needs no backend, so it is reported as satisfied by
    # "builtin" rather than as uncovered — otherwise `backend check` would flag a healthy profile.
    coverage = {
        capability: (["builtin"] if is_builtin_capability(capability) else index.get(capability, []))
        for capability in KNOWN_CAPABILITIES
    }
    uncovered = sorted(capability for capability, backends in coverage.items() if not backends)
    session.close()

    payload = {
        "profile": (runtime and str(runtime)) or None,
        "probes": {
            name: {
                "backend_id": result.backend_id,
                "version": result.version,
                "health": result.health,
                "capabilities": list(result.capabilities),
                "device": result.device,
                "determinism": result.determinism.value,
                "safe_to_retry": result.safe_to_retry,
                "external": result.external,
                "message": result.message,
            }
            for name, result in probes.items()
        },
        "coverage": coverage,
        "uncovered_capabilities": uncovered,
    }
    output.payload(payload)
    if json_mode:
        return
    if not probes:
        output.line("probe disabled; reporting declared capabilities only")
        return
    for name, result in probes.items():
        mark = "ok " if result.health == "ready" else ("warn" if result.health == "degraded" else "FAIL")
        output.line(f"{mark} {name:22s} v{result.version:8s} {result.health:11s} {result.message}")
        for capability in result.capabilities:
            output.line(f"      -> {capability}")
    if uncovered:
        output.line("")
        output.line("capabilities with no backend:")
        for capability in uncovered:
            output.line(f"  {capability:44s} {describe_capability(capability)}")
    failures = [name for name, result in probes.items() if result.health == "unavailable"]
    if failures:
        output.line("")
        output.line(f"{len(failures)} backend(s) unavailable: {', '.join(failures)}")
        raise typer.Exit(code=1)


def builtin_capabilities() -> tuple[str, ...]:
    """Capabilities the pipeline measures itself, with no backend involved."""
    return BUILTIN_CAPABILITIES


def _index_from_probes(probes: dict[str, Any]) -> dict[str, list[str]]:
    """Map each advertised capability to the backends that reported it."""

    index: dict[str, list[str]] = {}
    for name, result in probes.items():
        for capability in getattr(result, "capabilities", ()):
            index.setdefault(str(capability), []).append(name)
    return index
