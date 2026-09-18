"""``vidliner init``: create a workspace and a starter runtime profile."""

from __future__ import annotations

from pathlib import Path

import typer

from vidliner.cli.common import Output, fail, resolve_workspace
from vidliner.core.errors import ValidationFailure
from vidliner.pipeline.service import write_default_profile
from vidliner.runtime.profile import default_profile
from vidliner.storage.state import StateStore
from vidliner.storage.workspace import Workspace

__all__ = ["init_command"]


def init_command(
    directory: Path | None = typer.Argument(None, help="Directory to initialise (default: the workspace)."),
    force: bool = typer.Option(False, "--force", help="Replace an existing runtime profile."),
    device: str = typer.Option("cpu", "--device", help="Device hint: cpu, cuda, mps, or auto."),
    quiet: bool = typer.Option(False, "--quiet", help="Print nothing on success."),
    json_mode: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Create the workspace layout, a state database, and a starter runtime profile."""
    output = Output(json_mode=json_mode, quiet=quiet)
    try:
        root = directory.expanduser().resolve() if directory is not None else resolve_workspace(None)
        workspace = Workspace(root).initialize()
        state = StateStore(workspace.state_path).initialize()
        state.close()
        profile_path = workspace.root / "runtime.yaml"
        existed = profile_path.exists()
        if not existed or force:
            write_default_profile(profile_path, force=force)
        profile = default_profile(device=device)
        if device != "cpu" and profile_path.is_file():
            from vidliner.runtime.profile import load_profile

            profile = load_profile(profile_path).model_copy(update={"device": device})
    except ValidationFailure as exc:
        raise typer.Exit(code=fail(exc, output=output)) from exc

    output.payload(
        {
            "workspace": str(workspace.root),
            "state": str(workspace.state_path),
            "runtime": str(profile_path),
            "profile": profile.profile,
            "device": profile.device,
            "backends": list(profile.backend_names()),
        },
        fallback=(
            f"initialised workspace {workspace.root}\n"
            f"  state    {workspace.state_path}\n"
            f"  runtime  {profile_path}" + ("" if not existed or force else "  (kept existing)")
        ),
    )
    if json_mode:
        return
    output.line("")
    output.line("next steps:")
    output.line("  vidliner inspect ./your-images --format coco-instance")
    output.line("  vidliner recipe init car-swap.yaml")
    output.line("  vidliner plan car-swap.yaml --dry-run")
