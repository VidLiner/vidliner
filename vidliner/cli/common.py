"""Shared CLI plumbing: context objects, output helpers, and error rendering.

Every command funnels its exceptions through :func:`fail`, which turns a :class:`VidlinerError` into
a one-line message plus a machine-readable code and the right exit status. That keeps the CLI honest
about *why* it stopped instead of printing a traceback for a user error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from vidliner.core.errors import ValidationFailure, VidlinerError

__all__ = [
    "EXIT_FALSE",
    "EXIT_INVALID",
    "EXIT_OK",
    "Output",
    "SessionContext",
    "fail",
    "resolve_workspace",
]

EXIT_OK = 0
EXIT_FALSE = 1
EXIT_INVALID = 2


class Output:
    """Console output that respects ``--json`` and ``--quiet``."""

    def __init__(self, *, json_mode: bool = False, quiet: bool = False) -> None:
        self.json_mode = json_mode
        self.quiet = quiet

    def line(self, text: str = "") -> None:
        """Print a line, unless quiet."""
        if not self.quiet:
            typer.echo(text)

    def payload(self, data: Any, *, fallback: str = "") -> None:
        """Print machine-readable output, or a human fallback when not in JSON mode."""
        if self.json_mode:
            typer.echo(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        elif fallback:
            self.line(fallback)

    def table(self, headers: list[str], rows: list[list[str]], *, fallback: str = "") -> None:
        """Print a simple aligned table (or JSON rows when in JSON mode)."""
        if self.json_mode:
            typer.echo(
                json.dumps(
                    [dict(zip(headers, row, strict=True)) for row in rows], indent=2, ensure_ascii=False
                )
            )
            return
        if not rows:
            self.line(fallback or "no rows")
            return
        widths = [len(header) for header in headers]
        for row in rows:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))
        header_line = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
        self.line(header_line)
        self.line("  ".join("-" * width for width in widths))
        for row in rows:
            self.line("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


@dataclass(frozen=True, slots=True)
class SessionContext:
    """The options every command shares."""

    workspace: Path
    runtime: Path | None
    json_mode: bool
    quiet: bool

    @property
    def output(self) -> Output:
        """An :class:`Output` bound to this context."""
        return Output(json_mode=self.json_mode, quiet=self.quiet)


def resolve_workspace(explicit: Path | None) -> Path:
    """Determine the workspace directory.

    Precedence: the explicit option, then ``VIDLINER_WORKSPACE``, then the current directory.
    """
    import os

    if explicit is not None:
        return explicit.expanduser().resolve()
    from_env = os.environ.get("VIDLINER_WORKSPACE")
    if from_env:
        return Path(from_env).expanduser().resolve()
    return Path.cwd().resolve()


def fail(error: BaseException, *, output: Output | None = None, code: int = EXIT_INVALID) -> int:
    """Render an error and return the exit code.

    A :class:`ValidationFailure` is a user error and exits ``2``; a quality rejection or any other
    VidLiner failure exits ``1``; an unexpected exception is re-raised so the traceback is visible
    (it is a bug, not a user mistake).
    """
    if isinstance(error, (ValidationFailure,)):
        out = output or Output()
        out.line(f"error [{error.code.value}]: {error.message}")
        if error.detail:
            detail = json.dumps(error.detail, ensure_ascii=False, default=str)
            out.line(f"  detail: {detail}")
        return EXIT_INVALID
    if isinstance(error, VidlinerError):
        out = output or Output()
        out.line(f"{error.failure_class.value} failure [{error.code.value}]: {error.message}")
        if error.detail:
            out.line(f"  detail: {json.dumps(error.detail, ensure_ascii=False, default=str)}")
        return EXIT_FALSE if code == EXIT_INVALID else code
    raise error


def workspace_options(workspace: Path | None) -> Path:
    """Resolve the workspace for a command, mirroring :func:`resolve_workspace`."""
    return resolve_workspace(workspace)
