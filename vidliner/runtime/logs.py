"""Structured logging.

Every event is one JSON object on one line. Two sinks are supported and both are used:

* ``events.jsonl`` inside the job's run directory, so a run can be replayed after the fact;
* an optional console sink, which prints a short human line (and the full JSON when verbose).

Redaction happens at the sink boundary, not at the call site, so a secret cannot reach a log file by
someone forgetting to redact it. The resolver's known secrets are scrubbed from every serialized
event.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from vidliner.core.results import utc_now
from vidliner.runtime.secrets import CredentialResolver

__all__ = ["EventLog", "NullEventLog", "read_events"]


class NullEventLog:
    """An event log that discards events. Used by tests and by ``--quiet`` runs."""

    def emit(self, name: str, **fields: Any) -> None:
        """Discard one event."""
        del name, fields

    def close(self) -> None:
        """No-op."""


class EventLog:
    """Append-only JSON-lines event log with optional console mirroring."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        verbose: bool = False,
        console: TextIO | None = None,
        mirror: bool = False,
        resolver: CredentialResolver | None = None,
        clock: Callable[[], Any] = utc_now,
    ) -> None:
        self._path = path
        self._verbose = verbose
        self._console = console
        self._mirror = mirror
        self._resolver = resolver
        self._clock = clock
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, name: str, **fields: Any) -> None:
        """Write one event.

        The event *name* is a positional parameter rather than a key in ``fields`` so a caller cannot
        accidentally shadow it with payload data.
        """
        payload: dict[str, Any] = {
            "ts": self._clock().isoformat(),
            "event": name,
            **{key: _jsonable(value) for key, value in fields.items() if key != "event"},
        }
        payload = self._redact(payload)
        if self._path is not None:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        if self._mirror and self._console is not None:
            self._print(payload)

    def _redact(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._resolver is None:
            return payload
        redacted = self._resolver.redact_mapping(payload)
        if isinstance(redacted, dict):
            return redacted
        return payload

    def _print(self, payload: dict[str, Any]) -> None:
        assert self._console is not None
        if self._verbose:
            self._console.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            return
        summary = " ".join(
            f"{key}={value}"
            for key, value in payload.items()
            if key not in {"ts"} and isinstance(value, (str, int, float, bool))
        )
        self._console.write(f"[{payload['event']}] {summary}\n")

    def close(self) -> None:
        """Flush the console mirror."""
        if self._console is not None:
            self._console.flush()

    @staticmethod
    def console(verbose: bool = False) -> EventLog:
        """An event log that mirrors to stderr."""
        return EventLog(None, verbose=verbose, console=sys.stderr, mirror=True)


def read_events(path: Path) -> list[dict[str, Any]]:
    """Read an event log back, skipping any truncated trailing line."""
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            events.append(json.loads(stripped))
        except json.JSONDecodeError:
            continue
    return events


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value"):  # StrEnum and friends
        return value.value
    return str(value)
