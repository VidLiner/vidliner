"""Credential resolution and redaction.

Secrets enter the process only here. They are:

* never stored in a recipe, a manifest, or a log line;
* never written into a plan or an artifact;
* redacted by value from any string that is about to be logged or persisted.

The resolver keeps resolved values in memory for the lifetime of the run so a rate-limited service
is not asked for the same secret on every call, but it never exposes them through
:meth:`CredentialResolver.describe`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.runtime.profile import CredentialRef

__all__ = ["REDACTED", "CredentialResolver", "redact_text"]

REDACTED: Final = "<redacted>"
_MIN_SECRET_LENGTH: Final = 6


class CredentialResolver:
    """Resolves :class:`CredentialRef` objects into values, and remembers them for redaction."""

    def __init__(self, *, environ: dict[str, str] | None = None, base_dir: Path | None = None) -> None:
        self._environ = dict(os.environ if environ is None else environ)
        self._base_dir = base_dir
        self._cache: dict[str, str] = {}
        self._known_secrets: set[str] = set()

    def resolve(self, reference: CredentialRef, *, backend: str, name: str) -> str | None:
        """Resolve one credential reference.

        Args:
            reference: the reference to resolve.
            backend: backend name, used only for error messages.
            name: credential slot name, used only for error messages and caching.

        Returns:
            The secret value, or ``None`` when the reference is optional and unresolved.

        Raises:
            ValidationFailure: when a required reference cannot be resolved.
        """
        cache_key = f"{backend}:{name}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        value = self._read(reference, backend=backend, name=name)
        if value is None:
            if reference.required:
                raise ValidationFailure(
                    f"backend {backend!r} requires credential {name!r} from "
                    f"{reference.source}:{reference.name or '<inline>'} but it is not available",
                    code=ErrorCode.CREDENTIAL_MISSING,
                    detail={"backend": backend, "credential": name, "source": reference.source},
                )
            return None
        self._cache[cache_key] = value
        if len(value) >= _MIN_SECRET_LENGTH:
            self._known_secrets.add(value)
        return value

    def _read(self, reference: CredentialRef, *, backend: str, name: str) -> str | None:
        if reference.source == "env":
            return self._environ.get(reference.name)
        if reference.source == "file":
            path = Path(reference.name)
            if not path.is_absolute() and self._base_dir is not None:
                path = self._base_dir / path
            if not path.is_file():
                return None
            return path.read_text(encoding="utf-8").strip()
        if reference.source == "value":
            return reference.name or None
        raise ValidationFailure(
            f"credential source {reference.source!r} is not supported",
            code=ErrorCode.CREDENTIAL_MISSING,
            detail={"backend": backend, "credential": name},
        )

    def register_secret(self, value: str) -> None:
        """Register a value that must be redacted from logs even though it was not resolved here."""
        if len(value) >= _MIN_SECRET_LENGTH:
            self._known_secrets.add(value)

    def redact(self, text: str) -> str:
        """Replace every known secret occurrence in ``text``."""
        redacted = text
        for secret in sorted(self._known_secrets, key=len, reverse=True):
            redacted = redacted.replace(secret, REDACTED)
        return redacted

    def redact_mapping(self, payload: object) -> object:
        """Recursively redact secrets inside a JSON-like structure."""
        if isinstance(payload, dict):
            return {key: self.redact_mapping(value) for key, value in payload.items()}
        if isinstance(payload, list):
            return [self.redact_mapping(item) for item in payload]
        if isinstance(payload, str):
            return self.redact(payload)
        return payload

    def describe(self, reference: CredentialRef) -> dict[str, str]:
        """Reference description safe to print or persist."""
        return reference.redacted()

    @property
    def secret_count(self) -> int:
        """How many distinct secret values are currently known (never the values themselves)."""
        return len(self._known_secrets)


def redact_text(text: str, secrets: set[str]) -> str:
    """Redact a fixed set of secrets from ``text``."""
    redacted = text
    for secret in sorted(secrets, key=len, reverse=True):
        if secret:
            redacted = redacted.replace(secret, REDACTED)
    return redacted
