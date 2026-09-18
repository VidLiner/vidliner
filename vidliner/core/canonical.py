"""Canonical serialization and content hashing.

Everything that needs a stable identity — node ids, cache keys, recipe hashes, artifact digests —
goes through :func:`canonical_json`. The rules are deliberately boring so that two processes with
different dict insertion orders, different float formatting, or different time zones still agree:

* mapping keys are sorted lexicographically;
* separators are ``","`` and ``":"`` with no whitespace;
* ``None`` values inside mappings are dropped, ``None`` inside sequences becomes ``null``;
* ``float`` values are emitted with the shortest representation that round-trips;
* ``-0.0`` is normalised to ``0.0`` so it cannot produce a second identity;
* ``datetime`` values are ISO-8601 in UTC with a trailing ``Z``;
* ``Enum`` values use their ``value``;
* Pydantic models are dumped in JSON mode before encoding.

Content digests additionally carry a short type tag, so a mask PNG and an annotation JSON with the
same bytes cannot collide in the artifact catalogue.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel

__all__ = [
    "DIGEST_ALGORITHM",
    "canonical_bytes",
    "canonical_json",
    "digest_bytes",
    "digest_file",
    "digest_json",
    "object_digest",
    "short_digest",
]

DIGEST_ALGORITHM: Final = "sha256"
_HEX_CHARS: Final = set("0123456789abcdef")


def _normalise(value: Any) -> Any:
    """Recursively convert ``value`` into a JSON-encodable form with canonical ordering."""
    if isinstance(value, BaseModel):
        return _normalise(value.model_dump(mode="json", exclude_none=False))
    if isinstance(value, Enum):
        return _normalise(value.value)
    if isinstance(value, datetime):
        return _isoformat(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("canonical JSON cannot encode NaN or Infinity")
        return 0.0 if value == 0 else value
    if isinstance(value, int):
        return value
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                key = str(key)
            if item is None:
                continue
            out[key] = _normalise(item)
        return dict(sorted(out.items()))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    if isinstance(value, (set, frozenset)):
        return sorted(_normalise(item) for item in value)
    if isinstance(value, Sequence):
        return [_normalise(item) for item in value]
    raise TypeError(f"value of type {type(value).__name__} is not canonically serializable")


def _isoformat(value: datetime) -> str:
    """Render a datetime as ISO-8601 UTC with a ``Z`` suffix."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    """Serialize ``value`` to its canonical JSON string."""
    return json.dumps(_normalise(value), separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_bytes(value: Any) -> bytes:
    """Serialize ``value`` to canonical UTF-8 bytes."""
    return canonical_json(value).encode("utf-8")


def digest_bytes(data: bytes) -> str:
    """SHA-256 of raw bytes, lower-case hex."""
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, read in chunks so large media does not have to fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value: Any) -> str:
    """SHA-256 of the canonical JSON form of ``value``."""
    return digest_bytes(canonical_bytes(value))


def object_digest(prefix: str, payload: Any, length: int = 16) -> str:
    """Derive a stable identifier from a prefix and a payload.

    Args:
        prefix: short identifier namespace, e.g. ``"s_"``.
        payload: any canonically serializable value.
        length: number of hex characters to keep from the digest.

    Returns:
        ``prefix`` followed by the first ``length`` hex characters of the payload digest.
    """
    if not 4 <= length <= 64:
        raise ValueError("identifier digest length must be between 4 and 64")
    return f"{prefix}{digest_json(payload)[:length]}"


def short_digest(digest: str, length: int = 12) -> str:
    """Abbreviate a digest for display, validating that it looks like one."""
    if not set(digest.lower()) <= _HEX_CHARS:
        raise ValueError(f"not a hexadecimal digest: {digest!r}")
    return digest[:length]
