"""The naming convention for per-candidate evidence files.

Evidence is written per candidate — a sample with three candidates keeps three annotations, three
quality reports, and three rendered images. The writer and the readers must agree on those names
exactly, or a resumed run silently finds nothing and reports zero candidates.

That failure is what this module exists to prevent. The convention lives in one place, the writer uses
it, and every reader uses it, so a name can never be spelled two different ways.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["CANDIDATE_JSON", "evidence_path", "file_key", "kinds_for", "matches_kind"]


def file_key(candidate_key: str, *, fallback: str = "candidate") -> str:
    """Turn a candidate key into a filesystem-safe token.

    ``candidate-0`` is already safe and stays as it is; anything else is sanitised rather than
    rejected, because a key comes from a recipe and a recipe must not be able to write outside the
    sample directory.
    """
    raw = candidate_key.strip() or fallback
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in raw) or fallback


def evidence_path(sample_dir: Path, kind: str, candidate_key: str, *, suffix: str = "json") -> Path:
    """Where one candidate's evidence of a given kind is written."""
    return sample_dir / f"{kind}-{file_key(candidate_key)}.{suffix}"


#: The one evidence file a candidate's identity can be recovered from without reading a port.
CANDIDATE_JSON = "candidate"


def kinds_for(candidate_key: str) -> tuple[str, ...]:
    """File-name stems a candidate's evidence may use, most specific first.

    The second form exists for evidence written before the key was always present; a reader tries the
    keyed name first so a sample with several candidates can never be confused about which one it is
    looking at.
    """
    return (file_key(candidate_key), candidate_key)


def matches_kind(path: Path, kind: str, *, suffix: str = "json") -> bool:
    """Whether a path is evidence of ``kind`` in the expected format.

    The suffix is checked as well as the prefix: a reader that accepted any extension would happily
    read a rendered preview as if it were a report.
    """
    if path.suffix != f".{suffix}":
        return False
    return path.name.startswith(f"{kind}-") or path.name == f"{kind}.{suffix}"
