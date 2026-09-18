"""Deterministic stand-in backends.

A fake backend exists so the entire pipeline — including generation and quality evaluation — can be
exercised end to end in tests and in a first-run demo without a paid API or a downloaded model. It
is deterministic given its seed, which is what makes cache and provenance tests meaningful.
"""

from __future__ import annotations

__all__: list[str] = []
