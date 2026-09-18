"""Heuristic, dependency-free implementations of the perception and planning capabilities.

These backends are the default in :func:`vidliner.runtime.profile.default_profile`: they make a
complete job runnable without downloading a model. They are *not* stubs — each one produces real
measurements — but they are baselines, and a production deployment binds trained models for the
capabilities where accuracy matters.
"""

from __future__ import annotations

__all__: list[str] = []
