"""Declarative stand-ins for production backends, used to test the production guard.

The guard decides whether a job's bindings may produce training data by reading one thing: the
``demo_only`` marker on the backend, taken from the runtime profile entry and from the class itself.
Deciding that costs no import, no model download, and no network call, which is why the check can run
in ``vidliner plan`` and in pre-flight.

These classes exist so that decision can be tested for the *positive* case — a stack whose four
production-critical capabilities are served by something that does not declare itself a stand-in.
They deliberately implement nothing at all, and no test instantiates them: the guard never does
either, and a class that is only ever read as an attribute is honest about being exactly that.

Do not copy these into a real profile. A production backend is a class that actually performs the
work and says so in its capability declaration; ``docs/backends.md`` describes how to write one.
"""

from __future__ import annotations

from vidliner.capabilities.names import (
    CAP_INSTANCE_SEGMENTATION,
    CAP_OBJECT_DETECTION,
    CAP_OBJECT_REPLACEMENT,
    CAP_QUALITY_SEMANTIC,
)

__all__ = [
    "ReadyDetectorBackend",
    "ReadyEvaluatorBackend",
    "ReadyReplacementBackend",
    "ReadySegmenterBackend",
]


class ReadyDetectorBackend:
    """A detector that does not declare itself a stand-in. Implements nothing; never constructed."""

    declared_capabilities = (CAP_OBJECT_DETECTION,)


class ReadySegmenterBackend:
    """A segmenter that does not declare itself a stand-in. Implements nothing; never constructed."""

    declared_capabilities = (CAP_INSTANCE_SEGMENTATION,)


class ReadyReplacementBackend:
    """A generator that does not declare itself a stand-in. Implements nothing; never constructed."""

    declared_capabilities = (CAP_OBJECT_REPLACEMENT,)


class ReadyEvaluatorBackend:
    """A semantic evaluator that does not declare itself a stand-in. Implements nothing."""

    declared_capabilities = (CAP_QUALITY_SEMANTIC,)
