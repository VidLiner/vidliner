"""Quality measurement and acceptance.

* :mod:`vidliner.quality.metrics` — pure pixel measurements (SSIM, change ratio, boundary, geometry,
  artifact heuristics, annotation consistency).
* :mod:`vidliner.quality.gates` — the acceptance policy engine and its reason codes.
* :mod:`vidliner.quality.fingerprint` — perceptual hashing for duplicate control.
"""

from __future__ import annotations

from vidliner.quality.fingerprint import (
    Fingerprint,
    hamming_distance,
    perceptual_hash,
    similarity_from_distance,
)
from vidliner.quality.gates import build_outcome, evaluate_gates, metric_for_reason

__all__ = [
    "Fingerprint",
    "build_outcome",
    "evaluate_gates",
    "hamming_distance",
    "metric_for_reason",
    "perceptual_hash",
    "similarity_from_distance",
]
