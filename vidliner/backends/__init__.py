"""The built-in, dependency-free backends.

These exist so that a complete job — ingest, detect, segment, plan, replace, refine, evaluate,
annotate, export — runs on a fresh checkout with no downloaded model and no paid API. They are
honest implementations, not stubs:

* :mod:`~vidliner.backends.heuristic.detector` finds foreground regions by saliency;
* :mod:`~vidliner.backends.heuristic.segmentation` grows a mask from a box prompt;
* :mod:`~vidliner.backends.heuristic.scene` measures orientation, lighting, ground contact;
* :mod:`~vidliner.backends.heuristic.planner` applies the preservation rules from the recipe;
* :mod:`~vidliner.backends.fake.replacement` produces deterministic, seed-driven candidate images;
* :mod:`~vidliner.backends.local.evaluator` measures semantics and background preservation;
* :mod:`~vidliner.backends.local.refiner` cleans, blends, and harmonises;
* :mod:`~vidliner.backends.local.fingerprint` produces perceptual hashes;
* :mod:`~vidliner.backends.http_replacement` is a generic adapter for any HTTP image-edit service.

A production deployment swaps in real models by editing :mod:`vidliner.runtime.profile` — no core,
operator, or recipe change is involved.
"""

from __future__ import annotations

__all__: list[str] = []
