"""VidLiner — object-centric synthetic data augmentation with mandatory verification.

VidLiner turns existing images (and, later, video) plus optional annotations into new,
*verified* training samples: a target object is replaced, the scene outside the target is proven
unchanged, the annotation is rebuilt from the generated pixels, and only candidates that pass an
explicit acceptance policy enter the exported dataset.

The public entry points are:

* :mod:`vidliner.domain` — typed data objects shared by every layer.
* :mod:`vidliner.core` — graph engine primitives, identity, errors, results.
* :mod:`vidliner.pipeline` — recipe loading, graph compilation, job orchestration.
* :mod:`vidliner.cli` — the ``vidliner`` command line application.

Nothing outside :mod:`vidliner.backends` may import a vendor SDK.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
