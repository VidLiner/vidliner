"""Local implementations that need only numpy and Pillow.

These cover the parts of the pipeline that must never depend on a generative service: metric
measurement, mask refinement, compositing, harmonisation, and perceptual fingerprinting.
"""

from __future__ import annotations

__all__: list[str] = []
