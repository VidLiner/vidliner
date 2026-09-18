"""Annotation format conversion around the internal IR.

* :mod:`vidliner.annotations.base` — the reader/writer protocols and shared conversions.
* :mod:`vidliner.annotations.coco` — COCO detection and instance segmentation.
* :mod:`vidliner.annotations.yolo` — YOLO detection and segmentation.

The IR itself lives in :mod:`vidliner.domain.annotations`; this package only moves data between the
IR and the file formats. Operators never import a format module.
"""

from __future__ import annotations

from vidliner.annotations.base import (
    AnnotationReader,
    AnnotationWriter,
    ObjectRecord,
    build_bundle,
    category_ids_from,
    converter,
    read_annotations,
    supported_formats,
    write_annotations,
)
from vidliner.annotations.coco import CocoReader, CocoWriter
from vidliner.annotations.yolo import YoloReader, YoloWriter

__all__ = [
    "AnnotationReader",
    "AnnotationWriter",
    "CocoReader",
    "CocoWriter",
    "ObjectRecord",
    "YoloReader",
    "YoloWriter",
    "build_bundle",
    "category_ids_from",
    "converter",
    "read_annotations",
    "supported_formats",
    "write_annotations",
]
