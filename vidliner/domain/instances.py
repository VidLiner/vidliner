"""Detected and segmented objects.

``ObjectInstance`` is the unit the whole pipeline is organised around: the requirement is explicit
that the smallest execution unit is the *tracked object instance*, not the frame. For stills the
frame index is always ``0``, so the same model serves the video phase unchanged.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vidliner.domain.masks import MaskRef, PolygonMask
from vidliner.domain.shapes import BoundingBox, ImageShape

__all__ = ["ObjectInstance", "ObjectItem", "ObjectSelection"]


class ObjectInstance(BaseModel):
    """One object found in one frame.

    Attributes:
        object_id: deterministic identity derived from the asset, frame, class, box, and score.
        class_name: the label the detector/annotator assigned.
        bbox: pixel-space box, clipped to the image when built by an operator.
        score: detector or segmenter confidence in ``[0, 1]``.
        mask_ref: stored pixel-level mask, when segmentation has run.
        polygon: vector outline, when available. Kept alongside the mask because exporters need it
            and because a polygon survives format conversion better than a raster.
        attributes: free-form measured properties (``pose``, ``orientation``, ``occluded``, ...).
        source_frame: which frame and asset this instance was found in.
        track_id: identity across frames once tracking has run.
        area_px: measured pixel area, when a mask exists.
        prompt: how the instance was requested (``bbox``, ``point``, ``text``, ``auto``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    object_id: str
    class_name: str
    bbox: BoundingBox
    score: float = Field(ge=0.0, le=1.0)
    mask_ref: MaskRef | None = None
    polygon: PolygonMask | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    source_frame: str = ""
    """Frame reference in ``<asset-digest>#<frame-index>`` form."""
    track_id: str | None = None
    area_px: int | None = Field(default=None, ge=0)
    prompt: str | None = None

    @model_validator(mode="after")
    def _check_geometry(self) -> Self:
        if self.bbox.is_degenerate:
            raise ValueError(f"object {self.object_id} has a degenerate bounding box")
        if self.mask_ref is not None and self.mask_ref.is_empty and self.area_px:
            raise ValueError(f"object {self.object_id} declares a non-zero area with an empty mask")
        return self

    @property
    def frame_index(self) -> int:
        """Frame index parsed from ``source_frame``, or ``0`` when unset."""
        if "#" not in self.source_frame:
            return 0
        _, _, raw = self.source_frame.partition("#")
        try:
            return int(raw)
        except ValueError:
            return 0

    @property
    def asset_digest(self) -> str:
        """Asset digest parsed from ``source_frame``, or an empty string when unset."""
        head, _, _ = self.source_frame.partition("#")
        return head

    @property
    def coverage(self) -> float:
        """Mask coverage as a fraction of the image, or ``0.0`` without a mask."""
        return self.mask_ref.coverage if self.mask_ref is not None else 0.0

    @property
    def is_segmented(self) -> bool:
        """True when pixel-level segmentation has run for this instance."""
        return self.mask_ref is not None

    def with_mask(self, mask_ref: MaskRef, polygon: PolygonMask | None = None) -> ObjectInstance:
        """Return a copy carrying a mask (and optional polygon)."""
        return self.model_copy(update={"mask_ref": mask_ref, "polygon": polygon, "area_px": mask_ref.area_px})

    def with_attributes(self, **attributes: Any) -> ObjectInstance:
        """Return a copy with additional measured attributes."""
        merged = {**self.attributes, **attributes}
        return self.model_copy(update={"attributes": merged})

    def clipped(self, shape: ImageShape) -> ObjectInstance:
        """Return a copy whose box and polygon are clipped to the image."""
        return self.model_copy(
            update={
                "bbox": self.bbox.clipped(shape),
                "polygon": self.polygon.clipped(shape) if self.polygon is not None else None,
            }
        )


class ObjectItem(BaseModel):
    """One target object plus the context of the selection it came from.

    Per-target nodes receive one item rather than the whole selection, which is what lets the graph
    fan out by *object* rather than process every object in one node. The surrounding sample facts
    travel with the item so a per-target operator never has to look them up.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    instance: ObjectInstance
    sample_id: str
    selection_key: str
    classes_requested: tuple[str, ...] = ()
    neighbours: tuple[ObjectInstance, ...] = ()
    ordinal: int = Field(default=0, ge=0)

    @property
    def object_id(self) -> str:
        """Identity of the wrapped object."""
        return self.instance.object_id

    @property
    def class_name(self) -> str:
        """Class of the wrapped object."""
        return self.instance.class_name

    def with_instance(self, instance: ObjectInstance) -> ObjectItem:
        """Return a copy carrying a different (for example newly segmented) instance."""
        return self.model_copy(update={"instance": instance})


class ObjectSelection(BaseModel):
    """The targets the recipe chose inside one sample.

    Kept as an explicit object rather than a bare list because the *reasoning* about the selection
    (how many were dropped, and why) is part of the job's evidence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str
    selected: tuple[ObjectInstance, ...] = ()
    considered: int = Field(default=0, ge=0)
    dropped_below_score: int = Field(default=0, ge=0)
    dropped_below_area: int = Field(default=0, ge=0)
    dropped_by_class: int = Field(default=0, ge=0)
    dropped_by_cap: int = Field(default=0, ge=0)
    classes_requested: tuple[str, ...] = ()

    @property
    def count(self) -> int:
        """Number of selected objects."""
        return len(self.selected)

    @property
    def is_empty(self) -> bool:
        """True when no target object matched the recipe in this sample."""
        return not self.selected
