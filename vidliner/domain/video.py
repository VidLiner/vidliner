"""Video-native label transforms and variant records.

This module is the Python home for the small, composable contract that made
``vidLens`` useful: a video operator changes pixels or the clock only when it
also declares how supervision moves.  The models intentionally do not know
about ffmpeg or Hypit; renderers and generators consume these records later.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vidliner.core.canonical import canonical_json, object_digest
from vidliner.core.seedtree import derive_seed
from vidliner.domain.shapes import BoundingBox, Point

__all__ = [
    "ContrastPair",
    "GeometryTransform",
    "LabelTransform",
    "PlacementTransform",
    "TimeSpan",
    "TimeTransform",
    "VideoAugmentSpec",
    "VideoOperation",
    "VideoOperatorSpec",
    "VideoVariant",
    "WordStamp",
    "build_contrast_pairs",
    "compose_transforms",
    "enumerate_video_variants",
    "project_box",
    "project_time_span",
    "project_words",
]


class GeometryTransform(BaseModel):
    """Affine mapping in normalised content coordinates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    tx: float = 0.0
    ty: float = 0.0

    def point(self, x: float, y: float) -> tuple[float, float]:
        return (self.a * x + self.c * y + self.tx, self.b * x + self.d * y + self.ty)


class PlacementTransform(BaseModel):
    """Content viewport inside the output frame, all values normalised."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: float = Field(default=0.0, ge=0.0, le=1.0)
    y: float = Field(default=0.0, ge=0.0, le=1.0)
    w: float = Field(default=1.0, gt=0.0, le=1.0)
    h: float = Field(default=1.0, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _inside_frame(self) -> Self:
        if self.x + self.w > 1.0 + 1e-9 or self.y + self.h > 1.0 + 1e-9:
            raise ValueError("placement viewport must stay inside the output frame")
        return self


class TimeTransform(BaseModel):
    """Mapping from source seconds to variant seconds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scale: float = Field(default=1.0, gt=0.0)
    offset: float = 0.0

    def map(self, seconds: float) -> float:
        return self.scale * seconds + self.offset

    def inverse(self, seconds: float) -> float:
        return (seconds - self.offset) / self.scale


class TimeSpan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    t0: float
    t1: float

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.t1 < self.t0:
            raise ValueError("time span end must not precede its start")
        return self


class WordStamp(TimeSpan):
    """One word aligned to the source or variant clock."""

    word: str = Field(min_length=1)
    role: str | None = None


class LabelTransform(BaseModel):
    """The supervision movement declared by one operator chain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    geometry: GeometryTransform = Field(default_factory=GeometryTransform)
    placement: PlacementTransform = Field(default_factory=PlacementTransform)
    time: TimeTransform = Field(default_factory=TimeTransform)
    semantics: Literal["preserved", "edited", "synthesized"] = "preserved"
    identity: dict[str, str] = Field(default_factory=dict)
    text: dict[str, str] = Field(default_factory=dict)


class VideoOperatorSpec(BaseModel):
    """Declarative operator axis used by the deterministic video sampler."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    axis: str = Field(min_length=1)
    kind: Literal["local", "generative"] = "local"
    label_class: Literal["preserving", "varying", "enriching"] = "preserving"
    domains: dict[str, tuple[str | int | float | bool, ...]] = Field(min_length=1)
    probability: float = Field(default=1.0, gt=0.0, le=1.0)


class VideoAugmentSpec(BaseModel):
    """A portable video augmentation plan, independent of a renderer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format: Literal["vidliner/video-spec@1"] = "vidliner/video-spec@1"
    id: str = Field(min_length=1)
    seed: int = Field(ge=0)
    strategy: Literal["full-factorial", "random", "paired"] = "paired"
    max_variants: int = Field(default=128, ge=1)
    max_active_ops: int | None = Field(default=None, ge=1)
    operators: tuple[VideoOperatorSpec, ...] = Field(min_length=1)


class VideoOperation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    op: str
    axis: str
    kind: Literal["local", "generative"]
    label_class: Literal["preserving", "varying", "enriching"]
    params: dict[str, str | int | float | bool] = Field(default_factory=dict)


class VideoVariant(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    slug: str
    seed: int
    variant_seed: int
    ops: tuple[VideoOperation, ...] = ()
    fingerprint: dict[str, str] = Field(default_factory=dict)


class ContrastPair(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    left: str
    right: str
    axis: str
    relation: str


def _apply_geometry(g: GeometryTransform, box: BoundingBox) -> BoundingBox:
    points = (
        g.point(box.x_min, box.y_min),
        g.point(box.x_max, box.y_min),
        g.point(box.x_min, box.y_max),
        g.point(box.x_max, box.y_max),
    )
    return BoundingBox.from_points([Point(x=x, y=y) for x, y in points])


def project_box(transform: LabelTransform, box: BoundingBox) -> BoundingBox:
    """Project a normalised box through geometry and output placement."""
    mapped = _apply_geometry(transform.geometry, box)
    p = transform.placement
    return BoundingBox(
        x_min=p.x + mapped.x_min * p.w,
        y_min=p.y + mapped.y_min * p.h,
        x_max=p.x + mapped.x_max * p.w,
        y_max=p.y + mapped.y_max * p.h,
    )


def project_time_span(transform: LabelTransform, span: TimeSpan) -> TimeSpan:
    """Project a source-clock span onto the variant clock."""
    a, b = transform.time.map(span.t0), transform.time.map(span.t1)
    return TimeSpan(t0=min(a, b), t1=max(a, b))


def project_words(
    transform: LabelTransform, words: tuple[WordStamp, ...], duration_s: float | None = None
) -> tuple[WordStamp, ...]:
    """Project words and clip them to the output clock when its duration is known."""
    projected: list[WordStamp] = []
    for word in words:
        span = project_time_span(transform, word)
        t0, t1 = max(0.0, span.t0), span.t1
        if duration_s is not None:
            t1 = min(duration_s, t1)
        if t1 - t0 <= 1e-6:
            continue
        projected.append(WordStamp(word=word.word, t0=t0, t1=t1, role=word.role))
    return tuple(projected)


def _matrix_multiply(left: GeometryTransform, right: GeometryTransform) -> GeometryTransform:
    return GeometryTransform(
        a=left.a * right.a + left.c * right.b,
        b=left.b * right.a + left.d * right.b,
        c=left.a * right.c + left.c * right.d,
        d=left.b * right.c + left.d * right.d,
        tx=left.a * right.tx + left.c * right.ty + left.tx,
        ty=left.b * right.tx + left.d * right.ty + left.ty,
    )


def compose_transforms(first: LabelTransform, second: LabelTransform) -> LabelTransform:
    """Compose source→intermediate ``first`` with intermediate→output ``second``."""
    semantics = (
        "synthesized"
        if "synthesized" in {first.semantics, second.semantics}
        else "edited"
        if "edited" in {first.semantics, second.semantics}
        else "preserved"
    )
    return LabelTransform(
        geometry=_matrix_multiply(second.geometry, first.geometry),
        placement=second.placement,
        time=TimeTransform(
            scale=second.time.scale * first.time.scale,
            offset=second.time.scale * first.time.offset + second.time.offset,
        ),
        semantics=semantics,
        identity={**first.identity, **second.identity},
        text={**first.text, **second.text},
    )


def build_contrast_pairs(variants: tuple[VideoVariant, ...]) -> tuple[ContrastPair, ...]:
    """Return pairs whose fingerprints differ on exactly one axis."""
    pairs: list[ContrastPair] = []
    for index, left in enumerate(variants):
        for right in variants[index + 1 :]:
            axes = set(left.fingerprint) | set(right.fingerprint)
            differing = [axis for axis in axes if left.fingerprint.get(axis) != right.fingerprint.get(axis)]
            if len(differing) != 1:
                continue
            axis = differing[0]
            relation = (
                "identity-invariance"
                if not left.fingerprint or not right.fingerprint
                else f"{axis}-invariance"
            )
            pairs.append(ContrastPair(left=left.id, right=right.id, axis=axis, relation=relation))
    return tuple(pairs)


def _domain_values(operator: VideoOperatorSpec) -> tuple[dict[str, Any], ...]:
    keys = tuple(sorted(operator.domains))
    return tuple(
        dict(zip(keys, values, strict=True)) for values in product(*(operator.domains[key] for key in keys))
    )


def enumerate_video_variants(spec: VideoAugmentSpec) -> tuple[VideoVariant, ...]:
    """Enumerate stable variants; IDs depend on content, not list position."""
    choices = [_domain_values(operator) for operator in spec.operators]
    rows: list[tuple[VideoOperation, ...]] = [()]
    if spec.strategy == "full-factorial":
        rows = []
        for selected in product(*choices):
            ops = tuple(
                VideoOperation(
                    op=operator.name,
                    axis=operator.axis,
                    kind=operator.kind,
                    label_class=operator.label_class,
                    params=params,
                )
                for operator, params in zip(spec.operators, selected, strict=True)
            )
            rows.append(ops)
    elif spec.strategy == "random":
        for index in range(spec.max_variants):
            ops = tuple(
                VideoOperation(
                    op=operator.name,
                    axis=operator.axis,
                    kind=operator.kind,
                    label_class=operator.label_class,
                    params=values[derive_seed(spec.seed, "variant", index, operator.axis) % len(values)],
                )
                for operator, values in zip(spec.operators, choices, strict=True)
            )
            rows.append(ops)
    else:  # paired: identity plus one-axis siblings, then deterministic truncation.
        rows = [()]
        for operator, values in zip(spec.operators, choices, strict=True):
            for params in values:
                rows.append(
                    (
                        VideoOperation(
                            op=operator.name,
                            axis=operator.axis,
                            kind=operator.kind,
                            label_class=operator.label_class,
                            params=params,
                        ),
                    )
                )

    variants: list[VideoVariant] = []
    for ops in rows[: spec.max_variants]:
        if spec.max_active_ops is not None and len(ops) > spec.max_active_ops:
            continue
        payload = [op.model_dump(mode="json") for op in ops]
        variant_seed = derive_seed(spec.seed, "variant", canonical_json(payload))
        digest = object_digest("v_", {"spec": spec.id, "ops": payload, "seed": variant_seed}, 16)
        fingerprint = {op.axis: canonical_json({"op": op.op, "params": op.params}) for op in ops}
        variants.append(
            VideoVariant(
                id=f"{spec.id}__{digest}",
                slug=f"{spec.id}-{digest[-12:]}",
                seed=spec.seed,
                variant_seed=variant_seed,
                ops=ops,
                fingerprint=fingerprint,
            )
        )
    return tuple(variants)
