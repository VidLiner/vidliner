"""Measured scene analysis.

Everything this backend reports is *measured from pixels*: orientation from image moments, lighting
direction from a luminance gradient, ground contact from the lowest mask row, shadow from a
darkened region beneath the object, occlusion from overlap with the other detected objects. Nothing
is invented, and anything that cannot be measured stays ``None`` so the planner can be conservative
instead of confidently wrong.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vidliner.backends.base import LocalBackend
from vidliner.backends.imageops import dilate, erode, mask_geometry
from vidliner.capabilities.backend import BackendProbe, PipelineContext, SceneAnalysisRequest
from vidliner.capabilities.names import CAP_SCENE_ANALYSIS
from vidliner.domain.enums import Determinism, SceneScaleClass
from vidliner.domain.masks import MaskRef
from vidliner.domain.scene import LightingEstimate, OcclusionRelation, SceneContext
from vidliner.domain.shapes import BoundingBox

__all__ = ["HeuristicSceneBackend"]


class HeuristicSceneBackend(LocalBackend):
    """Measure orientation, lighting, scale, ground contact, shadow, and occlusion."""

    backend_id = "heuristic_scene"
    backend_version = "1.0.0"
    capabilities = (CAP_SCENE_ANALYSIS,)
    determinism = Determinism.DETERMINISTIC
    declared_capabilities = (CAP_SCENE_ANALYSIS,)

    def __init__(self, spec: Any, options: dict[str, Any], credentials: object | None = None) -> None:
        super().__init__(spec, options, credentials)
        self._sync = _SceneSync(self)

    async def analyse(self, request: SceneAnalysisRequest, context: PipelineContext) -> SceneContext:
        """Protocol method; the runtime executes ``self._sync.analyse`` on a worker thread."""
        return self._sync.analyse(request, context)

    async def probe(self) -> BackendProbe:
        """Report readiness."""
        probe = await super().probe()
        return BackendProbe(
            backend_id=probe.backend_id,
            version=probe.version,
            health=probe.health,
            capabilities=self.capabilities,
            device=probe.device,
            determinism=self.determinism,
            safe_to_retry=True,
            external=False,
            message="moment- and gradient-based scene measurements",
            details=probe.details,
        )


class _SceneSync:
    """Synchronous implementation of the scene analyser."""

    def __init__(self, backend: HeuristicSceneBackend) -> None:
        self._backend = backend

    def analyse(self, request: SceneAnalysisRequest, context: PipelineContext) -> SceneContext:
        """Measure the scene around one target object."""
        del context  # pixels and mask already travel with the request
        mask = _mask_from(request)
        shape = request.image_shape
        luminance = _luminance_of(request)
        geometry = mask_geometry(mask)
        lighting = self._lighting(mask, luminance, geometry.bbox)
        shadow = self._shadow(mask, luminance)
        ground = self._ground_contact(mask, luminance)
        occlusions = self._occlusions(request)
        return SceneContext(
            object_category=request.target.class_name,
            object_id=request.target.object_id,
            image_shape=shape,
            bbox=geometry.bbox,
            mask=_mask_ref(request),
            orientation_deg=self._orientation(mask, geometry.bbox),
            pose=None,
            scale_class=_scale_class(geometry.area_px / max(1, shape.pixel_count)),
            lighting=lighting,
            shadow=shadow,
            depth_order=None,
            occlusions=occlusions,
            background_summary=self._background_summary(luminance, mask),
            ground_contact=ground,
            neighbours=tuple(neighbour.object_id for neighbour in request.neighbours),
            measured_by=self._backend.backend_id,
        )

    def _orientation(self, mask: np.ndarray, bbox: BoundingBox) -> float | None:
        """In-plane orientation of the object's principal axis, in degrees."""
        rows, columns = np.nonzero(mask)
        if rows.size < 16:
            return None
        y = rows.astype(np.float64) - float(rows.mean())
        x = columns.astype(np.float64) - float(columns.mean())
        covariance_xx = float((x * x).mean())
        covariance_yy = float((y * y).mean())
        covariance_xy = float((x * y).mean())
        angle = 0.5 * np.arctan2(2.0 * covariance_xy, covariance_xx - covariance_yy)
        degrees = float(np.degrees(angle))
        if bbox.width < bbox.height:
            degrees += 90.0
        return float(((degrees + 90.0) % 180.0) - 90.0)

    def _lighting(self, mask: np.ndarray, luminance: np.ndarray, bbox: BoundingBox) -> LightingEstimate:
        """Estimate the dominant illumination direction and strength on the object."""
        inner = erode(mask, 1)
        region = inner if inner.any() else mask
        values = luminance[region]
        if values.size == 0:
            return LightingEstimate(measured_by=self._backend.backend_id)
        intensity = float(np.clip(values.mean(), 0.0, 1.0))
        contrast = float(np.clip(values.std(), 0.0, 1.0))
        # The direction is the gradient of a blurred luminance patch: light comes from the bright side.
        patch = _blur(luminance, 5)
        height, width = patch.shape
        y0 = int(np.clip(bbox.y_min - 4, 0, max(0, height - 1)))
        y1 = int(np.clip(bbox.y_max + 4, 1, height))
        x0 = int(np.clip(bbox.x_min - 4, 0, max(0, width - 1)))
        x1 = int(np.clip(bbox.x_max + 4, 1, width))
        window = patch[y0:y1, x0:x1]
        if window.size < 4:
            return LightingEstimate(
                intensity=intensity, contrast=contrast, measured_by=self._backend.backend_id
            )
        gradient_y, gradient_x = np.gradient(window)
        mean_gx = float(gradient_x.mean())
        mean_gy = float(gradient_y.mean())
        if abs(mean_gx) < 1e-4 and abs(mean_gy) < 1e-4:
            direction = None
        else:
            direction = float(np.degrees(np.arctan2(mean_gy, mean_gx)) % 360.0)
        return LightingEstimate(
            direction_deg=direction,
            intensity=intensity,
            contrast=contrast,
            measured_by=self._backend.backend_id,
        )

    def _shadow(self, mask: np.ndarray, luminance: np.ndarray) -> BoundingBox | None:
        """Locate a darkened region under the object, if any."""
        rows = np.nonzero(mask.any(axis=1))[0]
        if rows.size == 0:
            return None
        bottom = int(rows[-1])
        height, _width = mask.shape
        band_end = min(height, bottom + max(6, int(0.06 * height)))
        if band_end <= bottom + 1:
            return None
        below_columns = mask[bottom, :]
        if not below_columns.any():
            return None
        band = luminance[bottom + 1 : band_end, :]
        reference = float(luminance[mask].mean()) if mask.any() else 0.5
        dark = band < reference * float(self._backend.option("shadow_darkness", 0.82))
        dark[:, ~below_columns] = False
        if not dark.any():
            return None
        ys, xs = np.nonzero(dark)
        return BoundingBox(
            x_min=float(xs.min()),
            y_min=float(bottom + 1 + ys.min()),
            x_max=float(xs.max() + 1),
            y_max=float(bottom + 1 + ys.max() + 1),
        )

    def _ground_contact(self, mask: np.ndarray, luminance: np.ndarray) -> int | None:
        """The image row where the object meets a support surface, when one is visible."""
        rows = np.nonzero(mask.any(axis=1))[0]
        if rows.size == 0:
            return None
        bottom = int(rows[-1])
        height = mask.shape[0]
        if bottom + 2 >= height:
            return None
        band = luminance[bottom + 1 : min(height, bottom + 4), :]
        columns = mask[bottom, :]
        if not columns.any() or band.size == 0:
            return None
        support = float(band[:, columns].std())
        object_spread = float(luminance[mask].std()) if mask.any() else 0.0
        if support <= max(0.02, object_spread * 0.5):
            return None
        return bottom

    def _occlusions(self, request: SceneAnalysisRequest) -> tuple[OcclusionRelation, ...]:
        relations: list[OcclusionRelation] = []
        target_box = request.target.bbox
        for neighbour in request.neighbours:
            if neighbour.object_id == request.target.object_id:
                continue
            other = neighbour.bbox
            intersection_width = max(
                0.0, min(target_box.x_max, other.x_max) - max(target_box.x_min, other.x_min)
            )
            intersection_height = max(
                0.0, min(target_box.y_max, other.y_max) - max(target_box.y_min, other.y_min)
            )
            intersection = intersection_width * intersection_height
            smaller = min(target_box.area, other.area)
            overlap = intersection / smaller if smaller > 0 else 0.0
            if overlap <= 0.0:
                relation = "adjacent"
            elif overlap >= 0.5:
                # The taller box in image space is nearer the viewer in a typical ground-plane scene.
                relation = "in_front_of" if other.y_max > target_box.y_max else "behind"
            else:
                relation = "overlapping"
            relations.append(
                OcclusionRelation(
                    other_object_id=neighbour.object_id,
                    relation=relation,
                    overlap_fraction=float(np.clip(overlap, 0.0, 1.0)),
                )
            )
        relations.sort(key=lambda item: item.overlap_fraction, reverse=True)
        return tuple(relations)

    def _background_summary(self, luminance: np.ndarray, mask: np.ndarray) -> str:
        """A short, measured statement about the background, used verbatim in a generation prompt."""
        outside = ~dilate(mask, 4)
        values = luminance[outside]
        if values.size == 0:
            return "the surrounding scene"
        mean = float(values.mean())
        spread = float(values.std())
        if spread < 0.05:
            tone = "flat"
        elif spread < 0.15:
            tone = "smoothly graded"
        else:
            tone = "busy"
        if mean < 0.3:
            light = "dark"
        elif mean < 0.6:
            light = "mid-tone"
        else:
            light = "bright"
        return f"a {tone} {light} background"


def _mask_from(request: SceneAnalysisRequest) -> np.ndarray:
    """Materialise the target object's mask as a boolean array.

    The analysis backend needs pixels, and the mask travels as an artifact. The operator that builds
    this request therefore attaches the decoded mask to the request metadata; when it is absent the
    bounding box is used, and that substitution is visible in the returned context because
    ``mask.coverage`` will match a box rather than a silhouette.
    """
    mask = request.mask_array
    if isinstance(mask, np.ndarray):
        return mask
    raise ValueError("scene analysis requires the decoded mask array on the request")


def _mask_ref(request: SceneAnalysisRequest) -> MaskRef:
    mask_ref = request.target.mask_ref
    if mask_ref is None:
        raise ValueError("scene analysis requires the target object to be segmented")
    return mask_ref


def _luminance_of(request: SceneAnalysisRequest) -> np.ndarray:
    luminance = request.luminance
    if isinstance(luminance, np.ndarray):
        return luminance
    raise ValueError("scene analysis requires the decoded luminance array on the request")


def _blur(values: np.ndarray, window: int) -> np.ndarray:
    """Box blur via an integral image, edge-padded."""
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")
    height, width = values.shape
    total = (
        integral[window : window + height, window : window + width]
        - integral[0:height, window : window + width]
        - integral[window : window + height, 0:width]
        + integral[0:height, 0:width]
    )
    return total / float(window * window)


def _scale_class(coverage: float) -> SceneScaleClass:
    if coverage < 0.01:
        return SceneScaleClass.TINY
    if coverage < 0.05:
        return SceneScaleClass.SMALL
    if coverage < 0.20:
        return SceneScaleClass.MEDIUM
    return SceneScaleClass.LARGE
