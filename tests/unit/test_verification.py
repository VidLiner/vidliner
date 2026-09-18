"""Tests for the closed loop: post-generation re-detection and re-segmentation.

These are the tests that could not have been written before the ``verify.redetect`` stage existed.
They assert the property the stage is for: the pipeline describes the object that is **actually in
the generated image**, so a generator that ignores the mask, moves the object, or returns nothing is
caught instead of passing a comparison with the mask it was handed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from vidliner.capabilities.backend import (
    BackendProbe,
    DetectionRequest,
    DetectionResult,
    SegmentationResult,
)
from vidliner.core.errors import ErrorCode, VidlinerError
from vidliner.core.identity import object_id
from vidliner.domain.enums import ArtifactKind
from vidliner.domain.instances import ObjectInstance, ObjectItem
from vidliner.domain.masks import MaskRef
from vidliner.domain.replacement import (
    CandidateSpec,
    GeometryBudget,
    PreservationRules,
    ReplacementIntent,
    ReplacementPlan,
)
from vidliner.domain.shapes import BoundingBox, ImageShape
from vidliner.operators.base import ExecutionContext, Operator
from vidliner.operators.registry import create_operator
from vidliner.storage.workspace import ArtifactStore, BackendIO, Workspace

WIDTH, HEIGHT = 160, 120
ASSET_DIGEST = "b" * 64


class StubHandle:
    """A resolved capability whose single method returns a canned result, or raises."""

    def __init__(self, capability: str, backend_id: str, method: str, outcome: Any) -> None:
        self.capability = capability
        self.backend_id = backend_id
        self.capabilities = (capability,)
        self._method = method
        self._outcome = outcome
        self.calls: list[tuple[Any, ...]] = []

    async def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((method, *args))
        if method != self._method:
            raise AssertionError(f"stub for {self.capability} received unexpected method {method!r}")
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        if callable(self._outcome):
            return self._outcome(*args, **kwargs)
        return self._outcome

    async def probe(self) -> BackendProbe:
        return BackendProbe(backend_id=self.backend_id, version="stub")


class StubResolver:
    """Maps a capability name to a :class:`StubHandle`."""

    def __init__(self, handles: dict[str, StubHandle]) -> None:
        self._handles = handles
        self.metadata: dict[str, Any] = {}

    def __call__(self, capability: str) -> StubHandle:
        return self._handles[capability]


@pytest.fixture(name="io")
def io_fixture(tmp_path: Any) -> BackendIO:
    """Artifact I/O backed by a real store, so masks are genuinely written and read back."""
    return BackendIO(ArtifactStore(Workspace(tmp_path / "ws").initialize()))


def _scene_array(*, with_object: bool, box: tuple[int, int, int, int] = (30, 20, 90, 80)) -> np.ndarray:
    """A flat background, optionally with a bright block standing in for the object."""
    array = np.full((HEIGHT, WIDTH, 3), 110, dtype=np.uint8)
    if with_object:
        x0, y0, x1, y1 = box
        array[y0:y1, x0:x1] = (210, 60, 60)
    return array


def _image(io: BackendIO, *, with_object: bool, box: tuple[int, int, int, int] = (30, 20, 90, 80)) -> Any:
    from PIL import Image

    array = _scene_array(with_object=with_object, box=box)
    return io.save_image(Image.fromarray(array, mode="RGB"), kind=ArtifactKind.IMAGE)


def _mask_ref(io: BackendIO, box: tuple[int, int, int, int] = (30, 20, 90, 80)) -> MaskRef:
    """A stored mask plus the facts a consumer needs, exactly as segmentation produces it."""
    array = np.zeros((HEIGHT, WIDTH), dtype=bool)
    x0, y0, x1, y1 = box
    array[y0:y1, x0:x1] = True
    artifact = io.save_mask(array).with_kind(ArtifactKind.MASK)
    return MaskRef(
        artifact=artifact,
        shape=ImageShape(width=WIDTH, height=HEIGHT),
        area_px=int(array.sum()),
    )


def _mask(io: BackendIO, box: tuple[int, int, int, int] = (30, 20, 90, 80)) -> MaskRef:
    return _mask_ref(io, box)


def _instance(
    mask_ref: MaskRef, *, class_name: str = "car", box: tuple[int, int, int, int] = (30, 20, 90, 80)
) -> ObjectInstance:
    bbox = BoundingBox(x_min=float(box[0]), y_min=float(box[1]), x_max=float(box[2]), y_max=float(box[3]))
    return ObjectInstance(
        object_id=object_id(ASSET_DIGEST, 0, class_name, bbox, 0.9),
        class_name=class_name,
        bbox=bbox,
        score=0.9,
        mask_ref=mask_ref,
        area_px=mask_ref.area_px,
        source_frame=f"{ASSET_DIGEST}#0",
    )


def _item(io: BackendIO, *, class_name: str = "car") -> ObjectItem:
    mask_ref = _mask(io)
    return ObjectItem(
        instance=_instance(mask_ref, class_name=class_name),
        sample_id="s_1",
        selection_key="object:o_1",
        classes_requested=(class_name,),
    )


def _plan(*, expected_category: str = "sedan") -> ReplacementPlan:
    return ReplacementPlan(
        plan_id="p_1",
        sample_id="s_1",
        intent=ReplacementIntent(target_object="o_1", replacement_category=expected_category),
        expected_category=expected_category,
        positive_prompt=f"replace the car with a {expected_category}",
        candidate_specs=(
            CandidateSpec(candidate_key="sedan-1", category=expected_category, description="a sedan", seed=7),
        ),
        preservation=PreservationRules(),
        geometry_budget=GeometryBudget(),
        plan_seed=7,
        planner_id="stub",
    )


def _candidate(*, category: str = "sedan") -> dict[str, Any]:
    return {"candidate_key": "sedan-1", "category": category, "object_id": "o_1", "sample_id": "s_1"}


def _context(
    io: BackendIO,
    handles: dict[str, StubHandle],
    *,
    config: dict[str, Any] | None = None,
) -> ExecutionContext:
    from vidliner.core.graph import OperationNode
    from vidliner.domain.enums import Determinism, PortType, StageName

    node = OperationNode(
        node_id="n_verify",
        operator="verify.redetect",
        operator_version="1.0.0",
        stage=StageName.VERIFY,
        outputs={"item": PortType.INSTANCES, "mask": PortType.MASK_REF, "found": PortType.FLAG},
        config=config or {},
        determinism=Determinism.DETERMINISTIC,
        lineage=("sample:s_1", "object:o_1", "candidate:candidate-0"),
    )
    return ExecutionContext(
        job_id="j_test",
        node=node,
        seed=11,
        io=io,
        backends=StubResolver(handles),
    )


def _detection(
    box: tuple[int, int, int, int] = (30, 20, 90, 80), *, class_name: str = "car", score: float = 0.9
) -> DetectionResult:
    bbox = BoundingBox(x_min=float(box[0]), y_min=float(box[1]), x_max=float(box[2]), y_max=float(box[3]))
    return DetectionResult(
        instances=(
            ObjectInstance(
                object_id=object_id(ASSET_DIGEST, 0, class_name, bbox, score),
                class_name=class_name,
                bbox=bbox,
                score=score,
                source_frame=f"{ASSET_DIGEST}#0",
            ),
        ),
        backend_id="stub_detector",
    )


def _segmentation(io: BackendIO, box: tuple[int, int, int, int]) -> SegmentationResult:
    mask_ref = _mask_ref(io, box)
    return SegmentationResult(
        mask=mask_ref.artifact,
        shape=mask_ref.shape,
        area_px=mask_ref.area_px,
    )


def _operator() -> Operator:
    return create_operator("verify.redetect", {})


# --------------------------------------------------------------------------- #
# The happy path: the object is where it was put
# --------------------------------------------------------------------------- #


async def test_redetect_finds_the_object_and_rebuilds_its_geometry(io: BackendIO) -> None:
    handles = {
        "vision.object_detection.v1": StubHandle(
            "vision.object_detection.v1", "stub_detector", "detect", _detection()
        ),
        "vision.instance_segmentation.v1": StubHandle(
            "vision.instance_segmentation.v1",
            "stub_segmenter",
            "segment",
            lambda request, context: _segmentation(io, (36, 26, 84, 74)),
        ),
    }
    context = _context(io, handles)
    outputs = await _operator().run(
        {
            "image": _image(io, with_object=True),
            "item": _item(io),
            "mask": _mask(io),
            "plan": _plan(),
            "candidate": _candidate(),
        },
        context,
    )
    assert outputs["found"] is True
    regenerated = outputs["item"]
    assert regenerated.instance.attributes["regenerated"] is True
    # The geometry is the *found* object's, not the mask that was handed to the generator.
    assert regenerated.instance.bbox.as_xyxy() == (36.0, 26.0, 84.0, 74.0)
    assert regenerated.instance.area_px == 48 * 48
    assert regenerated.instance.polygon is not None
    assert regenerated.instance.polygon.point_count >= 3
    assert regenerated.instance.attributes["source_bbox"] == [30.0, 20.0, 90.0, 80.0]


async def test_redetect_requests_the_source_class_not_the_replacement_category(io: BackendIO) -> None:
    """A detector's label set does not contain sub-categories such as ``sedan``."""
    detector = StubHandle("vision.object_detection.v1", "stub_detector", "detect", _detection())
    handles = {
        "vision.object_detection.v1": detector,
        "vision.instance_segmentation.v1": StubHandle(
            "vision.instance_segmentation.v1",
            "stub_segmenter",
            "segment",
            lambda request, context: _segmentation(io, (30, 20, 90, 80)),
        ),
    }
    await _operator().run(
        {
            "image": _image(io, with_object=True),
            "item": _item(io, class_name="car"),
            "mask": _mask(io),
            "plan": _plan(expected_category="sedan"),
            "candidate": _candidate(category="sedan"),
        },
        _context(io, handles),
    )
    request: DetectionRequest = detector.calls[0][1]
    assert request.classes == ("car",)
    assert request.known_classes == ("car",)


async def test_redetect_can_be_configured_to_look_for_the_replacement_category(io: BackendIO) -> None:
    detector = StubHandle(
        "vision.object_detection.v1", "stub_detector", "detect", _detection(class_name="sedan")
    )
    handles = {
        "vision.object_detection.v1": detector,
        "vision.instance_segmentation.v1": StubHandle(
            "vision.instance_segmentation.v1",
            "stub_segmenter",
            "segment",
            lambda request, context: _segmentation(io, (30, 20, 90, 80)),
        ),
    }
    outputs = await _operator().run(
        {
            "image": _image(io, with_object=True),
            "item": _item(io),
            "mask": _mask(io),
            "plan": _plan(),
            "candidate": _candidate(),
        },
        _context(io, handles, config={"detect_class": "replacement"}),
    )
    request: DetectionRequest = detector.calls[0][1]
    assert request.classes == ("sedan",)
    assert outputs["found"] is True


# --------------------------------------------------------------------------- #
# The failures the loop exists to catch
# --------------------------------------------------------------------------- #


async def test_redetect_reports_not_found_when_the_object_is_gone(io: BackendIO) -> None:
    """A generator that removes the object must be caught, not scored against its own mask."""
    handles = {
        "vision.object_detection.v1": StubHandle(
            "vision.object_detection.v1",
            "stub_detector",
            "detect",
            DetectionResult(instances=(), backend_id="stub_detector"),
        ),
        "vision.instance_segmentation.v1": StubHandle(
            "vision.instance_segmentation.v1",
            "stub_segmenter",
            "segment",
            _segmentation(io, (30, 20, 90, 80)),
        ),
    }
    outputs = await _operator().run(
        {
            "image": _image(io, with_object=False),
            "item": _item(io),
            "mask": _mask(io),
            "plan": _plan(),
            "candidate": _candidate(),
        },
        _context(io, handles),
    )
    assert outputs["found"] is False
    assert outputs["mask"].area_px == 0
    assert outputs["mask"].is_empty
    # The original item is passed through unchanged, so the failure is visible downstream rather
    # than replaced by a plausible-looking empty instance.
    assert outputs["item"].instance.mask_ref is not None


async def test_redetect_reports_not_found_when_the_detection_is_far_from_the_target(io: BackendIO) -> None:
    """An object elsewhere in the frame is not the replacement."""
    handles = {
        "vision.object_detection.v1": StubHandle(
            "vision.object_detection.v1",
            "stub_detector",
            "detect",
            _detection(box=(2, 2, 14, 14)),
        ),
        "vision.instance_segmentation.v1": StubHandle(
            "vision.instance_segmentation.v1", "stub_segmenter", "segment", _segmentation(io, (2, 2, 14, 14))
        ),
    }
    outputs = await _operator().run(
        {
            "image": _image(io, with_object=True),
            "item": _item(io),
            "mask": _mask(io),
            "plan": _plan(),
            "candidate": _candidate(),
        },
        _context(io, handles, config={"min_iou": 0.15}),
    )
    assert outputs["found"] is False


async def test_redetect_tolerates_a_partial_overlap_above_the_threshold(io: BackendIO) -> None:
    """A slightly resized object is still the replacement; a far one is not."""
    handles = {
        "vision.object_detection.v1": StubHandle(
            "vision.object_detection.v1",
            "stub_detector",
            "detect",
            _detection(box=(38, 28, 98, 88)),
        ),
        "vision.instance_segmentation.v1": StubHandle(
            "vision.instance_segmentation.v1",
            "stub_segmenter",
            "segment",
            lambda request, context: _segmentation(io, (38, 28, 98, 88)),
        ),
    }
    outputs = await _operator().run(
        {
            "image": _image(io, with_object=True),
            "item": _item(io),
            "mask": _mask(io),
            "plan": _plan(),
            "candidate": _candidate(),
        },
        _context(io, handles, config={"min_iou": 0.10}),
    )
    assert outputs["found"] is True
    assert outputs["item"].instance.attributes["match_iou"] > 0.10


async def test_redetect_refuses_a_detection_of_the_wrong_class(io: BackendIO) -> None:
    handles = {
        "vision.object_detection.v1": StubHandle(
            "vision.object_detection.v1", "stub_detector", "detect", _detection(class_name="bus")
        ),
        "vision.instance_segmentation.v1": StubHandle(
            "vision.instance_segmentation.v1",
            "stub_segmenter",
            "segment",
            _segmentation(io, (30, 20, 90, 80)),
        ),
    }
    outputs = await _operator().run(
        {
            "image": _image(io, with_object=True),
            "item": _item(io, class_name="car"),
            "mask": _mask(io),
            "plan": _plan(),
            "candidate": _candidate(),
        },
        _context(io, handles, config={"require_same_class": True}),
    )
    assert outputs["found"] is False


async def test_redetect_reports_not_found_when_segmentation_produces_nothing(io: BackendIO) -> None:
    """Detection agreeing is not enough; a mask must exist for a label to be written."""
    handles = {
        "vision.object_detection.v1": StubHandle(
            "vision.object_detection.v1", "stub_detector", "detect", _detection()
        ),
        "vision.instance_segmentation.v1": StubHandle(
            "vision.instance_segmentation.v1",
            "stub_segmenter",
            "segment",
            VidlinerError("no mask", code=ErrorCode.MASK_EMPTY),
        ),
    }
    outputs = await _operator().run(
        {
            "image": _image(io, with_object=True),
            "item": _item(io),
            "mask": _mask(io),
            "plan": _plan(),
            "candidate": _candidate(),
        },
        _context(io, handles),
    )
    assert outputs["found"] is False
    assert outputs["mask"].is_empty


async def test_redetect_propagates_a_backend_failure_that_is_not_a_missing_mask(io: BackendIO) -> None:
    from vidliner.core.errors import BackendFailure

    handles = {
        "vision.object_detection.v1": StubHandle(
            "vision.object_detection.v1", "stub_detector", "detect", _detection()
        ),
        "vision.instance_segmentation.v1": StubHandle(
            "vision.instance_segmentation.v1",
            "stub_segmenter",
            "segment",
            BackendFailure("service down", backend_id="stub_segmenter", safe_to_retry=True),
        ),
    }
    with pytest.raises(BackendFailure):
        await _operator().run(
            {
                "image": _image(io, with_object=True),
                "item": _item(io),
                "mask": _mask(io),
                "plan": _plan(),
                "candidate": _candidate(),
            },
            _context(io, handles),
        )


async def test_redetect_rejects_a_mask_that_covers_the_whole_frame(io: BackendIO) -> None:
    handles = {
        "vision.object_detection.v1": StubHandle(
            "vision.object_detection.v1", "stub_detector", "detect", _detection()
        ),
        "vision.instance_segmentation.v1": StubHandle(
            "vision.instance_segmentation.v1",
            "stub_segmenter",
            "segment",
            lambda request, context: SegmentationResult(
                mask=_mask_ref(io, (0, 0, WIDTH, HEIGHT)).artifact,
                shape=ImageShape(width=WIDTH, height=HEIGHT),
                area_px=WIDTH * HEIGHT,
            ),
        ),
    }
    with pytest.raises(VidlinerError) as error:
        await _operator().run(
            {
                "image": _image(io, with_object=True),
                "item": _item(io),
                "mask": _mask(io),
                "plan": _plan(),
                "candidate": _candidate(),
            },
            _context(io, handles),
        )
    assert error.value.code is ErrorCode.MASK_INVALID


# --------------------------------------------------------------------------- #
# The stage contract
# --------------------------------------------------------------------------- #


def test_redetect_spec_declares_both_capabilities() -> None:
    from vidliner.capabilities.names import CAP_INSTANCE_SEGMENTATION, CAP_OBJECT_DETECTION

    spec = _operator().spec
    assert spec.name == "verify.redetect"
    assert spec.stage.value == "verify"
    assert CAP_OBJECT_DETECTION in spec.capabilities
    assert CAP_INSTANCE_SEGMENTATION in spec.capabilities
    assert set(spec.outputs) == {"item", "mask", "found"}
    assert spec.outputs["found"].port_type.value == "flag"


def test_verify_stage_sits_between_refine_and_evaluate() -> None:
    """Stage order is what makes the loop closed: verify must precede evaluate."""
    from vidliner.core.graph import _STAGE_ORDER
    from vidliner.domain.enums import StageName

    order = list(_STAGE_ORDER)
    assert order.index(StageName.REFINE) < order.index(StageName.VERIFY)
    assert order.index(StageName.VERIFY) < order.index(StageName.EVALUATE)
    assert order.index(StageName.EVALUATE) < order.index(StageName.ANNOTATE)


def test_graph_wires_verify_between_refine_and_annotate() -> None:
    """The assembled graph must route the regenerated object into evaluation and annotation."""
    from vidliner.core.graph import NodeOutput
    from vidliner.domain.recipe import Recipe
    from vidliner.pipeline.assemble import assemble_graph

    recipe = Recipe.model_validate(
        {
            "recipe": "t",
            "dataset": {"input": "./d"},
            "target": {"classes": ["car"]},
            "replacement": {"values": ["sedan"]},
            "limits": {"per_node_timeout_s": 60},
        }
    )
    assembly = assemble_graph(recipe, ("s_1",))
    graph = assembly.graph
    verify = [node for node in graph.nodes if node.operator == "verify.redetect"]
    evaluate = [node for node in graph.nodes if node.operator == "evaluate.candidate"]
    annotate = [node for node in graph.nodes if node.operator == "annotate.rebuild"]
    assert verify and evaluate and annotate

    verify_ids = {node.node_id for node in verify}
    for node in evaluate:
        bound = {binding.node_id for binding in node.inputs.values() if isinstance(binding, NodeOutput)}
        assert verify_ids & bound, "evaluate must read the regenerated object"
        assert node.inputs["input_mask"] is not None
    for node in annotate:
        bound = {binding.node_id for binding in node.inputs.values() if isinstance(binding, NodeOutput)}
        assert verify_ids & bound, "annotation must be rebuilt from the regenerated object"
    assert assembly.nodes("verify"), "the assembler must group the verification nodes for reporting"
