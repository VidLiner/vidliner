"""Per-sample export: collect an accepted candidate's outputs into the run directory.

The dataset-level export (COCO/YOLO documents, images, manifest) is performed once per job by
:mod:`vidliner.pipeline.exporter`, because it needs every sample at once to build a consistent
category vocabulary and to check duplicates across the whole accepted set.

What this node does is the per-sample part: it turns a candidate into the exact records the dataset
exporter consumes — the output image, the rebuilt annotation, the decision, and the reason codes —
and writes them into the job's run directory as evidence. Keeping this in the graph means an
accepted sample is materialised by the same engine that produced it, with the same caching and
resume behaviour as every other stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from vidliner.core.errors import ErrorCode, OperatorFailure
from vidliner.domain.annotations import AnnotationBundle
from vidliner.domain.enums import Determinism, PortType, StageName
from vidliner.domain.instances import ObjectItem
from vidliner.operators.base import ExecutionContext, InputSpec, Operator, OperatorSpec, OutputSpec
from vidliner.operators.shared import annotation_from_payload
from vidliner.pipeline.export_names import file_key

__all__ = ["ExportConfig", "MaterialiseSampleOperator", "register_all"]


class ExportConfig(BaseModel):
    """Configuration of the per-sample export operator."""

    model_config = ConfigDict(extra="forbid")

    keep_rejected: bool = True
    write_difference: bool = True
    write_overlay: bool = True


class MaterialiseSampleOperator(Operator):
    """Write one candidate's evidence into the job run directory."""

    @property
    def spec(self) -> OperatorSpec:
        """Declared contract of the per-sample export operator."""
        return OperatorSpec(
            name="export.sample",
            version="1.0.0",
            stage=StageName.EXPORT,
            summary="materialise an evaluated candidate's evidence for the dataset exporter",
            inputs={
                "item": InputSpec(PortType.INSTANCES, "the target object"),
                "candidate": InputSpec(PortType.CANDIDATE_REF, "the generated candidate payload"),
                "image": InputSpec(PortType.IMAGE_REF, "the refined candidate image"),
                "source_image": InputSpec(PortType.IMAGE_REF, "the untouched source frame"),
                "annotation": InputSpec(PortType.ANNOTATION, "the rebuilt annotation"),
                "report": InputSpec(PortType.QUALITY_REPORT, "the quality report"),
                "decision": InputSpec(PortType.ANY, "the acceptance decision", required=False),
            },
            outputs={"slice": OutputSpec(PortType.DATASET_SLICE, "per-sample export record")},
            config_model=ExportConfig,
            capabilities=(),
            determinism=Determinism.DETERMINISTIC,
            timeout_s=120.0,
            max_parallelism=4,
        )

    async def run(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        """Write the sample's artifacts and return its export record."""
        config = ExportConfig.model_validate(context.config)
        item = _item(inputs.get("item"))
        candidate = inputs.get("candidate")
        if not isinstance(candidate, dict):
            raise OperatorFailure(
                "export needs the generation payload produced upstream",
                code=ErrorCode.PORT_TYPE_MISMATCH,
            )
        image = inputs.get("image")
        source_image = inputs.get("source_image")
        annotation = annotation_from_payload(inputs.get("annotation"))
        report = inputs.get("report")
        decision = inputs.get("decision")
        decision_state = _decision_state(decision, report)

        sample_dir = _sample_dir(context, item.sample_id)
        if decision_state != "accepted" and not config.keep_rejected:
            return {
                "slice": _slice(
                    item, candidate, annotation, report, decision_state, sample_dir, written=False
                )
            }

        # One evidence set per *candidate*: a sample may produce several, and a later candidate must
        # not overwrite the earlier one's evidence.
        key = _candidate_file_key(candidate, item)
        written: list[str] = []
        written.append(_write(context, image, sample_dir / f"refined-{key}.png"))
        written.append(_write(context, source_image, sample_dir / "source.png"))
        written.append(_write_json(annotation.model_dump(mode="json"), sample_dir / f"annotation-{key}.json"))
        if report is not None:
            payload = report.model_dump(mode="json") if hasattr(report, "model_dump") else dict(report)
            written.append(_write_json(payload, sample_dir / f"quality-{key}.json"))
        if decision is not None:
            payload = decision.model_dump(mode="json") if hasattr(decision, "model_dump") else dict(decision)
            written.append(_write_json(payload, sample_dir / f"decision-{key}.json"))
        written.append(_write_json(candidate, sample_dir / f"candidate-{key}.json"))

        selection = _slice(item, candidate, annotation, report, decision_state, sample_dir, written=True)
        context.publish(
            "export.sample",
            sample_id=item.sample_id,
            object_id=item.object_id,
            decision=decision_state,
            files=written,
        )
        return {"slice": selection}


def _slice(
    item: ObjectItem,
    candidate: dict[str, Any],
    annotation: AnnotationBundle,
    report: Any,
    decision_state: str,
    sample_dir: Path,
    *,
    written: bool,
) -> dict[str, Any]:
    """The dataset exporter's per-sample record."""
    generation = candidate.get("generation") or {}
    image = generation.get("image") or {}
    return {
        "sample_id": item.sample_id,
        "object_id": item.object_id,
        "candidate_key": candidate.get("candidate_key"),
        "category": candidate.get("category"),
        "seed": candidate.get("seed"),
        "decision": decision_state,
        "output_digest": image.get("digest"),
        "output_media_type": image.get("media_type"),
        "annotation": annotation.model_dump(mode="json"),
        "metrics": (
            {outcome.metric.value: outcome.value for outcome in report.metrics if outcome.value is not None}
            if report is not None and hasattr(report, "metrics")
            else {}
        ),
        "reason_codes": list(getattr(report, "reason_codes", ()) or ()),
        "overall_score": getattr(report, "overall_score", None),
        "evidence_dir": sample_dir.as_posix(),
        "written": written,
    }


def _candidate_file_key(candidate: dict[str, Any], item: ObjectItem) -> str:
    """A filesystem-safe key that distinguishes the candidates of one sample.

    The convention is shared with every reader (see :mod:`vidliner.pipeline.export_names`); the
    writer must not invent its own spelling of it.
    """
    raw = str(candidate.get("candidate_key") or "")
    return file_key(raw, fallback=file_key(item.object_id, fallback="candidate"))


def _decision_state(decision: Any, report: Any) -> str:
    if decision is not None:
        state = getattr(decision, "state", None) or (
            decision.get("state") if isinstance(decision, dict) else None
        )
        if state is not None:
            return str(getattr(state, "value", state))
    if report is not None:
        outcome = getattr(report, "outcome", None)
        if outcome is not None:
            return str(getattr(outcome.state, "value", outcome.state))
    return "rejected"


def _sample_dir(context: ExecutionContext, sample_id: str) -> Path:
    workspace = context.io.store.workspace
    return workspace.sample_dir(context.job_id, sample_id)


def _write(context: ExecutionContext, value: Any, target: Path) -> str:
    """Copy an artifact's bytes to a run-directory path."""
    from vidliner.core.results import ArtifactRef

    artifact = (
        value
        if isinstance(value, ArtifactRef)
        else (ArtifactRef.model_validate(value) if isinstance(value, dict) else None)
    )
    if artifact is None:
        return ""
    data = context.io.load(artifact)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return context.io.store.workspace.relative(target)


def _write_json(payload: Any, target: Path) -> str:
    import json

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return target.name


def _item(value: Any) -> ObjectItem:
    if isinstance(value, ObjectItem):
        return value
    if isinstance(value, dict):
        return ObjectItem.model_validate(value)
    raise OperatorFailure(
        f"export needs a target item, received {type(value).__name__}",
        code=ErrorCode.PORT_TYPE_MISMATCH,
    )


def register_all(registry: Any) -> None:
    """Register this module's operators in ``registry``."""
    registry.register(MaterialiseSampleOperator)
