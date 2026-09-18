"""Compilation of a recipe into an operation graph.

The assembler is the only place that knows the *shape* of the pipeline. Operators declare what they
need; the assembler wires them together and derives each node's identity from its lineage path
(``sample → object → candidate``), which is what keeps node ids stable when a recipe changes
somewhere unrelated.

The graph shape:

```
ingest ─┬─ detect ─ select ─┬─ per object: segment ─ scene ─ plan ─┬─ per candidate: generate ─ refine ─ evaluate ─ annotate ─ export
        │                   │                                     │
        └───────────────────┴─────────────────────────────────────┴─ (source image / masks reused by every branch)
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.core.graph import (
    NodeOutput,
    OperationGraph,
    OperationNode,
    RetryPolicy,
    SampleSource,
)
from vidliner.core.identity import (
    lineage_key_candidate,
    lineage_key_object,
    lineage_key_sample,
    node_id,
)
from vidliner.domain.enums import StageName
from vidliner.domain.recipe import Recipe, RefinementStepSpec
from vidliner.operators.registry import describe_operator

__all__ = ["AssemblyResult", "PipelineAssembler", "assemble_graph"]


@dataclass(frozen=True, slots=True)
class AssemblyResult:
    """The compiled graph plus the facts the planner and the reports need."""

    graph: OperationGraph
    node_ids: dict[str, tuple[str, ...]]
    """Named node groups: ``sample``, ``detect``, ``select``, ``scene``, ``plan``, ``candidate``, ``export``."""

    def nodes(self, group: str) -> tuple[str, ...]:
        """Node ids of one group, empty when the group does not exist."""
        return self.node_ids.get(group, ())


class PipelineAssembler:
    """Builds an :class:`~vidliner.core.graph.OperationGraph` from a recipe and a sample list."""

    def __init__(self, recipe: Recipe) -> None:
        self._recipe = recipe
        self._nodes: list[OperationNode] = []
        self._groups: dict[str, list[str]] = {}

    def assemble(self, sample_ids: tuple[str, ...]) -> AssemblyResult:
        """Compile the graph for the given samples.

        Args:
            sample_ids: identities of the source samples the job will process.

        Returns:
            The compiled graph and its named node groups.
        """
        if not sample_ids:
            raise ValidationFailure(
                "cannot assemble a pipeline with no samples",
                code=ErrorCode.EXPORT_INVALID,
            )
        for sample in sample_ids:
            self._assemble_sample(sample)
        graph = OperationGraph(nodes=tuple(self._nodes))
        graph.validate()
        graph.validate_against_operators(describe_operator)
        return AssemblyResult(graph=graph, node_ids={name: tuple(ids) for name, ids in self._groups.items()})

    # -- per sample -------------------------------------------------------- #

    def _assemble_sample(self, sample: str) -> None:
        recipe = self._recipe
        sample_lineage = lineage_key_sample(sample)
        ingest = self._add(
            "ingest.image",
            lineage=sample_lineage,
            stage=StageName.INGEST,
            group="sample",
            config={"reject_video": True},
            inputs={
                "source": SampleSource("source_asset"),
                "annotation": SampleSource("source_annotation"),
            },
        )
        detect = self._add(
            "detect.objects",
            lineage=sample_lineage,
            stage=StageName.DETECT,
            group="detect",
            config={
                "classes": list(recipe.target.classes),
                "min_score": float(recipe.target.min_score),
                "max_objects": int(max(8, recipe.target.max_per_sample * 8)),
            },
            inputs={
                "image": NodeOutput(ingest, "image"),
                "asset": NodeOutput(ingest, "asset"),
                "annotation": NodeOutput(ingest, "annotation"),
            },
        )
        select = self._add(
            "select.targets",
            lineage=sample_lineage,
            stage=StageName.SELECT,
            group="select",
            config={
                "classes": list(recipe.target.classes),
                "min_score": float(recipe.target.min_score),
                "min_area_px": int(recipe.target.min_area_px),
                "max_per_sample": int(recipe.target.max_per_sample),
                "top_k_by": str(recipe.target.top_k_by),
                "strategy": str(recipe.target.strategy),
                "object_ids": list(recipe.target.object_ids),
            },
            inputs={"instances": NodeOutput(detect, "instances")},
        )
        self._groups.setdefault("export", [])
        for key in self._object_keys(sample):
            self._assemble_object(sample, key, ingest, select)

    def _object_keys(self, sample: str) -> tuple[str, ...]:
        """Object keys this sample fans out to.

        How many objects a sample actually contains is only known after detection runs, so the
        assembler reserves a fixed number of object branches and the runtime *skips* the branches a
        sample does not fill. A fixed number keeps node identities stable across runs, which is what
        makes cache reuse and resume work at all.

        The reserved count is the number of candidates per object times the object budget, capped by
        the recipe: enough to cover every realistic case without building a graph for objects that
        cannot exist.
        """
        return tuple(f"{sample}#object{index}" for index in range(self._recipe.target.max_per_sample))

    def _assemble_object(self, sample: str, object_key: str, ingest: str, select: str) -> None:
        recipe = self._recipe
        lineage = lineage_key_object(sample, object_key)
        segment = self._add(
            "segment.target",
            lineage=lineage,
            stage=StageName.SEGMENT,
            group="segment",
            config={
                "prompt": "bbox",
                "min_area_px": int(max(16, recipe.target.min_area_px // 8)),
                "max_coverage": 0.98,
            },
            inputs={
                "image": NodeOutput(ingest, "image"),
                "asset": NodeOutput(ingest, "asset"),
                "item": NodeOutput(select, "items"),
            },
            selector=object_key,
            selects=("item",),
        )
        scene = self._add(
            "scene.analyse",
            lineage=lineage,
            stage=StageName.SCENE,
            group="scene",
            config={"neighbours": 8},
            inputs={
                "image": NodeOutput(ingest, "image"),
                "asset": NodeOutput(ingest, "asset"),
                "item": NodeOutput(segment, "item"),
                "instances": NodeOutput(select, "selection"),
            },
            selector=object_key,
            selects=("item",),
        )
        plan = self._add(
            "plan.replacement",
            lineage=lineage,
            stage=StageName.PLAN,
            group="plan",
            config={
                "values": list(recipe.replacement.values),
                "description": recipe.replacement.description,
                "references": list(recipe.replacement.references),
                "candidates_per_object": int(recipe.replacement.candidates_per_object),
                "mode": recipe.replacement.mode,
                "strategy": recipe.replacement.strategy,
                "preserve": {
                    "pose": recipe.preserve.pose,
                    "scale": recipe.preserve.scale,
                    "position": recipe.preserve.position,
                    "lighting": recipe.preserve.lighting,
                    "occlusion": recipe.preserve.occlusion,
                },
                "allow_geometry_change": False,
                "mask_expansion_px": int(recipe.replacement.mask_expansion_px),
                "prompt_template": recipe.replacement.prompt_template,
                "negative_constraints": list(recipe.replacement.negative_constraints),
            },
            inputs={"item": NodeOutput(segment, "item"), "scene": NodeOutput(scene, "scene")},
            selector=object_key,
        )
        for candidate_index in range(recipe.replacement.candidates_per_object):
            self._assemble_candidate(sample, object_key, candidate_index, ingest, segment, plan)

    def _assemble_candidate(
        self,
        sample: str,
        object_key: str,
        candidate_index: int,
        ingest: str,
        segment: str,
        plan: str,
    ) -> None:
        recipe = self._recipe
        candidate_key = f"candidate-{candidate_index}"
        lineage = lineage_key_candidate(sample, object_key, candidate_key)
        category = _category_for(recipe, candidate_index)
        generate = self._add(
            "generate.replacement",
            lineage=lineage,
            stage=StageName.GENERATE,
            group="candidate",
            config={
                "candidate_key": candidate_key,
                "category": category,
                "description": recipe.replacement.description,
                "ordinal": int(candidate_index),
                "intensity": 1.0,
                "parameters": {},
                "context_padding_px": recipe.replacement.context_padding_px,
            },
            inputs={
                "item": NodeOutput(segment, "item"),
                "plan": NodeOutput(plan, "plan"),
                "source_image": NodeOutput(ingest, "image"),
            },
            selector=object_key,
            cacheable=False,
        )
        refine = self._add(
            "refine.candidate",
            lineage=lineage,
            stage=StageName.REFINE,
            group="refine",
            config=_refine_config(recipe.refine.steps),
            inputs={
                "item": NodeOutput(segment, "item"),
                "candidate": NodeOutput(generate, "candidate"),
                "mask": NodeOutput(segment, "mask"),
                "source_image": NodeOutput(ingest, "image"),
            },
            selector=object_key,
        )
        # Closing the loop: the object that the generator actually produced is found and masked in
        # the generated image, so every downstream check and the exported label describe what is
        # there rather than what was asked for.
        verify = self._add(
            "verify.redetect",
            lineage=lineage,
            stage=StageName.VERIFY,
            group="verify",
            config={
                # Look for the class the object *had*: a detector's label set is fixed, and a
                # replacement category such as "sedan" is usually not in it. Whether the replacement
                # belongs to the requested category is the semantic evaluator's verdict.
                "detect_class": "source",
                "min_iou": 0.15,
                "min_score": 0.10,
                "require_same_class": True,
                "max_objects": 16,
                "prompt": "bbox",
                "max_coverage": 0.98,
                "max_polygon_points": 96,
            },
            inputs={
                "image": NodeOutput(refine, "image"),
                "item": NodeOutput(segment, "item"),
                "mask": NodeOutput(refine, "mask"),
                "plan": NodeOutput(plan, "plan"),
                "candidate": NodeOutput(generate, "candidate"),
            },
            selector=object_key,
        )
        evaluate = self._add(
            "evaluate.candidate",
            lineage=lineage,
            stage=StageName.EVALUATE,
            group="evaluate",
            config={
                "minimum_target_presence": float(recipe.quality.minimum_target_presence),
                "background_change_ceiling": float(recipe.quality.background_change_ceiling),
                "min_annotation_area_px": int(recipe.export.min_annotation_area_px),
                "artifact_threshold": float(recipe.quality.maximum_artifact_score),
                "use_vision_evaluator": True,
                "hard_gates": dict(recipe.quality.hard_gates),
                "warn_gates": dict(recipe.quality.warn_gates),
                "geometry": {
                    "centroid_shift_max": float(recipe.preserve.geometry_tolerance_centroid),
                    "area_ratio_min": float(recipe.preserve.geometry_tolerance_area_min),
                    "area_ratio_max": float(recipe.preserve.geometry_tolerance_area_max),
                    "aspect_ratio_delta_max": float(recipe.preserve.geometry_tolerance_aspect),
                    "ground_contact_max_px": float(recipe.preserve.geometry_tolerance_ground_px),
                },
            },
            inputs={
                "source_image": NodeOutput(ingest, "image"),
                "image": NodeOutput(refine, "image"),
                "source_mask": NodeOutput(segment, "mask"),
                "input_mask": NodeOutput(refine, "mask"),
                "mask": NodeOutput(verify, "mask"),
                "item": NodeOutput(verify, "item"),
                "found": NodeOutput(verify, "found"),
                "candidate": NodeOutput(generate, "candidate"),
                "plan": NodeOutput(plan, "plan"),
            },
            selector=object_key,
        )
        annotate = self._add(
            "annotate.rebuild",
            lineage=lineage,
            stage=StageName.ANNOTATE,
            group="annotate",
            config={"min_area_px": recipe.export.min_annotation_area_px, "inherit_other_objects": True},
            inputs={
                "item": NodeOutput(verify, "item"),
                "mask": NodeOutput(verify, "mask"),
                "found": NodeOutput(verify, "found"),
                "plan": NodeOutput(plan, "plan"),
                "candidate": NodeOutput(generate, "candidate"),
                "source_annotation": SampleSource("source_annotation"),
                "source_image": NodeOutput(ingest, "image"),
                "image": NodeOutput(refine, "image"),
            },
            selector=object_key,
        )
        export = self._add(
            "export.sample",
            lineage=lineage,
            stage=StageName.EXPORT,
            group="export",
            config={"keep_rejected": recipe.acceptance.on_quality_reject == "keep_diagnostics"},
            inputs={
                "item": NodeOutput(segment, "item"),
                "candidate": NodeOutput(generate, "candidate"),
                "image": NodeOutput(refine, "image"),
                "source_image": NodeOutput(ingest, "image"),
                "annotation": NodeOutput(annotate, "annotation"),
                "report": NodeOutput(evaluate, "report"),
                "decision": NodeOutput(evaluate, "report"),
            },
            selector=object_key,
        )
        self._attribute(lineage, export)

    # -- helpers ----------------------------------------------------------- #

    def _add(
        self,
        operator: str,
        *,
        lineage: str,
        stage: StageName,
        group: str,
        config: dict[str, object],
        inputs: dict[str, object],
        selector: str | None = None,
        cacheable: bool | None = None,
        selects: tuple[str, ...] = (),
    ) -> str:
        """Create one node with a deterministic identity and register it in its group."""
        spec = describe_operator(operator)
        identifier = node_id(operator, f"{lineage}/{selector}" if selector else lineage)
        index = _branch_index(selector)
        node = OperationNode(
            node_id=identifier,
            operator=spec.name,
            operator_version=spec.version,
            stage=stage,
            inputs=cast("dict[str, Any]", dict(inputs)),
            outputs={name: port.port_type for name, port in spec.outputs.items()},
            config=dict(config),
            needs=spec.capabilities,
            name=spec.name,
            determinism=spec.determinism,
            cacheable=spec.cacheable if cacheable is None else cacheable,
            retry=RetryPolicy(attempts=spec.retry_attempts),
            timeout_s=spec.timeout_s,
            max_parallelism=spec.max_parallelism,
            lineage=tuple(lineage.split("/")),
            selects=tuple((port, index) for port in selects),
            description=spec.summary,
        )
        self._nodes.append(node)
        self._groups.setdefault(group, []).append(identifier)
        return identifier

    def _attribute(self, lineage: str, node: str) -> None:
        """Record a lineage-to-node mapping, used by the reports and by ``plan``."""
        self._groups.setdefault(f"lineage:{lineage}", []).append(node)


def _branch_index(selector: str | None) -> int:
    """The object ordinal encoded in a branch selector such as ``s_ab12#object2``."""
    if not selector or "#object" not in selector:
        return 0
    _, _, raw = selector.partition("#object")
    try:
        return int(raw)
    except ValueError:
        return 0


def _refine_config(steps: tuple[RefinementStepSpec, ...]) -> dict[str, object]:
    """Translate the recipe's refinement steps into the refinement operator's configuration."""
    feather = 3
    close_px = 5
    dilate_px = 0
    harmonize = False
    harmonize_strength = 0.45
    composite = True
    for step in steps:
        if not step.enabled:
            continue
        if step.operator == "refine.mask_edges":
            feather = int(str(step.config.get("feather_px", feather)))
            close_px = int(str(step.config.get("close_px", close_px)))
            dilate_px = int(str(step.config.get("dilate_px", dilate_px)))
        elif step.operator == "refine.alpha_blend":
            composite = True
        elif step.operator == "refine.harmonize":
            harmonize = True
            harmonize_strength = float(str(step.config.get("strength", harmonize_strength)))
        else:
            raise ValidationFailure(
                f"unknown refinement operator {step.operator!r} in the recipe",
                code=ErrorCode.OPERATOR_UNKNOWN,
                detail={"operator": step.operator},
            )
    return {
        "feather_px": feather,
        "close_px": close_px,
        "dilate_px": dilate_px,
        "harmonize": harmonize,
        "harmonize_strength": harmonize_strength,
        "composite": composite,
    }


def _category_for(recipe: Recipe, candidate_index: int) -> str:
    """The category a given candidate index will target.

    The planner may override this once it has measured the scene (for example when the recipe names
    a single value), but the node's declared config must be complete before the plan exists, so the
    assembler states the recipe's own intent here and the planner reconciles it at run time.
    """
    values = list(recipe.replacement.values)
    if not values:
        return recipe.replacement.description or "object"
    return values[candidate_index % len(values)]


def assemble_graph(recipe: Recipe, sample_ids: tuple[str, ...]) -> AssemblyResult:
    """Compile ``recipe`` into an operation graph for ``sample_ids``."""
    return PipelineAssembler(recipe).assemble(sample_ids)
