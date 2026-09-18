"""Rule-based replacement planning.

Planning is a decision, not a model call. Given the recipe's intent and the measured scene context,
this backend produces the exact plan the rest of the pipeline executes: what changes, what must
remain, how much to expand the mask, what the prompt says, what the negative constraints are, how
many candidates to generate, and what seed each one uses.

Keeping planning rule-based and deterministic for the default profile is deliberate: ``vidliner plan``
must be free, and a plan must be reproducible from the manifest. A learned planner can be bound in
its place without touching the pipeline.
"""

from __future__ import annotations

from typing import Any

from vidliner.backends.base import LocalBackend
from vidliner.capabilities.backend import BackendProbe, PipelineContext, PlanningRequest
from vidliner.capabilities.names import CAP_PLANNING
from vidliner.core.identity import node_id
from vidliner.core.seedtree import SeedTree
from vidliner.domain.enums import Determinism
from vidliner.domain.replacement import (
    CandidateSpec,
    GeometryBudget,
    PreservationRules,
    ReplacementPlan,
)

__all__ = ["RuleBasedPlannerBackend"]


class RuleBasedPlannerBackend(LocalBackend):
    """Turn an intent plus a measured scene into an executable plan."""

    backend_id = "rule_based_planner"
    backend_version = "1.0.0"
    capabilities = (CAP_PLANNING,)
    determinism = Determinism.DETERMINISTIC
    declared_capabilities = (CAP_PLANNING,)

    def __init__(self, spec: Any, options: dict[str, Any], credentials: object | None = None) -> None:
        super().__init__(spec, options, credentials)
        self._sync = _PlannerSync(self)

    async def plan(self, request: PlanningRequest, context: PipelineContext) -> ReplacementPlan:
        """Protocol method; the runtime executes ``self._sync.plan`` on a worker thread."""
        return self._sync.plan(request, context)

    async def probe(self) -> BackendProbe:
        """Report readiness. This planner calls no model, by construction."""
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
            message="deterministic rule-based planner; makes no model calls",
            details=probe.details,
        )


class _PlannerSync:
    """Synchronous implementation of the planner."""

    def __init__(self, backend: RuleBasedPlannerBackend) -> None:
        self._backend = backend

    def plan(self, request: PlanningRequest, context: PipelineContext) -> ReplacementPlan:
        """Build the plan for one target object."""
        intent = request.intent
        scene = request.scene
        tree = SeedTree(context.seed)
        values = self._candidate_values(request)
        specs: list[CandidateSpec] = []
        for ordinal, value in enumerate(values[: request.candidates_per_object]):
            category = value or intent.replacement_category
            description = self._describe(category, intent)
            candidate_key = _candidate_key(category, ordinal)
            specs.append(
                CandidateSpec(
                    candidate_key=candidate_key,
                    category=category,
                    description=description,
                    seed=tree.for_candidate(request.sample_id, intent.target_object, candidate_key).seed,
                    ordinal=ordinal,
                    reference=intent.reference,
                    parameters={"mode": intent.mode.value, "intensity": self._intensity(intent)},
                )
            )
        prompt = self._prompt(request, scene.object_category, specs[0].description if specs else "")
        return ReplacementPlan(
            plan_id=node_id("plan", f"sample:{request.sample_id}/object:{intent.target_object}"),
            sample_id=request.sample_id,
            intent=intent,
            expected_category=specs[0].category if specs else intent.replacement_category,
            positive_prompt=prompt,
            negative_constraints=self._negatives(request),
            mask_expansion_px=self._mask_expansion(request),
            depth_expansion_px=0,
            preservation=self._preservation(intent),
            geometry_budget=self._geometry_budget(intent),
            candidate_specs=tuple(specs),
            plan_seed=context.seed,
            planner_id=self._backend.backend_id,
            notes=self._notes(scene),
        )

    def _candidate_values(self, request: PlanningRequest) -> list[str]:
        """The replacement value for each candidate the plan must produce.

        The recipe's declared values are used in order; when the recipe names fewer values than there
        are candidates, the list is cycled. Cycling by *index* rather than generating new names keeps
        each candidate's key — and therefore its identity, seed, and cache entry — stable when a
        recipe later adds a fourth value.
        """
        intent = request.intent
        base: list[str] = [value for value in request.candidate_values if value]
        if not base and intent.replacement_category:
            base.append(intent.replacement_category)
        hint = intent.replacement_description.strip()
        if hint and hint not in base:
            base.append(hint)
        if not base:
            base = [intent.replacement_category or "object"]
        return [base[index % len(base)] for index in range(request.candidates_per_object)]

    def _describe(self, category: str, intent: object) -> str:
        description = getattr(intent, "replacement_description", "") or ""
        if description and description != category:
            return f"{category} ({description})"
        return category

    def _prompt(self, request: PlanningRequest, source_category: str, description: str) -> str:
        scene = request.scene
        lighting = scene.lighting
        light_phrase = "the existing lighting"
        if lighting.direction_deg is not None:
            light_phrase = f"light coming from {lighting.direction_deg:.0f} degrees"
        elif lighting.intensity is not None:
            light_phrase = "the existing lighting intensity"
        template = request.prompt_template
        try:
            return template.format(
                category=description or request.intent.replacement_category,
                description=description or request.intent.replacement_category,
                source=source_category,
                lighting=light_phrase,
                background=scene.background_summary,
            )
        except KeyError:
            # A template with an unknown placeholder is the user's business, not a crash: fall back
            # to the documented default phrasing so the job still runs, and record why.
            return (
                f"Replace the {source_category} in the masked region with "
                f"{description or request.intent.replacement_category}. Keep the background, camera "
                "viewpoint, lighting direction, shadows, and ground contact unchanged."
            )

    def _negatives(self, request: PlanningRequest) -> tuple[str, ...]:
        negatives = list(request.negative_constraints)
        if request.intent.preserve_pose:
            negatives.append("no change of pose")
        if request.intent.preserve_scale:
            negatives.append("no change of object scale")
        if request.intent.preserve_occlusion:
            negatives.append("do not remove or move occluding objects")
        return tuple(dict.fromkeys(negatives))

    def _mask_expansion(self, request: PlanningRequest) -> int:
        """Grow the mask slightly so shadows and contact edges are regenerated with the object."""
        base = int(request.mask_expansion_px)
        scene = request.scene
        if scene.ground_contact is not None:
            base = max(base, int(self._backend.option("ground_contact_expansion_px", 6)))
        if scene.shadow is not None:
            base = max(base, int(self._backend.option("shadow_expansion_px", 4)))
        return min(base, int(self._backend.option("max_expansion_px", 32)))

    def _preservation(self, intent: object) -> PreservationRules:
        return PreservationRules(
            background="strict",
            geometry=bool(getattr(intent, "preserve_position", True)),
            lighting=bool(getattr(intent, "preserve_lighting", True)),
            pose=bool(getattr(intent, "preserve_pose", True)),
            scale=bool(getattr(intent, "preserve_scale", True)),
            position=bool(getattr(intent, "preserve_position", True)),
            occlusion=bool(getattr(intent, "preserve_occlusion", True)),
        )

    def _geometry_budget(self, intent: object) -> GeometryBudget:
        """Tighten the geometry budget in strict mode and widen it when the user allowed deformation."""
        width = float(self._backend.option("strict_centroid_shift_max", 0.08))
        area_min = float(self._backend.option("strict_area_ratio_min", 0.70))
        area_max = float(self._backend.option("strict_area_ratio_max", 1.45))
        aspect = float(self._backend.option("strict_aspect_delta_max", 0.30))
        creative = _mode_value(intent) == "creative"
        if getattr(intent, "allowed_geometry_change", False) or creative:
            width *= 2.0
            area_min *= 0.7
            area_max *= 1.4
            aspect = min(1.0, aspect * 2.0)
        return GeometryBudget(
            centroid_shift_max=width,
            area_ratio_min=area_min,
            area_ratio_max=area_max,
            aspect_ratio_delta_max=aspect,
            ground_contact_max_px=float(self._backend.option("ground_contact_max_px", 24.0)),
        )

    def _intensity(self, intent: object) -> float:
        """How strongly to apply the edit: creative replacements get a gentler touch."""
        return 0.6 if _mode_value(intent) == "creative" else 1.0

    def _notes(self, scene: object) -> tuple[str, ...]:
        notes: list[str] = []
        if getattr(scene, "is_occluded", False):
            notes.append("target is partially occluded; preservation of occluders is enforced")
        if getattr(scene, "is_near_frame_edge", False):
            notes.append("target touches the frame edge; geometry budget is the limiting factor")
        if getattr(scene, "shadow", None) is not None:
            notes.append("a shadow region was measured below the object and is included in the mask")
        return tuple(notes)


def _mode_value(intent: object) -> str:
    """The mode's string value, whether the intent carries an enum or a plain string."""
    mode = getattr(intent, "mode", None)
    value = getattr(mode, "value", mode)
    return str(value) if value is not None else ""


def _candidate_key(category: str, ordinal: int) -> str:
    slug = "".join(char if char.isalnum() else "_" for char in category.strip().lower()) or "replacement"
    return f"{slug}-{ordinal + 1}"
