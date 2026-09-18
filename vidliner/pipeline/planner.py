"""Planning: compile a recipe, resolve capabilities, and estimate the work — without running it.

``vidliner plan`` and ``vidliner run --dry-run`` both go through here. The rule is absolute: planning
never calls a generative backend and never spends money. It resolves capability *bindings*, counts
nodes, and multiplies declared estimates.
"""

from __future__ import annotations

from dataclasses import dataclass

from vidliner.capabilities.names import describe_capability
from vidliner.core.graph import OperationGraph
from vidliner.domain.enums import StageName
from vidliner.domain.jobs import JobPlan, PlanEstimate, StageStat
from vidliner.domain.recipe import Recipe
from vidliner.pipeline.assemble import AssemblyResult, assemble_graph
from vidliner.runtime.registry import BackendRegistry, ResolutionReport

__all__ = ["CompiledPlan", "build_plan", "estimate_from_graph"]


@dataclass(frozen=True, slots=True)
class CompiledPlan:
    """The compiled graph, its resolved bindings, and its estimate."""

    recipe: Recipe
    assembly: AssemblyResult
    resolution: ResolutionReport
    job_plan: JobPlan

    @property
    def graph(self) -> OperationGraph:
        """The compiled operation graph."""
        return self.assembly.graph

    @property
    def is_runnable(self) -> bool:
        """Whether every required capability resolved to a backend."""
        return self.resolution.is_complete

    def unmet_explanation(self) -> tuple[str, ...]:
        """Human-readable lines explaining every unresolved capability."""
        lines: list[str] = []
        for capability in self.resolution.unmet:
            try:
                description = describe_capability(capability)
            except KeyError:  # pragma: no cover - the vocabulary is closed
                description = "unknown capability"
            lines.append(f"{capability}: {description} — no configured backend provides it")
        return tuple(lines)


def build_plan(
    recipe: Recipe,
    sample_ids: tuple[str, ...],
    registry: BackendRegistry,
    *,
    instantiate: bool = True,
    job_id: str = "plan",
) -> CompiledPlan:
    """Compile a recipe into a plan and resolve its capabilities.

    Args:
        recipe: the recipe to compile.
        sample_ids: the samples the job will process.
        registry: the runtime profile's backend registry.
        instantiate: whether capability resolution may import backend modules. ``plan`` passes
            ``False`` when it must not import heavy dependencies, accepting a less certain answer
            for capabilities that are only advertised at run time.
        job_id: identifier recorded in the plan document.

    Raises:
        ValidationFailure: when the graph itself is invalid.
    """
    assembly = assemble_graph(recipe, sample_ids)
    resolution = registry.resolve(assembly.graph.capabilities(), instantiate=instantiate)
    estimate = estimate_from_graph(
        assembly.graph,
        registry=registry,
        samples=len(sample_ids),
        registry_hint=recipe.estimates,
    )
    plan = JobPlan(
        job_id=job_id,
        recipe_hash=recipe.recipe_hash(),
        graph_hash=_graph_hash(assembly.graph),
        node_count=len(assembly.graph.nodes),
        capability_bindings=resolution.as_mapping(),
        unmet_capabilities=resolution.unmet,
        estimate=estimate,
        dry_run=not resolution.is_complete,
    )
    return CompiledPlan(recipe=recipe, assembly=assembly, resolution=resolution, job_plan=plan)


def estimate_from_graph(
    graph: OperationGraph,
    *,
    registry: BackendRegistry,
    samples: int,
    registry_hint: object | None = None,
) -> PlanEstimate:
    """Estimate nodes, external calls, accelerator work, time, and cost.

    Estimates come from the runtime profile's per-backend declarations, never from a measurement.
    When a backend declares nothing, the stage is counted but contributes zero cost and zero time,
    and that fact is recorded in the notes rather than hidden behind a fabricated average.
    """
    profile = registry.profile
    per_backend: dict[str, int] = {}
    stages: dict[StageName, dict[str, float]] = {}
    external_calls = 0
    accelerator_operations = 0
    estimated_seconds = 0.0
    estimated_cost = 0.0
    currency: str | None = None
    missing_estimates: set[str] = set()

    for node in graph.nodes:
        bucket = stages.setdefault(
            node.stage,
            {"nodes": 0.0, "external": 0.0, "accelerator": 0.0, "seconds": 0.0, "cost": 0.0},
        )
        bucket["nodes"] += 1
        backend_name = profile.binding_for(node.needs[0]) if node.needs else None
        if backend_name is None and node.needs:
            index = registry.capability_index(instantiate=False)
            candidates = [name for capability in node.needs for name in index.get(capability, [])]
            backend_name = candidates[0] if len(set(candidates)) == 1 else None
        if backend_name is None:
            if node.needs:
                missing_estimates.add(node.needs[0])
            continue
        per_backend[backend_name] = per_backend.get(backend_name, 0) + 1
        spec = profile.backends.get(backend_name)
        if spec is None:
            continue
        estimate = spec.estimates
        bucket["seconds"] += estimate.unit_seconds
        bucket["cost"] += estimate.unit_cost
        estimated_seconds += estimate.unit_seconds
        estimated_cost += estimate.unit_cost
        if estimate.currency:
            currency = estimate.currency
        if estimate.external:
            external_calls += 1
            bucket["external"] += 1
        if estimate.accelerator or profile.device_for(backend_name) in {"cuda", "mps"}:
            accelerator_operations += 1
            bucket["accelerator"] += 1

    node_count = len(graph.nodes)
    candidate_nodes = sum(1 for node in graph.nodes if node.stage is StageName.GENERATE)
    notes: list[str] = []
    if missing_estimates:
        notes.append("no estimate is available for capabilities: " + ", ".join(sorted(missing_estimates)))
    if estimated_cost == 0.0 and candidate_nodes:
        notes.append("every bound generative backend reports zero cost")
    if registry_hint is not None:
        generation_unit_cost = float(getattr(registry_hint, "generation_unit_cost", 0.0) or 0.0)
        generation_unit_seconds = float(getattr(registry_hint, "generation_unit_seconds", 0.0) or 0.0)
        hint_currency = getattr(registry_hint, "currency", None)
        if candidate_nodes and generation_unit_cost:
            estimated_cost += candidate_nodes * generation_unit_cost
            notes.append(
                f"recipe-supplied estimate added {candidate_nodes} x {generation_unit_cost} "
                f"{hint_currency or ''}".rstrip()
            )
        if candidate_nodes and generation_unit_seconds:
            estimated_seconds += candidate_nodes * generation_unit_seconds
        if hint_currency:
            currency = str(hint_currency)

    stage_stats = tuple(
        StageStat(
            stage=stage,
            nodes=int(values["nodes"]),
            external_nodes=int(values["external"]),
            accelerator_nodes=int(values["accelerator"]),
            estimated_seconds=round(
                values["seconds"] + _hint_seconds(stage, registry_hint, values["nodes"]), 3
            ),
            estimated_cost=round(values["cost"], 6),
        )
        for stage, values in sorted(stages.items(), key=lambda item: _stage_rank(item[0]))
    )
    return PlanEstimate(
        samples=samples,
        targets=samples * 1,
        candidates=candidate_nodes,
        nodes=node_count,
        external_calls=external_calls,
        accelerator_operations=accelerator_operations,
        estimated_seconds=round(estimated_seconds, 3),
        estimated_cost=round(estimated_cost, 6),
        currency=currency,
        stages=stage_stats,
        per_backend=per_backend,
        notes=tuple(notes),
    )


def _hint_seconds(stage: StageName, hint: object | None, nodes: float) -> float:
    if hint is None or stage is not StageName.GENERATE:
        return 0.0
    unit = float(getattr(hint, "generation_unit_seconds", 0.0) or 0.0)
    return unit * nodes if unit else 0.0


def _stage_rank(stage: StageName) -> int:
    order = (
        StageName.INGEST,
        StageName.DETECT,
        StageName.SELECT,
        StageName.SEGMENT,
        StageName.SCENE,
        StageName.PLAN,
        StageName.GENERATE,
        StageName.REFINE,
        StageName.VERIFY,
        StageName.EVALUATE,
        StageName.ANNOTATE,
        StageName.EXPORT,
    )
    return order.index(stage)


def _graph_hash(graph: OperationGraph) -> str:
    from vidliner.core.canonical import digest_json

    return digest_json(
        [
            {
                "node": node.node_id,
                "operator": node.operator,
                "version": node.operator_version,
                "config": node.config,
                "inputs": {port: binding.encode() for port, binding in node.inputs.items()},
            }
            for node in graph.topological_order()
        ]
    )
