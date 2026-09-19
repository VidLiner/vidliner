"""The job service: one place that knows how to run VidLiner.

Every user-facing command is a thin wrapper over a function here, so the CLI, the tests, and an
embedding application all get identical behaviour — including the parts that are easy to get subtly
wrong (which profile is loaded, how a job id is derived, when a job may be resumed, and when the
accepted set is exported).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vidliner.capabilities.names import PRODUCTION_CAPABILITIES
from vidliner.control.prepare import build_sample_contexts, load_source_annotations
from vidliner.control.sources import DatasetSource, DiscoveryResult
from vidliner.core.errors import ErrorCode, InfrastructureFailure, ValidationFailure
from vidliner.core.identity import object_digest
from vidliner.core.results import JobCounters, utc_now
from vidliner.domain.enums import DecisionState, JobState
from vidliner.domain.jobs import JobManifest
from vidliner.domain.provenance import ProvenanceRecord, SeedRecord
from vidliner.domain.quality import QualityReport
from vidliner.domain.recipe import Recipe
from vidliner.pipeline.export_names import kinds_for, matches_kind
from vidliner.pipeline.exporter import DatasetExporter, ExportOutcome, ExportRecord
from vidliner.pipeline.planner import CompiledPlan, build_plan
from vidliner.pipeline.runner import JobOptions, JobRunner, SampleOutcome
from vidliner.runtime.production import (
    DemoUsage,
    ProductionVerdict,
    assert_production_ready,
    assess_production_readiness,
)
from vidliner.runtime.profile import RuntimeProfile, default_profile, load_profile
from vidliner.runtime.registry import BackendRegistry
from vidliner.storage.state import StateStore
from vidliner.storage.workspace import ArtifactStore, Workspace

__all__ = [
    "JobRequest",
    "RunOutcome",
    "Session",
    "load_recipe",
    "open_session",
    "write_default_profile",
]

DEFAULT_RUNTIME_FILE = "runtime.yaml"
DEFAULT_RECIPE_FILE = "recipe.yaml"


# --------------------------------------------------------------------------- #
# Recipe and profile loading
# --------------------------------------------------------------------------- #


def load_recipe(path: Path) -> Recipe:
    """Load and validate a recipe document.

    Raises:
        ValidationFailure: when the file is missing, unparseable, or fails schema validation.
    """
    import yaml

    if not path.is_file():
        raise ValidationFailure(
            f"recipe file {path} does not exist",
            code=ErrorCode.RECIPE_INVALID,
            detail={"path": str(path)},
        )
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValidationFailure(
            f"recipe {path} is not valid YAML: {exc}",
            code=ErrorCode.RECIPE_INVALID,
            detail={"path": str(path)},
        ) from exc
    if not isinstance(document, dict):
        raise ValidationFailure(
            f"recipe {path} must contain a mapping at the top level",
            code=ErrorCode.RECIPE_INVALID,
        )
    try:
        return Recipe.model_validate(document)
    except Exception as exc:  # pydantic ValidationError
        raise ValidationFailure(
            f"recipe {path} is invalid: {exc}",
            code=ErrorCode.RECIPE_INVALID,
            detail={"path": str(path)},
        ) from exc


def write_default_profile(target: Path, *, force: bool = False) -> Path:
    """Write a starter runtime profile, refusing to overwrite an existing one unless forced."""
    if target.exists() and not force:
        raise ValidationFailure(
            f"runtime profile {target} already exists; pass --force to replace it",
            code=ErrorCode.RUNTIME_INVALID,
            detail={"path": str(target)},
        )
    import yaml

    profile = default_profile()
    payload = profile.model_dump(mode="json", exclude={"source_path"})
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #


@dataclass
class JobRequest:
    """Everything a run needs beyond the session itself."""

    recipe: Recipe
    options: JobOptions = field(default_factory=JobOptions)
    limit: int = 0
    job_id: str | None = None
    export: bool = True
    include_review: bool = False


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """The result of a run or plan."""

    manifest: JobManifest
    plan: CompiledPlan
    discovery: DiscoveryResult
    export: ExportOutcome | None = None
    outcomes: dict[str, SampleOutcome] = field(default_factory=dict)

    @property
    def accepted(self) -> int:
        """Accepted candidate count."""
        return self.manifest.counters.accepted

    @property
    def counters(self) -> JobCounters:
        """Counters recorded in the manifest."""
        return self.manifest.counters


class Session:
    """A workspace plus its state store, artifact store, and runtime profile."""

    def __init__(self, workspace: Workspace, profile: RuntimeProfile) -> None:
        self._workspace = workspace
        self._profile = profile
        self._store: ArtifactStore | None = None
        self._state: StateStore | None = None

    # -- construction ------------------------------------------------------ #

    @classmethod
    def open(
        cls,
        workspace_path: Path,
        *,
        runtime_path: Path | None = None,
        create: bool = False,
    ) -> Session:
        """Open a workspace, loading its runtime profile.

        Args:
            workspace_path: directory that holds (or will hold) the workspace.
            runtime_path: explicit profile path; otherwise ``runtime.yaml`` in the workspace, then
                ``VIDLINER_RUNTIME``, then the built-in local profile.
            create: initialise the workspace directories when they are missing.
        """
        workspace = Workspace(workspace_path)
        if create:
            workspace.initialize()
        elif not workspace.is_initialized():
            raise ValidationFailure(
                f"{workspace.root} is not a VidLiner workspace; run 'vidliner init' first",
                code=ErrorCode.WORKSPACE_INVALID,
                detail={"workspace": str(workspace.root)},
            )
        return cls(workspace, _load_profile(workspace, runtime_path))

    @property
    def workspace(self) -> Workspace:
        """The workspace being operated on."""
        return self._workspace

    @property
    def profile(self) -> RuntimeProfile:
        """The runtime profile in effect."""
        return self._profile

    @property
    def store(self) -> ArtifactStore:
        """The artifact store (opened lazily, and initialised on first use)."""
        if self._store is None:
            self._store = ArtifactStore(
                self._workspace,
                max_artifact_bytes=max(1, self._profile.storage.max_artifact_mb) * 1024 * 1024,
            )
        return self._store

    @property
    def state(self) -> StateStore:
        """The SQLite state store (opened and migrated lazily)."""
        if self._state is None:
            path = self._workspace.root / self._profile.storage.state_filename
            self._state = StateStore(path).initialize()
        return self._state

    @property
    def registry(self) -> BackendRegistry:
        """A backend registry for the current profile."""
        return BackendRegistry(self._profile)

    def reload_profile(self, runtime_path: Path | None = None) -> None:
        """Re-read the runtime profile and drop anything derived from the previous one.

        A command opens a session, reads the profile, and uses it for one operation; a long-lived
        embedding of VidLiner may want to switch profiles between operations. The backend registry is
        derived from the profile, so it is dropped here rather than left pointing at stale bindings.
        """
        self._profile = _load_profile(self._workspace, runtime_path)
        self._store = None

    def close(self) -> None:
        """Close the state store if it was opened."""
        if self._state is not None:
            self._state.close()
            self._state = None

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- discovery --------------------------------------------------------- #

    def dataset_root(self, recipe: Recipe) -> Path:
        """Resolve the recipe's dataset directory inside the workspace boundary."""
        return self._workspace.resolve_inside(recipe.dataset.input, must_exist=True)

    def discover(self, recipe: Recipe, *, limit: int = 0) -> DiscoveryResult:
        """Discover the recipe's dataset."""
        root = self.dataset_root(recipe)
        source = DatasetSource(root=root, recipe=recipe)
        discovery = source.discover()
        if limit and len(discovery.samples) > limit:
            trimmed = discovery.samples[:limit]
            return DiscoveryResult(
                root=discovery.root,
                samples=trimmed,
                splits=tuple(sorted({sample.split for sample in trimmed})),
                augment_splits=discovery.augment_splits,
                skipped=discovery.skipped,
                notes=discovery.notes,
            )
        return discovery

    def sample_contexts(self, recipe: Recipe, discovery: DiscoveryResult) -> dict[str, Any]:
        """Build the runtime sample contexts for a discovery result."""
        annotations = load_source_annotations(recipe, discovery)
        return build_sample_contexts(discovery, recipe=recipe, store=self.store, annotations=annotations)

    # -- planning ---------------------------------------------------------- #

    def plan(
        self,
        recipe: Recipe,
        *,
        limit: int = 0,
        job_id: str | None = None,
        instantiate: bool = True,
    ) -> tuple[CompiledPlan, DiscoveryResult]:
        """Compile and cost a recipe without running it."""
        discovery = self.discover(recipe, limit=limit)
        contexts = self.sample_contexts(recipe, discovery)
        if not contexts:
            raise ValidationFailure(
                f"no augmentable sample was found in {discovery.root} for split(s) "
                f"{list(recipe.dataset.augment_splits)}; discovered splits are {list(discovery.splits)}",
                code=ErrorCode.SPLIT_NOT_AUGMENTABLE,
                detail={"splits": list(discovery.splits)},
            )
        identifier = job_id or self.job_id_for(recipe, tuple(contexts))
        plan = build_plan(
            recipe,
            tuple(contexts),
            self.registry,
            instantiate=instantiate,
            job_id=identifier,
        )
        return plan, discovery

    def job_id_for(self, recipe: Recipe, sample_ids: tuple[str, ...]) -> str:
        """Derive an idempotent job id from the recipe, the seed, and the sample set.

        Re-running the same recipe against the same samples therefore resumes the same job instead of
        piling up near-identical run directories; pass an explicit ``--job-id`` (or ``--new``) to
        force a fresh attempt.
        """
        seed = recipe.replacement.seed or 0
        return object_digest(
            "j_",
            {
                "recipe": recipe.recipe_hash(),
                "seed": seed,
                "samples": sorted(sample_ids),
            },
            16,
        )

    # -- running ----------------------------------------------------------- #

    async def run(self, request: JobRequest) -> RunOutcome:
        """Run a job end to end and export its accepted samples."""
        recipe = request.recipe
        plan, discovery = self.plan(recipe, limit=request.limit, job_id=request.job_id)
        if not plan.is_runnable:
            raise ValidationFailure(
                "the job cannot start because these capabilities have no backend: "
                + "; ".join(plan.unmet_explanation()),
                code=ErrorCode.CAPABILITY_UNBOUND,
                detail={"unmet": list(plan.resolution.unmet)},
            )
        # Pre-flight: a job that could never be exported fails in a second, not after paying for
        # candidates nobody may keep. The check is repeated at export, because a plan's bindings and
        # a run's bindings can differ when the profile changed between the two.
        registry = self.registry
        verdict = assess_production_readiness(registry, plan.resolution.as_mapping())
        if not recipe.acceptance.allow_demo_backends:
            assert_production_ready(verdict)
        contexts = self.sample_contexts(recipe, discovery)
        runner = JobRunner(
            recipe=recipe,
            plan=plan,
            workspace=self._workspace,
            store=self.store,
            state=self.state,
            registry=registry,
            sample_contexts=contexts,
            options=request.options,
            demo_verdict=verdict,
        )
        runner.register_sources()
        manifest = await runner.execute()
        outcome = RunOutcome(manifest=manifest, plan=plan, discovery=discovery, outcomes=runner.outcomes)
        if request.options.dry_run:
            # A dry run still writes its plan, so the reviewed graph is on disk exactly as planned.
            self.write_plan_document(outcome)
            return outcome
        if manifest.state is not JobState.SUCCEEDED:
            return outcome
        runner.register_accepted()
        if request.export:
            export = self.export_job(
                manifest.job_id,
                recipe=recipe,
                outcomes=runner.outcomes,
                include_review=request.include_review,
            )
            outcome = RunOutcome(
                manifest=manifest,
                plan=plan,
                discovery=discovery,
                export=export,
                outcomes=runner.outcomes,
            )
        self._write_job_reports(manifest, runner)
        self.write_plan_document(outcome)
        return outcome

    async def resume(self, job_id: str, *, recipe: Recipe | None = None, export: bool = True) -> RunOutcome:
        """Resume a job from its stored manifest and run directory."""
        stored = self.state.require_job(job_id)
        loaded = recipe or Recipe.model_validate(stored.recipe_snapshot)
        if stored.state is JobState.SUCCEEDED:
            raise InfrastructureFailure(
                f"job {job_id} already finished with state {stored.state.value}; nothing to resume",
                code=ErrorCode.JOB_STATE_INVALID,
                detail={"job_id": job_id, "state": stored.state.value},
            )
        plan, discovery = self.plan(loaded, job_id=job_id)
        contexts = self.sample_contexts(loaded, discovery)
        runner = JobRunner(
            recipe=loaded,
            plan=plan,
            workspace=self._workspace,
            store=self.store,
            state=self.state,
            registry=self.registry,
            sample_contexts=contexts,
            options=JobOptions(seed=stored.seed, resume=True, use_cache=True),
            manifest=stored.model_copy(update={"state": JobState.PLANNED}),
        )
        manifest = await runner.execute()
        outcome = RunOutcome(manifest=manifest, plan=plan, discovery=discovery, outcomes=runner.outcomes)
        if manifest.state is JobState.SUCCEEDED and export:
            runner.register_accepted()
            export_outcome = self.export_job(
                manifest.job_id, recipe=loaded, outcomes=runner.outcomes, include_review=False
            )
            outcome = RunOutcome(
                manifest=manifest,
                plan=plan,
                discovery=discovery,
                export=export_outcome,
                outcomes=runner.outcomes,
            )
        self._write_job_reports(manifest, runner)
        self.write_plan_document(outcome)
        return outcome

    # -- queries ----------------------------------------------------------- #

    def jobs(self, *, limit: int = 25, states: tuple[JobState, ...] = ()) -> list[JobManifest]:
        """List jobs, newest first."""
        return self.state.list_jobs(limit=limit, states=states)

    def job(self, job_id: str) -> JobManifest:
        """Load one job manifest."""
        return self.state.require_job(job_id)

    def plan_document(self, job_id: str) -> dict[str, Any] | None:
        """Read a written plan document, when the job produced one."""
        path = self._workspace.run_dir(job_id) / "plan.json"
        if not path.is_file():
            return None
        import json

        document = json.loads(path.read_text(encoding="utf-8"))
        return document if isinstance(document, dict) else None

    def write_plan_document(self, outcome: RunOutcome) -> Path:
        """Write the compiled plan to the job's run directory."""
        import json

        path = self._workspace.run_dir(outcome.manifest.job_id) / "plan.json"
        payload = {
            "plan": outcome.plan.job_plan.model_dump(mode="json"),
            "capability_bindings": outcome.plan.resolution.as_mapping(),
            "unmet_capabilities": list(outcome.plan.resolution.unmet),
            "stages": [
                {"stage": stage.value, "nodes": count}
                for stage, count in outcome.plan.graph.describe_stages()
            ],
            "nodes": [
                {
                    "node_id": node.node_id,
                    "operator": node.operator,
                    "operator_version": node.operator_version,
                    "stage": node.stage.value,
                    "lineage": list(node.lineage),
                    "needs": list(node.needs),
                    "cacheable": node.cacheable,
                    "config": node.config,
                    "inputs": {port: binding.encode() for port, binding in node.inputs.items()},
                }
                for node in outcome.plan.graph.topological_order()
            ],
            "discovery": {
                "root": str(outcome.discovery.root),
                "samples": outcome.discovery.count,
                "splits": outcome.discovery.split_counts,
                "augment_splits": list(outcome.discovery.augment_splits),
            },
            "generated_at": utc_now().isoformat(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.state.record_report(outcome.manifest.job_id, "plan", self._workspace.relative(path))
        return path

    # -- export ------------------------------------------------------------ #

    def export_job(
        self,
        job_id: str,
        *,
        recipe: Recipe | None = None,
        outcomes: dict[str, SampleOutcome] | None = None,
        include_review: bool = False,
        format_override: str | None = None,
        path_override: str | None = None,
    ) -> ExportOutcome:
        """Export a job's accepted samples as a dataset."""
        manifest = self.state.require_job(job_id)
        loaded = recipe or Recipe.model_validate(manifest.recipe_snapshot)
        from vidliner.domain.annotations import ExportProfile

        profile = ExportProfile(
            format=format_override or loaded.export.format,
            path=path_override or loaded.export.path,
            copy_images=loaded.export.copy_images,
            image_format=loaded.export.image_format,
            include_rejected=loaded.export.include_rejected,
            include_provenance=loaded.export.include_provenance,
            include_review_report=loaded.export.include_review_report,
            min_area_px=loaded.export.min_annotation_area_px,
        )
        self._assert_export_allowed(loaded, manifest)
        payload = outcomes if outcomes is not None else self._outcomes_from_state(job_id, loaded)
        records = self.export_records(job_id, payload, loaded, include_review=include_review)
        exporter = DatasetExporter(
            workspace=self._workspace,
            store=self.store,
            state=self.state,
            recipe=loaded,
            profile=profile,
        )
        return exporter.export(records, job_id=job_id, seed_tree=manifest.seed_tree)

    def export_records(
        self,
        job_id: str,
        outcomes: dict[str, SampleOutcome],
        recipe: Recipe,
        *,
        include_review: bool,
    ) -> list[ExportRecord]:
        """Turn recorded outcomes into export records, reading evidence from the run directory."""

        manifest = self.state.require_job(job_id)
        records: list[ExportRecord] = []
        seed_tree = manifest.seed_tree
        for sample_outcome in outcomes.values():
            for accepted in sample_outcome.accepted_records:
                records.append(
                    self._record_for(
                        job_id,
                        manifest=manifest,
                        recipe=recipe,
                        outcome=sample_outcome,
                        entry=accepted,
                        seed_tree=seed_tree,
                        include_review=include_review,
                    )
                )
        return records

    def _record_for(
        self,
        job_id: str,
        *,
        manifest: JobManifest,
        recipe: Recipe,
        outcome: SampleOutcome,
        entry: dict[str, Any],
        seed_tree: dict[str, int],
        include_review: bool,
    ) -> ExportRecord:
        import json

        from vidliner.core.results import ArtifactRef
        from vidliner.domain.annotations import AnnotationBundle
        from vidliner.domain.enums import ArtifactKind

        candidate_identifier = str(entry["candidate_id"])
        sample_dir = self._workspace.sample_dir(job_id, outcome.sample_id)
        annotation_path = _candidate_evidence(sample_dir, "annotation", entry)
        if annotation_path is None:
            raise ValidationFailure(
                f"cannot export {candidate_identifier}: its rebuilt annotation is missing from "
                f"{sample_dir}, so the label for this candidate was never produced",
                code=ErrorCode.ARTIFACT_MISSING,
                detail={"candidate_id": candidate_identifier, "path": str(sample_dir)},
            )
        annotation_document = json.loads(annotation_path.read_text(encoding="utf-8"))
        annotation = AnnotationBundle.model_validate(annotation_document)
        decision = entry.get("decision")
        state = decision.state if hasattr(decision, "state") else DecisionState.ACCEPTED
        if state is DecisionState.NEEDS_REVIEW and not include_review and not recipe.acceptance.export_review:
            state = DecisionState.NEEDS_REVIEW
        metrics = entry.get("metrics") or {}
        quality: QualityReport | None = entry.get("quality")
        output_digest = str(entry.get("output_digest") or "")
        image_artifact = None
        if output_digest:
            image_artifact = ArtifactRef(
                kind=ArtifactKind.REFINED_IMAGE,
                digest=output_digest,
                media_type="image/png",
                size_bytes=self._artifact_size(output_digest),
                suffix="png",
            )
        provenance = ProvenanceRecord(
            source_sample_id=outcome.sample_id,
            source_digest=outcome.source_digest,
            source_path=str(entry.get("source_path") or ""),
            split=outcome.split,
            lineage_root=outcome.source_digest,
            ancestors=(outcome.sample_id,),
            augmentation_depth=1,
            recipe_name=recipe.name,
            recipe_hash=recipe.recipe_hash(),
            job_id=job_id,
            job_seed=manifest.seed,
            target_object_id=str(entry.get("object_id") or ""),
            target_class=str(entry.get("source_class") or entry.get("category") or ""),
            replacement_category=str(entry.get("category") or ""),
            replacement_description=recipe.replacement.description,
            replacement_mode=recipe.replacement.mode,
            intent=dict(entry.get("intent") or {}),
            seeds=SeedRecord(
                job_seed=manifest.seed,
                sample_seed=seed_tree.get(outcome.sample_id, manifest.seed),
                candidate_seed=int(entry.get("seed") or 0),
            ),
            operator_versions={item.operator: item.version for item in manifest.operator_versions},
            backend_ids=dict(manifest.capability_bindings),
            model_ids={},
            generation_parameters={
                "candidate_key": entry.get("candidate_key"),
                "category": entry.get("category"),
            },
            output_digest=output_digest,
            output_path=str(entry.get("output_path") or ""),
            annotation_path=self._workspace.relative(annotation_path),
            annotation_format=recipe.export.format,
            annotation_object_count=annotation.object_count,
            quality_scores={str(key): float(value) for key, value in metrics.items() if value is not None},
            overall_score=entry.get("overall_score"),
            decision=state,
            reason_codes=tuple(str(code) for code in entry.get("reason_codes", ())),
            policy_hash=quality.outcome.policy_hash if quality is not None and quality.outcome else "",
            perceptual_hash=None,
        )
        return ExportRecord(
            candidate_id=candidate_identifier,
            sample_id=str(entry.get("candidate_id")),
            source_sample_id=outcome.sample_id,
            category=str(entry.get("category") or ""),
            decision=state,
            output_digest=output_digest,
            annotation=annotation,
            provenance=provenance,
            split=outcome.split,
            reason_codes=tuple(str(code) for code in entry.get("reason_codes", ())),
            image_artifact=image_artifact,
        )

    def sample_state_samples(self, job_id: str) -> list[str]:
        """Sample ids this job touched, for diagnostics and tests."""
        return [row["sample_id"] for row in self.state.samples(job_id=job_id)]

    def _assert_export_allowed(self, recipe: Recipe, manifest: JobManifest) -> None:
        """Refuse to write a dataset that a demonstration stack produced.

        The manifest is the source of truth here rather than the current profile: a job records the
        bindings it actually ran with, and a profile edited afterwards must not retroactively make an
        old run look production-grade.
        """
        if manifest.demo_backends and not recipe.acceptance.allow_demo_backends:
            registry = self.registry
            verdict = ProductionVerdict(
                demo_usage=tuple(
                    DemoUsage(
                        capability,
                        manifest.capability_bindings.get(capability, "unknown"),
                        "recorded when the job ran",
                    )
                    for capability in manifest.demo_backends
                ),
                considered=PRODUCTION_CAPABILITIES,
            )
            del registry
            assert_production_ready(verdict)

    def _artifact_size(self, digest: str) -> int:
        try:
            return len(self.store.read(digest, verify=False))
        except ValidationFailure:
            return 0

    def _outcomes_from_state(self, job_id: str, recipe: Recipe) -> dict[str, SampleOutcome]:
        """Rebuild sample outcomes from the database, so ``vidliner export`` works after a restart."""
        outcomes: dict[str, SampleOutcome] = {}
        manifest = self.state.require_job(job_id)
        for candidate in self.state.candidates(job_id):
            outcome = outcomes.get(candidate.sample_id)
            if outcome is None:
                outcome = SampleOutcome(
                    sample_id=candidate.sample_id,
                    source_digest=candidate.source_digest,
                    split=self.state.split_of(candidate.sample_id) or "train",
                )
                outcomes[candidate.sample_id] = outcome
            if candidate.state.value == "accepted":
                outcome.accepted_records.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "sample_id": candidate.sample_id,
                        "object_id": candidate.target_object_id,
                        "candidate_key": candidate.candidate_key,
                        "category": candidate.category,
                        "seed": candidate.seed,
                        "output_digest": candidate.output_digest,
                        "overall_score": candidate.overall_score,
                        "reason_codes": list(candidate.reason_codes),
                        "decision": DecisionState.ACCEPTED,
                    }
                )
            else:
                outcome.candidates += 1
                if candidate.state.value == "review":
                    outcome.review += 1
                else:
                    outcome.rejected += 1
        del manifest, recipe
        return outcomes

    # -- QA ---------------------------------------------------------------- #

    def re_evaluate(self, job_id: str, recipe: Recipe | None = None) -> dict[str, Any]:
        """Re-apply the acceptance policy to a job's stored metric evidence.

        No pixels are recomputed and no backend is called: this is the operation that makes a
        threshold change cheap. Candidates whose decision changes are updated in place.
        """
        from vidliner.pipeline.policy import decide_candidate, policy_from_recipe

        manifest = self.state.require_job(job_id)
        loaded = recipe or Recipe.model_validate(manifest.recipe_snapshot)
        policy = policy_from_recipe(loaded)
        changed = 0
        evaluated = 0
        for candidate in self.state.candidates(job_id):
            rows = self.state.metrics_for(candidate.candidate_id)
            if not rows:
                continue
            report = QualityReport(
                candidate_id=candidate.candidate_id,
                sample_id=candidate.sample_id,
                metrics=tuple(_metric_outcome(row) for row in rows),
            )
            decision, _ = decide_candidate(report, policy, candidate_id=candidate.candidate_id, job_id=job_id)
            evaluated += 1
            if decision.state.value != candidate.state.value:
                changed += 1
            self.state.set_candidate_state(
                candidate.candidate_id,
                _candidate_state(decision.state),
                reason_codes=decision.reason_codes,
                output_digest=candidate.output_digest,
                overall_score=decision.overall_score,
            )
        return {
            "job_id": job_id,
            "policy_hash": policy.policy_hash,
            "evaluated": evaluated,
            "changed": changed,
        }

    # -- internal ---------------------------------------------------------- #

    def _write_job_reports(self, manifest: JobManifest, runner: JobRunner) -> None:
        """Write the job summary and the static review report."""
        import json

        from vidliner.control.review import write_review_report

        run_dir = self._workspace.run_dir(manifest.job_id)
        summary = runner.summary()
        (run_dir / "job-summary.json").write_text(
            json.dumps(summary.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self.state.record_report(
            manifest.job_id, "summary", self._workspace.relative(run_dir / "job-summary.json")
        )
        review_path = write_review_report(
            workspace=self._workspace,
            state=self.state,
            job_id=manifest.job_id,
            manifest=manifest,
            counters=manifest.counters,
        )
        if review_path is not None:
            self.state.record_report(manifest.job_id, "review", self._workspace.relative(review_path))


def open_session(
    workspace_path: Path,
    *,
    runtime_path: Path | None = None,
    create: bool = False,
) -> Session:
    """Open a session, the single entry point every command uses."""
    return Session.open(workspace_path, runtime_path=runtime_path, create=create)


def _load_profile(workspace: Workspace, runtime_path: Path | None) -> RuntimeProfile:
    candidates: list[Path] = []
    if runtime_path is not None:
        candidates.append(runtime_path)
    else:
        from_env = os.environ.get("VIDLINER_RUNTIME")
        if from_env:
            candidates.append(Path(from_env))
        candidates.append(workspace.root / DEFAULT_RUNTIME_FILE)
    for candidate in candidates:
        if candidate.is_file():
            return load_profile(candidate)
    return default_profile()


def _candidate_evidence(sample_dir: Path, kind: str, entry: dict[str, Any]) -> Path | None:
    """Locate one candidate's evidence file.

    The keyed name is authoritative: exporting one candidate's label under another candidate's image
    is exactly the corruption this project exists to prevent, so a missing keyed file is reported
    rather than worked around by taking whichever file happens to be first.
    """
    candidate_key = str(entry.get("candidate_key") or "")
    for stem in kinds_for(candidate_key):
        candidate = sample_dir / f"{kind}-{stem}.json"
        if candidate.is_file():
            return candidate
    unkeyed = sample_dir / f"{kind}.json"
    if unkeyed.is_file():
        return unkeyed
    matches = sorted(path for path in sample_dir.glob(f"{kind}-*.json") if matches_kind(path, kind))
    # Only unambiguous: a single candidate's evidence, with no key to tell them apart.
    return matches[0] if len(matches) == 1 and not candidate_key else None


def _metric_outcome(row: dict[str, Any]) -> Any:
    from vidliner.domain.enums import Comparison, GateStatus, MetricName, Severity
    from vidliner.domain.quality import MetricOutcome

    return MetricOutcome(
        metric=MetricName(row["metric"]),
        value=row["value"],
        threshold=row["threshold"],
        comparison=Comparison(row["comparison"]),
        severity=Severity(row["severity"]),
        status=GateStatus(row["status"]),
        evaluator_id=row["evaluator_id"],
        reason_codes=tuple(row["reason_codes"]) if isinstance(row["reason_codes"], list) else (),
        detail=dict(row["detail"]) if isinstance(row["detail"], dict) else {},
    )


def _candidate_state(decision: DecisionState) -> Any:
    from vidliner.domain.enums import CandidateState

    if decision is DecisionState.ACCEPTED:
        return CandidateState.ACCEPTED
    if decision is DecisionState.NEEDS_REVIEW:
        return CandidateState.REVIEW
    return CandidateState.REJECTED
