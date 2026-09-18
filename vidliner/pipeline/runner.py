"""Job orchestration.

A :class:`JobRunner` owns one job end to end:

```
CREATED → PLANNED → RUNNING → FINALISING → SUCCEEDED | FAILED | CANCELLED
```

It builds the sample contexts the graph injects, executes the graph with the engine, turns each
evaluated candidate into a persisted candidate row plus a quality report and an acceptance decision,
materialises the accepted set, and writes the job's manifest, summary, and reports.

Two invariants are enforced here rather than left to convention:

* **a quality rejection never fails a job** — it produces a ``REJECTED`` candidate and a counter;
* **a job with zero accepted candidates can still succeed** — the counters say what happened, and a
  run that produced nothing usable is a data outcome, not a crash.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from vidliner.control.duplicates import DuplicateIndex
from vidliner.core.errors import (
    Cancellation,
    ErrorCode,
    InfrastructureFailure,
    VidlinerError,
    describe_failure,
)
from vidliner.core.graph import OperationNode
from vidliner.core.identity import candidate_id
from vidliner.core.results import JobCounters, JobSummary, NodeResult, utc_now
from vidliner.core.seedtree import SeedTree
from vidliner.domain.enums import CandidateState, DecisionState, JobState, NodeStatus, SampleState
from vidliner.domain.jobs import BackendVersion, JobManifest, OperatorVersion
from vidliner.domain.quality import QualityReport
from vidliner.domain.recipe import Recipe
from vidliner.operators.base import SampleContext
from vidliner.operators.registry import describe_operator
from vidliner.pipeline.backends import CapabilityBroker
from vidliner.pipeline.executor import ArtifactExecutor, PortState, seed_for_node
from vidliner.pipeline.planner import CompiledPlan
from vidliner.pipeline.policy import decide_candidate, policy_from_recipe
from vidliner.runtime.engine import EngineOptions, ExecutionReport, OperationEngine
from vidliner.runtime.logs import EventLog
from vidliner.runtime.registry import BackendRegistry
from vidliner.storage.state import EvidenceRecorder, StateStore
from vidliner.storage.workspace import ArtifactStore, Workspace

__all__ = ["JobOptions", "JobRunner", "SampleOutcome"]


@dataclass(frozen=True, slots=True)
class JobOptions:
    """Knobs a caller controls for one run."""

    dry_run: bool = False
    use_cache: bool = True
    resume: bool = True
    prefer_cache: bool = False
    fail_fast: bool = False
    workers: int = 4
    job_timeout_s: float | None = None
    seed: int | None = None
    collect_reports: bool = True


@dataclass
class SampleOutcome:
    """What happened for one sample: its targets, candidates, and evidence."""

    sample_id: str
    source_digest: str
    split: str
    targets: int = 0
    candidates: int = 0
    accepted: int = 0
    rejected: int = 0
    review: int = 0
    duplicates: int = 0
    accepted_records: list[dict[str, Any]] = field(default_factory=list)
    reason_histogram: dict[str, int] = field(default_factory=dict)

    def observe(self, decision: DecisionState, reason_codes: tuple[str, ...], record: dict[str, Any]) -> None:
        """Record one decided candidate."""
        self.candidates += 1
        for code in reason_codes:
            self.reason_histogram[code] = self.reason_histogram.get(code, 0) + 1
        if decision is DecisionState.ACCEPTED:
            self.accepted += 1
            self.accepted_records.append(record)
        elif decision is DecisionState.NEEDS_REVIEW:
            self.review += 1
        else:
            self.rejected += 1


class JobRunner:
    """Runs one job over one dataset with one runtime profile."""

    def __init__(
        self,
        *,
        recipe: Recipe,
        plan: CompiledPlan,
        workspace: Workspace,
        store: ArtifactStore,
        state: StateStore,
        registry: BackendRegistry,
        sample_contexts: dict[str, SampleContext],
        options: JobOptions | None = None,
        manifest: JobManifest | None = None,
    ) -> None:
        self._recipe = recipe
        self._plan = plan
        self._workspace_ref = workspace
        self._store = store
        self._state = state
        self._registry = registry
        self._sample_contexts = sample_contexts
        self._options = options or JobOptions()
        self._broker = CapabilityBroker(registry=registry)
        self._ports = PortState()
        self._events = EventLog(workspace.run_dir(plan.job_plan.job_id) / "events.jsonl")
        self._cancel = asyncio.Event()
        self._outcomes: dict[str, SampleOutcome] = {}
        self._candidate_index: dict[str, dict[str, Any]] = {}
        self._duplicates = DuplicateIndex(
            enabled=recipe.duplicates.enabled,
            algorithm=recipe.duplicates.perceptual_hash,
            hamming_threshold=recipe.duplicates.hamming_threshold,
            existing=state.perceptual_hashes(),
        )
        self._policy = policy_from_recipe(recipe)
        self._report: ExecutionReport | None = None
        self._manifest = manifest or self._build_manifest()
        self._engine: OperationEngine | None = None
        self._executor: ArtifactExecutor | None = None

    # -- public API -------------------------------------------------------- #

    @property
    def manifest(self) -> JobManifest:
        """The job manifest, updated as the job progresses."""
        return self._manifest

    @property
    def report(self) -> ExecutionReport | None:
        """The engine's report, available after :meth:`execute`."""
        return self._report

    @property
    def outcomes(self) -> dict[str, SampleOutcome]:
        """Per-sample outcomes, available after :meth:`execute`."""
        return self._outcomes

    def cancel(self) -> None:
        """Request cancellation of the running job."""
        self._cancel.set()
        if self._engine is not None:
            self._engine.cancel()
        self._events.emit("job.cancel_requested", job_id=self._manifest.job_id)

    def _ensure_job_row(self) -> None:
        """Insert the job row the first time, and leave an existing row alone on a resume."""
        if self._state.get_job(self._manifest.job_id) is None:
            self._state.create_job(self._manifest)

    async def execute(self) -> JobManifest:
        """Run the job to completion and return its final manifest."""
        self._ensure_job_row()
        self._events.emit(
            "job.created",
            job_id=self._manifest.job_id,
            recipe=self._recipe.name,
            nodes=self._plan.job_plan.node_count,
        )
        self._manifest = self._transition(JobState.PLANNED)
        if self._options.dry_run:
            self._manifest = self._finalize(JobState.SUCCEEDED, dry_run=True)
            return self._manifest
        self._manifest = self._transition(JobState.RUNNING)
        try:
            report = await self._run_graph()
        except Cancellation:
            self._manifest = self._finalize(JobState.CANCELLED)
            return self._manifest
        except VidlinerError as exc:
            self._manifest = self._fail(exc)
            return self._manifest
        except Exception as exc:
            self._manifest = self._fail(
                InfrastructureFailure(f"{type(exc).__name__}: {exc}", code=ErrorCode.STATE_STORE_FAILED)
            )
            return self._manifest
        self._report = report
        self._manifest = self._transition(JobState.FINALIZING)
        self._manifest = self._finalize(JobState.SUCCEEDED)
        return self._manifest

    async def resume(self) -> JobManifest:
        """Resume a job whose run directory and state rows already exist."""
        self._options = JobOptions(
            dry_run=False,
            use_cache=self._options.use_cache,
            resume=True,
            fail_fast=self._options.fail_fast,
            workers=self._options.workers,
            job_timeout_s=self._options.job_timeout_s,
            seed=self._manifest.seed,
        )
        return await self.execute()

    def summary(self) -> JobSummary:
        """Build the job summary from the persisted state."""
        counters = self._counters()
        stored = self._state.get_job(self._manifest.job_id)
        manifest = stored.manifest if stored else self._manifest
        metric_averages = self._state.metric_averages(self._manifest.job_id)
        histogram: dict[str, int] = {}
        for outcome in self._outcomes.values():
            for code, count in outcome.reason_histogram.items():
                histogram[code] = histogram.get(code, 0) + count
        duration = manifest.duration_s
        return JobSummary(
            job_id=self._manifest.job_id,
            state=manifest.state,
            created_at=manifest.created_at,
            finished_at=manifest.finished_at,
            duration_s=duration,
            counters=counters,
            average_quality=metric_averages.get("overall"),
            estimated_cost=self._plan.job_plan.estimate.estimated_cost or None,
            actual_cost=self._actual_cost(),
            currency=self._plan.job_plan.estimate.currency,
            backends=list(self._broker.used_backends),
            reason_code_histogram=histogram,
            failure_class=manifest.failure_class,
            failure_code=manifest.failure_code,
            failure_message=manifest.failure_message,
        )

    # -- execution --------------------------------------------------------- #

    async def _run_graph(self) -> ExecutionReport:
        graph = self._plan.graph
        self._outcomes = {
            sample_id_value: SampleOutcome(
                sample_id=sample_id_value,
                source_digest=context.source_digest,
                split=context.split,
            )
            for sample_id_value, context in self._sample_contexts.items()
        }
        self._executor = executor = ArtifactExecutor(
            job_id=self._manifest.job_id,
            store=self._store,
            broker=self._broker,
            sample_for=self._sample_for_node,
            on_event=self._on_node_event,
            ports=self._ports,
        )
        engine = OperationEngine(
            graph,
            executor,
            EvidenceRecorder(
                self._state, job_id=self._manifest.job_id, artifact_exists=self._artifact_exists
            ),
            options=EngineOptions(
                workers=self._options.workers,
                use_cache=self._options.use_cache,
                resume=self._options.resume,
                prefer_cache=self._options.prefer_cache,
                fail_fast=self._options.fail_fast,
                job_timeout_s=self._options.job_timeout_s,
                default_node_timeout_s=self._recipe.limits.per_node_timeout_s,
                on_resolved=self._rehydrate,
            ),
            seed_for_node=lambda node: seed_for_node(self._manifest.seed, node),
            on_event=self._on_engine_event,
        )
        self._engine = engine
        report = await engine.execute()
        self._report = report
        self._record_targets(executor)
        self._decide_evaluated_candidates(report)
        if report.cancelled:
            raise Cancellation("job was cancelled")
        return report

    def _record_targets(self, executor: ArtifactExecutor) -> None:
        """Count actual target objects per sample from the selection node's output.

        Counting *selected objects* rather than object branches matters: the assembler reserves
        branches so node identity stays stable, and reporting the reservation as "targets" would
        overstate what the job actually did.
        """
        for node_id_value, values in executor.ports.values.items():
            node = (
                self._plan.graph.node(node_id_value)
                if node_id_value in {candidate.node_id for candidate in self._plan.graph.nodes}
                else None
            )
            if node is None or node.operator != "select.targets":
                continue
            sample = self._sample_for_node(node)
            if sample is None:
                continue
            selection = values.get("selection")
            selected = getattr(selection, "selected", None)
            outcome = self._outcomes.get(sample.sample_id)
            if outcome is not None and selected is not None:
                outcome.targets = len(selected)

    def _rehydrate(self, node_id_value: str, result: NodeResult) -> None:
        """Restore a resolved node's port values before its dependents are dispatched."""
        if result.status is not NodeStatus.SUCCEEDED:
            return
        executor = self._executor
        if executor is not None and not executor.ports.has(node_id_value, "__present__"):
            executor.restore_from_result(result)

    def _decide_evaluated_candidates(self, report: ExecutionReport) -> None:
        """Apply the acceptance policy to every evaluated candidate and persist the outcome."""
        import os as _os

        _debug = bool(_os.environ.get("VIDLINER_DEBUG_REPORT"))
        _seen = 0
        for node_id_value, result in report.nodes.items():
            if result.status is not NodeStatus.SUCCEEDED or result.operator != "evaluate.candidate":
                continue
            _seen += 1
            if _debug:
                _v = self._ports.values.get(node_id_value, {}).get("report")
                print(
                    "EVALNODE",
                    node_id_value,
                    type(_v).__name__,
                    list(_v.keys())[:4] if isinstance(_v, dict) else "",
                )
        if _debug:
            print("EVALCOUNT", _seen)
        for node_id_value, result in report.nodes.items():
            if result.status is not NodeStatus.SUCCEEDED or result.operator != "evaluate.candidate":
                continue
            report_payload = self._report_from_result(result)
            if report_payload is None:
                import os as _os

                if _os.environ.get("VIDLINER_DEBUG_REPORT"):
                    print(
                        "NOPAYLOAD", node_id_value, sorted(self._ports.values.get(node_id_value, {}).keys())
                    )
                continue
            node = self._plan.graph.node(node_id_value)
            sample = self._sample_for_node(node)
            if sample is None:
                continue
            record = self._record_for_evaluate(node_id_value, node)
            if record is None:
                # A cached run may serve an evaluation node without ever executing its generator. The
                # candidate record is on disk, so the decision is recovered from there rather than
                # from this run's memory.
                record = self._record_from_disk(sample, node)
            if record is None:
                import os as _os

                if _os.environ.get("VIDLINER_DEBUG_REPORT"):
                    print("NORECORD", node_id_value, node.lineage)
                self._events.emit(
                    "candidate.missing_record",
                    job_id=self._manifest.job_id,
                    node_id=node_id_value,
                    detail="no generated candidate is associated with this evaluation node",
                )
                continue
            identifier = record.get("candidate_id") or candidate_id(
                sample.sample_id, record.get("object_id", ""), record.get("candidate_key", "")
            )
            # The row must exist before its metrics and decision are written. Creating it here rather
            # than assuming it is what makes a rejected candidate's evidence complete even when the
            # node that materialises that evidence was legitimately skipped.
            self._ensure_candidate_row(sample, record, identifier)
            decision, evaluated = decide_candidate(
                report_payload,
                self._policy,
                candidate_id=identifier,
                job_id=self._manifest.job_id,
            )
            reasons = decision.reason_codes
            self._state.record_metrics(
                identifier,
                [
                    {
                        "metric": outcome.metric.value,
                        "value": outcome.value,
                        "threshold": outcome.threshold,
                        "comparison": outcome.comparison.value,
                        "severity": outcome.severity.value,
                        "status": outcome.status.value,
                        "evaluator_id": outcome.evaluator_id,
                        "reason_codes": list(outcome.reason_codes),
                        "detail": _jsonable(outcome.detail),
                    }
                    for outcome in evaluated.metrics
                ],
            )
            state = _candidate_state(decision.state)
            self._state.set_candidate_state(
                identifier,
                state,
                reason_codes=reasons,
                output_digest=record.get("output_digest"),
                overall_score=decision.overall_score,
            )
            outcome = self._outcomes.get(sample.sample_id)
            if outcome is None:
                continue
            outcome.observe(
                decision.state,
                reasons,
                {
                    "candidate_id": identifier,
                    "sample_id": sample.sample_id,
                    "object_id": record.get("object_id"),
                    "candidate_key": record.get("candidate_key"),
                    "category": record.get("category"),
                    "output_digest": record.get("output_digest"),
                    "overall_score": decision.overall_score,
                    "reason_codes": list(reasons),
                    "quality": evaluated,
                    "decision": decision,
                    "lineage_root": sample.root_digest,
                    "split": sample.split,
                    "seed": record.get("seed"),
                    "metrics": {
                        outcome_metric.metric.value: outcome_metric.value
                        for outcome_metric in evaluated.metrics
                    },
                },
            )
            self._events.emit(
                "candidate.decided",
                job_id=self._manifest.job_id,
                candidate_id=identifier,
                sample_id=sample.sample_id,
                state=state.value,
                overall_score=decision.overall_score,
                reason_codes=list(reasons),
            )

    def _record_for_evaluate(self, node_id_value: str, node: OperationNode) -> dict[str, Any] | None:
        """Find the generated candidate an evaluation node belongs to.

        Generation and evaluation are separate nodes, so the association is recovered from the graph
        lineage rather than from a shared id: a candidate is identified by
        ``sample → object → candidate``, which is exactly the lineage both nodes carry.
        """
        direct = self._candidate_index.get(node_id_value)
        if direct is not None:
            return direct
        key = _candidate_lineage_key(node)
        if key is None:
            return None
        return self._candidate_index.get(key)

    def _record_from_disk(self, sample: SampleContext, node: OperationNode) -> dict[str, Any] | None:
        """Recover a candidate's identity from the sample's evidence directory.

        Only reached when an evaluation node was served from the cache without its generator having
        run in this process; the candidate JSON written by the export stage is authoritative.
        """
        import json

        candidate_key = None
        for part in node.lineage:
            prefix, _, value = part.partition(":")
            if prefix == "candidate":
                candidate_key = value
        sample_dir = self._workspace_ref.sample_dir(self._manifest.job_id, sample.sample_id)
        for candidate_file in sorted(sample_dir.glob("candidate-*.json")):
            try:
                payload = json.loads(candidate_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            if candidate_key is not None and payload.get("candidate_key") != candidate_key:
                continue
            generation = payload.get("generation") or {}
            image = generation.get("image") or {}
            identifier = candidate_id(
                sample.sample_id,
                str(payload.get("object_id", "")),
                str(payload.get("candidate_key", "")),
            )
            return {
                "candidate_id": identifier,
                "sample_id": sample.sample_id,
                "object_id": payload.get("object_id"),
                "candidate_key": payload.get("candidate_key"),
                "category": payload.get("category"),
                "seed": payload.get("seed"),
                "output_digest": image.get("digest"),
                "intent": payload.get("intent") or {},
                "plan_id": payload.get("plan_id"),
            }
        return None

    def _report_from_result(self, result: NodeResult) -> QualityReport | None:
        value = self._ports.get(result.node_id, "report")
        if isinstance(value, QualityReport):
            return value
        if isinstance(value, dict):
            try:
                return QualityReport.model_validate(value)
            except Exception:
                return None
        return None

    # -- hooks ------------------------------------------------------------- #

    def _on_node_event(self, kind: str, node: OperationNode, detail: dict[str, Any]) -> None:
        """Record per-node side effects and forward the operator's own structured events.

        An operator that publishes ``verify.not_found`` produces the event name
        ``operator.verify.not_found`` here; the prefix is stripped so the run log carries the name the
        operator chose, and a reader can search for it without knowing how it was wrapped.
        """
        if kind == "node.values":
            return  # the runner already records artifacts and counters for this node
        if node.operator == "generate.replacement":
            self._register_generated_candidate(node)
        event = kind.removeprefix("operator.")
        self._events.emit(event, job_id=self._manifest.job_id, node_id=node.node_id, **detail)

    def _register_generated_candidate(self, node: OperationNode) -> None:
        sample = self._sample_for_node(node)
        if sample is None:
            return
        payload = self._ports.get(node.node_id, "candidate")
        if not isinstance(payload, dict):
            return
        generation = payload.get("generation") or {}
        image = generation.get("image") or {}
        identifier = candidate_id(
            sample.sample_id,
            str(payload.get("object_id", "")),
            str(payload.get("candidate_key", "")),
        )
        record = {
            "candidate_id": identifier,
            "sample_id": sample.sample_id,
            "object_id": payload.get("object_id"),
            "candidate_key": payload.get("candidate_key"),
            "category": payload.get("category"),
            "seed": payload.get("seed"),
            "output_digest": image.get("digest"),
            "intent": payload.get("intent") or {},
            "plan_id": payload.get("plan_id"),
        }
        self._candidate_index[node.node_id] = record
        self._candidate_index[_candidate_lineage_key(node) or node.node_id] = record
        self._ensure_candidate_row(sample, record, identifier)
        self._events.emit(
            "candidate.generated",
            job_id=self._manifest.job_id,
            candidate_id=identifier,
            sample_id=sample.sample_id,
            candidate_key=payload.get("candidate_key"),
        )

    def _ensure_candidate_row(self, sample: SampleContext, record: dict[str, Any], identifier: str) -> None:
        """Create a candidate row if it does not exist yet.

        Both the generator and the decision stage call this. Generation is the natural owner, but a
        cached or partially materialised run can reach a decision without the generator having run in
        this process, and a rejected candidate's evidence is written by a node that a rejection
        legitimately skips — so the stage that persists a lifecycle must not assume the row is there.
        """
        existed = self._state.get_candidate(identifier) is not None
        self._state.upsert_candidate(
            candidate_id=identifier,
            job_id=self._manifest.job_id,
            sample_id=sample.sample_id,
            target_object_id=str(record.get("object_id") or ""),
            candidate_key=str(record.get("candidate_key") or ""),
            category=str(record.get("category") or ""),
            state=CandidateState.GENERATED,
            seed=int(record.get("seed") or 0),
            source_digest=sample.source_digest,
            output_digest=record.get("output_digest"),
            overall_score=None,
            policy_hash=self._policy.policy_hash,
            plan_json={"plan_id": record.get("plan_id")},
            intent_json=record.get("intent") or {},
        )
        if not existed:
            self._state.append_candidate_event(identifier, to_state=CandidateState.GENERATED)

    def _on_engine_event(self, event: dict[str, Any]) -> None:
        self._events.emit("node.finished", **event)

    @property
    def _workspace(self) -> Workspace:
        """The workspace this job writes evidence into."""
        return self._workspace_ref

    def _artifact_exists(self, artifact: Any) -> bool:
        digest = getattr(artifact, "digest", None)
        return bool(digest) and self._store.exists(str(digest))

    def _sample_for_node(self, node: OperationNode) -> SampleContext | None:
        """The sample context a node belongs to, derived from its lineage path."""
        for part in node.lineage:
            prefix, _, value = part.partition(":")
            if prefix == "sample":
                return self._sample_contexts.get(value)
        return None

    # -- lifecycle --------------------------------------------------------- #

    def _build_manifest(self) -> JobManifest:
        from vidliner.runtime.profile import redact_profile

        job_id_value = self._plan.job_plan.job_id
        run_dir = self._workspace_ref.run_dir(job_id_value)
        operators: dict[str, OperatorVersion] = {}
        for node in self._plan.graph.nodes:
            spec = describe_operator(node.operator, node.operator_version)
            operators[spec.name] = OperatorVersion(
                operator=spec.name,
                version=spec.version,
                name=spec.name,
                capabilities=spec.capabilities,
            )
        backends: list[BackendVersion] = []
        for capability, backend_name in self._plan.resolution.as_mapping().items():
            spec = self._registry.profile.backends.get(backend_name)
            backends.append(
                BackendVersion(
                    backend_id=backend_name,
                    version="declared",
                    kind=capability,
                    capabilities=(capability,),
                    device=self._registry.profile.device_for(backend_name),
                    external=bool(spec.estimates.external) if spec else False,
                )
            )
        seed = self._options.seed if self._options.seed is not None else (self._recipe.replacement.seed or 0)
        tree = SeedTree(seed)
        return JobManifest(
            job_id=job_id_value,
            state=JobState.CREATED,
            recipe_name=self._recipe.name,
            recipe_hash=self._recipe.recipe_hash(),
            recipe_snapshot=self._recipe.snapshot(),
            runtime_snapshot=redact_profile(self._registry.profile),
            runtime_profile_name=self._registry.profile.profile,
            seed=seed,
            workspace_root=str(self._workspace_ref.root),
            run_dir=str(run_dir),
            dataset_input=self._recipe.dataset.input,
            output_path=self._recipe.export.path,
            operator_versions=tuple(operators.values()),
            backend_versions=tuple(backends),
            capability_bindings=self._plan.resolution.as_mapping(),
            seed_tree={sample: tree.for_sample(sample).seed for sample in self._sample_contexts},
            node_count=self._plan.job_plan.node_count,
        )

    def _transition(self, state: JobState) -> JobManifest:
        self._state.transition_job(self._manifest.job_id, state)
        stored = self._state.get_job(self._manifest.job_id)
        updated = stored.manifest if stored else self._manifest.model_copy(update={"state": state})
        self._events.emit("job.state", job_id=updated.job_id, state=state.value)
        return updated

    def _finalize(self, state: JobState, *, dry_run: bool = False) -> JobManifest:
        counters = self._counters()
        duration = self._manifest.model_copy(update={"finished_at": utc_now()}).duration_s or 0.0
        if dry_run:
            self._events.emit(
                "job.dry_run",
                job_id=self._manifest.job_id,
                nodes=self._plan.job_plan.node_count,
                estimated_cost=self._plan.job_plan.estimate.estimated_cost,
            )
        updated = self._manifest.model_copy(
            update={
                "state": state,
                "counters": counters,
                "updated_at": utc_now(),
                "finished_at": utc_now(),
            }
        )
        self._state.save_job(updated)
        summary = self.summary().model_copy(update={"counters": counters, "duration_s": duration})
        self._state.set_summary(updated.job_id, summary)
        self._manifest = updated.model_copy(update={"summary": summary})
        self._state.save_job(self._manifest)
        self._events.emit(
            "job.finished",
            job_id=self._manifest.job_id,
            state=state.value,
            duration_s=round(duration, 3),
            counters=counters.model_dump(),
        )
        return self._manifest

    def _fail(self, error: Exception) -> JobManifest:
        payload = describe_failure(error)
        self._events.emit("job.failed", job_id=self._manifest.job_id, **payload)
        updated = self._manifest.model_copy(
            update={
                "state": JobState.FAILED,
                "failure_class": payload["failure_class"],
                "failure_code": payload["code"],
                "failure_message": payload["message"],
                "counters": self._counters(),
                "updated_at": utc_now(),
                "finished_at": utc_now(),
            }
        )
        self._state.save_job(updated)
        summary = self.summary().model_copy(update={"counters": updated.counters})
        self._state.set_summary(updated.job_id, summary)
        self._manifest = updated.model_copy(update={"summary": summary})
        self._state.save_job(self._manifest)
        return self._manifest

    def _counters(self) -> JobCounters:
        return JobCounters(
            samples=len(self._sample_contexts),
            targets=sum(outcome.targets for outcome in self._outcomes.values()),
            generated=sum(outcome.candidates for outcome in self._outcomes.values()),
            accepted=sum(outcome.accepted for outcome in self._outcomes.values()),
            rejected=sum(outcome.rejected for outcome in self._outcomes.values()),
            review=sum(outcome.review for outcome in self._outcomes.values()),
            duplicates=sum(outcome.duplicates for outcome in self._outcomes.values()),
            failed=len(self._report.failed_nodes) if self._report else 0,
            nodes_succeeded=self._report.count(NodeStatus.SUCCEEDED) if self._report else 0,
            nodes_failed=len(self._report.failed_nodes) if self._report else 0,
            nodes_cached=len(self._report.cache_hits) if self._report else 0,
            nodes_resumed=len(self._report.resumed_nodes) if self._report else 0,
        )

    def _actual_cost(self) -> float | None:
        if self._report is None:
            return None
        total = 0.0
        seen = False
        for result in self._report.nodes.values():
            if result.telemetry.cost is not None:
                total += result.telemetry.cost
                seen = True
        return round(total, 6) if seen else None

    # -- dataset registration --------------------------------------------- #

    def register_sources(self) -> None:
        """Record every source sample in the state store, so lineage is queryable from the start."""
        for identifier, context in self._sample_contexts.items():
            self._state.register_sample(
                sample_id=identifier,
                job_id=None,
                root_digest=context.root_digest,
                digest=context.source_digest,
                rel_path=context.relative_path,
                split=context.split,
                state=SampleState.SOURCE,
                perceptual_hash=context.perceptual_hash,
            )

    def register_accepted(self) -> None:
        """Record every accepted sample and its lineage edge."""
        for outcome in self._outcomes.values():
            for record in outcome.accepted_records:
                identifier = f"a_{record['candidate_id'][2:]}"
                self._state.register_sample(
                    sample_id=identifier,
                    job_id=self._manifest.job_id,
                    source_sample_id=outcome.sample_id,
                    root_digest=outcome.source_digest,
                    digest=str(record.get("output_digest") or ""),
                    rel_path=str(record.get("candidate_key") or identifier),
                    split=outcome.split,
                    state=SampleState.ACCEPTED,
                    augmentation_depth=1,
                    class_histogram={str(record.get("category") or "unknown"): 1},
                )
                self._state.link_lineage(identifier, outcome.sample_id)


def _candidate_lineage_key(node: OperationNode) -> str | None:
    """The lineage suffix that identifies a candidate branch, or ``None`` for other nodes."""
    parts = [part for part in node.lineage if part.startswith(("object:", "candidate:"))]
    if not parts:
        return None
    return "/".join(parts)


def _is_object_node(node: OperationNode) -> bool:
    return any(part.startswith("object:") for part in node.lineage)


def _candidate_state(decision: DecisionState) -> CandidateState:
    if decision is DecisionState.ACCEPTED:
        return CandidateState.ACCEPTED
    if decision is DecisionState.NEEDS_REVIEW:
        return CandidateState.REVIEW
    return CandidateState.REJECTED


def _jsonable(payload: Any) -> Any:
    import json

    return json.loads(json.dumps(payload, default=str))
