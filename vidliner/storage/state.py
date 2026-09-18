"""SQLite persistence for jobs, nodes, candidates, artifacts, cache, and lineage.

State is the only thing in the database; pixels live in the artifact store. The schema is created
by :meth:`StateStore.initialize` and versioned through ``schema_meta`` so a future release can
migrate rather than guess.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from vidliner.core.errors import ErrorCode, InfrastructureFailure
from vidliner.core.graph import OperationNode
from vidliner.core.results import (
    ArtifactRef,
    JobCounters,
    JobSummary,
    NodeResult,
    OperationEvidence,
    utc_now,
)
from vidliner.domain.enums import CandidateState, JobState, NodeStatus, SampleState
from vidliner.domain.jobs import JobManifest

__all__ = [
    "SCHEMA_VERSION",
    "EvidenceRecorder",
    "InMemoryRunState",
    "StateStore",
    "StoredCandidate",
    "StoredJob",
]

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id            TEXT PRIMARY KEY,
    state             TEXT NOT NULL,
    recipe_name       TEXT NOT NULL,
    recipe_hash       TEXT NOT NULL,
    recipe_snapshot   TEXT NOT NULL,
    runtime_snapshot  TEXT NOT NULL,
    runtime_profile   TEXT NOT NULL DEFAULT '',
    seed              INTEGER NOT NULL,
    workspace_root    TEXT NOT NULL,
    run_dir           TEXT NOT NULL,
    dataset_input     TEXT NOT NULL DEFAULT '',
    output_path       TEXT NOT NULL DEFAULT '',
    node_count        INTEGER NOT NULL DEFAULT 0,
    manifest          TEXT NOT NULL,
    counters          TEXT NOT NULL DEFAULT '{}',
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    finished_at       TEXT,
    failure_class     TEXT,
    failure_code      TEXT,
    failure_message   TEXT
);
CREATE INDEX IF NOT EXISTS jobs_state_created ON jobs(state, created_at DESC);

CREATE TABLE IF NOT EXISTS node_runs (
    job_id            TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    node_id           TEXT NOT NULL,
    operator          TEXT NOT NULL,
    operator_version  TEXT NOT NULL,
    stage             TEXT NOT NULL,
    lineage           TEXT NOT NULL DEFAULT '',
    status            TEXT NOT NULL,
    attempt           INTEGER NOT NULL DEFAULT 0,
    cache_hit         INTEGER NOT NULL DEFAULT 0,
    resumed           INTEGER NOT NULL DEFAULT 0,
    backend_id        TEXT,
    config_hash       TEXT NOT NULL DEFAULT '',
    seed              INTEGER NOT NULL DEFAULT 0,
    input_digests     TEXT NOT NULL DEFAULT '[]',
    output_digests    TEXT NOT NULL DEFAULT '[]',
    result            TEXT NOT NULL DEFAULT '{}',
    started_at        TEXT,
    finished_at       TEXT,
    duration_ms       INTEGER,
    error_class       TEXT,
    error_code        TEXT,
    error_message     TEXT,
    PRIMARY KEY (job_id, node_id)
);
CREATE INDEX IF NOT EXISTS node_runs_job_stage ON node_runs(job_id, stage, status);

CREATE TABLE IF NOT EXISTS candidates (
    candidate_id      TEXT PRIMARY KEY,
    job_id            TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    sample_id         TEXT NOT NULL,
    target_object_id  TEXT NOT NULL,
    candidate_key     TEXT NOT NULL,
    category          TEXT NOT NULL,
    state             TEXT NOT NULL,
    seed              INTEGER NOT NULL,
    source_digest     TEXT NOT NULL DEFAULT '',
    output_digest     TEXT,
    overall_score     REAL,
    policy_hash       TEXT NOT NULL DEFAULT '',
    plan_json         TEXT NOT NULL DEFAULT '{}',
    intent_json       TEXT NOT NULL DEFAULT '{}',
    artifacts         TEXT NOT NULL DEFAULT '[]',
    reason_codes      TEXT NOT NULL DEFAULT '[]',
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS candidates_job_state ON candidates(job_id, state);
CREATE INDEX IF NOT EXISTS candidates_sample ON candidates(job_id, sample_id);

CREATE TABLE IF NOT EXISTS candidate_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id      TEXT NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    from_state        TEXT,
    to_state          TEXT NOT NULL,
    reason_codes      TEXT NOT NULL DEFAULT '[]',
    detail            TEXT,
    occurred_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quality_metrics (
    candidate_id      TEXT NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    metric            TEXT NOT NULL,
    value             REAL,
    threshold         REAL,
    comparison        TEXT NOT NULL DEFAULT 'at_least',
    severity          TEXT NOT NULL DEFAULT 'hard',
    status            TEXT NOT NULL DEFAULT 'skipped',
    evaluator_id      TEXT,
    reason_codes      TEXT NOT NULL DEFAULT '[]',
    detail            TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (candidate_id, metric)
);

CREATE TABLE IF NOT EXISTS artifacts (
    digest            TEXT NOT NULL,
    kind              TEXT NOT NULL,
    media_type        TEXT NOT NULL,
    rel_path          TEXT NOT NULL,
    size_bytes        INTEGER NOT NULL,
    width             INTEGER,
    height            INTEGER,
    producer_node     TEXT,
    job_id            TEXT,
    created_at        TEXT NOT NULL,
    PRIMARY KEY (digest, kind)
);
CREATE INDEX IF NOT EXISTS artifacts_kind ON artifacts(kind);

CREATE TABLE IF NOT EXISTS node_cache (
    cache_key         TEXT PRIMARY KEY,
    operator          TEXT NOT NULL,
    operator_version  TEXT NOT NULL,
    implementation    TEXT NOT NULL DEFAULT '',
    config_hash       TEXT NOT NULL DEFAULT '',
    output_digests    TEXT NOT NULL DEFAULT '[]',
    payload           TEXT NOT NULL,
    hits              INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL,
    last_used_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS samples (
    sample_id         TEXT PRIMARY KEY,
    job_id            TEXT REFERENCES jobs(job_id) ON DELETE SET NULL,
    -- Deliberately not a foreign key: a source sample discovered by 'vidliner inspect' may not be
    -- registered yet, and a source sample from another job must still be referable.
    source_sample_id  TEXT,
    root_digest       TEXT NOT NULL DEFAULT '',
    digest            TEXT NOT NULL DEFAULT '',
    rel_path          TEXT NOT NULL DEFAULT '',
    split             TEXT NOT NULL DEFAULT 'train',
    augmentation_depth INTEGER NOT NULL DEFAULT 0,
    perceptual_hash   TEXT,
    embedding_ref     TEXT,
    class_histogram   TEXT NOT NULL DEFAULT '{}',
    state             TEXT NOT NULL DEFAULT 'source',
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS samples_split_state ON samples(split, state);
CREATE INDEX IF NOT EXISTS samples_root ON samples(root_digest);
CREATE INDEX IF NOT EXISTS samples_phash ON samples(perceptual_hash);

CREATE TABLE IF NOT EXISTS lineage_edges (
    child_sample_id   TEXT NOT NULL,
    parent_sample_id  TEXT NOT NULL,
    relation          TEXT NOT NULL DEFAULT 'augmentation',
    PRIMARY KEY (child_sample_id, parent_sample_id)
);

CREATE TABLE IF NOT EXISTS duplicates (
    job_id            TEXT NOT NULL,
    candidate_id      TEXT NOT NULL,
    duplicate_of      TEXT NOT NULL,
    kind              TEXT NOT NULL,
    distance          REAL NOT NULL,
    PRIMARY KEY (job_id, candidate_id, duplicate_of)
);

CREATE TABLE IF NOT EXISTS job_reports (
    job_id            TEXT NOT NULL,
    report_name       TEXT NOT NULL,
    rel_path          TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    PRIMARY KEY (job_id, report_name)
);
"""


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _loads(text: str | None) -> Any:
    if not text:
        return None
    return json.loads(text)


@dataclass(frozen=True, slots=True)
class StoredJob:
    """A job row as read back from the database."""

    manifest: JobManifest
    counters: JobCounters

    @property
    def job_id(self) -> str:
        """Identifier of the stored job."""
        return self.manifest.job_id

    @property
    def state(self) -> JobState:
        """Current lifecycle state."""
        return self.manifest.state


@dataclass(frozen=True, slots=True)
class StoredCandidate:
    """A candidate row as read back from the database."""

    candidate_id: str
    job_id: str
    sample_id: str
    target_object_id: str
    candidate_key: str
    category: str
    state: CandidateState
    seed: int
    source_digest: str
    output_digest: str | None
    overall_score: float | None
    policy_hash: str
    reason_codes: tuple[str, ...]
    artifacts: tuple[str, ...]
    created_at: datetime
    updated_at: datetime


class StateStore:
    """SQLite-backed persistence for one workspace."""

    def __init__(self, path: Path, *, timeout_s: float = 30.0) -> None:
        self._path = Path(path)
        self._timeout = timeout_s
        self._connection: sqlite3.Connection | None = None

    @property
    def path(self) -> Path:
        """Path of the database file."""
        return self._path

    # -- lifecycle --------------------------------------------------------- #

    def initialize(self) -> StateStore:
        """Create the schema if needed and enable WAL. Idempotent."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        connection.executescript(_SCHEMA)
        connection.execute(
            "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?) ON CONFLICT(key) DO NOTHING",
            (str(SCHEMA_VERSION),),
        )
        connection.commit()
        return self

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            connection = sqlite3.connect(str(self._path), timeout=self._timeout, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA synchronous=NORMAL")
            self._connection = connection
        return self._connection

    @property
    def connection(self) -> sqlite3.Connection:
        """The live connection, initialised on first use."""
        return self._connect()

    def close(self) -> None:
        """Close the connection if open."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> StateStore:
        return self.initialize()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a block inside a transaction, rolling back on any exception."""
        connection = self._connect()
        connection.execute("BEGIN")
        try:
            yield connection
        except Exception:
            connection.execute("ROLLBACK")
            raise
        connection.execute("COMMIT")

    def schema_version(self) -> int:
        """Schema version recorded in the database."""
        row = self.connection.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()
        return int(row["value"]) if row else 0

    # -- jobs -------------------------------------------------------------- #

    def create_job(self, manifest: JobManifest) -> None:
        """Insert a new job row."""
        payload = manifest.model_dump(mode="json")
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO jobs (job_id, state, recipe_name, recipe_hash, recipe_snapshot,
                                  runtime_snapshot, runtime_profile, seed, workspace_root, run_dir,
                                  dataset_input, output_path, node_count, manifest, counters,
                                  created_at, updated_at, finished_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest.job_id,
                    manifest.state.value,
                    manifest.recipe_name,
                    manifest.recipe_hash,
                    _json(manifest.recipe_snapshot),
                    _json(manifest.runtime_snapshot),
                    manifest.runtime_profile_name,
                    manifest.seed,
                    manifest.workspace_root,
                    manifest.run_dir,
                    manifest.dataset_input,
                    manifest.output_path,
                    manifest.node_count,
                    _json(payload),
                    _json(manifest.counters.model_dump(mode="json")),
                    manifest.created_at.isoformat(),
                    manifest.updated_at.isoformat(),
                    manifest.finished_at.isoformat() if manifest.finished_at else None,
                ),
            )

    def save_job(self, manifest: JobManifest) -> None:
        """Persist the current form of a job manifest."""
        payload = manifest.model_dump(mode="json")
        with self.transaction() as connection:
            connection.execute(
                """
                UPDATE jobs SET state=?, manifest=?, counters=?, node_count=?, updated_at=?,
                                finished_at=?, failure_class=?, failure_code=?, failure_message=?,
                                output_path=?
                WHERE job_id=?
                """,
                (
                    manifest.state.value,
                    _json(payload),
                    _json(manifest.counters.model_dump(mode="json")),
                    manifest.node_count,
                    manifest.updated_at.isoformat(),
                    manifest.finished_at.isoformat() if manifest.finished_at else None,
                    manifest.failure_class,
                    manifest.failure_code,
                    manifest.failure_message,
                    manifest.output_path,
                    manifest.job_id,
                ),
            )

    def transition_job(self, job_id: str, state: JobState, **fields: Any) -> None:
        """Move a job to ``state`` and update the named manifest fields."""
        stored = self.get_job(job_id)
        if stored is None:
            raise InfrastructureFailure(
                f"job {job_id} does not exist",
                code=ErrorCode.JOB_NOT_FOUND,
                detail={"job_id": job_id},
            )
        updated = stored.manifest.model_copy(update={"state": state, "updated_at": utc_now(), **fields})
        if state in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED} and updated.finished_at is None:
            updated = updated.model_copy(update={"finished_at": utc_now()})
        self.save_job(updated)

    def get_job(self, job_id: str) -> StoredJob | None:
        """Read one job row, or ``None`` when absent."""
        row = self.connection.execute(
            "SELECT manifest, counters FROM jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        if row is None:
            return None
        manifest = JobManifest.model_validate(_loads(row["manifest"]))
        counters = JobCounters.model_validate(_loads(row["counters"]) or {})
        return StoredJob(manifest=manifest, counters=counters)

    def require_job(self, job_id: str) -> JobManifest:
        """Read one job manifest, raising when it does not exist."""
        stored = self.get_job(job_id)
        if stored is None:
            raise InfrastructureFailure(
                f"job {job_id} was not found in {self._path}",
                code=ErrorCode.JOB_NOT_FOUND,
                detail={"job_id": job_id},
            )
        return stored.manifest

    def list_jobs(self, *, states: tuple[JobState, ...] = (), limit: int = 50) -> list[JobManifest]:
        """List jobs, newest first, optionally filtered by state."""
        sql = "SELECT manifest FROM jobs"
        params: list[Any] = []
        if states:
            placeholders = ",".join("?" for _ in states)
            sql += f" WHERE state IN ({placeholders})"
            params.extend(state.value for state in states)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.connection.execute(sql, params).fetchall()
        return [JobManifest.model_validate(_loads(row["manifest"])) for row in rows]

    def latest_job(self) -> JobManifest | None:
        """The most recently created job."""
        row = self.connection.execute("SELECT manifest FROM jobs ORDER BY created_at DESC LIMIT 1").fetchone()
        return JobManifest.model_validate(_loads(row["manifest"])) if row else None

    def set_summary(self, job_id: str, summary: JobSummary) -> None:
        """Attach a summary to a job manifest."""
        stored = self.get_job(job_id)
        if stored is None:
            return
        updated = stored.manifest.model_copy(
            update={"summary": summary, "counters": summary.counters, "updated_at": utc_now()}
        )
        self.save_job(updated)

    def record_report(self, job_id: str, name: str, rel_path: str) -> None:
        """Record the location of a generated report."""
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO job_reports (job_id, report_name, rel_path, created_at) VALUES (?, ?, ?, ?)
                ON CONFLICT(job_id, report_name) DO UPDATE SET rel_path=excluded.rel_path,
                                                               created_at=excluded.created_at
                """,
                (job_id, name, rel_path, utc_now().isoformat()),
            )

    def reports_for(self, job_id: str) -> dict[str, str]:
        """Mapping of report name to relative path for one job."""
        rows = self.connection.execute(
            "SELECT report_name, rel_path FROM job_reports WHERE job_id=?", (job_id,)
        ).fetchall()
        return {row["report_name"]: row["rel_path"] for row in rows}

    # -- nodes and evidence ------------------------------------------------ #

    def record_node_evidence(self, evidence: OperationEvidence) -> None:
        """Upsert one node's evidence row."""
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO node_runs (job_id, node_id, operator, operator_version, stage, lineage,
                                       status, attempt, cache_hit, resumed, backend_id, config_hash,
                                       seed, input_digests, output_digests, started_at, finished_at,
                                       duration_ms, error_class, error_code, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id, node_id) DO UPDATE SET
                    status=excluded.status, attempt=excluded.attempt, cache_hit=excluded.cache_hit,
                    resumed=excluded.resumed, backend_id=excluded.backend_id,
                    output_digests=excluded.output_digests, started_at=excluded.started_at,
                    finished_at=excluded.finished_at, duration_ms=excluded.duration_ms,
                    error_class=excluded.error_class, error_code=excluded.error_code,
                    error_message=excluded.error_message
                """,
                (
                    evidence.job_id,
                    evidence.node_id,
                    evidence.operator,
                    evidence.operator_version,
                    evidence.stage,
                    evidence.lineage,
                    evidence.status.value,
                    evidence.attempt,
                    int(evidence.cache_hit),
                    int(evidence.resumed),
                    evidence.backend_id,
                    evidence.config_hash,
                    evidence.seed,
                    _json(evidence.input_digests),
                    _json(evidence.output_digests),
                    evidence.started_at.isoformat() if evidence.started_at else None,
                    evidence.finished_at.isoformat() if evidence.finished_at else None,
                    evidence.duration_ms,
                    evidence.error_class,
                    evidence.error_code,
                    evidence.error_message,
                ),
            )

    def record_node_result(
        self,
        job_id: str,
        result: NodeResult,
        *,
        stage: str,
        lineage: str,
        config_hash: str,
        seed: int,
        input_digests: list[str],
    ) -> None:
        """Store a node result payload so a resumed run can reuse it verbatim."""
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO node_runs (job_id, node_id, operator, operator_version, stage, lineage,
                                       status, attempt, cache_hit, resumed, backend_id, config_hash,
                                       seed, input_digests, output_digests, result, started_at,
                                       finished_at, duration_ms, error_class, error_code, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id, node_id) DO UPDATE SET
                    status=excluded.status, attempt=excluded.attempt, cache_hit=excluded.cache_hit,
                    resumed=excluded.resumed, backend_id=excluded.backend_id,
                    input_digests=excluded.input_digests, output_digests=excluded.output_digests,
                    result=excluded.result, started_at=excluded.started_at,
                    finished_at=excluded.finished_at, duration_ms=excluded.duration_ms,
                    error_class=excluded.error_class, error_code=excluded.error_code,
                    error_message=excluded.error_message
                """,
                (
                    job_id,
                    result.node_id,
                    result.operator,
                    result.operator_version,
                    stage,
                    lineage,
                    result.status.value,
                    result.attempt,
                    int(result.cache_hit),
                    int(result.resumed),
                    result.backend_id,
                    config_hash,
                    seed,
                    _json(input_digests),
                    _json([artifact.digest for artifact in result.artifacts]),
                    _json(result.model_dump(mode="json")),
                    result.started_at.isoformat(),
                    result.finished_at.isoformat(),
                    result.duration_ms,
                    result.error_class,
                    result.error_code,
                    result.error_message,
                ),
            )

    def node_result(self, job_id: str, node_id: str) -> NodeResult | None:
        """Read back a stored node result."""
        row = self.connection.execute(
            "SELECT result FROM node_runs WHERE job_id=? AND node_id=?", (job_id, node_id)
        ).fetchone()
        if row is None or not row["result"] or row["result"] == "{}":
            return None
        return NodeResult.model_validate(_loads(row["result"]))

    def node_evidence(self, job_id: str, node_id: str | None = None) -> list[dict[str, Any]]:
        """Evidence rows for a job, optionally narrowed to one node."""
        sql = "SELECT * FROM node_runs WHERE job_id=?"
        params: list[Any] = [job_id]
        if node_id:
            sql += " AND node_id=?"
            params.append(node_id)
        sql += " ORDER BY started_at"
        return [dict(row) for row in self.connection.execute(sql, params).fetchall()]

    def stage_totals(self, job_id: str) -> dict[str, dict[str, int]]:
        """Per-stage counts of node outcomes, for the job summary."""
        rows = self.connection.execute(
            "SELECT stage, status, COUNT(*) AS count FROM node_runs WHERE job_id=? GROUP BY stage, status",
            (job_id,),
        ).fetchall()
        totals: dict[str, dict[str, int]] = {}
        for row in rows:
            totals.setdefault(row["stage"], {})[row["status"]] = row["count"]
        return totals

    # -- cache ------------------------------------------------------------- #

    def cache_get(self, cache_key: str) -> NodeResult | None:
        """Read a cached node result and bump its hit counter."""
        row = self.connection.execute(
            "SELECT payload FROM node_cache WHERE cache_key=?", (cache_key,)
        ).fetchone()
        if row is None:
            return None
        with self.transaction() as connection:
            connection.execute(
                "UPDATE node_cache SET hits=hits+1, last_used_at=? WHERE cache_key=?",
                (utc_now().isoformat(), cache_key),
            )
        return NodeResult.model_validate(_loads(row["payload"]))

    def cache_put(
        self,
        cache_key: str,
        result: NodeResult,
        *,
        operator: str,
        operator_version: str,
        implementation: str,
        config_hash: str,
    ) -> None:
        """Store a node result in the cache."""
        now = utc_now().isoformat()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO node_cache (cache_key, operator, operator_version, implementation,
                                        config_hash, output_digests, payload, created_at, last_used_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO NOTHING
                """,
                (
                    cache_key,
                    operator,
                    operator_version,
                    implementation,
                    config_hash,
                    _json([artifact.digest for artifact in result.artifacts]),
                    _json(result.model_dump(mode="json")),
                    now,
                    now,
                ),
            )

    def cache_stats(self) -> dict[str, int]:
        """Cache size and total hits."""
        row = self.connection.execute(
            "SELECT COUNT(*) AS entries, COALESCE(SUM(hits), 0) AS hits FROM node_cache"
        ).fetchone()
        return {"entries": int(row["entries"]), "hits": int(row["hits"])}

    def clear_cache(self) -> int:
        """Delete every cache entry; returns how many were removed."""
        count = int(self.connection.execute("SELECT COUNT(*) AS n FROM node_cache").fetchone()["n"])
        with self.transaction() as connection:
            connection.execute("DELETE FROM node_cache")
        return count

    # -- artifacts --------------------------------------------------------- #

    def record_artifact(
        self, artifact: ArtifactRef, *, rel_path: str, job_id: str = "", producer_node: str = ""
    ) -> None:
        """Catalogue an artifact. Identical digests of the same kind are stored once."""
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO artifacts (digest, kind, media_type, rel_path, size_bytes, width, height,
                                       producer_node, job_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(digest, kind) DO NOTHING
                """,
                (
                    artifact.digest,
                    artifact.kind.value,
                    artifact.media_type,
                    rel_path,
                    artifact.size_bytes,
                    artifact.width,
                    artifact.height,
                    producer_node,
                    job_id,
                    utc_now().isoformat(),
                ),
            )

    def artifact_kinds(self, digest: str) -> tuple[str, ...]:
        """Kinds recorded for a digest."""
        rows = self.connection.execute("SELECT kind FROM artifacts WHERE digest=?", (digest,)).fetchall()
        return tuple(row["kind"] for row in rows)

    # -- candidates -------------------------------------------------------- #

    def upsert_candidate(
        self,
        *,
        candidate_id: str,
        job_id: str,
        sample_id: str,
        target_object_id: str,
        candidate_key: str,
        category: str,
        state: CandidateState,
        seed: int,
        source_digest: str,
        output_digest: str | None,
        overall_score: float | None,
        policy_hash: str,
        plan_json: dict[str, Any],
        intent_json: dict[str, Any],
        artifacts: tuple[str, ...] = (),
        reason_codes: tuple[str, ...] = (),
    ) -> None:
        """Insert or update a candidate row."""
        now = utc_now().isoformat()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO candidates (candidate_id, job_id, sample_id, target_object_id, candidate_key,
                                        category, state, seed, source_digest, output_digest, overall_score,
                                        policy_hash, plan_json, intent_json, artifacts, reason_codes,
                                        created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    state=excluded.state, output_digest=excluded.output_digest,
                    overall_score=excluded.overall_score, policy_hash=excluded.policy_hash,
                    artifacts=excluded.artifacts, reason_codes=excluded.reason_codes,
                    updated_at=excluded.updated_at
                """,
                (
                    candidate_id,
                    job_id,
                    sample_id,
                    target_object_id,
                    candidate_key,
                    category,
                    state.value,
                    seed,
                    source_digest,
                    output_digest,
                    overall_score,
                    policy_hash,
                    _json(plan_json),
                    _json(intent_json),
                    _json(list(artifacts)),
                    _json(list(reason_codes)),
                    now,
                    now,
                ),
            )

    def append_candidate_event(
        self,
        candidate_id: str,
        *,
        to_state: CandidateState,
        from_state: CandidateState | None = None,
        reason_codes: tuple[str, ...] = (),
        detail: str = "",
    ) -> None:
        """Append a lifecycle transition for a candidate."""
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO candidate_events (candidate_id, from_state, to_state, reason_codes, detail, occurred_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    from_state.value if from_state else None,
                    to_state.value,
                    _json(list(reason_codes)),
                    detail,
                    utc_now().isoformat(),
                ),
            )

    def candidate_events(self, candidate_id: str) -> list[dict[str, Any]]:
        """Lifecycle transitions recorded for one candidate."""
        rows = self.connection.execute(
            "SELECT * FROM candidate_events WHERE candidate_id=? ORDER BY id", (candidate_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def candidates(self, job_id: str, *, states: tuple[CandidateState, ...] = ()) -> list[StoredCandidate]:
        """Candidates of a job, optionally filtered by state."""
        sql = "SELECT * FROM candidates WHERE job_id=?"
        params: list[Any] = [job_id]
        if states:
            placeholders = ",".join("?" for _ in states)
            sql += f" AND state IN ({placeholders})"
            params.extend(state.value for state in states)
        sql += " ORDER BY created_at, candidate_id"
        rows = self.connection.execute(sql, params).fetchall()
        return [_candidate_from_row(row) for row in rows]

    def get_candidate(self, candidate_id: str) -> StoredCandidate | None:
        """Read one candidate row."""
        row = self.connection.execute(
            "SELECT * FROM candidates WHERE candidate_id=?", (candidate_id,)
        ).fetchone()
        return _candidate_from_row(row) if row else None

    def candidate_counts(self, job_id: str) -> dict[str, int]:
        """Candidate counts per state."""
        rows = self.connection.execute(
            "SELECT state, COUNT(*) AS count FROM candidates WHERE job_id=? GROUP BY state", (job_id,)
        ).fetchall()
        return {row["state"]: row["count"] for row in rows}

    def set_candidate_state(
        self,
        candidate_id: str,
        state: CandidateState,
        *,
        reason_codes: tuple[str, ...] = (),
        output_digest: str | None = None,
        overall_score: float | None = None,
    ) -> None:
        """Update a candidate's state, recording the transition."""
        current = self.get_candidate(candidate_id)
        if current is None:
            raise InfrastructureFailure(
                f"candidate {candidate_id} does not exist",
                code=ErrorCode.JOB_STATE_INVALID,
                detail={"candidate_id": candidate_id},
            )
        with self.transaction() as connection:
            connection.execute(
                """
                UPDATE candidates SET state=?, reason_codes=?, updated_at=?,
                       output_digest=COALESCE(?, output_digest),
                       overall_score=COALESCE(?, overall_score)
                WHERE candidate_id=?
                """,
                (
                    state.value,
                    _json(list(reason_codes)),
                    utc_now().isoformat(),
                    output_digest,
                    overall_score,
                    candidate_id,
                ),
            )
        if current.state is not state:
            self.append_candidate_event(
                candidate_id, to_state=state, from_state=current.state, reason_codes=reason_codes
            )

    # -- quality ----------------------------------------------------------- #

    def record_metrics(self, candidate_id: str, metrics: list[dict[str, Any]]) -> None:
        """Replace the metric rows of one candidate."""
        if not metrics:
            return
        with self.transaction() as connection:
            connection.execute("DELETE FROM quality_metrics WHERE candidate_id=?", (candidate_id,))
            connection.executemany(
                """
                INSERT INTO quality_metrics (candidate_id, metric, value, threshold, comparison, severity,
                                             status, evaluator_id, reason_codes, detail)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        candidate_id,
                        metric["metric"],
                        metric.get("value"),
                        metric.get("threshold"),
                        metric.get("comparison", "at_least"),
                        metric.get("severity", "hard"),
                        metric.get("status", "skipped"),
                        metric.get("evaluator_id"),
                        _json(metric.get("reason_codes", [])),
                        _json(metric.get("detail", {})),
                    )
                    for metric in metrics
                ],
            )

    def metrics_for(self, candidate_id: str) -> list[dict[str, Any]]:
        """Metric rows of one candidate."""
        rows = self.connection.execute(
            "SELECT * FROM quality_metrics WHERE candidate_id=? ORDER BY metric", (candidate_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def metric_averages(self, job_id: str) -> dict[str, float]:
        """Mean metric value per metric across a job's candidates."""
        rows = self.connection.execute(
            """
            SELECT m.metric AS metric, AVG(m.value) AS average
            FROM quality_metrics m JOIN candidates c ON c.candidate_id = m.candidate_id
            WHERE c.job_id=? AND m.value IS NOT NULL
            GROUP BY m.metric
            """,
            (job_id,),
        ).fetchall()
        return {row["metric"]: float(row["average"]) for row in rows}

    # -- samples and lineage ----------------------------------------------- #

    def register_sample(
        self,
        *,
        sample_id: str,
        job_id: str | None,
        root_digest: str,
        digest: str,
        rel_path: str,
        split: str,
        state: SampleState,
        source_sample_id: str | None = None,
        augmentation_depth: int = 0,
        perceptual_hash: str | None = None,
        class_histogram: dict[str, int] | None = None,
        embedding_ref: str | None = None,
    ) -> None:
        """Insert or update a sample row."""
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO samples (sample_id, job_id, source_sample_id, root_digest, digest, rel_path,
                                     split, augmentation_depth, perceptual_hash, embedding_ref,
                                     class_histogram, state, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sample_id) DO UPDATE SET
                    job_id=excluded.job_id, split=excluded.split, state=excluded.state,
                    perceptual_hash=excluded.perceptual_hash, rel_path=excluded.rel_path,
                    class_histogram=excluded.class_histogram
                """,
                (
                    sample_id,
                    job_id,
                    source_sample_id,
                    root_digest,
                    digest,
                    rel_path,
                    split,
                    augmentation_depth,
                    perceptual_hash,
                    embedding_ref,
                    _json(class_histogram or {}),
                    state.value,
                    utc_now().isoformat(),
                ),
            )

    def link_lineage(
        self, child_sample_id: str, parent_sample_id: str, *, relation: str = "augmentation"
    ) -> None:
        """Record a lineage edge."""
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO lineage_edges (child_sample_id, parent_sample_id, relation) VALUES (?, ?, ?)
                ON CONFLICT(child_sample_id, parent_sample_id) DO NOTHING
                """,
                (child_sample_id, parent_sample_id, relation),
            )

    def ancestors(self, sample_id: str) -> tuple[str, ...]:
        """Every ancestor of a sample, transitive, nearest first."""
        seen: set[str] = set()
        ordered: list[str] = []
        frontier = [sample_id]
        while frontier:
            current = frontier.pop(0)
            rows = self.connection.execute(
                "SELECT parent_sample_id FROM lineage_edges WHERE child_sample_id=?", (current,)
            ).fetchall()
            for row in rows:
                parent = row["parent_sample_id"]
                if parent not in seen:
                    seen.add(parent)
                    ordered.append(parent)
                    frontier.append(parent)
        return tuple(ordered)

    def sample(self, sample_id: str) -> dict[str, Any] | None:
        """Read one sample row."""
        row = self.connection.execute("SELECT * FROM samples WHERE sample_id=?", (sample_id,)).fetchone()
        return dict(row) if row else None

    def samples(
        self, *, job_id: str | None = None, split: str | None = None, state: SampleState | None = None
    ) -> list[dict[str, Any]]:
        """List sample rows with optional filters."""
        sql = "SELECT * FROM samples WHERE 1=1"
        params: list[Any] = []
        if job_id:
            sql += " AND job_id=?"
            params.append(job_id)
        if split:
            sql += " AND split=?"
            params.append(split)
        if state:
            sql += " AND state=?"
            params.append(state.value)
        sql += " ORDER BY created_at"
        return [dict(row) for row in self.connection.execute(sql, params).fetchall()]

    def split_of(self, sample_id: str) -> str | None:
        """Split recorded for a sample, if known."""
        row = self.connection.execute("SELECT split FROM samples WHERE sample_id=?", (sample_id,)).fetchone()
        return row["split"] if row else None

    def find_leakage(self) -> list[tuple[str, str, str]]:
        """Find lineage edges whose endpoints disagree about their split.

        Returns:
            ``(child, parent, child_split)`` triples for every violation.
        """
        rows = self.connection.execute(
            """
            SELECT e.child_sample_id AS child, e.parent_sample_id AS parent,
                   child.split AS child_split, parent.split AS parent_split
            FROM lineage_edges e
            JOIN samples child ON child.sample_id = e.child_sample_id
            JOIN samples parent ON parent.sample_id = e.parent_sample_id
            WHERE child.split <> parent.split
            """
        ).fetchall()
        return [(row["child"], row["parent"], row["child_split"]) for row in rows]

    def record_duplicate(
        self, job_id: str, candidate_id: str, duplicate_of: str, *, kind: str, distance: float
    ) -> None:
        """Record a duplicate decision so exports are auditable."""
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO duplicates (job_id, candidate_id, duplicate_of, kind, distance)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(job_id, candidate_id, duplicate_of) DO NOTHING
                """,
                (job_id, candidate_id, duplicate_of, kind, distance),
            )

    def duplicates(self, job_id: str) -> list[dict[str, Any]]:
        """Duplicate decisions recorded for a job."""
        rows = self.connection.execute("SELECT * FROM duplicates WHERE job_id=?", (job_id,)).fetchall()
        return [dict(row) for row in rows]

    def perceptual_hashes(self, *, exclude_job: str | None = None) -> dict[str, str]:
        """Every known perceptual hash, keyed by sample id, for duplicate comparison."""
        sql = "SELECT sample_id, perceptual_hash FROM samples WHERE perceptual_hash IS NOT NULL"
        params: list[Any] = []
        if exclude_job:
            sql += " AND (job_id IS NULL OR job_id <> ?)"
            params.append(exclude_job)
        rows = self.connection.execute(sql, params).fetchall()
        return {row["sample_id"]: row["perceptual_hash"] for row in rows}

    # -- maintenance ------------------------------------------------------- #

    def vacuum(self) -> None:
        """Compact the database file."""
        self.connection.execute("VACUUM")

    def counts(self) -> dict[str, int]:
        """Row counts per table, for diagnostics."""
        tables = ("jobs", "node_runs", "candidates", "artifacts", "node_cache", "samples", "lineage_edges")
        return {
            table: int(self.connection.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])
            for table in tables
        }


def _candidate_from_row(row: sqlite3.Row) -> StoredCandidate:
    return StoredCandidate(
        candidate_id=row["candidate_id"],
        job_id=row["job_id"],
        sample_id=row["sample_id"],
        target_object_id=row["target_object_id"],
        candidate_key=row["candidate_key"],
        category=row["category"],
        state=CandidateState(row["state"]),
        seed=row["seed"],
        source_digest=row["source_digest"],
        output_digest=row["output_digest"],
        overall_score=row["overall_score"],
        policy_hash=row["policy_hash"],
        reason_codes=tuple(_loads(row["reason_codes"]) or ()),
        artifacts=tuple(_loads(row["artifacts"]) or ()),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


class EvidenceRecorder:
    """Bridges an engine run to durable state.

    It implements the engine's ``RunState`` protocol: node results are stored verbatim so a resumed
    job can reuse them, evidence rows are written for audit, and the node cache is consulted only
    after the referenced artifacts have been verified to exist on disk.
    """

    def __init__(
        self,
        store: StateStore,
        *,
        job_id: str,
        artifact_exists: Callable[[ArtifactRef], bool] | None = None,
    ) -> None:
        self._store = store
        self._job_id = job_id
        self._artifact_exists = artifact_exists

    @property
    def job_id(self) -> str:
        """Job this recorder is attached to."""
        return self._job_id

    def completed_node(self, node_id: str) -> NodeResult | None:
        """Return the stored result of a finished node, for resume."""
        result = self._store.node_result(self._job_id, node_id)
        if result is None or result.status is not NodeStatus.SUCCEEDED:
            return None
        return result

    def record_node(self, result: NodeResult, node: OperationNode, *, seed: int, config_hash: str) -> None:
        """Persist a node result and its evidence row."""
        input_digests: list[str] = []
        self._store.record_node_result(
            self._job_id,
            result,
            stage=node.stage.value,
            lineage=node.lineage_label,
            config_hash=config_hash,
            seed=seed,
            input_digests=input_digests,
        )
        self._store.record_node_evidence(
            OperationEvidence(
                job_id=self._job_id,
                node_id=node.node_id,
                operator=node.operator,
                operator_version=node.operator_version,
                stage=node.stage.value,
                lineage=node.lineage_label,
                status=result.status,
                attempt=result.attempt,
                cache_hit=result.cache_hit,
                resumed=result.resumed,
                backend_id=result.backend_id,
                config_hash=config_hash,
                seed=seed,
                input_digests=input_digests,
                output_digests=[artifact.digest for artifact in result.artifacts],
                started_at=result.started_at,
                finished_at=result.finished_at,
                duration_ms=result.duration_ms,
                error_class=result.error_class,
                error_code=result.error_code,
                error_message=result.error_message,
            )
        )

    def cached_result(self, cache_key: str) -> NodeResult | None:
        """Return a verified cached result, or ``None`` when absent or no longer valid."""
        result = self._store.cache_get(cache_key)
        if result is None:
            return None
        if not self.artifacts_present(result):
            return None
        return result

    def store_cache(self, cache_key: str, result: NodeResult, node: OperationNode) -> None:
        """Persist a node result under ``cache_key``."""
        self._store.cache_put(
            cache_key,
            result,
            operator=node.operator,
            operator_version=node.operator_version,
            implementation=f"operator:{node.operator}@{node.operator_version}",
            config_hash=node.config_hash,
        )

    def artifacts_present(self, result: NodeResult) -> bool:
        """Whether every artifact referenced by ``result`` is still readable.

        Artifacts created *during* this execution are not registered yet, so a missing check
        function means "assume present" only for freshly produced results; a resumed or cached
        result is always verified because it was produced by an earlier process.
        """
        if self._artifact_exists is None:
            return True
        return all(self._artifact_exists(artifact) for artifact in result.artifacts)


class InMemoryRunState:
    """Engine state kept in process memory. Used by tests and by ``--no-resume`` smoke runs."""

    def __init__(self, artifact_exists: Callable[[ArtifactRef], bool] | None = None) -> None:
        self._nodes: dict[str, NodeResult] = {}
        self._cache: dict[str, NodeResult] = {}
        self._artifact_exists = artifact_exists

    def completed_node(self, node_id: str) -> NodeResult | None:
        """Return a previously recorded successful node result."""
        result = self._nodes.get(node_id)
        if result is None or result.status is not NodeStatus.SUCCEEDED:
            return None
        return result

    def record_node(self, result: NodeResult, node: OperationNode, *, seed: int, config_hash: str) -> None:
        """Store a node result in memory."""
        del node, seed, config_hash
        self._nodes[result.node_id] = result

    def cached_result(self, cache_key: str) -> NodeResult | None:
        """Return an in-memory cached result, verifying its artifacts when a checker was supplied."""
        result = self._cache.get(cache_key)
        if result is None:
            return None
        if self._artifact_exists is not None and not all(
            self._artifact_exists(artifact) for artifact in result.artifacts
        ):
            return None
        return result

    def store_cache(self, cache_key: str, result: NodeResult, node: OperationNode) -> None:
        """Store a node result in the in-memory cache."""
        del node
        self._cache[cache_key] = result

    def artifacts_present(self, result: NodeResult) -> bool:
        """Whether every artifact of a result exists on disk."""
        if self._artifact_exists is None:
            return True
        return all(self._artifact_exists(artifact) for artifact in result.artifacts)

    @property
    def cached_entries(self) -> int:
        """How many entries the in-memory cache holds."""
        return len(self._cache)
