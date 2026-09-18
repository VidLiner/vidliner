# VidLiner — Persistence Schema (SQLite)

One database per workspace: `<workspace>/state.db`, WAL mode, `foreign_keys=ON`,
`synchronous=NORMAL`. Schema version is stored in `schema_meta` and migrated forward by
`vidliner.storage.migrate`. The database stores *state*, never pixels: images, masks, and HTML
reports live in the content-addressed artifact store.

```sql
CREATE TABLE schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- One row per job attempt.
CREATE TABLE jobs (
    job_id            TEXT PRIMARY KEY,
    state             TEXT NOT NULL,              -- JobState
    recipe_name       TEXT NOT NULL,
    recipe_hash       TEXT NOT NULL,
    recipe_snapshot   TEXT NOT NULL,              -- canonical JSON
    runtime_snapshot  TEXT NOT NULL,              -- canonical JSON, credentials redacted
    seed              INTEGER NOT NULL,
    workspace_root    TEXT NOT NULL,
    run_dir           TEXT NOT NULL,
    node_count        INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    finished_at       TEXT,
    counters          TEXT NOT NULL DEFAULT '{}', -- processed/accepted/rejected/review/failed
    failure_class     TEXT,
    failure_code      TEXT,
    failure_message   TEXT
);
CREATE INDEX jobs_state_created ON jobs(state, created_at DESC);

-- One row per executed (or skipped) graph node.
CREATE TABLE node_runs (
    job_id            TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    node_id           TEXT NOT NULL,
    operator          TEXT NOT NULL,
    operator_version  TEXT NOT NULL,
    stage             TEXT NOT NULL,
    lineage           TEXT NOT NULL,              -- ':'-joined identity path
    status            TEXT NOT NULL,              -- pending|running|succeeded|failed|skipped|cancelled
    attempt           INTEGER NOT NULL DEFAULT 0,
    cache_hit         INTEGER NOT NULL DEFAULT 0,
    resumed           INTEGER NOT NULL DEFAULT 0,
    backend_id        TEXT,
    config_hash       TEXT NOT NULL,
    seed              INTEGER NOT NULL,
    input_digests     TEXT NOT NULL DEFAULT '[]', -- JSON array
    output_digests    TEXT NOT NULL DEFAULT '[]', -- JSON array
    started_at        TEXT,
    finished_at       TEXT,
    duration_ms       INTEGER,
    error_class       TEXT,
    error_code        TEXT,
    error_message     TEXT,
    PRIMARY KEY (job_id, node_id)
);
CREATE INDEX node_runs_job_stage ON node_runs(job_id, stage, status);

-- One row per generated candidate; the lifecycle is first-class data (requirement 15).
CREATE TABLE candidates (
    candidate_id      TEXT PRIMARY KEY,
    job_id            TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    sample_id         TEXT NOT NULL,
    target_object_id  TEXT NOT NULL,
    candidate_key     TEXT NOT NULL,
    category          TEXT NOT NULL,
    state             TEXT NOT NULL,              -- CandidateState
    seed              INTEGER NOT NULL,
    source_digest     TEXT NOT NULL,
    output_digest     TEXT,
    overall_score     REAL,
    policy_hash       TEXT NOT NULL,
    plan_json         TEXT NOT NULL,
    intent_json       TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
CREATE INDEX candidates_job_state ON candidates(job_id, state);
CREATE INDEX candidates_sample ON candidates(job_id, sample_id);

-- Append-only lifecycle transitions.
CREATE TABLE candidate_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id      TEXT NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    from_state        TEXT,
    to_state          TEXT NOT NULL,
    reason_codes      TEXT NOT NULL DEFAULT '[]',
    detail            TEXT,
    occurred_at       TEXT NOT NULL
);

-- Quality evidence, one row per metric.
CREATE TABLE quality_metrics (
    candidate_id      TEXT NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    metric            TEXT NOT NULL,
    value             REAL,
    threshold         REAL,
    comparison        TEXT NOT NULL,
    severity          TEXT NOT NULL,
    status            TEXT NOT NULL,
    evaluator_id      TEXT,
    reason_codes      TEXT NOT NULL DEFAULT '[]',
    detail            TEXT,
    PRIMARY KEY (candidate_id, metric)
);

-- Artifact catalogue. `digest` values also exist on disk; presence is re-verified on use.
CREATE TABLE artifacts (
    digest            TEXT PRIMARY KEY,
    kind              TEXT NOT NULL,              -- ArtifactKind
    media_type        TEXT NOT NULL,
    rel_path          TEXT NOT NULL,
    size_bytes        INTEGER NOT NULL,
    width             INTEGER,
    height            INTEGER,
    producer_node     TEXT,
    job_id            TEXT,
    created_at        TEXT NOT NULL
);
CREATE INDEX artifacts_kind ON artifacts(kind);

-- Node-level result cache (ADR-005). Survives job deletion.
CREATE TABLE node_cache (
    cache_key         TEXT PRIMARY KEY,
    operator          TEXT NOT NULL,
    operator_version  TEXT NOT NULL,
    implementation    TEXT NOT NULL,
    config_hash       TEXT NOT NULL,
    output_digests    TEXT NOT NULL,
    payload           TEXT NOT NULL,              -- canonical JSON of the operator result
    hits              INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL,
    last_used_at      TEXT NOT NULL
);

-- Dataset samples: both sources and produced outputs, for lineage and duplicate control.
CREATE TABLE samples (
    sample_id         TEXT PRIMARY KEY,
    job_id            TEXT REFERENCES jobs(job_id) ON DELETE SET NULL,
    source_sample_id  TEXT,
    root_digest       TEXT NOT NULL,
    digest            TEXT NOT NULL,
    rel_path          TEXT NOT NULL,
    split             TEXT NOT NULL,
    augmentation_depth INTEGER NOT NULL DEFAULT 0,
    perceptual_hash   TEXT,
    embedding_ref     TEXT,
    class_histogram   TEXT NOT NULL DEFAULT '{}',
    state             TEXT NOT NULL,              -- source|accepted|rejected|review
    created_at        TEXT NOT NULL
);
CREATE INDEX samples_split_state ON samples(split, state);
CREATE INDEX samples_root ON samples(root_digest);
CREATE INDEX samples_phash ON samples(perceptual_hash);

-- Lineage edges for leakage validation (requirement 24).
CREATE TABLE lineage_edges (
    child_sample_id   TEXT NOT NULL,
    parent_sample_id  TEXT NOT NULL,
    relation          TEXT NOT NULL,              -- augmentation|derived
    PRIMARY KEY (child_sample_id, parent_sample_id)
);

-- Per-split duplicate decisions, so export is auditable.
CREATE TABLE duplicates (
    job_id            TEXT NOT NULL,
    candidate_id      TEXT NOT NULL,
    duplicate_of      TEXT NOT NULL,
    kind              TEXT NOT NULL,              -- exact|near
    distance          REAL NOT NULL,
    PRIMARY KEY (job_id, candidate_id, duplicate_of)
);
```

## Access layer

`vidliner.storage.state.Store` wraps a `sqlite3` connection with:

* `initialize()` — create tables, set pragmas, stamp schema version; idempotent.
* Typed helpers per aggregate (`create_job`, `transition_job`, `record_node`, `upsert_candidate`,
  `append_candidate_event`, `record_metrics`, `record_artifact`, `cache_lookup`, `cache_store`,
  `register_sample`, `link_lineage`, `find_leakage`, `record_duplicate`).
* All timestamps are ISO-8601 UTC with a trailing `Z`; all JSON columns are canonical JSON.
* Writes happen on the event loop's thread through a single connection guarded by an
  `asyncio.Lock`; the store is not shared across processes.
