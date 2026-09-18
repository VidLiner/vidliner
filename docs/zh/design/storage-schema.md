# VidLiner —— 持久化 Schema（SQLite）

[English](../../design/storage-schema.md)

每个工作区一个数据库：`<workspace>/state.db`，WAL 模式，`foreign_keys=ON`，`synchronous=NORMAL`。Schema 版本记录在 `schema_meta`，由 `vidliner.storage.state` 向前迁移。数据库只存*状态*，不存像素：图像、mask 与 HTML 报告放在内容寻址的产物存储中。

```sql
CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

-- 每次 job 尝试一行
CREATE TABLE jobs (
    job_id            TEXT PRIMARY KEY,
    state             TEXT NOT NULL,              -- JobState
    recipe_name       TEXT NOT NULL,
    recipe_hash       TEXT NOT NULL,
    recipe_snapshot   TEXT NOT NULL,              -- 规范化 JSON
    runtime_snapshot  TEXT NOT NULL,              -- 规范化 JSON，凭据已脱敏
    runtime_profile   TEXT NOT NULL DEFAULT '',
    seed              INTEGER NOT NULL,
    workspace_root    TEXT NOT NULL,
    run_dir           TEXT NOT NULL,
    dataset_input     TEXT NOT NULL DEFAULT '',
    output_path       TEXT NOT NULL DEFAULT '',
    node_count        INTEGER NOT NULL DEFAULT 0,
    manifest          TEXT NOT NULL,
    counters          TEXT NOT NULL DEFAULT '{}', -- processed/accepted/rejected/review/failed
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    finished_at       TEXT,
    failure_class     TEXT, failure_code TEXT, failure_message TEXT
);
CREATE INDEX jobs_state_created ON jobs(state, created_at DESC);

-- 每个已执行（或被跳过）的图节点一行
CREATE TABLE node_runs (
    job_id            TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    node_id           TEXT NOT NULL,
    operator          TEXT NOT NULL,
    operator_version  TEXT NOT NULL,
    stage             TEXT NOT NULL,
    lineage           TEXT NOT NULL,
    status            TEXT NOT NULL,              -- pending|running|succeeded|failed|skipped|cancelled
    attempt           INTEGER NOT NULL DEFAULT 0,
    cache_hit         INTEGER NOT NULL DEFAULT 0,
    resumed           INTEGER NOT NULL DEFAULT 0,
    backend_id        TEXT,
    config_hash       TEXT NOT NULL DEFAULT '',
    seed              INTEGER NOT NULL DEFAULT 0,
    input_digests     TEXT NOT NULL DEFAULT '[]',
    output_digests    TEXT NOT NULL DEFAULT '[]',
    result            TEXT NOT NULL DEFAULT '{}', -- 完整 NodeResult，供恢复时回填
    started_at        TEXT, finished_at TEXT, duration_ms INTEGER,
    error_class       TEXT, error_code TEXT, error_message TEXT,
    PRIMARY KEY (job_id, node_id)
);
CREATE INDEX node_runs_job_stage ON node_runs(job_id, stage, status);

-- 每个生成的候选一行；生命周期是一等数据（需求 §15）
CREATE TABLE candidates (
    candidate_id      TEXT PRIMARY KEY,
    job_id            TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    sample_id         TEXT NOT NULL,
    target_object_id  TEXT NOT NULL,
    candidate_key     TEXT NOT NULL,
    category          TEXT NOT NULL,
    state             TEXT NOT NULL,              -- CandidateState
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
CREATE INDEX candidates_job_state ON candidates(job_id, state);
CREATE INDEX candidates_sample ON candidates(job_id, sample_id);

-- 只追加的生命周期迁移
CREATE TABLE candidate_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id      TEXT NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    from_state        TEXT, to_state TEXT NOT NULL,
    reason_codes      TEXT NOT NULL DEFAULT '[]',
    detail            TEXT, occurred_at TEXT NOT NULL
);

-- 质量证据：每个指标一行
CREATE TABLE quality_metrics (
    candidate_id      TEXT NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    metric            TEXT NOT NULL,
    value             REAL, threshold REAL,
    comparison        TEXT NOT NULL DEFAULT 'at_least',
    severity          TEXT NOT NULL DEFAULT 'hard',
    status            TEXT NOT NULL DEFAULT 'skipped',
    evaluator_id      TEXT,
    reason_codes      TEXT NOT NULL DEFAULT '[]',
    detail            TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (candidate_id, metric)
);

-- 产物目录；磁盘上同样以摘要为名，取用时重新验证存在性
CREATE TABLE artifacts (
    digest            TEXT NOT NULL,
    kind              TEXT NOT NULL,              -- ArtifactKind
    media_type        TEXT NOT NULL,
    rel_path          TEXT NOT NULL,
    size_bytes        INTEGER NOT NULL,
    width INTEGER, height INTEGER,
    producer_node     TEXT, job_id TEXT,
    created_at        TEXT NOT NULL,
    PRIMARY KEY (digest, kind)
);
CREATE INDEX artifacts_kind ON artifacts(kind);

-- 节点级结果缓存（ADR-005）；删除 job 后依然保留
CREATE TABLE node_cache (
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

-- 数据集样本：源样本与产出样本，用于 lineage 与去重
CREATE TABLE samples (
    sample_id         TEXT PRIMARY KEY,
    job_id            TEXT REFERENCES jobs(job_id) ON DELETE SET NULL,
    -- 刻意不加外键：'vidliner inspect' 发现的源样本可能尚未登记，跨 job 的源样本也必须可引用
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
CREATE INDEX samples_split_state ON samples(split, state);
CREATE INDEX samples_root ON samples(root_digest);
CREATE INDEX samples_phash ON samples(perceptual_hash);

-- 用于防泄漏校验的血缘边（需求 §24）
CREATE TABLE lineage_edges (
    child_sample_id   TEXT NOT NULL,
    parent_sample_id  TEXT NOT NULL,
    relation          TEXT NOT NULL DEFAULT 'augmentation',
    PRIMARY KEY (child_sample_id, parent_sample_id)
);

-- 逐 split 的重复判定，使导出可审计
CREATE TABLE duplicates (
    job_id            TEXT NOT NULL,
    candidate_id      TEXT NOT NULL,
    duplicate_of      TEXT NOT NULL,
    kind              TEXT NOT NULL,              -- exact | near
    distance          REAL NOT NULL,
    PRIMARY KEY (job_id, candidate_id, duplicate_of)
);

-- 生成的报告文件位置
CREATE TABLE job_reports (
    job_id            TEXT NOT NULL,
    report_name       TEXT NOT NULL,
    rel_path          TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    PRIMARY KEY (job_id, report_name)
);
```

## 访问层

`vidliner.storage.state.Store` 封装一个 `sqlite3` 连接，提供：

* `initialize()` —— 建表、设置 pragma、写入 schema 版本；幂等。
* 每个聚合的带类型辅助方法：`create_job`、`transition_job`、`record_node_result`、
  `upsert_candidate`、`append_candidate_event`、`record_metrics`、`record_artifact`、
  `cache_get`/`cache_put`、`register_sample`、`link_lineage`、`find_leakage`、`record_duplicate`。
* 所有时间戳为带 `Z` 的 ISO-8601 UTC；所有 JSON 列为规范化 JSON。
* 写入发生在事件循环线程上，由单个连接串行执行；连接不跨进程共享。

`EvidenceRecorder` 把引擎的运行对接到持久化状态：节点结果原样存储（以便恢复时回填端口值），证据行用于审计，而缓存只在引用到的产物被验证存在之后才被采用。`InMemoryRunState` 提供同一协议的纯内存实现，供测试使用。
