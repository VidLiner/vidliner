"""Durable local canvas drafts and execution intents, separate from accepted dataset jobs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.domain.workflow import WorkflowDocument
from vidliner.domain.workflow_edit import WorkflowEditRequest
from vidliner.pipeline.workflow_edit import apply_workflow_edits, workflow_digest


class CanvasStore:
    """SQLite compare-and-replace for drafts and unique claims for execution requests."""

    def __init__(self, path: Path, document: WorkflowDocument) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        path.chmod(0o600)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript(
            "CREATE TABLE IF NOT EXISTS draft (id INTEGER PRIMARY KEY, base_digest TEXT, document TEXT);"
            "CREATE TABLE IF NOT EXISTS executions (id TEXT PRIMARY KEY, identity TEXT, snapshot TEXT);"
        )
        initial = workflow_digest(document)
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO draft VALUES (1, ?, ?)", (initial, document.model_dump_json())
            )
        base = self.connection.execute("SELECT base_digest FROM draft WHERE id=1").fetchone()[0]
        if base != initial:
            self.connection.close()
            raise ValidationFailure(
                "source workflow changed; use a separate workspace to preserve the saved draft",
                code=ErrorCode.GRAPH_INVALID,
            )

    def document(self) -> WorkflowDocument:
        """Load the current validated canvas draft."""
        row = self.connection.execute("SELECT document FROM draft WHERE id=1").fetchone()
        return WorkflowDocument.model_validate_json(row[0])

    def edit(self, request: WorkflowEditRequest) -> WorkflowDocument:
        """Compare and validate the entire postimage under a SQLite write lock."""
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            edited = apply_workflow_edits(self.document(), request)
            self.connection.execute("UPDATE draft SET document=? WHERE id=1", (edited.model_dump_json(),))
            self.connection.commit()
            return edited
        except Exception:
            self.connection.rollback()
            raise

    def claim(self, job_id: str, identity: str, snapshot: dict[str, Any]) -> dict[str, Any] | None:
        """Claim once; return a previous identical intent or reject reused IDs."""
        try:
            with self.connection:
                self.connection.execute(
                    "INSERT INTO executions VALUES (?, ?, ?)", (job_id, identity, json.dumps(snapshot))
                )
        except sqlite3.IntegrityError:
            old = self.connection.execute(
                "SELECT identity, snapshot FROM executions WHERE id=?", (job_id,)
            ).fetchone()
            if old[0] != identity:
                raise ValidationFailure(
                    "job id already names another intent", code=ErrorCode.GRAPH_INVALID
                ) from None
            return json.loads(old[1])
        return None

    def save_job(self, snapshot: dict[str, Any]) -> None:
        """Persist node evidence, task handles and current execution status."""
        with self.connection:
            self.connection.execute(
                "UPDATE executions SET snapshot=? WHERE id=?",
                (json.dumps(snapshot), snapshot["job_id"]),
            )

    def job(self, job_id: str) -> dict[str, Any]:
        """Read a private execution snapshot or report a missing job."""
        row = self.connection.execute("SELECT snapshot FROM executions WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise ValidationFailure("unknown canvas job", code=ErrorCode.GRAPH_INVALID)
        return json.loads(row[0])

    def jobs(self) -> list[dict[str, Any]]:
        """Return the most recent private execution snapshots."""
        rows = self.connection.execute("SELECT snapshot FROM executions ORDER BY rowid DESC LIMIT 50")
        return [json.loads(row[0]) for row in rows]

    def close(self) -> None:
        """Release the database connection."""
        self.connection.close()
