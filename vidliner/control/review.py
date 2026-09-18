"""Static HTML review report.

``NEEDS_REVIEW`` candidates need a human, and the first version of that interface is a single HTML
file that opens from disk with no server: the source frame, the target mask, the generated
candidate, an amplified difference map, the rebuilt annotation overlay, and every metric with its
gate and reason codes.

Each row carries ``data-candidate-id`` and its decision, so a future web UI can reuse the same
document — and so a reviewer can record a verdict by editing one JSON file rather than by clicking
in a page that does not exist yet. The interface is defined by :func:`read_review_decisions` and
:func:`write_review_decisions`, which is the contract a web UI would implement.
"""

from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vidliner.core.results import JobCounters
from vidliner.domain.jobs import JobManifest
from vidliner.storage.state import StateStore
from vidliner.storage.workspace import Workspace

__all__ = [
    "ReviewEntry",
    "read_review_decisions",
    "write_review_decisions",
    "write_review_report",
]

_CSS = """
body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; background: #101215; color: #e8eaed; }
header { padding: 20px 28px; background: #171a1f; border-bottom: 1px solid #262b33; }
h1 { font-size: 18px; margin: 0 0 6px; font-weight: 600; }
h2 { font-size: 14px; margin: 0 0 12px; font-weight: 600; color: #9aa4b2; text-transform: uppercase; letter-spacing: .06em; }
.meta { font-size: 13px; color: #9aa4b2; }
.counters { display: flex; gap: 18px; padding: 14px 28px; flex-wrap: wrap; font-size: 13px; }
.counters b { color: #fff; font-size: 16px; display: block; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 18px; padding: 20px 28px 60px; }
.card { background: #171a1f; border: 1px solid #262b33; border-radius: 10px; overflow: hidden; }
.card header { background: transparent; border: 0; padding: 14px 16px 6px; }
.card h3 { font-size: 14px; margin: 0; }
.card .sub { font-size: 12px; color: #9aa4b2; margin-top: 4px; }
.images { display: grid; grid-template-columns: 1fr 1fr; gap: 2px; background: #262b33; }
.images img { width: 100%; display: block; background: #0b0d10; }
.metrics { padding: 12px 16px 16px; font-size: 12px; }
table { width: 100%; border-collapse: collapse; }
td { padding: 3px 0; border-bottom: 1px solid #21252c; }
td.value { text-align: right; font-variant-numeric: tabular-nums; }
.fail { color: #ff7b72; font-weight: 600; }
.warn { color: #e3b341; }
.pass { color: #3fb950; }
.reasons { padding: 0 16px 16px; display: flex; flex-wrap: wrap; gap: 6px; }
.reason { font-size: 11px; padding: 3px 8px; border-radius: 999px; background: #2d1d1f; color: #ff9d94; }
.reason.ok { background: #16281a; color: #7ee787; }
.decision { font-size: 12px; padding: 2px 10px; border-radius: 999px; display: inline-block; }
.decision.accepted { background: #16281a; color: #7ee787; }
.decision.rejected { background: #2d1d1f; color: #ff9d94; }
.decision.review { background: #2b2412; color: #e3b341; }
footer { padding: 0 28px 40px; color: #6b7280; font-size: 12px; }
code { background: #0b0d10; padding: 1px 5px; border-radius: 4px; }
"""


@dataclass(frozen=True, slots=True)
class ReviewEntry:
    """One candidate in the review report."""

    candidate_id: str
    sample_id: str
    candidate_key: str
    category: str
    decision: str
    overall_score: float | None
    metrics: list[dict[str, Any]]
    reason_codes: tuple[str, ...]
    images: dict[str, str]
    evidence_dir: str

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form, used by :func:`write_review_decisions` round trips."""
        return {
            "candidate_id": self.candidate_id,
            "sample_id": self.sample_id,
            "candidate_key": self.candidate_key,
            "category": self.category,
            "decision": self.decision,
            "overall_score": self.overall_score,
            "metrics": self.metrics,
            "reason_codes": list(self.reason_codes),
            "evidence_dir": self.evidence_dir,
        }


def write_review_report(
    *,
    workspace: Workspace,
    state: StateStore,
    job_id: str,
    manifest: JobManifest,
    counters: JobCounters,
    include_accepted: bool = False,
) -> Path | None:
    """Write ``review/index.html`` for a job and return its path."""
    entries = _collect_entries(
        workspace=workspace,
        state=state,
        job_id=job_id,
        include_accepted=include_accepted,
    )
    review_dir = workspace.run_dir(job_id) / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    document = _render(manifest, counters, entries)
    target = review_dir / "index.html"
    target.write_text(document, encoding="utf-8")
    write_review_decisions(review_dir / "decisions.json", entries)
    return target


def write_review_decisions(path: Path, entries: list[ReviewEntry]) -> Path:
    """Write the reviewer's decision file, pre-filled with the machine decision.

    The file is the interface a web UI would read and write: one entry per candidate with a
    ``reviewer_decision`` field left empty.
    """
    payload = {
        "format": "vidliner.review-decisions@1",
        "instructions": (
            "Set reviewer_decision to 'accepted' or 'rejected' for each candidate, then re-export "
            "with 'vidliner export <job>'. Leave it empty to keep the machine decision."
        ),
        "entries": [{**entry.as_dict(), "reviewer_decision": "", "reviewer_note": ""} for entry in entries],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_review_decisions(path: Path) -> dict[str, str]:
    """Read a reviewer decision file into ``candidate_id → decision``."""
    if not path.is_file():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    entries = document.get("entries") if isinstance(document, dict) else None
    if not isinstance(entries, list):
        return {}
    decisions: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        candidate = entry.get("candidate_id")
        decision = entry.get("reviewer_decision")
        if isinstance(candidate, str) and decision in {"accepted", "rejected"}:
            decisions[candidate] = decision
    return decisions


def _collect_entries(
    *,
    workspace: Workspace,
    state: StateStore,
    job_id: str,
    include_accepted: bool,
) -> list[ReviewEntry]:
    wanted = {"review", "rejected"} | ({"accepted"} if include_accepted else set())
    entries: list[ReviewEntry] = []
    for candidate in state.candidates(job_id):
        if candidate.state.value not in wanted:
            continue
        sample_dir = workspace.sample_dir(job_id, candidate.sample_id)
        metrics = state.metrics_for(candidate.candidate_id)
        entries.append(
            ReviewEntry(
                candidate_id=candidate.candidate_id,
                sample_id=candidate.sample_id,
                candidate_key=candidate.candidate_key,
                category=candidate.category,
                decision=candidate.state.value,
                overall_score=candidate.overall_score,
                metrics=metrics,
                reason_codes=candidate.reason_codes,
                images={
                    "source": _data_uri(sample_dir / "source.png"),
                    "candidate": _data_uri(sample_dir / "refined.png"),
                    "difference": _data_uri(sample_dir / "difference.png"),
                },
                evidence_dir=workspace.relative(sample_dir),
            )
        )
    return entries


def _data_uri(path: Path) -> str:
    if not path.is_file():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _render(manifest: JobManifest, counters: JobCounters, entries: list[ReviewEntry]) -> str:
    rows = "\n".join(_card(entry) for entry in entries) or (
        '<p class="meta" style="padding:0 28px">No candidate needs review for this job.</p>'
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VidLiner review — {html.escape(manifest.job_id)}</title>
<style>{_CSS}</style>
</head>
<body>
<header>
  <h1>VidLiner review report</h1>
  <div class="meta">
    job <code>{html.escape(manifest.job_id)}</code> ·
    recipe <code>{html.escape(manifest.recipe_name)}</code> ·
    profile <code>{html.escape(manifest.runtime_profile_name or "default")}</code> ·
    state <code>{html.escape(manifest.state.value)}</code>
  </div>
</header>
<div class="counters">
  <div><b>{counters.samples}</b>samples</div>
  <div><b>{counters.targets}</b>targets</div>
  <div><b>{counters.generated}</b>candidates</div>
  <div><b>{counters.accepted}</b>accepted</div>
  <div><b>{counters.rejected}</b>rejected</div>
  <div><b>{counters.review}</b>review</div>
  <div><b>{counters.acceptance_rate:.0%}</b>acceptance</div>
</div>
<div class="grid">
{rows}
</div>
<footer>
  Decisions are recorded in <code>decisions.json</code> next to this file. Set
  <code>reviewer_decision</code> to <code>accepted</code> or <code>rejected</code> and re-export.
</footer>
</body>
</html>
"""


def _card(entry: ReviewEntry) -> str:
    metric_rows = "\n".join(_metric_row(metric) for metric in entry.metrics)
    reasons = (
        "".join(f'<span class="reason">{html.escape(code)}</span>' for code in entry.reason_codes)
        or '<span class="reason ok">no failure reason</span>'
    )
    score = f"{entry.overall_score:.3f}" if entry.overall_score is not None else "n/a"
    images = "".join(
        f'<img alt="{html.escape(name)}" src="{uri}">' for name, uri in entry.images.items() if uri
    )
    return f"""  <article class="card" data-candidate-id="{html.escape(entry.candidate_id)}" data-decision="{html.escape(entry.decision)}">
    <header>
      <h3>{html.escape(entry.category or "object")} <span class="decision {html.escape(entry.decision)}">{html.escape(entry.decision)}</span></h3>
      <div class="sub">sample {html.escape(entry.sample_id)} · candidate {html.escape(entry.candidate_key)} · overall {score}</div>
    </header>
    <div class="images">{images}</div>
    <div class="metrics"><table>{metric_rows}</table></div>
    <div class="reasons">{reasons}</div>
  </article>"""


def _metric_row(metric: dict[str, Any]) -> str:
    value = metric.get("value")
    threshold = metric.get("threshold")
    status = str(metric.get("status", "skipped"))
    css = {"passed": "pass", "failed": "fail", "warned": "warn"}.get(status, "")
    value_text = "—" if value is None else f"{float(value):.3f}"
    threshold_text = "—" if threshold is None else f"{float(threshold):.3f}"
    return (
        f"<tr><td>{html.escape(str(metric.get('metric', '')))}</td>"
        f'<td class="value {css}">{value_text} / {threshold_text}</td></tr>'
    )
