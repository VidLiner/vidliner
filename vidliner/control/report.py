"""Dataset distribution reporting.

The point of augmentation is not to make a dataset bigger; it is to change its *distribution*
deliberately. That requires measuring the distribution before and after: class counts, object size
buckets, aspect ratio buckets, acceptance rate per replacement category, and the reason-code
histogram of everything that was rejected.

The report is written as JSON next to the exported dataset so a training run can point at it, and it
is summarised in the console by ``vidliner run``.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vidliner.domain.annotations import AnnotationBundle
from vidliner.domain.recipe import Recipe
from vidliner.storage.state import StateStore
from vidliner.storage.workspace import Workspace

__all__ = ["DistributionReport", "build_dataset_report", "build_distribution"]


@dataclass(frozen=True, slots=True)
class DistributionReport:
    """Everything the report contains, in a form that can also be printed."""

    job_id: str
    recipe: str
    counts: dict[str, int]
    class_distribution: dict[str, int]
    size_distribution: dict[str, int]
    aspect_distribution: dict[str, int]
    replacement_distribution: dict[str, int]
    acceptance_by_category: dict[str, dict[str, float]]
    reason_histogram: dict[str, int]
    quality_averages: dict[str, float]
    duplicate_count: int
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "format": "vidliner.dataset-report@1",
            "job_id": self.job_id,
            "recipe": self.recipe,
            "counts": self.counts,
            "class_distribution": self.class_distribution,
            "object_size_distribution": self.size_distribution,
            "aspect_ratio_distribution": self.aspect_distribution,
            "replacement_distribution": self.replacement_distribution,
            "acceptance_by_category": self.acceptance_by_category,
            "reason_code_histogram": self.reason_histogram,
            "quality_averages": self.quality_averages,
            "duplicate_count": self.duplicate_count,
            "notes": list(self.notes),
        }

    def summary_lines(self) -> list[str]:
        """Short human-readable summary for the console."""
        lines = [
            f"accepted {self.counts.get('accepted', 0)}  rejected {self.counts.get('rejected', 0)}  "
            f"review {self.counts.get('review', 0)}  duplicates {self.duplicate_count}",
        ]
        if self.class_distribution:
            lines.append(
                "classes: "
                + ", ".join(f"{name}={count}" for name, count in sorted(self.class_distribution.items()))
            )
        if self.reason_histogram:
            top = sorted(self.reason_histogram.items(), key=lambda item: -item[1])[:5]
            lines.append("top reasons: " + ", ".join(f"{code}={count}" for code, count in top))
        if self.quality_averages:
            lines.append(
                "quality: "
                + ", ".join(
                    f"{metric}={value:.3f}" for metric, value in sorted(self.quality_averages.items())
                )
            )
        return lines


def build_dataset_report(
    *,
    workspace: Workspace,
    state: StateStore,
    job_id: str,
    recipe: Recipe,
    accepted: list[Any],
    rejected: list[Any],
    review: list[Any],
    duplicates: dict[str, tuple[str, str, float]],
    target: Path,
) -> Path:
    """Build and write the dataset distribution report."""
    report = build_distribution(
        workspace=workspace,
        state=state,
        job_id=job_id,
        recipe=recipe,
        accepted=accepted,
        rejected=rejected,
        review=review,
        duplicates=duplicates,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    state.record_report(job_id, "dataset-report", target.as_posix())
    return target


def build_distribution(
    *,
    workspace: Workspace,
    state: StateStore,
    job_id: str,
    recipe: Recipe,
    accepted: list[Any],
    rejected: list[Any],
    review: list[Any],
    duplicates: dict[str, tuple[str, str, float]],
) -> DistributionReport:
    """Measure the augmented distribution and the acceptance behaviour."""
    del workspace
    class_counter: Counter[str] = Counter()
    size_counter: Counter[str] = Counter()
    aspect_counter: Counter[str] = Counter()
    replacement_counter: Counter[str] = Counter()
    reason_counter: Counter[str] = Counter()
    per_category: dict[str, Counter[str]] = {}

    for record in accepted:
        annotation: AnnotationBundle = record.annotation
        for obj in annotation.objects:
            name = annotation.category_name(obj.category_id) or f"category_{obj.category_id}"
            class_counter[name] += 1
            size_counter[_size_bucket(obj.area)] += 1
            aspect_counter[_aspect_bucket(obj.bbox.aspect_ratio)] += 1
        replacement_counter[str(record.category or "unknown")] += 1
        stats = per_category.setdefault(str(record.category or "unknown"), Counter())
        stats["accepted"] += 1

    for record in rejected:
        for code in getattr(record, "reason_codes", ()) or ():
            reason_counter[str(code)] += 1
        stats = per_category.setdefault(str(getattr(record, "category", "unknown") or "unknown"), Counter())
        stats["rejected"] += 1

    for record in review:
        stats = per_category.setdefault(str(getattr(record, "category", "unknown") or "unknown"), Counter())
        stats["review"] += 1

    acceptance = {
        category: {
            "accepted": float(stats.get("accepted", 0)),
            "rejected": float(stats.get("rejected", 0)),
            "review": float(stats.get("review", 0)),
            "acceptance_rate": round(
                stats.get("accepted", 0)
                / max(1, stats.get("accepted", 0) + stats.get("rejected", 0) + stats.get("review", 0)),
                4,
            ),
        }
        for category, stats in sorted(per_category.items())
    }
    notes: list[str] = []
    if not accepted:
        notes.append("no candidate was accepted; inspect the reason-code histogram before retuning")
    protected = set(recipe.dataset.augment_splits) & {"test", "validation"}
    if protected:
        notes.append(f"protected split(s) {sorted(protected)} were augmented by explicit recipe opt-in")
    return DistributionReport(
        job_id=job_id,
        recipe=recipe.name,
        counts={
            "accepted": len(accepted),
            "rejected": len(rejected),
            "review": len(review),
            "duplicates": len(duplicates),
            "objects": sum(class_counter.values()),
        },
        class_distribution=dict(sorted(class_counter.items())),
        size_distribution=dict(sorted(size_counter.items())),
        aspect_distribution=dict(sorted(aspect_counter.items())),
        replacement_distribution=dict(sorted(replacement_counter.items())),
        acceptance_by_category=acceptance,
        reason_histogram=dict(sorted(reason_counter.items(), key=lambda item: -item[1])),
        quality_averages=state.metric_averages(job_id),
        duplicate_count=len(duplicates),
        notes=tuple(notes),
    )


def _size_bucket(area: float) -> str:
    if area < 32 * 32:
        return "tiny(<32^2)"
    if area < 96 * 96:
        return "small(<96^2)"
    if area < 224 * 224:
        return "medium(<224^2)"
    return "large(>=224^2)"


def _aspect_bucket(ratio: float) -> str:
    if ratio <= 0:
        return "degenerate"
    if ratio < 0.5:
        return "tall(<0.5)"
    if ratio < 0.9:
        return "portrait(0.5-0.9)"
    if ratio <= 1.1:
        return "square(0.9-1.1)"
    if ratio <= 2.0:
        return "landscape(1.1-2.0)"
    return "wide(>2.0)"
