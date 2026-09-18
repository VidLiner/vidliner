"""Split safety: keeping an augmentation descendant in its source's split.

Dataset leakage is the failure mode where a model is evaluated on a near-copy of something it
trained on. Augmentation creates exactly that risk unless lineage is tracked, so the rule is applied
twice:

* **at planning time**, a recipe that names ``test`` or ``validation`` in ``augment_splits`` is
  refused unless it also explicitly acknowledges the risk, and never inside strict mode;
* **at export time**, every lineage edge is checked and any edge whose endpoints disagree about
  their split is a hard failure with ``SPLIT_LEAKAGE``.
"""

from __future__ import annotations

from dataclasses import dataclass

from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.domain.enums import SplitName
from vidliner.domain.recipe import Recipe

__all__ = ["SplitPlan", "assert_no_leakage", "plan_splits", "protected_splits"]

PROTECTED_SPLITS: tuple[str, ...] = (SplitName.VALIDATION.value, SplitName.TEST.value)


def protected_splits() -> tuple[str, ...]:
    """Splits that must not receive augmentation descendants by default."""
    return PROTECTED_SPLITS


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """The split policy a recipe implies."""

    augment_splits: tuple[str, ...]
    protected: tuple[str, ...]
    allow_test_augmentation: bool
    notes: tuple[str, ...]

    @property
    def touches_protected(self) -> bool:
        """Whether the policy permits augmenting a protected split."""
        return bool(set(self.augment_splits) & set(self.protected))


def plan_splits(recipe: Recipe) -> SplitPlan:
    """Derive the split policy from a recipe and validate it.

    Raises:
        ValidationFailure: when the recipe would augment a protected split without the explicit
            opt-in, or when it opts in but is running in strict mode.
    """
    requested = tuple(dict.fromkeys(recipe.dataset.augment_splits))
    protected = tuple(split for split in requested if split in PROTECTED_SPLITS)
    notes: list[str] = []
    if protected and not recipe.dataset.allow_test_augmentation:
        raise ValidationFailure(
            f"recipe requests augmentation of split(s) {sorted(protected)} without "
            "dataset.allow_test_augmentation: true",
            code=ErrorCode.SPLIT_NOT_AUGMENTABLE,
            detail={"splits": sorted(protected)},
        )
    if protected and recipe.replacement.mode == "strict":
        raise ValidationFailure(
            "strict replacement does not permit augmenting a protected split; leakage protection "
            "is not opt-out inside strict mode",
            code=ErrorCode.SPLIT_LEAKAGE,
            detail={"splits": sorted(protected)},
        )
    if protected:
        notes.append(
            f"augmenting protected split(s) {sorted(protected)} because the recipe explicitly allowed it"
        )
    if not protected:
        notes.append(f"augmentation is restricted to split(s) {list(requested)}")
    return SplitPlan(
        augment_splits=requested,
        protected=PROTECTED_SPLITS,
        allow_test_augmentation=recipe.dataset.allow_test_augmentation,
        notes=tuple(notes),
    )


def assert_no_leakage(edges: list[tuple[str, str, str, str]]) -> None:
    """Fail when a lineage edge crosses a split boundary.

    Args:
        edges: ``(child, parent, child_split, parent_split)`` tuples.

    Raises:
        ValidationFailure: listing every offending edge, so the whole problem is visible at once
            rather than one sample at a time.
    """
    offenders = [edge for edge in edges if edge[2] != edge[3]]
    if not offenders:
        return
    detail = [
        f"{child} ({child_split}) <- {parent} ({parent_split})"
        for child, parent, child_split, parent_split in offenders
    ]
    raise ValidationFailure(
        f"{len(offenders)} augmentation descendant(s) would cross a split boundary: " + "; ".join(detail[:5]),
        code=ErrorCode.SPLIT_LEAKAGE,
        detail={"edges": detail},
    )
