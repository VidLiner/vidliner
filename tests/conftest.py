"""Shared pytest fixtures.

Everything a test needs is built here from synthetic data, so no test downloads an image, calls a
paid API, or depends on a model being installed. The fixtures also demonstrate the supported usage
pattern for embedding VidLiner: open a session, load a recipe, run it, export.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from vidliner.fixtures import SyntheticDataset, build_dataset
from vidliner.pipeline.service import Session
from vidliner.runtime.profile import default_profile

__all__ = ["make_recipe"]


@pytest.fixture(name="workspace")
def workspace_fixture(tmp_path: Path) -> Path:
    """An initialised workspace directory with the default local runtime profile."""
    from vidliner.storage.state import StateStore
    from vidliner.storage.workspace import Workspace

    workspace = Workspace(tmp_path / "workspace").initialize()
    StateStore(workspace.state_path).initialize().close()
    (workspace.root / "runtime.yaml").write_text(_profile_yaml(), encoding="utf-8")
    return workspace.root


@pytest.fixture(name="session")
def session_fixture(workspace: Path) -> Iterator[Session]:
    """An open session over the fixture workspace."""
    session = Session.open(workspace)
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(name="dataset_root")
def dataset_root_fixture(tmp_path: Path, workspace: Path) -> Path:
    """Ten synthetic images with a matching COCO annotation file, inside the workspace."""
    root = workspace / "data" / "source"
    build_dataset(root, count=10, annotations=True)
    return root


@pytest.fixture(name="synthetic_dataset")
def synthetic_dataset_fixture(workspace: Path) -> SyntheticDataset:
    """The synthetic dataset object, for tests that need to inspect what was drawn."""
    return build_dataset(workspace / "data" / "source", count=10, annotations=True)


def _profile_yaml() -> str:
    """The default local profile, serialized for a fixture workspace."""
    import yaml

    return yaml.safe_dump(default_profile().model_dump(mode="json", exclude={"source_path"}), sort_keys=False)


def make_recipe(
    dataset_input: str,
    *,
    recipe_name: str = "test-recipe",
    split_manifest: str | None = None,
    classes: tuple[str, ...] = ("car",),
    values: tuple[str, ...] = ("sedan", "suv"),
    candidates: int = 2,
    output: str = "./data/output",
    fmt: str = "coco-instance",
    source_format: str | None = "coco-instance",
    minimum_overall: float = 0.10,
    hard_gates: dict[str, float] | None = None,
    split_mode: str = "none",
    augment_splits: tuple[str, ...] = ("train",),
    seed: int | None = 4242,
    max_per_sample: int = 1,
) -> dict[str, Any]:
    """Build a recipe mapping that is valid for the synthetic fixture dataset.

    The default gate values are deliberately permissive: a test that wants to exercise *rejection*
    tightens them explicitly, while general pipeline tests are not gated on a heuristic quality score
    they are not testing.
    """
    return {
        "recipe": recipe_name,
        "dataset": {
            "input": dataset_input,
            "format": source_format,
            "splits": {
                "mode": split_mode,
                "names": ["train", "val", "test"],
                **({"manifest": split_manifest} if split_manifest else {}),
            },
            "augment_splits": list(augment_splits),
        },
        "target": {
            "classes": list(classes),
            "min_score": 0.10,
            "min_area_px": 64,
            "max_per_sample": max_per_sample,
            "top_k_by": "score",
            "strategy": "all",
        },
        "replacement": {
            "mode": "strict",
            "strategy": "category",
            "values": list(values),
            "candidates_per_object": candidates,
            "seed": seed,
        },
        "quality": {
            "minimum_overall": minimum_overall,
            "hard_gates": hard_gates if hard_gates is not None else {},
            "maximum_artifact_score": 1.0,
            "minimum_target_presence": 0.10,
            "review_band": 0.0,
        },
        # The fixtures bind the built-in demonstration backends, so a job in a test acknowledges
        # that deliberately — exactly as the example recipe and an operator running a demo must. The
        # production guard has its own tests with production backends.
        "acceptance": {"allow_demo_backends": True},
        "export": {"format": fmt, "path": output, "copy_images": True},
        "limits": {"workers": 2, "resume": True, "cache": True, "per_node_timeout_s": 60},
    }
