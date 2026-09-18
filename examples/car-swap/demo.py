#!/usr/bin/env python3
"""Run the car-swap example end to end, with no model download and no paid API.

    python examples/car-swap/demo.py [output-directory]

The demo:

1. creates a workspace;
2. generates a small annotated dataset (a synthetic road scene per image);
3. runs `plan`, `run`, `qa`, and `export` through the library API — the same functions the CLI calls;
4. prints the dataset, the report, and one provenance record.

The point is to show the whole flow, including the parts that reject data.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

# Allow running this file directly from a checkout that has not been installed. When the package is
# installed (the documented `pip install -e .`) this is a no-op, because the path is already
# importable and the edit below changes nothing.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vidliner.fixtures import SceneObject, render_scene
from vidliner.pipeline.runner import JobOptions
from vidliner.pipeline.service import JobRequest, Session, load_recipe
from vidliner.runtime.profile import default_profile
from vidliner.storage.state import StateStore
from vidliner.storage.workspace import Workspace


def main() -> int:
    """Run the example and print what happened."""
    destination = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="vidliner-example-"))
    )
    workspace_root = destination / "workspace"
    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    workspace = Workspace(workspace_root).initialize()
    StateStore(workspace.state_path).initialize().close()

    import yaml

    (workspace.root / "runtime.yaml").write_text(
        yaml.safe_dump(default_profile().model_dump(mode="json", exclude={"source_path"}), sort_keys=False),
        encoding="utf-8",
    )

    print(f"workspace   {workspace.root}")

    # 1. A dataset that looks like a small annotated road-scene set.
    source = workspace.root / "data" / "source" / "train"
    _write_scene_dataset(source, count=10)
    _write_coco(workspace.root / "data" / "source", source)
    print(f"dataset     10 images in {source.relative_to(workspace.root)}")

    # 2. The recipe ships with the example.
    recipe_path = Path(__file__).with_name("car-swap.yaml")
    recipe = load_recipe(recipe_path)
    print(f"recipe      {recipe.name} ({recipe.recipe_hash()[:12]})")

    session = Session.open(workspace.root)
    try:
        # 3. Plan: resolve capabilities and cost the job without running anything.
        plan, _discovery = session.plan(recipe)
        estimate = plan.job_plan.estimate
        print(
            f"plan        {plan.job_plan.node_count} nodes, {estimate.candidates} candidates, "
            f"{estimate.estimated_seconds:.2f}s estimated"
        )
        for capability, backend in sorted(plan.job_plan.capability_bindings.items()):
            print(f"            {capability:44s} -> {backend}")

        # 4. Run: generate, verify, re-annotate, gate, export.
        outcome = asyncio.run(session.run(JobRequest(recipe=recipe, options=JobOptions(workers=4))))
        counters = outcome.manifest.counters
        print()
        print(f"run         {outcome.manifest.state.value}")
        print(
            f"            samples {counters.samples}  targets {counters.targets}  "
            f"generated {counters.generated}  accepted {counters.accepted}  "
            f"rejected {counters.rejected}  review {counters.review}"
        )
        print(f"            acceptance {counters.acceptance_rate:.0%}   failures {counters.failed}")
        print(f"            evidence {Path(outcome.manifest.run_dir).relative_to(workspace.root)}")

        # 5. Explain the rejections.
        rescored = session.re_evaluate(outcome.manifest.job_id, recipe)
        print(f"qa          evaluated {rescored['evaluated']}, changed {rescored['changed']}")
        for candidate in session.state.candidates(outcome.manifest.job_id)[:6]:
            reasons = ", ".join(candidate.reason_codes) or "-"
            score = f"{candidate.overall_score:.3f}" if candidate.overall_score is not None else "n/a"
            print(f"            {candidate.candidate_id}  {candidate.state.value:<8s} {score}  {reasons}")

        # 6. The dataset.
        if outcome.export is not None:
            root = outcome.export.path
            print()
            print(f"dataset     {root.relative_to(workspace.root)} ({outcome.export.format})")
            print(f"            images      {len(list((root / 'images').glob('*')))}")
            print(f"            provenance  {len(list((root / 'provenance').glob('*.json')))}")
            report = json.loads((root / "dataset-report.json").read_text(encoding="utf-8"))
            print(f"            classes     {report['class_distribution']}")
            print(f"            acceptance  {report['acceptance_by_category']}")
            print(
                f"            reasons     {report['reason_codes'] if 'reason_codes' in report else report['reason_code_histogram']}"
            )
            records = sorted((root / "provenance").glob("*.json"))
            if records:
                record = json.loads(records[0].read_text(encoding="utf-8"))
                print()
                print("one provenance record:")
                for key in (
                    "source_sample_id",
                    "source_digest",
                    "recipe_hash",
                    "job_id",
                    "target_object_id",
                    "replacement_category",
                    "decision",
                    "output_digest",
                    "overall_score",
                    "seeds",
                ):
                    value = record.get(key)
                    if isinstance(value, dict):
                        value = {name: seed for name, seed in value.items() if name != "node_seeds"}
                    print(f"  {key:22s} {value}")
    finally:
        session.close()

    print()
    print(f"done. Explore {workspace.root}")
    return 0


#: Every scene draws one object inside these bounds, so the annotation always matches the drawing.
BOX_WIDTH = 56
BOX_HEIGHT = 40


def _box_origin(index: int) -> tuple[int, int]:
    """Deterministic object position that stays inside the 200x150 frame."""
    return 30 + (index * 5) % 30, 25 + (index * 7) % 25


def _write_scene_dataset(target: Path, *, count: int) -> None:
    """Render synthetic road scenes with a car, deterministic per index."""
    from PIL import Image

    target.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        x0, y0 = _box_origin(index)
        car = SceneObject(
            class_name="car",
            bbox=(x0, y0, x0 + BOX_WIDTH, y0 + BOX_HEIGHT),
            # Contrasting colours so the saliency detector finds the object in every scene.
            colour=(210 - (index * 11) % 60, 30 + (index * 17) % 50, 40 + (index * 13) % 60),
            label=0,
        )
        image = render_scene(width=200, height=150, objects=(car,), seed=index)
        Image.fromarray(image, mode="RGB").save(target / f"img_{index:03d}.png")


def _write_coco(root: Path, image_dir: Path) -> None:
    """Write a COCO document describing the generated scenes."""
    from PIL import Image

    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    annotation_id = 1
    for image_id, path in enumerate(sorted(image_dir.glob("*.png")), start=1):
        with Image.open(path) as handle:
            width, height = handle.size
        relative = path.relative_to(root).as_posix()
        images.append({"id": image_id, "file_name": relative, "width": width, "height": height})
        # The demo draws one car per scene, at the same place the annotator records here.
        index = image_id - 1
        x0, y0 = _box_origin(index)
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x0, y0, BOX_WIDTH, BOX_HEIGHT],
                "area": float(BOX_WIDTH * BOX_HEIGHT),
                "iscrowd": 0,
                "segmentation": [
                    [x0, y0, x0 + BOX_WIDTH, y0, x0 + BOX_WIDTH, y0 + BOX_HEIGHT, x0, y0 + BOX_HEIGHT]
                ],
            }
        )
        annotation_id += 1
    document = {
        "info": {"description": "vidliner example dataset"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "car"}],
    }
    target = root / "annotations" / "instances_train.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
