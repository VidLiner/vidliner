"""Identity construction for every derived object in VidLiner.

Two runs over the same bytes with the same recipe must produce the same identifiers, otherwise
cache reuse, resume, and provenance comparison are all impossible. Every function here is a pure
function of its documented inputs. None of them consult the clock, the filesystem, or a random
source.
"""

from __future__ import annotations

from typing import Final

from vidliner.core.canonical import object_digest
from vidliner.domain.enums import ArtifactKind
from vidliner.domain.shapes import BoundingBox

__all__ = [
    "ARTIFACT_DIGEST_LENGTH",
    "JOB_ID_LENGTH",
    "artifact_digest_key",
    "candidate_id",
    "job_id",
    "lineage_key_candidate",
    "lineage_key_object",
    "lineage_key_sample",
    "node_id",
    "object_id",
    "sample_id",
]

JOB_ID_LENGTH: Final = 16
ARTIFACT_DIGEST_LENGTH: Final = 64
_BOX_QUANTUM: Final = 2.0
_SCORE_QUANTUM: Final = 0.05


def _quantise(value: float, quantum: float) -> float:
    return round(value / quantum) * quantum


def job_id(recipe_hash: str, seed: int, created_marker: str) -> str:
    """Identify one job attempt.

    ``created_marker`` is a caller-supplied, coarse (minute-resolution or explicit) string. Passing
    it explicitly keeps this function pure while still allowing two intentional runs of the same
    recipe to be distinct.
    """
    return object_digest("j_", {"recipe": recipe_hash, "seed": seed, "marker": created_marker}, JOB_ID_LENGTH)


def sample_id(root_digest: str, relative_path: str) -> str:
    """Identify a dataset sample by its content root and its path within the dataset.

    The path is normalised to POSIX separators before hashing, so a dataset moved between operating
    systems keeps the same sample identities and therefore the same cache entries.
    """
    normalised = relative_path.replace("\\", "/").lstrip("./")
    return object_digest("s_", {"root": root_digest, "path": normalised}, 16)


def object_id(
    asset_digest: str,
    frame_index: int,
    class_name: str,
    bbox: BoundingBox,
    score: float,
) -> str:
    """Identify an object instance inside one frame.

    The box and score are quantised before hashing so that a one-pixel or one-percent jitter from
    a detector does not create a brand-new identity (and therefore a brand-new cache lineage) for
    what is plainly the same object. The quantisation is deliberate and documented rather than
    accidental: it is what makes re-running detection cheap and reproducible.
    """
    payload = {
        "asset": asset_digest,
        "frame": frame_index,
        "class": class_name,
        "box": [
            _quantise(bbox.x_min, _BOX_QUANTUM),
            _quantise(bbox.y_min, _BOX_QUANTUM),
            _quantise(bbox.x_max, _BOX_QUANTUM),
            _quantise(bbox.y_max, _BOX_QUANTUM),
        ],
        "score": _quantise(score, _SCORE_QUANTUM),
    }
    return object_digest("o_", payload, 12)


def candidate_id(sample: str, target_object: str, candidate_key: str) -> str:
    """Identify one replacement candidate for one target object."""
    return object_digest("c_", {"sample": sample, "object": target_object, "key": candidate_key}, 16)


def node_id(operator: str, lineage: str, ordinal: int = 0) -> str:
    """Identify one graph node.

    The identity is ``(operator, lineage path, ordinal)``. ``lineage`` already contains the sample,
    object, and candidate keys, so unrelated edits elsewhere in the dataset cannot renumber it.
    """
    if ordinal < 0:
        raise ValueError("node ordinal must not be negative")
    return object_digest("n_", {"operator": operator, "lineage": lineage, "ordinal": ordinal}, 16)


def lineage_key_sample(sample: str) -> str:
    """Lineage key for a sample-level node."""
    return f"sample:{sample}"


def lineage_key_object(sample: str, target_object: str) -> str:
    """Lineage key for a target-object-level node."""
    return f"{lineage_key_sample(sample)}/object:{target_object}"


def lineage_key_candidate(sample: str, target_object: str, candidate_key: str) -> str:
    """Lineage key for a candidate-level node."""
    return f"{lineage_key_object(sample, target_object)}/candidate:{candidate_key}"


def artifact_digest_key(kind: ArtifactKind, digest: str) -> str:
    """Catalogue key for an artifact, namespaced by kind so equal bytes with different roles differ."""
    return f"{kind.value}:{digest}"
