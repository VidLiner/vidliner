"""Perceptual fingerprints for duplicate control.

Three classic hashes are implemented with nothing but numpy so that duplicate detection works in
every deployment, including one with no imaging library beyond Pillow for decoding:

* ``ahash`` — average hash: robust, cheap, weak on gradients;
* ``dhash`` — difference hash: better on smooth gradients;
* ``phash`` — a 32-point DCT low-frequency hash, the default because it survives mild
  recompression and scaling, which is exactly what an augmentation pipeline produces.

A fingerprint is a 64-bit integer rendered as 16 hex characters, so it sorts, stores in SQLite, and
diffs cheaply.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "Fingerprint",
    "hamming_distance",
    "hash_array",
    "perceptual_hash",
    "similarity_from_distance",
]

HASH_BITS = 64


class Fingerprint:
    """A perceptual hash plus the algorithm that produced it."""

    __slots__ = ("algorithm", "value")

    def __init__(self, value: int, algorithm: str = "phash") -> None:
        self.value = value & ((1 << HASH_BITS) - 1)
        self.algorithm = algorithm

    @property
    def hex(self) -> str:
        """16-character hexadecimal form, suitable for storage."""
        return f"{self.value:016x}"

    @property
    def bits(self) -> np.ndarray:
        """The 64 bits as a boolean array, most significant first."""
        return np.array(
            [(self.value >> (HASH_BITS - 1 - index)) & 1 for index in range(HASH_BITS)], dtype=bool
        )

    def distance(self, other: Fingerprint) -> int:
        """Hamming distance to another fingerprint."""
        return hamming_distance(self.value, other.value)

    def similarity(self, other: Fingerprint) -> float:
        """Similarity in ``[0, 1]``: 1.0 for identical fingerprints."""
        return similarity_from_distance(self.distance(other))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Fingerprint) and other.value == self.value and other.algorithm == self.algorithm
        )

    def __hash__(self) -> int:
        return hash((self.value, self.algorithm))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Fingerprint({self.algorithm}:{self.hex})"

    @classmethod
    def parse(cls, text: str, algorithm: str = "phash") -> Fingerprint:
        """Parse a stored hexadecimal fingerprint."""
        return cls(int(text, 16), algorithm)


def hamming_distance(left: int, right: int) -> int:
    """Number of differing bits between two 64-bit fingerprints."""
    return int((left ^ right).bit_count())


def similarity_from_distance(distance: int) -> float:
    """Map a Hamming distance onto ``[0, 1]``."""
    return float(np.clip(1.0 - distance / HASH_BITS, 0.0, 1.0))


def perceptual_hash(image: np.ndarray, *, algorithm: str = "phash") -> Fingerprint:
    """Compute a 64-bit perceptual fingerprint of a greyscale or colour image.

    Args:
        image: ``float32``/``uint8`` array in ``(H, W)`` or ``(H, W, C)`` layout.
        algorithm: ``phash``, ``ahash``, ``dhash``, or ``none``.

    Raises:
        ValueError: when the algorithm is unknown.
    """
    if algorithm == "none":
        return Fingerprint(0, "none")
    grey = _to_grey(image)
    if algorithm == "ahash":
        return Fingerprint(_ahash(grey), "ahash")
    if algorithm == "dhash":
        return Fingerprint(_dhash(grey), "dhash")
    if algorithm == "phash":
        return Fingerprint(_phash(grey), "phash")
    raise ValueError(f"unknown perceptual hash algorithm {algorithm!r}")


def hash_array(image: np.ndarray, *, algorithm: str = "phash") -> str:
    """Hexadecimal fingerprint of an image array."""
    return perceptual_hash(image, algorithm=algorithm).hex


def _to_grey(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 3:
        array = array[..., :3].mean(axis=2)
    if float(np.max(array)) > 1.5:
        array = array / 255.0
    return array


def _resize_nearest(grey: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize to ``size x size``; adequate for perceptual hashing."""
    height, width = grey.shape
    if height == 0 or width == 0:
        raise ValueError("cannot hash an empty image")
    rows = np.clip((np.arange(size) + 0.5) * height / size, 0, height - 1).astype(int)
    columns = np.clip((np.arange(size) + 0.5) * width / size, 0, width - 1).astype(int)
    return grey[rows][:, columns]


def _ahash(grey: np.ndarray) -> int:
    small = _resize_nearest(grey, 8)
    threshold = small.mean()
    return _bits_to_int(small > threshold)


def _dhash(grey: np.ndarray) -> int:
    small = _resize_nearest(grey, 9)
    differences = small[:, 1:] > small[:, :-1]
    return _bits_to_int(differences.reshape(-1)[:HASH_BITS])


def _phash(grey: np.ndarray) -> int:
    small = _resize_nearest(grey, 32)
    dct = _dct2(small)
    low = dct[:8, :8].copy()
    # The DC term carries overall brightness, which an augmentation may legitimately change.
    low[0, 0] = 0.0
    threshold = low.mean()
    return _bits_to_int(low.reshape(-1)[:HASH_BITS] > threshold)


def _dct2(matrix: np.ndarray) -> np.ndarray:
    """Separable 2-D DCT-II on a square matrix, then scaled to the low-frequency corner."""
    size = matrix.shape[0]
    index = np.arange(size)
    basis = np.cos(np.pi * (2 * index[:, None] + 1) * index[None, :] / (2 * size))
    basis[:, 0] *= 1.0 / np.sqrt(2.0)
    scaled = basis * np.sqrt(2.0 / size) / np.sqrt(2.0)
    return scaled.T @ matrix @ scaled


def _bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits.reshape(-1)[:HASH_BITS]:
        value = (value << 1) | int(bool(bit))
    return value
