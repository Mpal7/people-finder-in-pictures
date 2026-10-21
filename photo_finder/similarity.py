"""Perceptual hashing and near-duplicate grouping for review pages.

Exact duplicates are handled at index time by ``recon/build_dupmap.py``. This
module catches the cases that survive that filter: WhatsApp re-encodes,
HEIC -> JPG conversions and other recompressions that produce two visibly
identical review cards from different files.

The hash is a 64-bit difference hash (dHash): the image is reduced to 9x8
grayscale samples and each bit records whether the right neighbour is brighter.
It is invariant to scaling and JPEG recompression, cheap to compute, and needs
no extra dependency beyond Pillow + numpy.
"""
from __future__ import annotations

import numpy as np


def dhash(rgb: np.ndarray, hash_size: int = 8) -> int:
    """Return a ``hash_size**2``-bit difference hash for an RGB uint8 array."""
    from PIL import Image

    img = Image.fromarray(rgb).convert("L").resize(
        (hash_size + 1, hash_size), Image.LANCZOS
    )
    a = np.asarray(img, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return value


def hamming(a: int, b: int) -> int:
    return (int(a) ^ int(b)).bit_count()


def near_duplicate_groups(hashes: list[int], aspects: list[float] | None = None,
                          threshold: int = 6, aspect_tol: float = 0.04
                          ) -> tuple[list[int], list[int], list[int]]:
    """Cluster items whose dHash differs by at most ``threshold`` bits.

    ``hashes`` are in review order (best score first), so the first member of a
    group is its representative. Returns ``(group_id, rank_in_group,
    group_size)`` lists aligned with the input.

    ``aspects`` (width / height) is optional; when given, items whose aspect
    ratio differs by more than ``aspect_tol`` (relative) are not grouped even if
    their hashes match. This keeps a square crop and a wide version of the same
    scene apart. ``threshold <= 0`` disables grouping entirely.
    """
    n = len(hashes)
    if n == 0:
        return [], [], []
    if threshold <= 0:
        return list(range(n)), [0] * n, [1] * n

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            # Keep the lowest index as root so representatives are deterministic.
            if ri < rj:
                parent[rj] = ri
            else:
                parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if aspects is not None:
                ai, aj = aspects[i], aspects[j]
                if max(ai, aj) > 0 and abs(ai - aj) / max(ai, aj) > aspect_tol:
                    continue
            if hamming(hashes[i], hashes[j]) <= threshold:
                union(i, j)

    root_to_gid: dict[int, int] = {}
    group_of: list[int] = []
    for i in range(n):
        root = find(i)
        if root not in root_to_gid:
            root_to_gid[root] = len(root_to_gid)
        group_of.append(root_to_gid[root])

    sizes = [0] * len(root_to_gid)
    for gid in group_of:
        sizes[gid] += 1
    ranks = [0] * n
    seen = [0] * len(root_to_gid)
    for i, gid in enumerate(group_of):
        ranks[i] = seen[gid]
        seen[gid] += 1
    return group_of, ranks, [sizes[g] for g in group_of]
