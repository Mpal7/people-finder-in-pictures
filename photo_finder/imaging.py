"""Image loading and duplicate-map helpers.

Same preprocessing as measured in Phase 0 (recon/sample_timing.py):
EXIF orientation fix -> RGB -> downscale max side (default 1600).
CR2 uses the embedded full-size JPEG preview (no demosaicing).
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

IMAGE_EXT = {
    ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".heic", ".heif", ".avif", ".webp",
    ".gif", ".bmp", ".tif", ".tiff",
    ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".dng",
    ".orf", ".rw2", ".raf", ".pef", ".srw",
}


def load_folder_rank(path: str | Path | None) -> dict[str, int]:
    """Read an optional top-level-folder preference map used when choosing the
    canonical copy of an exact-duplicate group.

    The file is a JSON object mapping folder names to integer ranks, e.g.
    ``{"camera": 80, "phone_backup": 70, "exports": 10}``. Higher wins; folders
    not listed rank 0. Only relevant when ``index_faces.py --dedup`` is used.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in data.items()}


def canonical_rank(relpath: str, folder_rank: dict[str, int] | None = None) -> int:
    top = relpath.split(os.sep, 1)[0] if os.sep in relpath else "(root)"
    return (folder_rank or {}).get(top, 0)


def top_of(relpath: str) -> str:
    return relpath.split(os.sep, 1)[0] if os.sep in relpath else "(root)"


def load_dup_map(csv_path: Path,
                 folder_rank: dict[str, int] | None = None) -> dict[str, tuple[int, bool]]:
    """Read recon duplicates.csv -> {relpath: (group_id, is_canonical)}."""
    groups: dict[int, list[str]] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            groups.setdefault(int(row["group_id"]), []).append(row["relpath"])
    out: dict[str, tuple[int, bool]] = {}
    for gid, members in groups.items():
        members_sorted = sorted(members, key=lambda p: (-canonical_rank(p, folder_rank), p))
        for i, rel in enumerate(members):
            out[rel] = (gid, i == 0)
    return out


def iter_images(root: Path, dup_map: dict[str, tuple[int, bool]] | None = None,
                include_non_canonical: bool = False):
    """Yield relpaths of indexable images (canonical copy per exact-dup group)."""
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d.lower() not in {
            "$recycle.bin", "system volume information", ".trash-1000",
            ".trash", "found.000",
        })
        for name in sorted(filenames):
            ext = os.path.splitext(name)[1].lower()
            if ext not in IMAGE_EXT:
                continue
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            if dup_map is not None and not include_non_canonical:
                info = dup_map.get(rel)
                if info is not None and not info[1]:
                    continue  # exact duplicate: index only the canonical copy
            yield rel


def load_image(path: Path, max_side: int = 1600) -> np.ndarray:
    """Return RGB uint8 array, EXIF-corrected, max side limited. Read-only."""
    from PIL import Image, ImageOps
    import pillow_heif

    pillow_heif.register_heif_opener()
    ext = path.suffix.lower()
    if ext in (".cr2", ".crw"):
        import io
        import rawpy
        with rawpy.imread(str(path)) as raw:
            thumb = raw.extract_thumb()
        if thumb.format == rawpy.ThumbFormat.JPEG:
            img = Image.open(io.BytesIO(thumb.data))
        else:
            img = Image.fromarray(thumb.data)
    else:
        img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    m = max(h, w)
    if m > max_side:
        import cv2
        scale = max_side / m
        arr = cv2.resize(arr, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(arr)


def load_thumb(path: Path, px: int = 320, width_first: bool = False):
    """Fast PIL thumbnail: JPEGs use draft() (decode at reduced scale), CR2 via
    the embedded preview. Returns a PIL RGB image.

    width_first=False: max side <= px (grid cards)
    width_first=True : width normalized to px, height follows (status-bar
                       analysis - keeps tiny top-band glyphs legible on tall
                       screenshots)
    """
    from PIL import Image, ImageOps

    ext = path.suffix.lower()
    if ext in (".cr2", ".crw"):
        img = Image.fromarray(load_image(path, max_side=max(px * 3, 900)))
    else:
        import pillow_heif
        pillow_heif.register_heif_opener()
        img = Image.open(path)
        try:
            img.draft("RGB", (px * 2, px * 4) if width_first else (px * 2, px * 2))
        except Exception:
            pass
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    if width_first:
        if img.width > px:
            scale = px / img.width
            img = img.resize((px, max(1, round(img.height * scale))), Image.LANCZOS)
    else:
        img.thumbnail((px, px), Image.LANCZOS)
    return img
