"""Cross-platform path helpers.

Windows accepts both ``\\`` and ``/`` as separators; POSIX accepts only ``/``.
Collection-relative paths are normalized here before they are joined to a
root, so the same saved selection works on either platform.

These helpers also reject absolute paths, drive-relative paths (``C:foo``),
UNC roots (``\\\\server\\share``) and ``..`` traversal before any filesystem
access. That matters on Windows, where ``Path("C:/photos") / "\\\\foo"`` is
``C:\\foo`` rather than a child of the collection.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")


def expand_path(value: str | os.PathLike[str]) -> Path:
    """Expand ``~`` and environment variables, then return a Path."""
    text = os.path.expandvars(os.path.expanduser(os.fspath(value)))
    return Path(text)


def rel_parts(value: str | os.PathLike[str]) -> tuple[str, ...]:
    """Split a collection-relative path into safe, normalized components.

    Both separator styles are accepted. Empty and ``.`` components are
    ignored, so ``a//b`` and ``a\\b`` both become ``("a", "b")``.
    """
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError(f"not a path: {value!r}")
    text = os.fspath(value).strip()
    if not text:
        raise ValueError("empty relative path")
    text = text.replace("\\", "/")
    if text.startswith("/") or text.startswith("//") or _DRIVE_PREFIX.match(text):
        raise ValueError(f"absolute path is not allowed: {value!r}")
    parts = tuple(part for part in text.split("/") if part not in ("", "."))
    if not parts:
        raise ValueError(f"empty relative path: {value!r}")
    if ".." in parts:
        raise ValueError(f"path escapes the root: {value!r}")
    if any(_DRIVE_PREFIX.match(part) for part in parts):
        raise ValueError(f"drive/stream syntax is not allowed: {value!r}")
    return parts


def safe_join(root: str | os.PathLike[str], rel: str | os.PathLike[str]) -> Path:
    """Join a relative path to a root, rejecting traversal and absolutes."""
    return Path(root).joinpath(*rel_parts(rel))


def is_within(child: str | os.PathLike[str], parent: str | os.PathLike[str]) -> bool:
    """True when *child* is *parent* or lives below it (resolved, case-aware)."""
    child_p = Path(child).resolve()
    parent_p = Path(parent).resolve()
    try:
        child_p.relative_to(parent_p)
        return True
    except ValueError:
        return False
