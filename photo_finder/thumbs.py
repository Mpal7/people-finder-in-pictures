"""Bounded on-disk thumbnail cache for the web UI.

The cache key includes the source relpath, size, mtime and requested pixel
size, so edits to a photo naturally produce a new entry. Old entries are
evicted oldest-first when the cache exceeds a size budget or an age limit.

Knobs (environment variables, read once at startup):
  PHOTO_THUMB_CACHE_MB    total cache budget in MB (default 512)
  PHOTO_THUMB_CACHE_DAYS  delete entries older than N days (default 30)
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path

from .imaging import load_thumb

DEFAULT_MAX_BYTES = int(float(os.environ.get("PHOTO_THUMB_CACHE_MB", "512")) * 2 ** 20)
DEFAULT_MAX_AGE_S = int(float(os.environ.get("PHOTO_THUMB_CACHE_DAYS", "30")) * 86400)


class ThumbCache:
    def __init__(self, directory: Path, max_bytes: int | None = None,
                 max_age_s: int | None = None):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = DEFAULT_MAX_BYTES if max_bytes is None else max_bytes
        self.max_age_s = DEFAULT_MAX_AGE_S if max_age_s is None else max_age_s
        self._lock = threading.Lock()
        self._bytes = 0
        self._files = 0
        self._writes = 0
        self._rescan_locked()
        with self._lock:
            self._evict_locked()

    # ------------------------------------------------------------- public --
    def path_for(self, source: Path, rel: Path | str, px: int) -> Path:
        """Return the cached thumbnail path, creating it if needed."""
        st = source.stat()
        key = hashlib.sha1(
            f"{rel}|{st.st_mtime_ns}|{st.st_size}|{px}".encode()
        ).hexdigest()
        out = self.dir / f"{key}.jpg"
        with self._lock:
            if not out.exists():
                tmp = out.with_name(f"{out.name}.tmp")
                try:
                    img = load_thumb(source, px)
                    img.save(tmp, format="JPEG", quality=80, optimize=True)
                    os.replace(tmp, out)
                except Exception:
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
                    raise
                try:
                    self._bytes += out.stat().st_size
                except OSError:
                    pass
                self._files += 1
                self._writes += 1
                # Size violations are handled immediately; age is checked
                # periodically so a quiet cache still gets cleaned up.
                if ((self.max_bytes > 0 and self._bytes > self.max_bytes)
                        or self._writes % 200 == 0):
                    # Never remove the file we are about to serve, even when a
                    # single thumbnail is larger than the whole budget.
                    self._evict_locked(keep=out)
        return out

    def stats(self) -> dict:
        with self._lock:
            return {"files": self._files, "bytes": self._bytes,
                    "max_bytes": self.max_bytes, "max_age_days": round(self.max_age_s / 86400)}

    def clear(self) -> int:
        with self._lock:
            removed = 0
            for p in list(self.dir.glob("*.jpg")) + list(self.dir.glob("*.tmp")):
                try:
                    p.unlink()
                    if p.suffix == ".jpg":
                        removed += 1
                except OSError:
                    pass
            self._bytes = 0
            self._files = 0
            return removed

    # ------------------------------------------------------------ internal --
    def _rescan_locked(self) -> None:
        self._bytes = 0
        self._files = 0
        for p in self.dir.glob("*.jpg"):
            try:
                self._bytes += p.stat().st_size
                self._files += 1
            except OSError:
                pass

    def _evict_locked(self, keep: Path | None = None) -> None:
        now = time.time()
        entries: list[tuple[float, int, Path]] = []
        for p in self.dir.glob("*.jpg"):
            try:
                st = p.stat()
            except OSError:
                continue
            entries.append((st.st_mtime, st.st_size, p))
        entries.sort(key=lambda e: e[0])

        removed_bytes = 0
        removed_files = 0
        for mtime, size, p in entries:
            if keep is not None and p == keep:
                continue
            too_old = self.max_age_s > 0 and now - mtime > self.max_age_s
            too_big = self.max_bytes > 0 and self._bytes - removed_bytes > self.max_bytes
            if not (too_old or too_big):
                continue
            try:
                p.unlink()
                removed_bytes += size
                removed_files += 1
            except OSError:
                pass
        self._bytes = max(0, self._bytes - removed_bytes)
        self._files = max(0, self._files - removed_files)
