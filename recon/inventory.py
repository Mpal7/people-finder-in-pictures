#!/usr/bin/env python3
"""Phase 0 recon: read-only inventory of the photo collection.

Walks the collection root, classifies every file by extension, and writes:
  - inventory.json : summary (totals, per top-folder, per extension)
  - files.csv      : full file list (relpath, size, mtime, ext, category)

Never opens file contents; never writes inside the collection root.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

IMAGE_EXT = {
    ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".heic", ".heif", ".avif", ".webp",
    ".gif", ".bmp", ".tif", ".tiff",
    # raw formats
    ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".dng",
    ".orf", ".rw2", ".raf", ".pef", ".srw",
}
VIDEO_EXT = {
    ".mov", ".mp4", ".m4v", ".avi", ".mkv", ".wmv", ".3gp", ".3g2",
    ".mts", ".m2ts", ".mpg", ".mpeg", ".webm", ".flv", ".vob",
}
SIDECAR_EXT = {".aae", ".xmp", ".thm", ".xml", ".json"}
SKIP_DIR_NAMES = {
    "$recycle.bin", "system volume information", ".trash-1000", ".trash",
    "found.000", ".fseventsd", ".spotlight-v100", ".documentrevisions-v100",
}

CATEGORY_OF = {}
for _e in IMAGE_EXT:
    CATEGORY_OF[_e] = "images"
for _e in VIDEO_EXT:
    CATEGORY_OF[_e] = "videos"
for _e in SIDECAR_EXT:
    CATEGORY_OF[_e] = "sidecars"


def classify(ext: str) -> str:
    return CATEGORY_OF.get(ext, "other")


def walk(root: str):
    """Yield (relpath, size, mtime, ext) for every regular file under root."""
    root = os.path.abspath(root)
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            entries = list(os.scandir(d))
        except OSError as e:
            print(f"WARN: cannot scan {d}: {e}", file=sys.stderr)
            continue
        for e in entries:
            name_l = e.name.lower()
            try:
                if e.is_dir(follow_symlinks=False):
                    if name_l in SKIP_DIR_NAMES:
                        continue
                    stack.append(e.path)
                elif e.is_file(follow_symlinks=False):
                    st = e.stat(follow_symlinks=False)
                    ext = os.path.splitext(name_l)[1]
                    yield (
                        os.path.relpath(e.path, root),
                        st.st_size,
                        st.st_mtime,
                        ext,
                    )
            except OSError as ex:
                print(f"WARN: cannot stat {e.path}: {ex}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", help="collection root (any folder with photos)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    args = ap.parse_args()

    root = os.path.abspath(os.path.expandvars(os.path.expanduser(args.root)))
    if not os.path.isdir(root):
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 1
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    rows: list[tuple[str, int, float, str]] = list(walk(root))
    elapsed = time.time() - t0

    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    by_top: dict[str, dict] = {}
    by_ext: dict[str, dict] = {}
    by_top_ext: dict[str, dict[str, int]] = {}

    for rel, size, _mtime, ext in rows:
        cat = classify(ext)
        top = rel.split(os.sep, 1)[0] if os.sep in rel else "(root)"
        if top == rel:  # file directly under root
            top = "(root)"

        totals[cat]["files"] += 1
        totals[cat]["bytes"] += size

        d = by_top.setdefault(top, {"files": 0, "bytes": 0, "by_category": defaultdict(lambda: {"files": 0, "bytes": 0})})
        d["files"] += 1
        d["bytes"] += size
        d["by_category"][cat]["files"] += 1
        d["by_category"][cat]["bytes"] += size

        e = by_ext.setdefault(ext or "(none)", {"files": 0, "bytes": 0, "category": cat})
        e["files"] += 1
        e["bytes"] += size

        te = by_top_ext.setdefault(top, {})
        te[ext or "(none)"] = te.get(ext or "(none)", 0) + 1

    files_csv = os.path.join(args.out, "files.csv")
    with open(files_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["relpath", "size", "mtime", "ext", "category"])
        for rel, size, mtime, ext in sorted(rows):
            w.writerow([rel, size, f"{mtime:.0f}", ext, classify(ext)])

    summary = {
        "root": root,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "walk_seconds": round(elapsed, 1),
        "total_files": len(rows),
        "totals_by_category": {k: dict(v) for k, v in sorted(totals.items())},
        "by_top_folder": {
            k: {
                "files": v["files"],
                "bytes": v["bytes"],
                "by_category": {c: dict(cv) for c, cv in sorted(v["by_category"].items())},
                "by_ext": by_top_ext.get(k, {}),
            }
            for k, v in sorted(by_top.items())
        },
        "by_ext": {k: dict(v) for k, v in sorted(by_ext.items(), key=lambda kv: -kv[1]["files"])},
    }
    inv_json = os.path.join(args.out, "inventory.json")
    with open(inv_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    t = summary["totals_by_category"]
    print(f"walked {len(rows)} files in {elapsed:.1f}s")
    print("totals:", {k: (v['files'], human(v['bytes'])) for k, v in sorted(t.items())})
    print(f"wrote {inv_json}")
    print(f"wrote {files_csv}")
    return 0


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n:.0f}B"
        n /= 1024
    return f"{n:.1f}TB"


if __name__ == "__main__":
    sys.exit(main())
