#!/usr/bin/env python3
"""Copy confirmed matches (selection.json from the review page) into a folder.

The source collection is only ever read. Files are copied with their folder
structure preserved under --dest; a CSV manifest records size and sha256 of
every copy. Nothing is moved, renamed or modified at the source.

Usage:
  .venv/bin/python export_matches.py --root "<root>" \
      --selection review/run1/selection.json --dest found/run1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

from photo_finder.paths import expand_path, is_within, rel_parts
from photo_finder.progress import Progress

PROJECT = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def same_content(src: Path, dst: Path, no_sha256: bool) -> bool:
    """Whether an existing destination file is already the same copy.

    Size alone is not enough: different photos can share a name and size.
    With --no-sha256 we fall back to the old size-only check (documented).
    """
    if src.stat().st_size != dst.stat().st_size:
        return False
    if no_sha256:
        return True
    return sha256_file(src) == sha256_file(dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="collection root (any folder; read-only recommended)")
    ap.add_argument("--selection", required=True, help="selection.json exported from the review page")
    ap.add_argument("--dest", default=str(PROJECT / "found"), help="destination folder for copies")
    ap.add_argument("--flat", action="store_true", help="copy all files directly into the dest root (no subfolders); renames on collision")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-sha256", action="store_true", help="skip hashing (faster, less I/O)")
    ap.add_argument("--progress-json", default="", help="write live progress JSON here (web UI)")
    args = ap.parse_args()

    prog = Progress(args.progress_json)

    root = expand_path(args.root).resolve()
    dest = expand_path(args.dest).resolve()
    if not root.is_dir():
        print(f"ERROR: collection root not found: {root}", file=sys.stderr)
        return 1
    if is_within(dest, root) or is_within(root, dest):
        print("ERROR: destination must be outside the collection root", file=sys.stderr)
        return 1

    sel_path = expand_path(args.selection)
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    items = sel.get("selection", sel if isinstance(sel, list) else [])
    if not items:
        print("ERROR: empty selection", file=sys.stderr)
        return 1

    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "manifest.csv"
    new_manifest = not manifest_path.exists()
    copied = skipped = failed = 0
    total_bytes = 0

    with open(manifest_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new_manifest:
            w.writerow(["file", "relpath", "score", "size", "sha256", "copied_at"])
        total = len(items)
        for item_i, item in enumerate(items):
            prog.write(stage="copying", done=item_i, total=total,
                       detail=item["path"] if isinstance(item, dict) else item)
            rel = item["path"] if isinstance(item, dict) else item
            score = item.get("score", "") if isinstance(item, dict) else ""
            try:
                parts = rel_parts(rel)
            except ValueError:
                print(f"ERROR: suspicious path in selection: {rel}", file=sys.stderr)
                failed += 1
                continue
            rel_display = "/".join(parts)
            src = root.joinpath(*parts)
            if not src.is_file():
                print(f"ERROR: source missing: {src}", file=sys.stderr)
                failed += 1
                continue
            if args.flat:
                name = parts[-1]
                dst = dest / name
                if dst.exists() and not same_content(src, dst, args.no_sha256):
                    n = 2
                    while True:
                        cand = dest / f"{Path(name).stem}_{n}{Path(name).suffix}"
                        if not cand.exists() or same_content(src, cand, args.no_sha256):
                            dst = cand
                            break
                        n += 1
                file_col = dst.name
            else:
                dst = dest.joinpath(*parts)
                file_col = rel_display
            if dst.exists() and same_content(src, dst, args.no_sha256):
                skipped += 1
                continue
            if args.dry_run:
                print(f"would copy: {rel} -> {dst.name}")
                copied += 1
                continue
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                digest = "" if args.no_sha256 else sha256_file(dst)
                size = dst.stat().st_size
                total_bytes += size
                w.writerow([file_col, rel_display, score, size, digest, time.strftime("%Y-%m-%d %H:%M:%S")])
                copied += 1
            except Exception as e:
                print(f"ERROR copying {rel}: {type(e).__name__}: {e}", file=sys.stderr)
                failed += 1

    verb = "would copy" if args.dry_run else "copied"
    print(f"{verb}: {copied}  skipped (already there): {skipped}  failed: {failed}  "
          f"({total_bytes / 2**20:.1f} MB new)")
    print(f"destination: {dest}")
    print(f"manifest:    {manifest_path}")
    prog.done(stage="finished", done=copied + skipped + failed, total=copied + skipped + failed,
              copied=copied, skipped=skipped, failed=failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
