#!/usr/bin/env python3
"""Resume indexing from any folder, detached, on Windows or POSIX.

The collection root is an ordinary path: a local disk, an external/removable
drive, a NAS/network share, or a read-only mount. It is never hardcoded.

Examples:
  .venv\\Scripts\\python.exe -m cli.resume_index --root "D:/Photos"
  .venv/bin/python -m cli.resume_index --root "/mnt/foto"

The worker is the same resumable CLI used by the web UI. It is registered in
``data/jobs.json``, so progress and logs also appear in the Jobs tab.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from photo_finder.jobs import JobManager  # noqa: E402
from photo_finder.paths import expand_path  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("PHOTO_ROOT", ""),
                    help="collection root (or set PHOTO_ROOT)")
    ap.add_argument("--db", default=str(PROJECT / "data" / "faces.db"))
    ap.add_argument("--models", default="w600k",
                    help="comma-separated: w600k,adaface,lvface")
    ap.add_argument("--det-sizes", default="1280",
                    help="comma-separated SCRFD input sizes, e.g. 1280 or 640,1280")
    ap.add_argument("--no-flip", action="store_true", help="disable flip TTA")
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dedup", default="", help="optional exact-duplicate map CSV")
    ap.add_argument("--folder-rank", default="", help="optional folder-rank JSON")
    ap.add_argument("--include-non-canonical", action="store_true")
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--check", action="store_true", help="only check root/job state")
    args = ap.parse_args()

    if not args.root:
        print("ERROR: --root is required (or set PHOTO_ROOT), e.g.\n"
              '  python -m cli.resume_index --root "D:/Photos"\n'
              '  python -m cli.resume_index --root "/mnt/foto"', file=sys.stderr)
        return 1
    root = expand_path(args.root).resolve()
    if not root.is_dir():
        print(f"ERROR: collection root not found: {root}", file=sys.stderr)
        return 1

    jm = JobManager(PROJECT / "data")
    active = jm.active("index")
    if active:
        print(f"index already running: {active['id']} (pid {active['pid']})")
        print(f"log: {active['log']}")
        return 1
    if args.check:
        print(f"ready: {root}")
        return 0

    db_path = expand_path(args.db).resolve()
    cmd = [sys.executable, "-m", "cli.index_faces", str(root),
           "--db", str(db_path), "--models", args.models,
           "--det-sizes", args.det_sizes, "--max-side", str(args.max_side),
           "--threads", str(args.threads), "--progress-json", "{progress}"]
    if args.no_flip:
        cmd.append("--no-flip")
    if args.dedup:
        cmd += ["--dedup", str(expand_path(args.dedup))]
    if args.folder_rank:
        cmd += ["--folder-rank", str(expand_path(args.folder_rank))]
    if args.include_non_canonical:
        cmd.append("--include-non-canonical")
    if args.retry_errors:
        cmd.append("--retry-errors")

    job = jm.start("index", cmd, f"Resume index ({root.name or root})",
                   meta={"root": str(root)})
    print(f"started {job['id']} (pid {job['pid']})")
    print(f"log: {job['log']}")
    print(f"progress: {sys.executable} -m cli.index_status")
    return 0


if __name__ == "__main__":
    sys.exit(main())
