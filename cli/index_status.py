#!/usr/bin/env python3
"""One-command status of the face index (safe to run while indexing is active).

Examples:
  .venv/bin/python -m cli.index_status                 # current index progress
  .venv/bin/python -m cli.index_status --watch 60      # refresh every 60 s
  .venv/bin/python -m cli.index_status --db data/faces.db
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from photo_finder.jobs import pid_alive  # noqa: E402
from photo_finder.paths import expand_path  # noqa: E402


def fmt_h(seconds: float) -> str:
    m = int(seconds // 60)
    return f"{m // 60}h{m % 60:02d}m"


def snapshot(db_path: Path) -> str:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    meta = {r["key"]: r["value"] for r in conn.execute("SELECT key,value FROM meta")}
    total = int(meta.get("run_total", 0) or 0)
    processed = int(meta.get("run_processed", 0) or 0)
    skipped = int(meta.get("run_skipped", 0) or 0)
    errors = int(meta.get("run_errors", 0) or 0)
    avg = float(meta.get("run_avg_s", 0) or 0)
    pid = int(meta.get("run_pid", 0) or 0)
    started = meta.get("run_started_at", "?")
    finished = meta.get("run_finished_at")
    ok = conn.execute("SELECT COUNT(*) c, COALESCE(SUM(n_faces),0) f FROM images WHERE status='ok'").fetchone()
    err_rows = list(conn.execute("SELECT relpath, error FROM images WHERE status='error' LIMIT 5"))
    db_mb = db_path.stat().st_size / 2**20
    wal = db_path.with_suffix(db_path.suffix + "-wal")
    wal_mb = wal.stat().st_size / 2**20 if wal.exists() else 0

    running = finished is None and pid_alive(pid)
    state = "RUNNING" if running else ("DONE" if finished else ("STOPPED (resume by re-running the same command)" if pid else "NOT STARTED"))
    lines = []
    lines.append(f"index status: {state}   started: {started}   finished: {finished or '-'}   pid: {pid}")
    if total:
        done = ok["c"]
        pct = 100.0 * done / total
        bar_len = 40
        filled = int(bar_len * pct / 100)
        lines.append(f"[{'#' * filled}{'.' * (bar_len - filled)}] {done}/{total} images ({pct:.1f}%)  faces: {ok['f']}")
    lines.append(f"this run: processed={processed} skipped={skipped} errors={errors} avg={avg:.2f}s/img")
    if running and total and avg > 0:
        remaining = total - ok["c"]
        eta_s = avg * remaining
        lines.append(f"approx remaining: {fmt_h(eta_s)}  (finished around {datetime.fromtimestamp(time.time() + eta_s).strftime('%a %H:%M')})")
    lines.append(f"db: {db_mb:.0f} MB (+{wal_mb:.0f} MB wal)  at {db_path}")
    for r in err_rows:
        lines.append(f"  error: {r['relpath']}: {r['error']}")
    if errors > len(err_rows):
        lines.append(f"  ... {errors - len(err_rows)} more errors in the run log")
    conn.close()
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(PROJECT / "data" / "faces.db"))
    ap.add_argument("--watch", type=int, default=0, help="refresh every N seconds")
    args = ap.parse_args()
    db_path = expand_path(args.db)
    if not db_path.exists():
        print(f"no database at {db_path} (index not started yet)")
        return 1
    while True:
        print(snapshot(db_path))
        if not args.watch:
            return 0
        print(f"--- refresh every {args.watch}s, Ctrl-C to stop ---", flush=True)
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
