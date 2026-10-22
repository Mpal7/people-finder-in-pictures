#!/usr/bin/env python3
"""Classify indexed images as phone screenshots and store the result in the DB.

Phase A (default): header-only pass - format, EXIF camera fields, original
dimensions, exact screen resolutions, filenames, aspect ratios. Fast
(~5 ms/image over the collection) and resumable (skips already-classified rows).

Phase B (--visual): re-score the "possible" group (class 1) with the
status-bar/UI heuristics on small thumbnails; promotes to 2 or demotes to 0.

Usage:
  .venv/bin/python detect_screenshots.py "<root>" [--visual] [--redo] [--progress-json PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from photo_finder import db
from photo_finder.paths import expand_path
from photo_finder.progress import Progress
from photo_finder.screenshots import VISUAL_T, VISUAL_T_LOW, classify_header, visual_score

PROJECT = Path(__file__).resolve().parent
DEFAULT_DB = PROJECT / "data" / "faces.db"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", help="collection root")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--visual", action="store_true", help="phase B: rescore 'possible' rows")
    ap.add_argument("--redo", action="store_true", help="recompute everything")
    ap.add_argument("--progress-json", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = expand_path(args.root).resolve()
    if not root.is_dir():
        print(f"ERROR: collection root not found: {root}", file=sys.stderr)
        return 1
    conn = db.connect(expand_path(args.db))
    prog = Progress(args.progress_json)

    if args.visual and not args.redo:
        rows = list(conn.execute(
            "SELECT id, relpath, shot_class, shot_reason FROM images WHERE status='ok' AND shot_class=1 ORDER BY id"))
        mode = "visual"
    else:
        where = "status='ok'" + ("" if args.redo else " AND shot_class IS NULL")
        rows = list(conn.execute(f"SELECT id, relpath, shot_reason FROM images WHERE {where} ORDER BY id"))
        mode = "header"
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("nothing to do (all classified - use --redo to recompute)")
        return 0

    total = len(rows)
    t0 = time.time()
    counts = {0: 0, 1: 0, 2: 0}
    promoted = demoted = 0

    for i, r in enumerate(rows, 1):
        p = root / r["relpath"]
        if mode == "visual":
            reason = r["shot_reason"] or ""
            s = visual_score(p)
            if s >= VISUAL_T:
                res = {"cls": 2, "reason": f"{reason}+visual({s})", "score": 0.9}
                promoted += 1
            elif s <= VISUAL_T_LOW:
                res = {"cls": 0, "reason": f"{reason}+visual-reject({s})", "score": 0.2}
                demoted += 1
            else:
                res = {"cls": 1, "reason": f"{reason}+visual?({s})", "score": 0.5}
            res.update(orig_w=None, orig_h=None, exif_cam=None)
            counts[res["cls"]] = counts.get(res["cls"], 0) + 1
            conn.execute(
                "UPDATE images SET shot_class=?, shot_reason=?, shot_score=? WHERE id=?",
                (res["cls"], res["reason"], res["score"], r["id"]))
            if i % 20 == 0 or i == total:
                conn.commit()
                rate = (time.time() - t0) / i
                prog.write(stage=mode, done=i, total=total, cls2=counts.get(2, 0),
                           cls1=counts.get(1, 0), cls0=counts.get(0, 0),
                           eta_s=round(rate * (total - i)), detail=r["relpath"])
            continue

        res = classify_header(p)
        counts[res["cls"]] = counts.get(res["cls"], 0) + 1
        conn.execute(
            """UPDATE images SET orig_w=?, orig_h=?, exif_cam=?, shot_class=?, shot_reason=?, shot_score=?
               WHERE id=?""",
            (res["orig_w"], res["orig_h"], res["exif_cam"], res["cls"], res["reason"],
             res["score"], r["id"]))
        if i % 50 == 0 or i == total:
            conn.commit()
            rate = (time.time() - t0) / i
            prog.write(stage=mode, done=i, total=total, cls2=counts.get(2, 0),
                       cls1=counts.get(1, 0), cls0=counts.get(0, 0),
                       eta_s=round(rate * (total - i)), detail=r["relpath"])
        if i % 1000 == 0:
            print(f"  {i}/{total}  screenshots={counts.get(2, 0)} possible={counts.get(1, 0)}", flush=True)

    conn.commit()
    wall = time.time() - t0
    prog.done(stage="finished", done=total, total=total, cls2=counts.get(2, 0),
              cls1=counts.get(1, 0), cls0=counts.get(0, 0))
    if mode == "visual":
        print(f"visual pass: {total} candidates, promoted={promoted}, demoted={demoted}")
    summary = {"mode": mode, "processed": total, "screenshots": counts.get(2, 0),
               "possible": counts.get(1, 0), "clean": counts.get(0, 0),
               "wall_s": round(wall, 1)}
    print(json.dumps(summary, indent=1))
    stats = conn.execute("""SELECT shot_class, COUNT(*) c FROM images WHERE status='ok'
                            GROUP BY shot_class ORDER BY shot_class""").fetchall()
    print("db totals by class:", {r["shot_class"]: r["c"] for r in stats})
    return 0


if __name__ == "__main__":
    sys.exit(main())
