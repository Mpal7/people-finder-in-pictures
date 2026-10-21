#!/usr/bin/env python3
"""Validate the Phase B status-bar heuristic on labeled groups from a collection.

Positives (known screenshots): shots classified 'png+screen-res' and 'wa+screen-res'
Negatives (known photos):      shots classified 'camera-exif'
Unknown:                       shots classified 'noexif+tall' / 'noexif+screen-res'

Prints score distributions per group so thresholds (VISUAL_T / VISUAL_T_LOW) can
be chosen before any reclassification.

Usage:
  .venv/bin/python recon/screenshot_visual_probe.py "<root>" [--db data/faces.db]
"""
from __future__ import annotations

import argparse
import random
import sqlite3
import statistics
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from photo_finder.screenshots import visual_score  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="collection root")
    ap.add_argument("--db", default=str(PROJECT / "data" / "faces.db"))
    args = ap.parse_args()

    root = Path(args.root)
    if not (Path(args.db)).exists():
        print(f"no database at {args.db}", file=sys.stderr)
        return 1
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rng = random.Random(11)

    def sample(reason: str, n: int = 120):
        rows = [r["relpath"] for r in conn.execute(
            "SELECT relpath FROM images WHERE status='ok' AND shot_reason LIKE ? "
            "AND shot_reason NOT LIKE '%visual%'", (reason + "%",))]
        rng.shuffle(rows)
        return rows[:n]

    groups = {
        "POS png+screen-res": sample("png+screen-res"),
        "POS wa+screen-res": sample("wa+screen-res"),
        "NEG camera-exif": sample("camera-exif", 150),
        "NEG no-exif(other)": sample("no-exif(other)", 150),
        "??? noexif+screen-res": sample("noexif+screen-res", 120),
        "??? noexif+tall": sample("noexif+tall", 150),
    }

    for label, rels in groups.items():
        scores = []
        for rel in rels:
            try:
                scores.append(visual_score(root / rel))
            except Exception:
                pass
        if not scores:
            print(f"{label}: no samples")
            continue
        scores.sort()
        n = len(scores)

        def pct(p):
            return scores[min(n - 1, int(p * n))]

        print(f"{label}: n={n}  median={statistics.median(scores):.2f} p75={pct(.75):.2f} "
              f"| >=0.75: {sum(s >= 0.75 for s in scores)}/{n} "
              f"| >=0.85: {sum(s >= 0.85 for s in scores)}/{n} "
              f"| >=0.90: {sum(s >= 0.90 for s in scores)}/{n} "
              f"| <=0.25: {sum(s <= 0.25 for s in scores)}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
