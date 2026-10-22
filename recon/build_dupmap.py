#!/usr/bin/env python3
"""Phase 0 recon: exact-duplicate map for the image collection (size + MD5).

Read-only. Reads the file list produced by inventory.py, groups images by size,
hashes only size-collision candidates with MD5, and writes:
  - duplicates.csv : group_id,size,md5,relpath  (one row per file in a dup group)
  - dup_summary.json : stats incl. per-folder counts, cross-folder matrix,
                       largest groups, unique-image count, wasted bytes

Quasi-duplicates (recompressed WhatsApp copies, HEIC/JPG conversions) are NOT
detected here by design: byte comparison only. Perceptual dedup is a later,
optional decision (plan section 7).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

CHUNK = 1024 * 1024


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def top_of(rel: str) -> str:
    p = rel.replace("\\", "/")
    return p.split("/", 1)[0] if "/" in p else "(root)"


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", help="collection root")
    ap.add_argument("--files-csv", default=os.path.join(here, "out", "files.csv"))
    ap.add_argument("--out", default=os.path.join(here, "out"))
    args = ap.parse_args()

    root = os.path.abspath(os.path.expandvars(os.path.expanduser(args.root)))
    os.makedirs(args.out, exist_ok=True)

    rows = []
    with open(args.files_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["category"] == "images":
                rows.append((r["relpath"], int(r["size"])))

    by_size: dict[int, list[str]] = defaultdict(list)
    for rel, size in rows:
        by_size[size].append(rel)

    candidates = []  # (size, rel)
    for size, rels in by_size.items():
        if len(rels) > 1:
            for rel in rels:
                candidates.append((size, rel))
    # sort by folder then path so reads are locality-friendly
    candidates.sort(key=lambda t: (top_of(t[1]), os.path.dirname(t[1]), t[1]))

    print(f"images: {len(rows)}  unique sizes: {len(by_size)}  size-candidates: {len(candidates)}", flush=True)

    t0 = time.time()
    hashed_bytes = 0
    hashes: dict[str, str] = {}
    for i, (size, rel) in enumerate(candidates, 1):
        p = os.path.join(root, rel)
        try:
            hashes[rel] = md5_file(p)
            hashed_bytes += size
        except OSError as e:
            print(f"WARN: cannot read {p}: {e}", file=sys.stderr, flush=True)
        if i % 200 == 0 or i == len(candidates):
            el = time.time() - t0
            rate = hashed_bytes / el / 1024 / 1024 if el > 0 else 0
            print(f"  hashed {i}/{len(candidates)} files, {hashed_bytes/2**30:.1f} GB, {rate:.0f} MB/s", flush=True)

    # group by (size, digest)
    groups: dict[tuple[int, str], list[str]] = defaultdict(list)
    for size, rel in candidates:
        if rel in hashes:
            groups[(size, hashes[rel])].append(rel)
    groups = {k: v for k, v in groups.items() if len(v) > 1}

    dup_files = sum(len(v) for v in groups.values())
    dup_bytes = sum(k[0] * len(v) for k, v in groups.items())
    unique_images = len(rows) - dup_files + len(groups)
    wasted = dup_bytes - sum(k[0] for k in groups)  # bytes avoidable by keeping 1 copy per group

    gid_of = {}
    ordered = sorted(groups.items(), key=lambda kv: (-kv[0][0], kv[1][0]))
    with open(os.path.join(args.out, "duplicates.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["group_id", "size", "md5", "relpath"])
        for gid, ((size, digest), rels) in enumerate(ordered, 1):
            for rel in sorted(rels):
                w.writerow([gid, size, digest, rel])
                gid_of[rel] = gid

    # stats
    per_folder = defaultdict(lambda: {"dup_files": 0, "dup_bytes": 0})
    pair_matrix = defaultdict(int)
    for (size, _digest), rels in groups.items():
        tops = sorted({top_of(r) for r in rels})
        for r in rels:
            t = top_of(r)
            per_folder[t]["dup_files"] += 1
            per_folder[t]["dup_bytes"] += size
        for i, a in enumerate(tops):
            for b in tops[i + 1:]:
                pair_matrix[f"{a} -> {b}"] += 1

    largest = []
    for gid, ((size, digest), rels) in enumerate(ordered, 1):
        if gid > 25:
            break
        largest.append({"group_id": gid, "size": size, "count": len(rels), "paths": sorted(rels)[:12]})

    summary = {
        "root": root,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hash_seconds": round(time.time() - t0, 1),
        "images_total": len(rows),
        "images_unique_exact": unique_images,
        "size_candidates": len(candidates),
        "hashed_bytes": hashed_bytes,
        "dup_groups": len(groups),
        "dup_files": dup_files,
        "dup_bytes": dup_bytes,
        "avoidable_bytes": wasted,
        "per_folder": dict(sorted(per_folder.items())),
        "cross_folder_groups": dict(sorted(pair_matrix.items(), key=lambda kv: -kv[1])),
        "largest_groups": largest,
    }
    with open(os.path.join(args.out, "dup_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"hashed {hashed_bytes/2**30:.2f} GB in {summary['hash_seconds']}s")
    print(f"exact-duplicate groups: {len(groups)}  files: {dup_files}  avoidable: {wasted/2**30:.2f} GB")
    print(f"unique images (exact): {unique_images}")
    print(f"wrote {os.path.join(args.out, 'duplicates.csv')}")
    print(f"wrote {os.path.join(args.out, 'dup_summary.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
