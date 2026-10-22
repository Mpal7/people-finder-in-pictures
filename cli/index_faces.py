#!/usr/bin/env python3
"""Index faces in the photo collection into a resumable SQLite database.

Read-only with respect to the collection. Exact duplicates are indexed once
(canonical copy per group, from recon duplicates.csv). Interrupting and
re-running is always safe: images already fully indexed for the requested
detection config and models are skipped; adding a model later only computes
the missing embeddings for existing faces.

Usage (smoke test):
  .venv/bin/python -m cli.index_faces "<root>" --db data/faces.db \
      --dedup recon/out/duplicates.csv --models w600k --limit 30

Overnight quality-first run (recommended staging):
  .venv/bin/python -m cli.index_faces "<root>" --models w600k
  # next nights: rerun with --models w600k,adaface and then ...,lvface
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from photo_finder import db
from photo_finder.imaging import (iter_images, load_dup_map, load_folder_rank,
                                  load_image)
from photo_finder.models import FaceModels, aligned_crop
from photo_finder.paths import expand_path, is_within
from photo_finder.progress import Progress

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT / "data" / "faces.db"


def fmt_h(seconds: float) -> str:
    m = int(seconds // 60)
    return f"{m // 60}h{m % 60:02d}m"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="collection root (any folder; read-only recommended)")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--dedup", default=str(PROJECT / "recon" / "out" / "duplicates.csv"),
                    help="duplicates.csv from Phase 0; index one canonical copy per group")
    ap.add_argument("--folder-rank", default="",
                    help="optional JSON map {top-level folder: rank} used to pick the canonical duplicate copy")
    ap.add_argument("--include-non-canonical", action="store_true",
                    help="index every file even if an exact duplicate")
    ap.add_argument("--models", default="w600k", help="comma-separated: w600k,adaface,lvface")
    ap.add_argument("--det-sizes", default="1280", help="comma-separated SCRFD input sizes, e.g. 1280 or 640,1280")
    ap.add_argument("--no-flip", action="store_true", help="disable horizontal-flip TTA (faster)")
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--limit", type=int, default=0, help="process at most N images (smoke tests)")
    ap.add_argument("--retry-errors", action="store_true", help="retry images marked as errors")
    ap.add_argument("--threads", type=int, default=4, help="ORT intra-op threads per model")
    ap.add_argument("--insightface-root", default=str(PROJECT / "models" / "insightface"))
    ap.add_argument("--adaface", default=str(PROJECT / "models" / "adaface" / "adaface_ir_101.onnx"))
    ap.add_argument("--lvface", default=str(PROJECT / "models" / "lvface" / "LVFace-B_Glint360K.onnx"))
    ap.add_argument("--log", default=str(PROJECT / "data" / "index_faces.log"))
    ap.add_argument("--progress-json", default="", help="write live progress JSON here (web UI)")
    args = ap.parse_args()

    root = expand_path(args.root).resolve()
    db_path = expand_path(args.db).resolve()
    if not root.is_dir():
        print(f"ERROR: collection root not found: {root}", file=sys.stderr)
        return 1
    if is_within(db_path, root):
        print("ERROR: refusing to put the database inside the photo collection", file=sys.stderr)
        return 1

    enabled = tuple(m.strip() for m in args.models.split(",") if m.strip())
    det_sizes = tuple(int(s) for s in args.det_sizes.split(",") if s)

    dup_map = None
    dedup_path = expand_path(args.dedup)
    if not args.include_non_canonical and dedup_path.exists():
        folder_rank = load_folder_rank(expand_path(args.folder_rank) if args.folder_rank else None)
        if folder_rank:
            print(f"folder-rank preferences: {len(folder_rank)} folders")
        dup_map = load_dup_map(dedup_path, folder_rank)
        print(f"dedup map: {len(dup_map)} files in groups; indexing canonical copies only")
    rels = list(iter_images(root, dup_map, args.include_non_canonical))
    if args.limit:
        rels = rels[: args.limit]
    print(f"images to consider: {len(rels)}")

    log_path = expand_path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "a", encoding="utf-8")
    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, file=log_fh, flush=True)
        tqdm.write(msg)

    log(f"run start: root={root} db={db_path} models={enabled} det={det_sizes} "
        f"flip={not args.no_flip} max_side={args.max_side} n_images={len(rels)}")

    models = FaceModels(
        insightface_root=args.insightface_root,
        adaface_path=args.adaface,
        lvface_path=args.lvface,
        enabled=enabled,
        det_sizes=det_sizes,
        flip=not args.no_flip,
        threads=args.threads,
    )
    conn = db.connect(db_path)
    # Include every setting that changes the stored boxes/embeddings, so a
    # resumed run with different options re-detects instead of mixing configs.
    det_config = (f"{models.det_config}|flip={int(not args.no_flip)}"
                  f"|max_side={args.max_side}")
    db.meta_set(conn, "index_config", json.dumps({
        "models": enabled, "det_sizes": det_sizes, "flip": not args.no_flip,
        "max_side": args.max_side,
    }))
    db.meta_set(conn, "run_total", str(len(rels)))
    db.meta_set(conn, "run_started_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    db.meta_set(conn, "run_pid", str(os.getpid()))
    conn.execute("DELETE FROM meta WHERE key='run_finished_at'")
    conn.commit()

    prog = Progress(args.progress_json)
    prog.write(stage="starting", total=len(rels), done=0)

    processed = skipped = errors = 0
    t_process = 0.0
    total_faces_new = 0
    t0 = time.time()
    pbar = tqdm(total=len(rels), unit="img", smoothing=0.05)

    for rel in rels:
        p = root / rel
        try:
            st = p.stat()
            size, mtime = st.st_size, int(st.st_mtime)
        except OSError as e:
            log(f"stat failed {rel}: {e}")
            errors += 1
            pbar.update(1)
            continue

        row = db.get_image(conn, rel)
        complete_models: set[str] = set()
        n_faces = 0
        if row is not None and row["status"] == "ok" and row["size"] == size and row["mtime"] == mtime \
                and row["det_config"] == det_config:
            n_faces, complete_models = db.faces_with_models(conn, row["id"])
            if n_faces == 0 or all(m in complete_models for m in enabled):
                skipped += 1
                pbar.update(1)
                continue
        elif row is not None and row["status"] == "error" and row["size"] == size and row["mtime"] == mtime \
                and not args.retry_errors:
            skipped += 1
            pbar.update(1)
            continue

        t_start = time.perf_counter()
        try:
            arr = load_image(p, args.max_side)
            bgr = np.ascontiguousarray(arr[:, :, ::-1])
            h, w = arr.shape[:2]
            del arr

            if row is not None and row["status"] == "ok" and row["size"] == size and row["mtime"] == mtime \
                    and row["det_config"] == det_config:
                # image already detected: fill in missing model embeddings only
                missing = [m for m in enabled if m not in complete_models]
                face_rows = list(conn.execute(
                    "SELECT id, idx, kps_json FROM faces WHERE image_id=? ORDER BY idx", (row["id"],)))
                done = 0
                for fr in face_rows:
                    kps = np.asarray(json.loads(fr["kps_json"]), dtype=np.float32)
                    crop = aligned_crop(bgr, kps)
                    for m in missing:
                        vec = models.embed(m, crop)
                        db.store_embeddings(conn, fr["id"], m, vec.tobytes())
                    done += 1
                conn.commit()
                processed += 1
            else:
                boxes, kpss = models.detect(bgr)
                faces = []
                for i in range(len(boxes)):
                    faces.append({
                        "bbox": [round(float(v), 1) for v in boxes[i, :4]],
                        "kps": [[round(float(x), 2), round(float(y), 2)] for x, y in kpss[i]],
                        "det_score": round(float(boxes[i, 4]), 4),
                    })
                emb: dict[str, list[bytes]] = {m: [] for m in enabled}
                for i, f in enumerate(faces):
                    kps = np.asarray(f["kps"], dtype=np.float32)
                    crop = aligned_crop(bgr, kps)
                    for m in enabled:
                        emb[m].append(models.embed(m, crop).tobytes())
                gid = dup_map.get(rel, (None, True))[0] if dup_map else None
                db.store_image(conn, rel, size, mtime, w, h, det_config, gid, faces, emb)
                total_faces_new += len(faces)
                processed += 1

            t_process += time.perf_counter() - t_start
        except Exception as e:
            errors += 1
            db.mark_error(conn, rel, size, mtime, f"{type(e).__name__}: {e}")
            log(f"ERROR {rel}: {type(e).__name__}: {e}")

        pbar.update(1)
        n_done = processed + skipped
        avg_now = t_process / processed if processed else 0.0
        prog.write(stage="indexing", done=n_done, total=len(rels), processed=processed,
                   skipped=skipped, errors=errors, faces=total_faces_new,
                   avg_s=round(avg_now, 3), eta_s=round(avg_now * (len(rels) - n_done)))
        if processed >= 20 and processed % 25 == 0:
            avg = t_process / processed
            remaining = len(rels) - n_done
            pbar.set_postfix_str(
                f"proc={processed} skip={skipped} err={errors} avg={avg:.2f}s/img "
                f"eta_index={fmt_h(avg * remaining)}")
            if processed % 200 == 0:
                log(f"progress {n_done}/{len(rels)}: processed={processed} skipped={skipped} "
                    f"errors={errors} faces={total_faces_new} avg={avg:.2f}s/img "
                    f"elapsed={fmt_h(time.time() - t0)} eta={fmt_h(avg * remaining)}")
                db.meta_set(conn, "run_processed", str(processed))
                db.meta_set(conn, "run_skipped", str(skipped))
                db.meta_set(conn, "run_errors", str(errors))
                db.meta_set(conn, "run_avg_s", f"{avg:.3f}")
                conn.commit()

    pbar.close()
    wall = time.time() - t0
    avg = t_process / processed if processed else 0.0
    log(f"run done: processed={processed} skipped={skipped} errors={errors} "
        f"new_faces={total_faces_new} wall={fmt_h(wall)} avg_processed={avg:.2f}s/img")
    stats = conn.execute("SELECT COUNT(*) imgs, SUM(n_faces) faces FROM images WHERE status='ok'").fetchone()
    log(f"db now: {stats['imgs']} indexed images, {stats['faces']} faces")
    db.meta_set(conn, "run_processed", str(processed))
    db.meta_set(conn, "run_skipped", str(skipped))
    db.meta_set(conn, "run_errors", str(errors))
    db.meta_set(conn, "run_avg_s", f"{avg:.3f}")
    db.meta_set(conn, "run_finished_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    conn.commit()
    prog.done(stage="finished", done=len(rels), total=len(rels), processed=processed,
              skipped=skipped, errors=errors, faces=total_faces_new)
    log_fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
