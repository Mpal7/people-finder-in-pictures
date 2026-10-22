#!/usr/bin/env python3
"""Score every indexed face against reference photos of one person.

Pipeline: embed the reference photos with the same model ensemble as the index
(flip setting and detection size default to the index's own config, stored in
the DB), score all faces (max cosine over references per model), z-normalize
per model, fuse (mean by default), rank, and generate the static review page
plus the full score dump for calibration.

Outputs (in --out):
  index.html            review page (crops, scores, checkboxes, selection export)
  data.json / candidates.csv
  crops/, thumbs/
  scores_all.csv        every face with per-model scores (Phase 2 calibration)

Usage:
  .venv/bin/python match_person.py --db data/faces.db --root "<root>" \
      --refs people/alice/refs --out review/run1 --top-faces 600
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from photo_finder import db, review
from photo_finder.imaging import IMAGE_EXT, load_image
from photo_finder.models import FaceModels, MODEL_NAMES, aligned_crop
from photo_finder.paths import expand_path
from photo_finder.progress import Progress

PROJECT = Path(__file__).resolve().parent


def collect_refs(paths: list[str]) -> list[Path]:
    refs: list[Path] = []
    for p in paths:
        p = expand_path(p)
        if p.is_dir():
            refs.extend(sorted(f for f in p.iterdir()
                               if f.is_file() and f.suffix.lower() in IMAGE_EXT))
        elif p.is_file():
            refs.append(p)
        else:
            print(f"WARN: reference not found: {p}", file=sys.stderr)
    return refs


def _center_face(boxes, w: int, h: int) -> int:
    """Index of the face closest to the image center (area breaks ties)."""
    if len(boxes) == 1:
        return 0
    cx = (boxes[:, 0] + boxes[:, 2]) / (2 * w)
    cy = (boxes[:, 1] + boxes[:, 3]) / (2 * h)
    dist = np.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return int(np.lexsort((-areas, dist))[0])


def embed_reference(path: Path, fm, models: tuple[str, ...], max_side: int,
                    label: str) -> dict[str, np.ndarray] | None:
    """Embed one reference photo (positive or negative).

    Prefers the landmark sidecar written by pick_reference_faces.py; otherwise
    detects faces and uses the one closest to the center of the frame.
    """
    arr = load_image(path, max_side)
    bgr = np.ascontiguousarray(arr[:, :, ::-1])
    h, w = bgr.shape[:2]
    sidecar = Path(str(path) + ".kps.json")
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        kps_norm = data.get("kps_norm")
        if kps_norm:
            kps = np.asarray(kps_norm, dtype=np.float32) * np.array([w, h], dtype=np.float32)
            crop = aligned_crop(bgr, kps)
            return {m: fm.embed(m, crop) for m in models}
    boxes, kpss = fm.detect(bgr)
    if len(boxes) == 0:
        print(f"WARN: no face detected in {label} {path.name}; skipped")
        return None
    if len(boxes) > 1:
        print(f"note: {len(boxes)} faces in {label} {path.name}; using the one closest to center")
    i = _center_face(boxes, w, h)
    crop = aligned_crop(bgr, kpss[i])
    return {m: fm.embed(m, crop) for m in models}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(PROJECT / "data" / "faces.db"))
    ap.add_argument("--root", required=True, help="collection root (any folder; read-only recommended)")
    ap.add_argument("--refs", nargs="+", required=True, help="reference photo files or directories")
    ap.add_argument("--out", default=str(PROJECT / "review" / time.strftime("run-%Y%m%d-%H%M%S")))
    ap.add_argument("--progress-json", default="", help="write live progress JSON here (web UI)")
    ap.add_argument("--api-review", default="", help="URL the review page posts its selection to")
    ap.add_argument("--models", default="", help="override model set (default: all with embeddings in DB)")
    ap.add_argument("--top-faces", type=int, default=600, help="candidate faces to keep for review")
    ap.add_argument("--min-z", type=float, default=None, help="drop faces below this fused z-score")
    ap.add_argument("--fusion", choices=["mean", "max"], default="mean")
    ap.add_argument("--weights", default="", help="per-model weights, e.g. w600k=1,adaface=1,lvface=1")
    ap.add_argument("--neg-refs", nargs="+", default=[],
                    help="photos of look-alikes to suppress (files or directories)")
    ap.add_argument("--neg-weight", type=float, default=0.5,
                    help="penalty weight for negative-reference similarity (0 disables)")
    ap.add_argument("--no-near-dup", action="store_true",
                    help="do not group near-identical photos in the review page")
    ap.add_argument("--near-dup-threshold", type=int, default=6,
                    help="dHash Hamming distance for similar-photo grouping (default 6)")
    ap.add_argument("--flip", action="store_true", default=None, help="override ref-embedding flip TTA")
    ap.add_argument("--no-flip", action="store_false", dest="flip")
    ap.add_argument("--max-side", type=int, default=0, help="override (default: index config)")
    ap.add_argument("--insightface-root", default=str(PROJECT / "models" / "insightface"))
    ap.add_argument("--adaface", default=str(PROJECT / "models" / "adaface" / "adaface_ir_101.onnx"))
    ap.add_argument("--lvface", default=str(PROJECT / "models" / "lvface" / "LVFace-B_Glint360K.onnx"))
    args = ap.parse_args()

    root = expand_path(args.root).resolve()
    out_dir = expand_path(args.out).resolve()
    if not root.is_dir():
        print(f"ERROR: collection root not found: {root}", file=sys.stderr)
        return 1

    conn = db.connect(expand_path(args.db), create=False)
    cfg = json.loads(db.meta_get(conn, "index_config") or "{}")
    det_sizes = tuple(cfg.get("det_sizes", [1280]))
    flip = cfg.get("flip", True) if args.flip is None else args.flip
    max_side = args.max_side or cfg.get("max_side", 1600)

    db_models = {r["model"] for r in conn.execute("SELECT DISTINCT model FROM embeddings")}
    if args.models:
        models = tuple(m.strip() for m in args.models.split(",") if m.strip())
        missing = [m for m in models if m not in db_models]
        if missing:
            print(f"ERROR: models without embeddings in DB: {missing}", file=sys.stderr)
            return 1
    else:
        models = tuple(m for m in MODEL_NAMES if m in db_models)
    if not models:
        print("ERROR: no embeddings in DB; run index_faces.py first", file=sys.stderr)
        return 1
    weights = {m: 1.0 for m in models}
    if args.weights:
        for part in args.weights.split(","):
            try:
                k, v = part.split("=", 1)
                if k.strip() in weights:
                    weights[k.strip()] = float(v)
            except ValueError:
                print(f"ERROR: bad --weights entry '{part}' (expected model=value)", file=sys.stderr)
                return 1
    if sum(weights.values()) <= 0:
        print("ERROR: at least one model weight must be > 0", file=sys.stderr)
        return 1
    if args.neg_weight < 0:
        print("ERROR: --neg-weight must be >= 0", file=sys.stderr)
        return 1

    ref_paths = collect_refs(args.refs)
    if not ref_paths:
        print("ERROR: no reference photos found", file=sys.stderr)
        return 1
    neg_paths = collect_refs(args.neg_refs) if args.neg_refs else []
    print(f"references: {len(ref_paths)}  negatives: {len(neg_paths)}  models: {models}  "
          f"flip: {flip}  det: {det_sizes}")
    prog = Progress(args.progress_json)
    prog.write(stage="starting", total=len(ref_paths), done=0)

    fm = FaceModels(
        insightface_root=args.insightface_root,
        adaface_path=args.adaface,
        lvface_path=args.lvface,
        enabled=models, det_sizes=det_sizes, flip=flip,
    )

    # ---- embed references (sidecar landmarks, else best detected face) ----
    ref_vecs: dict[str, list[np.ndarray]] = {m: [] for m in models}
    ref_names: list[str] = []
    for ri, rp in enumerate(ref_paths):
        prog.write(stage="refs", done=ri, total=len(ref_paths), detail=rp.name)
        vecs = embed_reference(rp, fm, models, max_side, "reference")
        if vecs is None:
            continue
        for m in models:
            ref_vecs[m].append(vecs[m])
        ref_names.append(rp.name)
    n_refs = len(ref_names)
    if n_refs == 0:
        print("ERROR: no usable reference faces", file=sys.stderr)
        return 1
    print(f"usable references: {n_refs}")
    ref_mat = {m: np.stack(ref_vecs[m]) for m in models}  # (n_refs, 512)

    neg_vecs: dict[str, list[np.ndarray]] = {m: [] for m in models}
    neg_names: list[str] = []
    for ni, npth in enumerate(neg_paths):
        prog.write(stage="neg_refs", done=ni, total=len(neg_paths), detail=npth.name)
        vecs = embed_reference(npth, fm, models, max_side, "negative reference")
        if vecs is None:
            continue
        for m in models:
            neg_vecs[m].append(vecs[m])
        neg_names.append(npth.name)
    neg_mat = {m: np.stack(neg_vecs[m]) for m in models} if neg_names else None
    if neg_paths:
        print(f"usable negative references: {len(neg_names)}")

    # ---- load gallery ----
    prog.write(stage="scoring", done=0)
    rows = list(conn.execute(
        """SELECT f.id face_id, f.image_id, f.bbox_json, f.det_score, i.relpath, e.model, e.vec
           FROM embeddings e JOIN faces f ON f.id=e.face_id JOIN images i ON i.id=f.image_id
           WHERE e.model IN ({})""".format(",".join("?" * len(models))), models))
    face_meta: dict[int, dict] = {}
    by_face: dict[int, dict[str, np.ndarray]] = {}
    for r in rows:
        fid = r["face_id"]
        if fid not in face_meta:
            face_meta[fid] = {"relpath": r["relpath"], "image_id": r["image_id"],
                              "bbox": json.loads(r["bbox_json"]), "det_score": r["det_score"]}
        by_face.setdefault(fid, {})[r["model"]] = np.frombuffer(r["vec"], dtype=np.float32)
    # Only score faces that have an embedding for EVERY selected model; a face
    # with partial coverage would misalign the per-model score arrays.
    incomplete = sum(1 for d in by_face.values() if len(d) < len(models))
    face_order = [fid for fid in face_meta if len(by_face[fid]) == len(models)]
    if incomplete:
        print(f"note: skipping {incomplete} faces with partial model coverage "
              f"(re-run indexing with the same models to complete them)")
    if not face_order:
        print("ERROR: no faces have embeddings for all selected models", file=sys.stderr)
        return 1
    vecs: dict[str, list[np.ndarray]] = {
        m: [by_face[fid][m] for fid in face_order] for m in models}
    n_faces = len(face_order)
    print(f"gallery: {n_faces} faces in {len({m['image_id'] for m in face_meta.values()})} images")

    sims: dict[str, np.ndarray] = {}
    zs: dict[str, np.ndarray] = {}
    neg_sims: dict[str, np.ndarray] = {}
    neg_zs: dict[str, np.ndarray] = {}
    for m in models:
        X = np.stack(vecs[m])                     # (n_faces, 512), L2-normalized
        S = X @ ref_mat[m].T                      # cosine to each ref
        s = S.max(axis=1)
        sims[m] = s
        mu, sd = float(s.mean()), float(s.std()) + 1e-9
        zs[m] = (s - mu) / sd
        if neg_mat is not None:
            ns = (X @ neg_mat[m].T).max(axis=1)
            neg_sims[m] = ns
            # Same gallery normalization as the positive score, so the penalty
            # is expressed in comparable z units across models.
            neg_zs[m] = (ns - mu) / sd

    if args.fusion == "mean":
        fused_pos = sum(weights[m] * zs[m] for m in models) / sum(weights.values())
    else:
        fused_pos = np.max(np.stack([weights[m] * zs[m] for m in models]), axis=0)
    if neg_zs and args.neg_weight > 0:
        # Suppress faces that look more like the negatives than the gallery
        # average. Only above-average negative similarity is penalized.
        neg_penalty = np.mean(np.stack([np.maximum(0.0, neg_zs[m]) for m in models]), axis=0)
        fused = fused_pos - args.neg_weight * neg_penalty
    else:
        neg_penalty = np.zeros(len(face_order), dtype=np.float64)
        fused = fused_pos
    raw_mean = sum(sims[m] for m in models) / len(models)

    order = np.argsort(-fused)
    keep = order if args.min_z is None else order[fused[order] >= args.min_z]
    keep = keep[: args.top_faces]

    # ---- full score dump for calibration ----
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "scores_all.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        header = (["relpath", "face_id", "det_score", "fused_z", "fused_pos_z", "neg_penalty",
                   "raw_mean"] + [f"sim_{m}" for m in models]
                  + [f"z_{m}" for m in models])
        if neg_mat is not None:
            header += [f"neg_sim_{m}" for m in models] + [f"neg_z_{m}" for m in models]
        w.writerow(header)
        for k, rank_i in enumerate(order):
            fid = face_order[rank_i]
            meta = face_meta[fid]
            row = [meta["relpath"], fid, round(meta["det_score"], 4),
                   round(float(fused[rank_i]), 4), round(float(fused_pos[rank_i]), 4),
                   round(float(neg_penalty[rank_i]), 4), round(float(raw_mean[rank_i]), 4)]
            row += [round(float(sims[m][rank_i]), 4) for m in models]
            row += [round(float(zs[m][rank_i]), 3) for m in models]
            if neg_mat is not None:
                row += [round(float(neg_sims[m][rank_i]), 4) for m in models]
                row += [round(float(neg_zs[m][rank_i]), 3) for m in models]
            w.writerow(row)

    # ---- group kept faces into image candidates ----
    by_image: dict[int, dict] = {}
    for rank_i in keep:
        fid = face_order[rank_i]
        meta = face_meta[fid]
        neg_val = (max(float(neg_sims[m][rank_i]) for m in models)
                   if neg_mat is not None else None)
        img = by_image.setdefault(meta["image_id"], {
            "relpath": meta["relpath"], "score": float(fused[rank_i]), "faces": [],
            "per_model": {m: float(sims[m][rank_i]) for m in models},
            "neg": neg_val,
        })
        img["score"] = max(img["score"], float(fused[rank_i]))
        if neg_val is not None:
            img["neg"] = max(img.get("neg") or 0.0, neg_val)
        img["faces"].append({"bbox": meta["bbox"], "score": float(fused[rank_i])})
    candidates = sorted(by_image.values(), key=lambda c: -c["score"])
    for c in candidates:
        c["faces"].sort(key=lambda f: -f["score"])
        c["date"] = ""
    print(f"candidates: {len(keep)} faces in {len(candidates)} images -> {out_dir}")

    review.generate(out_dir, candidates, ref_names, list(models), root, load_image, max_side,
                    progress=prog, api_url=args.api_review,
                    near_dup_threshold=0 if args.no_near_dup else args.near_dup_threshold)
    print(f"review page: {out_dir / 'index.html'}")
    print(f"score dump:  {out_dir / 'scores_all.csv'}")
    (out_dir / "run_meta.json").write_text(json.dumps({
        "refs": ref_names, "models": list(models), "fusion": args.fusion,
        "top_faces": args.top_faces, "faces_kept": len(keep),
        "candidate_images": len(candidates), "min_z": args.min_z,
        "weights": args.weights, "negative_refs": neg_names,
        "neg_weight": args.neg_weight,
        "near_dup_threshold": 0 if args.no_near_dup else args.near_dup_threshold,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=1), encoding="utf-8")
    prog.done(stage="finished", done=len(candidates), total=len(candidates))
    return 0


if __name__ == "__main__":
    sys.exit(main())
