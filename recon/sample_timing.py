#!/usr/bin/env python3
"""Phase 0 recon: time a stratified 300-image sample with the real stack.

Runs, per image (read-only):
  1. load  - decode (PIL/pillow-heif; rawpy embedded preview for CR2), EXIF fix,
             downscale to max side 1600 (same preprocessing planned for indexing)
  2. detect - SCRFD-10G (from buffalo_l) at each requested det size, timed separately
  3. embed  - per face from the largest det size: w600k_r50, AdaFace IR-101,
              LVFace-B (if present); each with an optional horizontal-flip TTA pass

Outputs:
  timing_details.csv  - per image: load s, det s per size, faces, embed s per model
  timing_sample.csv   - the frozen sample (so later phases can reuse it)
  timing_summary.json - aggregates + ETA projection for the full collection
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- sample ----

def build_sample(files_csv: Path, n: int, seed: int) -> list[tuple[str, str, str]]:
    """Stratified sample proportional to (top_folder, ext); min 1 per stratum."""
    rows: list[tuple[str, str, str]] = []
    with open(files_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["category"] != "images":
                continue
            rel = r["relpath"]
            top = rel.split(os.sep, 1)[0] if os.sep in rel else "(root)"
            rows.append((rel, top, r["ext"]))

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for rel, top, ext in rows:
        strata[(top, ext)].append(rel)

    total = len(rows)
    keys = sorted(strata)
    exact = {k: len(strata[k]) * n / total for k in keys}
    min_per = 1 if n >= len(keys) else 0
    quota = {k: max(min_per, min(len(strata[k]), int(exact[k]))) for k in keys}
    # largest-remainder adjustment towards exactly n
    while sum(quota.values()) < n:
        k = max(keys, key=lambda k: exact[k] - quota[k] if quota[k] < len(strata[k]) else -1)
        if exact[k] - quota[k] <= 0:
            break
        quota[k] += 1
    while sum(quota.values()) > n:
        k = max(keys, key=lambda k: quota[k] - exact[k] if quota[k] > 1 else -1)
        if quota[k] <= 1:
            break
        quota[k] -= 1

    rng = random.Random(seed)
    sample: list[tuple[str, str, str]] = []
    for k in keys:
        pool = sorted(strata[k])
        rng.shuffle(pool)
        for rel in pool[: quota[k]]:
            sample.append((rel, k[0], k[1]))
    sample.sort()
    return sample


# ---------------------------------------------------------------- loader ----

def load_image(path: Path, max_side: int) -> np.ndarray:
    """Return RGB uint8 array, EXIF-corrected, max side limited (like indexer will)."""
    from PIL import Image, ImageOps
    import pillow_heif

    pillow_heif.register_heif_opener()
    ext = path.suffix.lower()
    if ext == ".cr2":
        import rawpy  # embedded full-size JPEG preview; avoids demosaicing
        with rawpy.imread(str(path)) as raw:
            thumb = raw.extract_thumb()
        if thumb.format == rawpy.ThumbFormat.JPEG:
            import io
            img = Image.open(io.BytesIO(thumb.data))
        else:  # BITMAP
            img = Image.fromarray(thumb.data)
    else:
        img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    m = max(h, w)
    if m > max_side:
        import cv2
        scale = max_side / m
        arr = cv2.resize(arr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------- models ----

class Embedder:
    name = "?"

    def embed(self, bgr_crop: np.ndarray) -> float:
        """Return seconds spent computing one embedding (no flip)."""
        raise NotImplementedError


class InsightFaceEmbedder(Embedder):
    def __init__(self, app):
        self.name = "w600k_r50"
        self.rec = app.models["recognition"]

    def embed(self, bgr_crop: np.ndarray) -> float:
        t0 = time.perf_counter()
        self.rec.get_feat(bgr_crop)
        return time.perf_counter() - t0


class OnnxEmbedder(Embedder):
    """Generic BGR/RGB 112x112 ONNX embedder with the given normalization."""

    def __init__(self, name: str, model_path: str, rgb: bool, mean: float, std: float):
        import onnxruntime
        self.name = name
        self.rgb = rgb
        self.mean = mean
        self.std = std
        so = onnxruntime.SessionOptions()
        so.intra_op_num_threads = 4
        self.sess = onnxruntime.InferenceSession(
            model_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        self.hw = (shape[2], shape[3]) if shape[2] != "batch" else (112, 112)

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        import cv2
        img = cv2.resize(crop, self.hw)
        x = img.astype(np.float32)
        if self.rgb:
            x = x[..., ::-1]
        x = (x - self.mean) / self.std
        return np.expand_dims(np.transpose(x, (2, 0, 1)), 0)

    def embed(self, bgr_crop: np.ndarray) -> float:
        blob = self.preprocess(bgr_crop)
        t0 = time.perf_counter()
        self.sess.run(None, {self.input_name: blob})
        return time.perf_counter() - t0


# ---------------------------------------------------------------- main ------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", help="collection root")
    ap.add_argument("--files-csv", default=str(OUT / "files.csv"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--det-sizes", default="640,1280", help="comma-separated square det sizes, timed in order; faces taken from the last")
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--insightface-root", default=str(PROJECT / "models" / "insightface"))
    ap.add_argument("--adaface", default=str(PROJECT / "models" / "adaface" / "adaface_ir_101.onnx"))
    ap.add_argument("--lvface", default=str(PROJECT / "models" / "lvface" / "LVFace-B_Glint360K.onnx"))
    ap.add_argument("--face-cap", default="", help="per-model face count cap, e.g. adaface=120,lvface=80")
    args = ap.parse_args()

    root = Path(os.path.abspath(os.path.expandvars(os.path.expanduser(args.root))))
    OUT.mkdir(parents=True, exist_ok=True)
    det_sizes = [int(s) for s in args.det_sizes.split(",") if s]
    caps = {}
    if args.face_cap:
        for part in args.face_cap.split(","):
            k, v = part.split("=")
            caps[k.strip()] = int(v)

    sample = build_sample(Path(args.files_csv), args.n, args.seed)
    with open(OUT / "timing_sample.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["relpath", "top", "ext"])
        w.writerows(sample)
    log(f"sample: {len(sample)} images across {len({s[1] for s in sample})} folders")

    # ---- models ----
    log("loading insightface buffalo_l (detection + w600k_r50)...")
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l",
        root=args.insightface_root,
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1)

    embedders: list[Embedder] = [InsightFaceEmbedder(app)]
    if Path(args.adaface).exists():
        log("loading AdaFace IR-101...")
        embedders.append(OnnxEmbedder("adaface_ir101", args.adaface, rgb=False, mean=127.5, std=127.5))
    else:
        log(f"NOTE: AdaFace model not found at {args.adaface} - skipping")
    if caps.get("lvface", 1) != 0 and Path(args.lvface).exists():
        log("loading LVFace-B...")
        # LVFace consumes 112x112 RGB crops, normalized to [-1, 1] (verify in Phase 2)
        embedders.append(OnnxEmbedder("lvface_b", args.lvface, rgb=True, mean=127.5, std=127.5))
    else:
        log("NOTE: LVFace model not present (or capped to 0) - skipping")

    from insightface.utils import face_align

    # ---- warmup ----
    log(f"warmup on {args.warmup} images...")
    for rel, _top, _ext in sample[: args.warmup]:
        img = load_image(root / rel, args.max_side)  # noqa: F841
        bgr = img[:, :, ::-1]
        for s in det_sizes:
            app.det_model.detect(bgr, input_size=(s, s))
        faces, kpss = app.det_model.detect(bgr, input_size=(det_sizes[-1], det_sizes[-1]))
        for i in range(min(len(faces), 1)):
            crop = face_align.norm_crop(bgr, kpss[i])
            for e in embedders:
                e.embed(crop)
                e.embed(np.fliplr(crop).copy())

    # ---- passes ----
    det_times: dict[int, dict[str, float]] = {s: {} for s in det_sizes}
    det_faces: dict[int, dict[str, int]] = {s: {} for s in det_sizes}
    kpss_cache: dict[str, np.ndarray] = {}
    img_cache: dict[str, np.ndarray] = {}
    loads: dict[str, float] = {}
    failures: dict[str, str] = {}

    for rel, _top, _ext in sample:
        p = root / rel
        t0 = time.perf_counter()
        try:
            img = load_image(p, args.max_side)
        except Exception as e:  # decode failure: record and continue
            failures[rel] = f"{type(e).__name__}: {e}"
            log(f"  DECODE FAIL {rel}: {e}")
            continue
        loads[rel] = time.perf_counter() - t0
        img_cache[rel] = img
        bgr = img[:, :, ::-1]
        for s in det_sizes:
            t0 = time.perf_counter()
            faces, kpss = app.det_model.detect(bgr, input_size=(s, s))
            det_times[s][rel] = time.perf_counter() - t0
            det_faces[s][rel] = int(len(faces))
            if s == det_sizes[-1]:
                kpss_cache[rel] = kpss
        log(f"  {rel}: faces={det_faces[det_sizes[-1]][rel]}")

    # ---- embedding ----
    emb_rows: list[dict] = []
    emb_stats: dict[str, list[float]] = defaultdict(list)  # per-face seconds (plain and flip recorded separately)
    flip_stats: dict[str, list[float]] = defaultdict(list)
    face_counts: dict[str, int] = defaultdict(int)
    total_faces = 0
    for rel, top, ext in sample:
        if rel not in kpss_cache:
            continue
        img = img_cache.pop(rel, None)
        if img is None:
            continue
        bgr = img[:, :, ::-1]
        kpss = kpss_cache[rel]
        row = {
            "relpath": rel, "top": top, "ext": ext,
            "load_s": round(loads.get(rel, -1), 4),
            "n_faces": len(kpss),
        }
        for s in det_sizes:
            row[f"det{s}_s"] = round(det_times[s].get(rel, -1), 4)
        for i in range(len(kpss)):
            crop = face_align.norm_crop(bgr, kpss[i])
            flip = np.fliplr(crop).copy()
            for e in embedders:
                cap_key = e.name.split("_")[0]
                if cap_key in caps and face_counts[e.name] >= caps[cap_key]:
                    continue
                dt = e.embed(crop)
                df = e.embed(flip)
                emb_stats[e.name].append(dt)
                flip_stats[e.name].append(df)
                face_counts[e.name] += 1
                total_faces += 1
                row[f"t_{e.name}"] = round(row.get(f"t_{e.name}", 0.0) + dt + df, 4)
        emb_rows.append(row)
        log(f"  embedded {rel} ({row['n_faces']} faces)")

    # ---- outputs ----
    with open(OUT / "timing_details.csv", "w", newline="", encoding="utf-8") as fh:
        cols = ["relpath", "top", "ext", "load_s", "n_faces"] + [f"det{s}_s" for s in det_sizes] + [f"t_{e.name}" for e in embedders]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in emb_rows:
            w.writerow({c: r.get(c, "") for c in cols})

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    loads_by_ext: dict[str, list[float]] = defaultdict(list)
    for rel, _t, ext in sample:
        if rel in loads:
            loads_by_ext[ext].append(loads[rel])

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_sample": len(sample),
        "n_ok": len(loads),
        "decode_failures": failures,
        "det_sizes": det_sizes,
        "loads": {
            "avg_s": avg(list(loads.values())),
            "by_ext": {ext: {"n": len(v), "avg_s": avg(v)} for ext, v in sorted(loads_by_ext.items())},
        },
        "detection": {
            str(s): {
                "avg_s": avg(list(det_times[s].values())),
                "avg_faces": avg(list(det_faces[s].values())),
            }
            for s in det_sizes
        },
        "embedding": {
            e.name: {
                "faces_timed": face_counts[e.name],
                "avg_plain_s": avg(emb_stats[e.name]),
                "avg_flip_s": avg(flip_stats[e.name]),
                "avg_both_s": avg(emb_stats[e.name]) + avg(flip_stats[e.name]),
            }
            for e in embedders
        },
        "avg_faces_per_image_1280": avg(list(det_faces[det_sizes[-1]].values())),
    }

    # ---- ETA projection ----
    inv_path = OUT / "inventory.json"
    dup_path = OUT / "dup_summary.json"
    if inv_path.exists():
        inv = json.loads(inv_path.read_text())
        images_total = inv["totals_by_category"]["images"]["files"]
        by_ext_counts = {e: v["files"] for e, v in inv["by_ext"].items() if v["category"] == "images"}
        unique = images_total
        if dup_path.exists():
            dup = json.loads(dup_path.read_text())
            unique = dup["images_unique_exact"]
        ratio = unique / images_total
        ext_unique = {e: c * ratio for e, c in by_ext_counts.items()}

        avg_load = avg(list(loads.values()))
        load_total_s = sum(ext_unique.get(ext, 0) * (avg(loads_by_ext.get(ext, [avg_load])) if loads_by_ext.get(ext) else avg_load) for ext in ext_unique)
        faces_total = unique * summary["avg_faces_per_image_1280"]

        def config_eta(det_s: int, models: list[str]) -> dict:
            det_total = unique * summary["detection"][str(det_s)]["avg_s"]
            emb_total = 0.0
            for m in models:
                st = summary["embedding"].get(m)
                if st:
                    emb_total += faces_total * st["avg_both_s"]
            total = load_total_s + det_total + emb_total
            return {
                "det_size": det_s,
                "models": models,
                "load_h": round(load_total_s / 3600, 2),
                "detect_h": round(det_total / 3600, 2),
                "embed_h": round(emb_total / 3600, 2),
                "total_h": round(total / 3600, 2),
            }

        models_all = [e.name for e in embedders]
        summary["projection"] = {
            "images_total": images_total,
            "images_unique_exact": unique,
            "estimated_faces": round(faces_total),
            "configs": [
                config_eta(1280, ["w600k_r50"]),
                config_eta(1280, models_all),
            ],
        }
        if str(640) in summary["detection"]:
            summary["projection"]["configs"].append(config_eta(640, ["w600k_r50"]))

    (OUT / "timing_summary.json").write_text(json.dumps(summary, indent=2))
    log("\n=== summary ===")
    log(json.dumps({k: v for k, v in summary.items() if k != "decode_failures"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
