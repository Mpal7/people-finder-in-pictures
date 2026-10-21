# Design notes

Technical background for Family Photo Finder: why the pipeline looks the way it
does, what each stage does, and where its limits are. The README covers usage;
this document covers the reasoning.

## 1. Goal and constraints

Find every photo of a specific person in a large local photo collection, then
copy the confirmed matches into a separate folder for review.

- **Local only**: no cloud upload, no external analysis. After the one-time
  model download, the tool works with networking disabled.
- **Originals are never modified**: the collection is opened read-only; matches
  are copied, verified with sha256, and recorded in a manifest.
- **Interruption-safe**: indexing, matching, screenshot analysis and copying are
  all resumable. The index is transaction-per-image; long jobs survive UI
  restarts and are tracked in a persistent job registry.
- **Quality first**: CPU-only is supported; the ensemble is designed to squeeze
  recall out of old, scanned or low-resolution photos, not to win benchmarks.

## 2. Stack

- Python 3.10-3.12, SQLite, FastAPI + a vanilla-JS UI.
- Face detection: SCRFD-10G from the InsightFace `buffalo_l` pack, at 1280 px
  input by default (optionally merging 640 + 1280 detections).
- Recognition: aligned 112×112 ArcFace crops embedded by up to three models:
  - `w600k` — InsightFace `w600k_r50` (ResNet50, WebFace600K). The base model.
  - `adaface` — AdaFace IR-101 ONNX.
  - `lvface` — LVFace-B Glint360K ONNX.
- Optional flip TTA (average an embedding with its horizontal mirror) and
  per-model z-normalization before fusion.
- Image I/O: Pillow + pillow-heif (HEIC/HEIF/AVIF), rawpy for CR2/CRW previews,
  OpenCV for resize and the screenshot heuristics.

Public accuracy references (IJB-C TAR @ FAR=1e-4) are close but not identical
protocols: `w600k_r50` ≈ 97.25, AdaFace IR-101 ≈ 97.66, LVFace-B ≈ 97.70.
The ensemble is worth more than any individual number because the failure modes
differ across models and photo eras.

## 3. Index

`data/faces.db` (SQLite, WAL) stores:

- `images` — relpath, size, mtime, dimensions, status, detection config,
  duplicate-group id, screenshot classification.
- `faces` — bbox, 5 landmarks, detection score per detected face.
- `embeddings` — one L2-normalized float32 vector per (face, model).
- `meta` — index config and run progress.

Resumability is keyed on `(relpath, size, mtime, det_config)`. `det_config`
includes detection sizes **and** flip/max-side settings, so changing the
pipeline cannot silently mix embeddings computed under different settings.
Adding a model later only embeds the missing vectors; it does not re-detect.

Exact duplicates (byte-identical) can be indexed once through a duplicates map
produced by `recon/build_dupmap.py`. Which copy is kept is decided by an
optional `--folder-rank` JSON map (higher rank wins; ties broken by path).

## 4. Matching

1. Detect the face in each reference photo (or reuse the landmark sidecar
   written by the picker), align, and embed with the same ensemble and flip
   setting as the index.
2. Score every indexed face against every reference; keep the best cosine per
   model.
3. z-normalize each model's score distribution across the gallery, fuse
   (weighted mean or max), rank faces, group them per image.
4. Emit a static review page: original thumbnail, face crops, per-model scores,
   checkboxes, score cutoff, and JSON selection export.

Because embeddings are cached, changing references, weights or the score cutoff
means a re-match in minutes, not a re-index.

Score guidance (fused z, per collection): ≥ 8 near-certain; 5–8 strong;
2.5–5 worth a look; below that usually other people. Old scans legitimately
score lower. Always calibrate on a small labeled sample of your own collection
if you need hard precision/recall numbers.

## 5. Export

Selections are JSON lists of relpaths. The exporter copies files flat (or keeps
the folder structure), skips existing identical copies, resolves same-name
collisions by renaming, records size + sha256 in `manifest.csv`, and refuses a
destination inside the collection root. Originals are only ever read.

## 6. Screenshot filtering

The Faces view can hide phone screenshots. Phase A is a header-only pass:
PNG/no-camera-EXIF, known phone screen resolutions, screenshot filenames,
scrolled aspect ratios, app-export names. Phase B re-scores only the ambiguous
"possible" group using status-bar/UI heuristics on small thumbnails. Thresholds
are deliberately conservative; expect a few false negatives rather than false
positives on labeled camera photos.

## 7. Phases and calibration

- **Recon** — inventory the collection, build the exact-duplicate map, time a
  stratified sample to project the indexing ETA.
- **Index** — start with `w600k`; add `adaface`/`lvface` overnight later.
- **Calibrate** — label a small sample (known positives across eras plus a few
  hundred negatives), compare model/fusion/threshold choices locally.
- **Match and review** — 1–3 review rounds; confirmed matches stay in `found/`.
- **Screenshot pass** — optional, after indexing.

## 8. Known limits

- Turned-away, tiny or heavily obscured faces are missed by any detector.
- Large age gaps are the hardest case; collect references from several eras.
- Screenshots and photos without a visible face are not findable by face
  matching (the Faces view exists for the "any person" case).
- Exact dedup only; recompressed WhatsApp copies and HEIC→JPG conversions are
  not grouped (an optional perceptual pass could be added).
- Model weights are non-commercial research assets; see `NOTICE.md`.
