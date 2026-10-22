# People Finder

**No cloud, no accounts, no uploads — your photos never leave your machine.**

People Finder is a local, offline face-recognition tool for personal photo
collections. Build a resumable face index once, then use it in two ways:

- **Find a person** — give it 5–20 reference photos, review the ranked matches,
  and copy the confirmed shots into a separate folder.
- **Find photos with people** — browse or filter the whole collection down to
  images containing faces, so you can skip screenshots, documents and memes.

Originals are only ever read. Everything runs on your hardware.

## Features

- **Resumable index** — SQLite face index, transaction per image; stop and
  restart any time. Adding a model later only computes the missing embeddings.
- **Person search** — multiple people, landmark-assisted alignment, negative
  (look-alike) references, and ranked match runs with a review page.
- **Model ensemble** — InsightFace `w600k_r50` base, optionally AdaFace IR-101
  and LVFace-B, fused with per-model z-normalization.
- **Readable review pages** — EXIF capture dates, score filter, and
  near-duplicate grouping for WhatsApp/HEIC pairs.
- **Faces view** — filter by face count, detection score, face size, folder and
  screenshot class.
- **Safe export** — copies only, sha256 manifest, refuses destinations inside
  the collection.
- **Local web UI** — background jobs with live progress/logs, person/run
  deletion, index reset and a bounded thumbnail cache.

## Requirements

- **Windows 10/11 or Linux.** CI runs the tests on both.
- Python **3.10–3.12**, 64-bit.
- CPU-only works; a GPU is optional.
- Roughly 1–3 GB for the index of 20k images, plus space for exported copies.
- **Any folder containing pictures** — local disk, external/removable drive,
  NAS/network share, or a read-only mount. The path is chosen at runtime; there
  is no hardcoded drive letter or `/mnt` location.

## Install

**Windows (PowerShell):**

```powershell
git clone <this-repo> people-finder
cd people-finder
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

`setup.ps1` creates `.venv`, installs `requirements.txt`, and downloads the
model weights once (~1 GB). Afterwards the tool runs fully offline.

Manual equivalent:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe recon\setup_models.py
```

**Linux:**

```bash
git clone <this-repo> people-finder
cd people-finder
./setup.sh
```

Manual equivalent:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python recon/setup_models.py
```

If pip tries to build a dependency from source on Windows, install the
Microsoft C++ Build Tools and retry. The pinned InsightFace 1.x wheel does not
need a compiler. On Debian/Ubuntu use `sudo apt install build-essential
python3-dev` instead.

> **Model licenses:** the code is MIT (`LICENSE`), but the pretrained weights are
> non-commercial research assets. See [NOTICE.md](NOTICE.md) before commercial use.

## Run

**Windows:**

```powershell
.\run_app.ps1               # http://127.0.0.1:8770
.\run_app.ps1 --port 9000   # if 8770 is taken
```

**Linux:**

```bash
./run_app.sh               # http://127.0.0.1:8770
./run_app.sh --port 9000   # if 8770 is taken
```

On first start, set the **collection root** in **Dashboard → Settings** to any
absolute path, for example `D:\Photos`, `\\NAS\photos`, `/mnt/foto`, or
`~/Pictures`. Windows drive letters, UNC shares, spaces and Unicode paths all
work. If you move the collection, update the setting; the index stores paths
relative to that root, so the same folder structure keeps working.

There is no mount requirement. A read-only mount is still the safest choice for
an archive, but People Finder only ever reads originals. The server binds to
`127.0.0.1` by default. Keep it there: the API is unauthenticated and can read
your collection.

## 1. Dashboard — build the index

The index is built once and reused by every person.

- **Index / add models** — choose models (`w600k` is the base; `adaface` and/or
  `lvface` sharpen ranking), flip TTA and detection size (1280 = best recall).
  Already-indexed images are skipped, and adding a model only computes its
  missing embeddings.
- **Progress** — images indexed, faces, ETA; the **Jobs** tab has the live log.
- **Embedding coverage** — how many faces have each model's embedding. Fully
  covered models are fused automatically.
- **Reset index...** — deletes the SQLite index so you can rebuild from scratch.
  Photos, references and runs are untouched. Changing detection size or flip
  TTA warns first, because those settings re-detect every image.
- **Settings** — collection root and the thumbnail cache (512 MB / 30 days by
  default; see `PHOTO_THUMB_CACHE_MB` and `PHOTO_THUMB_CACHE_DAYS`).

Indexing time depends on CPU and collection size. As a rough reference, a
low-end 4-core CPU took about 12 h for ~20k images with `w600k` at 1280 px and
flip TTA. Run long passes overnight — everything is resumable.

## 2. People — find someone

Create a person (for example `Alice`) and work through the three steps in the
person card. **Delete person** removes `people/<slug>/`; exports already in
`found/<slug>/` are kept.

1. **Reference photos** — drop **5–20 photos** of the person, across ages,
   angles and lighting, into `people/<slug>/refs_src/` (or type another folder)
   and press **Scan folder**.
2. **Pick the face** — open the picker, click the right face in each photo (or
   **skip**), export the selection, then press **Apply picks**. This writes
   aligned crops plus landmark sidecars into `people/<slug>/refs/`.
3. **Match** — choose the number of top faces and press **Run match**. The
   matcher embeds the references, scores every indexed face, z-normalizes and
   fuses the models, then generates the review page.

Optional: put photos of **look-alikes** into `people/<slug>/refs_neg/`. They are
used automatically on the next match and suppress false positives; the
suppression strength is the **negative weight** (default `0.5`, `0` disables).

The match card also exposes **fusion** (`mean`/`max`), **min z** (drop weak
candidates), **weights** (e.g. `w600k=1,adaface=1.5`) and **group similar
photos**.

### Review and export

- Cards are sorted best-first and show the photo, its EXIF capture date (file
  mtime fallback), score and face crops.
- Near-identical photos are grouped by perceptual hash. The best card gets a
  `+N similar` button; **Select shown** only selects currently visible cards.
- Filter with the score box, tick photos, and **Export selection**.
- Back in the app, **Copy to found/<slug>/** runs a resumable copy job. Files
  are copied flat, sha256-verified and listed in `manifest.csv`. Originals are
  never moved or modified.
- Delete obsolete runs from the match table; copies already in `found/` stay.

Score guide (fused z, per collection): **≥ 8** near-certain, **5–8** strong,
**2.5–5** worth a look, below that usually other people. Old scans legitimately
score lower.

## 3. Faces — photos with people

Browse the whole indexed collection for images containing faces, regardless of
who is in them.

- **Filters** — minimum face count, detection score, face size as % of frame,
  folder substring, and sort order (most faces / newest / path).
- **Screenshot filter** — classify images as **screenshot**, **possible** or
  clean, then hide or show them. The optional visual pass refines ambiguous
  cases; both scans are resumable.
- **Copy N photos** — copy the filtered set to any destination outside the
  collection, as a background job with a sha256 manifest.
- **CSV list** — download the filtered file list without copying anything.

## 4. Jobs

Indexing, scanning, applying picks, matching, export and screenshot analysis
all appear in **Jobs** with status, progress, ETA, a log and a **Stop** button.
They are detached processes: closing the UI does not stop them, and re-running
resumes or continues.

## CLI (advanced)

| File | Role |
|---|---|
| `index_faces.py` | resumable face index → SQLite (`data/faces.db`), dedup-aware |
| `pick_reference_faces.py` | reference scan + face picker (crops + landmark sidecars) |
| `match_person.py` | match a person's refs against the index → review page + scores |
| `export_matches.py` | copy selected matches (`found/<person>/`, sha256 manifest) |
| `detect_screenshots.py` | screenshot classification for indexed images |
| `index_status.py` | index status, also while the UI runs |
| `resume_index.py` | resume indexing from any folder, detached (Windows/Linux) |
| `webapp.py` | local UI (FastAPI, 127.0.0.1 only) |

Use `.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on Linux below.

```powershell
# Windows
.\.venv\Scripts\python.exe index_faces.py "D:\Photos" --models w600k --det-sizes 1280
.\.venv\Scripts\python.exe pick_reference_faces.py --src people\alice\refs_src
.\.venv\Scripts\python.exe pick_reference_faces.py --apply people\alice\refs_pick\selection.json `
    --src people\alice\refs_src --dest people\alice\refs
.\.venv\Scripts\python.exe match_person.py --root "D:\Photos" --refs people\alice\refs `
    --neg-refs people\alice\refs_neg --out review\alice --top-faces 600
.\.venv\Scripts\python.exe export_matches.py --root "D:\Photos" `
    --selection review\alice\selection.json --dest found\alice --flat
.\.venv\Scripts\python.exe index_status.py --watch 60
```

```bash
# Linux
.venv/bin/python index_faces.py "/mnt/foto/Foto" --models w600k --det-sizes 1280
.venv/bin/python pick_reference_faces.py --src people/alice/refs_src
.venv/bin/python pick_reference_faces.py --apply people/alice/refs_pick/selection.json \
    --src people/alice/refs_src --dest people/alice/refs
.venv/bin/python match_person.py --root "/mnt/foto/Foto" --refs people/alice/refs \
    --neg-refs people/alice/refs_neg --out review/alice --top-faces 600
.venv/bin/python export_matches.py --root "/mnt/foto/Foto" \
    --selection review/alice/selection.json --dest found/alice --flat
.venv/bin/python index_status.py --watch 60
```

Useful options:

- `index_faces.py --dedup recon/out/duplicates.csv` indexes one canonical copy
  per exact-duplicate group. `--folder-rank data/folder_rank.json` prefers
  copies from named top-level folders (higher rank wins; unlisted rank 0).
- `match_person.py --neg-refs DIR` / `--neg-weight` control look-alike
  suppression; `--fusion`, `--weights` and `--min-z` tune ranking;
  `--near-dup-threshold` (or `--no-near-dup`) controls review grouping.
- `resume_index.py` relaunches the indexer after a reboot and works on both
  platforms: `.\resume_index.ps1 --root "D:\Photos"` or
  `./resume_index.sh --root /mnt/foto/Foto`. The worker is detached and appears
  in the Jobs tab.

## Layout

```
data/faces.db        shared face index (images, faces, embeddings)
data/jobs.json       background job registry + logs/progress
data/config.json     UI settings (collection root)
data/people.json     person registry
data/thumbs_cache/   bounded thumbnail cache (safe to delete)
people/<slug>/       refs_src/ refs_pick/ refs/ refs_neg/ runs/
found/<slug>/        exported copies (flat) + manifest.csv
models/              one-time model downloads
recon/               inventory / duplicate-map / timing scripts
web/                 UI frontend
tests/               dependency-light unit tests
docs/DESIGN.md       technical design notes
```

Runtime data, models and exports are git-ignored.

## Privacy and security

- All processing is local. The only network access is the one-time model
  download in `recon/setup_models.py` (and pip).
- The collection is opened read-only; originals are never moved or modified.
- References, the index, review pages and exports stay on your machine.
- The UI is unauthenticated. Keep it bound to `127.0.0.1`; `--host` exposes
  your photos and API to the network.

## Troubleshooting

- **Collection folder not found** — update the path in **Dashboard → Settings**
  (a drive letter can change when a removable disk is reconnected; a network
  share must be online). There is no mount step and no fixed drive.
- **Model not found** — run `recon/setup_models.py`, or rerun `.\setup.ps1` /
  `./setup.sh`.
- **PowerShell blocks the scripts** — run them with
  `powershell -ExecutionPolicy Bypass -File .\run_app.ps1` (same for setup).
- **Python launcher missing** — install Python 3.10–3.12 from python.org and
  enable the `py` launcher; avoid the Microsoft Store stub if pip cannot find
  your venv.
- **No face in a reference photo** — use a clearer or larger crop; if the face
  fills the frame, use the picker so its landmark sidecar is written.
- **A photo is missed** — turned-away, tiny or obscured faces are hard for every
  face system. Age gaps are the hardest case; add references from several eras.
- **Index re-detects on resume** — expected after changing detection size, flip
  TTA or max-side; those settings are part of the per-image config key.
- **Near-duplicates not grouped** — raise `--near-dup-threshold` on the CLI, or
  turn grouping off to compare cards individually.
- **A Windows path is very long** — enable Win32 long paths or move the
  collection closer to the drive root; Windows still limits some operations.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The suite covers dates, cross-platform path safety, perceptual grouping,
thumbnail cache, review generation and picker collisions, and runs without
models or FastAPI. CI runs Python 3.10–3.12 on Linux and 3.12 on Windows.

## License

Code: MIT — see [LICENSE](LICENSE).
Model weights: separate, non-commercial research terms — see [NOTICE.md](NOTICE.md).

## Docs

- [docs/DESIGN.md](docs/DESIGN.md) — pipeline design, model strategy, limits
- [recon/README.md](recon/README.md) — inventory, duplicate map, timing scripts
