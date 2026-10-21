# Family Photo Finder

Local, fully offline face-recognition tool that finds every photo of a person in
a personal photo collection. Build a resumable face index once, then match any
number of people against it, review candidates in a local web UI, and copy the
confirmed photos into a separate folder.

Everything runs on your machine. Nothing is uploaded. Original photos are only
ever read.

## Features

- **Resumable index** — SQLite face index, transaction per image; stop and
  restart any time. Adding a model later only computes the missing embeddings.
- **Multiple people** — per-person reference sets, landmark-assisted alignment,
  and match runs with a ranked review page.
- **Model ensemble** — InsightFace `w600k_r50` base, optionally AdaFace IR-101
  and LVFace-B, fused with per-model z-normalization.
- **Faces view** — filter the whole collection to photos containing people;
  filter by face count, detection score, face size, folder and screenshot class.
- **Screenshot filter** — classify phone screenshots (header pass + optional
  visual refinement) and hide them from the Faces view.
- **Safe export** — copies only, flat or structured, sha256 manifest, refuses
  destinations inside the collection.
- **Job manager** — indexing, scanning, matching and copying run as detached
  jobs with live progress, logs and a Stop button; they survive UI restarts.
- **Local web UI** — FastAPI backend + vanilla JS frontend, no external assets.

## Requirements

- **Linux** (the scripts, mount checks and job management target Linux).
- Python **3.10–3.12**.
- CPU-only works. A GPU helps but is not required.
- Enough disk space for `data/faces.db` (roughly 1–3 GB for 20k images,
  depending on models) and for exported copies.
- A photo drive you can mount **read-only** (strongly recommended).

## Install

```bash
git clone <this-repo> family-photo-finder
cd family-photo-finder
./setup.sh
```

`setup.sh` creates `.venv`, installs `requirements.txt`, and downloads the model
weights once (~1 GB) via `recon/setup_models.py`. Afterwards the tool can run
fully offline.

Manual equivalent:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python recon/setup_models.py
```

If `insightface` fails to build from source, install the build toolchain first
(for Debian/Ubuntu: `sudo apt install build-essential python3-dev`) and retry.

> **Model licenses:** the code is MIT (`LICENSE`), but the pretrained weights are
> non-commercial research assets. See [NOTICE.md](NOTICE.md) before any
> commercial use.

## Prepare the photo collection

Mount the drive read-only so originals cannot be modified by accident:

```bash
# check current mount options
findmnt -no TARGET,OPTIONS /path/to/mount

# remount read-only if needed
sudo mount -o remount,ro /path/to/mount
```

The app opens images read-only regardless, but a read-only mount is a useful
safety belt.

## Run

```bash
./run_app.sh               # http://127.0.0.1:8770
./run_app.sh --port 9000   # if 8770 is taken
```

On first start, set the **collection root** (the top folder containing your
photos) in **Dashboard → Settings**. The UI has four tabs: **Dashboard**,
**People**, **Faces**, **Jobs**.

The server binds to `127.0.0.1` by default. Do not expose it to a network: the
API is unauthenticated and can read your collection.

---

## 1. Dashboard — the shared face index

The index is built once and reused by every person. For every image it stores
the detected faces and one embedding vector per enabled model.

- **Index / add models** — pick models (`w600k` = base; add `adaface` and/or
  `lvface` for sharper ranking), flip TTA, detection size (1280 = best recall).
  - Already-indexed images are skipped automatically; picking an extra model
    only computes the missing embeddings (no re-detection).
  - Safe to stop any time: it is transaction-per-image and resumes where it
    left off.
- **Progress** — images indexed/total, faces, ETA. The **Jobs** tab has the
  live log.
- **Embedding coverage** — how many faces have each model's embedding. The
  matcher automatically uses every fully-covered model and fuses them.
- **Unreadable files** — corrupt originals are reported and skipped, never
  modified.
- **Settings** — collection root path.

Indexing time depends on CPU and collection size. As a reference point, a
low-end 4-core CPU took about 12 h for ~20k images with `w600k` at detection
size 1280 and flip TTA; a modern CPU is substantially faster. AdaFace is the
most expensive model; LVFace is a cheaper addition. Run long passes overnight —
everything is resumable.

## 2. People — find someone in the collection

Create a person (for example `Alice`), then work down the three steps in the
person card.

### Step 1 — Reference photos

Collect **5–20 photos of the person** across different ages, angles and lighting
and drop them into `people/<slug>/refs_src/` (or another folder and type its
path). Then press **Scan folder**. Scanning detects all faces in each reference
photo and builds the picker page.

### Step 2 — Pick the face

Press **Open picker**. For every reference photo, click the numbered chip of the
right face; click "not this person / skip" for photos that do not contain the
person. Then **Export selection** — the selection is saved on the server. Back
in the app press **Apply picks**: this writes one crop per reference into
`people/<slug>/refs/` as `<name>_face.jpg`, together with a `.kps.json`
landmark sidecar. The sidecar lets the matcher align tight crops exactly — the
detector cannot re-find faces that fill a whole frame.

### Step 3 — Match

Choose how many top faces to review (default 1000) and press **Run match**. It
embeds the references, scores every indexed face with max-cosine per model,
z-normalizes, fuses, and generates the review page.

### Review and export

**Open review** next to the run:

- Cards are sorted best-first; each shows the photo, its score and face crops.
- Filter with the score box, tick photos (or **Select shown** for everything
  above the cutoff), then **Export selection** — saved server-side.
- Back in the app, **Copy to found/<slug>/** starts a copy job. Files are copied
  flat (folder structure dropped); originals are never moved or modified. Every
  copy is sha256-verified and listed in `found/<slug>/manifest.csv`.
- Multiple review rounds: tick more, export again, copy again — files already
  copied are skipped.

Score guide (fused z, per collection): **≥ 8** near-certain (references that are
copies of collection photos score ~14); **5–8** strong; **2.5–5** worth a look;
below that it is usually other people. Old scanned photos legitimately score
lower.

## 3. Faces — photos that contain people (any person)

Filter the whole indexed collection down to pictures with people, to skip junk
like screenshots, documents and memes.

- **Filters**: minimum number of faces, minimum detection score, minimum face
  size as % of the frame (for example 8% drops screenshots with tiny avatars),
  folder substring, sort order (most faces / newest / path).
- **Summary bar**: matching photo count, face count and total size, live.
- **Browse**: lazy-loaded thumbnail grid; click a thumbnail to open the photo
  (HEIC/CR2 are shown through their embedded preview). Badges show faces,
  face %, file size and screenshot class.
- **Screenshot filter**: a built-in scan classifies indexed images as
  **screenshot** / **possible** / clean (PNG + screen resolution, filenames,
  scrolled captures, WhatsApp origins, plus an optional visual status-bar pass
  for ambiguous cases). Choose `hide` / `hide + possible` / `show all` /
  `only screenshots`; badged cards let you spot-check. Scan and refine buttons
  are on the same card; both are resumable.
- **Copy N photos** → type any destination folder (relative, e.g.
  `found/people`, or absolute). The whole filtered set is copied as a
  background job (Jobs tab; resumable; copies only; duplicates skipped; sha256
  manifest). Destinations inside the collection are refused, so originals can
  never be overwritten.
- **CSV list** → download the file list (path, #faces, size, best score, face %,
  date) without copying anything.

## 4. Jobs — monitoring

Every long step (index, scan, apply, match, export, screenshot analysis) appears
with status, progress %, ETA, a collapsible log and a **Stop** button. Jobs are
detached processes: closing the UI does not stop them, and re-running any job
resumes or continues.

---

## CLI (advanced / debugging)

| File | Role |
|---|---|
| `index_faces.py` | resumable face index → SQLite (`data/faces.db`), dedup-aware |
| `pick_reference_faces.py` | reference scan + face picker (crops + landmark sidecars) |
| `match_person.py` | matches a person's refs against the index → review page + scores |
| `export_matches.py` | copies selected matches (`found/<person>/`, sha256 manifest) |
| `detect_screenshots.py` | screenshot classification for indexed images |
| `index_status.py` | one-command index status (also works while the UI runs) |
| `webapp.py` | the local UI (FastAPI, 127.0.0.1 only) |

Examples:

```bash
.venv/bin/python index_faces.py /mnt/foto/Foto --models w600k --det-sizes 1280
.venv/bin/python pick_reference_faces.py --src people/alice/refs_src
.venv/bin/python pick_reference_faces.py --apply people/alice/refs_pick/selection.json \
    --src people/alice/refs_src --dest people/alice/refs
.venv/bin/python match_person.py --root /mnt/foto/Foto --refs people/alice/refs \
    --out review/alice --top-faces 600
.venv/bin/python export_matches.py --root /mnt/foto/Foto \
    --selection review/alice/selection.json --dest found/alice --flat
.venv/bin/python index_status.py --watch 60
```

`index_faces.py` options worth knowing:

- `--dedup recon/out/duplicates.csv` indexes one canonical copy per exact-
  duplicate group (built by `recon/build_dupmap.py`).
- `--folder-rank data/folder_rank.json` optionally prefers copies from specific
  top-level folders when choosing the canonical copy. Example:
  `{"camera": 80, "phone_backup": 70, "exports": 10}` (higher wins; unlisted
  folders rank 0, ties broken by path).
- `--retry-errors` retries previously unreadable files.

`resume_index.sh` relaunches the indexer after a reboot with mount checks. It is
configured through environment variables:

```bash
PHOTO_MOUNT=/mnt/foto PHOTO_ROOT=/mnt/foto/Foto ./resume_index.sh
# optional: PHOTO_DRIVE_LABEL="My Drive" tries udisksctl mount via blkid
```

## Layout

```
data/faces.db        shared face index (images, faces, embeddings)
data/jobs.json       background job registry + per-job logs/progress
data/config.json     UI settings (collection root)
data/people.json     person registry
people/<slug>/       per-person: refs_src/ refs_pick/ refs/ runs/
found/<slug>/        exported copies (flat) + manifest.csv
models/              one-time model downloads (buffalo_l, AdaFace, LVFace)
recon/               inventory / duplicate-map / timing scripts
web/                 UI frontend
docs/DESIGN.md       technical design notes
```

All runtime data, models, personal folders and exports are git-ignored.

## Privacy and security

- All processing is local; the only network access is the one-time model
  download in `recon/setup_models.py` (and pip).
- The collection is opened read-only; originals are never moved or modified.
- Reference photos, the index, review pages and exports stay on your machine.
- The web UI is unauthenticated. Keep it bound to `127.0.0.1` (the default);
  `--host` is available for advanced use but exposes your photos and API to the
  network.

## Troubleshooting

- **"collection drive not mounted"** — set the root in Dashboard → Settings and
  make sure the folder exists.
- **Model not found** — run `recon/setup_models.py` (or `./setup.sh` again).
- **No faces in the references** — use clearer, larger, more frontal crops.
  Faces that fill the whole frame cannot be re-detected; the picker's landmark
  sidecars handle that case, so use the picker instead of dropping crops into
  `refs/` manually.
- **A photo of the person is missed** — turned-away/tiny/obscured faces are
  missed by all face systems. Big age gaps are the hardest case; that is what
  the model ensemble and several reference eras are for.
- **Index re-detect on resume** — expected if you changed detection size, flip
  TTA or max-side; those settings are part of the per-image config key.
- **Duplicate-looking cards** — exact duplicates are indexed once, but
  recompressed WhatsApp copies are not grouped; review them manually.

## License

Code: MIT — see [LICENSE](LICENSE).
Model weights: separate, non-commercial research terms — see
[NOTICE.md](NOTICE.md).

## Docs

- [docs/DESIGN.md](docs/DESIGN.md) — pipeline design, model strategy, limits
- [recon/README.md](recon/README.md) — inventory, duplicate map, timing scripts
