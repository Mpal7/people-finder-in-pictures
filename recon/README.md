# Phase 0 recon (read-only)

Tiny scripts that inventory any photo collection folder and measure real speed
on this machine. Nothing here writes to the collection; the only network access
is `setup_models.py` (one-time model downloads).

## Scripts

| Script | Purpose | Outputs |
|---|---|---|
| `inventory.py <root>` | Walk the collection, classify files by extension | `out/inventory.json`, `out/files.csv` |
| `build_dupmap.py <root>` | Exact-duplicate map (size + MD5; images only) | `out/duplicates.csv`, `out/dup_summary.json` |
| `setup_models.py` | One-time download of buffalo_l + AdaFace IR-101 + LVFace-B into `../models/` | `out/setup_models.log` |
| `sample_timing.py <root>` | Time a stratified 300-image sample end-to-end (load/detect/embed+flip) and project an ETA | `out/timing_details.csv`, `out/timing_sample.csv`, `out/timing_summary.json` |

Run everything from the project root with the project venv:

```powershell
# Windows
.\.venv\Scripts\python.exe recon\inventory.py "D:\Photos"
.\.venv\Scripts\python.exe recon\build_dupmap.py "D:\Photos"
.\.venv\Scripts\python.exe recon\setup_models.py
.\.venv\Scripts\python.exe recon\sample_timing.py "D:\Photos" --n 300
```

```bash
# Linux
.venv/bin/python recon/inventory.py "/mnt/foto/Foto"
.venv/bin/python recon/build_dupmap.py "/mnt/foto/Foto"
.venv/bin/python recon/setup_models.py
.venv/bin/python recon/sample_timing.py "/mnt/foto/Foto" --n 300
```

## Notes

- `<root>` is any folder containing the photos (local, removable, or network).
- `sample_timing.py --det-sizes 640,1280` times SCRFD-10G at both scales; faces
  are taken from the last (largest) size for the embedding stage.
- Exact duplicates are detected at byte level only; recompressed WhatsApp
  look-alikes and HEIC→JPG conversions need perceptual dedup later (optional).
- `screenshot_visual_probe.py <root> [--db data/faces.db]` prints score
  distributions for the screenshot classifier on labeled groups.
