"""Static HTML review page for match candidates.

Generates a self-contained local page (no server, no external resources):
  <out>/index.html   - card grid, checkboxes, score filter, selection export
  <out>/data.json    - machine-readable candidate data
  <out>/candidates.csv
  <out>/crops/*.jpg  - face crops (with margin)
  <out>/thumbs/*.jpg - image thumbnails

Selection is exported client-side as JSON (download); export_matches.py reads it.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from PIL import Image


def _crop_with_margin(rgb, bbox, margin: float = 0.45, size: int = 200) -> Image.Image:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(w, h) * (1 + margin) / 2
    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half))
    x2 = min(rgb.shape[1], int(cx + half))
    y2 = min(rgb.shape[0], int(cy + half))
    img = Image.fromarray(rgb[y1:y2, x1:x2])
    img.thumbnail((size, size), Image.LANCZOS)
    return img


def _thumb(rgb, size: int = 480) -> Image.Image:
    img = Image.fromarray(rgb)
    img.thumbnail((size, size), Image.LANCZOS)
    return img


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Photo review - __TITLE__</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; font: 14px/1.4 system-ui, sans-serif; background: #14161a; color: #e8e8e8; }
  header { position: sticky; top: 0; z-index: 10; background: #1c1f26; padding: 10px 16px; border-bottom: 1px solid #2c313a;
           display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  header h1 { font-size: 15px; margin: 0 12px 0 0; font-weight: 600; }
  header .muted { color: #8b93a1; }
  input[type=number] { width: 90px; background: #0f1115; color: inherit; border: 1px solid #333a45; border-radius: 6px; padding: 5px 8px; }
  button { background: #2d6cdf; color: #fff; border: 0; border-radius: 6px; padding: 6px 12px; cursor: pointer; }
  button.secondary { background: #333a45; }
  button:hover { filter: brightness(1.1); }
  #grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 14px; padding: 16px; }
  .card { background: #1c1f26; border: 1px solid #2c313a; border-radius: 10px; overflow: hidden; }
  .card.sel { outline: 2px solid #2d6cdf; }
  .thumbwrap { position: relative; }
  .thumbwrap img { width: 100%; display: block; aspect-ratio: 4/3; object-fit: cover; background: #000; }
  .score { position: absolute; top: 8px; left: 8px; background: rgba(0,0,0,.72); padding: 2px 8px; border-radius: 999px;
           font-weight: 700; font-size: 13px; }
  .pick { position: absolute; top: 8px; right: 8px; width: 22px; height: 22px; }
  .meta { padding: 8px 10px 4px; }
  .meta .path { font-size: 12px; color: #aab2bf; word-break: break-all; }
  .meta .sub { font-size: 11px; color: #78808d; margin-top: 2px; }
  .faces { display: flex; gap: 6px; padding: 6px 10px 10px; overflow-x: auto; }
  .faces figure { margin: 0; text-align: center; }
  .faces img { width: 84px; height: 84px; object-fit: cover; border-radius: 8px; display: block; }
  .faces figcaption { font-size: 10px; color: #8b93a1; margin-top: 2px; }
  .empty { padding: 40px; color: #8b93a1; }
</style>
</head>
<body>
<header>
  <h1>Family photo review</h1>
  <span class="muted" id="summary"></span>
  <label>show score &ge; <input type="number" id="cutoff" step="0.01" value="0"></label>
  <button id="selectAll" class="secondary">Select shown</button>
  <button id="clearAll" class="secondary">Clear</button>
  <button id="exportBtn">Export selection (JSON)</button>
  <span class="muted" id="selInfo"></span>
</header>
<div id="grid"></div>
<script>
const DATA = __DATA__;
const grid = document.getElementById('grid');
const cutoff = document.getElementById('cutoff');
const selInfo = document.getElementById('selInfo');
const state = new Map(DATA.images.map(im => [im.path, false]));

document.getElementById('summary').textContent =
  `${DATA.images.length} candidate photos - refs: ${DATA.refs.length} - models: ${DATA.models.join(', ')}`;

function fmtScore(x) { return x.toFixed(3); }
function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function render() {
  const c = parseFloat(cutoff.value) || 0;
  grid.innerHTML = '';
  let shown = 0;
  for (const im of DATA.images) {
    if (im.score < c) continue;
    shown++;
    const card = document.createElement('div');
    card.className = 'card' + (state.get(im.path) ? ' sel' : '');
    const faces = im.faces.map(f =>
      `<figure><img loading="lazy" src="${f.crop}" alt=""><figcaption>${fmtScore(f.score)}</figcaption></figure>`).join('');
    card.innerHTML = `
      <div class="thumbwrap">
        <img loading="lazy" src="${im.thumb}" alt="">
        <span class="score">${fmtScore(im.score)}</span>
        <input class="pick" type="checkbox" ${state.get(im.path) ? 'checked' : ''}>
      </div>
      <div class="meta">
        <div class="path">${esc(im.path)}</div>
        <div class="sub">${esc(im.date || '')} ${im.per_model ? ' - ' + esc(im.per_model) : ''}</div>
      </div>
      <div class="faces">${faces}</div>`;
    card.querySelector('.pick').addEventListener('change', e => {
      state.set(im.path, e.target.checked);
      card.classList.toggle('sel', e.target.checked);
      updateSel();
    });
    grid.appendChild(card);
  }
  if (!shown) grid.innerHTML = '<div class="empty">No images at this score.</div>';
  updateSel();
}

function updateSel() {
  const n = [...state.values()].filter(Boolean).length;
  selInfo.textContent = n ? `${n} selected` : '';
}

document.getElementById('selectAll').onclick = () => {
  const c = parseFloat(cutoff.value) || 0;
  for (const im of DATA.images) if (im.score >= c) state.set(im.path, true);
  render();
};
document.getElementById('clearAll').onclick = () => { for (const k of state.keys()) state.set(k, false); render(); };
document.getElementById('exportBtn').onclick = async () => {
  const selected = DATA.images.filter(im => state.get(im.path)).map(im => ({ path: im.path, score: im.score }));
  const payload = { generated: new Date().toISOString(), refs: DATA.refs, models: DATA.models,
                    selection: selected };
  if (DATA.api) {
    try {
      const r = await fetch(DATA.api, { method: 'POST',
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      const j = await r.json();
      selInfo.textContent = `saved on server: ${j.n} selected`;
    } catch (e) { alert('save failed: ' + e); }
    return;
  }
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'selection.json';
  a.click();
  URL.revokeObjectURL(a.href);
};
cutoff.addEventListener('input', render);
render();
</script>
</body>
</html>
"""


def generate(out_dir: Path, candidates: list[dict], refs: list[str], models: list[str],
             root: Path, loader, max_side: int = 1600, progress=None, api_url: str | None = None) -> None:
    """Write crops, thumbs, CSV and index.html.

    candidates: [{relpath, score, per_model: {...}, faces: [{bbox, score}]}], sorted best-first.
    loader: photo_finder.imaging.load_image(root/path, max_side) callable.
    progress: optional photo_finder.progress.Progress for live updates.
    api_url: if set, the page POSTs the selection there instead of downloading it.
    """
    out_dir = Path(out_dir)
    crops_dir = out_dir / "crops"
    thumbs_dir = out_dir / "thumbs"
    crops_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    images_data = []
    csv_rows = []
    for i, cand in enumerate(candidates):
        if progress is not None:
            progress.write(stage="page", done=i, total=len(candidates), detail=cand["relpath"])
        rel = cand["relpath"]
        arr = loader(Path(root) / rel, max_side)
        thumb = _thumb(arr)
        thumb_name = f"t{i:04d}.jpg"
        thumb.save(thumbs_dir / thumb_name, quality=82)

        faces_data = []
        for j, f in enumerate(cand["faces"]):
            crop = _crop_with_margin(arr, f["bbox"])
            crop_name = f"c{i:04d}_{j}.jpg"
            crop.save(crops_dir / crop_name, quality=85)
            faces_data.append({"crop": f"crops/{crop_name}", "score": f["score"]})
            csv_rows.append({"rank": i + 1, "relpath": rel, "image_score": cand["score"],
                             "face_score": f["score"], "face_idx": j, "crop": crop_name})
        per_model = " ".join(f"{m}:{v:.3f}" for m, v in cand.get("per_model", {}).items())
        images_data.append({
            "path": rel,
            "score": round(cand["score"], 4),
            "per_model": per_model,
            "date": cand.get("date", ""),
            "thumb": f"thumbs/{thumb_name}",
            "faces": faces_data,
        })

    with open(out_dir / "candidates.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["rank", "relpath", "image_score", "face_score", "face_idx", "crop"])
        w.writeheader()
        w.writerows(csv_rows)

    data = {"refs": [os.path.basename(r) for r in refs], "models": models, "images": images_data,
            "api": api_url}
    (out_dir / "data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")

    html = HTML.replace("__TITLE__", out_dir.name).replace(
        "__DATA__", json.dumps(data).replace("</", "<\\/"))
    (out_dir / "index.html").write_text(html, encoding="utf-8")
