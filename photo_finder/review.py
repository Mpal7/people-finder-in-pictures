"""Static HTML review page for match candidates.

Generates a self-contained local page (no server, no external resources):
  <out>/index.html   - card grid, checkboxes, score filter, selection export
  <out>/data.json    - machine-readable candidate data
  <out>/candidates.csv
  <out>/crops/*.jpg  - face crops (with margin)
  <out>/thumbs/*.jpg - image thumbnails

Selection is exported client-side as JSON (download); export_matches.py reads it.
Near-identical photos (WhatsApp re-encodes, HEIC/JPG pairs) are grouped by
perceptual hash so the grid shows one representative plus an expandable
"similar" set.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from PIL import Image

from .imaging import image_date
from .similarity import dhash, near_duplicate_groups


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
  .linkbtn { background: none; color: #6fa5ff; padding: 0 2px; font-size: 11px; }
  #grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 14px; padding: 16px; }
  .card { background: #1c1f26; border: 1px solid #2c313a; border-radius: 10px; overflow: hidden; }
  .card.sel { outline: 2px solid #2d6cdf; }
  .card.similar { border-style: dashed; }
  .thumbwrap { position: relative; }
  .thumbwrap img { width: 100%; display: block; aspect-ratio: 4/3; object-fit: cover; background: #000; }
  .score { position: absolute; top: 8px; left: 8px; background: rgba(0,0,0,.72); padding: 2px 8px; border-radius: 999px;
           font-weight: 700; font-size: 13px; }
  .pick { position: absolute; top: 8px; right: 8px; width: 22px; height: 22px; }
  .meta { padding: 8px 10px 4px; }
  .meta .path { font-size: 12px; color: #aab2bf; word-break: break-all; }
  .meta .sub { font-size: 11px; color: #78808d; margin-top: 2px; }
  .meta .group { font-size: 11px; color: #8b93a1; margin-top: 4px; }
  .pill { font-size: 10px; color: #8b93a1; border: 1px solid #333a45; border-radius: 999px; padding: 0 6px; }
  .faces { display: flex; gap: 6px; padding: 6px 10px 10px; overflow-x: auto; }
  .faces figure { margin: 0; text-align: center; }
  .faces img { width: 84px; height: 84px; object-fit: cover; border-radius: 8px; display: block; }
  .faces figcaption { font-size: 10px; color: #8b93a1; margin-top: 2px; }
  .empty { padding: 40px; color: #8b93a1; }
</style>
</head>
<body>
<header>
  <h1>Photo review</h1>
  <span class="muted" id="summary"></span>
  <label>show score &ge; <input type="number" id="cutoff" step="0.01" value="0"></label>
  <label id="groupWrap"><input type="checkbox" id="groupDups" checked> group similar</label>
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
const groupToggle = document.getElementById('groupDups');
const groupWrap = document.getElementById('groupWrap');
const selInfo = document.getElementById('selInfo');
const state = new Map(DATA.images.map(im => [im.path, false]));
const expanded = new Set();
const groupCount = DATA.images.filter(im => im.g_size > 1 && im.g_rank === 0).length;

document.getElementById('summary').textContent =
  `${DATA.images.length} candidate photos - ${groupCount} similar group(s) - refs: ${DATA.refs.length} - models: ${DATA.models.join(', ')}`;
if (!groupCount) groupWrap.style.display = 'none';

function fmtScore(x) { return x.toFixed(3); }
function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function shownImages() {
  const c = parseFloat(cutoff.value) || 0;
  const grouping = groupToggle.checked;
  return DATA.images.filter(im => im.score >= c &&
    (!grouping || im.g_size < 2 || im.g_rank === 0 || expanded.has(im.g)));
}

function render() {
  const visible = shownImages();
  grid.innerHTML = '';
  for (const im of visible) {
    const card = document.createElement('div');
    card.className = 'card' + (state.get(im.path) ? ' sel' : '') + (im.g_rank > 0 ? ' similar' : '');
    const faces = im.faces.map(f =>
      `<figure><img loading="lazy" src="${f.crop}" alt=""><figcaption>${fmtScore(f.score)}</figcaption></figure>`).join('');
    let groupLine = '';
    if (im.g_size > 1) {
      if (im.g_rank === 0) {
        groupLine = `<span class="pill">${im.g_size - 1} similar</span>` +
          (expanded.has(im.g) ? ' <button class="linkbtn" data-g="' + im.g + '" data-collapse="1">hide</button>'
                              : ' <button class="linkbtn" data-g="' + im.g + '">show</button>');
      } else {
        groupLine = '<span class="pill">similar</span>';
      }
    }
    const neg = im.neg == null ? '' : ' - neg ' + im.neg.toFixed(3);
    card.innerHTML = `
      <div class="thumbwrap">
        <img loading="lazy" src="${im.thumb}" alt="">
        <span class="score">${fmtScore(im.score)}</span>
        <input class="pick" type="checkbox" ${state.get(im.path) ? 'checked' : ''}>
      </div>
      <div class="meta">
        <div class="path">${esc(im.path)}</div>
        <div class="sub">${esc(im.date || 'date unknown')} ${im.per_model ? ' - ' + esc(im.per_model) : ''}${neg}</div>
        ${groupLine ? `<div class="group">${groupLine}</div>` : ''}
      </div>
      <div class="faces">${faces}</div>`;
    card.querySelector('.pick').addEventListener('change', e => {
      state.set(im.path, e.target.checked);
      card.classList.toggle('sel', e.target.checked);
      updateSel();
    });
    const gb = card.querySelector('.linkbtn');
    if (gb) gb.addEventListener('click', () => {
      const gid = parseInt(gb.dataset.g, 10);
      if (gb.dataset.collapse) expanded.delete(gid); else expanded.add(gid);
      render();
    });
    grid.appendChild(card);
  }
  if (!visible.length) grid.innerHTML = '<div class="empty">No images at this score.</div>';
  updateSel();
}

function updateSel() {
  const n = [...state.values()].filter(Boolean).length;
  selInfo.textContent = n ? `${n} selected` : '';
}

document.getElementById('selectAll').onclick = () => {
  for (const im of shownImages()) state.set(im.path, true);
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
groupToggle.addEventListener('change', render);
render();
</script>
</body>
</html>
"""


def generate(out_dir: Path, candidates: list[dict], refs: list[str], models: list[str],
             root: Path, loader, max_side: int = 1600, progress=None, api_url: str | None = None,
             near_dup_threshold: int = 6) -> None:
    """Write crops, thumbs, CSV and index.html.

    candidates: [{relpath, score, per_model: {...}, faces: [{bbox, score}],
    date?: str, neg?: float}], sorted best-first.
    loader: photo_finder.imaging.load_image(root/path, max_side) callable.
    progress: optional photo_finder.progress.Progress for live updates.
    api_url: if set, the page POSTs the selection there instead of downloading it.
    near_dup_threshold: dHash Hamming threshold for grouping; <= 0 disables.
    """
    out_dir = Path(out_dir)
    crops_dir = out_dir / "crops"
    thumbs_dir = out_dir / "thumbs"
    crops_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    hashes: list[int] = []
    aspects: list[float] = []
    for i, cand in enumerate(candidates):
        if progress is not None:
            progress.write(stage="page", done=i, total=len(candidates), detail=cand["relpath"])
        rel = cand["relpath"]
        full = Path(root) / rel
        arr = loader(full, max_side)
        h, w = arr.shape[:2]
        hashes.append(dhash(arr))
        aspects.append(w / h if h else 1.0)

        thumb = _thumb(arr)
        thumb_name = f"t{i:04d}.jpg"
        thumb.save(thumbs_dir / thumb_name, quality=82)

        faces_data = []
        for j, f in enumerate(cand["faces"]):
            crop = _crop_with_margin(arr, f["bbox"])
            crop_name = f"c{i:04d}_{j}.jpg"
            crop.save(crops_dir / crop_name, quality=85)
            faces_data.append({"crop": f"crops/{crop_name}", "score": f["score"]})

        records.append({
            "path": rel,
            "score": round(cand["score"], 4),
            "per_model": " ".join(f"{m}:{v:.3f}" for m, v in cand.get("per_model", {}).items()),
            "date": cand.get("date") or image_date(full),
            "thumb": f"thumbs/{thumb_name}",
            "faces": faces_data,
            "neg": cand.get("neg"),
        })

    group_of, rank_in_group, group_size = near_duplicate_groups(
        hashes, aspects, threshold=near_dup_threshold)

    images_data = []
    for i, rec in enumerate(records):
        item = dict(rec)
        item["g"] = group_of[i]
        item["g_rank"] = rank_in_group[i]
        item["g_size"] = group_size[i]
        images_data.append(item)

    with open(out_dir / "candidates.csv", "w", newline="", encoding="utf-8") as fh:
        fields = ["rank", "relpath", "image_score", "face_score", "face_idx", "crop",
                  "shoot_date", "near_group", "near_group_rank", "near_group_size", "neg_sim"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i, rec in enumerate(records):
            for j, f in enumerate(rec["faces"]):
                w.writerow({
                    "rank": i + 1, "relpath": rec["path"], "image_score": rec["score"],
                    "face_score": f["score"], "face_idx": j,
                    "crop": os.path.basename(f["crop"]),
                    "shoot_date": rec["date"], "near_group": group_of[i],
                    "near_group_rank": rank_in_group[i], "near_group_size": group_size[i],
                    "neg_sim": "" if rec.get("neg") is None else round(rec["neg"], 4),
                })

    data = {"refs": [os.path.basename(r) for r in refs], "models": models,
            "images": images_data, "api": api_url,
            "near_dup_threshold": near_dup_threshold}
    (out_dir / "data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")

    html = HTML.replace("__TITLE__", out_dir.name).replace(
        "__DATA__", json.dumps(data).replace("</", "<\\/"))
    (out_dir / "index.html").write_text(html, encoding="utf-8")
