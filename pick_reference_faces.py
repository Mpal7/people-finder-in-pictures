#!/usr/bin/env python3
"""Choose the target person's face in each reference photo, then crop it.

Stage A - build the picking page (no changes to your files):
  .venv/bin/python pick_reference_faces.py --src people/alice/refs_src
  -> refs_pick/index.html  (open in a browser, click the person's face in every
     photo, then click "Export selection" - or just use the web UI)

Stage B - apply the selection (writes crops, originals untouched):
  .venv/bin/python pick_reference_faces.py --apply refs_pick/selection.json \
      --src people/alice/refs_src --dest people/alice/refs
  -> people/alice/refs/<name>_face.jpg  (margin crops around the chosen face,
     up to 1200 px, plus a .kps.json landmark sidecar)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from photo_finder.imaging import IMAGE_EXT, load_image
from photo_finder.models import FaceModels
from photo_finder.progress import Progress

PROJECT = Path(__file__).resolve().parent

HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Pick the face - references</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; font: 14px/1.4 system-ui, sans-serif; background: #14161a; color: #e8e8e8; }
  header { position: sticky; top: 0; z-index: 10; background: #1c1f26; padding: 10px 16px; border-bottom: 1px solid #2c313a;
           display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  button { background: #2d6cdf; color: #fff; border: 0; border-radius: 6px; padding: 6px 12px; cursor: pointer; }
  button.secondary { background: #333a45; }
  button:hover { filter: brightness(1.1); }
  #grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 14px; padding: 16px; }
  .card { background: #1c1f26; border: 1px solid #2c313a; border-radius: 10px; padding: 10px; }
  .card.skip { opacity: .5; }
  .card img.prev { width: 100%; border-radius: 8px; display: block; background: #000; }
  .name { font-size: 12px; color: #aab2bf; margin: 6px 0; word-break: break-all; }
  .chips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
  .chip { border: 2px solid transparent; border-radius: 8px; padding: 2px; cursor: pointer; background: #0f1115; position: relative; }
  .chip img { width: 72px; height: 72px; object-fit: cover; border-radius: 6px; display: block; }
  .chip.sel { border-color: #35c26a; }
  .chip span { position: absolute; top: 2px; left: 4px; font-size: 11px; font-weight: 700;
               color: #fff; text-shadow: 0 0 4px #000, 0 0 4px #000; }
  .chip.none { display: flex; align-items: center; justify-content: center; width: 76px; height: 76px;
               color: #8b93a1; font-size: 11px; text-align: center; }
  .row { display: flex; gap: 8px; align-items: center; margin-top: 8px; }
  textarea { width: 100%; height: 120px; margin-top: 10px; background: #0f1115; color: inherit;
             border: 1px solid #333a45; border-radius: 6px; font: 11px monospace; }
  .muted { color: #8b93a1; }
</style>
</head>
<body>
<header>
  <h1 style="font-size:15px;margin:0">Pick the face in each photo</h1>
  <span class="muted" id="count"></span>
  <button id="export">Export selection</button>
  <button id="copy" class="secondary">Copy JSON</button>
</header>
<div id="grid"></div>
<textarea id="json" placeholder="The exported JSON will appear here - copy it back to the agent (or save as selection.json)"></textarea>
<script>
const DATA = __DATA__;
const state = new Map(DATA.items.map(it => [it.file, { face: null, skip: false }]));
const grid = document.getElementById('grid');
const jsonBox = document.getElementById('json');

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function answered() { return [...state.values()].filter(s => s.face !== null || s.skip).length; }
function updateCount() { document.getElementById('count').textContent = `${answered()}/${DATA.items.length} answered`; }

function render() {
  grid.innerHTML = '';
  DATA.items.forEach((it, ii) => {
    const st = state.get(it.file);
    const card = document.createElement('div');
    card.className = 'card' + (st.skip ? ' skip' : '');
    const chips = it.faces.map((f, fi) => `
      <div class="chip ${st.face === fi ? 'sel' : ''}" data-face="${fi}">
        <span>${fi + 1}</span><img loading="lazy" src="${f.thumb}" alt="">
      </div>`).join('');
    card.innerHTML = `
      <img class="prev" loading="lazy" src="${it.preview}" alt="">
      <div class="name">${esc(it.file)} &nbsp; <span class="muted">${it.faces.length} face(s)</span></div>
      <div class="chips">
        ${chips || '<div class="muted">no face detected</div>'}
        <div class="chip none ${st.skip ? 'sel' : ''}" data-skip="1">not this person /<br>skip</div>
      </div>`;
    card.querySelectorAll('.chip').forEach(ch => ch.addEventListener('click', () => {
      if (ch.dataset.skip) {
        st.face = null; st.skip = !st.skip;
      } else {
        const fi = parseInt(ch.dataset.face);
        st.face = (st.face === fi) ? null : fi; st.skip = false;
      }
      render();
    }));
    grid.appendChild(card);
  });
  updateCount();
}

function selection() {
  return {
    version: 1, generated: new Date().toISOString(),
    selections: DATA.items.filter(it => state.get(it.file).face !== null || state.get(it.file).skip)
      .map(it => {
        const st = state.get(it.file);
        if (st.skip) return { file: it.file, skip: true };
        const f = it.faces[st.face];
        const out = { file: it.file, face: st.face, bbox_norm: f.bbox_norm };
        if (f.kps_norm) out.kps_norm = f.kps_norm;
        return out;
      })
  };
}

document.getElementById('export').onclick = async () => {
  if (answered() < DATA.items.length && !confirm(`${DATA.items.length - answered()} photo(s) unanswered - export anyway?`)) return;
  const payload = selection();
  jsonBox.value = JSON.stringify(payload, null, 1);
  if (DATA.api && DATA.api.pick) {
    try {
      const r = await fetch(DATA.api.pick, { method: 'POST',
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      const j = await r.json();
      alert(`Saved on server: ${j.picked} picked, ${j.skipped} skipped. Now press "Apply picks" in the web app.`);
    } catch (e) { alert('save failed: ' + e); }
    return;
  }
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'selection.json'; a.click(); URL.revokeObjectURL(a.href);
};
document.getElementById('copy').onclick = async () => {
  jsonBox.value = JSON.stringify(selection(), null, 1);
  await navigator.clipboard.writeText(jsonBox.value);
  alert('Copied to clipboard');
};
render();
</script>
</body>
</html>
"""


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def plan_crop(shape, bbox, margin: float = 0.35, max_side: int = 1200):
    """Crop box (integers) around a bbox with margin, capped at max_side.
    Returns (x1, y1, x2, y2, scale) where scale maps the box to the saved size."""
    H, W = shape[:2]
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(w, h) * (1 + margin) / 2
    x1 = int(clamp(cx - half, 0, W))
    y1 = int(clamp(cy - half, 0, H))
    x2 = int(clamp(cx + half, 0, W))
    y2 = int(clamp(cy + half, 0, H))
    x2 = max(x2, x1 + 1)
    y2 = max(y2, y1 + 1)
    scale = min(1.0, max_side / max(x2 - x1, y2 - y1))
    return x1, y1, x2, y2, scale


def margin_crop(arr: np.ndarray, bbox: tuple[float, float, float, float],
                margin: float = 0.35, max_side: int = 1200) -> Image.Image:
    x1, y1, x2, y2, scale = plan_crop(arr.shape, bbox, margin, max_side)
    img = Image.fromarray(arr[y1:y2, x1:x2])
    if scale < 1.0:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                         Image.LANCZOS)
    return img


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def stage_build(src: Path, out: Path, progress=None, api_pick: str = "") -> int:
    refs = sorted(f for f in src.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXT)
    if not refs:
        print(f"no images in {src}", file=sys.stderr)
        return 1
    (out / "previews").mkdir(parents=True, exist_ok=True)
    (out / "thumbs").mkdir(parents=True, exist_ok=True)

    fm = FaceModels(
        insightface_root=str(PROJECT / "models" / "insightface"),
        adaface_path="", lvface_path="", enabled=(), det_sizes=(1280,),
    )
    items = []
    for i, p in enumerate(refs):
        if progress is not None:
            progress.write(stage="scanning", done=i, total=len(refs), detail=p.name)
        arr = load_image(p, max_side=1400)
        h, w = arr.shape[:2]
        bgr = np.ascontiguousarray(arr[:, :, ::-1])
        boxes, kpss = fm.detect(bgr)

        preview = Image.fromarray(arr.copy())
        preview.thumbnail((1000, 1000), Image.LANCZOS)
        draw = ImageDraw.Draw(preview)
        sx, sy = preview.width / w, preview.height / h
        faces = []
        for j in range(len(boxes)):
            x1, y1, x2, y2 = [float(v) for v in boxes[j, :4]]
            for k in range(2):
                draw.rectangle([x1 * sx + k, y1 * sy + k, x2 * sx - k, y2 * sy - k], outline="#35c26a")
            draw.text((x1 * sx + 3, y1 * sy + 2), str(j + 1),
                      fill="#35c26a", stroke_width=2, stroke_fill="#000",
                      font=ImageFont.load_default(size=22))
            thumb = margin_crop(arr, (x1, y1, x2, y2), margin=0.4, max_side=160)
            thumb_name = f"f{i:03d}_{j}.jpg"
            thumb.save(out / "thumbs" / thumb_name, quality=85)
            faces.append({
                "thumb": f"thumbs/{thumb_name}",
                "bbox_norm": [round(x1 / w, 5), round(y1 / h, 5), round(x2 / w, 5), round(y2 / h, 5)],
                "kps_norm": [[round(float(px) / w, 5), round(float(py) / h, 5)] for px, py in kpss[j]],
            })
        prev_name = f"p{i:03d}.jpg"
        preview.save(out / "previews" / prev_name, quality=85)
        items.append({"file": p.name, "preview": f"previews/{prev_name}", "faces": faces})
        print(f"  {p.name}: {len(faces)} face(s)")

    data = {"items": items, "api": {"pick": api_pick} if api_pick else None}
    html = HTML.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    (out / "index.html").write_text(html, encoding="utf-8")
    if progress is not None:
        progress.done(stage="scanned", done=len(refs), total=len(refs))
    print(f"\nopen this in your browser: {out / 'index.html'}")
    return 0


def stage_apply(sel_path: Path, src: Path, dest: Path, progress=None) -> int:
    """Write margin crops + landmark sidecars (<crop>.kps.json) for each picked face.

    Landmarks come from the export (kps_norm) when present; otherwise they are
    recovered by re-running detection on the source photo and matching the picked
    box by IoU. The sidecar lets the matcher align the tight crop exactly, since
    SCRFD cannot re-detect faces that fill almost the whole frame.
    """
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    dest.mkdir(parents=True, exist_ok=True)
    fm = None  # lazily created only if re-detection is needed
    done = skipped = 0
    for si, s in enumerate(sel.get("selections", [])):
        if progress is not None:
            progress.write(stage="applying", done=si, total=len(sel.get("selections", [])),
                           detail=s.get("file", ""))
        p = src / s["file"]
        if s.get("skip") or s.get("face") is None or "bbox_norm" not in s:
            print(f"skip: {s['file']}")
            skipped += 1
            continue

        arr = load_image(p, max_side=100_000)  # full resolution
        h, w = arr.shape[:2]
        x1, y1, x2, y2 = s["bbox_norm"]
        box_px = (x1 * w, y1 * h, x2 * w, y2 * h)

        kps_norm = s.get("kps_norm")
        if kps_norm is None:
            if fm is None:
                fm = FaceModels(
                    insightface_root=str(PROJECT / "models" / "insightface"),
                    adaface_path="", lvface_path="", enabled=(), det_sizes=(1280,),
                )
            arr_s = load_image(p, max_side=1400)  # same scale as the picker page
            hs, ws = arr_s.shape[:2]
            boxes, kpss = fm.detect(np.ascontiguousarray(arr_s[:, :, ::-1]))
            if len(boxes):
                boxes_n = boxes[:, :4] / np.array([ws, hs, ws, hs])
                scores = [iou(b, s["bbox_norm"]) for b in boxes_n]
                i = int(np.argmax(scores))
                if scores[i] > 0.3:
                    kps_norm = [[float(px) / ws, float(py) / hs] for px, py in kpss[i]]
                    print(f"  {s['file']}: landmarks recovered (IoU {scores[i]:.2f})")
                else:
                    print(f"  WARN {s['file']}: picked face not matched on re-detect "
                          f"(best IoU {scores[i]:.2f}); writing crop without landmarks")
            else:
                print(f"  WARN {s['file']}: no face on re-detect; writing crop without landmarks")

        cx1, cy1, cx2, cy2, scale = plan_crop(arr.shape, box_px, margin=0.35, max_side=1200)
        crop = Image.fromarray(arr[cy1:cy2, cx1:cx2])
        if scale < 1.0:
            crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))),
                               Image.LANCZOS)
        base_stem = p.stem.lower().replace(" ", "_")
        out_path = dest / f"{base_stem}_face.jpg"
        # Two source photos can share a stem (e.g. IMG_0001.jpg and IMG_0001.png,
        # or the same name in a case-sensitive folder). Don't overwrite a crop
        # that came from a different source: add a short deterministic suffix.
        # If the existing file has no sidecar we cannot prove it came from this
        # source, so also suffix instead of silently overwriting it.
        existing_sidecar = Path(str(out_path) + ".kps.json")
        if out_path.exists():
            prev_src = None
            if existing_sidecar.exists():
                try:
                    prev_src = json.loads(existing_sidecar.read_text(encoding="utf-8")).get("src")
                except Exception:
                    prev_src = None
            if prev_src != s["file"]:
                tag = hashlib.sha1(s["file"].encode("utf-8")).hexdigest()[:6]
                out_path = dest / f"{base_stem}_{tag}_face.jpg"
        crop.save(out_path, quality=92)

        # Always write a sidecar with the source name. Landmarks are optional:
        # when re-detection failed, the matcher can still fall back to detecting
        # the face in the margin crop.
        sidecar_data = {"src": s["file"]}
        extra = ""
        if kps_norm is not None:
            kps_px = np.asarray(kps_norm, dtype=np.float64) * np.array([w, h])
            kps_crop = (kps_px - np.array([cx1, cy1])) * scale
            kps_out = kps_crop / np.array([crop.width, crop.height])
            sidecar_data["kps_norm"] = [[round(float(a), 6), round(float(b), 6)]
                                        for a, b in kps_out]
            extra = " + landmarks"
        Path(str(out_path) + ".kps.json").write_text(json.dumps(sidecar_data, indent=1))
        print(f"wrote {out_path} ({crop.width}x{crop.height}){extra}")
        done += 1
    print(f"\nrefs ready: {done} crops in {dest}  ({skipped} skipped)")
    if progress is not None:
        progress.done(stage="applied", done=done, total=done + skipped)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="folder with the reference photos (e.g. people/alice/refs_src)")
    ap.add_argument("--out", default=str(PROJECT / "refs_pick"), help="stage A output folder")
    ap.add_argument("--apply", default="", help="path to selection.json (stage B)")
    ap.add_argument("--dest", default="", help="stage B crop destination (required with --apply)")
    ap.add_argument("--progress-json", default="", help="write live progress JSON here (web UI)")
    ap.add_argument("--api-pick", default="", help="URL the picker page posts its selection to")
    args = ap.parse_args()

    prog = Progress(args.progress_json)
    src = Path(args.src)
    if not src.is_dir():
        print(f"ERROR: {src} is not a directory", file=sys.stderr)
        return 1
    if args.apply:
        if not args.dest:
            print("ERROR: --dest is required with --apply", file=sys.stderr)
            return 1
        return stage_apply(Path(args.apply), src, Path(args.dest), progress=prog)
    return stage_build(src, Path(args.out), progress=prog, api_pick=args.api_pick)


if __name__ == "__main__":
    sys.exit(main())
