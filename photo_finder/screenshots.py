"""Phone-screenshot classification for indexed images.

Phase A (header-only, deterministic):
  class 0 = not a screenshot, 1 = possible, 2 = screenshot
  Signals: PNG + no camera EXIF + exact phone screen resolution, filename
  patterns, scrolled/tall aspect ratio, app-export names, WhatsApp origin.

Phase B (visual, optional): status-bar band scoring on small thumbnails,
used to reclassify the "possible" group (visual_score()).
"""
from __future__ import annotations

from pathlib import Path

# Common phone/tablet screen resolutions (± tolerance), portrait or landscape
SCREEN_RES = [
    (750, 1334), (828, 1792), (1125, 2436), (1170, 2532), (1179, 2556),
    (1242, 2208), (1242, 2688), (1284, 2778), (1290, 2796), (1320, 2868),
    (1206, 2622), (1080, 1920), (1080, 2340), (1080, 2400), (1080, 2412),
    (1440, 2560), (1440, 2960), (1440, 3040), (1440, 3120), (1440, 3200),
    (1536, 2048), (1620, 2160), (1668, 2388), (2048, 2732),
]

RAW_EXT = {".cr2", ".crw", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf",
           ".pef", ".srw", ".sr2", ".srf"}


def is_screen_res(w: int, h: int, tol: int = 6):
    for sw, sh in SCREEN_RES:
        if (abs(w - sw) <= tol and abs(h - sh) <= tol) or \
           (abs(w - sh) <= tol and abs(h - sw) <= tol):
            return (sw, sh)
    return None


def classify_header(path: Path) -> dict:
    """Phase A classification from file header only (fast, no full decode)."""
    ext = path.suffix.lower()
    name = path.name.lower()
    out = {"orig_w": None, "orig_h": None, "exif_cam": None,
           "cls": 0, "reason": "", "score": 0.0}

    if ext in RAW_EXT:
        return {**out, "exif_cam": 1, "reason": "raw-camera", "score": 0.01}

    try:
        from PIL import Image
        if ext in (".heic", ".heif"):
            import pillow_heif
            pillow_heif.register_heif_opener()
        with Image.open(path) as im:
            w, h = im.size
            ex = im.getexif()
            has_cam = any(ex.get(k) for k in (271, 272, 42036))  # Make/Model/LensModel
    except Exception as e:
        return {**out, "reason": f"unreadable:{type(e).__name__}", "score": 0.0}

    out.update(orig_w=w, orig_h=h, exif_cam=int(bool(has_cam)))
    ratio = max(w, h) / max(1, min(w, h))
    screen = is_screen_res(w, h)

    if "screenshot" in name or "screen shot" in name:
        return {**out, "cls": 2, "reason": "filename", "score": 0.97}

    if not has_cam:
        if ext == ".png":
            if screen:
                return {**out, "cls": 2, "reason": "png+screen-res", "score": 0.99}
            if ratio >= 2.0:
                return {**out, "cls": 2, "reason": "png+tall(scrolled)", "score": 0.90}
            if "snapshot" in name or "reactnative" in name:
                return {**out, "cls": 2, "reason": "app-export-name", "score": 0.90}
            if 1.55 <= ratio <= 2.45:
                return {**out, "cls": 1, "reason": "png-noexif-phone-ratio", "score": 0.55}
            return {**out, "cls": 0, "reason": "png-export(no-exif)", "score": 0.15}
        # non-PNG: a no-EXIF image with a smartphone aspect ratio (9:18.5-9:21)
        # cannot come from a camera (4:3 / 3:2 / 16:9) -> screenshot, also when
        # resized (e.g. by WhatsApp to 1600px)
        if 2.04 <= ratio <= 2.34:
            return {**out, "cls": 2, "reason": "noexif+screen-aspect", "score": 0.90}
        if screen:
            return {**out, "cls": 1, "reason": "noexif+screen-res", "score": 0.65}
        if 2.0 <= ratio <= 3.2:
            return {**out, "cls": 1, "reason": "noexif+tall", "score": 0.55}
        return {**out, "cls": 0, "reason": "no-exif(other)", "score": 0.25}

    return {**out, "cls": 0, "reason": "camera-exif", "score": 0.02}


# ----------------------------------------------------------------- phase B --
VISUAL_T = 0.90       # score >= -> promote to screenshot (0 FP on 300 labeled photos)
VISUAL_T_LOW = 0.25   # score <= -> demote to not-screenshot


def visual_score(path: Path) -> float:
    """Status-bar / UI-likeness score in [0, 1] from a 320px thumbnail.

    Discriminative signals (tuned against labeled groups from the collection):
      - the top band (status bar zone, ~4.5% of height) is mostly flat
      - 1..6 small high-contrast glyph clusters inside that flat band, located
        in the left/right thirds (time, wifi, battery)
      - a crisp full-width horizontal edge just below (bar boundary)
      - a mostly uniform thin strip at the bottom (home indicator / nav bar)
    """
    import cv2
    import numpy as np

    from photo_finder.imaging import load_thumb

    img = load_thumb(path, 400, width_first=True)
    a = np.asarray(img, dtype=np.uint8)
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    H, W = g.shape
    if H < 80 or W < 120:
        return 0.0

    th = max(8, int(H * 0.045))
    top = g[:th, :]
    uni = float((np.abs(top - np.median(top)) < 20).mean())

    edges = cv2.Canny(top.astype(np.uint8), 60, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)  # merge letters -> word clusters
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats((edges > 0).astype(np.uint8), 8)
    glyphs_lr = 0
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if 6 <= area <= 0.10 * th * W and 0.30 * th <= h <= 1.3 * th and w <= 0.5 * W:
            if x + w <= 0.55 * W or x >= 0.55 * W:   # left (time) or right (battery) zone
                glyphs_lr += 1

    gy = np.abs(np.diff(g, axis=0))
    rowsharp = (gy > 30).mean(axis=1)
    ui_edge = bool((rowsharp[: max(10, int(H * 0.15))] > 0.5).any())

    bh = max(4, int(H * 0.03))
    bot = g[-bh:, :]
    bot_uni = float((np.abs(bot - np.median(bot)) < 20).mean())

    s = 0.0
    s += 0.35 if uni >= 0.85 else (0.20 if uni >= 0.70 else 0.0)
    if 1 <= glyphs_lr <= 8:
        s += 0.30
    if ui_edge:
        s += 0.20
    if bot_uni >= 0.90:
        s += 0.15
    return round(min(1.0, s), 3)
