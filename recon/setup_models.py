#!/usr/bin/env python3
"""One-time model downloads for People Finder.

This is the ONLY script in the project that accesses the network (besides pip).
It fetches, once:
  - InsightFace buffalo_l pack (SCRFD-10G det_10g.onnx + w600k_r50.onnx, ~326 MB)
  - AdaFace IR-101 ONNX (~249 MB)
  - LVFace-B Glint360K ONNX (~434 MB)

Everything is stored under <project>/models/ and reused offline afterwards.
No photo data is ever uploaded.
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
MODELS = PROJECT / "models"

BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
ADA_URL = "https://github.com/yakhyo/adaface-onnx/releases/download/weights/adaface_ir_101.onnx"
LVFACE_URL = "https://huggingface.co/bytedance-research/LVFace/resolve/main/LVFace-B_Glint360K/LVFace-B_Glint360K.onnx"


def human(n: float) -> str:
    return f"{n / 2**20:.1f} MB"


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"exists: {dest} ({human(dest.stat().st_size)})")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        next_report = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if done >= next_report:
                rate = done / (time.time() - t0 + 1e-9) / 2**20
                print(f"  {done / 2**20:.0f}/{total / 2**20:.0f} MB  ({rate:.1f} MB/s)", flush=True)
                next_report = done + (64 << 20)
    os.replace(tmp, dest)
    print(f"saved: {dest} ({human(dest.stat().st_size)})")


def ensure_buffalo(insight_root: Path) -> Path:
    """Download and unzip buffalo_l with urllib.

    InsightFace's own downloader uses requests/certifi. On Windows, especially
    behind a corporate TLS proxy, the OS certificate store (used by urllib)
    is often the only one that verifies. Doing the download here keeps setup
    working on both platforms.
    """
    models_dir = insight_root / "models"
    model_dir = models_dir / "buffalo_l"
    required = [model_dir / "det_10g.onnx", model_dir / "w600k_r50.onnx"]
    if all(p.exists() and p.stat().st_size > 0 for p in required):
        print(f"exists: {model_dir}")
        return model_dir

    zip_path = models_dir / "buffalo_l.zip"
    fetch(BUFFALO_URL, zip_path)
    print(f"extracting {zip_path} ...")
    model_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(model_dir)
    zip_path.unlink(missing_ok=True)
    missing = [p.name for p in required if not (p.exists() and p.stat().st_size > 0)]
    if missing:
        raise RuntimeError(f"buffalo_l archive is missing: {', '.join(missing)}")
    return model_dir


def main() -> int:
    # 1) InsightFace buffalo_l (download, then load once to verify ONNX)
    from insightface.app import FaceAnalysis

    insight_root = MODELS / "insightface"
    model_dir = ensure_buffalo(insight_root)
    app = FaceAnalysis(
        name="buffalo_l",
        root=str(insight_root),
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    print(f"buffalo_l ready at {Path(app.model_dir)}")
    for f in sorted(model_dir.glob("*.onnx")):
        print(f"  {f.name:24} {human(f.stat().st_size)}")

    # 2) AdaFace IR-101
    fetch(ADA_URL, MODELS / "adaface" / "adaface_ir_101.onnx")

    # 3) LVFace-B
    fetch(LVFACE_URL, MODELS / "lvface" / "LVFace-B_Glint360K.onnx")

    print("\nAll models present. The pipeline can now run fully offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
