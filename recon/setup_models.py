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
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
MODELS = PROJECT / "models"

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
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
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


def main() -> int:
    # 1) InsightFace buffalo_l (downloads + verifies the pack via the library)
    from insightface.app import FaceAnalysis

    insight_root = str(MODELS / "insightface")
    app = FaceAnalysis(
        name="buffalo_l",
        root=insight_root,
        allowed_modules=["detection", "recognition"],
    )
    model_dir = Path(app.model_dir)
    print(f"buffalo_l ready at {model_dir}")
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
