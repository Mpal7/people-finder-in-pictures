"""Family photo finder: local, fully offline face indexing and matching.

Modules:
  imaging - image loading (EXIF, HEIC/CR2/DNG), duplicate-map helpers
  models  - SCRFD detection + embedding ensemble (w600k_r50, AdaFace, LVFace)
  db      - SQLite schema and helpers for the resumable face index
"""
