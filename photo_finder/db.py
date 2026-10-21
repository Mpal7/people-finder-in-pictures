"""SQLite schema and helpers for the resumable face index.

Layout:
  images  - one row per canonical image (path+size+mtime identity for resumability)
  faces   - detected faces (bbox, 5 landmarks, det score) per detection config
  embeddings - L2-normalized float32 vectors per (face, model)

WAL journal; one transaction per image, so an interrupted run loses at most
the image being processed.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY,
    relpath TEXT UNIQUE NOT NULL,
    size INTEGER NOT NULL,
    mtime INTEGER NOT NULL,
    width INTEGER,
    height INTEGER,
    n_faces INTEGER,
    status TEXT NOT NULL DEFAULT 'pending',   -- ok | error
    error TEXT,
    det_config TEXT,
    dup_group INTEGER,
    indexed_at TEXT
);
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    bbox_json TEXT NOT NULL,
    kps_json TEXT NOT NULL,
    det_score REAL NOT NULL,
    UNIQUE(image_id, idx)
);
CREATE TABLE IF NOT EXISTS embeddings (
    face_id INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL,
    PRIMARY KEY (face_id, model)
);
CREATE INDEX IF NOT EXISTS idx_faces_image ON faces(image_id);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def connect(db_path: Path, create: bool = True) -> sqlite3.Connection:
    db_path = Path(db_path)
    if create:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    if not create and not db_path.exists():
        raise FileNotFoundError(db_path)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    if create:
        conn.executescript(SCHEMA)
        conn.commit()
    ensure_screenshot_columns(conn)
    return conn


def ensure_screenshot_columns(conn: sqlite3.Connection) -> None:
    """Add the screenshot-analysis columns on first use (idempotent)."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(images)")}
    added = False
    for name, typ in (("orig_w", "INTEGER"), ("orig_h", "INTEGER"), ("exif_cam", "INTEGER"),
                      ("shot_class", "INTEGER"), ("shot_reason", "TEXT"), ("shot_score", "REAL")):
        if name not in cols:
            conn.execute(f"ALTER TABLE images ADD COLUMN {name} {typ}")
            added = True
    if added:
        conn.commit()


def meta_get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def get_image(conn: sqlite3.Connection, relpath: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM images WHERE relpath=?", (relpath,)).fetchone()


def faces_with_models(conn: sqlite3.Connection, image_id: int) -> tuple[int, set[str]]:
    """(number of faces, set of models fully present for all faces)."""
    n_faces = conn.execute("SELECT COUNT(*) c FROM faces WHERE image_id=?", (image_id,)).fetchone()["c"]
    if n_faces == 0:
        return 0, set()
    models = {r["model"] for r in conn.execute(
        """SELECT DISTINCT e.model FROM embeddings e
           JOIN faces f ON f.id = e.face_id
           WHERE f.image_id=?""", (image_id,))}
    complete = {m for m in models if conn.execute(
        """SELECT COUNT(*) c FROM faces f
           LEFT JOIN embeddings e ON e.face_id=f.id AND e.model=?
           WHERE f.image_id=? AND e.face_id IS NULL""", (m, image_id)).fetchone()["c"] == 0}
    return n_faces, complete


def store_image(conn: sqlite3.Connection, relpath: str, size: int, mtime: int, width: int,
                height: int, det_config: str, dup_group: int | None,
                faces: list[dict], embeddings: dict[str, list[bytes]]) -> None:
    """Replace any previous data for this image with a fresh detection result."""
    cur = conn.cursor()
    old = get_image(conn, relpath)
    if old:
        cur.execute("DELETE FROM images WHERE id=?", (old["id"],))
    cur.execute(
        """INSERT INTO images(relpath,size,mtime,width,height,n_faces,status,det_config,dup_group,indexed_at)
           VALUES(?,?,?,?,?,?,?,?,?,datetime('now'))""",
        (relpath, size, mtime, width, height, len(faces), "ok", det_config, dup_group),
    )
    image_id = cur.lastrowid
    for i, f in enumerate(faces):
        cur.execute(
            "INSERT INTO faces(image_id,idx,bbox_json,kps_json,det_score) VALUES(?,?,?,?,?)",
            (image_id, i, json.dumps(f["bbox"]), json.dumps(f["kps"]), f["det_score"]),
        )
        face_id = cur.lastrowid
        for model, vecs in embeddings.items():
            v = vecs[i]
            cur.execute(
                "INSERT INTO embeddings(face_id,model,dim,vec) VALUES(?,?,?,?)",
                (face_id, model, len(v) // 4, v),
            )
    conn.commit()


def store_embeddings(conn: sqlite3.Connection, face_id: int, model: str, vec: bytes) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO embeddings(face_id,model,dim,vec) VALUES(?,?,?,?)",
        (face_id, model, len(vec) // 4, vec),
    )


def mark_error(conn: sqlite3.Connection, relpath: str, size: int, mtime: int, error: str) -> None:
    conn.execute(
        """INSERT INTO images(relpath,size,mtime,status,error,indexed_at)
           VALUES(?,?,?,'error',?,datetime('now'))
           ON CONFLICT(relpath) DO UPDATE SET size=excluded.size, mtime=excluded.mtime,
             status='error', error=excluded.error, indexed_at=excluded.indexed_at""",
        (relpath, size, mtime, error),
    )
    conn.commit()
