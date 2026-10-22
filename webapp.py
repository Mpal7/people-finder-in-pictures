#!/usr/bin/env python3
"""People Finder - local web UI (127.0.0.1 only, fully offline).

Run:  .venv/bin/python webapp.py            (Linux/macOS)
      .venv\\Scripts\\python.exe webapp.py  (Windows)

The UI orchestrates the same resumable CLI modules used from the terminal:
  cli.index_faces          - build / extend the shared face index
  cli.pick_reference_faces - scan a person's reference photos, pick the right face
  cli.match_person         - match a person's refs against the index
  cli.export_matches       - copy the selected matches to found/<person>/

Nothing leaves the machine; the collection is opened read-only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from photo_finder import db as fdb  # noqa: E402
from photo_finder import people as ppl  # noqa: E402
from photo_finder.imaging import IMAGE_EXT  # noqa: E402
from photo_finder.jobs import JobManager  # noqa: E402
from photo_finder.paths import expand_path, is_within, safe_join  # noqa: E402
from photo_finder.thumbs import ThumbCache  # noqa: E402

DATA = PROJECT / "data"
DB_PATH = DATA / "faces.db"
WEB = PROJECT / "web"
FOUND = PROJECT / "found"
RUNS_ROOT = PROJECT / "review"
PY = sys.executable
# No machine-specific default: set the collection root in Settings on first run.
DEFAULT_ROOT = ""

MODEL_ORDER = ["w600k", "adaface", "lvface"]


# ---------------------------------------------------------------- config ---
class Config(BaseModel):
    root: str = DEFAULT_ROOT


def load_config() -> dict:
    p = DATA / "config.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    cfg = {"root": DEFAULT_ROOT}
    DATA.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return cfg


def save_config(cfg: dict) -> None:
    (DATA / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")


def config_root() -> Path | None:
    """Resolve the configured collection root for any local/removable/share path."""
    raw = str(load_config().get("root") or "").strip()
    if not raw:
        return None
    root = expand_path(raw)
    if not root.is_absolute():
        root = PROJECT / root
    try:
        return root.resolve()
    except OSError:
        return root


# ------------------------------------------------------------------- app ---
jm = JobManager(DATA)
app = FastAPI(title="People Finder (local)")
THUMB_CACHE = ThumbCache(DATA / "thumbs_cache")

(WEB).mkdir(exist_ok=True)
ppl.PEOPLE_DIR.mkdir(parents=True, exist_ok=True)
RUNS_ROOT.mkdir(parents=True, exist_ok=True)


def index_stats() -> dict:
    out = {"exists": DB_PATH.exists(), "images_ok": 0, "images_error": 0, "faces": 0,
           "models": {}, "meta": {}, "config": {}, "db_mb": 0.0}
    if not DB_PATH.exists():
        return out
    conn = fdb.connect(DB_PATH, create=False)
    out["images_ok"] = conn.execute("SELECT COUNT(*) c FROM images WHERE status='ok'").fetchone()["c"]
    out["images_error"] = conn.execute("SELECT COUNT(*) c FROM images WHERE status='error'").fetchone()["c"]
    out["faces"] = conn.execute("SELECT COUNT(*) c FROM faces").fetchone()["c"]
    out["models"] = {r["model"]: r["c"] for r in
                     conn.execute("SELECT model, COUNT(*) c FROM embeddings GROUP BY model")}
    out["meta"] = {r["key"]: r["value"] for r in conn.execute("SELECT key,value FROM meta")}
    conn.close()
    try:
        out["config"] = json.loads(out["meta"].get("index_config") or "{}")
    except (TypeError, ValueError):
        out["config"] = {}
    out["db_mb"] = round(DB_PATH.stat().st_size / 2**20, 1)
    return out


def index_status() -> dict:
    st = index_stats()
    total = int(st["meta"].get("run_total", 0) or 0)
    done = st["images_ok"]
    finished = st["meta"].get("run_finished_at")
    running = jm.active("index") is not None
    eta = None
    avg = float(st["meta"].get("run_avg_s", 0) or 0)
    if running and total and avg:
        eta = round(avg * (total - done))
    return {**st, "running": running, "finished_at": finished, "total": total, "eta_s": eta,
            "active_job": jm.active("index")}


def person_detail(slug: str) -> dict:
    person = ppl.get(slug)
    if not person:
        raise HTTPException(404, f"no person '{slug}'")
    d = ppl.person_dir(slug)
    refs = ppl.refs_dir(person)
    refs_files = ([p for p in refs.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXT]
                  if refs.is_dir() else [])
    kps = list(refs.glob("*.kps.json")) if refs.is_dir() else []
    neg_dir = d / "refs_neg"
    neg_files = ([p for p in neg_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXT] if neg_dir.is_dir() else [])
    picker_index = d / "refs_pick" / "index.html"
    pick_sel = d / "refs_pick" / "selection.json"
    picked = 0
    if pick_sel.exists():
        try:
            sels = json.loads(pick_sel.read_text())
            picked = sum(1 for s in sels.get("selections", []) if s.get("face") is not None)
        except Exception:
            pass
    all_jobs = jm.list(200)
    jobs_by_id = {j["id"]: j for j in all_jobs}
    runs = []
    for run in person.get("runs", []):
        rd = PROJECT / run["dir"]
        sel = rd / "selection.json"
        meta = {}
        if (rd / "run_meta.json").exists():
            try:
                meta = json.loads((rd / "run_meta.json").read_text())
            except Exception:
                meta = {}
        sel_n = 0
        if sel.exists():
            try:
                sel_n = len(json.loads(sel.read_text()).get("selection", []))
            except Exception:
                pass
        job = jobs_by_id.get(run.get("job"))
        job_status = job["status"] if job else None
        review_exists = (rd / "index.html").exists()
        runs.append({**run,
                     "review_exists": review_exists,
                     "meta": meta,
                     "selection_n": sel_n,
                     "job_status": job_status,
                     "failed": bool(job_status in ("error", "stopped") and not review_exists)})
    found_count = count_found(slug)
    jobs = [j for j in all_jobs if j.get("meta", {}).get("person") == slug]
    return {"slug": slug, "name": person["name"], "refs_dir": str(refs),
            "refs_count": len(refs_files), "kps_count": len(kps),
            "neg_refs_dir": str(neg_dir), "neg_refs_count": len(neg_files),
            "picker_exists": picker_index.exists(),
            "pick_selection": {"exists": pick_sel.exists(), "picked": picked},
            "runs": runs, "found_count": found_count, "jobs": jobs[:8]}


def count_found(slug: str) -> int:
    fdir = FOUND / slug
    if not fdir.exists():
        return 0
    return sum(1 for p in fdir.iterdir() if p.is_file() and p.name != "manifest.csv")


# ---------------------------------------------------------------- models ---
class IndexReq(BaseModel):
    models: list[str] = ["w600k"]
    flip: bool = True
    det_sizes: str = "1280"


class PersonReq(BaseModel):
    name: str
    refs_dir: str | None = None


class ScanReq(BaseModel):
    src: str | None = None


class MatchReq(BaseModel):
    top_faces: int = 1000
    fusion: str = "mean"
    min_z: float | None = None
    weights: str = ""
    neg_weight: float = 0.5
    group_similar: bool = True


class ExportReq(BaseModel):
    ts: str


class ResetReq(BaseModel):
    confirm: bool = False


# ------------------------------------------------------------------ API ----
@app.get("/api/state")
def api_state():
    cfg = load_config()
    root = config_root()
    st = index_stats()
    total_faces = max(st["faces"], 1)
    coverage = {m: {"count": st["models"].get(m, 0),
                    "pct": round(100 * st["models"].get(m, 0) / total_faces, 1)}
                for m in MODEL_ORDER}
    return {
        "root": cfg.get("root") or "",
        "root_ok": bool(root and root.is_dir()),
        "index": index_status(),
        "coverage": coverage,
        "screenshots": api_shots_status(),
        "people": [{"slug": p["slug"], "name": p["name"],
                    "runs": len(p.get("runs", []))} for p in ppl.list_all()],
        "jobs": jm.list(20),
        "found_total": sum(count_found(p["slug"]) for p in ppl.list_all()),
        "thumbs_cache": THUMB_CACHE.stats(),
    }


@app.post("/api/config")
def api_config(body: Config):
    raw = body.root.strip()
    if raw:
        root = expand_path(raw)
        if not root.is_absolute():
            root = PROJECT / root
        try:
            raw = str(root.resolve())
        except OSError:
            raw = str(root)
    save_config({"root": raw})
    return {"ok": True, "root": raw, "root_ok": bool(raw and Path(raw).is_dir())}


@app.post("/api/index/start")
def api_index_start(body: IndexReq):
    if jm.active("index"):
        raise HTTPException(409, "an index run is already active")
    root = config_root()
    if not root or not root.is_dir():
        raise HTTPException(400, "collection root not set or not found - set it in Settings")
    models = [m for m in body.models if m in MODEL_ORDER]
    if not models:
        raise HTTPException(400, "no models selected")
    cmd = [PY, "-m", "cli.index_faces", str(root), "--db", str(DB_PATH),
           "--models", ",".join(models), "--det-sizes", body.det_sizes,
           "--progress-json", "{progress}"]
    if not body.flip:
        cmd.append("--no-flip")
    job = jm.start("index", cmd, f"Index faces ({'+'.join(models)})")
    return {"ok": True, "job": job["id"]}


@app.post("/api/index/stop")
def api_index_stop():
    job = jm.active("index")
    if not job:
        raise HTTPException(404, "no active index run")
    jm.stop(job["id"])
    return {"ok": True, "job": job["id"]}


@app.post("/api/index/reset")
def api_index_reset(body: ResetReq):
    if not body.confirm:
        raise HTTPException(400, "confirmation required")
    for kind in ("index", "match", "shots"):
        if jm.active(kind):
            raise HTTPException(409, f"stop the active {kind} job first")
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(DB_PATH) + suffix)
        if p.exists():
            p.unlink()
    return {"ok": True}


@app.post("/api/people")
def api_people_create(body: PersonReq):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "name required")
    p = ppl.create(name, body.refs_dir)
    return {"ok": True, "person": p}


@app.get("/api/people/{slug}")
def api_person(slug: str):
    return person_detail(slug)


@app.delete("/api/people/{slug}")
def api_person_delete(slug: str):
    person = ppl.get(slug)
    if not person:
        raise HTTPException(404, "no such person")
    active = [j for j in jm.list(200)
              if j.get("status") in ("running", "stopping")
              and j.get("meta", {}).get("person") == slug]
    if active:
        raise HTTPException(409, "stop this person's active jobs first")
    ppl.delete(slug)
    return {"ok": True, "slug": slug}


@app.delete("/api/people/{slug}/runs/{ts}")
def api_run_delete(slug: str, ts: str):
    person = ppl.get(slug)
    if not person:
        raise HTTPException(404, "no such person")
    run = next((r for r in person.get("runs", []) if r.get("ts") == ts), None)
    if not run:
        raise HTTPException(404, "no such run")
    job = next((j for j in jm.list(200) if j["id"] == run.get("job")), None)
    if job and job.get("status") in ("running", "stopping"):
        raise HTTPException(409, "stop the match job first")
    if not ppl.delete_run(slug, ts):
        raise HTTPException(404, "no such run")
    return {"ok": True, "ts": ts}


@app.post("/api/people/{slug}/refs/scan")
def api_refs_scan(slug: str, body: ScanReq):
    person = ppl.get(slug)
    if not person:
        raise HTTPException(404, "no such person")
    if jm.active() and jm.active()["kind"] in ("scan",):
        raise HTTPException(409, "a reference scan is already active")
    d = ppl.person_dir(slug)
    src = expand_path(body.src) if body.src else d / "refs_src"
    if not src.is_dir():
        raise HTTPException(400, f"folder not found: {src}")
    images = [p for p in src.iterdir()
              if p.is_file() and p.suffix.lower() in IMAGE_EXT]
    if not images:
        raise HTTPException(400, f"no images in {src} - drop reference photos there first")
    cmd = [PY, "-m", "cli.pick_reference_faces", "--src", str(src),
           "--out", str(d / "refs_pick"), "--progress-json", "{progress}",
           "--api-pick", f"/api/people/{slug}/refs/pick"]
    job = jm.start("scan", cmd, f"Scan references ({len(images)} photos)", meta={"person": slug})
    return {"ok": True, "job": job["id"], "images": len(images)}


@app.post("/api/people/{slug}/refs/pick")
def api_refs_pick(slug: str, payload: dict):
    person = ppl.get(slug)
    if not person:
        raise HTTPException(404, "no such person")
    d = ppl.person_dir(slug)
    (d / "refs_pick").mkdir(parents=True, exist_ok=True)
    picked = sum(1 for s in payload.get("selections", []) if s.get("face") is not None)
    skipped = sum(1 for s in payload.get("selections", []) if s.get("skip"))
    (d / "refs_pick" / "selection.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return {"ok": True, "n": picked + skipped, "picked": picked, "skipped": skipped}


@app.post("/api/people/{slug}/refs/apply")
def api_refs_apply(slug: str):
    person = ppl.get(slug)
    if not person:
        raise HTTPException(404, "no such person")
    d = ppl.person_dir(slug)
    sel = d / "refs_pick" / "selection.json"
    if not sel.exists():
        raise HTTPException(400, "no picks saved yet - finish the picker page first")
    src = d / "refs_src"
    if not src.is_dir():
        raise HTTPException(400, f"references folder missing: {src}")
    cmd = [PY, "-m", "cli.pick_reference_faces", "--apply", str(sel),
           "--src", str(src), "--dest", str(ppl.refs_dir(person)), "--progress-json", "{progress}"]
    job = jm.start("apply", cmd, "Apply reference picks (crops + landmarks)", meta={"person": slug})
    return {"ok": True, "job": job["id"]}


@app.post("/api/people/{slug}/match")
def api_match(slug: str, body: MatchReq):
    person = ppl.get(slug)
    if not person:
        raise HTTPException(404, "no such person")
    if jm.active("match"):
        raise HTTPException(409, "a match run is already active")
    if body.fusion not in ("mean", "max"):
        raise HTTPException(400, "fusion must be 'mean' or 'max'")
    if body.neg_weight < 0:
        raise HTTPException(400, "neg_weight must be >= 0")
    root = config_root()
    if not root or not root.is_dir():
        raise HTTPException(400, "collection root not set or not found")
    refs = ppl.refs_dir(person)
    if not refs.is_dir() or not any(refs.glob("*.jpg")):
        raise HTTPException(400, "no reference crops yet - run the picker/apply first")
    if not DB_PATH.exists():
        raise HTTPException(400, "no face index yet - run indexing first")
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = ppl.person_dir(slug) / "runs" / ts
    suffix = 2
    while run_dir.exists():
        ts = f"{time.strftime('%Y%m%d-%H%M%S')}-{suffix}"
        run_dir = ppl.person_dir(slug) / "runs" / ts
        suffix += 1
    review_dir = run_dir / "review"
    cmd = [PY, "-m", "cli.match_person", "--db", str(DB_PATH), "--root", str(root),
           "--refs", str(refs), "--out", str(review_dir), "--top-faces", str(body.top_faces),
           "--fusion", body.fusion, "--neg-weight", str(body.neg_weight),
           "--progress-json", "{progress}",
           "--api-review", f"/api/people/{slug}/review/{ts}/select"]
    if body.min_z is not None:
        cmd += ["--min-z", str(body.min_z)]
    if body.weights.strip():
        cmd += ["--weights", body.weights.strip()]
    if not body.group_similar:
        cmd.append("--no-near-dup")
    neg_dir = ppl.person_dir(slug) / "refs_neg"
    neg_count = 0
    if neg_dir.is_dir():
        neg_count = sum(1 for p in neg_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in IMAGE_EXT)
    if neg_count:
        cmd += ["--neg-refs", str(neg_dir)]
    job = jm.start("match", cmd, f"Match '{person['name']}' (top {body.top_faces})",
                   meta={"person": slug})
    ppl.add_run(slug, {"ts": ts, "dir": str(run_dir.relative_to(PROJECT)),
                       "top_faces": body.top_faces, "fusion": body.fusion,
                       "min_z": body.min_z, "weights": body.weights,
                       "neg_weight": body.neg_weight, "negatives": neg_count,
                       "group_similar": body.group_similar, "job": job["id"],
                       "review_url": f"/files/people/{slug}/runs/{ts}/review/index.html"})
    return {"ok": True, "job": job["id"], "ts": ts}


@app.post("/api/people/{slug}/review/{ts}/select")
def api_review_select(slug: str, ts: str, payload: dict):
    person = ppl.get(slug)
    if not person:
        raise HTTPException(404, "no such person")
    run_dir = ppl.person_dir(slug) / "runs" / ts
    if not run_dir.is_dir():
        raise HTTPException(404, "no such run")
    (run_dir / "selection.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    scores = [float(s.get("score", 0)) for s in payload.get("selection", [])]
    return {"ok": True, "n": len(scores), "min_score": min(scores) if scores else None,
            "max_score": max(scores) if scores else None}


@app.post("/api/people/{slug}/export")
def api_export(slug: str, body: ExportReq):
    person = ppl.get(slug)
    if not person:
        raise HTTPException(404, "no such person")
    active = jm.active("export")
    if active and active.get("meta", {}).get("person") == slug:
        raise HTTPException(409, "a copy job for this person is already active")
    root = config_root()
    if not root or not root.is_dir():
        raise HTTPException(400, "collection root not set or not found")
    run_dir = ppl.person_dir(slug) / "runs" / body.ts
    sel = run_dir / "selection.json"
    if not sel.exists():
        raise HTTPException(400, "no selection saved for this run yet")
    dest = FOUND / slug
    cmd = [PY, "-m", "cli.export_matches", "--root", str(root),
           "--selection", str(sel), "--dest", str(dest), "--flat",
           "--progress-json", "{progress}"]
    job = jm.start("export", cmd, f"Copy selection for '{person['name']}'",
                   meta={"person": slug})
    return {"ok": True, "job": job["id"], "dest": str(dest),
            "count": count_found(slug)}


@app.get("/api/jobs")
def api_jobs():
    return {"jobs": jm.list(50)}


@app.post("/api/jobs/{jid}/stop")
def api_job_stop(jid: str):
    job = jm.stop(jid)
    return {"ok": bool(job)}


# ------------------------------------------------------------ any-people --
class FacesExportReq(BaseModel):
    min_faces: int = 1
    min_det: float = 0.5
    min_face_pct: float = 0.0
    folder: str = ""
    sort: str = "faces"
    shots: str = "all"
    dest: str = "found/people"


def _faces_where(min_faces, min_det, min_face_pct, folder, shots="all"):
    conds, params = [], []
    if folder:
        conds.append("i.relpath LIKE ?")
        params.append(f"%{folder}%")
    if shots == "hide":
        conds.append("COALESCE(i.shot_class,0) < 2")
    elif shots == "hide2":
        conds.append("COALESCE(i.shot_class,0) = 0")
    elif shots == "only":
        conds.append("COALESCE(i.shot_class,0) >= 2")
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    having = ("HAVING i.n_faces >= ? AND MAX(f.det_score) >= ? AND "
              "MAX((json_extract(f.bbox_json,'$[3]') - json_extract(f.bbox_json,'$[1]')) "
              "* 1.0 / NULLIF(i.height,0)) >= ?")
    return where, params, having, [int(min_faces), float(min_det), float(min_face_pct)]


def _faces_list(conn, min_faces, min_det, min_face_pct, folder, sort, limit=None, offset=0,
                shots="all"):
    where, params, having, hp = _faces_where(min_faces, min_det, min_face_pct, folder, shots)
    # A stable tie-breaker is required for LIMIT/OFFSET pagination: without it
    # rows with equal scores can repeat or go missing between pages.
    order = {"faces": "i.n_faces DESC, mdet DESC, i.relpath",
             "recent": "i.mtime DESC, i.relpath",
             "path": "i.relpath"}.get(sort, "i.n_faces DESC, mdet DESC, i.relpath")
    base = f"FROM images i JOIN faces f ON f.image_id = i.id {where} GROUP BY i.id {having}"
    total = conn.execute(f"SELECT COUNT(*) c FROM (SELECT i.id {base})", params + hp).fetchone()["c"]
    agg = conn.execute(
        f"SELECT COALESCE(SUM(sz),0) sb, COALESCE(SUM(nf),0) nf FROM (SELECT i.size sz, i.n_faces nf {base})",
        params + hp).fetchone()
    sql = (f"SELECT i.relpath p, i.n_faces nf, i.size sz, i.mtime mt, "
           f"i.shot_class sc, i.shot_reason sr, "
           f"MAX(f.det_score) mdet, "
           f"MAX((json_extract(f.bbox_json,'$[3]') - json_extract(f.bbox_json,'$[1]')) "
           f"* 1.0 / NULLIF(i.height,0)) mfr "
           f"{base} ORDER BY {order}")
    if limit is not None:
        rows = conn.execute(sql + " LIMIT ? OFFSET ?",
                            params + hp + [int(limit), int(offset)]).fetchall()
    else:
        rows = conn.execute(sql, params + hp).fetchall()
    return total, agg["sb"], agg["nf"], rows


def _filters(min_faces, min_det, min_face_pct, folder, sort, shots="all",
             limit=None, offset=0):
    if not DB_PATH.exists():
        raise HTTPException(400, "no face index yet - run indexing first")
    conn = fdb.connect(DB_PATH, create=False)
    try:
        return _faces_list(conn, min_faces, min_det, min_face_pct / 100.0, folder, sort,
                           limit=limit, offset=offset, shots=shots)
    finally:
        conn.close()


def _safe_rel(path: str) -> Path:
    root = config_root()
    if not root:
        raise HTTPException(400, "collection root not set")
    try:
        full = safe_join(root, path)
    except ValueError:
        raise HTTPException(400, "bad path")
    if not full.is_file():
        raise HTTPException(404, "not found")
    return full


@app.get("/api/faces/images")
def api_faces_images(min_faces: int = 1, min_det: float = 0.5, min_face_pct: float = 0.0,
                     folder: str = "", sort: str = "faces", shots: str = "all",
                     limit: int = 60, offset: int = 0):
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    total, bytes_total, faces_total, rows = _filters(
        min_faces, min_det, min_face_pct, folder, sort, shots, limit=limit, offset=offset)
    items = [{"path": r["p"], "n_faces": r["nf"], "size": r["sz"], "mtime": r["mt"],
              "max_det": round(r["mdet"], 3), "max_face_pct": round(100 * (r["mfr"] or 0), 1),
              "shot_class": r["sc"] or 0, "shot_reason": r["sr"] or ""}
             for r in rows]
    return {"total": total, "bytes_total": bytes_total, "faces_total": faces_total,
            "items": items, "offset": offset, "limit": limit}


@app.get("/api/screenshots/status")
def api_shots_status():
    out = {"total": 0, "analyzed": 0, "possible": 0, "screenshots": 0,
           "running": jm.active("shots") is not None}
    if DB_PATH.exists():
        conn = fdb.connect(DB_PATH, create=False)
        r = conn.execute("""SELECT COUNT(*) t,
                 SUM(CASE WHEN shot_class IS NOT NULL THEN 1 ELSE 0 END) a,
                 SUM(CASE WHEN shot_class=1 THEN 1 ELSE 0 END) p,
                 SUM(CASE WHEN shot_class=2 THEN 1 ELSE 0 END) s
                 FROM images WHERE status='ok'""").fetchone()
        conn.close()
        out.update(total=r["t"], analyzed=r["a"] or 0, possible=r["p"] or 0, screenshots=r["s"] or 0)
    return out


class ShotsReq(BaseModel):
    visual: bool = False


@app.post("/api/screenshots/analyze")
def api_shots_analyze(body: ShotsReq):
    if jm.active("shots"):
        raise HTTPException(409, "a screenshot analysis is already running")
    root = config_root()
    if not root or not root.is_dir():
        raise HTTPException(400, "collection root not set or not found")
    cmd = [PY, "-m", "cli.detect_screenshots", str(root), "--db", str(DB_PATH),
           "--progress-json", "{progress}"]
    if body.visual:
        cmd.append("--visual")
    title = "Refine possible screenshots (visual)" if body.visual else "Scan screenshots (header pass)"
    job = jm.start("shots", cmd, title)
    return {"ok": True, "job": job["id"]}


@app.get("/api/faces/thumb")
def api_faces_thumb(path: str, px: int = 320):
    rel = Path(path)
    full = _safe_rel(path)
    px = max(64, min(px, 1600))
    return FileResponse(THUMB_CACHE.path_for(full, rel, px), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=86400"})


@app.get("/api/faces/view")
def api_faces_view(path: str):
    """Original for browser-friendly formats, large 1200px render otherwise."""
    rel = Path(path)
    full = _safe_rel(path)
    if full.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"):
        return FileResponse(full)
    return FileResponse(THUMB_CACHE.path_for(full, rel, 1200), media_type="image/jpeg")


@app.get("/api/faces/list.csv")
def api_faces_csv(min_faces: int = 1, min_det: float = 0.5, min_face_pct: float = 0.0,
                  folder: str = "", sort: str = "faces", shots: str = "all"):
    total, bytes_total, faces_total, rows = _filters(min_faces, min_det, min_face_pct, folder, sort, shots)
    lines = ["relpath,n_faces,size_bytes,max_det_score,max_face_pct,shoot_date"]
    for r in rows:
        lines.append(f'"{r["p"]}",{r["nf"]},{r["sz"]},{r["mdet"]:.3f},'
                     f'{100 * (r["mfr"] or 0):.1f},{time.strftime("%Y-%m-%d", time.localtime(r["mt"]))}')
    return Response("\n".join(lines) + "\n", media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=people_photos.csv"})


@app.post("/api/faces/export")
def api_faces_export(body: FacesExportReq):
    root_path = config_root()
    if not root_path or not root_path.is_dir():
        raise HTTPException(400, "collection root not set or not found")
    raw = (body.dest or "").strip()
    if not raw:
        raise HTTPException(400, "destination folder required")
    dest_path = expand_path(raw)
    if not dest_path.is_absolute():
        dest_path = PROJECT / dest_path
    dest_path = dest_path.resolve()
    if root_path and (is_within(dest_path, root_path) or is_within(root_path, dest_path)):
        raise HTTPException(400, "destination must be OUTSIDE the collection (originals stay read-only)")
    if dest_path.exists() and not dest_path.is_dir():
        raise HTTPException(400, "destination exists and is not a directory")
    total, bytes_total, faces_total, rows = _filters(body.min_faces, body.min_det,
                                                     body.min_face_pct, body.folder, body.sort,
                                                     body.shots)
    if total == 0:
        raise HTTPException(400, "no photos match the filters")
    exp_dir = DATA / "exports"
    exp_dir.mkdir(exist_ok=True)
    sel = exp_dir / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    sel.write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "selection": [{"path": r["p"], "score": r["nf"]} for r in rows],
    }, indent=1), encoding="utf-8")
    cmd = [PY, "-m", "cli.export_matches", "--root", str(root_path),
           "--selection", str(sel), "--dest", str(dest_path), "--flat",
           "--progress-json", "{progress}"]
    job = jm.start("export", cmd, f"Copy {total} photos to {dest_path}",
                   meta={"dest": str(dest_path), "count": total})
    return {"ok": True, "job": job["id"], "count": total, "bytes": bytes_total,
            "dest": str(dest_path)}


@app.post("/api/cache/clear")
def api_cache_clear():
    removed = THUMB_CACHE.clear()
    return {"ok": True, "removed": removed}


# Dynamic file route (not a startup mount): review/ and people/ files created by
# CLI runs after the server started must be served too. Only these two roots are
# exposed, and traversal outside them is rejected.
FILE_ROOTS = tuple(p.resolve() for p in (ppl.PEOPLE_DIR, RUNS_ROOT))


@app.get("/files/{path:path}")
def files(path: str):
    try:
        full = safe_join(PROJECT, path).resolve()
    except ValueError:
        raise HTTPException(404, "not found")
    if not any(is_within(full, root) for root in FILE_ROOTS):
        raise HTTPException(404, "not found")
    if not full.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(full)


@app.get("/")
def root():
    return FileResponse(WEB / "index.html")


if (WEB / "app.js").exists():
    app.mount("/assets", StaticFiles(directory=str(WEB)), name="assets")


def main() -> int:
    import argparse

    import uvicorn
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("WARNING: binding to a non-loopback address exposes your photo collection "
              "and the unauthenticated API to your network. Keep the default for local use.")
    print(f"People Finder UI -> http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
