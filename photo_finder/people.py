"""People profiles: one directory per person + a small JSON registry.

Layout (all under <project>/people/<slug>/):
  refs_src/   - drop your source photos of this person here
  refs_pick/  - picker page + selection.json (which face is hers/his)
  refs/       - final face crops + .kps.json landmark sidecars
  refs_neg/   - optional photos of look-alikes to suppress during matching
  runs/<ts>/  - per-match run: review/, scores_all.csv, selection.json

Registry: <project>/data/people.json
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PEOPLE_DIR = PROJECT / "people"
REGISTRY = PROJECT / "data" / "people.json"


def _load() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {"people": []}


def _save(data: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(data, indent=1), encoding="utf-8")


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "person"


def person_dir(slug: str) -> Path:
    return PEOPLE_DIR / slug


def refs_dir(person: dict) -> Path:
    """Reference crops directory (absolute)."""
    r = person.get("refs_dir")
    if r:
        p = Path(r)
        return p if p.is_absolute() else PROJECT / p
    return person_dir(person["slug"]) / "refs"


def create(name: str, refs_dir_override: str | None = None) -> dict:
    data = _load()
    base = slugify(name)
    slug, n = base, 2
    while any(p["slug"] == slug for p in data["people"]):
        slug, n = f"{base}-{n}", n + 1
    person = {"slug": slug, "name": name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": []}
    if refs_dir_override:
        person["refs_dir"] = refs_dir_override
    d = person_dir(slug)
    for sub in ("refs_src", "refs_pick", "refs", "refs_neg", "runs"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    data["people"].append(person)
    _save(data)
    return person


def get(slug: str) -> dict | None:
    for p in _load()["people"]:
        if p["slug"] == slug:
            return p
    return None


def list_all() -> list[dict]:
    return _load()["people"]


def update(slug: str, **fields) -> dict | None:
    data = _load()
    for p in data["people"]:
        if p["slug"] == slug:
            p.update(fields)
            _save(data)
            return p
    return None


def add_run(slug: str, run: dict) -> None:
    data = _load()
    for p in data["people"]:
        if p["slug"] == slug:
            p.setdefault("runs", []).append(run)
            _save(data)
            return


def delete_run(slug: str, ts: str) -> dict | None:
    """Delete one run directory and its registry entry. Leaves found/ copies."""
    data = _load()
    for p in data["people"]:
        if p["slug"] != slug:
            continue
        runs = p.get("runs", [])
        target = next((r for r in runs if r.get("ts") == ts), None)
        if target is None:
            return None
        run_dir = (PROJECT / target["dir"]).resolve()
        base = person_dir(slug).resolve()
        if run_dir != base and base not in run_dir.parents:
            raise ValueError("run path is outside the person directory")
        if run_dir.is_dir():
            shutil.rmtree(run_dir)
        p["runs"] = [r for r in runs if r.get("ts") != ts]
        _save(data)
        return target
    return None


def delete(slug: str) -> dict | None:
    """Delete a person's profile directory and registry entry.

    ``found/<slug>/`` exports are deliberately kept: they are user-visible
    copies, not application state. A custom external ``refs_dir`` is also left
    alone; only ``people/<slug>/`` is removed.
    """
    data = _load()
    target = next((p for p in data["people"] if p["slug"] == slug), None)
    if target is None:
        return None
    d = person_dir(slug).resolve()
    people_root = PEOPLE_DIR.resolve()
    if d.exists() and d != people_root and people_root in d.parents:
        shutil.rmtree(d)
    data["people"] = [p for p in data["people"] if p["slug"] != slug]
    _save(data)
    return target
