"""People profiles: one directory per person + a small JSON registry.

Layout (all under <project>/people/<slug>/):
  refs_src/   - drop your source photos of this person here
  refs_pick/  - picker page + selection.json (which face is hers/his)
  refs/       - final face crops + .kps.json landmark sidecars
  runs/<ts>/  - per-match run: review/, scores_all.csv, selection.json

Registry: <project>/data/people.json
"""
from __future__ import annotations

import json
import re
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
    for sub in ("refs_src", "refs_pick", "refs", "runs"):
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
