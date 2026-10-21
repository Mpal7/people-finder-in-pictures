"""Progress file helper for long-running worker scripts.

Worker scripts (index_faces.py, match_person.py, pick_reference_faces.py)
write a small JSON file periodically; the web app polls it to show live
progress. The file is tiny and rewritten atomically (tmp + rename).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


class Progress:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None

    def write(self, **fields) -> None:
        if self.path is None:
            return
        fields.setdefault("ts", time.time())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(fields), encoding="utf-8")
        os.replace(tmp, self.path)

    def done(self, **fields) -> None:
        self.write(finished=True, error=None, **fields)

    def fail(self, error: str, **fields) -> None:
        self.write(finished=True, error=str(error)[:500], **fields)
