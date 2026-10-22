"""Command-line entry points.

Run them as modules from the repository root, for example:

  python -m cli.index_faces "D:/Photos" --models w600k
  python -m cli.match_person --root "D:/Photos" --refs people/alice/refs
  python -m cli.resume_index --root "D:/Photos"

The reusable implementation lives in :mod:`photo_finder`; the web UI launches
these same modules so CLI and UI stay on one code path.
"""
