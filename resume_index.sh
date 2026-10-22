#!/usr/bin/env bash
# Linux convenience wrapper around resume_index.py. The collection path is
# never hardcoded: pass --root, or set PHOTO_ROOT.
#
#   PHOTO_ROOT=/mnt/foto/Foto ./resume_index.sh
#   ./resume_index.sh --root /mnt/foto/Foto --check
#
# Optional legacy helpers:
#   PHOTO_MOUNT        used as "$PHOTO_MOUNT/Foto" when PHOTO_ROOT is unset
#   PHOTO_DRIVE_LABEL  if set and the root is missing, try udisksctl mount
set -euo pipefail
cd "$(dirname "$0")"

if [[ -z "${PHOTO_ROOT:-}" && -n "${PHOTO_MOUNT:-}" ]]; then
  export PHOTO_ROOT="$PHOTO_MOUNT/Foto"
fi

if [[ -n "${PHOTO_DRIVE_LABEL:-}" && -n "${PHOTO_ROOT:-}" && ! -d "$PHOTO_ROOT" ]]; then
  DEV="$(blkid -L "$PHOTO_DRIVE_LABEL" || true)"
  if [[ -z "$DEV" ]]; then
    echo "drive with label '$PHOTO_DRIVE_LABEL' not found (not plugged in?)" >&2
    exit 1
  fi
  echo "drive not mounted, trying udisksctl ($DEV)..."
  udisksctl mount -b "$DEV" || {
    echo "-> could not mount $DEV automatically. Mount it manually, then rerun." >&2
    exit 1
  }
fi

if [[ -n "${PHOTO_ROOT:-}" ]] && command -v findmnt >/dev/null 2>&1; then
  OPTS="$(findmnt -no OPTIONS --target "$PHOTO_ROOT" 2>/dev/null || true)"
  if [[ -n "$OPTS" && ",$OPTS," != *",ro,"* ]]; then
    echo "note: $PHOTO_ROOT is writable; People Finder still only reads originals." >&2
  fi
fi

exec .venv/bin/python resume_index.py "$@"
