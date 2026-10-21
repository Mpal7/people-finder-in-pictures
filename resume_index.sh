#!/usr/bin/env bash
# Resume the face index after a reboot / suspension. Safe to run repeatedly:
# images already in the DB are skipped and processing continues where it stopped.
#
#   ./resume_index.sh          # check + launch detached (sleep/hibernation inhibited)
#   ./resume_index.sh --check  # only run the checks, do not launch
#
# Configure the mount with environment variables:
#   PHOTO_MOUNT        mount point of the photo drive          (default: /mnt/foto)
#   PHOTO_ROOT         collection folder under that mount      (default: $PHOTO_MOUNT/Foto)
#   PHOTO_DRIVE_LABEL  optional filesystem label; when set and the drive is not
#                      mounted, the script tries `udisksctl mount` via blkid
#
# Example:
#   PHOTO_MOUNT=/mnt/foto PHOTO_ROOT=/mnt/foto/Foto ./resume_index.sh
set -euo pipefail
cd "$(dirname "$0")"

MOUNT="${PHOTO_MOUNT:-/mnt/foto}"
ROOT="${PHOTO_ROOT:-$MOUNT/Foto}"
LABEL="${PHOTO_DRIVE_LABEL:-}"

if [[ -n "$LABEL" && ! -d "$ROOT" ]]; then
  DEV="$(blkid -L "$LABEL" || true)"
  if [[ -z "$DEV" ]]; then
    echo "drive with label '$LABEL' not found (not plugged in?)"; exit 1
  fi
  echo "drive not mounted, trying udisksctl ($DEV)..."
  udisksctl mount -b "$DEV" || {
    echo "-> could not mount $DEV automatically. Mount it manually, then rerun."; exit 1; }
fi

if [[ ! -d "$ROOT" ]]; then
  echo "collection folder not found: $ROOT"
  echo "set PHOTO_MOUNT / PHOTO_ROOT (and optionally PHOTO_DRIVE_LABEL), e.g.:"
  echo "  PHOTO_MOUNT=/mnt/foto PHOTO_ROOT=/mnt/foto/Foto ./resume_index.sh"
  exit 1
fi

if command -v findmnt >/dev/null 2>&1; then
  OPTS=$(findmnt -no OPTIONS "$MOUNT")
  if [[ ",$OPTS," != *",ro,"* ]]; then
    echo "WARNING: drive is mounted READ-WRITE (options: $OPTS)."
    echo "Make it read-only first:   sudo mount -o remount,ro \"$MOUNT\""
    exit 1
  fi
fi

if pgrep -f "index_faces.py" >/dev/null; then
  echo "an indexer is already running:"
  pgrep -af index_faces.py
  exit 1
fi

echo "checks passed: drive mounted read-only, no indexer running."
if [[ "${1:-}" == "--check" ]]; then exit 0; fi

echo "resuming face index (detached; sleep/hibernation inhibited)..."
nohup setsid systemd-inhibit --what=idle:sleep:handle-lid-switch --why="face indexing" \
  .venv/bin/python index_faces.py "$ROOT" --db data/faces.db --models w600k \
  > data/index_run.out 2>&1 < /dev/null &
sleep 3
.venv/bin/python index_status.py || true
echo
echo "check progress anytime:   .venv/bin/python index_status.py --watch 60"
