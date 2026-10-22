"""Background job manager.

Jobs are plain detached subprocesses (the same CLI tools as always) so they
survive app restarts and remain resumable. The manager tracks status in
data/jobs.json, reads each job's progress JSON, and tails its log.

Status rules:
  running   - pid alive
  done      - pid dead, progress says finished, no error
  error     - progress has an error string, or the log tail shows a traceback
  stopped   - pid dead without finishing (user stop or crash)
Adopted jobs (started by a previous app instance or from the CLI) work the
same way: the pid is stored on start and jobs.json is persistent.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def _popen_kwargs(log_fh) -> dict:
    """Make jobs survive the UI process on both Windows and POSIX."""
    kwargs = {
        "cwd": str(PROJECT),
        "stdout": log_fh,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        # Worker output is often redirected to a log file. Force UTF-8 so
        # photo paths with non-ASCII characters do not crash on Windows.
        "env": {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    }
    if os.name == "nt":
        # DETACHED_PROCESS keeps the worker alive after the console/server
        # exits; without it, closing the terminal can terminate indexing.
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS
    else:
        # New session = no controlling terminal, so Ctrl-C in the UI does not
        # also interrupt the detached worker.
        kwargs["start_new_session"] = True
    return kwargs


def _terminate_tree(pid: int) -> None:
    """Stop a job and its children.

    On Windows, ``.venv\\Scripts\\python.exe`` is a launcher that spawns the
    real interpreter as a child. Terminating only the launcher would leave the
    indexer running in the background, so use ``taskkill /T`` there.
    """
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            return
        except OSError:
            pass
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass


def pid_alive(pid: int) -> bool:
    """True only for a live process; zombies count as dead.

    Uses platform-specific probes. Never calls ``os.kill(pid, 0)`` on Windows:
    there it terminates the process instead of probing it.
    """
    if not pid:
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not handle:
                return False
            try:
                code = wintypes.DWORD()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return False
                return code.value == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False
    if sys.platform.startswith("linux"):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            state = stat.rsplit(")", 1)[1].split()[0] if ")" in stat else "?"
            return state not in ("Z", "z", "X", "x")
        except OSError:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class JobManager:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.jobs_dir = self.data_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_file = self.data_dir / "jobs.json"
        self.jobs: list[dict] = []
        self._procs: dict[str, subprocess.Popen] = {}
        if self.jobs_file.exists():
            try:
                self.jobs = json.loads(self.jobs_file.read_text(encoding="utf-8"))
            except Exception:
                self.jobs = []
        self.refresh()

    # ------------------------------------------------------------- utils --
    def _save(self) -> None:
        self.jobs_file.write_text(json.dumps(self.jobs, indent=1), encoding="utf-8")

    def _progress(self, job: dict) -> dict:
        try:
            return json.loads(Path(job["progress"]).read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _log_tail(self, job: dict, lines: int = 15) -> list[str]:
        try:
            text = Path(job["log"]).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []
        clean = [ln.rstrip() for ln in text.replace("\r", "\n").splitlines()]
        return [ln for ln in clean if ln][-lines:]

    # ------------------------------------------------------------- api -----
    def start(self, kind: str, cmd: list[str], title: str, meta: dict | None = None) -> dict:
        jid = f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}"
        suffix = 2
        while any(j["id"] == jid for j in self.jobs):
            jid = f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}-{suffix}"
            suffix += 1
        log = self.jobs_dir / f"{jid}.log"
        progress = self.jobs_dir / f"{jid}.progress.json"
        cmd = [a.replace("{progress}", str(progress)).replace("{log}", str(log)) for a in cmd]
        log_fh = open(log, "ab")
        proc = subprocess.Popen(cmd, **_popen_kwargs(log_fh))
        log_fh.close()
        self._procs[jid] = proc
        job = {
            "id": jid, "kind": kind, "title": title, "cmd": cmd,
            "pid": proc.pid, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ended": None, "status": "running",
            "log": str(log), "progress": str(progress),
            "meta": meta or {},
        }
        self.jobs.append(job)
        self._save()
        return job

    def refresh(self) -> None:
        changed = False
        for job in self.jobs:
            if job["status"] not in ("running", "stopping"):
                continue
            proc = self._procs.get(job["id"])
            if proc is not None:
                if proc.poll() is None:  # still running; also reaps zombies
                    continue
            elif pid_alive(job["pid"]):
                continue
            prog = self._progress(job)
            if prog.get("error"):
                job["status"] = "error"
            elif prog.get("finished"):
                job["status"] = "done"
            else:
                tail = "\n".join(self._log_tail(job, 40))
                job["status"] = "error" if "Traceback" in tail else "stopped"
            job["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
            changed = True
        if changed:
            self._save()

    def stop(self, jid: str) -> dict:
        for job in self.jobs:
            if job["id"] == jid and job["status"] in ("running", "stopping"):
                _terminate_tree(job["pid"])
                job["status"] = "stopping"
                self._save()
                return job
        return {}

    def get(self, jid: str) -> dict | None:
        self.refresh()
        for job in self.jobs:
            if job["id"] == jid:
                j = dict(job)
                j["progress_data"] = self._progress(job)
                j["log_tail"] = self._log_tail(job)
                return j
        return None

    def active(self, kind: str | None = None) -> dict | None:
        self.refresh()
        for job in reversed(self.jobs):
            if job["status"] in ("running", "stopping") and (kind is None or job["kind"] == kind):
                return job
        return None

    def list(self, limit: int = 50) -> list[dict]:
        self.refresh()
        out = []
        for job in reversed(self.jobs[-limit:]):
            j = dict(job)
            j["progress_data"] = self._progress(job)
            if job["status"] in ("running", "stopping"):
                j["log_tail"] = self._log_tail(job, 8)
            out.append(j)
        return out
