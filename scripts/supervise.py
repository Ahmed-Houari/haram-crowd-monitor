"""Run both collectors 24/7: restart them when they exit, keep the PC awake on AC power.

Started by Windows Task Scheduler at logon (and every 5 min as a watchdog; a second copy exits
at once because of the single-instance lock). Children:
  crowd  scripts/collect_crowd.py   (frames every 15 s -> crowd_minute)
  aux    scripts/collect_aux.py     (TomTom every 5 min, prayer times daily)
  api    uvicorn backend.app:app    (dashboard + API on http://localhost:8000, local only)
  lan    scripts/lan_dashboard.py   (forwards port 8000 of the home Wi-Fi to the API, for phones; Windows Firewall
                                     allows it on the Private profile only: scripts/allow_dashboard_lan.ps1)
Restart backoff: 10 s doubling to 10 min, reset after 10 min of healthy running. Children run
inside a Windows job object, so they are killed if the supervisor dies (no orphans/duplicates).
Keep-awake: while on AC power, Windows idle sleep is blocked (SetThreadExecutionState); on
battery the PC may sleep normally. Lid close / power button are user power settings.

Memory guard (user request 2026-10-25): every 30 s the crowd collector's private memory is read; its level is the
25th percentile of the last 10 minutes (it swings ~1 GB within seconds). Above 12 GB the collector is restarted at
the next quiet moment (no reading for 2 min, from data/.crowd_status.json), above 14 GB at once; at most once an hour.
The collector has the same guard inside (scripts/collect_crowd.py MemoryGuard); this one also works if it hangs.

Logs: data/logs/supervisor.log, data/logs/<child>.log, data/logs/<child>.out.log (stdout/stderr).
Manual run:  .venv/Scripts/python scripts/supervise.py
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge.runtime import (ALREADY_RUNNING_EXIT, LOG_DIR, PROJECT_ROOT, keep_awake, on_ac_power,  # noqa: E402
                          private_memory_mb, running_pid, setup_logging, single_instance, terminate_pid)

log = logging.getLogger("supervisor")
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"  # console python; window hidden below
CHILDREN = {"crowd": ["scripts/collect_crowd.py"], "aux": ["scripts/collect_aux.py"],
            # dashboard + API, local only: http://localhost:8000
            "api": ["-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", "8000", "--log-level", "warning"],
            "lan": ["scripts/lan_dashboard.py"],  # home Wi-Fi -> the API (2026-10-25)
            "shot": ["scripts/live_shot.py"],  # the shot on air every 1-2 s, the Mataf metric (2026-10-26)
            "dvr": ["scripts/grab_dvr.py", "--prayers"],  # raw frames around every prayer -> data/samples/dvr (2026-10-26)
            "housekeeping": ["scripts/housekeeping.py", "--loop"],  # fixed disk use: frames 2 d, logs 7 d, backups 7
            # the crowd forecast retrained at 03:00 Makkah every night, its daily error logged; not on the dashboard
            "forecast": ["scripts/forecast_nightly.py", "--loop"]}  # (2026-11-01) -> data/forecast
BACKOFF = (10, 600)  # first / max seconds before a restart
HEALTHY_S = 600
STATUS_PATH = PROJECT_ROOT / "data" / ".supervisor_status.json"
CREATE_NO_WINDOW = 0x08000000
CROWD_STATUS = PROJECT_ROOT / "data" / ".crowd_status.json"
MEMORY_LIMIT_MB, MEMORY_HARD_MB, QUIET_S = 12000, 14000, 120  # same as scripts/collect_crowd.py
MEMORY_EVERY_S, MEMORY_COOLDOWN_S = 30, 3600


class MemoryGuard:
    """Restart the crowd collector when its private memory stays above the limit (see the module docstring)."""

    def __init__(self):
        from collections import deque

        self.samples: deque = deque(maxlen=20)  # 10 minutes of 30-s samples
        self.next = 0.0
        self.last_restart = -MEMORY_COOLDOWN_S

    def check(self, child: "Child") -> None:
        now = time.monotonic()
        if now < self.next or child.proc is None:
            return
        self.next = now + MEMORY_EVERY_S
        pid = running_pid("crowd")
        mb = private_memory_mb(pid) if pid else None
        if mb is None:
            return
        self.samples.append(mb)
        if len(self.samples) < self.samples.maxlen or now - self.last_restart < MEMORY_COOLDOWN_S:
            return
        level = sorted(self.samples)[len(self.samples) // 4]
        quiet = True
        try:
            st = json.loads(CROWD_STATUS.read_text())
            last = max((datetime.fromisoformat(t) for t in (st.get("last_reading_utc"), st.get("last_m2_utc")) if t),
                       default=None)
            quiet = last is None or (datetime.now(timezone.utc) - last).total_seconds() >= QUIET_S
        except (OSError, ValueError):
            pass
        if level > MEMORY_HARD_MB or (level > MEMORY_LIMIT_MB and quiet):
            log.warning("memory guard: crowd collector at %.0f MB private (limit %d MB%s): restarting it", level,
                        MEMORY_LIMIT_MB, ", quiet moment" if quiet else ", hard limit")
            if terminate_pid(pid):  # the venv launcher exits with it; check() restarts as for any exit
                self.last_restart = now
                self.samples.clear()


class Job:
    """Windows job object with KILL_ON_JOB_CLOSE: children die together with this process."""

    def __init__(self):
        self.handle = None
        if sys.platform != "win32":
            return
        import ctypes
        from ctypes import wintypes

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in ("r_ops", "w_ops", "o_ops", "r_bytes", "w_bytes", "o_bytes")]

        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class EXTENDED(ctypes.Structure):
            _fields_ = [("Basic", BASIC), ("Io", IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32 = ctypes.windll.kernel32
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        self.handle = k32.CreateJobObjectW(None, None)
        info = EXTENDED()
        info.Basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            log.warning("could not configure job object; children may outlive the supervisor")
        self._k32 = k32

    def add(self, proc: subprocess.Popen) -> None:
        if self.handle and not self._k32.AssignProcessToJobObject(self.handle, int(proc._handle)):
            log.warning("could not add pid %d to the job object", proc.pid)


class Child:
    def __init__(self, name: str, args: list[str], job: Job):
        self.name, self.args, self.job = name, args, job
        self.proc: subprocess.Popen | None = None
        self.started = 0.0
        self.next_start = 0.0
        self.fails = 0
        self.restarts = 0

    def start(self) -> None:
        out_path = LOG_DIR / f"{self.name}.out.log"  # stdout/stderr: crashes and tracebacks end up here
        if out_path.exists() and out_path.stat().st_size > 20_000_000:
            out_path.replace(out_path.with_suffix(".old"))
        out = open(out_path, "ab")
        self.proc = subprocess.Popen([str(PYTHON), *self.args], cwd=PROJECT_ROOT, stdout=out, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        out.close()
        self.job.add(self.proc)
        self.started = time.monotonic()
        log.info("started %s (pid %d)", self.name, self.proc.pid)

    def check(self) -> None:
        now = time.monotonic()
        if self.proc is None:
            if now >= self.next_start:
                self.start()
            return
        code = self.proc.poll()
        if code is None:
            if now - self.started > HEALTHY_S:
                self.fails = 0
            return
        ran = now - self.started
        self.proc, self.restarts = None, self.restarts + 1
        if code == ALREADY_RUNNING_EXIT:
            wait = 60.0
            log.warning("%s: another instance is running; retry in 60 s", self.name)
        else:
            self.fails = 0 if ran > HEALTHY_S else self.fails + 1
            wait = min(BACKOFF[0] * 2 ** max(self.fails - 1, 0), BACKOFF[1])
            log.error("%s exited with code %s after %.0f s; restart in %.0f s", self.name, code, ran, wait)
        self.next_start = now + wait


def main() -> None:
    setup_logging("supervisor")
    if not single_instance("supervisor"):
        log.info("supervisor already running; exiting")  # normal: Task Scheduler watchdog trigger
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    job = Job()
    children = [Child(name, args, job) for name, args in CHILDREN.items()]
    log.info("supervisor started (pid-locked); children: %s", ", ".join(CHILDREN))
    guard = MemoryGuard()
    awake = None
    last_status = 0.0
    try:
        while True:
            for c in children:
                c.check()
                if c.name == "crowd":
                    try:
                        guard.check(c)
                    except Exception:
                        log.exception("memory guard failed")
            ac = on_ac_power()
            want = bool(ac)
            if want != awake:
                log.info("power: %s -> keep-awake %s", {True: "AC", False: "battery", None: "unknown"}[ac],
                         "ON" if want else "OFF")
                awake = want
            keep_awake(want)  # refresh every loop (cheap; state is per thread)
            if time.monotonic() - last_status > 60:
                STATUS_PATH.write_text(json.dumps({
                    "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "ac_power": ac,
                    "keep_awake": want,
                    "children": {c.name: {"pid": c.proc.pid if c.proc else None, "restarts": c.restarts} for c in children},
                }, indent=1))
                last_status = time.monotonic()
            time.sleep(5)
    except KeyboardInterrupt:
        log.info("stopped by user")
    finally:
        keep_awake(False)
        for c in children:
            if c.proc and c.proc.poll() is None:
                c.proc.terminate()


if __name__ == "__main__":
    main()
