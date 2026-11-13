"""Stop every process of this project right away (user request 2026-10-28: stop.bat next to the project folder) and
free their RAM and GPU memory: the supervisor and its services, FFmpeg, background jobs, a dashboard dev server.

The scheduled task HaramCrowdMonitor (logon + 5-min watchdog) is switched off first, so nothing comes back until
start.bat (scripts/start_all.py). Background jobs (python processes that are not services, e.g. data/jobs/*.py) are
added to data/.stopped_jobs.json so that start_all.py starts them again; their own sub-steps are not listed.

  .venv/Scripts/python scripts/stop_all.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = "HaramCrowdMonitor"
JOBS = ROOT / "data" / ".stopped_jobs.json"
PYTHONS = ("python.exe", "pythonw.exe")
SERVICES = ("supervise.py", "collect_crowd.py", "collect_aux.py", "backend.app", "lan_dashboard.py", "live_shot.py",
            "run_shot.py", "grab_dvr.py", "housekeeping.py", "forecast_nightly.py", "stop_all.py", "start_all.py")
NO_WINDOW = 0x08000000


def processes() -> dict[int, dict]:
    """pid -> {ProcessId, ParentProcessId, Name, ExecutablePath, CommandLine} for every process (CIM via PowerShell)."""
    ps = ("Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId, Name, ExecutablePath, CommandLine | "
          "ConvertTo-Json -Compress")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                         creationflags=NO_WINDOW).stdout
    return {p["ProcessId"]: p for p in json.loads(out or "[]")}


def cmd_of(p: dict | None) -> str:
    return ((p or {}).get("CommandLine") or "").lower().replace("/", "\\")


def ours(procs: dict[int, dict]) -> list[dict]:
    """This project's processes, not this one or its launcher: every program whose executable is in the project folder
    (the venv's python / pythonw, FFmpeg, ...; also when started with a relative path) and python, node or FFmpeg with
    the project folder in its command line (the real interpreter behind the venv's launcher, a dev server)."""
    me = os.getpid()
    skip = {me, procs.get(me, {}).get("ParentProcessId")}
    root = str(ROOT).lower() + "\\"
    out = []
    for pid, p in procs.items():
        name, exe = (p.get("Name") or "").lower(), (p.get("ExecutablePath") or "").lower()
        if pid in skip:
            continue
        if exe.startswith(root) or (root in cmd_of(p) and (name in PYTHONS or name == "node.exe"
                                                             or name.startswith("ffmpeg"))):
            out.append(p)
    return out


def jobs(procs: dict[int, dict], mine: list[dict]) -> list[str]:
    """Command lines of the background jobs among them: python, not a service, not a sub-step of another job (the venv
    launcher runs the real interpreter with the same command line: one entry)."""
    pids = {p["ProcessId"] for p in mine}
    out = []
    for p in mine:
        c = cmd_of(p)
        if (p.get("Name") or "").lower() not in PYTHONS or any(s in c for s in SERVICES) and "--frames" not in c:
            continue
        par = procs.get(p["ParentProcessId"])
        while par is not None and par["ProcessId"] in pids and cmd_of(par) == c:
            par = procs.get(par["ParentProcessId"])  # the launcher of this same job
        if par is not None and par["ProcessId"] in pids:
            continue  # started by another job or a service: that one is restarted, not this step
        if p["CommandLine"] not in out:
            out.append(p["CommandLine"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="only list what would be stopped and kept for start.bat")
    args = ap.parse_args()
    procs = processes()
    mine = ours(procs)
    found = jobs(procs, mine)
    if args.dry_run:
        for p in sorted(mine, key=lambda p: p["ProcessId"]):
            print(f"would stop {p['ProcessId']:>6}  {p['CommandLine'][:150]}")
        for c in found:
            print(f"job for start.bat: {c}")
        return
    for a in (["/Change", "/TN", TASK, "/DISABLE"], ["/End", "/TN", TASK]):  # the watchdog must not start it again
        subprocess.run(["schtasks", *a], capture_output=True, creationflags=NO_WINDOW)
    try:
        kept = json.loads(JOBS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        kept = []
    kept += [{"cmd": c} for c in found if c not in [k["cmd"] for k in kept]]
    if kept:
        JOBS.write_text(json.dumps(kept, indent=1), encoding="utf-8")
    first = [p for p in mine if "supervise.py" in cmd_of(p)]  # the supervisor first: it would restart its children
    for p in first + [p for p in mine if p not in first]:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p["ProcessId"])], capture_output=True, creationflags=NO_WINDOW)
    time.sleep(2)
    left = [p for p in ours(processes()) if p["ProcessId"] in {q["ProcessId"] for q in mine}]
    print(f"Stopped {len(mine) - len(left)} of {len(mine)} project processes; auto-start off until start.bat.")
    for c in found:
        print(f"  job start.bat starts again: {c}")
    for p in left:
        print(f"  still running: {p['ProcessId']} {p['CommandLine'][:120]}")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True, creationflags=NO_WINDOW).stdout.strip()
    if gpu:
        print(f"GPU memory in use now (all programs): {gpu}")


if __name__ == "__main__":
    main()
