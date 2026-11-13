"""Start this project again after stop.bat (user request 2026-10-28: start.bat next to the project folder).

The scheduled task HaramCrowdMonitor is switched on again (logon + 5-min watchdog) and run now: its supervisor starts
every service (a second supervisor exits at once). Then the background jobs scripts/stop_all.py stopped
(data/.stopped_jobs.json) start again from the beginning in the project folder, output appended to
data/logs/jobs.log (the jobs in data/jobs skip the steps they already finished). No priority set: normal, like any
program (user, 2026-10-28).

  .venv/Scripts/python scripts/start_all.py
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = "HaramCrowdMonitor"
JOBS = ROOT / "data" / ".stopped_jobs.json"
LOG = ROOT / "data" / "logs" / "jobs.log"
NO_WINDOW, NEW_GROUP = 0x08000000, 0x00000200


def main() -> None:
    for a in (["/Change", "/TN", TASK, "/ENABLE"], ["/Run", "/TN", TASK]):
        r = subprocess.run(["schtasks", *a], capture_output=True, text=True, creationflags=NO_WINDOW)
        if r.returncode:
            print(f"schtasks {' '.join(a)}: {(r.stderr or r.stdout).strip()}")
    print("Services starting (supervisor of the scheduled task HaramCrowdMonitor); auto-start on again.")
    try:
        kept = json.loads(JOBS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        kept = []
    LOG.parent.mkdir(parents=True, exist_ok=True)
    for j in kept:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(f"--- {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC started again by start.bat: {j['cmd']}\n")
        with LOG.open("ab") as out:
            subprocess.Popen(j["cmd"], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                             creationflags=NO_WINDOW | NEW_GROUP)
        print(f"  job started again: {j['cmd']}")
    JOBS.unlink(missing_ok=True)
    print("Dashboard: http://localhost:8000 (about a minute until the first reading)")


if __name__ == "__main__":
    main()
