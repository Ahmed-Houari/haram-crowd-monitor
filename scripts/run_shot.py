"""Keep the live shot service running until the supervisor manages it (2026-10-26). The supervisor starts its children
only when it starts, and restarting it would restart the other collectors too; this launcher starts
scripts/live_shot.py and starts it again when it exits (backoff 10 s doubling to 10 min, reset after 10 healthy
minutes). When the supervisor's own "shot" child runs, live_shot.py's single-instance lock makes this one wait.

  .venv/Scripts/pythonw.exe scripts/run_shot.py          (no window)
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge.runtime import setup_logging, single_instance  # noqa: E402

PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def main() -> None:
    import logging

    setup_logging("run_shot")
    log = logging.getLogger("run_shot")
    if not single_instance("run_shot"):
        sys.exit(0)
    wait = 10
    while True:
        t = time.monotonic()
        code = subprocess.call([str(PYTHON), str(ROOT / "scripts" / "live_shot.py")], cwd=ROOT,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        up = time.monotonic() - t
        wait = 10 if up > 600 else min(wait * 2, 600)
        log.warning("live_shot.py exited with code %s after %.0f s; starting again in %d s", code, up, wait)
        time.sleep(wait)


if __name__ == "__main__":
    main()
