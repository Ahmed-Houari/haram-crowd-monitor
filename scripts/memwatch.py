"""Log a collector's private memory every 30 s to data/logs/<name>_memory.csv (read-only diagnostics, 2026-10-25).

Private memory of the crowd collector swings by ~1 GB within seconds (large per-frame buffers), so single readings
say little; this log gives the trend (hourly minimum / median). Follows restarts through the collector's PID file.

  .venv/Scripts/python scripts/memwatch.py [--name crowd] [--every 30] [--hours 24]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge.runtime import LOG_DIR, private_memory_mb, running_pid  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="crowd")
    ap.add_argument("--every", type=float, default=30.0)
    ap.add_argument("--hours", type=float, default=24.0)
    args = ap.parse_args()
    path = LOG_DIR / f"{args.name}_memory.csv"
    new = not path.exists()
    end = time.monotonic() + args.hours * 3600
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc", "pid", "private_mb"])
        while time.monotonic() < end:
            pid = running_pid(args.name)
            mb = private_memory_mb(pid) if pid else None
            w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), pid or "",
                        round(mb, 1) if mb is not None else ""])
            f.flush()
            time.sleep(args.every)


if __name__ == "__main__":
    main()
