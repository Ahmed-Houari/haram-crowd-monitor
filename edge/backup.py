"""Data safety (user request 2026-10-24): daily SQLite backup + disk space check, run by the crowd collector.

backup_db() uses SQLite's online backup API (a consistent copy while the collectors keep writing; never a plain
file copy of a database in use) into data/backups/haram_<Makkah date>.db and keeps the newest KEEP files.
disk_status() reports free space on the project drive; LOW_GB or less is flagged in the collector status.

  .venv/Scripts/python -m edge.backup            back up now (if not done today) and print the disk status
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from edge.db import PROJECT_ROOT, database_url

log = logging.getLogger(__name__)

BACKUP_DIR = PROJECT_ROOT / "data" / "backups"
KEEP = 7
LOW_GB = 10.0
MAKKAH = timezone(timedelta(hours=3))


def db_path() -> Path | None:
    """The SQLite file behind DATABASE_URL (None for other databases)."""
    from sqlalchemy.engine import make_url

    url = make_url(database_url())
    return Path(url.database) if url.get_backend_name() == "sqlite" and url.database else None


def backup_db(force: bool = False) -> Path | None:
    """Today's backup (Makkah date), made with the SQLite backup API; prunes to the newest KEEP. None if skipped."""
    src = db_path()
    if src is None or not src.exists():
        return None
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    out = BACKUP_DIR / f"haram_{datetime.now(MAKKAH):%Y%m%d}.db"
    if (out.exists() or out.with_name(out.name + ".zip").exists()) and not force:  # .zip: compressed by housekeeping
        return None
    tmp = out.with_suffix(".tmp")
    t0 = time.monotonic()
    s, d = sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True), sqlite3.connect(tmp)
    try:
        s.backup(d, pages=4096, sleep=0.05)  # in steps, so the collectors' writes are not blocked for long
        d.execute("PRAGMA journal_mode=DELETE")  # the copy inherits WAL mode: make it one self-contained file
    finally:  # close explicitly: "with sqlite3.connect()" only commits, and Windows keeps open files locked
        d.close()
        s.close()
    tmp.replace(out)
    for old in sorted(BACKUP_DIR.glob("haram_*.db"))[:-KEEP]:
        for f in (old, old.with_name(old.name + "-wal"), old.with_name(old.name + "-shm")):
            f.unlink(missing_ok=True)
    log.warning("database backup %s (%.1f MB, %.1f s)", out.name, out.stat().st_size / 1e6, time.monotonic() - t0)
    return out


def disk_status(path: Path = PROJECT_ROOT) -> dict:
    u = shutil.disk_usage(path)
    free = u.free / 1e9
    return {"disk_free_gb": round(free, 1), "disk_used_pct": round(100 * u.used / u.total, 1), "disk_low": free <= LOW_GB}


def last_backup() -> str | None:
    """Newest backup, plain or compressed by scripts/housekeeping.py (haram_<date>.db.zip)."""
    files = sorted(list(BACKUP_DIR.glob("haram_*.db")) + list(BACKUP_DIR.glob("haram_*.db.zip")),
                   key=lambda f: f.name.split(".")[0])
    return files[-1].name if files else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="back up even if today's backup exists")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = backup_db(args.force)
    print("backup:", out or f"today's backup exists ({last_backup()})", "| disk:", disk_status())


if __name__ == "__main__":
    main()
