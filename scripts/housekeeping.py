"""Keep the project's disk use small and fixed (user request 2026-10-26). Deletes only files that are copies, temporary
or for debugging; never the readings (data/haram.db), the references, the models or the kept ground truth.

Rules (age = file modification time):
  data/samples/dvr          prayer-window frames (scripts/grab_dvr.py): 2 days; index.csv trimmed to the same days;
                            never a training or test frame (one a data/finetune*/<window>.json uses): those are
                            copied to data/kept_frames first and keep their index rows (user, 2026-10-28)
  data/kept_frames          training and test frames (edge/kept_frames.py): kept, never touched
  data/samples/truth        ground-truth frames: kept, never touched
  data/samples/pairs        inner-ring truth pairs (scripts/live_shot.py PairCapture): the newest PAIRS_GB kept
  data/samples/patches, probe, debug_align, data/vloc/*.jpg, *.tmp under data/   debug overlays and temp files: 2 days
  data/logs                 rotated logs (*.log.N) and logs nobody wrote to: 7 days
  data/backups              daily DB backups (edge/backup.py): the newest BACKUPS_KEEP kept, each compressed once it
                            is an hour old (zip with LZMA, ~11x smaller; restore: unzip, then use the .db);
                            pre_*.db (made before a one-off change): 7 days
Everything else under data/ is fixed in size or small (sampler-capped samples, references, thumbnails, the bank,
benchmarks).

  .venv/Scripts/python scripts/housekeeping.py [--dry-run]     # once
  .venv/Scripts/python scripts/housekeeping.py --loop          # every hour (supervisor child "housekeeping")
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge.kept_frames import KEPT, keep, window_stems  # noqa: E402
from edge.runtime import setup_logging, single_instance  # noqa: E402

DATA = ROOT / "data"
DAY = 86400
FRAMES_DAYS = 2
DEBUG_DAYS = 2
LOG_DAYS = 7
PRE_BACKUP_DAYS = 7
BACKUPS_KEEP = 7
PAIRS_GB = 2.0
DEBUG_DIRS = ["samples/patches", "samples/probe", "samples/debug_align"]
NEVER = [DATA / "samples" / "truth", DATA / "haram.db", KEPT]
log = logging.getLogger("housekeeping")


class Cleaner:
    def __init__(self, dry: bool):
        self.dry = dry
        self.freed = 0
        self.n = 0

    def delete(self, f: Path, why: str) -> None:
        if any(f == p or p in f.parents for p in NEVER):
            return
        size = f.stat().st_size
        if not self.dry:
            try:
                f.unlink()
            except OSError as e:  # in use (Windows): next round
                log.warning("could not delete %s: %s", f, e)
                return
        self.freed += size
        self.n += 1
        log.debug("%s %s (%s)", "would delete" if self.dry else "deleted", f.relative_to(ROOT), why)

    def older(self, files, days: float, why: str) -> None:
        cutoff = time.time() - days * DAY
        for f in files:
            if f.is_file() and f.stat().st_mtime < cutoff:
                self.delete(f, why)


def frames(c: Cleaner) -> None:
    d = DATA / "samples" / "dvr"
    if not d.exists():
        return
    used = window_stems()  # training / test frames: never deleted
    if not c.dry:
        keep(used)
    used |= {p.stem for p in KEPT.glob("*.jpg")}
    c.older([f for f in list(d.glob("*.jpg")) + list(d.glob("*.json")) if f.stem not in used], FRAMES_DAYS, "prayer frame")
    index = d / "index.csv"
    if index.exists() and not c.dry:
        cutoff = datetime.now(timezone.utc) - timedelta(days=FRAMES_DAYS)
        rows = list(csv.reader(index.open(encoding="utf-8")))
        kept = rows[:1] + [r for r in rows[1:] if r and r[1] and (datetime.fromisoformat(r[1]) >= cutoff
                                                                  or Path(r[0]).stem in used)]
        if len(kept) < len(rows):
            tmp = index.with_suffix(".tmp")
            with tmp.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerows(kept)
            tmp.replace(index)


def pairs(c: Cleaner) -> None:
    d = DATA / "samples" / "pairs"
    if not d.exists():
        return
    jpgs = sorted(d.glob("*.jpg"), key=lambda f: f.name, reverse=True)  # newest first (UTC stamp in the name)
    total = 0
    for f in jpgs:
        total += f.stat().st_size
        if total > PAIRS_GB * 1e9:
            c.delete(f, "pairs beyond the cap")
            side = f.with_suffix(".json")
            if side.exists():
                c.delete(side, "pairs beyond the cap")


def debug(c: Cleaner) -> None:
    for sub in DEBUG_DIRS:
        d = DATA / sub
        if d.exists():
            c.older(d.rglob("*"), DEBUG_DAYS, "debug output")
    c.older((DATA / "vloc").glob("*.jpg"), DEBUG_DAYS, "debug overlay")
    c.older(DATA.rglob("*.tmp"), DEBUG_DAYS, "temp file")


def logs(c: Cleaner) -> None:
    d = DATA / "logs"
    if d.exists():
        c.older([f for f in d.iterdir() if ".log." in f.name or f.suffix == ".log"], LOG_DAYS, "old log")


def backups(c: Cleaner) -> None:
    d = DATA / "backups"
    if not d.exists():
        return
    c.older(d.glob("pre_*.db"), PRE_BACKUP_DAYS, "one-off backup")
    daily = sorted(list(d.glob("haram_*.db")) + list(d.glob("haram_*.db.zip")), key=lambda f: f.name.split(".")[0])
    for f in daily[:-BACKUPS_KEEP]:
        c.delete(f, f"backup beyond the newest {BACKUPS_KEEP}")
    for f in daily[-BACKUPS_KEEP:]:  # compressed once complete (edge/backup.py writes it at Makkah midnight)
        if f.suffix == ".db" and not c.dry and time.time() - f.stat().st_mtime > 3600:
            z = f.with_name(f.name + ".zip")
            with zipfile.ZipFile(z.with_suffix(".tmp"), "w", zipfile.ZIP_LZMA) as zf:
                zf.write(f, f.name)
            z.with_suffix(".tmp").replace(z)
            before = f.stat().st_size
            f.unlink()
            c.freed += before - z.stat().st_size
            log.info("compressed %s: %.1f -> %.1f MB", f.name, before / 1e6, z.stat().st_size / 1e6)


def du(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def run(dry: bool) -> None:
    c = Cleaner(dry)
    for step in (frames, pairs, debug, logs, backups):
        try:
            step(c)
        except Exception:
            log.exception("housekeeping step %s failed", step.__name__)
    log.info("%s %d files, %.1f MB; data/ now %.2f GB", "would free" if dry else "freed", c.n, c.freed / 1e6, du(DATA) / 1e9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--loop", action="store_true", help="run every hour")
    args = ap.parse_args()
    setup_logging("housekeeping")
    if args.loop:
        if not single_instance("housekeeping"):
            sys.exit(3)
        while True:
            run(False)
            time.sleep(3600)
    run(args.dry_run)


if __name__ == "__main__":
    main()
