"""Counted minutes per hour, per camera and total (read-only): is the collector counting enough?

A counted minute = a crowd_minute row (an accepted reading of that camera in that minute). Also: frames per hour
(collector_minute), which reference matched (crowd_frame.ref_id, since 2026-10-24) and how many counted minutes saw
only part of the zone (visible-part metric, since 2026-10-24 17:00 Makkah).

Run:  .venv/Scripts/python scripts/coverage.py [--since "2026-10-24 14:15"] [--hours 24]   (Makkah time)
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
MAKKAH = timezone(timedelta(hours=3))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", help="Makkah time 'YYYY-MM-DD HH:MM' (default: --hours ago)")
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--db", type=Path, default=ROOT / "data" / "haram.db")
    args = ap.parse_args()
    since = (datetime.strptime(args.since, "%Y-%m-%d %H:%M").replace(tzinfo=MAKKAH) if args.since
             else datetime.now(MAKKAH) - timedelta(hours=args.hours)).astimezone(timezone.utc)
    s = since.strftime("%Y-%m-%d %H:%M:%S")
    e = create_engine(f"sqlite:///file:{args.db.resolve().as_posix()}?mode=ro&uri=true")
    with e.connect() as c:
        mins = c.execute(text("select minute_utc, view_id, coalesce(visible_avg, 1.0) from crowd_minute "
                              "where minute_utc >= :s"), {"s": s}).all()
        frames = c.execute(text("select minute_utc, frames from collector_minute where minute_utc >= :s"), {"s": s}).all()
        refs = c.execute(text("select view_id, coalesce(ref_id, 'primary'), count(*) from crowd_frame "
                              "where ts_utc >= :s and accepted group by 1, 2"), {"s": s}).all()
    hour = lambda t: (datetime.fromisoformat(str(t)[:19]).replace(tzinfo=timezone.utc).astimezone(MAKKAH)
                      .strftime("%m-%d %H:00"))  # noqa: E731
    per = defaultdict(lambda: defaultdict(int))
    for t, v, _ in mins:
        per[hour(t)][v] += 1
    fr = defaultdict(int)
    for t, n in frames:
        fr[hour(t)] += 1  # minutes the collector was alive in that hour
    cams = sorted({v for _, v, _ in mins})
    hours = sorted(set(fr) | set(per))
    print(f"Counted minutes per hour since {since.astimezone(MAKKAH):%Y-%m-%d %H:%M} Makkah")
    print(f"{'hour':>12} {'alive':>5} " + " ".join(f"{c[:16]:>16}" for c in cams) + f" {'total':>6}")
    for h in hours:
        tot = sum(per[h].values())
        print(f"{h:>12} {fr[h]:>5} " + " ".join(f"{per[h].get(c, 0):>16}" for c in cams) + f" {tot:>6}")
    n_hours = max(1, len(hours))
    total = len(mins)
    print(f"total {total} counted minutes in {len(hours)} hours = {total / n_hours:.2f} per hour "
          f"(target >= 10); per camera: " + ", ".join(f"{c} {sum(per[h].get(c, 0) for h in hours)}" for c in cams))
    part = [vis for *_, vis in mins if vis < 0.99]
    if part:
        print(f"counted minutes with only part of the zone in view: {len(part)} of {total} "
              f"(mean {100 * sum(part) / len(part):.0f} % visible)")
    if refs:
        print("accepted frames by reference: " + ", ".join(f"{v}/{r} {n}" for v, r, n in refs))


if __name__ == "__main__":
    main()
