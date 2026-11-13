"""Auxiliary data collector: TomTom traffic tiles every 5 min + prayer times once a day.

Runs next to the crowd collector and writes to the same database (DATABASE_URL, default
data/haram.db). Polls are aligned to a fixed grid (:00, :05, :10, ... for --interval 300).

    python scripts/collect_aux.py            # run until Ctrl+C
    python scripts/collect_aux.py --once     # one traffic poll + prayer check, then exit
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge import db, prayer_times, traffic  # noqa: E402
from edge.runtime import ALREADY_RUNNING_EXIT, setup_logging, single_instance  # noqa: E402

log = logging.getLogger("collect_aux")


def check_prayers() -> bool:
    """Ensure prayer times for the current + next month. True if the DB is complete."""
    try:
        calls = prayer_times.ensure_prayer_times()
        log.info("prayer times ok (%d Aladhan call%s)", calls, "" if calls == 1 else "s")
        return True
    except Exception as e:  # never stop the traffic loop because of Aladhan
        log.error("prayer times failed: %s", e)
        return False


def traffic_poll(args: argparse.Namespace) -> None:
    """One TomTom poll -> DB, one log line. Errors are logged, never raised."""
    try:
        result = traffic.poll(args.zoom, args.radius_km, args.daily_cap)
    except (traffic.BackoffActive, traffic.DailyCapReached) as e:
        log.warning("traffic poll skipped: %s", e)
        return
    except traffic.TrafficError as e:
        log.error("traffic poll failed: %s", traffic.redact(str(e)))
        return
    try:
        traffic.save_poll(result)
    except Exception as e:
        log.error("traffic poll %s not saved: %s", result.poll_utc, traffic.redact(str(e))[:300])
        return
    log.info(traffic.poll_line(result) + f" / cap {args.daily_cap}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect TomTom traffic (every 5 min) and prayer times (daily).")
    parser.add_argument("--once", action="store_true", help="one traffic poll + prayer check, then exit")
    parser.add_argument("--interval", type=float, default=300, help="seconds between traffic polls (default 300)")
    parser.add_argument("--daily-cap", type=int, default=traffic.DEFAULT_DAILY_CAP,
                        help=f"max TomTom requests per UTC day (default {traffic.DEFAULT_DAILY_CAP})")
    parser.add_argument("--zoom", type=int, default=traffic.DEFAULT_ZOOM)
    parser.add_argument("--radius-km", type=float, default=traffic.RADIUS_KM)
    args = parser.parse_args()
    setup_logging("aux")
    if not single_instance("aux"):
        log.warning("another collect_aux.py is already running; exiting")
        sys.exit(ALREADY_RUNNING_EXIT)

    engine = db.init_db()
    log.info("database %s; %d tiles per poll at zoom %d, every %.0f s, daily cap %d",
             db.describe(engine),
             len(traffic.tiles_for_area(radius_km=args.radius_km, z=args.zoom)), args.zoom, args.interval,
             args.daily_cap)

    if args.once:
        check_prayers()
        traffic_poll(args)
        return

    prayers_day = None  # Makkah date of the last successful prayer check
    next_prayer_try = 0.0
    try:
        while True:
            today = prayer_times.makkah_today()
            if prayers_day != today and time.time() >= next_prayer_try:
                if check_prayers():
                    prayers_day = today
                else:
                    next_prayer_try = time.time() + 3600  # retry in 1 h
            traffic_poll(args)
            wait = args.interval - time.time() % args.interval  # next grid point
            time.sleep(wait if wait > 1 else wait + args.interval)
    except KeyboardInterrupt:
        log.info("stopped by user")


if __name__ == "__main__":
    main()
