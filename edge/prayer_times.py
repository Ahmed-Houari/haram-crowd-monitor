"""Prayer times for Masjid al-Haram from the Aladhan API (method 4 = Umm al-Qura, Makkah).

One API call returns a whole Gregorian month (calendar endpoint). Times are stored in UTC
in the prayer_times table, keyed by the Makkah local date. Makkah is UTC+3 all year (no DST).
ensure_prayer_times() only calls the API for months with missing dates.
prayer_context() gives, for a UTC timestamp, the previous/next prayer and the minutes
since/until them (forecast features).

Endpoint: https://api.aladhan.com/v1/calendar/{year}/{month}
          ?latitude=..&longitude=..&method=4&iso8601=true&timezonestring=Asia/Riyadh
"""
from __future__ import annotations

import argparse
import bisect
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import requests
from sqlalchemy import Engine, select

from edge import db

log = logging.getLogger(__name__)

API_URL = "https://api.aladhan.com/v1/calendar/{year}/{month}"
METHOD = 4  # Umm Al-Qura University, Makkah (Fajr 18.5 deg, Isha 90 min after Maghrib)
HARAM_LAT, HARAM_LON = 21.4225, 39.8262
MAKKAH_TZ = timezone(timedelta(hours=3), "AST")  # Arabia Standard Time, no DST
TIMES = ("fajr", "sunrise", "dhuhr", "asr", "maghrib", "isha")  # stored columns
PRAYERS = ("fajr", "dhuhr", "asr", "maghrib", "isha")  # the five prayers (sunrise is not one)
TIMEOUT_S = 30
ATTEMPTS = 2
RETRY_WAIT_S = 10.0


class PrayerTimesError(RuntimeError):
    pass


@dataclass
class PrayerDay:
    date: date  # Makkah local date
    method: int
    hijri_date: str  # "YYYY-MM-DD"
    fajr: datetime  # all UTC
    sunrise: datetime
    dhuhr: datetime
    asr: datetime
    maghrib: datetime
    isha: datetime


@dataclass
class PrayerContext:
    prev_prayer: str
    prev_utc: datetime
    minutes_since_prev: float
    next_prayer: str
    next_utc: datetime
    minutes_to_next: float


def makkah_today() -> date:
    return datetime.now(MAKKAH_TZ).date()


def _parse_day(d: dict, method: int) -> PrayerDay:
    t = d["timings"]  # iso8601=true: "2026-10-23T05:03:00+03:00"
    times = {k: datetime.fromisoformat(t[k.capitalize()].split(" ")[0]).astimezone(timezone.utc) for k in TIMES}
    gd, gm, gy = d["date"]["gregorian"]["date"].split("-")  # DD-MM-YYYY
    hd, hm, hy = d["date"]["hijri"]["date"].split("-")
    return PrayerDay(date(int(gy), int(gm), int(gd)), method, f"{hy}-{hm}-{hd}", **times)


def fetch_month(year: int, month: int, method: int = METHOD) -> list[PrayerDay]:
    """One Aladhan call: prayer times for every day of a Gregorian month at the Haram."""
    params = {"latitude": HARAM_LAT, "longitude": HARAM_LON, "method": method,
              "iso8601": "true", "timezonestring": "Asia/Riyadh"}
    url = API_URL.format(year=year, month=month)
    err = ""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT_S)
            if r.status_code == 200:
                body = r.json()
                if body.get("code") == 200 and isinstance(body.get("data"), list):
                    days = [_parse_day(d, method) for d in body["data"]]
                    log.info("Aladhan: %d days for %04d-%02d (method %d)", len(days), year, month, method)
                    return days
                err = f"unexpected body: {str(body)[:200]}"
            else:
                err = f"HTTP {r.status_code}: {r.text[:200]}"
                if r.status_code < 500 and r.status_code != 429:
                    break  # bad request: retrying will not help
        except (requests.RequestException, ValueError, KeyError) as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
        if attempt < ATTEMPTS:
            log.warning("Aladhan %04d-%02d attempt %d failed (%s), retry in %.0f s", year, month, attempt, err,
                        RETRY_WAIT_S)
            time.sleep(RETRY_WAIT_S)
    raise PrayerTimesError(f"Aladhan {year:04d}-{month:02d} failed: {err}")


def save_days(days: list[PrayerDay], engine: Engine | None = None) -> int:
    """Upsert prayer days (key: date, method)."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    with (engine or db.get_engine()).begin() as conn:
        return db.upsert(conn, db.prayer_times, [{**d.__dict__, "fetched_utc": now} for d in days])


def stored_dates(start: date, end: date, method: int = METHOD, engine: Engine | None = None) -> set[date]:
    """Dates in [start, end] that are already in the DB."""
    t = db.prayer_times
    q = select(t.c.date).where(t.c.method == method, t.c.date >= start, t.c.date <= end)
    with (engine or db.get_engine()).connect() as conn:
        return set(conn.execute(q).scalars())


def _months(start: date, end: date) -> list[tuple[int, int]]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def ensure_prayer_times(days_ahead: int = 31, method: int = METHOD, engine: Engine | None = None) -> int:
    """Make sure the DB has every date from the 1st of the current month to today + days_ahead.

    Default covers the current and the next month. Calls the API (one call per month) only
    for months with missing dates. Returns the number of API calls made.
    """
    engine = engine or db.init_db()
    today = makkah_today()
    start, end = today.replace(day=1), today + timedelta(days=days_ahead)
    have = stored_dates(start, end, method, engine)
    calls = 0
    for y, m in _months(start, end):
        first = date(y, m, 1)
        last = (date(y + (m == 12), m % 12 + 1, 1)) - timedelta(days=1)
        wanted = {first + timedelta(days=i) for i in range((last - first).days + 1)
                  if start <= first + timedelta(days=i) <= end}
        if wanted <= have:
            continue
        days = fetch_month(y, m, method)
        calls += 1
        save_days(days, engine)
    if calls == 0:
        log.info("prayer times complete for %s .. %s (no API call)", start, end)
    return calls


def load_events(start_utc: datetime, end_utc: datetime, method: int = METHOD, include_sunrise: bool = False,
                engine: Engine | None = None) -> list[tuple[datetime, str]]:
    """Sorted (time_utc, name) of all prayers between start_utc and end_utc (+-1 day margin)."""
    t = db.prayer_times
    d0 = (db.utc(start_utc).astimezone(MAKKAH_TZ) - timedelta(days=1)).date()
    d1 = (db.utc(end_utc).astimezone(MAKKAH_TZ) + timedelta(days=1)).date()
    names = TIMES if include_sunrise else PRAYERS
    q = select(t).where(t.c.method == method, t.c.date >= d0, t.c.date <= d1).order_by(t.c.date)
    with (engine or db.get_engine()).connect() as conn:
        rows = conn.execute(q).mappings().all()
    return sorted((db.utc(r[n]), n) for r in rows for n in names)


def context_from_events(events: list[tuple[datetime, str]], ts_utc: datetime) -> PrayerContext | None:
    """Previous/next prayer around ts_utc from a sorted event list (fast, for many timestamps)."""
    ts = db.utc(ts_utc)
    i = bisect.bisect_right([e[0] for e in events], ts)
    if i == 0 or i >= len(events):
        return None
    (pt, pn), (nt, nn) = events[i - 1], events[i]
    return PrayerContext(pn, pt, (ts - pt).total_seconds() / 60, nn, nt, (nt - ts).total_seconds() / 60)


def prayer_context(ts_utc: datetime, method: int = METHOD, include_sunrise: bool = False,
                   engine: Engine | None = None) -> PrayerContext | None:
    """Previous/next prayer and minutes since/until them for one UTC timestamp (None if not in the DB)."""
    events = load_events(ts_utc, ts_utc, method, include_sunrise, engine)
    return context_from_events(events, ts_utc)


def print_day(day: date, method: int = METHOD, engine: Engine | None = None) -> None:
    """Print one day's prayer times in Makkah local time and UTC."""
    t = db.prayer_times
    with (engine or db.get_engine()).connect() as conn:
        r = conn.execute(select(t).where(t.c.date == day, t.c.method == method)).mappings().first()
    if not r:
        print(f"{day}: not in the DB (run --ensure)")
        return
    print(f"{day} (Hijri {r['hijri_date']}), method {method}, Makkah local (UTC):")
    for n in TIMES:
        u = db.utc(r[n])
        print(f"  {n:8s} {u.astimezone(MAKKAH_TZ):%H:%M}  ({u:%H:%M} UTC)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aladhan prayer times (Umm al-Qura) for the Haram.")
    parser.add_argument("--ensure", action="store_true", help="fetch missing months (current + next) into the DB")
    parser.add_argument("--days-ahead", type=int, default=31)
    parser.add_argument("--today", action="store_true", help="print today's times (Makkah local)")
    parser.add_argument("--at", help="UTC time, e.g. 2026-10-23T12:30: previous/next prayer")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    engine = db.init_db()
    if args.ensure:
        print(f"Aladhan calls: {ensure_prayer_times(args.days_ahead, engine=engine)}")
    if args.today:
        print_day(makkah_today(), engine=engine)
    if args.at:
        ts = db.utc(datetime.fromisoformat(args.at))
        c = prayer_context(ts, engine=engine)
        print(c if c is None else f"{ts:%Y-%m-%d %H:%M}Z: {c.minutes_since_prev:.0f} min after {c.prev_prayer} "
              f"({c.prev_utc:%H:%M}Z), {c.minutes_to_next:.0f} min before {c.next_prayer} ({c.next_utc:%H:%M}Z)")
    if not (args.ensure or args.today or args.at):
        parser.print_help()


if __name__ == "__main__":
    main()
