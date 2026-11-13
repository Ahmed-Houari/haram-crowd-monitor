"""Synthetic test database for step 7 (forecast + alerts): the real schema, fake data. Never the real DB.

  prayer_times   a smooth Makkah schedule (Isha = Maghrib + 90 min, as Umm al-Qura outside Ramadan), drifting a
                 little each day; Hijri dates counted from --hijri-start; --ramadan-days N marks the last N days as
                 Ramadan (month 9, bigger evening crowds) to exercise that feature.
  crowd_minute   per camera density (people per 10,000 zone px): a day/night profile, a build-up before every
                 prayer and dispersal after, a bigger Friday Dhuhr (Jumu'ah), busier Thu/Fri/Sat, a slow random
                 drift and a few unannounced surges (so time of day alone cannot explain everything). Readings exist
                 only while the camera is "on air": bursts of 2-6 min with gaps, like the real broadcast; the roof
                 camera is on air less often and not at night. Like the live collector since 2026-10-24, about
                 30 % of minutes show only part of the zone (visible 35-95 %, zoomed / panned shots): count = density
                 x visible zone area (config/views.json); people in the whole zone only for fully visible minutes.
  crowd_m2_minute  the view-independent Mataf series (people per m^2 per ring) whenever a camera is on air.
  traffic_area   1 km / 3 km congestion following the crowd ~20 min later, every 5 min (for the road alert).

Run:  .venv/Scripts/python scripts/make_synthetic_db.py [--days 21] [--out data/synthetic/haram_synth.db]
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge import db  # noqa: E402
from edge.prayer_times import METHOD, PrayerDay, save_days  # noqa: E402
from edge.view_matcher import load_config  # noqa: E402

MAKKAH = timezone(timedelta(hours=3))
REAL_DB = (ROOT / "data" / "haram.db").resolve()
# local Makkah prayer times on day 0 (hour, minute) and drift in minutes per day
SCHEDULE = {"fajr": (5, 3, 0.25), "sunrise": (6, 20, 0.3), "dhuhr": (12, 5, 0.05), "asr": (15, 24, -0.15),
            "maghrib": (17, 50, -0.4)}
BUMP = {"fajr": 0.35, "dhuhr": 0.42, "asr": 0.33, "maghrib": 0.55, "isha": 0.5}  # crowd swell around each prayer
CAMS = {  # view id: base density, mean off-air gap (min), off-air hours (Makkah)
    "cam0_mataf_low": (21.0, 16, ()),
    "cam5_roof": (28.0, 42, range(0, 5)),
}


def zone_area(view: dict) -> float:
    """Total zone area in reference px (shoelace)."""
    total = 0.0
    for p in view["zones"].values():
        a = np.asarray(p, float)
        total += abs(np.dot(a[:, 0], np.roll(a[:, 1], 1)) - np.dot(a[:, 1], np.roll(a[:, 0], 1))) / 2
    return total


def hijri_days(start: str, n: int, ramadan_days: int) -> list[str]:
    """Hijri dates for n consecutive days (months alternate 30/29 days); the last `ramadan_days` get month 9."""
    y, m, d = (int(x) for x in start.split("-"))
    out = []
    for i in range(n):
        month = 9 if i >= n - ramadan_days else m
        out.append(f"{y:04d}-{month:02d}-{d:02d}")
        d += 1
        if d > (30 if m % 2 else 29):
            d, m = 1, m % 12 + 1
            y += m == 1
    return out


def prayer_days(day0: date, n: int, hijri: list[str]) -> list[PrayerDay]:
    days = []
    for i in range(n):
        dt = day0 + timedelta(days=i)
        local = {}
        for name, (hh, mm, drift) in SCHEDULE.items():
            local[name] = datetime(dt.year, dt.month, dt.day, hh, mm, tzinfo=MAKKAH) + timedelta(minutes=drift * i)
        ramadan = hijri[i][5:7] == "09"
        local["isha"] = local["maghrib"] + timedelta(minutes=120 if ramadan else 90)
        utc = {k: v.astimezone(timezone.utc).replace(second=0, microsecond=0) for k, v in local.items()}
        days.append(PrayerDay(dt, METHOD, hijri[i], utc["fajr"], utc["sunrise"], utc["dhuhr"], utc["asr"],
                              utc["maghrib"], utc["isha"]))
    return days


def crowd_index(t_min: np.ndarray, days: list[PrayerDay], rng: np.random.Generator) -> np.ndarray:
    """Relative crowd level (around 1) per UTC minute (minutes since the Unix epoch)."""
    local_h = ((t_min + 180) % 1440) / 60
    x = 0.62 + 0.2 * np.cos(2 * np.pi * (local_h - 21) / 24)  # busier late evening, quietest late morning
    for d in days:
        ramadan = d.hijri_date[5:7] == "09"
        for name, amp in BUMP.items():
            p = getattr(d, name).timestamp() / 60
            if name == "dhuhr" and d.date.weekday() == 4:
                amp *= 1.9  # Friday prayer
            if ramadan and name in ("maghrib", "isha"):
                amp *= 1.5
            dt = t_min - p
            x += amp * np.exp(-((dt + 12) / 38) ** 2)  # fills ~45 min before, empties ~30 min after
    weekday = ((t_min + 180) // 1440 + 3) % 7  # 0 = Monday (1970-01-01 was a Thursday)
    x *= np.choose(weekday.astype(int), [1.0, 0.98, 0.98, 1.0, 1.06, 1.12, 1.07])
    ram = np.zeros_like(x)
    for d in days:
        if d.hijri_date[5:7] == "09":
            start = datetime(d.date.year, d.date.month, d.date.day, tzinfo=MAKKAH).timestamp() / 60
            ram[(t_min >= start) & (t_min < start + 1440)] = 1
    x *= 1 + 0.3 * ram
    drift = np.zeros_like(x)  # slow AR(1) drift in log space
    e = rng.normal(0, 0.008, len(x))
    for i in range(1, len(x)):
        drift[i] = 0.996 * drift[i - 1] + e[i]
    x *= np.exp(drift)
    for _ in range(int(len(x) / 1440 * 0.45)):  # unannounced surges (~3 a week): +20-35 % for 40-100 min
        c, w, a = rng.integers(0, len(x)), rng.integers(40, 100), rng.uniform(0.2, 0.35)
        idx = np.arange(len(x))
        x *= 1 + a * np.clip(1 - np.abs(idx - c) / w, 0, 1) ** 0.6
    return x


def on_air(n: int, mean_gap: float, off_hours, t_min: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Boolean per minute: bursts of 2-6 min, exponential gaps; never during `off_hours` (Makkah)."""
    on = np.zeros(n, bool)
    i = int(rng.integers(0, 20))
    while i < n:
        run = int(rng.integers(2, 7))
        on[i:i + run] = True
        i += run + max(3, int(rng.exponential(mean_gap)))
    if off_hours:
        h = ((t_min + 180) % 1440) // 60
        on &= ~np.isin(h, list(off_hours))
    return on


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--start", default="2026-09-01", help="first Makkah date")
    ap.add_argument("--hijri-start", default="1448-03-10")
    ap.add_argument("--ramadan-days", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "data" / "synthetic" / "haram_synth.db"))
    args = ap.parse_args()

    out = Path(args.out).resolve()
    if out == REAL_DB:
        sys.exit("refusing to write synthetic data into the real database")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    engine = db.init_db(create_engine(f"sqlite:///{out}"))
    rng = np.random.default_rng(args.seed)

    day0 = date.fromisoformat(args.start)
    hijri = hijri_days(args.hijri_start, args.days + 1, args.ramadan_days)
    pdays = prayer_days(day0 - timedelta(days=0), args.days + 1, hijri)  # +1 day: prayers after the last minute
    save_days(pdays, engine)

    t0 = datetime(day0.year, day0.month, day0.day, tzinfo=MAKKAH).astimezone(timezone.utc)
    n = args.days * 1440
    t_min = np.arange(n) + int(t0.timestamp() // 60)
    idx = crowd_index(t_min, pdays, rng)
    views = {v["id"]: v for v in load_config()["views"]}
    rows = []
    vis_rng = np.random.default_rng(args.seed + 1)  # own stream: the densities stay as before the visible-part metric
    for cam, (base, gap, off) in CAMS.items():
        dens = base * idx * (1 + rng.normal(0, 0.04, n))
        area = zone_area(views[cam]) if cam in views else 5e5
        live = on_air(n, gap, off, t_min, rng)
        vis = np.where(vis_rng.random(n) < 0.3, vis_rng.uniform(0.35, 0.95, n), 1.0)
        for i in np.nonzero(live)[0]:
            full = float(dens[i] * area / 1e4)
            rows.append(db.CrowdMinute(datetime.fromtimestamp(int(t_min[i]) * 60, timezone.utc), cam, "mataf_ground",
                                       full * float(vis[i]), float(dens[i]), float(1 + rng.normal(0, 0.015)),
                                       int(rng.integers(3, 5)), "youtube", float(vis[i]),
                                       full if vis[i] >= 1.0 else None))
    db.insert_crowd_minute(rows, engine)

    # view-independent Mataf series (people per m^2, all cameras together, 2026-10-24): whenever any camera is on air
    on_any = sorted({int(r.minute_utc.timestamp() // 60) for r in rows})
    m2_rows = []
    m2_rng = np.random.default_rng(args.seed + 2)
    for mi in on_any:
        i = mi - int(t_min[0])
        base = 0.55 * idx[i] * (1 + m2_rng.normal(0, 0.05))
        for ring, f in (("all", 1.0), ("inner", 1.25), ("outer", 0.92)):
            m2_rows.append({"minute_utc": datetime.fromtimestamp(mi * 60, timezone.utc), "ring": ring, "source": "youtube",
                            "people_m2": round(float(base * f), 4), "people_m2_raw": round(float(base * f * 0.78), 4),
                            "area_m2": {"all": 2300.0, "inner": 520.0, "outer": 1780.0}[ring],
                            "n_frames": int(m2_rng.integers(2, 5)), "cameras": "cam0_mataf_low,cam5_roof"})
    db.insert_m2([], m2_rows, engine)

    # roads follow the crowd ~20 min later (5-min polls)
    polls = np.arange(0, n, 5)
    lagged = idx[np.clip(polls - 20, 0, n - 1)]
    traffic = []
    for area, k in (("r1km", 30.0), ("r3km", 16.0)):
        pct = np.clip(k * (lagged - 0.7) + rng.normal(0, 2.5, len(polls)), 0, 95)
        for j, p in enumerate(polls):
            traffic.append({"poll_utc": datetime.fromtimestamp(int(t_min[p]) * 60, timezone.utc), "area": area,
                            "road_category": "all", "n_segments": 90 if area == "r1km" else 500,
                            "length_km": 32.8 if area == "r1km" else 200.1, "mean_level": float(1 - pct[j] / 160),
                            "congested_pct": float(pct[j]), "closed_pct": 0.0})
    with engine.begin() as conn:
        db.upsert(conn, db.traffic_area, traffic)
    print(f"{out}: {args.days} days from {day0}, {len(rows)} crowd minutes "
          f"({', '.join(f'{c}: {sum(r.view_id == c for r in rows)}' for c in CAMS)}), {len(traffic)} road rows, "
          f"Ramadan days: {args.ramadan_days}")


if __name__ == "__main__":
    main()
