"""Put the Mataf readings from before the metric's change (2026-10-26 ~19:02 Makkah) on the new scale (user request).

The change: head-size curve h0 6.5 / k 1.2 -> 5.17 / 1.96, camera factors (low 1.0, wide 1.0, roof 1.5,
config/m2.json "correction"), prayer rows (edge/prayer_rows.py), ~1 frame/s from the live shot service. Every old frame
row of ring "all" stores its raw count and floor area per head-width bin (head_bins), so its corrected value can be
recomputed with the new curve and factor: sum over bins raw / (max(g(bin middle), 0.2) x factor) / area. The inner and
outer rings take the same frame's ratio new / old. Minutes are pooled again from the frames, as the collector pooled
them (area-weighted). Prayer rows cannot be recovered without the frames: old minutes within PRAYER_WINDOW_MIN after an
adhan get note "prayer_v1" and stay out of the levels, the usual and the forecast (edge/db.m2_5min).

Nothing is deleted: the original value stays in people_m2_v1. A backup of the database is made first
(data/backups/pre_rescale_<date>.db, removed by scripts/housekeeping.py after 7 days).

  .venv/Scripts/python scripts/rescale_m2_history.py            # report only
  .venv/Scripts/python scripts/rescale_m2_history.py --apply
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge.m2_metric import G_FLOOR, HEAD_BINS, load_correction  # noqa: E402
from edge.prayer_rows import PRAYER_WINDOW_MIN  # noqa: E402

DB = ROOT / "data" / "haram.db"
CHANGE_UTC = "2026-10-26 16:02:30"  # the crowd collector's m2 path off, the shot service with the new correction on
OLD = (6.5, 1.2)
MIDS = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 7.0, 9.0, 11.5, 15.0, 21.0, 30.0])  # head px per HEAD_BINS bin


def g(h, h0, k):
    return np.maximum(1 - np.exp(-np.power(np.maximum(h, 0.1) / h0, k)), G_FLOOR)


def prayer_windows(con: sqlite3.Connection) -> list[tuple[str, str]]:
    out = []
    for row in con.execute("SELECT fajr, dhuhr, asr, maghrib, isha FROM prayer_times WHERE date >= '2026-10-20' AND date <= '2026-10-27'"):
        for t in row:
            t0 = datetime.fromisoformat(t)
            out.append((t0.strftime("%Y-%m-%d %H:%M:%S"), (t0 + timedelta(minutes=PRAYER_WINDOW_MIN)).strftime("%Y-%m-%d %H:%M:%S")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    assert len(MIDS) == len(HEAD_BINS) - 1
    h0, k, factors = load_correction()
    con = sqlite3.connect(DB, timeout=30)
    con.execute("PRAGMA busy_timeout=30000")
    cols = {r[1] for r in con.execute("PRAGMA table_info(crowd_m2_frame)")}
    if "people_m2_v1" in cols and con.execute("SELECT count(*) FROM crowd_m2_frame WHERE people_m2_v1 IS NOT NULL").fetchone()[0]:
        sys.exit("already rescaled (people_m2_v1 is filled)")
    frames = con.execute("SELECT ts_utc, ring, camera, people_m2, people_m2_raw, area_m2, head_px, head_bins FROM crowd_m2_frame "
                         "WHERE ts_utc < ? AND source = 'youtube'", (CHANGE_UTC,)).fetchall()
    by_ts = defaultdict(dict)
    for ts, ring, cam, v, raw, area, head, bins in frames:
        by_ts[ts][ring] = (cam, v, raw, area, head, bins)
    new_frame: dict[tuple[str, str], float] = {}
    check, how = [], defaultdict(int)
    for ts, rings in by_ts.items():
        a = rings.get("all")
        if a is None:
            continue
        cam, v, raw, area, head, bins = a
        q = factors.get(cam or "", 1.0)
        if bins:
            cnt, ar = (np.array(x, float) for x in json.loads(bins))
            old_b = float((cnt / g(MIDS, *OLD)).sum() / max(ar.sum(), 1e-6))
            new_all = float((cnt / (g(MIDS, h0, k) * q)).sum() / max(ar.sum(), 1e-6))
            check.append(old_b / v if v else np.nan)
            how["bins"] += 1
        else:  # early rows without bins: the median head width
            new_all = float(raw / (g(head, h0, k) * q)) if raw is not None and head else v
            how["median head"] += 1
        ratio = new_all / v if v else 1.0
        for ring, (_, rv, *_rest) in rings.items():
            new_frame[(ts, ring)] = round(new_all if ring == "all" else rv * ratio, 4)
    check = np.array([c for c in check if np.isfinite(c)])
    print(f"old frames {len(by_ts)} ({dict(how)}); old value recomputed from the bins / stored: median {np.median(check):.3f}, "
          f"p10-p90 {np.percentile(check, 10):.3f}-{np.percentile(check, 90):.3f}")
    # minutes, pooled again from the frames (area-weighted), and the prayer windows
    area = {(ts, ring): a for ts, rings in by_ts.items() for ring, (_, _, _, a, *_r) in rings.items()}
    pool = defaultdict(lambda: [0.0, 0.0, 0])
    for (ts, ring), v in new_frame.items():
        m = ts[:16] + ":00.000000"
        a = area[(ts, ring)] or 0.0
        pool[(m, ring)][0] += v * a
        pool[(m, ring)][1] += a
        pool[(m, ring)][2] += 1
    windows = prayer_windows(con)
    minutes = con.execute("SELECT minute_utc, ring, people_m2, n_frames FROM crowd_m2_minute WHERE minute_utc < ? AND source = 'youtube'",
                          (CHANGE_UTC[:16] + ":00",)).fetchall()
    upd, prayer, ratios, missing = [], 0, defaultdict(list), 0
    for m, ring, v, n in minutes:
        p = pool.get((m, ring))
        if p is None or p[1] <= 0:
            missing += 1
            continue
        nv = round(p[0] / p[1], 4)
        in_prayer = any(a <= m < b for a, b in windows)
        prayer += in_prayer
        upd.append((nv, "prayer_v1" if in_prayer else "rescaled_v1", m, ring))
        if ring == "all" and not in_prayer:
            ratios["all"].append(nv / v)
    r = np.array(ratios["all"])
    print(f"old minutes {len(minutes)}: rescaled {len(upd)} (of which in prayer windows, flagged: {prayer}), "
          f"no frames {missing}; new / old (ring all, outside prayers): median {np.median(r):.3f}, p10-p90 "
          f"{np.percentile(r, 10):.3f}-{np.percentile(r, 90):.3f}")
    per_cam = defaultdict(list)
    for ts, rings in by_ts.items():
        if "all" in rings and rings["all"][1]:
            per_cam[rings["all"][0]].append(new_frame[(ts, "all")] / rings["all"][1])
    for cam, rr in sorted(per_cam.items(), key=lambda x: str(x[0])):
        print(f"  {cam}: {len(rr)} frames, new / old median {np.median(rr):.3f}")
    if not args.apply:
        print("report only (--apply to write)")
        return
    from edge.backup import BACKUP_DIR

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    dst = sqlite3.connect(BACKUP_DIR / f"pre_rescale_{datetime.now(timezone.utc):%Y%m%d}.db")
    con.backup(dst)
    dst.close()
    for table in ("crowd_m2_frame", "crowd_m2_minute"):
        have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        if "people_m2_v1" not in have:
            con.execute(f"ALTER TABLE {table} ADD COLUMN people_m2_v1 FLOAT")
    if "note" not in {r[1] for r in con.execute("PRAGMA table_info(crowd_m2_minute)")}:
        con.execute("ALTER TABLE crowd_m2_minute ADD COLUMN note VARCHAR(16)")
    with con:
        con.executemany("UPDATE crowd_m2_frame SET people_m2_v1 = people_m2, people_m2 = ? WHERE ts_utc = ? AND ring = ? "
                        "AND source = 'youtube'", [(v, ts, ring) for (ts, ring), v in new_frame.items()])
        con.executemany("UPDATE crowd_m2_minute SET people_m2_v1 = people_m2, people_m2 = ?, note = ? WHERE minute_utc = ? "
                        "AND ring = ? AND source = 'youtube'", upd)
    print(f"written: {len(new_frame)} frame rows, {len(upd)} minute rows (originals in people_m2_v1); backup "
          f"{BACKUP_DIR.name}/pre_rescale_*.db")


if __name__ == "__main__":
    main()
