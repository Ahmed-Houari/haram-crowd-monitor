"""Put the Mataf readings from before v3's deploy on v3's scale (user request 2026-10-29), so the series stays one series
and the forecast's start (train from 2026-11-01) does not move.

The factors come from frames both models saw: every placed prayer-window frame on disk (data/finetune/<window>.json, its
camera as the live service placed it, the live model's density map at the time in <window>.npz). Per frame the old
reading (the original DM-Count with the live head-size correction and camera factor, edge/m2_pipeline) and v3's (its
own density, no correction, no camera factor: the live path since the deploy) of each ring; a factor sum(v3) / sum(old),
area-weighted, per camera, ring and period (user, 2026-10-29): "prayer" = within PRAYER_WINDOW_MIN after an adhan (the
live service's prayer window), "other" = the rest. Too few frames for a camera and period: that camera's factor over both
periods, then all cameras'. Frames with prayer rows are left out: their live reading is the rows estimator for both
models (edge/shot.py), so rows frames keep their value (factor 1).

Applied to crowd_m2_frame / crowd_m2_minute rows before --until (the shot service's restart on v3): every calibrated
frame row by its camera's and period's factor (rows frames: 1), every minute by the ratio of its own frame rows (new /
old, area-weighted), else by the factor of its cameras and period. The far Mataf the same way with the "far" factor.
Nothing is overwritten (user, 2026-10-29): people_m2 / people_m2_far keep the original value; the v3-scale value goes to
people_m2_v3 / people_m2_far_v3, which the readers use where it is set (edge/db.m2_value / m2_col); going back = clearing
those columns. A backup of the database is made first (data/backups/pre_v3_<date>.db). The approximate path is not
stored in the DB (no rows to rescale).

  .venv/Scripts/python scripts/rescale_m2_to_v3.py                        # factors only
  .venv/Scripts/python scripts/rescale_m2_to_v3.py --apply --until "2026-10-29 01:30:00"
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge.kept_frames import frame_file  # noqa: E402

DB = ROOT / "data" / "haram.db"
FT = ROOT / "data" / "finetune"
FACTORS = ROOT / "data" / "jobs" / "v3_rescale_factors.json"
RINGS = ("all", "inner", "outer", "far")
MIN_FRAMES = 20  # per camera and period, else the camera's over both periods, else all cameras'


def prayer_windows() -> list[tuple[str, str]]:
    """[adhan, adhan + PRAYER_WINDOW_MIN) per prayer, UTC 'YYYY-MM-DD HH:MM:SS' (prayer_times table)."""
    from edge.prayer_rows import PRAYER_WINDOW_MIN

    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    out = []
    for row in con.execute("SELECT fajr, dhuhr, asr, maghrib, isha FROM prayer_times"):
        for t in row:
            if t:
                t0 = datetime.fromisoformat(t)
                out.append((t0.strftime("%Y-%m-%d %H:%M:%S"),
                            (t0 + timedelta(minutes=PRAYER_WINDOW_MIN)).strftime("%Y-%m-%d %H:%M:%S")))
    return sorted(out)


def period(ts: str, windows: list[tuple[str, str]]) -> str:
    return "prayer" if any(a <= ts[:19] < b for a, b in windows) else "other"


def factors(weights: str) -> dict:
    from edge import kaaba_geom as kg
    from edge.crowd_model import CrowdModel
    from edge.m2_metric import measure
    from edge.m2_pipeline import CAMERA_FACTOR, g_for, kaaba_exclusion
    from edge.view_matcher import graphics_mask, load_config

    gm = graphics_mask(load_config(), (1920, 1080))
    new = CrowdModel(weights)
    one = lambda h: np.ones_like(h)  # noqa: E731
    windows = prayer_windows()
    sums = defaultdict(lambda: [0.0, 0.0, 0])  # (camera, period, ring) -> [sum new*area, sum old*area, frames]
    per_frame = defaultdict(list)
    for p in sorted(FT.glob("*.json")):
        if p.stem.startswith("report") or p.stem.endswith("-test"):
            continue
        meta = json.loads(p.read_text())["frames"]
        dens_old = np.load(p.with_suffix(".npz"))["density"]
        for fr in meta:
            if "+rows" in fr["how"]:
                continue
            img = cv2.imread(str(frame_file(fr["stem"])))
            if img is None:
                continue
            c = fr["cam"]
            cam = kg.Camera(c["f"], np.asarray(c["rvec"], float), np.asarray(c["tvec"], float), tuple(c["size"]))
            ex = np.maximum(kaaba_exclusion(cam, (1080, 1920)), gm)
            old = measure(cam, dens_old[fr["i"]].astype(np.float32), ex, g_for("youtube"),
                          CAMERA_FACTOR.get(fr["camera"], 1.0))
            nv = measure(cam, new.predict(img).density, ex, one, 1.0)
            per = period(datetime.strptime(fr["stem"], "%Y%m%dT%H%M%SZ").strftime("%Y-%m-%d %H:%M:%S"), windows)
            for ring in RINGS:
                if ring == "far":
                    o, n, a = (old.far or {}).get("corr"), (nv.far or {}).get("corr"), (old.far or {}).get("area")
                else:
                    o, n, a = old.density_corr.get(ring), nv.density_corr.get(ring), old.area.get(ring)
                if o is None or n is None or not a:
                    continue
                for key in ((fr["camera"], per, ring), (fr["camera"], "both", ring), ("*", per, ring), ("*", "both", ring)):
                    sums[key][0] += n * a
                    sums[key][1] += o * a
                    sums[key][2] += 1
                    if o > 0.05:
                        per_frame[key].append(n / o)
        print(f"{p.stem}: done", flush=True)
    out: dict = {}
    for (camera, per, ring), (sn, so, n) in sorted(sums.items()):
        r = per_frame.get((camera, per, ring), [])
        out.setdefault(camera, {}).setdefault(per, {})[ring] = {
            "factor": round(sn / so, 4) if so > 0 else None, "frames": n,
            "per_frame_p10_p50_p90": [round(float(np.percentile(r, q)), 3) for q in (10, 50, 90)] if r else None}
    return out


def factor(fs: dict, camera: str | None, ring: str, per: str) -> float:
    for cam, p in ((camera or "", per), (camera or "", "both"), ("*", per), ("*", "both")):
        c = fs.get(cam, {}).get(p, {}).get(ring)
        if c and c["factor"] and (c["frames"] >= MIN_FRAMES or cam == "*"):
            return c["factor"]
    return 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", default="qnrf_ft3")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--until", help="UTC 'YYYY-MM-DD HH:MM:SS': rows before it are rescaled (the restart on v3)")
    args = ap.parse_args()
    if FACTORS.exists() and json.loads(FACTORS.read_text()).get("weights") == args.weights:
        fs = json.loads(FACTORS.read_text())["factors"]
    else:
        fs = factors(args.weights)
        FACTORS.write_text(json.dumps({"weights": args.weights, "factors": fs}, indent=1))
    for camera, pers in fs.items():
        for per, rings in pers.items():
            print(f"{camera:18s} {per:6s} " + "  ".join(
                f"{r} {v['factor']} (n {v['frames']}, p10-p90 {(v['per_frame_p10_p50_p90'] or [None])[0]}-"
                f"{(v['per_frame_p10_p50_p90'] or [None])[-1]})" for r, v in rings.items()))
    if not args.apply:
        print("factors only (--apply --until ... to write)")
        return
    if not args.until:
        sys.exit("--until is required with --apply")
    con = sqlite3.connect(DB, timeout=60)
    con.execute("PRAGMA busy_timeout=60000")
    for table in ("crowd_m2_frame", "crowd_m2_minute"):
        have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        if "people_m2_v3" in have and con.execute(f"SELECT count(*) FROM {table} WHERE people_m2_v3 IS NOT NULL").fetchone()[0]:
            sys.exit(f"already rescaled ({table}.people_m2_v3 is filled)")
    from edge.backup import BACKUP_DIR

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    dst = sqlite3.connect(BACKUP_DIR / f"pre_v3_{datetime.now(timezone.utc):%Y%m%d}.db")
    con.backup(dst)
    dst.close()
    for table in ("crowd_m2_frame", "crowd_m2_minute"):
        have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        for col in ("people_m2_v3", "people_m2_far_v3"):
            if col not in have:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {col} FLOAT")
    windows = prayer_windows()
    frames = con.execute("SELECT ts_utc, ring, camera, path, people_m2, area_m2, people_m2_far, area_m2_far FROM crowd_m2_frame "
                         "WHERE ts_utc < ?", (args.until,)).fetchall()
    fu, pool = [], defaultdict(lambda: [0.0, 0.0])  # (minute, ring) -> [sum new*area, sum old*area]
    for ts, ring, cam, path, v, area, vf, af in frames:
        rows = "+rows" in (path or "")
        per = period(ts, windows)
        q = 1.0 if rows else factor(fs, cam, ring, per)
        qf = 1.0 if rows else factor(fs, cam, "far", per)
        nv = round(v * q, 4)
        nf = round(vf * qf, 4) if vf is not None else None
        fu.append((nv, nf, ts, ring))
        m = ts[:16] + ":00.000000"
        if area:
            pool[(m, ring)][0] += nv * area
            pool[(m, ring)][1] += v * area
        if ring == "all" and vf is not None and af:
            pool[(m, "far")][0] += nf * af
            pool[(m, "far")][1] += vf * af
    minutes = con.execute("SELECT minute_utc, ring, source, people_m2, cameras, people_m2_far FROM crowd_m2_minute "
                          "WHERE minute_utc < ?", (args.until,)).fetchall()
    mu, how = [], defaultdict(int)
    for m, ring, source, v, cams, vf in minutes:
        def ratio(r: str) -> float:
            p = pool.get((m, r))
            if p and p[1] > 0:
                how["own frames"] += r == ring
                return p[0] / p[1]
            cs = [c for c in (cams or "").split(",") if c and c != "kaaba"] or [None]
            how["camera factor"] += r == ring
            return float(np.mean([factor(fs, c, r, period(m, windows)) for c in cs]))
        mu.append((round(v * ratio(ring), 4), round(vf * ratio("far"), 4) if vf is not None else None, m, ring, source))
    with con:
        con.executemany("UPDATE crowd_m2_frame SET people_m2_v3 = ?, people_m2_far_v3 = ? WHERE ts_utc = ? AND ring = ?", fu)
        con.executemany("UPDATE crowd_m2_minute SET people_m2_v3 = ?, people_m2_far_v3 = ? WHERE minute_utc = ? AND ring = ? "
                        "AND source = ?", mu)
    old = np.array([x[3] for x in minutes if x[1] == "all"])
    new = np.array([r[0] for r, x in zip(mu, minutes) if x[1] == "all"])
    print(f"written: {len(fu)} frame rows, {len(mu)} minute rows ({dict(how)}); ring all: median {np.median(old):.3f} -> "
          f"{np.median(new):.3f} people/m2; v3 values in people_m2_v3 (people_m2 unchanged), backup "
          f"{BACKUP_DIR.name}/pre_v3_*.db")


if __name__ == "__main__":
    main()
