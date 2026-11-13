"""Evaluate the shadow run of the view-independent metric (research, 2026-10-24; read-only).

From data/shadow/m2_frames.csv (scripts/shadow_m2.py) and the prayer times in the database:
  coverage      counted minutes per hour, new metric vs the current one, by hour and day / night (Maghrib-sunrise)
  consistency   |log ratio| of people per m^2 for readings close in time (dt <= 60 / 120 / 300 s):
                  - different cameras (the key test: same moment, different framings)
                  - same camera, zoom differing by > 20 % (real optical zoom: tests the head-size correction)
                  - same camera, similar zoom (noise floor: the crowd changes, the model is noisy)
                  - the same frame through both camera paths (ptz vs Kaaba fit)
                and the current metric's spread for readings of different cameras (its density is per 10k px of
                each camera's own reference view, so it is not designed to agree across cameras)
  speed / VRAM  per stage, median and 90th percentile
  fallback      frames without a camera: DINOv2 framing clusters, each scaled to calibrated readings within
                +-5 min (factor = people per m^2 / raw people per 10k px); stable clusters would add coverage
Writes data/shadow/eval_m2.json and prints the report. With --live HOURS the same report (no speed, no fallback) comes
from the live tables instead (crowd_m2_frame vs crowd_frame, alive minutes from collector_minute; read-only) and
goes to data/shadow/eval_m2_live.json.

  .venv/Scripts/python scripts/eval_m2.py [--csv data/shadow/m2_frames.csv] [--live 24]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MAKKAH = timezone(timedelta(hours=3))


def fnum(x: str) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def night_flag(ts: list[datetime]) -> list[bool]:
    """Maghrib..sunrise from the prayer times table (read-only)."""
    from edge import prayer_times

    lo, hi = min(ts) - timedelta(days=1), max(ts) + timedelta(days=1)
    ev = prayer_times.load_events(lo, hi, include_sunrise=True)
    out = []
    for t in ts:
        prev = [n for e, n in ev if e <= t]
        out.append(bool(prev) and prev[-1] in ("maghrib", "isha", "fajr"))
    return out


def rows_from_db(db_path: Path, hours: float) -> list[dict]:
    """The live tables in the shadow CSV's shape: one row per alive collector minute, per new-metric frame
    (crowd_m2_frame, ring "all") and per counted current-metric frame (crowd_frame, densest zone)."""
    from sqlalchemy import create_engine, text

    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    e = create_engine(f"sqlite:///file:{db_path.resolve().as_posix()}?mode=ro&uri=true")
    with e.connect() as c:
        alive = c.execute(text("select minute_utc from collector_minute where minute_utc >= :s and frames > 0"),
                          {"s": since}).all()
        new = c.execute(text("select ts_utc, path, camera, people_m2, people_m2_raw, zoom, cam_x, cam_y "
                             "from crowd_m2_frame where ring = 'all' and ts_utc >= :s"), {"s": since}).all()
        old = c.execute(text("select ts_utc, view_id, max(density) from crowd_frame where accepted and ts_utc >= :s "
                             "group by ts_utc, view_id"), {"s": since}).all()

    def iso(t) -> str:
        return datetime.fromisoformat(str(t)[:19]).replace(tzinfo=timezone.utc).isoformat()

    rows = [{"ts_utc": iso(m), "old_accepted": "0"} for (m,) in alive]
    for t, path, cam, pm, raw, zoom, cx, cy in new:
        p, _, k = path.partition("/k")
        rows.append({"ts_utc": iso(t), "old_accepted": "0", "path": p, "ptz_depth": k, "ptz_view": cam or "",
                     "m2c_all": pm, "m2_all": raw, "ptz_zoom": zoom, "cam_x": cx, "cam_y": cy})
    rows += [{"ts_utc": iso(t), "old_accepted": "1", "old_view": v, "old_density": d} for t, v, d in old]
    rows.sort(key=lambda r: r["ts_utc"])
    return rows


def spread(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    a = np.abs(np.array(vals))
    return {"n": len(a), "median_pct": round(float(np.expm1(np.median(a)) * 100), 1),
            "p90_pct": round(float(np.expm1(np.percentile(a, 90)) * 100), 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, default=ROOT / "data" / "shadow" / "m2_frames.csv")
    ap.add_argument("--live", type=float, metavar="HOURS", help="evaluate the live tables of the last HOURS instead")
    ap.add_argument("--db", type=Path, default=ROOT / "data" / "haram.db")
    ap.add_argument("--embed", type=Path, nargs="*", help="embedding files of an archived run (default: the current)")
    ap.add_argument("--paths", nargs="*", help="count only readings of these camera paths (e.g. ptz: what goes live)")
    args = ap.parse_args()
    rows = rows_from_db(args.db, args.live) if args.live else list(csv.DictReader(open(args.csv)))
    if args.paths:  # readings of other paths count as frames without a reading
        for r in rows:
            if r.get("path") and r["path"] not in args.paths:
                r["m2c_all"] = r["m2_all"] = ""
    for r in rows:
        r["t"] = datetime.fromisoformat(r["ts_utc"])
    nights = night_flag([r["t"] for r in rows])
    for r, n in zip(rows, nights):
        r["night"] = n
    rep: dict = {"frames": len(rows), "from_utc": rows[0]["ts_utc"], "to_utc": rows[-1]["ts_utc"]}

    # ---------- coverage
    per_hour = defaultdict(lambda: {"frames": 0, "new": set(), "old": set(), "minutes": set(), "night": 0})
    for r in rows:
        hk = r["t"].astimezone(MAKKAH).strftime("%m-%d %H")
        mk = r["t"].replace(second=0)
        h = per_hour[hk]
        h["frames"] += 1
        h["minutes"].add(mk)
        h["night"] += r["night"]
        if fnum(r.get("m2c_all")) is not None:
            h["new"].add(mk)
        if r["old_accepted"] == "1":
            h["old"].add(mk)
    print(f"{'Live tables' if args.live else 'Shadow run'} {rows[0]['t'].astimezone(MAKKAH):%m-%d %H:%M} -> "
          f"{rows[-1]['t'].astimezone(MAKKAH):%m-%d %H:%M} Makkah, {len(rows)} rows")
    print(f"{'hour':>9} {'alive':>5} {'new':>4} {'old':>4}  (counted minutes)")
    tot = {"day": [0, 0, 0], "night": [0, 0, 0]}
    hours = []
    for hk in sorted(per_hour):
        h = per_hour[hk]
        alive = len(h["minutes"])
        period = "night" if h["night"] > h["frames"] / 2 else "day"
        print(f"{hk:>9} {alive:>5} {len(h['new']):>4} {len(h['old']):>4}  {period}")
        hours.append({"hour": hk, "alive": alive, "new": len(h["new"]), "old": len(h["old"]), "period": period})
        t = tot[period]
        t[0] += alive
        t[1] += len(h["new"])
        t[2] += len(h["old"])
    rep["coverage"] = {}
    for p, (alive, new, old) in tot.items():
        if alive:
            rep["coverage"][p] = {"alive_minutes": alive, "new_per_hour": round(60 * new / alive, 1),
                                  "old_per_hour": round(60 * old / alive, 1)}
            print(f"  {p}: new {60 * new / alive:.1f} counted min/h, current {60 * old / alive:.1f} (over {alive} live minutes)")
    rep["hours"] = hours
    paths = defaultdict(int)
    for r in rows:
        if fnum(r.get("m2c_all")) is not None:  # ptz readings by link count (ptz/k1: through a keyframe or extra ref)
            paths[r["path"] + (f"/k{r['ptz_depth']}" if r["path"] == "ptz" and r.get("ptz_depth") not in (None, "", "0")
                               else "")] += 1
    rep["readings_by_path"] = dict(paths)
    print("  readings by path:", dict(paths))
    checks = defaultdict(int)  # ptz cameras checked on the Kaaba (shadow runs from 2026-10-24 22:00)
    for r in rows:
        if r.get("ptz_check"):
            checks[r["ptz_check"]] += 1
    if checks:
        al = np.array([[fnum(r["align_rec"]), fnum(r["align_prec"])] for r in rows
                       if fnum(r.get("align_rec")) is not None and fnum(r.get("align_prec")) is not None])
        rep["ptz_checks"] = {"counts": dict(checks),
                             "align_rec_prec_median": np.median(al, 0).round(3).tolist() if len(al) else None}
        print("  ptz checks on the Kaaba:", rep["ptz_checks"])

    # ---------- consistency
    rd = [r for r in rows if fnum(r.get("m2c_all")) is not None and fnum(r.get("m2c_all")) > 0]
    for r in rd:
        r["cam_key"] = (r["ptz_view"] if r["path"] == "ptz" else r.get("anchor_view") or r["ptz_view"]
                        if r["path"] == "anchor" else f"kaaba@{round(fnum(r['cam_x']) or 0, -1)},{round(fnum(r['cam_y']) or 0, -1)}")
        r["zoomv"] = fnum(r["ptz_zoom"]) or 1.0
    for r in rd:
        sec = r.get("sectors_all")
        r["sect"] = np.array(json.loads(sec)) if sec else None
    cons: dict = {}
    for dt in (60, 120, 300):
        diff_cam, zoom_pair, same, diff_cam_raw, diff_cam_sect = [], [], [], [], []
        for i, a in enumerate(rd):
            for b in rd[i + 1:]:
                if (b["t"] - a["t"]).total_seconds() > dt:
                    break
                lc = np.log(fnum(a["m2c_all"]) / fnum(b["m2c_all"]))
                lr = np.log(max(fnum(a["m2_all"]), 1e-6) / max(fnum(b["m2_all"]), 1e-6))
                if a["cam_key"] != b["cam_key"]:
                    diff_cam.append(lc)
                    diff_cam_raw.append(lr)
                    if a["sect"] is not None and b["sect"] is not None:  # only the floor both cameras see
                        common = (a["sect"][1] >= 30) & (b["sect"][1] >= 30)
                        if common.any():
                            da = a["sect"][0][common].sum() / a["sect"][1][common].sum()
                            db_ = b["sect"][0][common].sum() / b["sect"][1][common].sum()
                            if da > 0 and db_ > 0:
                                diff_cam_sect.append(np.log(da / db_))
                elif abs(np.log(a["zoomv"] / b["zoomv"])) > np.log(1.2):
                    zoom_pair.append((lc, lr))
                else:
                    same.append(lc)
        cons[dt] = {"different_cameras": spread(diff_cam), "different_cameras_raw": spread(diff_cam_raw),
                    "different_cameras_common_sectors": spread(diff_cam_sect),
                    "same_camera_zoomed": spread([z[0] for z in zoom_pair]),
                    "same_camera_zoomed_raw": spread([z[1] for z in zoom_pair]),
                    "same_camera_similar_zoom": spread(same)}
    old = [r for r in rows if r["old_accepted"] == "1" and fnum(r["old_density"])]
    old_pairs = defaultdict(list)
    for i, a in enumerate(old):
        for b in old[i + 1:]:
            d = (b["t"] - a["t"]).total_seconds()
            if d > 300:
                break
            if a["old_view"] != b["old_view"]:
                for dt in (60, 120, 300):
                    if d <= dt:
                        old_pairs[dt].append(np.log(fnum(a["old_density"]) / fnum(b["old_density"])))
    for dt in (60, 120, 300):
        cons[dt]["current_metric_different_cameras"] = spread(old_pairs[dt])
    xk = [np.log(fnum(r["xk_m2c_all"]) / fnum(r["m2c_all"])) for r in rd if fnum(r.get("xk_m2c_all"))]
    cons["same_frame_ptz_vs_kaaba_fit"] = spread(xk)
    rep["consistency"] = cons
    print("\nConsistency, |difference| of people per m^2 (median / 90th percentile, %):")
    for dt in (60, 120, 300):
        c = cons[dt]
        print(f"  dt <= {dt:3d} s: different cameras {c['different_cameras']} (raw {c['different_cameras_raw']})")
        print(f"               different cameras, common sectors only: {c['different_cameras_common_sectors']}")
        print(f"               same camera zoomed > 20 %: {c['same_camera_zoomed']} (raw {c['same_camera_zoomed_raw']})")
        print(f"               same camera, similar zoom (noise floor): {c['same_camera_similar_zoom']}")
        print(f"               current metric, different cameras: {c['current_metric_different_cameras']}")
    print(f"  same frame, ptz camera vs Kaaba-fit camera: {cons['same_frame_ptz_vs_kaaba_fit']}")

    # ---------- speed / VRAM
    sp = {}
    for k in ("ms_ptz", "ms_detect", "ms_sam", "ms_fit", "ms_model", "ms_total"):
        v = [fnum(r[k]) for r in rows if fnum(r.get(k)) is not None]
        if v:
            sp[k] = {"median": round(float(np.median(v))), "p90": round(float(np.percentile(v, 90))), "n": len(v)}
    vr = [fnum(r["vram_gb"]) for r in rows if fnum(r.get("vram_gb"))]
    sp["vram_gb_max"] = max(vr) if vr else None
    rep["speed"] = sp
    print("\nSpeed (ms) and VRAM:", sp)

    # ---------- fallback for framings without a camera
    if not args.live:
        try:
            rep["fallback"] = fallback(rows, args.embed)
        except Exception as e:  # embeddings missing etc.
            rep["fallback"] = {"error": str(e)}
        print("\nFallback (frames without a camera):", rep["fallback"])
    out = ROOT / "data" / "shadow" / ("eval_m2_live.json" if args.live else "eval_m2.json")
    out.write_text(json.dumps(rep, indent=1, default=str))
    print("written", out.relative_to(ROOT))


def fallback(rows: list[dict], files: list[Path] | None = None) -> dict:
    from sklearn.cluster import KMeans

    emb = {}
    for p in files or (ROOT / "data" / "shadow").glob("m2_embed_*.npy"):  # m2_embed_<UTC date>[_<run>].npy
        emb[p.stem.split("_")[2]] = np.load(p).astype(np.float32)
    X, idx = [], []
    for i, r in enumerate(rows):
        day, k = r["emb_row"].split(":")
        if day in emb and int(k) < len(emb[day]):
            X.append(emb[day][int(k)])
            idx.append(i)
    X = np.array(X)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    nocam = [j for j, i in enumerate(idx) if not rows[i]["path"] and fnum(rows[i].get("raw_per_10kpx"))]
    if len(nocam) < 20:
        return {"frames_without_camera": len(nocam), "note": "too few frames yet"}
    k = min(12, max(3, len(nocam) // 25))
    km = KMeans(k, n_init=10, random_state=0).fit(X[nocam])
    calib = [i for i in idx if fnum(rows[i].get("m2c_all"))]
    out = {"frames_without_camera": len(nocam), "clusters": []}
    for c in range(k):
        members = [idx[nocam[j]] for j in np.nonzero(km.labels_ == c)[0]]
        factors = []
        for i in members:
            t = rows[i]["t"]
            near = [fnum(rows[q]["m2c_all"]) for q in calib if abs((rows[q]["t"] - t).total_seconds()) <= 300]
            raw = fnum(rows[i]["raw_per_10kpx"])
            if near and raw and raw > 0:
                factors.append(float(np.median(near)) / raw)
        f = np.array(factors)
        cv = float(f.std() / f.mean()) if len(f) >= 4 and f.mean() > 0 else None
        out["clusters"].append({"cluster": c, "frames": len(members), "with_calibrated_neighbour": len(f),
                                "factor_cv": round(cv, 3) if cv is not None else None,
                                "stable": bool(cv is not None and cv < 0.2)})
    stable = sum(c["frames"] for c in out["clusters"] if c["stable"])
    out["frames_in_stable_clusters"] = stable
    return out


if __name__ == "__main__":
    main()
