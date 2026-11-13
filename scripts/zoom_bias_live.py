"""Check / refit the head-size correction g(h) on live frames (research, 2026-10-24; read-only unless --write).

scripts/zoom_bias.py fits g(h) = 1 - exp(-(h / h0)^k) on digitally zoomed reference frames, a proxy: digital zoom adds
no detail. Live readings store the raw count and floor area per head-size bin (shadow runs: bins_all; the collector:
crowd_m2_frame.head_bins since 2026-10-25), so any (h0, k) can be replayed on real frames: the corrected people per
m^2 of readings close in time should agree between different cameras and between zoom levels of the same camera.
This grid-searches (h0, k) for the smallest median differences on the fit data and reports them on the check data.
Collector frames stored before the bins existed get two bins from their inner / outer rings (raw count, area and
median head width per ring): an approximation.
--write stores the best pair in models/zoom_bias.json (the synthetic one stays under "synthetic").

  .venv/Scripts/python scripts/zoom_bias_live.py --fit data/shadow/m2_frames_run4.csv --check data/shadow/m2_frames_run5.csv
  .venv/Scripts/python scripts/zoom_bias_live.py --live 12 [--since "2026-10-25 06:00"] [--day]
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge.m2_metric import G_FLOOR, HEAD_BINS  # noqa: E402
from edge.m2_pipeline import ZOOM_BIAS, load_zoom_bias  # noqa: E402

MID = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 7, 9, 11.5, 15, 21, 30])  # px, a middle value per HEAD_BINS bin
assert len(MID) == len(HEAD_BINS) - 1
DT_S = 300
MAKKAH = timezone(timedelta(hours=3))


def load(paths: list[Path]) -> list[tuple]:
    """Shadow runs: (time, camera, zoom, bin middles, raw count per bin, area per bin) of the ptz readings."""
    out = []
    for p in paths:
        for r in csv.DictReader(open(p)):
            if r.get("bins_all") and r.get("path") == "ptz":
                c, a = json.loads(r["bins_all"])
                if sum(a) > 40:
                    out.append((datetime.fromisoformat(r["ts_utc"]), r["ptz_view"], float(r["ptz_zoom"] or 1), MID,
                                np.array(c), np.array(a)))
    return sorted(out, key=lambda x: x[0])


def load_live(db: Path, since: datetime, day_only: bool = False) -> list[tuple]:
    """Collector readings (crowd_m2_frame, ptz paths) since `since`, in the same shape as load()."""
    con = sqlite3.connect(f"file:{db.resolve().as_posix()}?mode=ro", uri=True)
    cols = {r[1] for r in con.execute("pragma table_info(crowd_m2_frame)")}
    bins = "head_bins" if "head_bins" in cols else "null"  # the column comes with the collector start of 2026-10-25
    rows = con.execute(f"select ts_utc, ring, camera, zoom, people_m2_raw, area_m2, head_px, {bins} from crowd_m2_frame"
                       " where (path like 'ptz%' or path = 'anchor') and ts_utc >= ? order by ts_utc",
                       (since.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),)).fetchall()
    frames: dict[str, dict] = {}
    for ts, ring, cam, zoom, raw, area, hp, bins in rows:
        f = frames.setdefault(ts, {"cam": cam, "zoom": zoom or 1.0})
        f[ring] = (raw, area, hp, bins)
    out = []
    for ts, f in frames.items():
        t = datetime.fromisoformat(str(ts)[:19]).replace(tzinfo=timezone.utc)
        if day_only and not 6 <= t.astimezone(MAKKAH).hour < 17:
            continue
        if "all" in f and f["all"][3]:
            c, a = json.loads(f["all"][3])
            out.append((t, f["cam"], f["zoom"], MID, np.array(c), np.array(a)))
        elif "inner" in f and "outer" in f and all(f[r][0] is not None and f[r][2] for r in ("inner", "outer")):
            (ri, ai, hi, _), (ro, ao, ho, _) = f["inner"], f["outer"]
            out.append((t, f["cam"], f["zoom"], np.array([hi, ho]), np.array([ri * ai, ro * ao]), np.array([ai, ao])))
    return [x for x in sorted(out, key=lambda x: x[0]) if x[5].sum() > 40]


def spreads(frames: list[tuple], h0: float, k: float) -> tuple[float, float, int, int]:
    """(median % difference between cameras, between zoom levels > 20 % apart of one camera, n pairs, n pairs)."""
    d = []
    for t, v, z, mid, c, a in frames:
        g = 1 - np.exp(-np.power(mid / h0, k))
        d.append((t, v, z, (c / np.maximum(g, G_FLOOR)).sum() / a.sum()))
    cams, zooms = [], []
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            if (d[j][0] - d[i][0]).total_seconds() > DT_S:
                break
            if d[i][3] <= 0 or d[j][3] <= 0:
                continue
            x = abs(np.log(d[i][3] / d[j][3]))
            if d[i][1] != d[j][1]:
                cams.append(x)
            elif abs(np.log(d[i][2] / d[j][2])) > np.log(1.2):
                zooms.append(x)
    med = lambda a: float(np.expm1(np.median(a)) * 100) if a else float("nan")  # noqa: E731
    return med(cams), med(zooms), len(cams), len(zooms)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fit", type=Path, nargs="*", default=[])
    ap.add_argument("--check", type=Path, nargs="*", default=[])
    ap.add_argument("--live", type=float, metavar="HOURS", help="fit on the collector's readings of the last HOURS")
    ap.add_argument("--since", help="with --live: Makkah time 'YYYY-MM-DD HH:MM' instead of HOURS ago")
    ap.add_argument("--day", action="store_true", help="with --live: 06:00-17:00 Makkah only")
    ap.add_argument("--db", type=Path, default=ROOT / "data" / "haram.db")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    if args.live:
        since = (datetime.strptime(args.since, "%Y-%m-%d %H:%M").replace(tzinfo=MAKKAH) if args.since
                 else datetime.now(timezone.utc) - timedelta(hours=args.live))
        fit = load_live(args.db, since, args.day)
    else:
        fit = load(args.fit)
    chk = load(args.check)
    h0s, ks = load_zoom_bias()
    c0, z0, nc, nz = spreads(fit, h0s, ks)
    print(f"fit: {len(fit)} frames ({nc} camera pairs, {nz} zoom pairs), check: {len(chk)} frames")
    for name, (h0, k) in (("current", (h0s, ks)), ("synthetic", (3.51, 1.79)), ("none", (1e-3, 1.0))):
        c, z, _, _ = spreads(fit, h0, k)
        line = f"{name:9s} h0 {h0:.2f} k {k:.2f}: fit cameras {c:.1f} %, zoom {z:.1f} %"
        if chk:
            cc, zz, _, _ = spreads(chk, h0, k)
            line += f"; check cameras {cc:.1f} %, zoom {zz:.1f} %"
        print(line)
    grid = []
    for h0 in np.arange(2.0, 9.01, 0.25):
        for k in (1.0, 1.2, 1.4, 1.6, 1.79, 2.0, 2.4, 2.8):
            c, z, _, _ = spreads(fit, h0, k)
            grid.append((np.nansum([c, z]), c, z, float(h0), float(k)))
    grid.sort()
    for _, c, z, h0, k in grid[:5]:
        line = f"h0 {h0:.2f} k {k:.2f}: fit cameras {c:.1f} %, zoom {z:.1f} %"
        if chk:
            cc, zz, ncc, nzz = spreads(chk, h0, k)
            line += f"; check cameras {cc:.1f} % (n {ncc}), zoom {zz:.1f} % (n {nzz})"
        print(line)
    if args.write:
        _, c, z, h0, k = grid[0]
        d = json.loads(ZOOM_BIAS.read_text())
        d.setdefault("synthetic", {"h0": d["h0"], "k": d["k"]})
        d.update(h0=round(h0, 3), k=round(k, 3), live_refit={"fit": [str(p) for p in args.fit] or f"live {args.live} h",
                                                             "check": [str(p) for p in args.check],
                                                             "cameras_pct": round(c, 1), "zoom_pct": round(z, 1)})
        ZOOM_BIAS.write_text(json.dumps(d, indent=1))
        print("written", ZOOM_BIAS.relative_to(ROOT))


if __name__ == "__main__":
    main()
