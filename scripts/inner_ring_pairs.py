"""Inner-ring accuracy of the roof views against zoomed ground truth (investigation 2026-10-26, user request).

Truth: frames where the low or wide camera shows the ring with heads >= TRUTH_HEAD_PX (DM-Count was within 0.5-1.17x of
the prayer-row ground truth at 9-12 px). Paired with the roof camera / Roof camera B readings of the same ring within
+-WINDOW_S (crowd_m2_frame; the ring readings do not depend on which way round the Kaaba's frame was chosen, sectors
would). Per pair: truth people/m^2 (raw / g(head), the live curve), the roof's raw people/m^2 and head width, the roof
as read live (raw / (g x 1.5)), and what each candidate correction reads. Candidates are fitted on time blocks and
scored on the other blocks (leave-one-block-out), so a block's own pairs never tune its correction.

Prayer windows (PRAYER_BEFORE_MIN before an adhan to PRAYER_WINDOW_MIN after) are left out unless --with-prayers: people
stand, bow and prostrate in rows there, the model misses bent-over people (the roof read 0.00-0.10 on the inner ring
10-23 min after the Maghrib and Isha adhans while the zoomed views read 0.9-2.3), and the live metric uses the rows
estimator (edge/prayer_rows.py) in them. The question here is tawaf.

  .venv/Scripts/python scripts/inner_ring_pairs.py [--window 120] [--ring inner] [--since 2026-10-24] [--with-prayers]
"""
from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge.m2_metric import G_FLOOR, load_correction  # noqa: E402
from edge.prayer_rows import PRAYER_WINDOW_MIN  # noqa: E402

TRUTH_CAMS = ("cam0_mataf_low", "cam1_mataf_wide")
ROOF_CAMS = ("cam5_roof", "cam8_roof_b")
TRUTH_HEAD_PX = 9.0
MIN_AREA = {"inner": 20.0, "outer": 60.0}
BLOCK_MIN = 10
PRAYER_BEFORE_MIN = 10  # people already sit in rows waiting for the prayer


def g(h, h0, k):
    return np.maximum(1 - np.exp(-np.power(np.maximum(np.asarray(h, float), 0.1) / h0, k)), G_FLOOR)


def prayer_windows(con, since: str) -> list[tuple[datetime, datetime]]:
    out = []
    for row in con.execute("SELECT fajr, dhuhr, asr, maghrib, isha FROM prayer_times WHERE date >= ?", (since[:10],)):
        for t in row:
            if t:  # UTC, like ts_utc
                t0 = datetime.fromisoformat(t[:19])
                out.append((t0 - timedelta(minutes=PRAYER_BEFORE_MIN), t0 + timedelta(minutes=PRAYER_WINDOW_MIN)))
    return out


def load(con, ring: str, since: str, skip=()):
    out, dropped = [], 0
    for ts, cam, raw, area, head in con.execute(
            "SELECT ts_utc, camera, people_m2_raw, area_m2, head_px FROM crowd_m2_frame WHERE source = 'youtube' AND ring = ? "
            "AND ts_utc >= ? AND people_m2_raw IS NOT NULL AND head_px IS NOT NULL ORDER BY ts_utc", (ring, since)):
        if area and area >= MIN_AREA[ring]:
            t = datetime.fromisoformat(ts[:19])
            if any(a <= t < b for a, b in skip):
                dropped += 1
                continue
            out.append((t, cam, float(raw), float(area), float(head)))
    return out, dropped


def pairs(rows, window_s: float):
    truth = [r for r in rows if r[1] in TRUTH_CAMS and r[4] >= TRUTH_HEAD_PX]
    roof = [r for r in rows if r[1] in ROOF_CAMS]
    rt = [r[0] for r in roof]
    out = []
    for t in truth:
        i0 = bisect.bisect_left(rt, t[0] - timedelta(seconds=window_s))
        i1 = bisect.bisect_right(rt, t[0] + timedelta(seconds=window_s))
        if i1 > i0:
            rs = roof[i0:i1]
            a = sum(r[3] for r in rs)
            out.append({"t": t[0], "truth_cam": t[1], "truth_raw": t[2], "truth_head": t[4],
                        "roof_cam": max({r[1] for r in rs}, key=lambda c: sum(r[1] == c for r in rs)),
                        "roof_raw": sum(r[2] * r[3] for r in rs) / a, "roof_head": sum(r[4] * r[3] for r in rs) / a,
                        "n_roof": len(rs), "block": t[0].replace(minute=t[0].minute // BLOCK_MIN * BLOCK_MIN, second=0)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", type=float, default=120.0)
    ap.add_argument("--ring", default="inner", choices=["inner", "outer"])
    ap.add_argument("--since", default="2026-10-24")
    ap.add_argument("--with-prayers", action="store_true", help="keep the prayer windows")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    h0, k, factors = load_correction()
    con = sqlite3.connect(ROOT / "data" / "haram.db")
    rows, dropped = load(con, args.ring, args.since, () if args.with_prayers else prayer_windows(con, args.since))
    P = pairs(rows, args.window)
    if not P:
        sys.exit("no pairs")
    # one row per block (frames in a block are the same shots: averaged, so a long shot does not count many times)
    blocks = defaultdict(list)
    for p in P:
        blocks[p["block"]].append(p)
    B = []
    for b, ps in sorted(blocks.items()):
        truth = np.mean([p["truth_raw"] / g(p["truth_head"], h0, k) for p in ps])
        B.append({"block": b, "n": len(ps), "truth": float(truth), "roof_raw": float(np.mean([p["roof_raw"] for p in ps])),
                  "roof_head": float(np.mean([p["roof_head"] for p in ps])), "roof_cam": ps[0]["roof_cam"],
                  "truth_head": float(np.mean([p["truth_head"] for p in ps]))})
    T = np.array([b["truth"] for b in B]); R = np.array([b["roof_raw"] for b in B]); H = np.array([b["roof_head"] for b in B])
    q = np.array([factors.get(b["roof_cam"], 1.0) for b in B])
    live = R / (g(H, h0, k) * q)
    gonly = R / g(H, h0, k)
    print(f"{args.ring} ring, window +-{args.window:.0f} s: {len(P)} truth frames in {len(B)} ten-minute blocks"
          + ("" if args.with_prayers else f" (prayer windows left out: {dropped} frame rows)"))
    print("block (UTC)        n  truth  roof raw  head   live(/1.5)  g only")
    for b, lv, go in zip(B, live, gonly):
        print(f"{b['block']:%m-%d %H:%M}  {b['n']:4d}  {b['truth']:.2f}   {b['roof_raw']:.2f}    {b['roof_head']:.1f}   {lv:.2f}        {go:.2f}")

    def usable(idx):
        return np.array([i for i in idx if T[i] > 0 and R[i] > 0])

    def fit_const(idx):  # truth = raw / (g x c): c from the median log ratio
        idx = usable(idx)
        return float(np.exp(np.median(np.log(gonly[idx] / T[idx]))))

    def fit_sat(idx):  # truth = raw / (g x c x (1 - raw / rmax)): the model saturates as the crowd packs
        idx = usable(idx)

        def res(p):
            c, rmax = np.exp(p)
            pred = R[idx] / (g(H[idx], h0, k) * c * np.clip(1 - R[idx] / rmax, 0.15, 1))
            return np.log(pred) - np.log(T[idx])
        s = least_squares(res, [0.0, np.log(max(3.0, 2 * R[idx].max()))], loss="soft_l1", f_scale=0.2)
        return np.exp(s.x)

    def fit_loglin(idx):  # truth = raw / (g x exp(b0 + b1 log(head / 4 px) + b2 log(raw / 0.5))): head size AND density
        idx = usable(idx)
        X = np.column_stack([np.ones(len(idx)), np.log(H[idx] / 4), np.log(R[idx] / 0.5)])
        y = np.log(gonly[idx] / T[idx])
        return least_squares(lambda b: X @ b - y, np.zeros(3), loss="soft_l1", f_scale=0.2).x

    def loglin(b, i):
        return gonly[i] / np.exp(b[0] + b[1] * np.log(H[i] / 4) + b[2] * np.log(max(R[i], 1e-3) / 0.5))

    n = len(B)
    cv = {"live (/1.5)": [], "g only": [], "constant": [], "saturation": [], "head+density": []}
    for i in range(n):
        idx = np.array([j for j in range(n) if j != i])
        cv["live (/1.5)"].append(live[i])
        cv["g only"].append(gonly[i])
        if len(usable(idx)) >= 4:
            cv["constant"].append(gonly[i] / fit_const(idx))
            c, rmax = fit_sat(idx)
            cv["saturation"].append(R[i] / (g(H[i], h0, k) * c * np.clip(1 - R[i] / rmax, 0.15, 1)))
        if len(usable(idx)) >= 6:
            cv["head+density"].append(loglin(fit_loglin(idx), i))
    print("\nscored on held-out blocks (reading / truth):")
    for name, v in cv.items():
        if len(v) == n:
            r = np.array(v) / T
            zero = int((~(r > 0)).sum())
            print(f"  {name:12s} median {np.median(r):.2f}  p10-p90 {np.percentile(r, 10):.2f}-{np.percentile(r, 90):.2f}  "
                  f"mean |log error| {np.mean(np.abs(np.log(np.clip(r, 0.05, 20)))):.2f}"
                  + (f"  ({zero} block(s) read 0, counted as x0.05)" if zero else ""))
    if len(usable(np.arange(n))) >= 4:
        c_all = fit_const(np.arange(n)); s_all = fit_sat(np.arange(n))
        ceiling = f"raw ceiling {s_all[1]:.2f} people/m^2" if s_all[1] < 50 else "no ceiling (the fit runs it to infinity)"
        print(f"\nall blocks: constant c = {c_all:.2f}; saturation c = {s_all[0]:.2f}, {ceiling}")
    if len(usable(np.arange(n))) >= 6:
        b = fit_loglin(np.arange(n))
        rng = np.random.default_rng(0)
        boot = np.array([fit_loglin(rng.choice(n, n)) for _ in range(300)])
        lo, hi = np.percentile(boot, 10, axis=0), np.percentile(boot, 90, axis=0)
        print(f"head+density: factor at 4 px / raw 0.5 = {np.exp(b[0]):.2f}; head exponent {b[1]:+.2f} (80% {lo[1]:+.2f}..{hi[1]:+.2f}); "
              f"density exponent {b[2]:+.2f} (80% {lo[2]:+.2f}..{hi[2]:+.2f}; below 0 = the roof under-reads more as it packs)")
    if args.json:
        args.json.write_text(json.dumps([{**b, "block": b["block"].isoformat()} for b in B], indent=1))


if __name__ == "__main__":
    main()
