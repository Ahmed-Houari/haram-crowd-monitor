"""Nightly crowd forecast (user 2026-11-01): every night at 03:00 Makkah (00:00 UTC) the forecast is retrained on all
readings so far, and its error on the day that just ended is logged against the "same time yesterday" baseline.
NOT connected to the dashboard: it writes only data/forecast/ (the API's live forecast reads models/forecast, which this
never touches).

Each run, for the slot S (00:00 UTC of a day = 03:00 Makkah):
  1. Readings: crowd_m2_minute, ring 'all', 5-minute bins weighted by frames, everything before S.
  2. The day's error: models trained on the samples whose target is before S - 24 h, tested on the targets in
     [S - 24 h, S) (03:00 to 03:00 Makkah): MAE of the model, of "same time yesterday" (the reading 24 h before the
     target; the reading now where that is missing, as in forecast v0) and of persistence (the reading now), per
     horizon 15 / 30 / 60 min. One row per horizon in data/forecast/daily_errors.csv, and a line in the log.
  3. The models: retrained on all samples before S -> data/forecast/lgbm_resid_<h>min.txt, with data/forecast/run.json
     (when, how much data, the day's error, the top inputs).
  4. The running report, per horizon, in data/forecast/summary.json and the log: the error over the days after 31 Oct
     separately from the days up to 31 Oct (user 2026-11-01: the readings changed with v4 on 30 Oct and the speed
     sensor on 31 Oct), each the MAE over all their test bins.
Newer days weigh more (user 2026-11-01, same reason): every training sample has the weight 0.5 ** (age / HALF_LIFE_DAYS),
age = from its target time to the end of the training data (the test day's start for the day's error, S for the
models): a day 2 days older counts half.
Method = forecast v0 (data/jobs/overnight_20261101/forecast_v0.py --residual): one LightGBM model per horizon (MAE
objective) on the change from now; inputs: time of day, weekday (Makkah), the prayer schedule (minutes since the last
adhan, to the next, which prayer, prayer window), the last readings (now, -15, -30, -60 min, last hour's mean, slope)
and the readings 1 and 2 days before the target.
A night the PC was off runs when the job starts again (the latest slot only); each slot runs once. Normal priority, no
limits (user rule); a run takes seconds on the CPU.

  .venv/Scripts/python scripts/forecast_nightly.py --loop       # supervisor child "forecast": waits for 03:00 Makkah
  .venv/Scripts/python scripts/forecast_nightly.py --once       # the latest slot now (again if it already ran)
  .venv/Scripts/python scripts/forecast_nightly.py --backfill   # the day's error for every past day, then --once
  [--half-life DAYS]  (0 = no weighting; for comparisons, with --out DIR to keep them apart)
Log: data/logs/forecast.log.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge.runtime import ALREADY_RUNNING_EXIT, setup_logging, single_instance  # noqa: E402

DB = ROOT / "data" / "haram.db"
OUT = ROOT / "data" / "forecast"
RUN_HOUR_UTC = 0  # 03:00 Makkah
BIN = 5  # minutes
MK = timedelta(hours=3)
HORIZONS = {15: 3, 30: 6, 60: 12}  # minutes ahead -> bins
DAY = 24 * 60 // BIN
MIN_TRAIN, MIN_TEST = 200, 20
HALF_LIFE_DAYS = 2.0  # newer days weigh more: a day this much older counts half
NEW_READINGS_FROM = "2026-11-01"  # Makkah test days from here on are "after 31 Oct" (v4 since 30 Oct, speed sensor 31 Oct)
PERIODS = {"after_31_oct": "days after 31 Oct", "to_31_oct": "days up to 31 Oct"}
PARAMS = dict(objective="l1", n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20, subsample=0.9,
              subsample_freq=1, colsample_bytree=0.9, verbose=-1)
NAMES = ["sin_h", "cos_h", "dow", "since_prev", "to_next", "next_prayer", "in_window", "since_prev_now", "y_now", "y_15",
         "y_30", "y_60", "mean_1h", "slope_30", "y_tgt_1d", "y_tgt_2d", "y_now_1d", "horizon_min"]
FIELDS = ["slot_utc", "day_makkah", "period", "horizon_min", "test_bins", "train_bins", "half_life_days", "mae_model",
          "mae_yesterday", "mae_persist", "model_vs_yesterday_pct", "yesterday_available", "mean_level", "note"]
log = logging.getLogger("forecast")


def parse(s: str) -> datetime:
    return datetime.strptime(s[:16], "%Y-%m-%d %H:%M")


class Data:
    """The Mataf's 5-minute readings before `until` (naive UTC; from `since` when given) and the prayer schedule."""

    def __init__(self, until: datetime, since: datetime | None = None):
        c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        try:
            rows = c.execute("select minute_utc, people_m2, n_frames from crowd_m2_minute where ring='all' and "
                             "people_m2 is not null and n_frames > 0 and minute_utc < ? and minute_utc >= ? "
                             "order by minute_utc", (until.strftime("%Y-%m-%d %H:%M"),
                                                     (since or datetime(2000, 1, 1)).strftime("%Y-%m-%d %H:%M"))).fetchall()
            pts = c.execute("select fajr, dhuhr, asr, maghrib, isha from prayer_times").fetchall()
        finally:
            c.close()
        self.minutes, self.bins, self.first, self.last = len(rows), 0, None, None
        if not rows:
            self.n = 0
            return
        self.first, self.last = rows[0][0][:16], rows[-1][0][:16]
        t0 = parse(rows[0][0]).replace(minute=0)
        self.n = int((until - t0).total_seconds() // 60 // BIN)
        num, den = np.zeros(self.n), np.zeros(self.n)
        for m, v, k in rows:
            i = int((parse(m) - t0).total_seconds() // 60 // BIN)
            if 0 <= i < self.n:
                num[i] += v * k
                den[i] += k
        self.y = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
        self.bins = int((den > 0).sum())
        self.times = [t0 + timedelta(minutes=BIN * i) for i in range(self.n)]
        self.events = sorted((parse(t), k) for r in pts for k, t in enumerate(r) if t)
        self.ev_t = [e[0] for e in self.events]

    def at(self, i: int) -> float:
        return self.y[i] if 0 <= i < self.n else np.nan

    def prayer(self, t: datetime) -> tuple[float, float, int, float]:
        j = bisect.bisect_right(self.ev_t, t)
        prev = self.events[j - 1] if j else None
        nxt = self.events[j] if j < len(self.events) else None
        since = (t - prev[0]).total_seconds() / 60 if prev else np.nan
        to = (nxt[0] - t).total_seconds() / 60 if nxt else np.nan
        return since, to, (nxt[1] if nxt else -1), float(since <= 40)

    def feats(self, i: int, h: int) -> list[float]:
        j = i + h
        tm = self.times[j] + MK
        hod = tm.hour + tm.minute / 60
        since, to, nxt, win = self.prayer(self.times[j])
        hist = self.y[max(i - 12, 0):i + 1]
        return [np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24), tm.weekday(), since, to, nxt, win,
                self.prayer(self.times[i])[0], self.at(i), self.at(i - 3), self.at(i - 6), self.at(i - 12),
                np.nanmean(hist) if np.isfinite(hist).any() else np.nan, self.at(i) - self.at(i - 6),
                self.at(j - DAY), self.at(j - 2 * DAY), self.at(i - DAY), h * BIN]

    def samples(self, h: int):
        """Inputs, target, target time, same-time-yesterday baseline, reading now: every pair with both readings."""
        X, Y, T, B, P = [], [], [], [], []
        for i in range(self.n - h):
            j = i + h
            if np.isfinite(self.y[i]) and np.isfinite(self.y[j]):
                X.append(self.feats(i, h))
                Y.append(self.y[j])
                T.append(self.times[j])
                B.append(self.at(j - DAY) if np.isfinite(self.at(j - DAY)) else self.y[i])
                P.append(self.y[i])
        return (np.array(X, float).reshape(-1, len(NAMES)), np.array(Y, float), np.array(T, dtype="datetime64[m]"),
                np.array(B, float), np.array(P, float))


def weights(T: np.ndarray, end: datetime, half_life: float) -> np.ndarray:
    """Newer samples weigh more: 0.5 ** (age in days / half_life); all 1 when half_life is 0."""
    if half_life <= 0:
        return np.ones(len(T))
    age = (np.datetime64(end, "m") - T).astype("timedelta64[m]").astype(float) / 1440.0
    return 0.5 ** (np.maximum(age, 0) / half_life)


def fit(X, Y, P, w):
    import lightgbm as lgb

    return lgb.LGBMRegressor(**PARAMS).fit(X, Y - P, sample_weight=w)


def period(day_makkah: str) -> str:
    return "after_31_oct" if day_makkah >= NEW_READINGS_FROM else "to_31_oct"


def day_error(d: Data, slot: datetime, half_life: float, cache: dict) -> list[dict]:
    """The error on the 24 h before the slot (03:00 to 03:00 Makkah) of models trained on the days before it."""
    day0 = slot - timedelta(days=1)
    day = f"{(day0 + MK):%Y-%m-%d}"
    label = f"{(day0 + MK):%Y-%m-%d %H:%M} to {(slot + MK):%Y-%m-%d %H:%M} Makkah"
    rows = []
    for hmin, h in HORIZONS.items():
        X, Y, T, B, P = cache.setdefault(h, d.samples(h))
        tr = T < np.datetime64(day0, "m")
        te = ~tr & (T < np.datetime64(slot, "m"))
        row = {"slot_utc": f"{slot:%Y-%m-%d %H:%M}", "day_makkah": day, "period": period(day), "horizon_min": hmin,
               "test_bins": int(te.sum()), "train_bins": int(tr.sum()), "half_life_days": half_life, "note": ""}
        if tr.sum() >= MIN_TRAIN and te.sum() >= MIN_TEST:
            p = fit(X[tr], Y[tr], P[tr], weights(T[tr], day0, half_life)).predict(X[te]) + P[te]
            mae, mb, mp = (float(np.abs(v - Y[te]).mean()) for v in (p, B[te], P[te]))
            avail = float(np.isfinite(X[te][:, NAMES.index("y_tgt_1d")]).mean())
            row.update(mae_model=round(mae, 3), mae_yesterday=round(mb, 3), mae_persist=round(mp, 3),
                       model_vs_yesterday_pct=round(100 * (mae - mb) / max(mb, 1e-9), 1), yesterday_available=round(avail, 2),
                       mean_level=round(float(Y[te].mean()), 2))
            log.info("%s, %2d min ahead: MAE model %.3f, same time yesterday %.3f (%+.0f %%; yesterday's reading for %.0f %% "
                     "of the bins), persistence %.3f | %d test bins, trained on %d", label, hmin, mae, mb,
                     row["model_vs_yesterday_pct"], 100 * avail, mp, te.sum(), tr.sum())
        else:
            row["note"] = "not enough readings"
            log.warning("%s, %d min ahead: not enough readings for the day's error (%d test bins, %d to train on)",
                        label, hmin, te.sum(), tr.sum())
        rows.append(row)
    return rows


def save_rows(csv_path: Path, rows: list[dict]) -> list[dict]:
    """Add the rows to the CSV, replacing earlier rows of the same slot and horizon; returns every row."""
    old = []
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            old = list(csv.DictReader(f))
    keys = {(r["slot_utc"], str(r["horizon_min"])) for r in rows}
    allr = [r for r in old if (r.get("slot_utc"), str(r.get("horizon_min"))) not in keys] + rows
    allr.sort(key=lambda r: (r["slot_utc"], int(r["horizon_min"])))
    tmp = csv_path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(allr)
    tmp.replace(csv_path)
    return allr


def report(out: Path, allr: list[dict]) -> dict:
    """Per horizon and period (after 31 Oct / up to 31 Oct / all): MAE over all their test bins, model vs yesterday."""
    rep: dict = {}
    for hmin in HORIZONS:
        rep[f"{hmin} min"] = {}
        for key, name in (*PERIODS.items(), ("all", "all days")):
            rs = [r for r in allr if int(r["horizon_min"]) == hmin and r.get("mae_model") not in (None, "")
                  and (key == "all" or r["period"] == key)]
            n = sum(int(r["test_bins"]) for r in rs)
            if not n:
                rep[f"{hmin} min"][key] = {"days": 0}
                continue
            m = sum(float(r["mae_model"]) * int(r["test_bins"]) for r in rs) / n
            b = sum(float(r["mae_yesterday"]) * int(r["test_bins"]) for r in rs) / n
            p = sum(float(r["mae_persist"]) * int(r["test_bins"]) for r in rs) / n
            rep[f"{hmin} min"][key] = {"days": len(rs), "first": rs[0]["day_makkah"], "last": rs[-1]["day_makkah"],
                                       "test_bins": n, "mae_model": round(m, 3), "mae_yesterday": round(b, 3),
                                       "mae_persist": round(p, 3), "model_vs_yesterday_pct": round(100 * (m - b) / max(b, 1e-9), 1)}
    tmp = (out / "summary.json").with_suffix(".tmp")
    tmp.write_text(json.dumps({"by_horizon": rep, "periods": PERIODS, "new_readings_from": NEW_READINGS_FROM}, indent=1),
                   encoding="utf-8")
    tmp.replace(out / "summary.json")
    for key, name in (*PERIODS.items(), ("all", "all days")):
        parts = []
        for hmin in HORIZONS:
            r = rep[f"{hmin} min"][key]
            parts.append(f"{hmin} min {r['mae_model']:.3f} vs {r['mae_yesterday']:.3f} ({r['model_vs_yesterday_pct']:+.0f} %)"
                         if r["days"] else f"{hmin} min -")
        n_days = max(rep[f"{h} min"][key]["days"] for h in HORIZONS)
        log.info("report, %s (%d): MAE model vs same time yesterday: %s", name, n_days, "; ".join(parts))
    return rep


def run(slot: datetime, half_life: float = HALF_LIFE_DAYS, out: Path = OUT, backfill: bool = False) -> dict:
    """One night's run for the slot (naive UTC): the day's error (every past day too with backfill), the models on
    everything, the report."""
    t_run = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    d = Data(slot)
    cache: dict = {}
    rows = []
    if backfill and d.n:
        first = datetime.strptime(d.first, "%Y-%m-%d %H:%M").replace(hour=RUN_HOUR_UTC, minute=0) + timedelta(days=2)
        s = first
        while s < slot:
            rows += day_error(d, s, half_life, cache)
            s += timedelta(days=1)
    rows += day_error(d, slot, half_life, cache)
    res = {"slot_utc": f"{slot:%Y-%m-%d %H:%M}", "ran_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
           "half_life_days": half_life, "readings": {"minutes": d.minutes, "bins": d.bins, "from_utc": d.first, "to_utc": d.last},
           "day_error": [r for r in rows if r["slot_utc"] == f"{slot:%Y-%m-%d %H:%M}"], "models": {}}
    for hmin, h in HORIZONS.items():  # the models on all readings so far, newer days weighing more
        X, Y, T, B, P = cache.setdefault(h, d.samples(h)) if d.n else (np.zeros((0, len(NAMES))),) + (np.array([]),) * 4
        if len(Y) < MIN_TRAIN:
            res["models"][f"{hmin} min"] = None
            log.warning("%d min ahead: %d samples, not enough to train (kept the previous model)", hmin, len(Y))
            continue
        m = fit(X, Y, P, weights(T, slot, half_life))
        path = out / f"lgbm_resid_{hmin}min.txt"
        tmp = path.with_suffix(".tmp")
        m.booster_.save_model(str(tmp))
        tmp.replace(path)
        res["models"][f"{hmin} min"] = {"file": path.name, "samples": int(len(Y)),
                                        "top_inputs": [f"{k} {v}" for k, v in sorted(zip(NAMES, m.feature_importances_.tolist()),
                                                                                      key=lambda kv: -kv[1])[:6]]}
    res["report"] = report(out, save_rows(out / "daily_errors.csv", rows))
    res["seconds"] = round(time.monotonic() - t_run, 1)
    res["features"] = NAMES
    tmp = (out / "run.json").with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    tmp.replace(out / "run.json")
    log.info("forecast retrained for slot %s UTC on %d readings (%d five-minute bins, %s .. %s UTC), half-life %.1f days, "
             "in %.1f s -> %s", res["slot_utc"], d.minutes, d.bins, d.first, d.last, half_life, res["seconds"], out)
    return res


LIVE_HORIZONS = (30, 60)  # the forecast line on the dashboard and /simple (user 2026-11-07): in 30 min, in 1 hour
LIVE_FRESH_MIN = 15  # the latest 5-minute reading must have ended this recently (the dashboard's "live")
_boosters: dict = {}  # minutes ahead -> (model file mtime, Booster)


def predict_live(now: datetime | None = None, out: Path = OUT) -> dict:
    """The forecast line (user 2026-11-07, after the forecast beat "same time yesterday" on every day since 1 Nov):
    the nightly models (out/lgbm_resid_<h>min.txt, retrained at 03:00 Makkah) on the latest complete 5-minute reading,
    30 and 60 min ahead, with the same inputs as in training. Read-only; {"available": false, "reason"} when there is
    no fresh reading or no model. Times naive UTC in, ISO UTC out."""
    import lightgbm as lgb

    now = now or utcnow()
    cur = now.replace(second=0, microsecond=0) - timedelta(minutes=now.minute % BIN)  # start of the bin in progress
    d = Data(cur, since=cur - timedelta(days=3))  # the inputs reach 2 days + 1 hour back
    if not d.n:
        return {"available": False, "reason": "no readings in the last 3 days"}
    i = next((k for k in range(d.n - 1, -1, -1) if np.isfinite(d.y[k])), None)
    if i is None or (now - d.times[i]).total_seconds() / 60 > LIVE_FRESH_MIN + BIN:
        return {"available": False, "reason": "no live reading"}
    d.times += [d.times[-1] + timedelta(minutes=BIN * k) for k in range(1, max(HORIZONS.values()) + 1)]
    try:
        slot = json.loads((out / "run.json").read_text(encoding="utf-8")).get("slot_utc")
    except (OSError, ValueError):
        slot = None
    items = []
    for hmin in LIVE_HORIZONS:
        h, path = HORIZONS[hmin], out / f"lgbm_resid_{hmin}min.txt"
        try:
            mt = path.stat().st_mtime
        except OSError:
            return {"available": False, "reason": f"no model {path.name}"}
        if _boosters.get(hmin, (None,))[0] != mt:
            _boosters[hmin] = (mt, lgb.Booster(model_file=str(path)))
        v = float(_boosters[hmin][1].predict(np.array([d.feats(i, h)], float))[0] + d.y[i])
        items.append({"minutes": hmin, "target_utc": f"{d.times[i + h]:%Y-%m-%dT%H:%M}:00+00:00",
                      "people_m2": round(max(v, 0.0), 2)})
    return {"available": True, "from_utc": f"{d.times[i]:%Y-%m-%dT%H:%M}:00+00:00", "reading_m2": round(float(d.y[i]), 2),
            "items": items, "model_slot_utc": slot, "kind": "forecast"}


def latest_slot(now: datetime) -> datetime:
    s = now.replace(hour=RUN_HOUR_UTC, minute=0, second=0, microsecond=0)
    return s if now >= s else s - timedelta(days=1)


def done(slot: datetime) -> bool:
    try:
        return json.loads((OUT / "run.json").read_text(encoding="utf-8")).get("slot_utc") == f"{slot:%Y-%m-%d %H:%M}"
    except (OSError, ValueError):
        return False


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--loop", action="store_true", help="wait for 03:00 Makkah every night (supervisor child)")
    g.add_argument("--once", action="store_true", help="run the latest slot now")
    g.add_argument("--backfill", action="store_true", help="the day's error for every past day, then the latest slot")
    ap.add_argument("--half-life", type=float, default=HALF_LIFE_DAYS, help="days; 0 = no weighting (comparisons only)")
    ap.add_argument("--out", type=Path, default=OUT, help="output folder (comparisons only; the job uses data/forecast)")
    args = ap.parse_args()
    setup_logging("forecast")
    if args.once or args.backfill:
        run(latest_slot(utcnow()), args.half_life, args.out, backfill=args.backfill)
        return
    if not single_instance("forecast"):
        sys.exit(ALREADY_RUNNING_EXIT)
    log.info("nightly forecast job started: retrains at 03:00 Makkah on all readings (half-life %.1f days), writes %s "
             "(not the dashboard)", HALF_LIFE_DAYS, OUT)
    while True:
        slot = latest_slot(utcnow())
        if not done(slot):
            try:
                run(slot)
            except Exception:  # noqa: BLE001 - one bad night must not stop the job; the next check tries again
                log.exception("forecast run for slot %s UTC failed", f"{slot:%Y-%m-%d %H:%M}")
                time.sleep(600)
                continue
        nxt = slot + timedelta(days=1)
        time.sleep(max(5.0, min(300.0, (nxt - utcnow()).total_seconds())))  # short naps: robust to sleep / clock changes


if __name__ == "__main__":
    main()
