"""Step 7: crowd density forecast 15 / 30 / 60 min ahead per camera (LightGBM), baselines and metrics.

Used by scripts/train_forecast.py (train + report) and backend/app.py (live forecast for the dashboard and the
alert engine). Data: 5-minute averages of accepted readings (edge.db.crowd_5min), YouTube source only (the 720p
fallback counts differ), one series per camera/zone. The series is the density in the visible part
of the camera's zone (people per 10k primary-reference px, since 2026-10-24; earlier rows are fully visible zones,
same unit), so zoomed / partly visible shots continue the same series; forecasts are densities only (people in the
zone are shown only for fully visible readings). Readings come in bursts (a camera is on air for a few minutes,
then gone), so the 5-min grid has gaps (NaN); the model also sees the last known value and how old it is.

Rows: every 5-min grid time t (the "issue time") where the camera had a reading within the last MAX_AGE_MIN.
Features (all known at t): FEATURES below. yday_k = the density 24 h before t+k: it lets the model follow a new
regime (Ramadan starting, a holiday) that the training days did not contain; without it, a model trained before
Ramadan lost badly to "same time yesterday" in the synthetic test. Target for horizon h: the density in the bucket starting at t+h (no
reading there = row not used for that horizon). Densities are divided by the camera's median ("scale", from the
training days only), so one model serves all cameras; predictions are multiplied back.
Baselines: persistence (last known value), same time yesterday and same time last week (the bucket 1 / 7 days
before the target, or the nearest within +-10 min).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import Engine, func, inspect, select

from edge import db, prayer_times

log = logging.getLogger(__name__)

STEP = 5  # minutes per bucket
HORIZONS = (15, 30, 60)
LAGS = (0, 15, 30, 60)
MAX_AGE_MIN = 30  # issue a forecast only if the camera was seen within this many minutes
CROWDED_PCT = 80  # "crowded" = top 20 % of the camera's 5-min averages (same rule as the dashboard level)
PRAYER_CODE = {"fajr": 0, "dhuhr": 1, "asr": 2, "maghrib": 3, "isha": 4}
FEATURES = ["lag_0", "lag_15", "lag_30", "lag_60", "last", "last_age", "mean_60", "n_60",
            "yday_0", "yday_15", "yday_30", "yday_60",
            "min_to_next", "min_since_prev", "next_prayer", "prev_prayer",
            "hour", "weekday", "is_friday", "is_ramadan", "is_hajj", "hijri_month", "camera"]
CATEGORICAL = ["next_prayer", "prev_prayer", "weekday", "camera"]
MAKKAH = timezone(timedelta(hours=3))


def _minutes(dt: datetime) -> int:
    return int(db.utc(dt).timestamp() // 60)


def _utc(minutes: int) -> datetime:
    return datetime.fromtimestamp(int(minutes) * 60, timezone.utc)


@dataclass
class Series:
    """One camera/zone on a regular 5-min grid (minutes since the Unix epoch, UTC)."""
    view_id: str
    zone: str
    t: np.ndarray  # int64 grid times
    v: np.ndarray  # density, NaN where the camera was off air
    ratio: float  # people in the whole zone per unit of density (from fully visible readings; reference only)


def load_series(engine: Engine, start: datetime | None = None, end: datetime | None = None) -> dict[str, Series]:
    """5-min density series per camera (YouTube readings), keyed 'view_id/zone', plus the view-independent Mataf
    series (people per m^2 per ring, all cameras together, keyed 'mataf/<ring>') once crowd_m2_minute has rows."""
    t = db.crowd_minute
    with engine.connect() as c:
        lo, hi = c.execute(select(func.min(t.c.minute_utc), func.max(t.c.minute_utc))).one()
        lo2, hi2 = (c.execute(select(func.min(db.crowd_m2_minute.c.minute_utc), func.max(db.crowd_m2_minute.c.minute_utc))).one()
                    if db.crowd_m2_minute.name in inspect(engine).get_table_names() else (None, None))
    los = [db.utc(x) for x in (lo, lo2) if x is not None]
    his = [db.utc(x) for x in (hi, hi2) if x is not None]
    if not los:
        return {}
    start = max(min(los), db.utc(start)) if start else min(los)
    end = min(max(his) + timedelta(minutes=1), db.utc(end)) if end else max(his) + timedelta(minutes=1)
    groups: dict[str, list] = {}
    buckets = db.crowd_5min(start, end, source="youtube", engine=engine)  # per-camera metric: no SBA curve
    if lo2 is not None:
        buckets += db.m2_5min(start, end, source=db.MAIN, engine=engine)
    for b in buckets:
        groups.setdefault(f"{b.view_id}/{b.zone}", []).append(b)
    out = {}
    for key, bs in groups.items():
        tm = np.array([_minutes(b.bucket_utc) for b in bs], np.int64)
        g0 = tm.min() - tm.min() % STEP
        grid = np.arange(g0, tm.max() + STEP, STEP, dtype=np.int64)
        v = np.full(len(grid), np.nan)
        v[(tm - g0) // STEP] = [b.density_avg if b.density_avg is not None else np.nan for b in bs]
        ratios = [b.count_full_avg / b.density_avg for b in bs if b.density_avg and b.count_full_avg is not None]
        view_id, zone = key.split("/", 1)
        out[key] = Series(view_id, zone, grid, v, float(np.median(ratios)) if ratios else 1.0)
    return out


@dataclass
class Calendar:
    """Prayer times and Hijri dates for feature lookups."""
    ev_t: np.ndarray  # prayer times (minutes), sorted
    ev_code: np.ndarray  # prayer code per event
    hijri: dict = field(default_factory=dict)  # Makkah date -> (month, day)

    @classmethod
    def load(cls, engine: Engine, start_min: int, end_min: int) -> "Calendar":
        events = prayer_times.load_events(_utc(start_min) - timedelta(days=1), _utc(end_min) + timedelta(days=2),
                                          engine=engine)
        pt = db.prayer_times
        with engine.connect() as c:
            rows = c.execute(select(pt.c.date, pt.c.hijri_date).where(pt.c.method == prayer_times.METHOD)).all()
        hijri = {}
        for d, h in rows:
            try:
                hijri[d] = (int(h[5:7]), int(h[8:10]))
            except (TypeError, ValueError):
                pass
        return cls(np.array([_minutes(t) for t, _ in events], np.int64),
                   np.array([PRAYER_CODE[n] for _, n in events], np.int64), hijri)


def time_features(t: np.ndarray, cal: Calendar) -> dict[str, np.ndarray]:
    """Calendar features for issue times t (minutes, UTC)."""
    n = len(t)
    f = {k: np.full(n, np.nan) for k in ("min_to_next", "min_since_prev", "next_prayer", "prev_prayer",
                                         "hijri_month", "is_ramadan", "is_hajj")}
    if len(cal.ev_t):
        j = np.searchsorted(cal.ev_t, t, side="right")
        has_prev, has_next = j > 0, j < len(cal.ev_t)
        f["min_since_prev"][has_prev] = t[has_prev] - cal.ev_t[j[has_prev] - 1]
        f["prev_prayer"][has_prev] = cal.ev_code[j[has_prev] - 1]
        f["min_to_next"][has_next] = cal.ev_t[j[has_next]] - t[has_next]
        f["next_prayer"][has_next] = cal.ev_code[j[has_next]]
    local = t + 180  # Makkah = UTC+3
    days = local // 1440
    f["hour"] = (local % 1440) / 60
    f["weekday"] = ((days + 3) % 7).astype(float)  # 0 = Monday (1970-01-01 was a Thursday)
    f["is_friday"] = (f["weekday"] == 4).astype(float)
    epoch = datetime(1970, 1, 1).date()
    for d in np.unique(days):
        hm = cal.hijri.get(epoch + timedelta(days=int(d)))
        if hm:
            m = days == d
            f["hijri_month"][m] = hm[0]
            f["is_ramadan"][m] = float(hm[0] == 9)
            f["is_hajj"][m] = float(hm[0] == 12 and hm[1] <= 13)
    return f


def _nearest(v: np.ndarray, idx: np.ndarray, reach: int = 2) -> np.ndarray:
    """v[idx], or the nearest non-NaN value within +-reach buckets (NaN outside the grid)."""
    out = np.full(len(idx), np.nan)
    for off in [0] + [s * k for k in range(1, reach + 1) for s in (-1, 1)]:
        j = idx + off
        ok = (j >= 0) & (j < len(v)) & np.isnan(out)
        out[ok] = v[j[ok]]
    return out


@dataclass
class Rows:
    """Feature rows (all cameras stacked) with targets and baselines per horizon."""
    X: np.ndarray
    t: np.ndarray  # issue time (minutes)
    cam: np.ndarray  # camera key per row
    scale: np.ndarray  # camera scale per row
    y: dict  # h -> density at t+h (raw units)
    base: dict  # h -> {"persistence", "yesterday", "last_week"} (raw units)


def build_rows(series: dict[str, Series], cal: Calendar, scales: dict[str, float], cams: list[str],
               only_last: bool = False) -> Rows:
    """Feature rows for every issue time with a recent reading (or only the latest grid time: live forecast)."""
    blocks = []
    for key, s in series.items():
        if key not in scales:
            continue
        v, L = s.v, len(s.v)
        seen = np.where(np.isfinite(v), np.arange(L), -1)
        last_i = np.maximum.accumulate(seen)
        idx = np.arange(L)
        age = np.where(last_i >= 0, (idx - last_i) * STEP, np.inf)
        rows = np.nonzero(age <= MAX_AGE_MIN)[0]
        if only_last:
            rows = rows[rows == L - 1]
        if not len(rows):
            continue
        sc = scales[key]
        lag = {k: np.where(rows - k // STEP >= 0, v[np.clip(rows - k // STEP, 0, L - 1)], np.nan) for k in LAGS}
        win = np.lib.stride_tricks.sliding_window_view(np.concatenate([np.full(11, np.nan), v]), 12)[rows]
        n60 = np.isfinite(win).sum(1)
        mean60 = np.where(n60 > 0, np.nansum(np.nan_to_num(win, nan=0.0), 1) / np.maximum(n60, 1), np.nan)
        last = v[last_i[rows]]
        yday = {k: _nearest(v, rows + k // STEP - 1440 // STEP) for k in LAGS}
        tf = time_features(s.t[rows], cal)
        cols = {"lag_0": lag[0] / sc, "lag_15": lag[15] / sc, "lag_30": lag[30] / sc, "lag_60": lag[60] / sc,
                "last": last / sc, "last_age": age[rows], "mean_60": mean60 / sc, "n_60": n60.astype(float),
                **{f"yday_{k}": yday[k] / sc for k in LAGS},
                **tf, "camera": np.full(len(rows), float(cams.index(key)) if key in cams else np.nan)}
        X = np.column_stack([cols[f] for f in FEATURES])
        y, base = {}, {}
        for h in HORIZONS:
            ti = rows + h // STEP
            y[h] = np.where(ti < L, v[np.clip(ti, 0, L - 1)], np.nan)
            base[h] = {"persistence": last, "yesterday": _nearest(v, ti - 1440 // STEP),
                       "last_week": _nearest(v, ti - 10080 // STEP)}
        blocks.append((X, s.t[rows], np.full(len(rows), key, object), np.full(len(rows), sc), y, base))
    if not blocks:
        return Rows(np.zeros((0, len(FEATURES))), np.zeros(0, np.int64), np.zeros(0, object), np.zeros(0),
                    {h: np.zeros(0) for h in HORIZONS},
                    {h: {k: np.zeros(0) for k in ("persistence", "yesterday", "last_week")} for h in HORIZONS})
    cat = lambda i: np.concatenate([b[i] for b in blocks])  # noqa: E731
    return Rows(np.vstack([b[0] for b in blocks]), cat(1), cat(2), cat(3),
                {h: np.concatenate([b[4][h] for b in blocks]) for h in HORIZONS},
                {h: {k: np.concatenate([b[5][h][k] for b in blocks]) for k in ("persistence", "yesterday", "last_week")}
                 for h in HORIZONS})


def _mae(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    m = np.isfinite(a) & np.isfinite(b)
    return (float(np.mean(np.abs(a[m] - b[m]))) if m.any() else float("nan")), int(m.sum())


@dataclass
class Result:
    models: dict  # h -> lightgbm Booster
    meta: dict  # cameras, scales, thresholds, ratios, ranges, feature list
    metrics: dict  # h -> {...}
    early: dict  # early-warning summary
    test: dict  # h -> {"t", "cam", "y", "pred", "persistence", "yesterday"} (raw units) for plots


def train(series: dict[str, Series], cal: Calendar, test_days: float = 2.0, seed: int = 0) -> Result:
    """Time-based split (last `test_days` days = test, last train day = early stopping), no shuffling."""
    import lightgbm as lgb

    end = max(int(s.t[-1]) for s in series.values()) + STEP
    cut = end - int(test_days * 1440)
    val_cut = cut - 1440
    scales, thresholds, cams = {}, {}, sorted(series)
    for key, s in series.items():
        train_v = s.v[(s.t < cut) & np.isfinite(s.v)]
        if len(train_v) >= 12:
            scales[key] = float(np.median(train_v))
            thresholds[key] = float(np.percentile(train_v, CROWDED_PCT))
    rows = build_rows(series, cal, scales, cams)
    tr, va, te = rows.t < val_cut, (rows.t >= val_cut) & (rows.t < cut), rows.t >= cut
    if tr.sum() == 0 or te.sum() == 0:
        raise ValueError("not enough data for a time split (need earlier days to train and later days to test)")
    models, metrics, test = {}, {}, {}
    cat_idx = [FEATURES.index(c) for c in CATEGORICAL]
    last_raw = rows.X[:, FEATURES.index("last")] * rows.scale
    for h in HORIZONS:
        y = rows.y[h] / last_raw  # target = ratio to the last known value (scale-free: survives regime shifts)
        ok = np.isfinite(y) & (last_raw > 0)
        params = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=20,
                      feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=seed,
                      metric="l1", verbose=-1)
        dtr = lgb.Dataset(rows.X[tr & ok], y[tr & ok], feature_name=FEATURES, categorical_feature=cat_idx,
                          free_raw_data=False)
        valid = [lgb.Dataset(rows.X[va & ok], y[va & ok], reference=dtr)] if (va & ok).sum() >= 20 else []
        booster = lgb.train(params, dtr, num_boost_round=2000, valid_sets=valid,
                            callbacks=[lgb.early_stopping(100, verbose=False)] if valid else [])
        models[h] = booster
        m = te & ok
        pred = booster.predict(rows.X[m], num_iteration=booster.best_iteration or None) * last_raw[m]
        actual = rows.y[h][m]
        mm, n = _mae(pred, actual)
        entry = {"n_test": n, "n_train": int((tr & ok).sum()), "mae": mm, "baselines": {}, "per_camera": {}}
        for name in ("persistence", "yesterday", "last_week"):
            b = rows.base[h][name][m]
            have = np.isfinite(b)
            mb, nb = _mae(b, actual)
            mm_sub, _ = _mae(pred[have], actual[have])
            entry["baselines"][name] = {"n": nb, "coverage": nb / max(n, 1), "mae_baseline": mb, "mae_model": mm_sub,
                                        "improvement_pct": 100 * (1 - mm_sub / mb) if nb and mb > 0 else None}
        for key in cams:
            c = rows.cam[m] == key
            if c.any():
                pm, pn = _mae(pred[c], actual[c])
                bm, _ = _mae(rows.base[h]["persistence"][m][c], actual[c])
                entry["per_camera"][key] = {"n": pn, "mae": pm, "mae_persistence": bm}
        metrics[h] = entry
        test[h] = {"t": rows.t[m] + h, "cam": rows.cam[m], "y": actual, "pred": pred,
                   "persistence": rows.base[h]["persistence"][m], "yesterday": rows.base[h]["yesterday"][m]}
    meta = {"features": FEATURES, "categorical": CATEGORICAL, "horizons": list(HORIZONS), "cameras": cams,
            "target": "ratio_to_last",
            "scales": scales, "thresholds": thresholds, "ratios": {k: series[k].ratio for k in cams},
            "train_range_utc": [_utc(min(int(s.t[0]) for s in series.values())).isoformat(), _utc(cut).isoformat()],
            "test_range_utc": [_utc(cut).isoformat(), _utc(end).isoformat()],
            "best_iteration": {h: models[h].best_iteration for h in HORIZONS},
            "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    return Result(models, meta, metrics, early_warning(series, test, thresholds, cut), test)


def early_warning(series: dict[str, Series], test: dict, thresholds: dict, cut: int) -> dict:
    """When the observed density crosses into "crowded", how many minutes earlier did the forecast say so?

    A crossing = a test bucket at/above the camera's threshold whose previous reading (within 30 min) was below.
    Lead = the longest horizon whose forecast for that bucket was at/above the threshold (0 = not warned).
    Same for "same time yesterday". Also precision / recall of "crowded" per horizon on all test rows.
    """
    events = []
    for key, s in series.items():
        thr = thresholds.get(key)
        if thr is None:
            continue
        obs = np.nonzero(np.isfinite(s.v) & (s.t >= cut))[0]
        for j in obs:
            prev = [i for i in range(max(0, j - 6), j) if np.isfinite(s.v[i])]
            if s.v[j] >= thr and prev and s.v[prev[-1]] < thr:
                events.append((key, int(s.t[j]), thr))
    leads, leads_y, unavailable = [], [], 0
    for key, tj, thr in events:
        lead, lead_y, any_fc = 0, 0, False
        for h in sorted(HORIZONS, reverse=True):
            hit = (test[h]["cam"] == key) & (test[h]["t"] == tj)
            if hit.any():
                any_fc = True
                if lead == 0 and test[h]["pred"][hit][0] >= thr:
                    lead = h
                if lead_y == 0 and np.isfinite(test[h]["yesterday"][hit][0]) and test[h]["yesterday"][hit][0] >= thr:
                    lead_y = h
        unavailable += not any_fc
        if any_fc:
            leads.append(lead)
            leads_y.append(lead_y)
    pr = {}
    for h in HORIZONS:
        thr = np.array([thresholds.get(k, np.inf) for k in test[h]["cam"]])
        p, a = test[h]["pred"] >= thr, test[h]["y"] >= thr
        pr[h] = {"precision": float((p & a).sum() / p.sum()) if p.sum() else None,
                 "recall": float((p & a).sum() / a.sum()) if a.sum() else None, "n_crowded": int(a.sum())}
    lv = np.array(leads)
    return {"events": len(events), "with_forecast": len(leads), "no_forecast_issued": unavailable,
            "warned_15plus": int((lv >= 15).sum()) if len(lv) else 0,
            "warned_30plus": int((lv >= 30).sum()) if len(lv) else 0,
            "median_lead_min": float(np.median(lv)) if len(lv) else None,
            "mean_lead_min": float(lv.mean()) if len(lv) else None,
            "yesterday_median_lead_min": float(np.median(leads_y)) if leads_y else None,
            "precision_recall": pr}


def save(result: Result, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    for h, booster in result.models.items():
        booster.save_model(str(model_dir / f"lgb_h{h}.txt"), num_iteration=booster.best_iteration or None)
    (model_dir / "meta.json").write_text(json.dumps(result.meta, indent=2))


_cache: dict = {"key": None, "models": None, "meta": None}


def load(model_dir: Path) -> tuple[dict, dict] | None:
    """(models, meta) from model_dir, cached by file times; None if no model was trained yet."""
    meta_p = model_dir / "meta.json"
    if not meta_p.exists():
        return None
    import lightgbm as lgb

    files = [meta_p] + [model_dir / f"lgb_h{h}.txt" for h in HORIZONS]
    key = tuple(f.stat().st_mtime_ns if f.exists() else 0 for f in files)
    if _cache["key"] != key:
        meta = json.loads(meta_p.read_text())
        _cache.update(key=key, meta=meta, models={int(h): lgb.Booster(model_file=str(model_dir / f"lgb_h{h}.txt"))
                                                  for h in meta["horizons"]})
    return _cache["models"], _cache["meta"]


def predict_now(engine: Engine, model_dir: Path, now: datetime | None = None) -> dict:
    """Live forecast per camera for the latest 5-min bucket (cameras seen within MAX_AGE_MIN only)."""
    loaded = load(model_dir)
    if loaded is None:
        return {"available": False, "reason": "no model trained yet (scripts/train_forecast.py)"}
    models, meta = loaded
    now = db.utc(now or datetime.now(timezone.utc))
    series = load_series(engine, now - timedelta(hours=26), now + timedelta(minutes=1))  # yday_* need yesterday
    end_min = _minutes(now)
    for s in series.values():  # extend each grid to "now" so the age of the last reading is right
        last_grid = end_min - end_min % STEP
        if s.t[-1] < last_grid:
            extra = np.arange(s.t[-1] + STEP, last_grid + STEP, STEP, dtype=np.int64)
            s.t, s.v = np.concatenate([s.t, extra]), np.concatenate([s.v, np.full(len(extra), np.nan)])
    cal = Calendar.load(engine, end_min - 240, end_min + 120)
    rows = build_rows(series, cal, meta["scales"], meta["cameras"], only_last=True)
    out = []
    for i, key in enumerate(rows.cam):
        thr = meta["thresholds"].get(key)
        preds = []
        for h in meta["horizons"]:
            last_raw = rows.X[i, FEATURES.index("last")] * rows.scale[i]
            d = float(models[int(h)].predict(rows.X[i:i + 1])[0] * last_raw)  # model predicts the ratio to the last value
            preds.append({"h": int(h), "utc": _utc(int(rows.t[i]) + int(h)).isoformat(), "density": round(d, 2)})
        view_id, zone = key.split("/", 1)
        out.append({"view_id": view_id, "zone": zone, "issued_utc": _utc(int(rows.t[i])).isoformat(),
                    "threshold_density": thr,
                    "predictions": preds,
                    "crowded_within_30": bool(thr is not None and any(p["density"] >= thr for p in preds if p["h"] <= 30))})
    return {"available": True, "trained_utc": meta.get("trained_utc"), "train_range_utc": meta.get("train_range_utc"),
            "cameras": out}
