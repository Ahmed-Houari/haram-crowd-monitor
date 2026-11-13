"""Step 7: train and evaluate the 15 / 30 / 60-min crowd forecast (edge/forecast.py), print a clear report.

  --db PATH          SQLite database to read (opened READ-ONLY). Test on a synthetic DB first:
                     scripts/make_synthetic_db.py. The real one is data/haram.db.
  --check            only check whether the data is enough (days, readings, Fridays), print GO / NO-GO.
  --learning-curve   train on 3, 5, 7, 10, 14 ... days with the same test days: how much data is needed?
  --test-days N      days at the end used as the test set (default 2).
  --out DIR          model files. Default: models/forecast for the real data/haram.db (the API serves this one),
                     models/forecast_test for any other DB, so a synthetic model never reaches the dashboard.
                     The report + plot go to data/reports/forecast/<time>/.

Report: MAE per horizon for the model and three baselines (persistence = value now, same time yesterday, same
time last week), % improvement over each (on the rows where that baseline exists), per camera MAE, and the
early-warning check (how many minutes before the level crossed into "crowded" the forecast said so).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge import forecast as fc  # noqa: E402

MAKKAH = timezone(timedelta(hours=3))
# From the learning curve on synthetic data (2026-10-24, ratio-to-last target): with 3 train days the
# model still loses to "same time yesterday" at 30 min, from 5 train days it beats all baselines, around 10 it
# levels off. Real data is noisier and sparser (the roof camera is rarely on air), so these are minimums.
MIN_DAYS = 5  # 3 train + 2 test: a rough dry run only
GOOD_DAYS = 7  # 5 train + 2 test: first meaningful result
SOLID_DAYS = 12  # 10 train + 2 test: levelled off; 'same time last week' baseline needs 9+ days
MIN_TEST_ROWS = 100  # per horizon, for a stable MAE


def open_ro(path: Path):
    """SQLite engine that cannot write (safe next to the running collectors)."""
    if not path.exists():
        sys.exit(f"no database at {path}")
    return create_engine(f"sqlite:///file:{path.resolve().as_posix()}?mode=ro&uri=true")


def fmt(v, nd=2, pct=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:+.1f} %" if pct else f"{v:.{nd}f}"


def check_data(series: dict, cal: fc.Calendar, test_days: float) -> tuple[bool, list[str]]:
    """Is there enough data for a meaningful result? (span, readings, test rows, Fridays)."""
    lines, ok = [], True
    if not series:
        return False, ["no accepted crowd readings in this database"]
    t0 = min(int(s.t[0]) for s in series.values())
    t1 = max(int(s.t[-1]) for s in series.values()) + fc.STEP
    span = (t1 - t0) / 1440
    fridays = len({int((t + 180) // 1440) for s in series.values() for t in s.t[np.isfinite(s.v)]
                   if int(((t + 180) // 1440 + 3) % 7) == 4})
    lines.append(f"span {span:.1f} days ({fc._utc(t0):%Y-%m-%d} .. {fc._utc(t1):%Y-%m-%d} UTC), Fridays with data: {fridays}")
    for key, s in sorted(series.items()):
        obs = np.isfinite(s.v)
        test_obs = int((obs & (s.t >= t1 - test_days * 1440)).sum())
        lines.append(f"  {key}: {int(obs.sum())} five-minute averages ({100 * obs.mean():.0f} % of the grid), "
                     f"{test_obs} in the test days")
        if test_obs < MIN_TEST_ROWS // 2:
            lines.append(f"    -> too few test readings for this camera (want >= {MIN_TEST_ROWS // 2})")
    if span < MIN_DAYS:
        ok = False
        lines.append(f"NO-GO: {span:.1f} days < {MIN_DAYS} (need ~{MIN_DAYS - test_days:.0f} train + {test_days:.0f} test days)")
    elif span < GOOD_DAYS:
        lines.append(f"GO (rough): {span:.1f} days. Meaningful from {GOOD_DAYS} days; the model may not beat "
                     "'same time yesterday' yet.")
    else:
        lines.append(f"GO: {span:.1f} days >= {GOOD_DAYS}.")
    if span < test_days + 7.2:
        lines.append(f"  'same time last week' baseline needs >= {test_days + 7:.0f} days: not available yet")
    if span < SOLID_DAYS:
        lines.append(f"  results level off from ~{SOLID_DAYS} days (and 2+ Fridays to learn the Friday effect)")
    return ok, lines


def report(res: fc.Result) -> str:
    out = [f"Train {res.meta['train_range_utc'][0][:16]} .. {res.meta['train_range_utc'][1][:16]} UTC, "
           f"test {res.meta['test_range_utc'][0][:16]} .. {res.meta['test_range_utc'][1][:16]} UTC",
           f"Cameras: {', '.join(res.meta['cameras'])};  'crowded' threshold (p{fc.CROWDED_PCT} of the train days): "
           + ", ".join(f"{k.split('/')[0]} {v:.1f}" for k, v in res.meta["thresholds"].items()), "",
           "MAE in density units (people per 10,000 zone px); improvement = how much smaller the model's error is",
           f"{'horizon':>8} {'n test':>7} {'model':>7} | {'persistence':>21} | {'same time yesterday':>23} | {'same time last week':>23}"]
    for h, m in res.metrics.items():
        cells = []
        for name in ("persistence", "yesterday", "last_week"):
            b = m["baselines"][name]
            cells.append(f"{fmt(b['mae_baseline']):>6} {fmt(b['improvement_pct'], pct=True):>9} ({100 * b['coverage']:3.0f}%)")
        out.append(f"{h:>6} m {m['n_test']:>7} {fmt(m['mae']):>7} | {cells[0]:>21} | {cells[1]:>23} | {cells[2]:>23}")
    out.append("  (baseline MAE, model improvement over it, share of test rows where that baseline exists)")
    out.append("")
    out.append("Per camera (MAE model / persistence):")
    for h, m in res.metrics.items():
        out.append(f"  {h} min: " + ";  ".join(f"{k.split('/')[0]} {fmt(v['mae'])} / {fmt(v['mae_persistence'])} (n {v['n']})"
                                             for k, v in m["per_camera"].items()))
    e = res.early
    out += ["", "Early warning (test days): crossings into 'crowded' and how early the forecast saw them",
            f"  crossings {e['events']}, with a forecast issued before them {e['with_forecast']} "
            f"(no recent reading to forecast from: {e['no_forecast_issued']})",
            f"  warned >= 15 min ahead: {e['warned_15plus']}, >= 30 min: {e['warned_30plus']}, "
            f"median lead {fmt(e['median_lead_min'], 0)} min (same time yesterday: {fmt(e['yesterday_median_lead_min'], 0)} min; "
            "persistence: 0 by definition)"]
    for h, pr in e["precision_recall"].items():
        out.append(f"  {h} min 'crowded' calls: precision {fmt(pr['precision'])}, recall {fmt(pr['recall'])} "
                   f"({pr['n_crowded']} crowded test buckets)")
    return "\n".join(out)


def plot(res: fc.Result, path: Path, cal: fc.Calendar) -> None:
    """Forecast (30 min ahead) vs actual for the camera with the most test rows."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = res.test[30]
    if not len(d["t"]):
        return
    keys, counts = np.unique(d["cam"], return_counts=True)
    key = keys[counts.argmax()]
    m = d["cam"] == key
    order = np.argsort(d["t"][m])
    t = (d["t"][m][order] + 180) / 1440  # Makkah days since epoch
    gap = np.concatenate([[False], np.diff(t) > 30 / 1440])  # camera off air > 30 min: break the lines

    def broken(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.insert(t, np.nonzero(gap)[0], np.nan), np.insert(v, np.nonzero(gap)[0], np.nan)

    fig, ax = plt.subplots(figsize=(13, 4.2), dpi=110)
    ax.plot(t, d["y"][m][order], ".", ms=4, color="#2f6f69", label="actual (5-min average)")
    ax.plot(*broken(d["pred"][m][order]), "-", lw=1.2, color="#c8a45c", label="forecast made 30 min earlier")
    ax.plot(*broken(d["persistence"][m][order]), "-", lw=0.8, color="#9aa", alpha=0.7, label="persistence (value 30 min earlier)")
    thr = res.meta["thresholds"].get(key)
    if thr:
        ax.axhline(thr, color="#ff6f52", ls="--", lw=1, label=f"crowded (p{fc.CROWDED_PCT})")
    lo, hi = t.min(), t.max()
    for et in cal.ev_t:
        x = (et + 180) / 1440
        if lo <= x <= hi:
            ax.axvline(x, color="#ccc", lw=0.6, zorder=0)
    ticks = np.arange(np.floor(lo * 4) / 4, hi, 0.25)
    ax.set_xticks(ticks, [f"{datetime(1970, 1, 1) + timedelta(days=float(x)):%a %H:%M}" for x in ticks], fontsize=8)
    ax.set_ylabel("density (people / 10k px)")
    ax.set_title(f"{key.split('/')[0]}: forecast 30 min ahead vs actual (test days, Makkah time; grey = prayers)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def learning_curve(series: dict, cal: fc.Calendar, test_days: float, sizes: list[int]) -> str:
    """Same test days, growing amounts of training history: when does the model stop improving?"""
    end = max(int(s.t[-1]) for s in series.values()) + fc.STEP
    rows = ["train days | improvement vs persistence (15/30/60 min) | vs same time yesterday (15/30/60 min)"]
    for n in sizes:
        start = end - int((n + test_days) * 1440)
        cut = {k: fc.Series(s.view_id, s.zone, s.t[s.t >= start], s.v[s.t >= start], s.ratio) for k, s in series.items()}
        cut = {k: s for k, s in cut.items() if len(s.t)}
        if min(int(s.t[0]) for s in cut.values()) > start + 60:
            rows.append(f"{n:>10} | not enough history")
            continue
        try:
            res = fc.train(cut, cal, test_days)
        except ValueError as e:
            rows.append(f"{n:>10} | {e}")
            continue
        p = [res.metrics[h]["baselines"]["persistence"]["improvement_pct"] for h in fc.HORIZONS]
        y = [res.metrics[h]["baselines"]["yesterday"]["improvement_pct"] for h in fc.HORIZONS]
        rows.append(f"{n:>10} | {' '.join(f'{fmt(v, pct=True):>9}' for v in p)} | {' '.join(f'{fmt(v, pct=True):>9}' for v in y)}")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--learning-curve", action="store_true")
    ap.add_argument("--test-days", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    real = args.db.resolve() == (ROOT / "data" / "haram.db").resolve()
    args.out = args.out or ROOT / "models" / ("forecast" if real else "forecast_test")

    engine = open_ro(args.db)
    series = fc.load_series(engine)
    if not series:
        sys.exit("no accepted crowd readings in this database")
    t0 = min(int(s.t[0]) for s in series.values())
    t1 = max(int(s.t[-1]) for s in series.values())
    cal = fc.Calendar.load(engine, t0, t1)
    ok, lines = check_data(series, cal, args.test_days)
    print("\n".join(["Data check:"] + lines) + "\n")
    if args.check:
        sys.exit(0 if ok else 1)
    if args.learning_curve:
        span = (t1 - t0) / 1440
        sizes = [n for n in (3, 5, 7, 10, 14, 21, 28) if n + args.test_days <= span]
        print("Learning curve (same test days, growing training history):")
        print(learning_curve(series, cal, args.test_days, sizes))
        return
    if not ok:
        print("Training anyway: treat the numbers below as a dry run, not a result.\n")
    res = fc.train(series, cal, args.test_days)
    res.meta["source_db"] = str(args.db.resolve())
    text = report(res)
    print(text)
    fc.save(res, args.out)
    stamp = datetime.now(MAKKAH).strftime("%Y%m%d_%H%M")
    rep = ROOT / "data" / "reports" / "forecast" / stamp
    rep.mkdir(parents=True, exist_ok=True)
    plot(res, rep / "forecast_vs_actual.png", cal)
    (rep / "report.txt").write_text("Data check:\n" + "\n".join(lines) + "\n\n" + text + "\n", encoding="utf-8")
    (rep / "metrics.json").write_text(json.dumps({"db": str(args.db), "metrics": res.metrics, "early": res.early,
                                                  "meta": res.meta}, indent=2, default=str))
    print(f"\nModel: {args.out}  |  report + plot: {rep}")


if __name__ == "__main__":
    main()
