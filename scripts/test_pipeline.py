"""Step 2d: run the whole crowd pipeline live and check that zone counts do not depend on zoom.

For N minutes: grab a frame every 15 s, match the reference views (±20 % zoom), run DM-Count on
accepted frames, count per zone. Writes (numbers only) data/views/2d/frames.csv + zones.csv,
3 annotated samples to data/samples/pipeline_*.png (explicit test samples: zones + heatmap),
plots data/views/2d/counts_time.png + count_vs_zoom.png, and summary.json.

Zoom check per zone: Pearson r(count, zoom), and an OLS fit count = a + b*zoom + c*minutes
(the time term absorbs real crowd changes during the run). Drift = |b| * observed zoom range /
mean count. Rule (user): drift > ~10 % -> fall back to ±5 % zoom tolerance for that camera.

Run from the project root:  .venv/Scripts/python scripts/test_pipeline.py --minutes 30
Re-analyze only:            .venv/Scripts/python scripts/test_pipeline.py --analyze
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from edge.frame_grabber import SAMPLES_DIR, FrameGrabber  # noqa: E402
from edge.pipeline import CrowdPipeline  # noqa: E402
from edge.pipeline import annotate_reading as annotate  # noqa: E402
from edge.view_matcher import PROJECT_ROOT  # noqa: E402

OUT = PROJECT_ROOT / "data" / "views" / "2d"
DRIFT_LIMIT = 0.10
MIN_POINTS = 8
MIN_ZOOM_RANGE = 0.04


def run(minutes: float, interval: float, stop_after: int = 0) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    pipe = CrowdPipeline(auto_capture=False)  # the 24/7 collector captures missing references
    print(f"Views: {', '.join(pipe.view_ids)}")
    accepted = 0
    sampled_views: set[str] = set()
    extra: tuple[float, np.ndarray, str] | None = None  # (|zoom-1|, image, name) best 3rd sample, RAM only
    n_samples = 0
    deadline = time.monotonic() + minutes * 60
    with open(OUT / "frames.csv", "w", newline="") as ff, open(OUT / "zones.csv", "w", newline="") as fz:
        wf, wz = csv.writer(ff), csv.writer(fz)
        wf.writerow(["ts", "source", "view_id", "zoom", "inliers", "accepted", "n_zones", "infer_ms"])
        wz.writerow(["ts", "source", "view_id", "zone", "count", "density", "zoom", "accepted"])
        for frame in FrameGrabber.from_env(interval_s=interval).frames():
            reading, m, result = pipe.process(frame)
            ts = frame.captured_at.isoformat(timespec="seconds")
            wf.writerow([ts, frame.source, m.view_id if m else "", f"{m.zoom:.4f}" if m else "", m.inliers if m else 0,
                         int(bool(reading and reading.accepted)), len(reading.zones) if reading else 0,
                         f"{result.inference_ms:.0f}" if result else ""])
            line = f"{frame.captured_at:%H:%M:%S} UTC  {frame.source:8s} "
            if m:
                line += f"{m.view_id:16s} zoom={m.zoom:.3f} inl={m.inliers:4d} "
            if reading:
                for z, v in reading.zones.items():
                    wz.writerow([ts, frame.source, reading.view_id, z, v["count"], v["density"], f"{reading.zoom:.4f}",
                                 int(reading.accepted)])
                accepted += 1
                line += ("ACCEPTED " if reading.accepted else "outside tolerance (stored) ") + " ".join(f"{z}={v['count']:,.0f}" for z, v in reading.zones.items())
                if reading.view_id not in sampled_views and n_samples < 3:
                    path = SAMPLES_DIR / f"pipeline_{n_samples + 1}_{reading.view_id}_{frame.captured_at:%H%M%S}.png"
                    cv2.imwrite(str(path), annotate(pipe, frame, m, result, reading))
                    sampled_views.add(reading.view_id)
                    n_samples += 1
                elif extra is None or abs(reading.zoom - 1) > extra[0]:
                    extra = (abs(reading.zoom - 1), annotate(pipe, frame, m, result, reading),
                             f"{reading.view_id}_{frame.captured_at:%H%M%S}")
            elif m and abs(m.zoom - 1) > 0.2:
                line += "rejected (zoom > ±20 %)"
            elif m:
                line += "no zone fully visible"
            print(line, flush=True)
            ff.flush(), fz.flush()
            del frame
            pipe.reload_if_changed()
            if time.monotonic() >= deadline or (stop_after and accepted >= stop_after):
                break
    while n_samples < 3 and extra is not None:
        cv2.imwrite(str(SAMPLES_DIR / f"pipeline_{n_samples + 1}_{extra[2]}_zoom.png"), extra[1])
        n_samples += 1
        extra = None
    print(f"Samples written: {n_samples}")


def load_csv() -> tuple[list[tuple], dict, list[dict]]:
    """(minutes [(minute, frames, accepted, youtube, fallback)], accepted frames per view, zone rows) from the CSVs."""
    frames = list(csv.DictReader(open(OUT / "frames.csv")))
    per_min = defaultdict(lambda: [0, 0, 0, 0])
    per_view = defaultdict(int)
    for f in frames:
        m = per_min[datetime.fromisoformat(f["ts"]).replace(second=0)]
        m[0] += 1
        m[1] += f["accepted"] == "1"
        m[2] += f["source"] == "youtube"
        m[3] += f["source"] == "fallback"
        per_view[f["view_id"]] += f["accepted"] == "1"
    zones = [{"ts": datetime.fromisoformat(z["ts"]), "view_id": z["view_id"], "zone": z["zone"],
              "count": float(z["count"]), "zoom": float(z["zoom"])} for z in csv.DictReader(open(OUT / "zones.csv"))]
    return [(k, *v) for k, v in sorted(per_min.items())], dict(per_view), zones


def load_db(since: datetime) -> tuple[list[tuple], dict, list[dict]]:
    """Same, from the 24/7 collector's database (collector_minute + crowd_frame)."""
    from sqlalchemy import select

    from edge import db

    with db.get_engine().connect() as c:
        mins = c.execute(select(db.collector_minute).where(db.collector_minute.c.minute_utc >= since)
                         .order_by(db.collector_minute.c.minute_utc)).mappings().all()
        rows = c.execute(select(db.crowd_frame).where(db.crowd_frame.c.ts_utc >= since)).mappings().all()  # all ±20 %
    minutes = [(db.utc(r["minute_utc"]), r["frames"], r["accepted"], r["youtube"], r["fallback"]) for r in mins]
    zones = [{"ts": db.utc(r["ts_utc"]), "view_id": r["view_id"], "zone": r["zone"], "count": r["count"],
              "zoom": r["zoom"]} for r in rows]
    per_view = defaultdict(set)
    for r in rows:
        if r["accepted"]:
            per_view[r["view_id"]].add(r["ts_utc"])
    return minutes, {v: len(ts) for v, ts in per_view.items()}, zones


def analyze(minutes: list[tuple], per_view: dict, zones: list[dict]) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not minutes:
        raise SystemExit("no data to analyze")
    n = sum(m[1] for m in minutes)
    t0 = minutes[0][0]
    minute = lambda ts: (ts - t0).total_seconds() / 60  # noqa: E731
    acc_min = [m for m in minutes if m[2] > 0]
    all_5 = {m[0].replace(minute=m[0].minute // 5 * 5) for m in minutes}
    acc_5 = {m[0].replace(minute=m[0].minute // 5 * 5) for m in acc_min}
    summary = {"frames": n, "start": t0.isoformat(), "end": minutes[-1][0].isoformat(), "minutes": len(minutes),
               "views": {v: {"accepted_pct": round(100 * k / n, 1)} for v, k in sorted(per_view.items()) if v},
               "accepted_pct": round(100 * sum(m[2] for m in minutes) / n, 1),
               "readings_per_min": round(sum(m[2] for m in minutes) / len(minutes), 2),
               "minutes_with_reading_pct": round(100 * len(acc_min) / len(minutes), 1),
               "windows_5min_with_reading_pct": round(100 * len(acc_5) / len(all_5), 1),
               "sources": {"youtube": sum(m[3] for m in minutes), "fallback": sum(m[4] for m in minutes)},
               "zones": {}}

    groups = defaultdict(list)
    for z in zones:
        groups[(z["view_id"], z["zone"])].append((minute(z["ts"]), z["zoom"], z["count"]))
    keys = sorted(groups)
    fig1, ax1 = plt.subplots(len(keys) or 1, 1, figsize=(10, 3.2 * max(len(keys), 1)), squeeze=False)
    fig2, ax2 = plt.subplots(1, len(keys) or 1, figsize=(5.5 * max(len(keys), 1), 4.5), squeeze=False)
    for i, key in enumerate(keys):
        t, zoom, cnt = (np.array(c) for c in zip(*groups[key]))
        res = {"n": len(cnt), "mean_count": round(float(cnt.mean()), 1),
               "zoom_min": round(float(zoom.min()), 3), "zoom_max": round(float(zoom.max()), 3)}
        zr = float(zoom.max() - zoom.min())
        if len(cnt) >= 3 and zr > 0:
            res["pearson_r"] = round(float(np.corrcoef(zoom, cnt)[0, 1]), 3)
            X = np.column_stack([np.ones_like(zoom), zoom, t]) if np.ptp(t) > 0 else np.column_stack([np.ones_like(zoom), zoom])
            b = np.linalg.lstsq(X, cnt, rcond=None)[0][1]
            b_simple = np.polyfit(zoom, cnt, 1)[0]
            res["slope_per_zoom"] = round(float(b), 1)
            res["drift_pct_observed_range"] = round(100 * abs(b) * zr / cnt.mean(), 1)
            res["drift_pct_full_pm20"] = round(100 * abs(b) * 0.4 / cnt.mean(), 1)
            res["drift_pct_simple_fit"] = round(100 * abs(b_simple) * zr / cnt.mean(), 1)
        if len(cnt) < MIN_POINTS or zr < MIN_ZOOM_RANGE:
            res["verdict"] = f"not enough data (need >= {MIN_POINTS} readings and zoom range >= {MIN_ZOOM_RANGE})"
        else:
            res["verdict"] = "PASS" if res["drift_pct_observed_range"] <= 100 * DRIFT_LIMIT else "FAIL -> use ±5 % for this camera"
        summary["zones"][f"{key[0]}/{key[1]}"] = res

        a = ax1[i, 0]
        sc = a.scatter(t, cnt, c=zoom, cmap="viridis", s=22)
        fig1.colorbar(sc, ax=a, label="zoom")
        a.set_title(f"{key[0]} / {key[1]}: count over time")
        a.set_xlabel(f"minutes since {t0:%H:%M} UTC"), a.set_ylabel("people in zone")
        a = ax2[0, i]
        a.scatter(zoom, cnt, s=22)
        if zr > 0 and len(cnt) >= 3:
            xs = np.linspace(zoom.min(), zoom.max(), 10)
            a.plot(xs, np.polyval(np.polyfit(zoom, cnt, 1), xs), "r-")
        a.set_title(f"{key[0]} / {key[1]}\nr={res.get('pearson_r', 'n/a')}, drift={res.get('drift_pct_observed_range', 'n/a')}% "
                    f"({res['verdict'].split(' ')[0]})", fontsize=10)
        a.set_xlabel("zoom factor vs. reference"), a.set_ylabel("people in zone")
    fig1.tight_layout(), fig2.tight_layout()
    fig1.savefig(OUT / "counts_time.png", dpi=90), fig2.savefig(OUT / "count_vs_zoom.png", dpi=90)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def synthetic(apply: bool = False) -> dict:
    """Synthetic density zoom test (edge/zoom_range.py) on every approved view's primary reference: density per 10k
    primary px in the visible part of the zone, digital zoom 0.90..2.00x about several points of the zone.

    Writes data/views/2d/synthetic_density.json + density_vs_zoom_synthetic.png; with apply=True also each view's
    zoom_min / zoom_max (range rule in edge/zoom_range.py) into config/views.json.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from edge import zoom_range
    from edge.crowd_model import CrowdModel
    from edge.view_matcher import load_config, save_config, zones_in_ref_px

    cfg = load_config()
    views = [v for v in cfg["views"] if v.get("status") != "pending" and (PROJECT_ROOT / v["ref_image"]).exists()]
    model = CrowdModel()
    out = {"rule": zoom_range.RANGE_RULE, "views": {}}
    fig, axes = plt.subplots(1, len(views), figsize=(6 * len(views), 4.5), squeeze=False)
    for ax, v in zip(axes[0], views):
        img = cv2.imread(str(PROJECT_ROOT / v["ref_image"]))
        model.predict(img)  # warm-up
        res = zoom_range.run(img, zones_in_ref_px(v, cfg), model, cfg)
        for zone, z in res["zones"].items():
            rows = zoom_range.summarize(z["points"])
            lo, hi = zoom_range.pick_range(rows, v["zoom_tolerance"])
            out["views"][v["id"]] = {"zone": zone, "base_density": z["base_density"], "base_count": z["base_count"],
                                     "by_scale": rows, "zoom_min": lo, "zoom_max": hi, "points": z["points"]}
            print(f"\n{v['id']}/{zone}: unzoomed {z['base_density']:.1f} per 10k px ({z['base_count']:,.0f} people)")
            print(f"  {'zoom':>5} {'visible':>8} {'same patch mean':>16} {'max':>6} {'whole zone mean':>16} {'max':>6}")
            for r in rows:
                print(f"  {r['scale']:>5.2f} {100 * r['visible_mean']:>7.0f}% {100 * r['same_mean']:>+15.1f}% "
                      f"{100 * r['same_maxabs']:>5.0f}% {100 * r['whole_mean']:>+15.1f}% {100 * r['whole_maxabs']:>5.0f}%")
            print(f"  -> zoom range {lo:.2f}..{hi:.2f} (old {1 - v['zoom_tolerance']:.2f}..{1 + v['zoom_tolerance']:.2f})")
            xs = [r["scale"] for r in rows]
            ax.plot(xs, [100 * r["same_mean"] for r in rows], "o-", label="same floor patch (model bias)")
            ax.plot(xs, [100 * r["whole_mean"] for r in rows], "s--", label="vs whole zone (what the series sees)")
            ax.axvspan(lo, hi, color="green", alpha=0.1)
            ax.axhspan(-100 * zoom_range.DRIFT_LIMIT, 100 * zoom_range.DRIFT_LIMIT, color="grey", alpha=0.08)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(f"{v['id']}: density drift vs digital zoom")
        ax.set_xlabel("zoom factor (green = accepted range)"), ax.set_ylabel("density drift, mean over zoom centres (%)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "density_vs_zoom_synthetic.png", dpi=90)
    (OUT / "synthetic_density.json").write_text(json.dumps(out, indent=1))
    if apply:
        cfg = load_config()  # fresh: the collector may have changed it meanwhile
        for v in cfg["views"]:
            r = out["views"].get(v["id"])
            if r:
                v["zoom_min"], v["zoom_max"] = r["zoom_min"], r["zoom_max"]
                v["zoom_test"] = f"data/views/2d/synthetic_density.json ({datetime.now(timezone.utc):%Y-%m-%d})"
        save_config(cfg)
        print("zoom ranges written to config/views.json")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--stop-after-readings", type=int, default=0, help="stop early after this many accepted frames")
    parser.add_argument("--analyze", action="store_true", help="only re-run the analysis on the saved CSVs")
    parser.add_argument("--from-db", metavar="SINCE_UTC", help="analyze the 24/7 collector's data since this UTC time "
                        "(e.g. 2026-10-23T19:30) instead of a live run")
    parser.add_argument("--synthetic", action="store_true", help="density zoom test (0.9-2.0x) on the reference frames")
    parser.add_argument("--apply", action="store_true", help="with --synthetic: write each view's zoom range to the config")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        synthetic(args.apply)
        return
    if args.from_db:
        since = datetime.fromisoformat(args.from_db)
        analyze(*load_db(since if since.tzinfo else since.replace(tzinfo=timezone.utc)))
        return
    if not args.analyze:
        run(args.minutes, args.interval, args.stop_after_readings)
    analyze(*load_csv())


if __name__ == "__main__":
    main()
