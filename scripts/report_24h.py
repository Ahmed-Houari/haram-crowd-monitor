"""24-hour collection report (numbers + a few images), written to data/reports/<Makkah date>/.

- accepted minutes per Makkah-time hour for each camera, hours with zero readings
- collector uptime % (collector_minute), frame sources, yt-dlp lookups per day, TomTom polls
- coverage for forecasting: 5-min windows with >= 1 accepted reading, longest gap
- live 2d (count vs zoom) on crowd_frame, via scripts/test_pipeline.py analysis
- the annotated samples (data/samples/live_*.png) if any
- clustering of the saved NIGHT thumbnails that matched no reference (like the view survey),
  plus a contact sheet of the night thumbnails that did match cam0/cam5
Output: report.md (summary), report.json, night_clusters_sheet.jpg, night_matched_sheet.jpg, plots.

Run from the project root:  .venv/Scripts/python scripts/report_24h.py [--hours 24]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from sqlalchemy import select  # noqa: E402

from edge import db  # noqa: E402
from edge.frame_grabber import load_state  # noqa: E402
from edge.view_matcher import PROJECT_ROOT, load_config  # noqa: E402

MAKKAH = timezone(timedelta(hours=3))
THUMBS = PROJECT_ROOT / "data" / "views" / "thumbs"
NIGHT_CLUSTER_THRESHOLD = 0.25  # thumbnails: group camera states (0.15 split every zoom step in the survey)


def cluster_night_thumbs(out: Path) -> list[dict]:
    """Cluster night 'unmatched' thumbnails; write a contact sheet. Returns cluster infos (most frequent first)."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from view_survey import Embedder, _sheet_row, cluster, pick_thumbs, runs

    index = THUMBS / "index.csv"
    if not index.exists():
        return []
    rows = [r for r in csv.DictReader(open(index)) if r["kind"] == "unmatched" and r["night"] == "1"
            and (THUMBS / r["file"]).exists()]
    if len(rows) < 2:
        return []
    emb = Embedder()
    embs = np.stack([emb(cv2.imread(str(THUMBS / r["file"]))) for r in rows])
    labels = cluster(embs, NIGHT_CLUSTER_THRESHOLD)
    all_runs = runs(labels)
    infos, sheet = [], []
    for c in range(labels.max() + 1):
        idx = np.flatnonzero(labels == c)
        picks = pick_thumbs(idx, embs)
        c_runs = [n for lab, n in all_runs if lab == c]
        info = {"cluster": c, "frames": int(len(idx)), "pct": round(100 * len(idx) / len(rows), 1),
                "visits": len(c_runs), "median_stay_s": float(np.median(c_runs) * 300),  # 1 thumbnail / 5 min
                "makkah_times": [rows[i]["makkah_time"] for i in idx[:6]],
                "thumbs": [rows[i]["file"] for i in picks]}
        infos.append(info)
        if c < 16:
            sheet.append(_sheet_row(info, [THUMBS / f for f in info["thumbs"]]))
    if sheet:
        cv2.imwrite(str(out / "night_clusters_sheet.jpg"), np.vstack(sheet), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return infos


def night_matched_sheet(out: Path) -> dict[str, int]:
    files = sorted((THUMBS / "night_matched").glob("*.jpg"))
    counts: dict[str, int] = defaultdict(int)
    for f in files:
        counts[f.stem.rsplit("_", 1)[0]] += 1
    if files:
        tiles = [cv2.resize(cv2.imread(str(f)), (320, 180)) for f in files[:24]]
        while len(tiles) % 4:
            tiles.append(np.zeros((180, 320, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        cv2.imwrite(str(out / "night_matched_sheet.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hours", type=float, default=24)
    args = parser.parse_args()
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(hours=args.hours)
    out = PROJECT_ROOT / "data" / "reports" / f"{end.astimezone(MAKKAH):%Y-%m-%d_%H%M}"
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    views = [v["id"] for v in cfg["views"]]

    with db.get_engine().connect() as c:
        frames = c.execute(select(db.crowd_frame).where(db.crowd_frame.c.ts_utc >= start)).mappings().all()
        mins = c.execute(select(db.collector_minute).where(db.collector_minute.c.minute_utc >= start)).mappings().all()
        polls = c.execute(select(db.traffic_poll).where(db.traffic_poll.c.poll_utc >= start)).mappings().all()
        first = c.execute(select(db.collector_minute.c.minute_utc).order_by(db.collector_minute.c.minute_utc).limit(1)).scalar()

    # accepted minutes per Makkah hour per camera
    hours = [(start + timedelta(hours=h)).astimezone(MAKKAH).replace(minute=0, second=0, microsecond=0)
             for h in range(int(args.hours))]
    acc_min = defaultdict(set)  # (view, hour) -> minutes
    acc5 = defaultdict(set)  # view -> 5-min windows
    for f in frames:
        if not f["accepted"]:
            continue
        ts = db.utc(f["ts_utc"])
        acc_min[(f["view_id"], ts.astimezone(MAKKAH).replace(minute=0, second=0, microsecond=0))].add(ts.replace(second=0))
        acc5[f["view_id"]].add(ts.replace(minute=ts.minute // 5 * 5, second=0))
    table = [{"hour_makkah": h.strftime("%m-%d %H:00"), **{v: len(acc_min[(v, h)]) for v in views}} for h in hours]
    zero_all = sum(all(row[v] == 0 for v in views) for row in table)
    zero_per = {v: sum(row[v] == 0 for row in table) for v in views}

    # uptime
    up_start = max(start, db.utc(first)) if first else start
    window_min = max(int((end - up_start).total_seconds() // 60), 1)
    alive = sum(1 for m in mins if m["frames"] > 0)
    windows_total = int((end - up_start).total_seconds() // 300) or 1
    any5 = set().union(*acc5.values()) if acc5 else set()
    acc_ts = sorted({db.utc(f["ts_utc"]) for f in frames if f["accepted"]})
    gaps = [b - a for a, b in zip([up_start] + acc_ts, acc_ts + [end])]
    report = {
        "window_utc": [start.isoformat(), end.isoformat()], "collection_since_utc": db.utc(first).isoformat() if first else None,
        "uptime_pct": round(100 * alive / window_min, 1), "minutes_alive": alive, "minutes_in_window": window_min,
        "frames": sum(m["frames"] for m in mins), "accepted_frames": sum(m["accepted"] for m in mins),
        "fallback_frames": sum(m["fallback"] for m in mins), "yt_dlp_lookups_per_day": load_state().get("calls", {}),
        "tomtom_polls_ok": sum(p["tiles_ok"] == p["tiles_total"] for p in polls), "tomtom_polls": len(polls),
        "hours_with_zero_readings_all_cameras": zero_all, "hours_with_zero_readings_per_camera": zero_per,
        "coverage_5min_pct": {v: round(100 * len(acc5[v]) / windows_total, 1) for v in views},
        "coverage_5min_any_camera_pct": round(100 * len(any5) / windows_total, 1),
        "longest_gap_hours": round(max(gaps).total_seconds() / 3600, 1) if gaps else None,
        "accepted_minutes_per_hour": table,
        "samples": sorted(p.name for p in (PROJECT_ROOT / "data" / "samples").glob("live_*.png")),
        "night_matched_thumbs": night_matched_sheet(out),
        "night_unmatched_clusters": cluster_night_thumbs(out),
        "quality": {v["id"]: v.get("quality") for v in cfg["views"]},
        "zoom_tolerance": {v["id"]: v["zoom_tolerance"] for v in cfg["views"]},
    }
    if any(f["accepted"] for f in frames) or frames:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        import test_pipeline

        test_pipeline.OUT = out
        try:
            report["live_2d"] = test_pipeline.analyze(*test_pipeline.load_db(start))["zones"]
        except SystemExit:
            report["live_2d"] = {}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))

    md = [f"# 24 h collection report, {start.astimezone(MAKKAH):%Y-%m-%d %H:%M} to {end.astimezone(MAKKAH):%Y-%m-%d %H:%M} Makkah",
          "", f"- Uptime: {report['uptime_pct']} % ({alive} of {window_min} min)",
          f"- Frames: {report['frames']:,} checked, {report['accepted_frames']:,} accepted, {report['fallback_frames']:,} from the SBA fallback",
          f"- yt-dlp lookups per day: {report['yt_dlp_lookups_per_day']}",
          f"- TomTom: {report['tomtom_polls_ok']}/{report['tomtom_polls']} polls OK",
          f"- Hours with zero readings (all cameras): {zero_all} of {len(hours)}; per camera: {zero_per}",
          f"- 5-min windows with a reading: {report['coverage_5min_pct']}, any camera {report['coverage_5min_any_camera_pct']} %",
          f"- Longest gap without any reading: {report['longest_gap_hours']} h",
          f"- Annotated samples: {report['samples'] or 'none (no reference view accepted)'}",
          f"- Night thumbnails: {sum(i['frames'] for i in report['night_unmatched_clusters'])} unmatched in "
          f"{len(report['night_unmatched_clusters'])} clusters; matched per view: {report['night_matched_thumbs']}",
          "", "| Hour (Makkah) | " + " | ".join(views) + " |", "|---|" + "---|" * len(views)]
    md += [f"| {r['hour_makkah']} | " + " | ".join(str(r[v]) for v in views) + " |" for r in table]
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:12]))
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()
