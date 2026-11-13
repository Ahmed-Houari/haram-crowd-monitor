"""Step 2b: how many frames match a reference view, and how many readings per minute?

Runs the view matcher on the live stream for N minutes (no crowd model, frames stay in RAM)
and logs one row per frame to data/views/matcher_log.csv (numbers only). Summary: matched /
accepted % per view at the configured tolerance and at ±5 %, accepted readings per minute,
and the share of 1-min and 5-min windows with at least one reading.

Run from the project root:  .venv/Scripts/python scripts/measure_views.py --minutes 60
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge.frame_grabber import FrameGrabber  # noqa: E402
from edge.view_matcher import PROJECT_ROOT, ViewMatcher  # noqa: E402

LOG_PATH = PROJECT_ROOT / "data" / "views" / "matcher_log.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--minutes", type=float, default=60)
    parser.add_argument("--interval", type=float, default=15.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    matcher = ViewMatcher.from_config()
    if not matcher.refs:
        raise SystemExit("No reference images yet. Run scripts/capture_refs.py first.")
    print(f"Views: {', '.join(f'{r.id} (zoom {r.zoom_range[0]:.2f}..{r.zoom_range[1]:.2f})' for r in matcher.refs)}")

    rows = []
    deadline = time.monotonic() + args.minutes * 60
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "view_id", "zoom", "inliers", "accepted", "accepted_5pct", "match_ms"])
        for frame in FrameGrabber.from_env(interval_s=args.interval).frames():
            t0 = time.perf_counter()
            m = matcher.match(frame.image)
            ms = (time.perf_counter() - t0) * 1000
            row = {"ts": frame.captured_at, "view": m.view_id if m else "", "zoom": m.zoom if m else None,
                   "inl": m.inliers if m else 0, "acc": bool(m and m.accepted),
                   "acc5": bool(m and abs(m.zoom - 1) <= 0.05)}
            del frame
            rows.append(row)
            w.writerow([row["ts"].isoformat(timespec="seconds"), row["view"],
                        f"{row['zoom']:.4f}" if m else "", row["inl"], int(row["acc"]), int(row["acc5"]), round(ms)])
            f.flush()
            print(f"{row['ts']:%H:%M:%S} UTC  {row['view'] or '-':16s} "
                  f"{'zoom=%.3f inliers=%d' % (m.zoom, m.inliers) if m else '':26s} "
                  f"{'ACCEPTED' if row['acc'] else ''}  ({ms:.0f} ms)", flush=True)
            if time.monotonic() >= deadline:
                break

    n = len(rows)
    if not n:
        return
    print(f"\n=== {n} frames, {rows[0]['ts']:%H:%M}-{rows[-1]['ts']:%H:%M} UTC ===")
    for vid in dict.fromkeys(r.id for r in matcher.refs):
        mine = [r for r in rows if r["view"] == vid]
        print(f"{vid:16s} matched {100 * len(mine) / n:5.1f}%   accepted (zoom range): "
              f"{100 * sum(r['acc'] for r in mine) / n:5.1f}%   accepted ±5%: {100 * sum(r['acc5'] for r in mine) / n:5.1f}%")
    acc = [r for r in rows if r["acc"]]
    print(f"{'all views':16s} accepted {100 * len(acc) / n:5.1f}% of frames")
    for label, key in (("±tolerance", "acc"), ("±5%", "acc5")):
        per_min = Counter(r["ts"].replace(second=0, microsecond=0) for r in rows if r[key])
        minutes = {r["ts"].replace(second=0, microsecond=0) for r in rows}
        per_5 = defaultdict(int)
        for r in rows:
            per_5[r["ts"].replace(minute=r["ts"].minute // 5 * 5, second=0, microsecond=0)] += r[key]
        print(f"[{label}] readings/min: mean {sum(per_min.values()) / max(len(minutes), 1):.2f}; "
              f"minutes with >=1 reading: {100 * len(per_min) / max(len(minutes), 1):.0f}%; "
              f"5-min windows with >=1 reading: {100 * sum(v > 0 for v in per_5.values()) / max(len(per_5), 1):.0f}%")
    print(f"Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
