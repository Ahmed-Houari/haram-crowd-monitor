"""The shadow run's report (edge/shadow.py, table crowd_m2_shadow): the fine-tuned model next to the live one on the same
frames. Read-only on the database.

  - frames with a camera, outside prayer rows: shadow / live per frame (median, 10-90 %), per camera and per source;
  - frames where the live reading came from the prayer rows: the model's reading / the rows estimator's;
  - frames without a camera (approximate): shadow / live;
  - the headline as the dashboard would show it: per minute, area-weighted over the frames of the minute (outside
    prayer rows), shadow vs live, overall and per hour of the day (Makkah).

  .venv/Scripts/python -I scripts/shadow_report.py [--since "2026-10-27 15:00"] [--model qnrf_ft]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "haram.db"
MAKKAH = timezone(timedelta(hours=3))


def q(x: list[float]) -> str:
    a = np.asarray([v for v in x if v is not None and np.isfinite(v)])
    if not len(a):
        return "no data"
    return f"median {np.median(a):.2f} (10-90 % {np.percentile(a, 10):.2f}-{np.percentile(a, 90):.2f}, n {len(a)})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", help="UTC, default: the shadow run's start (config/m2.json shadow.since_utc)")
    ap.add_argument("--model", default="qnrf_ft")
    args = ap.parse_args()
    cfg = (json.loads((ROOT / "config" / "m2.json").read_text(encoding="utf-8")).get("shadow") or {})
    since = args.since or cfg.get("since_utc", "2026-10-27")
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute("SELECT * FROM crowd_m2_shadow WHERE model = ? AND ts_utc >= ? ORDER BY ts_utc",
                                         (args.model, since.replace("T", " ")[:19]))]
    if not rows:
        sys.exit(f"no shadow rows since {since}")
    t0, t1 = rows[0]["ts_utc"], rows[-1]["ts_utc"]
    print(f"shadow run of {args.model}: {len(rows)} frames, {t0[:16]} - {t1[:16]} UTC")
    cam = [r for r in rows if r["live_all"] is not None and r["shadow_all"] is not None and not r["rows"]]
    rat = lambda rs, a="shadow_all", b="live_all": [r[a] / r[b] for r in rs if r[b] and r[b] > 0.05]  # noqa: E731
    print(f"\nwith a camera, outside prayer rows ({len(cam)} frames): shadow / live {q(rat(cam))}")
    for key in ("camera", "source", "method"):
        for v in sorted({r[key] for r in cam if r[key]}):
            print(f"  {key} {v:18s} {q(rat([r for r in cam if r[key] == v]))}")
    for ring in ("inner", "outer", "far"):
        print(f"  ring {ring:6s} {q(rat(cam, f'shadow_{ring}', f'live_{ring}'))}")
    pr = [r for r in rows if r["rows"] and r["shadow_all"] is not None and r["live_all"]]
    print(f"\nprayer rows ({len(pr)} frames): the model / the rows estimator (the live reading then) {q(rat(pr))}")
    ap_ = [r for r in rows if r["live_approx"] and r["shadow_approx"]]
    print(f"without a camera, approximate ({len(ap_)} frames): shadow / live "
          f"{q([r['shadow_approx'] / r['live_approx'] for r in ap_ if r['live_approx'] > 0.05])}")
    # the headline per minute: area-weighted over the minute's frames (as scripts/live_shot.py pools it)
    mins = defaultdict(lambda: np.zeros(3))
    for r in cam:
        a = r["area_all"] or 0
        if a > 0:
            mins[r["ts_utc"][:16]] += [r["live_all"] * a, r["shadow_all"] * a, a]
    ml = {m: (v[0] / v[2], v[1] / v[2]) for m, v in mins.items() if v[2] > 0}
    print(f"\nheadline per minute (outside prayer rows, {len(ml)} minutes): live median "
          f"{np.median([v[0] for v in ml.values()]):.2f}, shadow median {np.median([v[1] for v in ml.values()]):.2f}; "
          f"shadow / live {q([v[1] / v[0] for v in ml.values() if v[0] > 0.05])}")
    hours = defaultdict(list)
    for m, (lv, sh) in ml.items():
        h = datetime.fromisoformat(m).replace(tzinfo=timezone.utc).astimezone(MAKKAH).hour
        hours[h].append((lv, sh))
    print("by hour (Makkah): live / shadow median people per m^2")
    print("  " + "  ".join(f"{h:02d}h {np.median([a for a, _ in v]):.2f}/{np.median([b for _, b in v]):.2f}"
                           for h, v in sorted(hours.items())))


if __name__ == "__main__":
    main()
