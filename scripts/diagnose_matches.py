"""Why are so few frames counted? Match the saved survey thumbnails against every reference view (read-only).

For each thumbnail in data/views/thumbs/index.csv (default: today, Makkah date): inliers + zoom against each
reference (thumbnail resolution, weak matches allowed), the best camera, and the period (day / evening / night).
The night_matched thumbnails (frames the collector DID match) are the control: they show how many inliers a real
match gets at this resolution. Prints a summary per period and a conclusion; --csv writes the per-frame table.

Run:  .venv/Scripts/python scripts/diagnose_matches.py [--date 2026-10-24 | --all] [--csv out.csv]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge.view_matcher import RATIO_TEST, RANSAC_PX, ViewMatcher, load_config, zoom_factor  # noqa: E402

THUMBS = ROOT / "data" / "views" / "thumbs"
WORK = 480


def period(makkah: str) -> str:
    h = int(makkah[11:13])
    return "day" if 6 <= h < 16 else "evening" if 16 <= h < 22 else "night"


def per_ref(matcher: ViewMatcher, img: np.ndarray) -> dict[str, tuple[int, float | None]]:
    """{view: (inliers, zoom)} against every reference (no minimum)."""
    fh, fw = img.shape[:2]
    kp, desc = matcher._features(img)
    out = {}
    for ref in matcher.refs:
        if len(desc) < 4 or len(ref.desc) < 4:
            out[ref.id] = (0, None)
            continue
        pairs = matcher.flann.knnMatch(ref.desc, desc, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO_TEST * p[1].distance]
        if len(good) < 4:
            out[ref.id] = (len(good), None)
            continue
        src, dst = ref.kp[[m.queryIdx for m in good]], kp[[m.trainIdx for m in good]]
        Hw, inl = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX * WORK / 960)
        if Hw is None:
            out[ref.id] = (0, None)
            continue
        sr, sf = matcher.work_width / ref.size[0], matcher.work_width / fw
        H = np.diag([1 / sf, 1 / sf, 1]) @ Hw @ np.diag([sr, sr, 1])
        out[ref.id] = (int(inl.sum()), round(zoom_factor(H, ref.size, (fw, fh)), 3))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="Makkah date (default today)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()

    cfg = load_config()
    refs = {v["id"]: cv2.imread(str(ROOT / v["ref_image"])) for v in cfg["views"] if (ROOT / v["ref_image"]).exists()}
    matcher = ViewMatcher(refs, cfg, work_width=WORK, min_inliers=8)
    rows = []
    with open(THUMBS / "index.csv", newline="") as f:
        for r in csv.DictReader(f):
            if not args.all and not r["makkah_time"].startswith(args.date):
                continue
            img = cv2.imread(str(THUMBS / r["file"]))
            if img is None:
                continue
            res = per_ref(matcher, img)
            best = max(res, key=lambda k: res[k][0])
            rows.append({"makkah": r["makkah_time"], "kind": r["kind"], "period": period(r["makkah_time"]),
                         "file": r["file"], "best": best, "inliers": res[best][0], "zoom": res[best][1],
                         **{f"in_{k}": v[0] for k, v in res.items()}})
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    ctrl = [r["inliers"] for r in rows if r["kind"] == "night_matched"]
    print(f"{len(rows)} thumbnails ({'all dates' if args.all else args.date}); control = night frames the collector matched: "
          f"{len(ctrl)} (median inliers {np.median(ctrl) if ctrl else '-'})")
    bands = [(0, 10), (10, 20), (20, 30), (30, 60), (60, 10**6)]
    by = defaultdict(list)
    for r in rows:
        if r["kind"] == "unmatched":
            by[r["period"]].append(r)
    print(f"{'unmatched':>10} {'n':>4}  best inliers: " + "  ".join(f"{a}-{b if b < 10**6 else ''}" for a, b in bands)
          + "   best camera when >= 20 inliers (zoom range)")
    for per in ("day", "evening", "night"):
        rs = by.get(per, [])
        if not rs:
            continue
        hist = [sum(a <= r["inliers"] < b for r in rs) for a, b in bands]
        strong = [r for r in rs if r["inliers"] >= 20]
        cams = defaultdict(list)
        for r in strong:
            cams[r["best"]].append(r["zoom"])
        cam_txt = ", ".join(f"{k} {len(v)} ({min(z for z in v if z):.2f}-{max(z for z in v if z):.2f})" for k, v in cams.items() if any(v))
        print(f"{per:>10} {len(rs):>4}  " + "  ".join(f"{h:>5}" for h in hist) + f"   {cam_txt or '-'}")
    weak = [r for r in rows if r["kind"] == "unmatched" and 20 <= r["inliers"] < 50]
    print(f"\nunmatched frames with 20-49 inliers (would match with a lower bar or a closer reference): {len(weak)}")
    for r in weak[:12]:
        print(f"  {r['makkah']} {r['period']:>7} {r['best']:<16} inliers {r['inliers']:>3} zoom {r['zoom']}  {r['file']}")


if __name__ == "__main__":
    main()
