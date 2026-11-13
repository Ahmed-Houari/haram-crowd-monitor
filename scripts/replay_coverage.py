"""Replay saved frames through the visible-part density logic: how many more minutes would be counted? (read-only)

Inputs (already on disk; nothing new is saved):
- survey thumbnails, data/views/thumbs/index.csv: 480x270, one unmatched frame every 5 min since 2026-10-23 23:07
  Makkah (+ the few night_matched ones). Until 2026-10-24 14:21 the live matcher found almost nothing, so they are a
  near-uniform time sample of the broadcast; after that, frames the new 640 px pass matched are no longer in it.
- test samples data/samples/*.png (side-by-side images: raw 720p frame in the left panel; acc_*.png full frames).
Each image is upscaled to 1920x1080 and matched like the collector does (960 px, 640 px second pass, strict = 50
inliers), then the zone's visible part and its zoom are computed (edge/view_matcher.py) and three rules decide:
  old    zone fully visible, zoom within the old tolerance (1 -+ zoom_tolerance)
  new    >= 30 % of the zone visible, zoom within the view's tested range (zoom_min..zoom_max; from the config, else
         from data/views/2d/synthetic_density.json)
  new2x  as new, but zoom-in accepted up to 2.0x (the user's first idea, before the zoom test)
No crowd model runs: this is coverage only.

Thumbnails have 1/4 of the pixels of a live frame, so they keep fewer inliers; the samples are matched twice (full
frame, and shrunk to a thumbnail and back) to show the loss. --min-inliers gives a relaxed estimate for thumbnails.
Pending cameras (cam6, cam7: no full-res reference yet) are matched by their survey thumbnail (zoom +-10 %) and
reported apart, as "if approved".

Counted minutes per hour from the share p of frames counted (a frame every 15 s, 4 per minute): between 60p (shots
of a minute or longer) and 60(1-(1-p)^4) (shots shorter than 15 s); the middle value is the estimate.

Run:  .venv/Scripts/python scripts/replay_coverage.py [--min-inliers 50] [--csv data/views/thumbs/replay.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge.view_matcher import (FULL_VISIBLE, MIN_INLIERS, MIN_VISIBLE, ViewMatcher, load_config,  # noqa: E402
                               zoom_range)

THUMBS = ROOT / "data" / "views" / "thumbs"
SAMPLES = ROOT / "data" / "samples"
SYNTH = ROOT / "data" / "views" / "2d" / "synthetic_density.json"
MAKKAH = timezone(timedelta(hours=3))
FALLBACK_LIVE = datetime(2026, 10, 24, 11, 21, tzinfo=timezone.utc)  # 640 px pass live in the collector
FULL = (1920, 1080)


def load_sample(path: Path) -> np.ndarray:
    """The raw frame of a test sample (left panel under a 40 px bar), else the image itself."""
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    return img[40:, : w // 2] if w > 2 * h and h > 40 else img


def as_thumbnail(img: np.ndarray) -> np.ndarray:
    """What the survey thumbnail of this frame looks like (480x270 JPEG q80)."""
    small = cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA)
    return cv2.imdecode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])[1], cv2.IMREAD_COLOR)


def ranges(cfg: dict) -> dict[str, tuple[float, float]]:
    out = {v["id"]: zoom_range(v) for v in cfg["views"]}
    if SYNTH.exists():  # tested ranges not yet written to the config
        for vid, r in json.loads(SYNTH.read_text())["views"].items():
            if not any(v["id"] == vid and "zoom_min" in v for v in cfg["views"]):
                out[vid] = (r["zoom_min"], r["zoom_max"])
    return out


def decide(matcher: ViewMatcher, img: np.ndarray, cfg: dict, rng: dict, need: int) -> dict:
    up = cv2.resize(img, FULL, interpolation=cv2.INTER_CUBIC) if img.shape[1] != FULL[0] else img
    m = matcher.match(up, min_inliers=25)
    row = {"view": m.view_id if m else "", "inliers": m.inliers if m else 0, "zoom": None, "visible": None,
           "old": False, "new": False, "new2x": False, "why": "no match"}
    if m is None:
        return row
    vis = matcher.visibility(m, up.shape)
    main = max(vis.values(), key=lambda v: v["visible"])
    tol = next(v["zoom_tolerance"] for v in cfg["views"] if v["id"] == m.view_id)
    lo, hi = rng[m.view_id]
    z, f = main["zoom"], main["visible"]
    strong = m.inliers >= need
    row.update(zoom=round(z, 3), visible=round(f, 3))
    row["old"] = strong and f >= FULL_VISIBLE and abs(z - 1) <= tol
    row["new"] = strong and f >= MIN_VISIBLE and lo <= z <= hi
    row["new2x"] = strong and f >= MIN_VISIBLE and lo <= z <= 2.0
    row["why"] = ("counted" if row["new"] else "weak match" if not strong else
                  "zone < 30 % visible" if f < MIN_VISIBLE else "zoom-in beyond range" if z > hi else "zoom-out")
    return row


def pending_matcher(cfg: dict) -> ViewMatcher | None:
    thumbs = {v["id"]: cv2.imread(str(ROOT / v["source_thumb"])) for v in cfg["views"]
              if v.get("status") == "pending" and (ROOT / v.get("source_thumb", "")).is_file()}
    thumbs = {k: v for k, v in thumbs.items() if v is not None}
    return ViewMatcher(thumbs, cfg, work_width=480, min_inliers=30, fallback_width=None) if thumbs else None


def per_hour(p: float) -> tuple[float, float, float]:
    lo, hi = 60 * p, 60 * (1 - (1 - p) ** 4)
    return lo, (lo + hi) / 2, hi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-inliers", type=int, default=MIN_INLIERS, help="strict match on thumbnails (live: 50)")
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()
    cfg = load_config()
    rng = ranges(cfg)
    matcher = ViewMatcher.from_config(cfg)
    pend = pending_matcher(cfg)
    print("zoom ranges: " + ", ".join(f"{k} {a:.3f}..{b:.3f}" for k, (a, b) in rng.items() if k in {r.id for r in matcher.refs}))

    rows = []
    with open(THUMBS / "index.csv", newline="") as f:
        for r in csv.DictReader(f):
            img = cv2.imread(str(THUMBS / r["file"]))
            if img is None:
                continue
            ts = datetime.fromisoformat(r["ts_utc"])
            row = {"file": r["file"], "kind": r["kind"], "ts_utc": r["ts_utc"], "makkah": r["makkah_time"],
                   "period": "night" if r["night"] == "1" else "day",
                   "era": "before" if ts < FALLBACK_LIVE else "after", **decide(matcher, img, cfg, rng, args.min_inliers)}
            if pend and not row["new"]:
                pm = pend.match(img)
                own = any(v.get("source_thumb", "").endswith(r["file"]) for v in cfg["views"])
                if pm and abs(pm.zoom - 1) <= 0.10 and not own:
                    row["pending"] = pm.view_id
            rows.append(row)

    print(f"\nThumbnails: {len(rows)} (strict match = {args.min_inliers} inliers on the thumbnail)")
    for period in ("day", "night"):
        for era in ("before", "after", "all"):
            sel = [r for r in rows if r["period"] == period and r["kind"] == "unmatched" and era in ("all", r["era"])]
            if not sel:
                continue
            n = len(sel)
            k = {rule: sum(r[rule] for r in sel) for rule in ("old", "new", "new2x")}
            pend_n = sum(bool(r.get("pending")) for r in sel)
            est = {rule: per_hour(k[rule] / n) for rule in k}
            print(f"  {period:5s} {era:6s} n={n:3d}  counted old {k['old']:2d}  new {k['new']:2d}  new2x {k['new2x']:2d}"
                  f"  | min/hour new {est['new'][0]:.1f}..{est['new'][2]:.1f} (~{est['new'][1]:.1f}), "
                  f"new2x ~{est['new2x'][1]:.1f}, old ~{est['old'][1]:.1f}"
                  f"  | pending cams matched {pend_n} (~{per_hour(pend_n / n)[1]:.1f} min/h if approved)")
    why = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["view"] and not r["new"]:
            why[r["period"]][r["why"]] += 1
    for period, d in why.items():
        print(f"  matched but not counted ({period}): " + ", ".join(f"{k}: {v}" for k, v in sorted(d.items())))
    per_cam = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["new"]:
            per_cam[r["period"]][r["view"]] += 1
    print("  counted (new) per camera: " + "; ".join(f"{p}: " + ", ".join(f"{v} {n}" for v, n in d.items())
                                                for p, d in per_cam.items()))
    zv = [(r["zoom"], r["visible"]) for r in rows if r["new"]]
    if zv:
        z, v = np.array(zv).T
        print(f"  counted frames: zoom {z.min():.2f}..{z.max():.2f} (median {np.median(z):.2f}), visible "
              f"{100 * v.min():.0f}..{100 * v.max():.0f} % (median {100 * np.median(v):.0f} %), "
              f"fully visible {int((v >= FULL_VISIBLE).sum())} of {len(v)}")

    print("\nSamples (full frames) vs the same frame as a thumbnail:")
    for p in sorted(SAMPLES.glob("*.png")):
        img = load_sample(p)
        full = decide(matcher, img, cfg, rng, MIN_INLIERS)
        th = decide(matcher, as_thumbnail(img), cfg, rng, MIN_INLIERS)
        print(f"  {p.name[:44]:44s} full: {full['view'] or '-':16s} inl {full['inliers']:4d} zoom {full['zoom']} vis "
              f"{full['visible']} -> {'COUNTED' if full['new'] else full['why']}  | as thumbnail: inl {th['inliers']}"
              f" -> {'counted' if th['new'] else th['why']}")
    if args.csv:
        keys = ["file", "kind", "ts_utc", "makkah", "period", "era", "view", "inliers", "zoom", "visible", "old", "new",
                "new2x", "why", "pending"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nper-frame table: {args.csv}")


if __name__ == "__main__":
    main()
