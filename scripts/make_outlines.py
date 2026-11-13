"""Architecture outlines per reference view, for the dashboard's "What the AI sees" pane.

Edges of the clean reference frame (data/views/refs/<view>.png) become long, simplified polylines, after
every area that can hold people is removed (privacy: outlines of the building only, never of people):
  1. inside the counted zones (dilated), where the crowd is;
  2. inside the broadcast graphics mask;
  3. wherever DM-Count sees any density (people anywhere else in the frame), blurred and dilated;
  4. short or compact edge pieces (people, noise): a line must be at least --min-len px long AND span at least
     --min-extent px in width or height, so small closed blobs (a person on an upper floor) are dropped.
Output: data/views/outlines/<view>.json = {"size": [w, h], "lines": [[[x, y], ...], ...]} in reference-frame
pixels (numbers only, no image). --review also writes data/design/outlines_<view>.png (lines on black) to check.
The crowd model runs on the CPU so the collector's GPU work is not disturbed.

Run:  .venv/Scripts/python scripts/make_outlines.py [--view cam5_roof] [--review]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge.view_matcher import load_config  # noqa: E402

OUT_DIR = ROOT / "data" / "views" / "outlines"
REVIEW_DIR = ROOT / "data" / "design"


def people_mask(img: np.ndarray, model, thresh: float, grow: int) -> np.ndarray:
    """255 where the crowd model sees anyone (density map upsampled, blurred, thresholded, dilated)."""
    h, w = img.shape[:2]
    d = cv2.resize(model.predict(img).density.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    d = cv2.GaussianBlur(d, (0, 0), 6)
    m = (d > thresh).astype(np.uint8) * 255
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))


def poly_mask(shape: tuple, polys: list, grow: int) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.int32(p)], 255)
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow))) if grow else m


def outlines(img: np.ndarray, blocked: np.ndarray, min_len: float, min_extent: float,
             eps: float) -> list[list[list[float]]]:
    """Long edge polylines outside `blocked`."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 9, 40, 40)
    med = float(np.median(smooth))
    edges = cv2.Canny(smooth, int(max(10, 0.6 * med)), int(min(255, 1.3 * med)))
    edges[blocked > 0] = 0
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    lines = []
    for c in contours:
        if cv2.arcLength(c, False) < 2 * min_len:  # a 1-px edge is traced there and back
            continue
        _, _, bw, bh = cv2.boundingRect(c)
        if max(bw, bh) < min_extent:
            continue
        lines.append(cv2.approxPolyDP(c, eps, False).reshape(-1, 2).astype(float).round(1).tolist())
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--view", help="one view id (default: all with a reference frame)")
    ap.add_argument("--min-len", type=float, default=70, help="shortest line kept, in reference px")
    ap.add_argument("--min-extent", type=float, default=80, help="smallest width or height of a kept line (px)")
    ap.add_argument("--eps", type=float, default=1.2, help="polyline simplification tolerance (px)")
    ap.add_argument("--density-thresh", type=float, default=0.004, help="density (per map cell) that counts as people")
    ap.add_argument("--review", action="store_true", help="also write data/design/outlines_<view>.png (lines only)")
    args = ap.parse_args()

    from edge.crowd_model import CrowdModel

    cfg = load_config()
    model = CrowdModel()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for v in cfg["views"]:
        if args.view and v["id"] != args.view:
            continue
        ref = ROOT / v["ref_image"]
        if not ref.exists():
            print(f"{v['id']}: no reference frame yet, skipped")
            continue
        img = cv2.imread(str(ref))
        blocked = (poly_mask(img.shape, list(v["zones"].values()), 25)
                   | poly_mask(img.shape, cfg.get("graphics_mask", []), 12)
                   | people_mask(img, model, args.density_thresh, 31))
        lines = outlines(img, blocked, args.min_len, args.min_extent, args.eps)
        h, w = img.shape[:2]
        out = OUT_DIR / f"{v['id']}.json"
        out.write_text(json.dumps({"view_id": v["id"], "size": [w, h], "lines": lines,
                                   "made_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                   "params": {"min_len": args.min_len, "min_extent": args.min_extent, "eps": args.eps,
                                              "density_thresh": args.density_thresh}}))
        print(f"{v['id']}: {len(lines)} lines, {sum(len(l) for l in lines)} points, "
              f"{100 * (blocked > 0).mean():.0f} % of the frame blocked -> {out.relative_to(ROOT)}")
        if args.review:
            REVIEW_DIR.mkdir(parents=True, exist_ok=True)
            canvas = np.zeros((h, w, 3), np.uint8)
            canvas[blocked > 0] = (40, 22, 22)  # blocked areas in dark blue, to see what was removed
            for line in lines:
                cv2.polylines(canvas, [np.int32(line)], False, (220, 230, 236), 1, cv2.LINE_AA)
            cv2.imwrite(str(REVIEW_DIR / f"outlines_{v['id']}.png"), canvas)
        del img


if __name__ == "__main__":
    main()
