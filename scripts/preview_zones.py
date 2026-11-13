"""Render one preview image per reference view: zones (green), graphics mask (red), % grid,
cyan safe box (10 % margin). Uses the full-res reference if captured, else the survey
thumbnail draft. Also checks each zone: graphics overlap and inside the safe margin.

Output: data/views/preview_<view_id>.jpg
Run from the project root:  .venv/Scripts/python scripts/preview_zones.py [--finalize]

--finalize: for views whose full-res reference is captured, move the thumbnail-drafted zones
onto it (sidecar homography), clip them to the safe area and store them in reference px.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from draw_zones import SAFE_MARGIN, draw_overlay  # noqa: E402
from edge.view_matcher import (PROJECT_ROOT, finalize_zones, graphics_mask, load_config,  # noqa: E402
                               save_config, zones_in_ref_px)

OUT_DIR = PROJECT_ROOT / "data" / "views"
WIDTH = 1280


def main() -> None:
    parser = argparse.ArgumentParser(description="Render zone previews per reference view.")
    parser.add_argument("--finalize", action="store_true",
                        help="move thumbnail-drafted zones onto captured references, clip to the safe area, save config")
    args = parser.parse_args()
    config = load_config()
    if args.finalize:
        done = [v["id"] for v in config["views"] if v.get("zones_base") == "thumb" and finalize_zones(v, config)]
        if done:
            save_config(config)
            print(f"Finalized zones on the full-res reference: {', '.join(done)}")
    for view in config["views"]:
        ref = cv2.imread(str(PROJECT_ROOT / view["ref_image"]))
        if ref is not None:
            source, zones = "full-res reference", zones_in_ref_px(view, config)
        else:
            thumb = cv2.imread(str(PROJECT_ROOT / view["source_thumb"]))
            ref = cv2.resize(thumb, tuple(config["ref_size"]), interpolation=cv2.INTER_CUBIC)
            source, zones = f"DRAFT on survey thumbnail {view['survey_cluster']}", view["zones"]
        h, w = ref.shape[:2]
        gmask = graphics_mask(config, (w, h))
        notes = []
        for name, poly in zones.items():
            zmask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(zmask, [np.int32(poly)], 1)
            overlap = (zmask & gmask).sum() / max(zmask.sum(), 1)
            p = np.array(poly)
            inside = (p[:, 0].min() >= SAFE_MARGIN * w - 1 and p[:, 0].max() <= (1 - SAFE_MARGIN) * w + 1
                      and p[:, 1].min() >= SAFE_MARGIN * h - 1 and p[:, 1].max() <= (1 - SAFE_MARGIN) * h + 1)
            notes.append(f"{name}: {100 * zmask.sum() / (w * h):.0f}% of frame, graphics overlap {100 * overlap:.1f}%, "
                         f"{'inside' if inside else 'OUTSIDE'} safe margin")
        img = draw_overlay(ref, config["graphics_mask"], zones, lw=3, vertex_numbers=True)
        img = cv2.resize(img, (WIDTH, round(h * WIDTH / w)), interpolation=cv2.INTER_AREA)
        bar = np.full((34 + 24 * len(notes), WIDTH, 3), 40, np.uint8)
        cv2.putText(bar, f"{view['id']}  ({source})", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)
        for k, note in enumerate(notes):
            cv2.putText(bar, note, (8, 52 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        out = OUT_DIR / f"preview_{view['id']}.jpg"
        cv2.imwrite(str(out), np.vstack([bar, img]), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{out}  |  " + "  |  ".join(notes))


if __name__ == "__main__":
    main()
