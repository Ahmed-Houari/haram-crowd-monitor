"""Ground truth vs the metric's setups on floor regions of frames with a known camera (investigation 2026-10-26).
A region: frame, wall-distance range (m), bearing range (deg, around the Kaaba), ground truth people/m^2 (from
scripts/rows_truth.py rows or a head count). Setups, all on the same cells of the head plane:
  before    DM-Count QNRF on the frame, / max(g(h), 0.2) with the curve before 2026-10-26 19:02 (h0 6.5, k 1.2)
  applied   DM-Count QNRF on the frame, / (max(g(h), 0.2) x camera factor) with config/m2.json "correction";
            prayer rows (edge/prayer_rows.py) instead when the frame shows them
  fixA      DM-Count QNRF on the frame, / (max(G(h), 0.2) x camera factor)           (fix A, approved before the GT)
  raw       DM-Count QNRF on the frame, no correction
  tiles     DM-Count QNRF on tiles enlarged so heads reach ~TARGET_HEAD px (1x-3x), no correction
  tilesG    tiles, / max(G(h x enlargement), 0.2) for what is still missed after the enlargement
GPU memory capped.

  .venv/Scripts/python scripts/gt_compare.py 20261026T150602Z:2:22:0:360:1.41 ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge import kaaba_geom as kg  # noqa: E402
from edge.m2_metric import HEAD_M, cell_floor, wall_distance  # noqa: E402
from edge.m2_metric import load_correction  # noqa: E402
from edge.m2_pipeline import kaaba_exclusion  # noqa: E402
from edge import prayer_rows  # noqa: E402
from edge.view_matcher import graphics_mask, load_config  # noqa: E402
from scripts.accuracy_probe import tiled_density  # noqa: E402
from scripts.patch_truth import A_CURVE, A_FACTOR, find  # noqa: E402

TARGET_HEAD = 9.0  # px: DM-Count matches the ground truth at 9-10 px heads (prayer rows, 2026-10-26)


BEFORE = (6.5, 1.2)


def G(h, h0=A_CURVE[0], k=A_CURVE[1]):
    return np.maximum(1 - np.exp(-np.power(np.maximum(h, 0.1) / h0, k)), 0.2)


def enlargement(head_px: float) -> float:
    return float(np.clip(np.ceil(TARGET_HEAD / max(head_px, 1.0)), 1, 3))


def tiles_for(up: float) -> tuple[int, int]:
    return {1.0: (1, 1), 2.0: (3, 3), 3.0: (4, 4)}[up]


def main() -> None:
    from edge.crowd_model import CrowdModel

    cfg = load_config()
    model = CrowdModel("qnrf")
    cache: dict = {}
    for spec in sys.argv[1:]:
        stem, w0, w1, b0, b1, gt = spec.split(":")
        w0, w1, b0, b1, gt = map(float, (w0, w1, b0, b1, gt))
        if stem not in cache:
            path, meta = find(stem)
            img = cv2.imread(str(path))
            h, w = img.shape[:2]
            c = meta["cam"]
            cam = kg.Camera(c["f"], np.asarray(c["rvec"], float), np.asarray(c["tvec"], float), tuple(c["size"]))
            d = model.predict(img).density
            geo = cell_floor(cam, d.shape)
            ex = np.maximum(kaaba_exclusion(cam, (h, w)), graphics_mask(cfg, (w, h)))
            dh, dw = d.shape
            ok = geo["valid"] & ~(ex[: dh * 8, : dw * 8].reshape(dh, 8, dw, 8).mean((1, 3)) >= 0.5)
            dist = wall_distance(geo["X"], geo["Y"])
            head = cam.f * HEAD_M / np.maximum(geo["depth"], 1e-3)
            ring = ok & (dist >= 1) & (dist < 25)
            up = enlargement(float(np.median(head[ring])))
            t = d if up == 1 else tiled_density(model, img, d.shape, grid=tiles_for(up), up=up)
            rows = prayer_rows.measure(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cam, ex)
            cache[stem] = (meta, geo, ok, dist, head, d, t, up, rows)
        meta, geo, ok, dist, head, d, t, up, rows = cache[stem]
        h0, k, factors = load_correction()
        view = meta.get("view") or meta.get("camera") or ""
        bear = np.degrees(np.arctan2(geo["Y"], geo["X"])) % 360
        inb = (bear >= b0) & (bear < b1) if b1 - b0 < 360 else np.ones_like(bear, bool)
        sel = ok & (dist >= w0) & (dist < w1) & inb
        a = float(geo["area"][sel].sum())
        f = lambda m: round(float(m[sel].sum()) / max(a, 1e-6), 3)  # noqa: E731
        q = A_FACTOR.get(view, 1.0)
        res = {"frame": stem, "view": view, "region": f"{w0:g}-{w1:g} m, {b0:g}-{b1:g} deg", "area_m2": round(a),
               "head_px": round(float(np.median(head[sel])), 1), "gt": gt, "enlarge": up,
               "before": f(d / G(head, *BEFORE)), "fixA": f(d / (G(head) * q)), "raw": f(d),
               "tiles": f(t), "tilesG": f(t / G(head * up)),
               "applied": rows.density if rows.density is not None else f(d / (G(head, h0, k) * factors.get(view, 1.0))),
               "rows_strength": rows.strength}
        res |= {f"{k}/gt": round(res[k] / gt, 2) for k in ("before", "fixA", "raw", "tiles", "tilesG", "applied")}
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
