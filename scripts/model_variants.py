"""Counting-model variants on frames with a known camera (investigation 2026-10-26, Maghrib accuracy): people per m^2
on the metric's ring (1-25 m) and its inner ring, raw (no head-size correction), for DM-Count QNRF at 1x and on 2x / 3x
tiles, DM-Count SHA at 1x / 2x tiles, and APGCC points at 1x / 2x. GPU memory capped.

  .venv/Scripts/python scripts/model_variants.py 20261026T150602Z m2_cam1_mataf_wide_20261025T150441Z ...
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
from edge.m2_metric import ALL, RINGS, cell_floor, wall_distance  # noqa: E402
from edge.m2_pipeline import kaaba_exclusion  # noqa: E402
from edge.view_matcher import graphics_mask, load_config  # noqa: E402
from scripts.accuracy_probe import points_to_cells, tiled_density  # noqa: E402
from scripts.patch_truth import find  # noqa: E402


def main() -> None:
    from edge.apgcc_model import PointCounter
    from edge.crowd_model import CrowdModel

    cfg = load_config()
    qnrf, sha = CrowdModel("qnrf"), CrowdModel("sha")
    apg = PointCounter(ROOT / "models" / "apgcc" / "SHHA_best.pth")
    for stem in sys.argv[1:]:
        path, meta = find(stem)
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        c = meta["cam"]
        cam = kg.Camera(c["f"], np.asarray(c["rvec"], float), np.asarray(c["tvec"], float), tuple(c["size"]))
        d = qnrf.predict(img).density
        geo = cell_floor(cam, d.shape)
        ex = np.maximum(kaaba_exclusion(cam, (h, w)), graphics_mask(cfg, (w, h)))
        dh, dw = d.shape
        exc = ex[: dh * 8, : dw * 8].reshape(dh, 8, dw, 8).mean((1, 3)) >= 0.5
        ok = geo["valid"] & ~exc
        dist = wall_distance(geo["X"], geo["Y"])
        maps = {"qnrf_1x": d, "qnrf_tile2x": tiled_density(qnrf, img, d.shape, up=2.0),
                "qnrf_tile3x": tiled_density(qnrf, img, d.shape, grid=(4, 4), up=3.0),
                "sha_1x": sha.predict(img).density}
        maps["sha_tile2x"] = tiled_density(sha, img, d.shape, up=2.0)
        maps["apgcc_1x"] = points_to_cells(apg.predict(img).points, d.shape)
        maps["apgcc_2x"] = points_to_cells(apg.predict(img, scale=2.0).points, d.shape)
        out = {"frame": stem, "camera": meta.get("view") or meta.get("camera"),
               "head_px": round(float(np.median((cam.f * 0.22 / np.maximum(geo["depth"], 1e-3))[ok & (dist >= 1) & (dist < 25)])), 1)}
        for ring, (r0, r1) in {"all": ALL, "inner": RINGS["inner"]}.items():
            sel = ok & (dist >= r0) & (dist < r1)
            a = float(geo["area"][sel].sum())
            out[f"area_{ring}"] = round(a)
            for k, m in maps.items():
                out[f"{k}_{ring}"] = round(float(m[sel].sum()) / max(a, 1e-6), 3)
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
