"""Refit the calibrated cameras' positions on frames where the Kaaba's base is visible (user request 2026-10-27, offline:
nothing live changes). The poses in use put the Kaaba's base 1-2 m of wall too low on the low and wide cameras while
its top and sides match (the roof cameras fit well): the silhouette fit lets people in front hide the base (z_occ,
pulled towards head level), so the camera position was never pinned by the base. Here the base is part of the fit.

Per frame (Kaaba mask from edge/kaaba_seg.py, the frame's stored pose as the start):
  fixed    the camera at its calibrated position (config/cameras.json centre_m), rotation and focal length fitted,
           base visible (how well the position in use can match the whole Kaaba);
  free     position, rotation and focal length fitted, base visible;
  height   as free, with the Kaaba's height free as well (the published height: the 14 m Kiswa above a 0.25 m base).
Per camera: the median free position (and its spread). Output: data/samples/debug_align/refit_positions.json and a
sheet refit_positions.jpg (old box red, refitted green).

  .venv/Scripts/python scripts/refit_positions.py [--frames WINDOW:STEM ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from edge import kaaba_geom as kg  # noqa: E402

DVR = ROOT / "data" / "samples" / "dvr"
FT = ROOT / "data" / "finetune"
OUT = ROOT / "data" / "samples" / "debug_align"
# frames with the Kaaba's base in view (checked by eye 2026-10-27: prayer rows start a few metres out)
FRAMES = ["2026-10-27-asr:20261027T123442Z", "2026-10-27-asr:20261027T123512Z", "2026-10-27-asr:20261027T123232Z",
          "2026-10-27-asr:20261027T123902Z", "2026-10-27-asr:20261027T123822Z", "2026-10-27-asr:20261027T123832Z",
          "2026-10-27-asr:20261027T123602Z", "2026-10-27-asr:20261027T123622Z", "2026-10-26-maghrib:20261026T150532Z",
          "2026-10-26-maghrib:20261026T150657Z", "2026-10-26-maghrib:20261026T150727Z"]


def cam_of(c: dict) -> kg.Camera:
    return kg.Camera(c["f"], np.asarray(c["rvec"], float), np.asarray(c["tvec"], float), tuple(c["size"]))


def sil_mask(cam: kg.Camera, z0: float = kg.Z_BOTTOM, z1: float = kg.Z_TOP) -> np.ndarray:
    m = np.zeros((1080, 1920), np.uint8)
    s = kg.silhouette(cam, kg.box_corners(z0=z0, z1=z1))
    if s is not None:
        cv2.fillPoly(m, [np.round(s[0]).astype(np.int32)], 1)
    return m


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames", nargs="*", default=FRAMES)
    args = ap.parse_args()
    from edge.kaaba_seg import KaabaSegmenter
    from edge.view_matcher import graphics_mask, load_config

    cams = json.loads((ROOT / "config" / "cameras.json").read_text())["cameras"]
    gm = graphics_mask(load_config(), (1920, 1080))
    seg = KaabaSegmenter()
    rows, tiles = [], []
    for item in args.frames:
        win, stem = item.split(":")
        meta = next(f for f in json.loads((FT / f"{win}.json").read_text())["frames"] if f["stem"] == stem)
        view, old = meta["camera"], cam_of(meta["cam"])
        img = cv2.imread(str(DVR / f"{stem}.jpg"))
        old_sil = sil_mask(old)
        cands = seg.find(img, gm, k=2)
        if not cands:
            print(f"{stem} {view}: no Kaaba mask", flush=True)
            continue
        km = max(cands, key=lambda c: iou(c.mask, old_sil)).mask
        out = {"stem": stem, "camera": view, "how": meta["how"], "old": {"centre": np.round(old.centre, 2).tolist(),
               "f": round(old.f, 1), "iou_full_box": round(iou(km, old_sil), 3)}}
        kg_z = kg.Z_OCC0
        kg.Z_OCC0 = kg.Z_BOTTOM  # the base is visible: the outline's bottom is the base, not a crowd line
        try:
            fx = kg.fit_from_centre(km, np.asarray(cams[view]["centre_m"], float), init=old, budget_s=20)
            fr = kg.fit(km, init=[old])
            fh = kg.fit(km, init=[old], free_height=True)
        finally:
            kg.Z_OCC0 = kg_z
        for name, ft in (("fixed", fx), ("free", fr), ("height", fh)):
            if ft is None:
                continue
            out[name] = {"centre": np.round(ft.cam.centre, 2).tolist(), "f": round(ft.cam.f, 1), "rms_px": round(ft.rms_px, 2),
                         "iou": round(ft.iou, 3), "z_occ": round(ft.z_occ, 2), "z_top": round(ft.z_top, 2),
                         "iou_full_box": round(iou(km, sil_mask(ft.cam, z1=ft.z_top)), 3),
                         "rvec": [round(float(x), 6) for x in ft.cam.rvec], "tvec": [round(float(x), 4) for x in ft.cam.tvec]}
        rows.append(out)
        print(json.dumps({k: (v if k in ("stem", "camera") else {kk: vv for kk, vv in v.items() if kk not in ("rvec", "tvec")}
                              if isinstance(v, dict) else v) for k, v in out.items()}), flush=True)
        vis = img.copy()
        for cam, col in ((old, (0, 0, 255)), (fr.cam if fr else None, (0, 220, 0))):
            if cam is not None:
                s = kg.silhouette(cam)
                cv2.polylines(vis, [np.round(s[0]).astype(np.int32).reshape(-1, 1, 2)], True, col, 2)
        ys, xs = np.nonzero(km)
        y0, y1, x0, x1 = max(0, ys.min() - 80), min(1080, ys.max() + 80), max(0, xs.min() - 80), min(1920, xs.max() + 80)
        t = cv2.resize(vis[y0:y1, x0:x1], (480, 360))
        cv2.putText(t, f"{stem[9:15]} {view}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        tiles.append(t)
    summary = {}
    for view in sorted({r["camera"] for r in rows}):
        cs = np.array([r["free"]["centre"] for r in rows if r["camera"] == view and "free" in r])
        hs = [r["height"]["z_top"] for r in rows if r["camera"] == view and "height" in r]
        if len(cs):
            summary[view] = {"old_centre": cams[view]["centre_m"], "free_centre_median": np.round(np.median(cs, 0), 2).tolist(),
                             "free_centre_spread": np.round(cs.std(0), 2).tolist(), "frames": len(cs),
                             "fitted_kaaba_top": [round(float(x), 2) for x in hs]}
    (OUT / "refit_positions.json").write_text(json.dumps({"frames": rows, "cameras": summary}, indent=1))
    if tiles:
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(OUT / "refit_positions.jpg"), np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
