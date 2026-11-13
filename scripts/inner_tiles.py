"""DM-Count on enlarged tiles over the inner ring only (investigation 2026-10-26, user request): the roof views see
heads of 3-4 px, where the model misses people, most of all where the crowd packs. The part of the frame that shows the
floor 1-10 m from the Kaaba's walls is cut into tiles, each enlarged UP x (heads ~10 px), counted, and the counts put
back on the frame's density grid; the rest of the frame keeps the normal pass. Per frame: the inner and outer rings'
raw people per m^2 at 1x and with the tiles, the corrected readings, and the GPU time of the tiles (synchronised).
Offline only, never in the live services (user request 2026-10-26); full speed, no pause or memory cap (user, 2026-10-28).

  .venv/Scripts/python scripts/inner_tiles.py data/samples/dvr/20261026T1807*.jpg [--up 3]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TILE_SRC = (640, 360)  # source px per tile before enlarging (x UP: 1920 x 1080 at 3x)


def inner_box(cam, shape, margin=24):
    """Frame px box around the floor 0-11 m from the Kaaba's walls (head plane), clipped to the frame."""
    from edge import kaaba_geom as kg
    from edge.m2_metric import wall_distance

    h, w = shape[:2]
    pts = []
    for a in np.linspace(0, 2 * np.pi, 72, endpoint=False):
        r = np.linspace(0, 40, 800)
        X, Y = r * np.cos(a), r * np.sin(a)
        j = int(np.argmin(np.abs(wall_distance(X, Y) - 11.0)))
        pts.append([X[j], Y[j], kg.HEAD_Z])
    uv, z = cam.project(np.array(pts))
    uv = uv[z > 0]
    x0, y0 = np.floor(uv.min(0)).astype(int) - margin
    x1, y1 = np.ceil(uv.max(0)).astype(int) + margin
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def tiled_region(model, img, dshape, box, up):
    """Density grid (dshape) with the box's cells replaced by the enlarged tiles' counts; (grid, ms, passes)."""
    import torch

    x0, y0, x1, y1 = box
    dh, dw = dshape
    out = np.zeros((dh, dw), np.float64)
    nx, ny = max(1, int(np.ceil((x1 - x0) / TILE_SRC[0]))), max(1, int(np.ceil((y1 - y0) / TILE_SRC[1])))
    xs = np.linspace(x0, x1, nx + 1).round().astype(int)
    ys = np.linspace(y0, y1, ny + 1).round().astype(int)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for i in range(ny):
        for j in range(nx):
            tile = img[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            th, tw = tile.shape[:2]
            # every tile padded to TILE_SRC: one input size, so cuDNN (benchmark mode) picks its kernels once, not per size
            pad = np.zeros((TILE_SRC[1], TILE_SRC[0], 3), img.dtype)
            pad[:th, :tw] = tile
            big = cv2.resize(pad, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
            d = model.predict(big).density
            d = d[: round(th / TILE_SRC[1] * d.shape[0]), : round(tw / TILE_SRC[0] * d.shape[1])]
            # the tile's people go back onto the frame's 8 x 8 px cells it covers, spread as its map shows
            c0, c1, r0, r1 = xs[j] // 8, min(xs[j + 1] // 8, dw), ys[i] // 8, min(ys[i + 1] // 8, dh)
            if c1 <= c0 or r1 <= r0:
                continue
            cells = cv2.resize(d.astype(np.float32), (c1 - c0, r1 - r0), interpolation=cv2.INTER_AREA)
            cells *= d.sum() / max(cells.sum(), 1e-9)
            out[r0:r1, c0:c1] += cells
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t) * 1e3, nx * ny


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("frames", nargs="+")
    ap.add_argument("--up", type=float, default=3.0)
    ap.add_argument("--out", type=Path, help="per-frame rows (JSON lines)")
    args = ap.parse_args()
    import torch

    from edge import kaaba_geom as kg
    from edge.crowd_model import CrowdModel
    from edge.m2_metric import cell_floor, wall_distance
    from edge.m2_pipeline import CAMERA_FACTOR, g, kaaba_exclusion
    from edge.view_matcher import graphics_mask, load_config

    cfg = load_config()
    model = CrowdModel()
    files = sorted({f for p in args.frames for f in glob.glob(p)})
    rows = []
    for f in files:
        side = Path(f).with_suffix(".json")
        if not side.exists():
            continue
        m = json.loads(side.read_text())
        if m.get("view") not in ("cam5_roof", "cam8_roof_b"):
            continue
        c = m["cam"]
        cam = kg.Camera(c["f"], np.asarray(c["rvec"], float), np.asarray(c["tvec"], float), tuple(c["size"]))
        img = cv2.imread(f)
        h, w = img.shape[:2]
        torch.cuda.synchronize(); t = time.perf_counter()
        d1 = model.predict(img).density
        torch.cuda.synchronize(); ms1 = (time.perf_counter() - t) * 1e3
        box = inner_box(cam, img.shape)
        dt, ms_t, n = tiled_region(model, img, d1.shape, box, args.up)
        dh, dw = d1.shape
        inbox = np.zeros((dh, dw), bool)
        inbox[box[1] // 8:box[3] // 8, box[0] // 8:box[2] // 8] = True
        d3 = np.where(inbox, dt, d1)
        geo = cell_floor(cam, d1.shape)
        dist = wall_distance(geo["X"], geo["Y"])
        ex = np.maximum(kaaba_exclusion(cam, (h, w)), graphics_mask(cfg, (w, h)))
        ok = geo["valid"] & ~(ex[: dh * 8, : dw * 8].reshape(dh, 8, dw, 8).mean((1, 3)) >= 0.5)
        head = cam.f * 0.22 / np.maximum(geo["depth"], 1e-3)
        q = CAMERA_FACTOR.get(m["view"], 1.0)
        row = {"frame": Path(f).stem, "view": m["view"], "box_px": [int(box[2] - box[0]), int(box[3] - box[1])], "tiles": int(n),
               "ms_1x": round(ms1), "ms_tiles": round(ms_t)}
        for ring, (r0, r1) in (("inner", (1, 10)), ("band_1_4", (1, 4)), ("outer", (10, 25))):
            sel = ok & (dist >= r0) & (dist < r1)
            a = float(geo["area"][sel].sum())
            if a < 10:
                continue
            row[ring] = {"raw_1x": round(float(d1[sel].sum()) / a, 3), "raw_tiles": round(float(d3[sel].sum()) / a, 3),
                         "live_1x": round(float((d1[sel] / (np.maximum(g(head[sel]), 0.2) * q)).sum()) / a, 3),
                         "head": round(float(np.median(head[sel])), 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        if args.out:
            with args.out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
    if rows:
        for ring in ("inner", "band_1_4", "outer"):
            r = [x[ring] for x in rows if ring in x]
            if r:
                gain = np.median([x["raw_tiles"] / max(x["raw_1x"], 1e-6) for x in r])
                print(f"{ring}: {len(r)} frames, raw 1x {np.median([x['raw_1x'] for x in r]):.2f}, tiles {np.median([x['raw_tiles'] for x in r]):.2f} "
                      f"(x{gain:.2f}), live {np.median([x['live_1x'] for x in r]):.2f}")
        print(f"GPU: 1x pass {np.median([x['ms_1x'] for x in rows]):.0f} ms, inner-ring tiles {np.median([x['ms_tiles'] for x in rows]):.0f} ms "
              f"({np.median([x['tiles'] for x in rows]):.0f} tiles)")


if __name__ == "__main__":
    main()
