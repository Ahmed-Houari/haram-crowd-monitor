"""Ground-truth patches (investigation 2026-10-26): small floor squares of known area on a frame with a known camera
(data/samples/dvr/<frame>.json from scripts/dvr_cameras.py, or an m2 sample's sidecar). For each patch: an enlarged crop
with the square and a 1 m grid drawn on the head plane (to count heads cell by cell; local only, data/samples/patches/),
and the models' counts inside the same square: DM-Count raw, DM-Count on 2x tiles, APGCC points, and the live and
proposed corrections. Counting only: nothing identifies anyone.

  .venv/Scripts/python scripts/patch_truth.py 20261026T151447Z --patch near:3:200:3 --patch far:15:200:3
      name:wall_distance_m:bearing_deg:size_m (bearing of the patch centre around the Kaaba, 0 = +X, counter-clockwise)
  .venv/Scripts/python scripts/patch_truth.py 20261026T151447Z --auto        # one near, one far, on the camera's side
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge import kaaba_geom as kg  # noqa: E402
from edge.m2_metric import HEAD_M, wall_distance  # noqa: E402

OUT = ROOT / "data" / "samples" / "patches"
SOURCES = [ROOT / "data" / "samples" / "dvr", ROOT / "data" / "samples" / "truth" / "frames", ROOT / "data" / "samples"]
A_CURVE = (5.17, 1.96)  # fix A (2026-10-26 investigation): head-size curve from zoom pairs ...
A_FACTOR = {"cam5_roof": 1.0, "cam1_mataf_wide": 0.67, "cam0_mataf_low": 0.66}  # ... and camera factors vs the roof


def find(stem: str) -> tuple[Path, dict]:
    for d in SOURCES:
        for ext in (".jpg", ".png"):
            p = d / f"{stem}{ext}"
            if p.exists() and p.with_suffix(".json").exists():
                return p, json.loads(p.with_suffix(".json").read_text())
    sys.exit(f"{stem}: no frame with a camera sidecar")


def square(cx: float, cy: float, s: float, n: int = 1) -> np.ndarray:
    """Corners (and edge points) of an s x s square on the head plane, axis-aligned to the patch's own frame."""
    h = s / 2
    t = np.linspace(-h, h, 8 * n + 1)
    edge = np.concatenate([np.column_stack([t, np.full_like(t, -h)]), np.column_stack([np.full_like(t, h), t]),
                           np.column_stack([t[::-1], np.full_like(t, h)]), np.column_stack([np.full_like(t, -h), t[::-1]])])
    return edge + [cx, cy]


R_TABLE = np.linspace(0, 90, 9001)


def patch_centre(d, bearing):
    """The point(s) at wall distance d along the ray from the Kaaba's centre at this bearing (degrees); scalars or
    arrays (broadcast)."""
    d, a = np.broadcast_arrays(np.asarray(d, float), np.radians(np.asarray(bearing, float)))
    ca, sa = np.cos(a), np.sin(a)
    out_x, out_y = np.empty(d.shape), np.empty(d.shape)
    for idx in np.ndindex(d.shape):
        w = wall_distance(R_TABLE * ca[idx], R_TABLE * sa[idx])  # non-decreasing along the ray
        r = float(np.interp(d[idx], w, R_TABLE))
        out_x[idx], out_y[idx] = r * ca[idx], r * sa[idx]
    if out_x.shape == ():
        return float(out_x), float(out_y)
    return out_x, out_y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frame")
    ap.add_argument("--patch", action="append", default=[], help="name:wall_m:bearing_deg:size_m")
    ap.add_argument("--auto", action="store_true", help="near (3 m) and far (15 m) patches on the camera's side")
    ap.add_argument("--zoom", type=float, default=0, help="crop enlargement (default: heads ~24 px)")
    ap.add_argument("--no-models", action="store_true")
    args = ap.parse_args()
    path, meta = find(args.frame)
    img = cv2.imread(str(path))
    c = meta["cam"]
    cam = kg.Camera(c["f"], np.asarray(c["rvec"], float), np.asarray(c["tvec"], float), tuple(c["size"]))
    patches = []
    for p in args.patch:
        name, d, b, s = p.split(":")
        patches.append((name, float(d), float(b), float(s)))
    if args.auto:
        b = float(np.degrees(np.arctan2(cam.centre[1], cam.centre[0])))
        patches += [("near", 3.0, b, 3.0), ("far", 15.0, b, 3.0)]
    dens = tiled = pts = None
    if not args.no_models:
        from edge.apgcc_model import PointCounter
        from edge.crowd_model import CrowdModel
        from scripts.accuracy_probe import tiled_density

        model = CrowdModel()
        dens = model.predict(img).density
        tiled = tiled_density(model, img, dens.shape)
        pts = PointCounter(ROOT / "models" / "apgcc" / "SHHA_best.pth").predict(img).points
    from edge.m2_pipeline import g as g_live

    h0, k = A_CURVE
    view = meta.get("view") or meta.get("camera") or ""
    OUT.mkdir(parents=True, exist_ok=True)
    H = cam.plane_to_image(kg.HEAD_Z)
    hh, ww = img.shape[:2]
    for name, d, b, s in patches:
        cx, cy = patch_centre(d, b)
        poly = square(cx, cy, s)
        q = np.column_stack([poly, np.ones(len(poly))]) @ H.T
        uv = q[:, :2] / q[:, 2:3]
        mask = np.zeros((hh, ww), np.uint8)
        cv2.fillPoly(mask, [np.round(uv).astype(np.int32)], 1)
        depth = float(cam.project(np.array([[cx, cy, kg.HEAD_Z]]))[1][0])
        head = cam.f * HEAD_M / depth
        row = {"frame": args.frame, "patch": name, "wall_m": d, "bearing": b, "area_m2": s * s, "head_px": round(head, 1),
               "in_frame": float(mask.sum() > 0 and uv[:, 0].min() >= 0 and uv[:, 1].min() >= 0 and uv[:, 0].max() < ww
                                 and uv[:, 1].max() < hh)}
        if dens is not None:
            dh, dw = dens.shape
            m8 = cv2.resize(mask[: dh * 8, : dw * 8].astype(np.float32), (dw, dh), interpolation=cv2.INTER_AREA)
            raw = float((dens * m8).sum())
            til = float((tiled * m8).sum())
            ap_n = int(sum(mask[int(round(y)), int(round(x))] for x, y in pts if 0 <= x < ww and 0 <= y < hh))
            gl = float(max(g_live(np.array([head]))[0], 0.2))
            gn = float(max(1 - np.exp(-((head / h0) ** k)), 0.2))
            row |= {"dm_raw": round(raw, 1), "dm_tiled2x": round(til, 1), "apgcc": ap_n,
                    "dm_live_corr": round(raw / gl, 1), "g_live": round(gl, 2),
                    "dm_fixA": round(raw / (gn * A_FACTOR.get(view, 1.0)), 1), "g_A": round(gn, 2), "view": view}
        if not row["in_frame"]:
            print(json.dumps(row), flush=True)
            continue
        # the crop: the square's bounding box + margin, enlarged so heads are ~24 px, with a 1 m grid on the head plane
        x0, y0 = np.floor(uv.min(0)).astype(int) - 12
        x1, y1 = np.ceil(uv.max(0)).astype(int) + 12
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, ww), min(y1, hh)
        z = args.zoom or float(np.clip(24 / max(head, 1), 2, 8))
        crop = cv2.resize(img[y0:y1, x0:x1], None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
        over = crop.copy()
        for t in np.arange(-s / 2, s / 2 + 1e-6, 1.0):
            for seg in (np.array([[t, -s / 2], [t, s / 2]]), np.array([[-s / 2, t], [s / 2, t]])):
                line = np.linspace(seg[0], seg[1], 20) + [cx, cy]
                qq = np.column_stack([line, np.ones(len(line))]) @ H.T
                p2 = (qq[:, :2] / qq[:, 2:3] - [x0, y0]) * z
                cv2.polylines(over, [np.round(p2).astype(np.int32)], False, (0, 255, 255) if abs(abs(t) - s / 2) < 1e-6
                              else (0, 200, 255), 1, cv2.LINE_AA)
        crop = cv2.addWeighted(over, 0.7, crop, 0.3, 0)
        out = OUT / f"{args.frame}_{name}.png"
        cv2.imwrite(str(out), crop)
        row["crop"] = str(out.relative_to(ROOT))
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
