"""Ground truth from prayer rows (investigation 2026-10-26): during prayer the Mataf fills with rows around the Kaaba,
so people per m^2 = (people per metre along a row) / (row spacing in metres). On a frame with a known camera, a floor
patch is resampled top-down (head plane, uniform metres) and the spacing of rows (across) and of people (along) is
read from its autocorrelation; the top-down patch is also saved, enlarged, to count by eye (data/samples/patches/).

  .venv/Scripts/python scripts/rows_truth.py 20261026T150602Z --at 6:200 --at 15:250
      wall_distance_m:bearing_deg = the patch centre; --size 6 (m), --res 0.03 (m per px)
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
from edge.m2_metric import wall_distance  # noqa: E402
from scripts.patch_truth import OUT, find, patch_centre  # noqa: E402


def topdown(img: np.ndarray, cam: kg.Camera, cx: float, cy: float, size: float, res: float, z: float = kg.HEAD_Z):
    """(patch, across, along): the floor around (cx, cy) resampled at res m/px; rows 'across' = away from the Kaaba
    (the gradient of the wall distance), columns 'along' the row."""
    e = 0.05
    gx = (wall_distance(np.array([cx + e]), np.array([cy]))[0] - wall_distance(np.array([cx - e]), np.array([cy]))[0])
    gy = (wall_distance(np.array([cx]), np.array([cy + e]))[0] - wall_distance(np.array([cx]), np.array([cy - e]))[0])
    across = np.array([gx, gy]) / max(np.hypot(gx, gy), 1e-9)
    along = np.array([-across[1], across[0]])
    n = int(round(size / res))
    s = (np.arange(n) - (n - 1) / 2) * res
    A, B = np.meshgrid(s, s, indexing="ij")  # row index = across, column = along
    X = cx + A * across[0] + B * along[0]
    Y = cy + A * across[1] + B * along[1]
    H = cam.plane_to_image(z)
    q = np.stack([X, Y, np.ones_like(X)], -1) @ H.T
    u = (q[..., 0] / q[..., 2]).astype(np.float32)
    v = (q[..., 1] / q[..., 2]).astype(np.float32)
    patch = cv2.remap(img, u, v, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return patch, u, v


def period(profile: np.ndarray, res: float, lo: float, hi: float) -> tuple[float, float]:
    """(period m, strength): the strongest autocorrelation peak of a detrended profile between lo and hi metres."""
    p = profile - cv2.GaussianBlur(profile.reshape(-1, 1).astype(np.float64), (0, 0), max(1.5 / res, 1)).ravel()
    p = (p - p.mean()) / (p.std() + 1e-9)
    ac = np.correlate(p, p, "full")[len(p) - 1:] / len(p)
    i0, i1 = int(lo / res), min(int(hi / res), len(ac) - 2)
    if i1 <= i0 + 2:
        return float("nan"), 0.0
    k = i0 + int(np.argmax(ac[i0:i1]))
    # parabolic refinement
    if 0 < k < len(ac) - 1:
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        den = a - 2 * b + c
        k = k + (float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if den < 0 else 0.0)
    return float(k * res), float(ac[int(round(k))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frame")
    ap.add_argument("--at", action="append", required=True, help="wall_m:bearing_deg")
    ap.add_argument("--size", type=float, default=6.0)
    ap.add_argument("--res", type=float, default=0.03)
    ap.add_argument("--z", type=float, default=kg.HEAD_Z, help="plane height (m): 1.55 standing, ~0.9 seated")
    ap.add_argument("--show", type=float, default=4.0, help="enlargement of the saved top-down patch")
    args = ap.parse_args()
    path, meta = find(args.frame)
    img = cv2.imread(str(path))
    c = meta["cam"]
    cam = kg.Camera(c["f"], np.asarray(c["rvec"], float), np.asarray(c["tvec"], float), tuple(c["size"]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    for at in args.at:
        d, b = map(float, at.split(":"))
        cx, cy = patch_centre(d, b)
        patch, u, v = topdown(gray, cam, cx, cy, args.size, args.res, args.z)
        inside = (u >= 0) & (v >= 0) & (u < img.shape[1]) & (v < img.shape[0])
        # source px per output px (how much real detail the top-down patch has)
        du = np.hypot(np.gradient(u, axis=1), np.gradient(v, axis=1)).mean()
        dv = np.hypot(np.gradient(u, axis=0), np.gradient(v, axis=0)).mean()
        row_pitch, s1 = period(patch.mean(1), args.res, 0.6, 2.0)  # across: rows 0.6-2 m apart
        # along: per row band, then the median (people 0.35-1.0 m apart)
        alongs = [period(patch[i], args.res, 0.35, 1.0) for i in range(0, patch.shape[0], max(1, int(0.15 / args.res)))]
        good = [p for p, s in alongs if s > 0.15 and np.isfinite(p)]
        person_pitch = float(np.median(good)) if good else float("nan")
        dens = 1.0 / (row_pitch * person_pitch) if good and np.isfinite(row_pitch) else float("nan")
        name = f"{args.frame}_rows_{int(d)}m_{int(b)}"
        show = cv2.resize(cv2.remap(img, u, v, cv2.INTER_CUBIC), None, fx=args.show, fy=args.show, interpolation=cv2.INTER_CUBIC)
        for t in np.arange(0, args.size + 1e-6, 1.0):  # 1 m grid
            k = int(round(t / args.res * args.show))
            cv2.line(show, (k, 0), (k, show.shape[0]), (0, 200, 255), 1)
            cv2.line(show, (0, k), (show.shape[1], k), (0, 200, 255), 1)
        cv2.imwrite(str(OUT / f"{name}.png"), show)
        print(json.dumps({"frame": args.frame, "wall_m": d, "bearing": b, "size_m": args.size, "in_frame": round(float(inside.mean()), 2),
                          "src_px_per_m_along": round(float(du) / args.res, 1), "src_px_per_m_across": round(float(dv) / args.res, 1),
                          "row_pitch_m": round(row_pitch, 3), "row_strength": round(s1, 2), "person_pitch_m": round(person_pitch, 3),
                          "n_bands": len(good), "rows_density_m2": round(dens, 2), "img": f"data/samples/patches/{name}.png"}), flush=True)


if __name__ == "__main__":
    main()
