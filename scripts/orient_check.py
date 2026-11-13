"""Which way round is each calibrated camera? (user request 2026-10-27: settle it with the door, the Hijr Ismail and the
Maqam Ibrahim.)

A box fitted to the Kaaba's silhouette cannot tell its sides apart: turned half a turn the silhouette is the same, and
mirrored (x -> -x or y -> -y) it is nearly the same. Each of these four readings g of the world is tested by
projecting g(landmark) with the camera as calibrated: if the image agrees with g, the true camera is "the calibrated
camera after g" (a half-turn is exact; a mirror needs a new fit, scripts/refit_orientation.py).
Landmarks (Kaaba frame of edge/kaaba_geom.py: door wall at +y, Black Stone corner (+L/2, +W/2), Hijr off the -x wall):
  door      the gold curtain (sitara) on the door wall, near the Black Stone corner, ~2-8.5 m up;
  hijr      the white marble semicircle off the -x wall (8.44 m out, half-width 4.95 m), ~1.3 m high;
  maqam     the gold Maqam Ibrahim in front of the door wall, ~(2.63, 16.5) m;
  stone     the Black Stone corner's vertical edge.
Scores per hypothesis (higher = more alike): gold share in the door curtain (only on a wall that faces the camera) and
around the Maqam, brightness of the Hijr wall's band. A sheet per camera (crops, the four hypotheses in four colours)
goes to data/samples/debug_align/ (debug, 2 days).

  .venv/Scripts/python scripts/orient_check.py [--cams cam0_mataf_low ...] [--frames 6]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge import kaaba_geom as kg  # noqa: E402

L, W = kg.KAABA_L, kg.KAABA_W
HYP = {"as calibrated": np.diag([1.0, 1, 1]), "half-turn": np.diag([-1.0, -1, 1]),
       "mirror x": np.diag([-1.0, 1, 1]), "mirror y": np.diag([1.0, -1, 1])}
COLOR = {"as calibrated": (60, 220, 60), "half-turn": (230, 60, 230), "mirror x": (40, 200, 255),
         "mirror y": (255, 160, 40)}
OUT = ROOT / "data" / "samples" / "debug_align"


def door_quad() -> np.ndarray:
    """The door curtain on the door wall (y = +W/2): 3.5 m wide, centre 3.0 m from the wall's centre towards the
    Black Stone corner, 2-8.5 m up."""
    x0, x1, y = 3.0 - 1.75, 3.0 + 1.75, W / 2
    return np.array([[x0, y, 2.0], [x1, y, 2.0], [x1, y, 8.5], [x0, y, 8.5]])


def hijr_curve(z: float, n: int = 40) -> np.ndarray:
    """The Hijr wall's line at height z: a half-ellipse off the -x wall, 8.44 m out, half-width 4.95 m."""
    a = np.linspace(-np.pi / 2, np.pi / 2, n)
    return np.column_stack([-L / 2 - 8.44 * np.cos(a), 4.95 * np.sin(a), np.full(n, z)])


def maqam_box() -> np.ndarray:
    c = np.array([2.63, 16.5])
    return np.array([[c[0] + dx, c[1] + dy, z] for z in (0.0, 3.0) for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def gold(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return ((hsv[..., 0] >= 10) & (hsv[..., 0] <= 30) & (hsv[..., 1] >= 80) & (hsv[..., 2] >= 90)).astype(np.float32)


def faces_camera(cam: kg.Camera, g: np.ndarray) -> bool:
    """The (transformed) door wall's outward normal points towards the camera."""
    n = g @ np.array([0, 1.0, 0])
    p = g @ np.array([3.0, W / 2, 5.0])
    return float(n @ (cam.centre - p)) > 0


def poly_mask(shape, pts) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m.astype(bool)


def scores(img: np.ndarray, cam: kg.Camera) -> dict:
    gimg = gold(img)
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32) / 255
    out = {}
    for name, g in HYP.items():
        s = {}
        uv, z = cam.project(door_quad() @ g.T)
        m = poly_mask(img.shape, uv) if (z > 0).all() else None
        s["door"] = round(float(gimg[m].mean()), 3) if m is not None and m.sum() > 30 and faces_camera(cam, g) else None
        uv, z = cam.project(maqam_box() @ g.T)
        if (z > 0).all():
            hull = cv2.convexHull(np.round(uv).astype(np.int32))
            mm = np.zeros(img.shape[:2], np.uint8)
            cv2.fillPoly(mm, [hull], 1)
            s["maqam"] = round(float(gimg[mm.astype(bool)].mean()), 3) if mm.sum() > 30 else None
        top, _ = cam.project(hijr_curve(1.3) @ g.T)
        bot, _ = cam.project(hijr_curve(0.2) @ g.T)
        band = np.vstack([top, bot[::-1]])
        hm = poly_mask(img.shape, band)
        s["hijr"] = round(float(v[hm].mean()), 3) if hm.sum() > 30 else None
        out[name] = s
    return out


def draw(img: np.ndarray, cam: kg.Camera) -> np.ndarray:
    out = img.copy()
    for name, g in HYP.items():
        c = COLOR[name]
        uv, z = cam.project(door_quad() @ g.T)
        if (z > 0).all():
            cv2.polylines(out, [np.round(uv).astype(np.int32)], True, c, 2 if faces_camera(cam, g) else 1)
        for zz in (0.2, 1.3):
            uv, z = cam.project(hijr_curve(zz) @ g.T)
            cv2.polylines(out, [np.round(uv).astype(np.int32)], False, c, 2)
        uv, z = cam.project(maqam_box() @ g.T)
        if (z > 0).all():
            x, y = uv.mean(0)
            cv2.circle(out, (int(x), int(y)), 14, c, 2)
        uv, _ = cam.project(np.array([[L / 2, W / 2, 0.25], [L / 2, W / 2, 14.25]]) @ g.T)
        cv2.line(out, tuple(np.round(uv[0]).astype(int)), tuple(np.round(uv[1]).astype(int)), c, 1)
    y = 30
    for name, c in COLOR.items():
        cv2.putText(out, name, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
        y += 30
    return out


def crop_kaaba(img: np.ndarray, cam: kg.Camera, pad: float = 2.2) -> np.ndarray:
    uv, _ = cam.project(kg.box_corners())
    c = uv.mean(0)
    r = pad * max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1]))
    h, w = img.shape[:2]
    x0, x1 = int(max(0, c[0] - r)), int(min(w, c[0] + r))
    y0, y1 = int(max(0, c[1] - 0.7 * r)), int(min(h, c[1] + 0.7 * r))
    return img[y0:y1, x0:x1]


def frames_for(cam_id: str, n: int) -> list[tuple[str, np.ndarray, kg.Camera]]:
    reg = json.loads((ROOT / "config" / "cameras.json").read_text())["cameras"][cam_id]
    views = {v["id"]: v for v in json.loads((ROOT / "config" / "views.json").read_text())["views"]}
    ref = reg.get("ref_image") or views.get(cam_id, {}).get("ref_image")
    out = []
    if ref and (ROOT / ref).exists():
        out.append(("reference", cv2.imread(str(ROOT / ref)),
                    kg.Camera(reg["f"], np.asarray(reg["rvec"], float), np.asarray(reg["tvec"], float), tuple(reg["size"]))))
    side = []
    for f in sorted(glob.glob(str(ROOT / "data" / "samples" / "dvr" / "*.json"))):
        try:
            m = json.loads(Path(f).read_text())
        except (OSError, ValueError):
            continue
        if m.get("view") == cam_id and m.get("cam"):
            side.append((f, m["cam"]))
    for f, c in side[:: max(1, len(side) // max(n - 1, 1))][: n - 1]:
        out.append((Path(f).stem, cv2.imread(str(Path(f).with_suffix(".jpg"))),
                    kg.Camera(c["f"], np.asarray(c["rvec"], float), np.asarray(c["tvec"], float), tuple(c["size"]))))
    return [o for o in out if o[1] is not None]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cams", nargs="*", default=["cam0_mataf_low", "cam1_mataf_wide", "cam5_roof", "cam8_roof_b"])
    ap.add_argument("--frames", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for cid in args.cams:
        rows, tiles = [], []
        for name, img, cam in frames_for(cid, args.frames):
            s = scores(img, cam)
            rows.append(s)
            print(cid, name, json.dumps(s))
            t = crop_kaaba(draw(img, cam), cam)
            t = cv2.resize(t, (900, int(900 * t.shape[0] / max(t.shape[1], 1))))
            cv2.putText(t, name, (10, t.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            tiles.append(t)
        for key in ("door", "maqam", "hijr"):
            med = {h: np.median([r[h][key] for r in rows if r[h].get(key) is not None] or [np.nan]) for h in HYP}
            print(f"  {cid} {key:5s} " + "  ".join(f"{h}: {v:.3f}" for h, v in med.items()))
        if tiles:
            wmax = max(t.shape[1] for t in tiles)
            sheet = np.vstack([cv2.copyMakeBorder(t, 0, 6, 0, wmax - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles[:4]])
            cv2.imwrite(str(OUT / f"orient_{cid}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
