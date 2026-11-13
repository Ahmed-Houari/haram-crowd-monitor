"""Refit a calibrated camera's pose so that its sides are right (user request 2026-10-27).

The silhouette fit (scripts/calibrate_cameras.py) matches the Kaaba's outline, but a box of the Kaaba's size seen
from a low angle has several poses with nearly the same outline (which corner is nearest, half-turns, mirrors). Here
the pose is fitted again with the focal length fixed (self-calibration, independent of the Kaaba) from many starting
directions around the Kaaba; every distinct solution is then checked against landmarks on the reference frame:
  mizab   the gold rain spout at the top of the Hijr wall's middle (-x wall), seen as a gold blob at the roof edge;
  door    the gold curtain on the door wall near the Black Stone corner (on a wall that faces the camera);
  hijr    the white semicircle off the -x wall (brightness of its band, weak with a crowd in white).
The segmenter's mask can miss part of the Kaaba (cam0, 2026-10-27: its left 50 px); --crowd-y repairs it with the dark
Kiswa above the crowd line. Landmark pixels read off the reference (--mizab-px, --corner-px with --corner) are scored
as reprojection errors; the chosen pose minimises silhouette rms + 0.2 x those errors. A sheet with the best solutions
(box edges, landmarks) goes to data/samples/debug_align/refit_<camera>.jpg; the chosen pose is written to
config/cameras_refit.json (not live: scripts/apply_orientation.py, only on approval).

  .venv/Scripts/python scripts/refit_orientation.py cam0_mataf_low --crowd-y 480 --az 196 232 --mizab-px 869 215       --corner W --corner-px 990 188        (2026-10-27: camera at (-52, -31, 15) m)
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
from scripts.orient_check import door_quad, faces_camera, gold, hijr_curve, poly_mask  # noqa: E402

L, W = kg.KAABA_L, kg.KAABA_W
MIZAB = np.array([-L / 2 - 1.0, 0.0, 13.9])  # spout tip ~2 m out from the roof edge, slightly below it
OUT = ROOT / "data" / "samples" / "debug_align"
REFIT = ROOT / "config" / "cameras_refit.json"


def mizab_blob(img: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Centre of the gold blob at the Kaaba's top edge (the Mizab), or None."""
    ys, xs = np.nonzero(mask)
    top, bottom = ys.min(), ys.max()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    g = (hsv[..., 0] >= 8) & (hsv[..., 0] <= 32) & (hsv[..., 1] >= 40) & (hsv[..., 2] >= 80)
    band = np.zeros_like(g)
    y0, y1 = max(0, top - 15), top + int(0.2 * (bottom - top))
    band[y0:y1, xs.min():xs.max() + 1] = True
    m = (g & band).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return cen[i] if st[i, cv2.CC_STAT_AREA] >= 15 else None


def landmarks(img: np.ndarray, cam: kg.Camera, blob) -> dict:
    gimg = gold(img)
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32) / 255
    s = {}
    uv, z = cam.project(MIZAB[None])
    s["mizab_px"] = round(float(np.hypot(*(uv[0] - blob))), 1) if blob is not None and z[0] > 0 else None
    s["mizab_facing"] = bool(np.array([-1.0, 0, 0]) @ (cam.centre - np.array([-L / 2, 0, 14])) > 0)
    uv, z = cam.project(door_quad())
    m = poly_mask(img.shape, uv) if (z > 0).all() else None
    s["door_gold"] = round(float(gimg[m].mean()), 3) if m is not None and m.sum() > 30 and faces_camera(cam, np.eye(3)) else None
    top, _ = cam.project(hijr_curve(1.3))
    bot, _ = cam.project(hijr_curve(0.2))
    hm = poly_mask(img.shape, np.vstack([top, bot[::-1]]))
    s["hijr_v"] = round(float(v[hm].mean()), 3) if hm.sum() > 30 else None
    return s


def draw(img: np.ndarray, cam: kg.Camera, color, blob) -> np.ndarray:
    out = img.copy()
    uv, _ = cam.project(kg.box_corners())
    for a, b in kg.BOX_EDGES:
        cv2.line(out, tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[b]).astype(int)), color, 1)
    cv2.putText(out, "BS", tuple(np.round(uv[6]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    uv, z = cam.project(door_quad())
    if (z > 0).all():
        cv2.polylines(out, [np.round(uv).astype(np.int32)], True, (0, 215, 255), 2 if faces_camera(cam, np.eye(3)) else 1)
    for zz in (0.2, 1.3):
        uv, _ = cam.project(hijr_curve(zz))
        cv2.polylines(out, [np.round(uv).astype(np.int32)], False, (255, 255, 255), 2)
    uv, _ = cam.project(MIZAB[None])
    cv2.drawMarker(out, tuple(np.round(uv[0]).astype(int)), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
    if blob is not None:
        cv2.circle(out, tuple(np.round(blob).astype(int)), 12, (255, 255, 0), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cams", nargs="+")
    ap.add_argument("--step", type=float, default=10.0, help="azimuth step of the starts, degrees")
    ap.add_argument("--az", type=float, nargs=2, default=(0.0, 360.0), help="azimuth range of the starts, degrees")
    ap.add_argument("--crowd-y", type=int, help="repair the mask with the dark Kiswa above this image row")
    ap.add_argument("--mizab-px", type=float, nargs=2)
    ap.add_argument("--corner", choices=["W", "S", "E", "N"], help="top corner seen at --corner-px")
    ap.add_argument("--corner-px", type=float, nargs=2)
    args = ap.parse_args()
    import torch

    from edge.kaaba_seg import KaabaSegmenter
    from edge.view_matcher import graphics_mask, load_config

    cfg = load_config()
    reg = json.loads((ROOT / "config" / "cameras.json").read_text())["cameras"]
    views = {v["id"]: v for v in cfg["views"]}
    seg = KaabaSegmenter()
    refit = json.loads(REFIT.read_text()) if REFIT.exists() else {}
    OUT.mkdir(parents=True, exist_ok=True)
    for cid in args.cams:
        r = reg[cid]
        ref = r.get("ref_image") or views[cid]["ref_image"]
        img = cv2.imread(str(ROOT / ref))
        h, w = img.shape[:2]
        old = kg.Camera(r["f"], np.asarray(r["rvec"], float), np.asarray(r["tvec"], float), tuple(r["size"]))
        cands = seg.find(img, graphics_mask(cfg, (w, h)))
        best_mask = None
        for c in cands:
            ft = kg.fit(c.mask, init=[old], top_k=1, fixed_f=r["f"])
            if ft is not None and (best_mask is None or ft.cost < best_mask[1]):
                best_mask = (c.mask, ft.cost)
        if best_mask is None:
            print(cid, "no Kaaba mask")
            continue
        mask = best_mask[0].astype(np.uint8)
        if args.crowd_y:  # the dark Kiswa connected to the mask, above the crowd (dark clothes below)
            v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2]
            dark = ((v < 60) | (cv2.dilate(mask, np.ones((3, 3), np.uint8)) > 0)).astype(np.uint8)
            dark[args.crowd_y:] = 0
            dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            _, lab = cv2.connectedComponents(dark, connectivity=8)
            ys0, xs0 = np.nonzero(mask)
            k = lab[int(np.median(ys0[ys0 < args.crowd_y])), int(np.median(xs0))]
            mask = cv2.morphologyEx(((lab == k) | (mask > 0)).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        blob = np.array(args.mizab_px) if args.mizab_px else mizab_blob(img, mask)
        corner = {"W": (-L / 2, -W / 2), "S": (L / 2, -W / 2), "E": (L / 2, W / 2), "N": (-L / 2, W / 2)}.get(args.corner)
        ys, xs = np.nonzero(mask)
        C = old.centre
        dist = float(np.linalg.norm(C - [0, 0, 7.25]))
        elev = float(np.arcsin((C[2] - 7.25) / dist))
        sols = []
        for az in np.radians(np.arange(args.az[0], args.az[1], args.step)):
            for el in (elev, np.radians(4.0), np.radians(16.0)) if args.crowd_y else (elev,):
                start = kg.look_at_camera(r["f"], az, el, dist, (w, h), target_px=(xs.mean(), ys.mean()))
                ft = kg.fit(mask, init=[start], top_k=1, fixed_f=r["f"])
                if ft is None:
                    continue
                c = ft.cam.centre
                if any(np.linalg.norm(c - s[0].cam.centre) < 1.0 for s in sols):
                    continue
                lm = landmarks(img, ft.cam, blob)
                if corner is not None and args.corner_px:
                    uv, _ = ft.cam.project(np.array([[corner[0], corner[1], kg.Z_TOP]]))
                    lm["corner_px"] = round(float(np.hypot(*(uv[0] - args.corner_px))), 1)
                sols.append((ft, lm))
        score = lambda s: s[0].rms_px + 0.2 * ((s[1].get("mizab_px") or 0) + (s[1].get("corner_px") or 0))
        sols.sort(key=score)
        print(f"{cid}: Mizab blob at {None if blob is None else blob.round().tolist()}; old camera at "
              f"{C.round(1).tolist()}, landmarks {landmarks(img, old, blob)}")
        for ft, lm in sols[:8]:
            c = ft.cam.centre
            print(f"  at ({c[0]:7.1f},{c[1]:7.1f},{c[2]:5.1f}) azim {np.degrees(np.arctan2(c[1], c[0])):6.1f}  rms {ft.rms_px:5.2f}"
                  f"  iou {ft.iou:.3f}  {lm}")
        tiles = [draw(img, old, (0, 255, 0), blob)] + [draw(img, ft.cam, (255, 0, 255), blob) for ft, _ in sols[:5]]
        x0, x1 = max(0, xs.min() - 300), min(w, xs.max() + 300)
        y0, y1 = max(0, ys.min() - 120), min(h, ys.max() + 220)
        crops = [t[y0:y1, x0:x1] for t in tiles]
        for i, cimg in enumerate(crops):
            label = "old" if i == 0 else f"sol {i}: rms {sols[i - 1][0].rms_px:.2f}"
            cv2.putText(cimg, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        rows = [np.hstack(crops[i:i + 2]) if i + 1 < len(crops) else np.hstack([crops[i], np.zeros_like(crops[i])])
                for i in range(0, len(crops), 2)]
        cv2.imwrite(str(OUT / f"refit_{cid}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
        ft, lm = sols[0]
        C2 = ft.cam.centre
        chosen = dict(r)
        chosen.update({"rvec": [round(float(x), 7) for x in ft.cam.rvec], "tvec": [round(float(x), 4) for x in ft.cam.tvec],
                       "centre_m": [round(float(x), 2) for x in C2], "distance_m": round(float(np.hypot(C2[0], C2[1])), 1),
                       "rms_px": round(ft.rms_px, 2), "iou": round(ft.iou, 3),
                       "orientation_fix": "pose fitted again (focal length fixed) from starts around the Kaaba; chosen by "
                                          f"silhouette and landmarks {lm}",
                       "previous": {k: r[k] for k in ("rvec", "tvec", "centre_m")}})
        refit[cid] = {"chosen": chosen, "solutions": [{"centre_m": [round(float(x), 2) for x in ft.cam.centre], "rvec": [float(x) for x in ft.cam.rvec],
                                     "tvec": [float(x) for x in ft.cam.tvec], "f": ft.cam.f, "rms_px": round(ft.rms_px, 2),
                                     "iou": round(ft.iou, 3), **lm} for ft, lm in sols[:8]],
                      "mizab_blob": None if blob is None else [float(x) for x in blob], "ref_image": ref}
        REFIT.write_text(json.dumps(refit, indent=1) + "\n")


if __name__ == "__main__":
    main()
