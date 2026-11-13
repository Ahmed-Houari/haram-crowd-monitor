"""Focal length of a calibrated camera chosen by the Kaaba check (research, 2026-10-24; read-only unless --write).

scripts/calibrate_cameras.py takes each camera's reference focal length from rotation consistency alone (rotating-
camera self-calibration). For the roof camera that left the projected Kaaba up to a Kaaba-width beside the real one
on many matched frames (replay 2026-10-24), although those homographies passed the rotation check. Here the focal
length is chosen by what the metric needs: for each candidate f the camera is fitted to the Kaaba on its reference
frame with that f (pose only), every saved thumbnail matched to the reference (ALIKED + LightGlue) gets its camera from
the homography, and the projected Kaaba is compared with the Kaaba's mask on that thumbnail (OWLv2 + SAM 2.1,
edge/m2_pipeline.kaaba_alignment). The f with the most aligned frames wins (ties: higher mean alignment).
With --write the camera's entry in config/cameras.json is replaced (old values kept under "previous").

  .venv/Scripts/python scripts/kaaba_check_calibration.py --views cam5_roof [--write]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from edge.kaaba_geom import Camera, fit  # noqa: E402
from edge.m2_pipeline import ALIGN_MIN, IN_VIEW_MIN, kaaba_alignment, kaaba_in_view, load_cameras  # noqa: E402
from edge.ptz import WORK, PtzMatcher, camera_from_homography, ptz_ok  # noqa: E402
from edge.view_matcher import graphics_mask, load_config  # noqa: E402

FULL = (1920, 1080)
MIN_INLIERS = 50


def main() -> None:
    from edge.kaaba_seg import KaabaSegmenter

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--views", nargs="+", required=True)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    gm = graphics_mask(cfg, FULL)
    every = load_cameras(cfg)
    cams = {k: v for k, v in every.items() if k in args.views}
    seg = KaabaSegmenter()
    pm = PtzMatcher(every, graphics_mask(cfg, WORK))  # all cameras: a frame goes to the one it matches best
    frames: dict[str, list] = {k: [] for k in cams}  # (H, Kaaba mask)
    for r in csv.DictReader(open(ROOT / "data" / "views" / "thumbs" / "index.csv")):
        th = cv2.imread(str(ROOT / "data" / "views" / "thumbs" / r["file"]))
        if th is None:
            continue
        img = cv2.resize(th, FULL, interpolation=cv2.INTER_CUBIC)
        m = pm.match(img)
        if m is None or m.inliers < MIN_INLIERS or m.view_id not in cams:
            continue
        cands = seg.find(img, gm, k=2)
        frames[m.view_id].append((m.H, [c.mask for c in cands]))
    reg = json.loads((ROOT / "config" / "cameras.json").read_text())
    for vid, (img, cam_now) in cams.items():
        ref_mask = max(seg.find(img, graphics_mask(cfg, (img.shape[1], img.shape[0]))), key=lambda c: c.score).mask
        rows = []
        for f in np.geomspace(cam_now.f * 0.6, cam_now.f * 1.8, 25):
            r = fit(ref_mask, fixed_f=float(f))
            if r is None:
                continue
            n_ok = n_checked = n_rot = 0
            scores = []
            for H, masks in frames[vid]:
                fc, err = camera_from_homography(H, r.cam, FULL)
                if fc is None or not ptz_ok(MIN_INLIERS, err):
                    continue
                n_rot += 1
                if not masks or kaaba_in_view(fc, (FULL[1], FULL[0]), gm) < IN_VIEW_MIN:
                    continue
                n_checked += 1
                s = max(min(rec / ALIGN_MIN[0], prec / ALIGN_MIN[1]) for rec, prec in
                        (kaaba_alignment(fc, mk) for mk in masks))
                scores.append(s)
                n_ok += s >= 1
            rows.append((n_ok, float(np.mean(scores)) if scores else 0.0, float(f), r, n_checked, n_rot))
            print(f"{vid} f {f:7.0f}: aligned {n_ok:3d} of {n_checked:3d} checked ({n_rot} pass the rotation check), "
                  f"mean score {rows[-1][1]:.2f}, reference fit rms {r.rms_px:.2f} px, camera at "
                  f"{r.cam.centre.round(1).tolist()}")
        best = max(rows, key=lambda x: (x[0], x[1]))
        n_ok, score, f, r, n_checked, n_rot = best
        now = next(x for x in rows if abs(np.log(x[2] / cam_now.f)) < 0.03) if any(
            abs(np.log(x[2] / cam_now.f)) < 0.03 for x in rows) else None
        print(f"{vid}: best f {f:.0f} ({n_ok} of {n_checked} aligned, mean score {score:.2f}); "
              f"current f {cam_now.f:.0f}" + (f" ({now[0]} of {now[4]} aligned)" if now else ""))
        if args.write:
            old = reg["cameras"][vid]
            C = r.cam.centre
            reg["cameras"][vid] = {
                "f": round(r.cam.f, 1), "rvec": [round(float(x), 7) for x in r.cam.rvec],
                "tvec": [round(float(x), 4) for x in r.cam.tvec], "size": list(r.cam.size),
                "centre_m": [round(float(x), 2) for x in C], "distance_m": round(float(np.hypot(C[0], C[1])), 1),
                "f_kaaba_fit": old.get("f_kaaba_fit"), "f_kaaba_range": old.get("f_kaaba_range"),
                "f_selfcal": old.get("f_selfcal"), "selfcal_frames": old.get("selfcal_frames"),
                "selfcal_ptz_err": old.get("selfcal_ptz_err"),
                "f_kaaba_check": round(f, 1), "kaaba_check_aligned": [n_ok, n_checked],
                "rms_px": round(r.rms_px, 2), "iou": round(r.iou, 3),
                "calibrated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "method": "Kaaba silhouette fit; focal length chosen by the Kaaba check on the matched thumbnails "
                          "(scripts/kaaba_check_calibration.py)",
                "previous": {k: old[k] for k in ("f", "rvec", "tvec", "centre_m", "method") if k in old},
            }
    if args.write:
        (ROOT / "config" / "cameras.json").write_text(json.dumps(reg, indent=1) + "\n")
        print("written config/cameras.json")


if __name__ == "__main__":
    main()
