"""Calibrate the known broadcast cameras once (research, 2026-10-24) -> config/cameras.json.

For every approved view with a reference frame:
  1. Kaaba mask on the reference (edge/kaaba_seg.py) and a camera fitted to it (edge/kaaba_geom.py, free focal length);
  2. focal length by self-calibration of the pan / tilt / zoom camera: the saved survey thumbnails matched to that
     reference (ALIKED + LightGlue, >= 100 inliers) must all be rotations + zooms of it (edge/ptz.py); the reference
     focal length minimising the median deviation is kept (it does not depend on the Kaaba at all);
  3. the Kaaba fit again with that focal length (pose only), which fixes the camera position.
The registry holds per camera: focal length, rotation, translation, position (m, Kaaba frame), the two focal length
estimates and the residuals. Reference frames stay where they are (data/views/refs); nothing else is written.

--new ID: a camera only the view-independent metric uses (not a view of config/views.json), its reference frame at
data/views/refs/ID.png (scripts/capture_new_camera.py); its self-calibration also uses the 1080p homographies the
watcher logged (data/views/refs/ID_selfcal.json), and every thumbnail is matched against all calibrated cameras too,
so it counts only for the camera it matches best. The registry entry gets "ref_image".

  .venv/Scripts/python scripts/calibrate_cameras.py [--views cam1_mataf_wide ...] [--new cam8_roof_b]
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
from edge.m2_pipeline import load_cameras  # noqa: E402
from edge.ptz import WORK, PtzMatcher, camera_from_homography  # noqa: E402
from edge.view_matcher import graphics_mask, load_config  # noqa: E402

OUT = ROOT / "config" / "cameras.json"
MIN_SELFCAL_INLIERS = 100


def self_calibrate(ref: Camera, Hs: list[np.ndarray], frame_size: tuple[int, int]) -> tuple[float, float, int]:
    """(focal length, median PTZ error, n) minimising the median deviation of K^-1 H K_ref from a rotation."""
    if len(Hs) < 4:
        return ref.f, float("nan"), len(Hs)
    fs = np.geomspace(ref.f / 3, ref.f * 3, 240)
    med = [np.median([camera_from_homography(H, Camera(f, ref.rvec, ref.tvec, ref.size), frame_size)[1] for H in Hs])
           for f in fs]
    i = int(np.argmin(med))
    return float(fs[i]), float(med[i]), len(Hs)


def main() -> None:
    from edge.kaaba_seg import KaabaSegmenter

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--views", nargs="*")
    ap.add_argument("--new", nargs="*", default=[], help="cameras with a reference at data/views/refs/<id>.png only")
    args = ap.parse_args()
    cfg = load_config()
    if args.new:
        views = [{"id": i, "ref_image": f"data/views/refs/{i}.png"} for i in args.new]
    else:
        views = [v for v in cfg["views"] if v.get("status") != "pending" and (ROOT / v["ref_image"]).exists()
                 and (not args.views or v["id"] in args.views)]
    seg = KaabaSegmenter()
    gm_full = graphics_mask(cfg, (1920, 1080))
    first: dict[str, tuple[np.ndarray, Camera, dict]] = {}
    for v in views:
        img = cv2.imread(str(ROOT / v["ref_image"]))
        h, w = img.shape[:2]
        best = None
        for c in seg.find(img, graphics_mask(cfg, (w, h))):
            r = fit(c.mask)
            if r is not None and (best is None or r.cost < best[1].cost):
                best = (c, r)
        if best is None:
            print(f"{v['id']}: no Kaaba fit, skipped")
            continue
        first[v["id"]] = (img, best[1].cam, {"mask": best[0].mask, "fit": best[1]})
        print(f"{v['id']}: Kaaba fit f {best[1].cam.f:.0f} [{best[1].f_range[0]:.0f}-{best[1].f_range[1]:.0f}] "
              f"rms {best[1].rms_px:.2f} px")
    others = {k: v for k, v in load_cameras(cfg).items() if k not in first} if args.new else {}
    pm = PtzMatcher({**others, **{k: (img, cam) for k, (img, cam, _) in first.items()}}, graphics_mask(cfg, WORK))
    Hs: dict[str, list] = {k: [] for k in first}
    for k in first:  # 1080p homographies logged live (scripts/capture_new_camera.py)
        log_path = ROOT / "data" / "views" / "refs" / f"{k}_selfcal.json"
        if log_path.exists():
            Hs[k] += [np.array(e["H"], np.float64).reshape(3, 3) for e in json.loads(log_path.read_text())]
            print(f"{k}: {len(Hs[k])} live homographies")
    for r in csv.DictReader(open(ROOT / "data" / "views" / "thumbs" / "index.csv")):
        th = cv2.imread(str(ROOT / "data" / "views" / "thumbs" / r["file"]))
        if th is None:
            continue
        m = pm.match(cv2.resize(th, (1920, 1080), interpolation=cv2.INTER_CUBIC))
        if m is not None and m.inliers >= MIN_SELFCAL_INLIERS and m.view_id in Hs:
            Hs[m.view_id].append(m.H)
    reg = json.loads(OUT.read_text()) if OUT.exists() else {"cameras": {}}
    for k, (img, cam0, extra) in first.items():
        f_sc, err, n = self_calibrate(cam0, Hs[k], (1920, 1080))
        r = fit(extra["mask"], fixed_f=f_sc) if np.isfinite(err) else extra["fit"]
        cam = r.cam
        C = cam.centre
        reg["cameras"][k] = {
            "f": round(cam.f, 1), "rvec": [round(float(x), 7) for x in cam.rvec],
            "tvec": [round(float(x), 4) for x in cam.tvec], "size": list(cam.size),
            "centre_m": [round(float(x), 2) for x in C], "distance_m": round(float(np.hypot(C[0], C[1])), 1),
            "f_kaaba_fit": round(cam0.f, 1), "f_kaaba_range": [round(x, 1) for x in extra["fit"].f_range],
            "f_selfcal": round(f_sc, 1), "selfcal_frames": n, "selfcal_ptz_err": round(err, 4) if np.isfinite(err) else None,
            "rms_px": round(r.rms_px, 2), "iou": round(r.iou, 3),
            "calibrated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "method": "Kaaba silhouette fit; focal length from pan/tilt/zoom self-calibration on matched thumbnails",
        }
        if args.new:
            reg["cameras"][k]["ref_image"] = f"data/views/refs/{k}.png"
            reg["cameras"][k]["method"] += " and on live 1080p homographies (scripts/capture_new_camera.py)"
        print(f"{k}: f {cam.f:.0f} (Kaaba fit {cam0.f:.0f}, self-calibration {f_sc:.0f} from {n} frames, median err "
              f"{err:.3f}); camera at {C.round(1).tolist()} m, {np.hypot(C[0], C[1]):.0f} m from the Kaaba")
    reg["note"] = ("Known broadcast cameras (research prototype 2026-10-24): world frame = Kaaba frame of "
                   "edge/kaaba_geom.py (origin on the floor under the Kaaba's centre, z up, m); written by "
                   "scripts/calibrate_cameras.py")
    OUT.write_text(json.dumps(reg, indent=1) + "\n")
    print("written", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
