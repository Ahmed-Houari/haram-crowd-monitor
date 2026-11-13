"""Capture a full-resolution reference of an uncalibrated broadcast camera, found through the Kaaba (2026-10-25).

The view-independent metric (edge/m2_pipeline.py) drops ptz cameras whose projected Kaaba misses the real one. In
the replay of the saved thumbnails about a quarter of the frames matched to the roof camera cam5_roof failed that
check for every focal length, and cameras fitted to their Kaaba sat on cam5's line of sight, closer to the Kaaba:
probably a second roof camera. This watcher runs beside the crowd collector (same broadcast, shared yt-dlp state;
frames in memory) and per frame:
  1. matches the calibrated cameras (edge/ptz.py); a frame whose camera passes the Kaaba check is a known camera;
  2. otherwise segments the Kaaba (OWLv2 + SAM 2.1) and fits a camera to it (edge/kaaba_geom.py, strict gate); a
     candidate is a good fit whose camera lies within --radius m of --near (or its 180-degree twin: the Kaaba looks
     the same from opposite sides) and >= 15 m from every calibrated camera;
  3. keeps the best candidate (the Kaaba whole and large in the frame, off the graphics, tightest fit) as the new
     camera's reference frame: data/views/refs/<id>.png (the place for references) + <id>.json (its numbers);
  4. after --refine-min minutes with a reference, logs the homography of every frame that matches it (>= 100
     inliers, a rotation + zoom of it) to data/views/refs/<id>_selfcal.json (numbers only), for the rotating-camera
     self-calibration of its focal length (scripts/calibrate_cameras.py --new).
Only the reference frame is written as an image.

  .venv/Scripts/python scripts/capture_new_camera.py --id cam8_roof_b --near 107 27 71 [--hours 12]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from edge import kaaba_geom as kg  # noqa: E402
from edge.m2_pipeline import (ALIGN_MIN, IN_VIEW_MIN, good_fit, kaaba_alignment, kaaba_in_view,  # noqa: E402
                              load_cameras, load_extra_refs)
from edge.view_matcher import graphics_mask, load_config  # noqa: E402

REFS = ROOT / "data" / "views" / "refs"
MIN_KAABA_PX = 90  # the Kaaba's mask at least this wide for a reference (a precise fit)
MIN_SELFCAL_INLIERS = 100
log = logging.getLogger("capture_new_camera")


def near(c: np.ndarray, p: np.ndarray) -> float:
    """Distance (m) from camera centre c to p, allowing for the 180-degree twin of c."""
    return float(min(np.linalg.norm(c - p), np.linalg.norm(np.array([-c[0], -c[1], c[2]]) - p)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--id", required=True)
    ap.add_argument("--near", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    ap.add_argument("--radius", type=float, default=45.0)
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--refine-min", type=float, default=30.0, help="keep looking for a better reference this long")
    ap.add_argument("--max-selfcal", type=int, default=60)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "data" / "logs" / "capture_new_camera.log",
                                                      encoding="utf-8"), logging.StreamHandler()])
    for name in ("httpx", "httpx2", "httpcore", "httpcore2", "transformers", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)

    from edge.frame_grabber import FrameGrabber
    from edge.kaaba_seg import KaabaSegmenter
    from edge.ptz import WORK, PtzMatcher, camera_from_homography

    cfg = load_config()
    cams = load_cameras(cfg)
    known = {k: c.centre for k, (_, c) in cams.items()}
    ptz = PtzMatcher(cams, graphics_mask(cfg, WORK), extra=load_extra_refs(cfg))
    seg = KaabaSegmenter()
    target = np.array(args.near)
    png, meta_path, selfcal_path = REFS / f"{args.id}.png", REFS / f"{args.id}.json", REFS / f"{args.id}_selfcal.json"
    best = json.loads(meta_path.read_text()) if meta_path.exists() else None
    found_at = time.monotonic() - 1e9 if best else None
    new_ptz = None
    selfcal = json.loads(selfcal_path.read_text()) if selfcal_path.exists() else []
    log.info("watching for %s near %s (radius %.0f m); calibrated cameras %s; reference %s", args.id, target.tolist(),
             args.radius, list(known), "exists" if best else "not yet")
    deadline = time.monotonic() + args.hours * 3600
    n = 0
    for frame in FrameGrabber.from_env(interval_s=15.0).frames():
        n += 1
        img = frame.image
        h, w = img.shape[:2]
        if (w, h) != (1920, 1080):  # references are 1080p (YouTube); the 720p fallback is skipped
            continue
        gm = graphics_mask(cfg, (w, h))
        try:
            if found_at is not None and time.monotonic() - found_at > args.refine_min * 60:
                if new_ptz is None:  # phase 2: homographies of frames of the new camera
                    ref_img = cv2.imread(str(png))
                    r = best["fit"]
                    cam = kg.Camera(r["f"], np.array(r["rvec"]), np.array(r["tvec"]), (1920, 1080))
                    new_ptz = PtzMatcher({args.id: (ref_img, cam)}, graphics_mask(cfg, WORK))
                    log.info("reference fixed (%s); collecting homographies for the self-calibration", best["captured_utc"])
                m = new_ptz.match(img)
                if m is not None and m.inliers >= MIN_SELFCAL_INLIERS:
                    _, err = camera_from_homography(m.H, new_ptz.cams[args.id], (w, h))
                    selfcal.append({"utc": frame.captured_at.isoformat(timespec="seconds"), "inliers": m.inliers,
                                    "ptz_err": round(err, 4), "H": [float(f"{x:.8g}") for x in (m.H / m.H[2, 2]).ravel()]})
                    selfcal_path.write_text(json.dumps(selfcal, indent=0))
                    log.info("self-calibration homography %d: %d inliers, err %.3f", len(selfcal), m.inliers, err)
                    if len(selfcal) >= args.max_selfcal:
                        break
                continue
            # phase 1: a known camera, checked on the Kaaba?
            pm = ptz.match(img)
            cands = None
            if pm is not None and pm.cam is not None and kaaba_in_view(pm.cam, (h, w), gm) >= IN_VIEW_MIN:
                cands = seg.find(img, gm, k=2)
                if any(min(a / ALIGN_MIN[0], b / ALIGN_MIN[1]) >= 1 for a, b in
                       (kaaba_alignment(pm.cam, c.mask) for c in cands)):
                    continue  # a calibrated camera, right
            cands = seg.find(img, gm, k=2) if cands is None else cands
            for c in cands:
                bx, by, bw, bh = cv2.boundingRect(c.mask)
                x0, y0, x1, y1 = bx, by, bx + bw, by + bh
                inside = x0 > 5 and y0 > 5 and x1 < w - 5 and y1 < h - 5
                if c.dark < 0.3 or not inside or x1 - x0 < MIN_KAABA_PX or float((gm[c.mask > 0]).mean()) > 0.02:
                    continue
                r = kg.fit(c.mask)
                ok, why = good_fit(r)
                if r is None:
                    continue
                d_t = near(r.cam.centre, target)
                d_k = min(near(r.cam.centre, p) for p in known.values())
                log.info("frame %d: ptz %s, Kaaba fit %s, camera %s, f %.0f [%.0f-%.0f], %.0f m from the target, "
                         "%.0f m from the nearest calibrated camera", n, f"{pm.view_id}:{pm.inliers}" if pm else "-",
                         why, r.cam.centre.round(1).tolist(), r.cam.f, *r.f_range, d_t, d_k)
                if not (ok and d_t <= args.radius and d_k >= 15):
                    continue
                score = r.iou - 0.02 * r.rms_px + 0.0005 * (x1 - x0)
                if best is not None and score <= best["score"]:
                    continue
                cv2.imwrite(str(png), img)
                best = {"id": args.id, "captured_utc": frame.captured_at.isoformat(timespec="seconds"),
                        "score": round(score, 4), "kaaba_box": [int(x0), int(y0), int(x1), int(y1)],
                        "ptz_match": {"view": pm.view_id, "inliers": pm.inliers, "err": round(pm.ptz_err, 4)} if pm
                        else None,
                        "fit": {"f": round(r.cam.f, 1), "rvec": [round(float(v), 7) for v in r.cam.rvec],
                                "tvec": [round(float(v), 4) for v in r.cam.tvec], "centre_m": r.cam.centre.round(2).tolist(),
                                "f_range": [round(v, 1) for v in r.f_range], "rms_px": round(r.rms_px, 2),
                                "iou": round(r.iou, 3)},
                        "note": "reference frame of an uncalibrated camera, found by scripts/capture_new_camera.py"}
                meta_path.write_text(json.dumps(best, indent=1))
                found_at = time.monotonic() if found_at is None else found_at
                log.warning("reference for %s saved (score %.3f): %s", args.id, score, png.relative_to(ROOT))
                break
        except Exception:
            log.exception("frame %d failed", n)
        finally:
            del img
        if time.monotonic() > deadline:
            break


if __name__ == "__main__":
    main()
