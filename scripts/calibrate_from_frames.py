"""Calibrate a new broadcast camera from saved frames (2026-10-26) instead of the live watcher
(scripts/capture_new_camera.py): one saved frame with the whole Kaaba becomes the reference
(data/views/refs/<id>.png), every other saved frame of the same camera that matches it (ALIKED + LightGlue, >= 100
inliers) gives a homography for the pan / tilt / zoom self-calibration of its focal length
(data/views/refs/<id>_selfcal.json, numbers only), then scripts/calibrate_cameras.py --new <id> fits it.

The Kaaba segmenter and the matcher run on the GPU (user, 2026-10-28: everything on the GPU by default), with no memory
cap (user, 2026-10-28: no limits).

  .venv/Scripts/python scripts/calibrate_from_frames.py --id cam8_roof_b --ref data/samples/dvr/20261026T150747Z.jpg \\
      --frames "data/samples/dvr/20261026T1506*.jpg" "data/samples/dvr/20261026T1507*.jpg"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
REFS = ROOT / "data" / "views" / "refs"
MIN_INLIERS = 100


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--id", required=True)
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--frames", nargs="+", required=True, help="glob patterns of saved frames of the same camera")
    args = ap.parse_args()
    from edge import kaaba_geom as kg
    from edge.kaaba_seg import KaabaSegmenter
    from edge.m2_pipeline import whole
    from edge.ptz import WORK, PtzMatcher
    from edge.view_matcher import graphics_mask, load_config

    cfg = load_config()
    ref = cv2.imread(str(args.ref))
    REFS.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(REFS / f"{args.id}.png"), ref)
    h, w = ref.shape[:2]
    seg = KaabaSegmenter()
    c = next((c for c in seg.find(ref, graphics_mask(cfg, (w, h)), k=2) if c.dark >= 0.3 and whole(c.mask)), None)
    if c is None:
        sys.exit("no whole Kaaba on the reference frame")
    r = kg.fit(c.mask)
    print(f"reference {args.ref.name}: Kaaba fit f {r.cam.f:.0f} [{r.f_range[0]:.0f}-{r.f_range[1]:.0f}], rms {r.rms_px:.2f} px, "
          f"camera at {r.cam.centre.round(1).tolist()} m")
    pm = PtzMatcher({args.id: (ref, r.cam)}, graphics_mask(cfg, WORK), keyframes=False)
    files = sorted({f for p in args.frames for f in glob.glob(str(ROOT / p) if not os.path.isabs(p) else p)})
    Hs = []
    for f in files:
        if Path(f).resolve() == args.ref.resolve():
            continue
        m = pm.match(cv2.imread(f))
        if m is not None and m.inliers >= MIN_INLIERS:
            Hs.append({"frame": Path(f).name, "H": np.asarray(m.H, float).ravel().round(8).tolist(), "inliers": m.inliers,
                       "zoom": round(float(m.zoom), 3)})
    (REFS / f"{args.id}_selfcal.json").write_text(json.dumps(Hs, indent=0))
    print(f"{len(Hs)} of {len(files)} frames match the reference with >= {MIN_INLIERS} inliers "
          f"(zoom {min((h['zoom'] for h in Hs), default=0):.2f}-{max((h['zoom'] for h in Hs), default=0):.2f})")
    del pm, seg
    sys.argv = ["calibrate_cameras.py", "--new", args.id]
    from scripts import calibrate_cameras

    calibrate_cameras.main()


if __name__ == "__main__":
    main()
