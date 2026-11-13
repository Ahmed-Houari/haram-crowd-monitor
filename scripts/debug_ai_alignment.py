"""Alignment check of "What the AI sees" against the real frame (2026-10-26). LOCAL ONLY: the overlays show video
pixels, so they go to data/samples/debug_align/ (gitignored, never served or uploaded) and are not committed.

For the newest saved m2 sample frames of each calibrated camera (data/samples/m2_<camera>_*.png + the sidecar with the
frame's camera), the dashboard's own pipeline is rebuilt end to end: DM-Count density on the frame (as the collector),
the collector's heatmap PNG (edge/heatmap.py density_png + heat_scale), the reference -> frame homography (from the
calibrated reference camera and the frame's camera, on the head plane), backend/floormap.image_density +
backend/aiview.heat_m2 (exactly what /api/aiview/<id>.png draws). Then, on the frame:
  floor share  = share of the model's density mass that lies on the Mataf floor (the AI view draws only the floor);
  in warm      = share of the floor's density mass inside the AI view's warm areas (heat drawn at >= 25 % opacity);
  shift        = offset (frame px) between the model's density and the AI view's heat (phase correlation);
  mirror       = correlation with the heat as drawn vs flipped left-right / upside down (mirroring would win).
Target: in warm >= 90 %, shift ~0, no mirroring. Overlay: the frame darkened, the AI view's heat on it, the Kaaba's
projected outline in white.

  .venv/Scripts/python -I scripts/debug_ai_alignment.py [--per-camera 3]
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

from backend import aiview, floormap  # noqa: E402
from edge.heatmap import density_png, heat_scale  # noqa: E402
from edge.kaaba_geom import HEAD_Z, Camera  # noqa: E402
from edge.m2_metric import load_correction  # noqa: E402
from edge.view_matcher import load_config  # noqa: E402

SAMPLES = ROOT / "data" / "samples"
OUT = SAMPLES / "debug_align"


def cam_of(d: dict) -> Camera:
    return Camera(d["f"], np.asarray(d["rvec"], float), np.asarray(d["tvec"], float), tuple(d["size"]))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-camera", type=int, default=3)
    args = ap.parse_args()
    from edge.crowd_model import CrowdModel

    model = CrowdModel()
    cfg = load_config()
    cams = json.loads((ROOT / "config" / "cameras.json").read_text(encoding="utf-8"))["cameras"]
    h0, k = load_correction()[:2]
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "_heatmap.png"
    rows = []
    for cid in cams:
        files = sorted(SAMPLES.glob(f"m2_{cid}_*.json"))[-args.per_camera:]
        for side in files:
            meta = json.loads(side.read_text())
            frame = cv2.imread(str(side.with_suffix(".png")))
            if frame is None:
                continue
            fh, fw = frame.shape[:2]
            dens = model.predict(frame).density  # people per 8 x 8 px cell
            tmp.write_bytes(density_png(dens, (fw, fh)))  # the collector's live heatmap, no zone lines
            top = heat_scale(dens, (fw, fh))
            ref, fr = cam_of(cams[cid]), cam_of(meta["cam"])
            H = fr.plane_to_image(HEAD_Z) @ np.linalg.inv(ref.plane_to_image(HEAD_Z))  # reference px -> frame px
            H /= H[2, 2]
            hm = {"h_primary": H.tolist(), "density_top": top, "frame_size": [fw, fh], "ts": meta["ts_utc"], "view_id": cid}
            m = floormap.image_density(tmp, hm, cams[cid], cfg, h0, k)
            if m is None:
                continue
            hw, hh = m["dens"].shape[1], m["dens"].shape[0]
            heat = aiview.heat_m2(m["dens"], (hw * aiview.UPSCALE, hh * aiview.UPSCALE), m.get("px_per_m"))
            heat = cv2.resize(heat, (fw, fh), interpolation=cv2.INTER_LINEAR)
            # the model's density at frame resolution (mass per px) and the floor mask
            mass = cv2.resize(dens, (fw, fh), interpolation=cv2.INTER_LINEAR) / 64.0
            floor = cv2.resize(m["floor"].astype(np.uint8), (fw, fh), interpolation=cv2.INTER_NEAREST) > 0
            warm = heat[..., 3] >= 64
            total, on_floor = mass.sum(), mass[floor].sum()
            in_warm = mass[floor & warm].sum() / max(on_floor, 1e-9)
            # offset and mirroring at heatmap resolution
            a = cv2.resize(np.where(floor, mass, 0).astype(np.float32), (hw, hh), interpolation=cv2.INTER_AREA)
            a = cv2.GaussianBlur(a, (0, 0), 2)
            b = cv2.resize(heat[..., 3].astype(np.float32) * (heat[..., 2].astype(np.float32) + 50), (hw, hh),
                           interpolation=cv2.INTER_AREA)
            (dx, dy), _ = cv2.phaseCorrelate(a, b)
            c0, cx, cy = corr(a, b), corr(a, b[:, ::-1]), corr(a, b[::-1, :])
            rows.append((cid, side.stem[-16:], on_floor / max(total, 1e-9), in_warm, dx * fw / hw, dy * fh / hh, c0, cx, cy))
            # overlay: darkened frame, the AI view's heat, the Kaaba's outline (local file only)
            a3 = heat[..., 3:4].astype(np.float32) / 255.0
            over = (frame.astype(np.float32) * 0.45 * (1 - a3) + heat[..., :3].astype(np.float32) * a3).astype(np.uint8)
            if m.get("kaaba") is not None:
                cv2.polylines(over, [np.int32(m["kaaba"] * [fw / hw, fh / hh])], True, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(over, f"{cid}  in warm {100 * in_warm:.0f} %  shift {dx * fw / hw:+.0f},{dy * fh / hh:+.0f} px",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(OUT / f"{side.stem}_overlay.jpg"), cv2.resize(over, (fw // 2, fh // 2)), [cv2.IMWRITE_JPEG_QUALITY, 85])
    tmp.unlink(missing_ok=True)
    print(f"{'camera':16s} {'frame':16s} floor%  inwarm%  shift_x  shift_y  corr  flipLR flipUD")
    for r in rows:
        print(f"{r[0]:16s} {r[1]:16s} {100 * r[2]:5.0f}  {100 * r[3]:6.0f}  {r[4]:+7.0f}  {r[5]:+7.0f}  {r[6]:.2f}  {r[7]:+.2f}  {r[8]:+.2f}")
    for cid in cams:
        rs = [r for r in rows if r[0] == cid]
        if rs:
            print(f"{cid}: in warm {100 * np.mean([r[3] for r in rs]):.0f} % (mean of {len(rs)}), on floor "
                  f"{100 * np.mean([r[2] for r in rs]):.0f} %, |shift| <= {max(max(abs(r[4]), abs(r[5])) for r in rs):.0f} px")
    print(f"overlays: {OUT}")


if __name__ == "__main__":
    main()
