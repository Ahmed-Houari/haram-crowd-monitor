"""Per-frame camera poses refitted on the Kaaba at the calibrated positions (user request 2026-10-27, offline: nothing
live changes). scripts/refit_positions.py found the positions of the wide and roof cameras right (with the Kaaba's base
in view the full box fits within ~1 px from them; a free position only slides along the line of sight) and the Kaaba's
height right (fitted 13.5-14.3 m to the top of the Kiswa; published: a 14 m Kiswa on a 0.25 m base): the base offsets
come from the per-frame poses (tracked, bank, ptz: rotation and zoom from image matching, checked against the Kaaba
only loosely). Here every placed frame of the fine-tune windows with the Kaaba in view gets the live anchor fit
(edge/kaaba_geom.fit_from_centre: rotation and focal length at the calibrated position, on the Kaaba's mask from
edge/kaaba_seg.py), kept when it fits (silhouette IoU >= MIN_IOU, rms <= MAX_RMS px); the rows strips
(finetune_data.rows_segments) are measured again with the new pose.
cam0_mataf_low keeps its stored poses and is marked: its saved frames come from two sides of the Kaaba (the door is in
view on 26 Oct, not on 27 Oct; positions (-52, +30) and (-52, -30) give the same box outline), so its floor positions
cannot be shared with other cameras (no carried-over rows labels to or from it).
Writes data/finetune_r/<window>.json (+ the density maps, unchanged) and prints the m2 reading (live model, live
correction, 1-25 m) before / after per camera.

  .venv/Scripts/python scripts/refit_frames.py [--windows 2026-10-26-maghrib ...]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from edge import kaaba_geom as kg  # noqa: E402
from edge.kept_frames import frame_file  # noqa: E402

DATA = ROOT / "data" / "finetune"
OUT = ROOT / "data" / "finetune_r"
MIN_IOU, MAX_RMS = 0.85, 3.0
IN_VIEW = 0.6  # share of the Kaaba (above head level) in the frame for a refit
SKIP = {"cam0_mataf_low"}


def cam_of(c: dict) -> kg.Camera:
    return kg.Camera(c["f"], np.asarray(c["rvec"], float), np.asarray(c["tvec"], float), tuple(c["size"]))


def cam_dict(cam: kg.Camera) -> dict:
    return {"f": float(cam.f), "rvec": [float(x) for x in cam.rvec], "tvec": [float(x) for x in cam.tvec],
            "size": list(cam.size)}


def sil(cam: kg.Camera, z0: float) -> np.ndarray:
    m = np.zeros((1080, 1920), np.uint8)
    s = kg.silhouette(cam, kg.box_corners(z0=z0))
    if s is not None:
        cv2.fillPoly(m, [np.round(s[0]).astype(np.int32)], 1)
    return m


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", nargs="*")
    args = ap.parse_args()
    from edge.kaaba_seg import KaabaSegmenter
    from edge.m2_metric import measure
    from edge.m2_pipeline import CAMERA_FACTOR, g_for, kaaba_exclusion, kaaba_in_view
    from edge.view_matcher import graphics_mask, load_config
    from finetune_data import kg_mask, rows_segments

    cams = json.loads((ROOT / "config" / "cameras.json").read_text())["cameras"]
    gm = graphics_mask(load_config(), (1920, 1080))
    seg = KaabaSegmenter()
    names = args.windows or sorted(p.stem for p in DATA.glob("*.json") if not p.stem.startswith("report")
                                   and not p.stem.endswith("-test"))
    OUT.mkdir(parents=True, exist_ok=True)
    cmp = defaultdict(list)
    t0 = time.time()
    for name in names:
        meta = json.loads((DATA / f"{name}.json").read_text())
        dens = np.load(DATA / f"{name}.npz")["density"]
        n_ok = 0
        for i, fr in enumerate(meta["frames"]):
            old = cam_of(fr["cam"])
            fr["cam_stored"] = fr["cam"]
            view = fr["camera"]
            if view in SKIP:
                fr["refit"] = {"status": "kept: two sides"}
                continue
            if view not in cams or kaaba_in_view(old, (1080, 1920), gm) < IN_VIEW:
                fr["refit"] = {"status": "kept: Kaaba not in view"}
                continue
            img = cv2.imread(str(frame_file(fr['stem'])))
            ref = sil(old, kg.CUT_Z)
            cands = seg.find(img, gm, k=2)
            if not cands:
                fr["refit"] = {"status": "kept: no Kaaba mask"}
                continue
            km = max(cands, key=lambda c: iou(c.mask, ref)).mask
            ft = kg.fit_from_centre(km, np.asarray(cams[view]["centre_m"], float), init=old, budget_s=6)
            before = iou(km, sil(old, kg.Z_BOTTOM))
            if ft is None or ft.iou < MIN_IOU or ft.rms_px > MAX_RMS:
                fr["refit"] = {"status": "kept: fit failed", "iou": None if ft is None else round(ft.iou, 3),
                               "rms": None if ft is None else round(ft.rms_px, 2)}
                continue
            new = ft.cam
            fr["cam"] = cam_dict(new)
            fr["refit"] = {"status": "refit", "iou": round(ft.iou, 3), "rms": round(ft.rms_px, 2), "z_occ": round(ft.z_occ, 2),
                           "box_iou_before": round(before, 3), "box_iou_after": round(iou(km, sil(new, kg.Z_BOTTOM)), 3)}
            blocked = np.maximum(gm, kg_mask(kg, new, img.shape[:2]))
            fr["segs_stored"] = fr.get("segs")
            fr["segs"] = rows_segments(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), new, blocked)
            # the m2 reading (live model, live correction) with the stored and the refitted pose
            d = dens[fr["i"]].astype(np.float32)
            g, fac = g_for("youtube"), CAMERA_FACTOR.get(view, 1.0)
            r = []
            for c in (old, new):
                rd = measure(c, d, np.maximum(kaaba_exclusion(c, (1080, 1920)), gm), g, fac)
                r.append(rd.density_corr.get("all"))
            if r[0] and r[1]:
                cmp[view].append((r[0], r[1], before, fr["refit"]["box_iou_after"]))
            n_ok += 1
        (OUT / f"{name}.json").write_text(json.dumps(meta))
        shutil.copyfile(DATA / f"{name}.npz", OUT / f"{name}.npz")
        st = defaultdict(int)
        for fr in meta["frames"]:
            st[fr["refit"]["status"]] += 1
        print(f"{name}: {len(meta['frames'])} frames, {dict(st)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print("\nm2 reading 1-25 m (live model, live correction): refitted / stored pose, and the full box's overlap with the "
          "Kaaba's mask before -> after")
    for view, v in sorted(cmp.items()):
        a = np.array(v)
        ratio = a[:, 1] / a[:, 0]
        print(f"  {view:16s} {len(a):4d} frames  ratio median {np.median(ratio):.3f} (10-90 % {np.percentile(ratio, 10):.3f}-"
              f"{np.percentile(ratio, 90):.3f}), reading median {np.median(a[:, 0]):.2f} -> {np.median(a[:, 1]):.2f}; "
              f"box IoU median {np.median(a[:, 2]):.3f} -> {np.median(a[:, 3]):.3f}")
    (OUT / "refit_summary.json").write_text(json.dumps({k: np.round(np.array(v), 4).tolist() for k, v in cmp.items()}))


if __name__ == "__main__":
    main()
