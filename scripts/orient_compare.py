"""Before deploying a camera orientation fix (user request 2026-10-27): do the ring values stay the same and do the
sector values now agree across cameras?

Saved frames with their camera (data/samples/dvr sidecars) are measured twice with DM-Count's map from one model run:
with the camera as it was found live (old calibration) and with the same camera moved onto the fixed calibration
(config/cameras_refit.json "chosen": a pan / tilt / zoom camera keeps its position, so the frame's rotation relative
to its reference and its zoom carry over: R_new = R_frame R_ref_old^T R_ref_new, f scaled by f_ref_new / f_ref_old).
Rings: per frame, new / old. Sectors (12 angles x inner / outer ring, people per m^2): per camera pooled over the
window, then each fixed camera against the roof cameras on the sectors both see, before and after.

  .venv/Scripts/python scripts/orient_compare.py --since 20261026T1800 --until 20261026T1820
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge import kaaba_geom as kg  # noqa: E402

REFS = ("cam5_roof", "cam8_roof_b")
NEAR_DEG = 90


def cam_of(d: dict) -> kg.Camera:
    return kg.Camera(d["f"], np.asarray(d["rvec"], float), np.asarray(d["tvec"], float), tuple(d["size"]))


def moved(frame: kg.Camera, ref_old: kg.Camera, ref_new: kg.Camera) -> kg.Camera:
    R = frame.R @ ref_old.R.T @ ref_new.R
    f = frame.f * ref_new.f / ref_old.f
    return kg.Camera(f, cv2.Rodrigues(R)[0].ravel(), -R @ ref_new.centre, frame.size)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", default="20261026T1800")
    ap.add_argument("--until", default="20261026T1820")
    ap.add_argument("--every", type=int, default=2, help="every n-th frame per camera")
    args = ap.parse_args()
    from edge.crowd_model import CrowdModel
    from edge.m2_metric import measure
    from edge.m2_pipeline import CAMERA_FACTOR, g, kaaba_exclusion
    from edge.view_matcher import graphics_mask, load_config

    reg = json.loads((ROOT / "config" / "cameras.json").read_text())["cameras"]
    fix = {k: v["chosen"] for k, v in json.loads((ROOT / "config" / "cameras_refit.json").read_text()).items() if "chosen" in v}
    cfg = load_config()
    model = CrowdModel()
    per = defaultdict(list)
    for f in sorted(glob.glob(str(ROOT / "data" / "samples" / "dvr" / "*.json"))):
        stem = Path(f).stem
        if not args.since <= stem < args.until:
            continue
        try:
            m = json.loads(Path(f).read_text())
        except (OSError, ValueError):
            continue
        if m.get("view") in reg and m.get("cam"):
            per[m["view"]].append((f, m))
    rings, sect = defaultdict(list), defaultdict(lambda: defaultdict(lambda: np.zeros(2)))
    for view, items in per.items():
        for f, m in items[:: args.every]:
            img = cv2.imread(str(Path(f).with_suffix(".jpg")))
            if img is None:
                continue
            h, w = img.shape[:2]
            dens = model.predict(img).density
            old = cam_of(m["cam"])
            cams = {"old": old}
            cams["new"] = moved(old, cam_of(reg[view]), cam_of(fix[view])) if view in fix else old
            q = CAMERA_FACTOR.get(view, 1.0)
            out = {}
            for k, cam in cams.items():
                ex = np.maximum(kaaba_exclusion(cam, (h, w)), graphics_mask(cfg, (w, h)))
                rd = measure(cam, dens, ex, g, q)
                out[k] = rd
                for s, v in (rd.sectors or {}).items():
                    sect[(view, k)][s] += [v[1], v[2]]  # corrected count, m^2
            rings[view].append({r: (out["old"].density_corr.get(r), out["new"].density_corr.get(r)) for r in ("all", "inner", "outer")})
    print(f"frames {args.since}-{args.until}: " + ", ".join(f"{v} {len(rings[v])}" for v in rings))
    print("rings, new / old per frame (median, range):")
    for view, rows in rings.items():
        parts = []
        for r in ("all", "inner", "outer"):
            x = [n / o for o, n in (row[r] for row in rows) if o and n]
            if x:
                parts.append(f"{r} {np.median(x):.3f} ({min(x):.3f}-{max(x):.3f})")
        print(f"  {view:16s} " + "; ".join(parts))
    dens = {k: {s: c / a for s, (c, a) in d.items() if a >= 30} for k, d in sect.items()}
    print("sectors, people per m^2 pooled over the window (i = inner ring, o = outer; sector 0 starts at -180 deg):")
    names = sorted({s for d in dens.values() for s in d}, key=lambda s: (s[0], int(s[1:])))
    print("  " + " " * 22 + " ".join(f"{s:>5s}" for s in names))
    for k in sorted(dens):
        print(f"  {k[0]:16s} {k[1]:4s}  " + " ".join(f"{dens[k][s]:5.2f}" if s in dens[k] else "    -" for s in names))
    # near sectors only: a camera reads its own side of the Kaaba well and the far side poorly (small heads, the
    # Kaaba in between); a pair is compared where both are near (sector centre within NEAR_DEG of both azimuths)
    azim = {}
    for view in per:
        for k in ("old", "new"):
            d = fix[view] if (k == "new" and view in fix) else reg[view]
            c = d["centre_m"]
            azim[(view, k)] = float(np.degrees(np.arctan2(c[1], c[0])))

    def near(s: str, a: float) -> bool:
        centre = -180 + 30 * int(s[1:]) + 15
        return abs((centre - a + 180) % 360 - 180) <= NEAR_DEG

    print(f"pairs on sectors near both cameras (within {NEAR_DEG} deg of each one's azimuth), median |log ratio| "
          "after each camera's own level (median) is taken out; lower = the same pattern:")
    views = list(per)
    for i, va in enumerate(views):
        for vb in views[i + 1:]:
            for k in ("old", "new"):
                a, b = dens.get((va, k), {}), dens.get((vb, k), {})
                common = [s for s in a if s in b and a[s] > 0 and b[s] > 0 and near(s, azim[(va, k)]) and near(s, azim[(vb, k)])]
                if len(common) < 2:
                    print(f"  {va} / {vb} {k:3s}: {len(common)} shared near sector(s)")
                    continue
                la, lb = np.log([a[s] for s in common]), np.log([b[s] for s in common])
                d = (la - np.median(la)) - (lb - np.median(lb))
                print(f"  {va} / {vb} {k:3s}: {len(common)} sectors, pattern |log ratio| {np.median(np.abs(d)):.2f}, "
                      f"level ratio {np.exp(np.median(la - lb)):.2f} ({', '.join(common)})")
    print("agreement with the roof cameras on shared sectors (median |log ratio|; lower = closer):")
    for view in per:
        if view in REFS:
            continue
        for ref in REFS:
            if (ref, "old") not in dens:
                continue
            for k in ("old", "new"):
                a, b = dens.get((view, k), {}), dens[(ref, "old")]
                common = [s for s in a if s in b and a[s] > 0 and b[s] > 0]
                if common:
                    lr = [abs(np.log(a[s] / b[s])) for s in common]
                    print(f"  {view} {k:3s} vs {ref}: {len(common)} sectors, median |log ratio| {np.median(lr):.2f} "
                          f"({', '.join(common)})")
                else:
                    print(f"  {view} {k:3s} vs {ref}: no shared sector")


if __name__ == "__main__":
    main()
