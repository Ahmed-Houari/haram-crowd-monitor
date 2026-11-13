"""Deploy camera orientation fixes (config/cameras_refit.json "chosen", from scripts/refit_orientation.py or the
half-turn) after the user approved them (2026-10-27). Run once, then restart the live shot service.

  config/cameras.json   the cameras' rvec / tvec / position (the old values kept under "previous")
  data/vloc/bank.npz    every bank frame of those camera places: a pan / tilt / zoom camera keeps its position, so the
                        frame's rotation relative to the reference and its zoom carry over (as scripts/orient_compare.py)
  config/twin.json      the dashboard twin's camera spot (angle, distance, height; scene = 0.116 per metre, the twin's
                        angle = atan2(-Y, -X) + TWIN_ROT)
Backups first: data/backups/pre_orient_<UTC>/ (cameras.json, twin.json, bank.npz).
Readings already stored keep their sectors as they were measured (rings do not depend on the side).

  .venv/Scripts/python scripts/apply_orientation.py [--cams cam0_mataf_low cam1_mataf_wide] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.orient_compare import cam_of, moved  # noqa: E402

CAMS, TWIN, BANK = ROOT / "config" / "cameras.json", ROOT / "config" / "twin.json", ROOT / "data" / "vloc" / "bank.npz"
REFIT = ROOT / "config" / "cameras_refit.json"
SCENE_PER_M = 0.116
TWIN_ROT = 0.32  # the twin's angles vs the Kaaba frame (all four cameras' config/twin.json "ang", 2026-10-27)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cams", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    fix = {k: v["chosen"] for k, v in json.loads(REFIT.read_text()).items() if "chosen" in v and (not args.cams or k in args.cams)}
    reg = json.loads(CAMS.read_text())
    twin = json.loads(TWIN.read_text(encoding="utf-8"))
    z = dict(np.load(BANK))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not args.dry_run:
        bdir = ROOT / "data" / "backups" / f"pre_orient_{stamp}"
        bdir.mkdir(parents=True)
        for f in (CAMS, TWIN, BANK):
            shutil.copy2(f, bdir / f.name)
        print("backup", bdir.relative_to(ROOT))
    cams = z["cams"].copy()
    for cid, new in fix.items():
        old_ref, new_ref = cam_of(reg["cameras"][cid]), cam_of(new)
        n = 0
        for i, place in enumerate(z["places"]):
            if place != cid:
                continue
            c = cams[i]
            frame = cam_of({"f": c[0], "rvec": c[1:4], "tvec": c[4:7], "size": (int(c[7]), int(c[8]))})
            m = moved(frame, old_ref, new_ref)
            cams[i] = np.r_[m.f, np.asarray(m.rvec).ravel(), np.asarray(m.tvec).ravel(), c[7], c[8]]
            n += 1
        X, Y, Z = new["centre_m"]
        tc = twin["cameras"].get(cid)
        if tc is not None:
            tc["ang"] = round(float((np.arctan2(-Y, -X) + TWIN_ROT) % (2 * np.pi)), 3)
            tc["r"] = round(float(np.hypot(X, Y) * SCENE_PER_M), 2) if tc["r"] < 10 else tc["r"]  # roof cams: drawn at the edge
            tc["h"] = round(float(Z * SCENE_PER_M), 2) if tc["h"] < 3.5 else tc["h"]
        reg["cameras"][cid] = {k: v for k, v in new.items()}
        print(f"{cid}: camera {reg['cameras'][cid]['previous']['centre_m']} -> {new['centre_m']} m; {n} bank frames moved; "
              f"twin {tc and {k: tc[k] for k in ('ang', 'r', 'h')}}")
    if args.dry_run:
        print("dry run: nothing written")
        return
    z["cams"] = cams
    tmp = BANK.with_suffix(".tmp.npz")
    np.savez(tmp, **z)
    tmp.replace(BANK)
    CAMS.write_text(json.dumps(reg, indent=1) + "\n")
    TWIN.write_text(json.dumps(twin, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("written: config/cameras.json, config/twin.json, data/vloc/bank.npz; restart the live shot service")


if __name__ == "__main__":
    main()
