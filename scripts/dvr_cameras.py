"""Which saved DVR frames (data/samples/dvr, scripts/grab_dvr.py) show a calibrated camera, and the metric on them
(investigation 2026-10-26, Maghrib accuracy). Per frame: the ptz match (ALIKED + LightGlue, no Kaaba check: the
rotation-error gate only, to stay small on the shared GPU; --anchor adds the OWLv2 + SAM check and the anchor fit),
DM-Count's density, the live ring readings. Writes a sidecar <frame>.json next to each matched frame (its camera and
readings) and data/samples/dvr/cameras.csv.

  .venv/Scripts/python scripts/dvr_cameras.py --since 20261026T1440 --until 20261026T1530
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DVR = ROOT / "data" / "samples" / "dvr"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="")
    ap.add_argument("--until", default="99999999T9999")
    ap.add_argument("--anchor", action="store_true")
    args = ap.parse_args()
    from edge.crowd_model import CrowdModel
    from edge.m2_pipeline import M2Pipeline
    from edge.view_matcher import graphics_mask, load_config

    cfg = load_config()
    model = CrowdModel()
    m2 = M2Pipeline(cfg, model, use_kaaba=False, verify=args.anchor, anchor=args.anchor)
    files = [f for f in sorted(DVR.glob("*.jpg")) if args.since <= f.stem < args.until]
    rows = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        gm = graphics_mask(cfg, (w, h))
        t = time.perf_counter()
        fc = m2.camera(img, gm, try_kaaba=False)
        row = {"file": f.name, "path": fc.path or "-", "view": fc.info.get("ptz_view") or fc.info.get("anchor_view") or "",
               "zoom": fc.info.get("ptz_zoom", ""), "err": fc.info.get("ptz_err", ""), "inliers": fc.info.get("ptz_inliers", "")}
        if fc.cam is not None:
            rd = m2.reading(img, fc, gm)
            row |= {"m2_all": rd.density_corr.get("all"), "raw_all": rd.density.get("all"), "area_all": rd.area.get("all"),
                    "head_all": rd.head_px.get("all"), "m2_inner": rd.density_corr.get("inner"),
                    "raw_inner": rd.density.get("inner"), "count_raw": float(m2.last_density.sum())}
            c = fc.cam
            f.with_suffix(".json").write_text(json.dumps({
                "cam": {"f": float(c.f), "rvec": [float(x) for x in c.rvec], "tvec": [float(x) for x in c.tvec],
                        "size": list(c.size)}, **{k: v for k, v in row.items() if k != "file"},
                "rings": {k: [rd.count.get(k), rd.area.get(k), rd.density.get(k), rd.density_corr.get(k), rd.head_px.get(k)]
                          for k in rd.area}}, default=str))
        row["ms"] = round((time.perf_counter() - t) * 1e3)
        rows.append(row)
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    out = DVR / "cameras.csv"
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with out.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, keys)
        wr.writeheader()
        wr.writerows(rows)
    n = sum(r["path"] != "-" for r in rows)
    print(f"{n} of {len(rows)} frames with a calibrated camera; {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
