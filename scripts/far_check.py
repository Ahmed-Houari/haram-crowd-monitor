"""How sure is the far Mataf? (user request 2026-10-27: 25 m from the Kaaba's walls out to the courtyard's edge.)

On saved frames (data/samples/dvr) of the calibrated cameras: each frame's camera is fitted to the Kaaba at the
camera's position (as the live Kaaba gate does; the live pose of the moment when there is no Kaaba fit), DM-Count
once, then edge/m2_metric.measure gives the far Mataf: people per m^2, the floor seen, the head size there, and per
sector (12 angles). Reported:
  - head sizes in the far ring per camera (the counting model needs ~5 px heads; the correction g(h) grows fast below);
  - cameras that see the same far floor (a sector both see) within the same window: their readings compared;
  - the same for the headline ring "all" (1-25 m), as the yardstick.
A confidence level follows from these (written into the report; the API's FAR_CONFIDENCE is set from it).

  .venv/Scripts/python scripts/far_check.py --since 20261026T1800 --until 20261026T1820
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


def cam_of(d: dict) -> kg.Camera:
    return kg.Camera(d["f"], np.asarray(d["rvec"], float), np.asarray(d["tvec"], float), tuple(d["size"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", default="20261026T1800")
    ap.add_argument("--until", default="20261026T1820")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--old-cameras", type=Path, help="cameras.json the sidecars' poses were made with (before a refit)")
    args = ap.parse_args()
    from edge.crowd_model import CrowdModel
    from edge.kaaba_seg import KaabaSegmenter
    from edge.m2_metric import measure
    from edge.m2_pipeline import CAMERA_FACTOR, g, kaaba_exclusion, kaaba_overlap, whole
    from edge.view_matcher import graphics_mask, load_config

    reg = json.loads((ROOT / "config" / "cameras.json").read_text())["cameras"]
    old = json.loads(args.old_cameras.read_text())["cameras"] if args.old_cameras else reg
    cfg = load_config()
    model = CrowdModel()
    seg = KaabaSegmenter()
    rows = []
    per = defaultdict(list)
    for f in sorted(glob.glob(str(ROOT / "data" / "samples" / "dvr" / "*.json"))):
        if not args.since <= Path(f).stem < args.until:
            continue
        m = json.loads(Path(f).read_text())
        if m.get("view") in reg and m.get("cam"):
            per[m["view"]].append((f, m))
    for view, items in per.items():
        for f, m in items[:: args.every]:
            img = cv2.imread(str(Path(f).with_suffix(".jpg")))
            h, w = img.shape[:2]
            gm = graphics_mask(cfg, (w, h))
            live = cam_of(m["cam"])
            R = live.R @ cam_of(old[view]).R.T @ cam_of(reg[view]).R  # the live pose on the current calibration
            cam = kg.Camera(live.f * reg[view]["f"] / old[view]["f"], cv2.Rodrigues(R)[0].ravel(),
                            -R @ np.asarray(reg[view]["centre_m"], float), live.size)
            how, ov = "live", None
            masks = [c.mask for c in seg.find(img, gm, k=2) if c.dark >= 0.3]
            for mk in masks:
                if whole(mk):
                    r = kg.fit_from_centre(mk, np.asarray(reg[view]["centre_m"], float), init=cam, budget_s=3)
                    if r is not None and kaaba_overlap(r.cam, mk) >= 0.8:
                        cam, how, ov = r.cam, "kaaba", round(kaaba_overlap(r.cam, mk), 3)
                        break
            dens = model.predict(img).density
            ex = np.maximum(kaaba_exclusion(cam, (h, w)), gm)
            rd = measure(cam, dens, ex, g, CAMERA_FACTOR.get(view, 1.0))
            rows.append({"view": view, "stem": Path(f).stem, "how": how, "overlap": ov, "far": rd.far,
                         "all": rd.density_corr.get("all"), "sectors": rd.sectors or {}})
    print(f"frames {args.since}-{args.until}: " + ", ".join(f"{v} {sum(r['view'] == v for r in rows)}" for v in per)
          + f"; fitted to the Kaaba {sum(r['how'] == 'kaaba' for r in rows)}")
    print("far Mataf per camera: frames with a reading, people per m^2 (median), floor seen (m^2), head px (median of "
          "frames' medians):")
    for v in per:
        fr = [r["far"] for r in rows if r["view"] == v and r["far"]]
        if fr:
            print(f"  {v:16s} {len(fr):3d}  {np.median([x['corr'] for x in fr]):.2f} people/m^2 (raw {np.median([x['raw'] for x in fr]):.2f}), "
                  f"{np.median([x['area'] for x in fr]):6.0f} m^2, heads {np.median([x['head_px'] for x in fr]):.1f} px; "
                  f"headline ring {np.median([r['all'] for r in rows if r['view'] == v and r['all']]):.2f}")
    # pooled per camera and sector, then pairs on shared sectors
    pool = defaultdict(lambda: np.zeros(2))
    for r in rows:
        for s, x in ((r["far"] or {}).get("sectors") or {}).items():
            pool[(r["view"], s)] += [x[1], x[2]]
        for s, x in r["sectors"].items():
            pool[(r["view"], s)] += [x[1], x[2]]
    dens = {k: c / a for k, (c, a) in pool.items() if a >= 30}
    views = list(per)
    print("cameras on the same floor (sectors both saw, >= 30 m^2 each), median |log ratio| (0.10 = 10 %):")
    for kind, pre in (("far Mataf", "f"), ("headline rings", ("i", "o"))):
        for i, a in enumerate(views):
            for b in views[i + 1:]:
                common = sorted({s for (v, s) in dens if v == a and s.startswith(pre) and (b, s) in dens})
                if len(common) >= 2:
                    lr = [abs(np.log(dens[(a, s)] / dens[(b, s)])) for s in common if dens[(a, s)] > 0 and dens[(b, s)] > 0]
                    print(f"  {kind:14s} {a} / {b}: {len(common)} sectors, median |log ratio| {np.median(lr):.2f} ({', '.join(common)})")
    out = ROOT / "data" / "samples" / "debug_align" / "far_check.json"
    out.write_text(json.dumps(rows, default=float))
    print("rows:", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
