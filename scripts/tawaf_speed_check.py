"""Offline check of the tawaf speed sensor (edge/tawaf_speed.py) on recorded 1 s frames (data/finetune_rec, the frames'
stored poses). Prints per ring the speed density, how much of the ring walks / is estimated / is stopped, and the
sectors; draws one frame's sector map (people per m^2 on the crowd-safety levels; hatched = estimated from the ring's
neighbours, grey = stopped: the count model's). Not used by the live services. Frames stay local; numbers only.

  .venv/Scripts/python scripts/tawaf_speed_check.py --window screen_20261029T145041Z_DISPLAY1 --camera cam0_mataf_low \
      --from 20261029T154300Z --to 20261029T154900Z [--step 3] [--overlay 20261029T154600Z --out x.jpg]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def ring_polygon(cam, d0: float, d1: float, a0: float, a1: float) -> np.ndarray | None:
    """Image polygon of the floor (head plane) between d0 and d1 m from the walls and bearings a0..a1 deg."""
    from edge import kaaba_geom as kg
    from edge.m2_metric import wall_distance

    rs = np.linspace(0, 80, 1601)
    pts = []
    for d, angs in ((d0, np.linspace(a0, a1, 8)), (d1, np.linspace(a1, a0, 8))):
        for a in np.radians(angs):
            wdl = wall_distance(rs * np.cos(a), rs * np.sin(a))
            r = rs[np.argmax(wdl >= d)]
            pts.append([r * np.cos(a), r * np.sin(a), kg.HEAD_Z])
    uv, z = cam.project(np.array(pts))
    if (z <= 0).any():
        return None
    return uv


def draw(img, cam, secs: dict, title: list[str], blocked: np.ndarray) -> np.ndarray:
    from backend.aiview import ramp_m2
    from edge.tawaf_speed import RINGS, SECTOR_DEG

    h, w = img.shape[:2]
    lay = img.copy()
    hatch = np.zeros((h, w), np.uint8)
    grey = np.zeros((h, w), np.uint8)
    for ring, (d0, d1) in RINGS.items():
        for s in secs[ring]:
            a0, a1 = s["deg"] - SECTOR_DEG / 2, s["deg"] + SECTOR_DEG / 2
            poly = ring_polygon(cam, d0, d1, a0, a1)
            if poly is None or s["how"] == "none":
                continue
            p = np.int32(np.round(poly))
            if s["rho"] is not None:
                c = ramp_m2(np.array([s["rho"]]))[0][::-1]
                cv2.fillPoly(lay, [p], tuple(int(x) for x in c))
                if s["how"] == "estimated":
                    cv2.fillPoly(hatch, [p], 1)
            else:
                cv2.fillPoly(grey, [p], 1)
            cv2.polylines(lay, [p], True, (40, 40, 40), 1, cv2.LINE_AA)
    out = cv2.addWeighted(lay, 0.55, img, 0.45, 0)
    yy, xx = np.indices((h, w))
    lines = ((xx + yy) % 12) < 3
    out[(hatch > 0) & lines] = (232, 236, 236)
    out[(grey > 0) & lines] = (150, 150, 150)
    out[blocked > 0] = img[blocked > 0]  # the Kaaba stays in front of the floor behind it
    band = np.zeros((20 + 34 * len(title), w, 3), np.uint8)
    for i, t in enumerate(title):
        cv2.putText(band, t, (16, 30 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return np.vstack([band, out])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--from", dest="t0", required=True)
    ap.add_argument("--to", dest="t1", required=True)
    ap.add_argument("--step", type=int, default=3, help="use every n-th 1 s pair")
    ap.add_argument("--overlay", help="stem of the frame to draw")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--note", default="", help="extra caption line")
    args = ap.parse_args()
    from edge import kaaba_geom as kg
    from edge.m2_pipeline import kaaba_exclusion
    from edge.tawaf_speed import TawafSpeed
    from edge.view_matcher import graphics_mask, load_config

    m = json.loads((ROOT / "data" / "finetune_rec" / f"{args.window}.json").read_text())
    t = lambda f: datetime.strptime(f["stem"], "%Y%m%dT%H%M%SZ").timestamp()  # noqa: E731
    fs = sorted((f for f in m["frames"] if f.get("cam") and f["camera"] == args.camera and args.t0 <= f["stem"] <= args.t1),
                key=t)
    prs = [(a, b) for a, b in zip(fs, fs[1:]) if t(b) - t(a) == 1][:: args.step]
    gmask = graphics_mask(load_config(), (1920, 1080))
    sensor = TawafSpeed()
    pairs, dropped, ov = [], 0, None
    for a, b in prs:
        c = a["cam"]
        cam = kg.Camera(c["f"], np.asarray(c["rvec"], float), np.asarray(c["tvec"], float), tuple(c["size"]))
        ia, ib = cv2.imread(str(ROOT / a["path"])), cv2.imread(str(ROOT / b["path"]))
        blocked = kaaba_exclusion(cam, ia.shape[:2])
        p = sensor.pair(ia, ib, cam, 1.0, gmask, blocked)
        if p is None:
            dropped += 1
            continue
        pairs.append(p)
        near = args.overlay and (ov is None or abs(t(a) - t({"stem": args.overlay})) < abs(t({"stem": ov[3]}) - t({"stem": args.overlay})))
        if ov is None and not args.overlay or near:
            ov = (ia, cam, blocked, a["stem"])  # the drawn frame: --overlay's nearest used pair
    secs = TawafSpeed.sectors(pairs)
    print(f"{args.window} {args.camera} {args.t0}-{args.t1}: {len(pairs)} pairs used, {dropped} dropped (camera motion)")
    lines = []
    for ring, ss in secs.items():
        r = TawafSpeed.ring_reading(ss)
        vt = [s["speed"] for s in ss if s.get("speed") is not None and s["how"] in ("walking", "slow", "fast")]
        line = (f"{ring} ring: speed density {r['rho'] if r['rho'] is not None else '-'} people/m2 | tawaf speed "
                f"{np.median(vt):.2f} m/s | " if vt else f"{ring} ring: no walking sectors | ")
        line += f"walking {r['walking']:.0%} (slow end {r['slow']:.0%}, fast {r['fast']:.0%}), estimated {r['estimated']:.0%}, stopped {r['stopped']:.0%}"
        print("  " + line)
        print("    " + " ".join(f"{s['deg']:.0f}:{s['how'][:4]}{'' if s['rho'] is None else '=' + str(s['rho'])}" for s in ss))
        lines.append(line)
    if args.out and ov is not None:
        ia, cam, blocked, stem = ov
        title = [f"Tawaf speed sensor (offline): {args.camera} {stem}, {len(pairs)} pairs 1 s apart",
                 "colour = people/m2 from speed; hatched = estimated from the ring's neighbours; grey = stopped (count model's)"] + lines + ([args.note] if args.note else [])
        cv2.imwrite(str(args.out), draw(ia, cam, secs, title, blocked), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("overlay", args.out)


if __name__ == "__main__":
    main()
