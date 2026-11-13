"""Coverage replay of the view-independent metric on the saved thumbnails (research, 2026-10-24; read-only).

For every labelled survey thumbnail (data/views/thumbs/labels.csv, 480x270, one every 5 min since 2026-10-23 23:07
Makkah) the frame is upscaled to 1080p and given a camera like the shadow run does: path "ptz" (ALIKED + LightGlue to a
calibrated reference, config/cameras.json, rotation + zoom check) or, failing that, path "kaaba" (a camera fitted to
the Kaaba's silhouette, strict gate). Thumbnails are replayed in time order, so the PTZ keyframes (edge/ptz.py) build up
as they would live (off with --no-keyframes); ptz cameras are checked on the Kaaba when it is in view (off with
--no-verify; edge/m2_pipeline.py). --sheets also writes contact sheets of the counted frames with their projected
Kaaba box to data/views/discover/ (day / night; lines on the stored thumbnails). A frame counts when it has a camera and >= MIN_AREA_M2 of the Mataf ring (1-25 m
from the Kaaba's walls) is in view. The current metric (visible part of a fixed zone, SIFT) is replayed on the same
thumbnails. No crowd model: coverage only; thumbnails have 1/4 of the pixels, so both are conservative.

Counted minutes per hour from the share p of frames counted: between 60p and 60(1-(1-p)^4), middle value reported.

  .venv/Scripts/python scripts/replay_m2.py [--no-kaaba] [--no-keyframes] [--no-verify] [--no-anchor] [--sheets]
                                            [--out data/views/thumbs/replay_m2.csv]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from edge import kaaba_geom as kg  # noqa: E402
from edge.m2_metric import ALL, MIN_AREA_M2, cell_floor, wall_distance  # noqa: E402
from edge.view_matcher import FULL_VISIBLE, MIN_VISIBLE, ViewMatcher, graphics_mask, load_config, zoom_range  # noqa: E402

FULL = (1920, 1080)


def ring_area(cam: kg.Camera, ex: np.ndarray) -> float:
    dshape = (FULL[1] // 8, FULL[0] // 8)
    geo = cell_floor(cam, dshape)
    dh, dw = dshape
    exs = ex[: dh * 8, : dw * 8].reshape(dh, 8, dw, 8).mean((1, 3)) < 0.5
    d = wall_distance(geo["X"], geo["Y"])
    sel = geo["valid"] & exs & (d >= ALL[0]) & (d < ALL[1])
    return float(geo["area"][sel].sum())


def draw_box(th: np.ndarray, cam: kg.Camera, text: str) -> np.ndarray:
    """The camera's Kaaba box (red lines) on a stored thumbnail, for the contact sheets."""
    vis = th.copy()
    s = th.shape[1] / FULL[0]
    uv, z = cam.project(kg.box_corners())
    if (z > 0).all():
        for a, b in kg.BOX_EDGES:
            cv2.line(vis, tuple(np.round(uv[a] * s).astype(int)), tuple(np.round(uv[b] * s).astype(int)), (0, 0, 255),
                     1, cv2.LINE_AA)
    cv2.putText(vis, text, (4, th.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def stamp(r: dict) -> datetime:
    """Capture time from the thumbnail's file name (...20261023T200717Z.jpg)."""
    return datetime.strptime(Path(r["file"]).stem[-16:], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def per_hour(p: float) -> tuple[float, float, float]:
    lo, hi = 60 * p, 60 * (1 - (1 - p) ** 4)
    return lo, (lo + hi) / 2, hi


def main() -> None:
    from edge.m2_pipeline import M2Pipeline, kaaba_exclusion

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-kaaba", action="store_true", help="ptz path only (fast)")
    ap.add_argument("--no-keyframes", action="store_true", help="calibrated and extra references only")
    ap.add_argument("--no-verify", action="store_true", help="no check of ptz cameras on the Kaaba")
    ap.add_argument("--no-anchor", action="store_true", help="no rotation + focal length fits at calibrated positions")
    ap.add_argument("--anchor-gate", type=float, nargs=2, metavar=("RMS", "IOU"),
                    help="anchor gate for the upscaled thumbnails (default: the live 1080p gate)")
    ap.add_argument("--sheets", action="store_true", help="contact sheets of the counted frames (projected Kaaba)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "views" / "thumbs" / "replay_m2.csv")
    args = ap.parse_args()
    cfg = load_config()
    gm = graphics_mask(cfg, FULL)
    mp = M2Pipeline(cfg, use_kaaba=not args.no_kaaba, keyframes=not args.no_keyframes, verify=not args.no_verify,
                    anchor=not args.no_anchor, **({"anchor_gate": tuple(args.anchor_gate)} if args.anchor_gate else {}))
    tiles: dict[str, list] = {"day": [], "night": []}
    sift = ViewMatcher.from_config(cfg)
    rng = {v["id"]: zoom_range(v) for v in cfg["views"]}
    rows = list(csv.DictReader(open(ROOT / "data" / "views" / "thumbs" / "labels.csv")))
    rows.sort(key=stamp)
    out = []
    for r in rows:
        th = cv2.imread(str(ROOT / "data" / "views" / "thumbs" / r["file"]))
        img = cv2.resize(th, FULL, interpolation=cv2.INTER_CUBIC)
        res = {"file": r["file"], "period": "night" if r["night"] == "1" else "day", "label": r["label"], "path": "",
               "area": 0.0, "old": False}
        # current metric (SIFT + visible part of the fixed zone)
        m = sift.match(img, min_inliers=25)
        if m is not None and m.inliers >= 50:
            vis = sift.visibility(m, img.shape)
            main = max(vis.values(), key=lambda v: v["visible"])
            lo, hi = rng[m.view_id]
            res["old"] = main["visible"] >= MIN_VISIBLE and lo <= main["zoom"] <= hi
        # new metric
        fc = mp.camera(img, gm, t=stamp(r).timestamp())
        cam, kmask, res["path"] = fc.cam, fc.kaaba_mask, fc.path
        if "ptz_view" in fc.info:
            res["ptz"] = f"{fc.info['ptz_view']}:{fc.info['ptz_inliers']}:{fc.info['ptz_err']}:{fc.info['ptz_depth']}"
        res["kaaba_why"] = fc.info.get("fit_why", "")
        res["check"] = fc.info.get("ptz_check", "") + (f" anchor {fc.info['anchor_view'][:5]}" if fc.path == "anchor" else "")
        if "align_rec" in fc.info:
            res["align"] = f"{fc.info['align_rec']}:{fc.info['align_prec']}"
        res["ms_anchor"] = fc.info.get("ms_anchor", "")
        if cam is not None:
            ex = np.maximum(kmask if kmask is not None else kaaba_exclusion(cam, (FULL[1], FULL[0])), gm)
            res["area"] = round(ring_area(cam, ex), 1)
        res["new"] = bool(cam is not None and res["area"] >= MIN_AREA_M2)
        out.append(res)
        if args.sheets and res["new"] and len(tiles[res["period"]]) < 48:
            tiles[res["period"]].append(draw_box(th, cam, f"{stamp(r):%H:%M}Z {fc.path} {res.get('ptz', '')[:5]} "
                                                          f"{res['check']} {res.get('align', '')}"))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, ["file", "period", "label", "path", "ptz", "check", "align", "kaaba_why", "area", "old",
                              "new", "ms_anchor"], extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} thumbnails (labelled), current metric vs view-independent metric")
    for period in ("day", "night"):
        sel = [o for o in out if o["period"] == period]
        n = len(sel)
        k_old, k_new = sum(o["old"] for o in sel), sum(o["new"] for o in sel)
        paths = defaultdict(int)
        for o in sel:
            if o["new"]:
                paths[o["path"]] += 1
        lab = defaultdict(lambda: [0, 0])
        for o in sel:
            lab[o["label"]][0] += 1
            lab[o["label"]][1] += o["new"]
        print(f"  {period:5s} n={n}: current {k_old} ({100 * k_old / n:.0f} %, ~{per_hour(k_old / n)[1]:.1f} min/h), "
              f"new {k_new} ({100 * k_new / n:.0f} %, {per_hour(k_new / n)[0]:.1f}-{per_hour(k_new / n)[2]:.1f} "
              f"~{per_hour(k_new / n)[1]:.1f} min/h) by path {dict(paths)}; per label new/total "
              + ", ".join(f"{k} {v[1]}/{v[0]}" for k, v in sorted(lab.items())))
    if mp.ptz is not None:
        print("keyframes at the end:", mp.ptz.n_keyframes())
    checks = defaultdict(int)
    for o in out:
        checks[o.get("check") or "-"] += 1
    print("ptz checks:", dict(checks))
    for period, tl in tiles.items():
        for i in range(0, len(tl), 24):
            sheet = tl[i:i + 24]
            sheet += [np.zeros_like(sheet[0])] * (-len(sheet) % 4)
            path = ROOT / "data" / "views" / "discover" / f"replay_m2_{period}_{i // 24}.jpg"
            cv2.imwrite(str(path), np.vstack([np.hstack(sheet[j:j + 4]) for j in range(0, len(sheet), 4)]),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            print("sheet", path.relative_to(ROOT))
    print("per-frame table:", args.out)


if __name__ == "__main__":
    main()
