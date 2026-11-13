"""Zoom bias of the crowd model in people per m^2, and its correction from the head size (research, 2026-10-24).

The crowd model finds more people when they look bigger. With the camera known (config/cameras.json), every density
cell has a floor position and a head size in px, so the bias can be measured and undone:
  1. each calibrated reference frame is digitally zoomed about its centre (0.8 .. 2.0x); the zoomed camera is exact
     (focal length x s, same pose);
  2. people per 2 m x 2 m floor tile (head plane, 1-25 m from the Kaaba's walls) are compared with the unzoomed frame
     on the tiles both show: ratio(s) = g(s * h) / g(h), h = the tile's head width in px at zoom 1;
  3. g(h) = 1 - exp(-(h / h0)^k) (the share of people the model finds at head width h; -> 1 for big heads) is fitted
     to all tiles, scales and cameras;
  4. check: people / m^2 on the common area, raw and corrected (each cell / g(its head size)), across zoom.
Writes models/zoom_bias.json (h0, k, the drift tables) and data/views/2d/zoom_bias.png. Frames stay in memory.
--g H0 K: no fit, the drift tables for that g only (e.g. a refit on live frames, scripts/zoom_bias_live.py); prints
them and writes data/views/2d/zoom_bias_g.json / .png, models/zoom_bias.json untouched.

  .venv/Scripts/python scripts/zoom_bias.py [--g 6.5 1.2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from edge.kaaba_geom import Camera  # noqa: E402
from edge.m2_metric import ALL, HEAD_M, cell_floor, wall_distance  # noqa: E402
from edge.view_matcher import graphics_mask, load_config  # noqa: E402

SCALES = (0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0)
TILE = 2.0  # m
OUT_JSON = ROOT / "models" / "zoom_bias.json"


def zoom_frame(img: np.ndarray, s: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def tiles(cam: Camera, dens: np.ndarray, valid_px: np.ndarray) -> dict:
    """{(i, j): [count, head px weighted by area, area]} per floor tile, cells fully valid only."""
    geo = cell_floor(cam, dens.shape)
    dh, dw = dens.shape
    ok = geo["valid"] & (valid_px[: dh * 8, : dw * 8].reshape(dh, 8, dw, 8).mean((1, 3)) > 0.99)
    d = wall_distance(geo["X"], geo["Y"])
    ok &= (d >= ALL[0]) & (d < ALL[1])
    hp = cam.f * HEAD_M / np.maximum(geo["depth"], 1e-3)
    ti = np.floor(geo["X"] / TILE).astype(int)
    tj = np.floor(geo["Y"] / TILE).astype(int)
    out: dict = {}
    for a, b, c, h_, ar in zip(ti[ok], tj[ok], dens[ok], hp[ok], geo["area"][ok]):
        t = out.setdefault((a, b), [0.0, 0.0, 0.0])
        t[0] += c
        t[1] += h_ * ar
        t[2] += ar
    return out


def g_fn(h: np.ndarray, h0: float, k: float) -> np.ndarray:
    return 1.0 - np.exp(-np.power(np.maximum(h, 0.05) / h0, k))


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.optimize import least_squares

    from edge.crowd_model import CrowdModel

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--g", type=float, nargs=2, metavar=("H0", "K"), help="evaluate this g instead of fitting one")
    args = ap.parse_args()
    cfg = load_config()
    reg = json.loads((ROOT / "config" / "cameras.json").read_text())["cameras"]
    model = CrowdModel()
    obs = []  # (camera, s, h1, ratio, weight)
    per_cam = {}
    for vid, c in reg.items():
        img = cv2.imread(str(ROOT / "data" / "views" / "refs" / f"{vid}.png"))
        h, w = img.shape[:2]
        cam1 = Camera(c["f"], np.array(c["rvec"]), np.array(c["tvec"]), tuple(c["size"]))
        gm = graphics_mask(cfg, (w, h))
        base = None
        runs = {}
        for s in SCALES:
            frame = zoom_frame(img, s)
            valid = zoom_frame(np.ones((h, w), np.uint8), s) & (1 - gm)  # inside the zoomed picture, not graphics
            cam = Camera(cam1.f * s, cam1.rvec, cam1.tvec, cam1.size)
            dens = model.predict(frame).density
            runs[s] = (cam, dens, tiles(cam, dens, valid))
        base = runs[1.0][2]
        for s, (cam, dens, t) in runs.items():
            if s == 1.0:
                continue
            for key, (cnt, hw, ar) in t.items():
                b = base.get(key)
                if b is None or b[0] < 3 or cnt < 1 or abs(ar / b[2] - 1) > 0.05:  # same, fully seen tile
                    continue
                obs.append((vid, s, b[1] / b[2], cnt / b[0], b[0]))
        per_cam[vid] = runs
        print(f"{vid}: {len(base)} tiles at zoom 1, head width {np.median([v[1] / v[2] for v in base.values()]):.1f} px")
    o = np.array([(s, h1, r, wgt) for _, s, h1, r, wgt in obs])

    def res(p):
        h0, k = np.exp(p)
        pred = g_fn(o[:, 0] * o[:, 1], h0, k) / g_fn(o[:, 1], h0, k)
        return (np.log(o[:, 2]) - np.log(pred)) * np.sqrt(o[:, 3])

    if args.g:
        h0, k = args.g
    else:
        fit = least_squares(res, np.log([4.0, 1.2]), loss="soft_l1", f_scale=0.3)
        h0, k = np.exp(fit.x)
    print(f"g(h) = 1 - exp(-(h / {h0:.2f})^{k:.2f})  from {len(o)} tile x zoom pairs; g(3) {g_fn(3, h0, k):.2f}, "
          f"g(5) {g_fn(5, h0, k):.2f}, g(8) {g_fn(8, h0, k):.2f}, g(12) {g_fn(12, h0, k):.2f}")
    report = {"h0": round(float(h0), 3), "k": round(float(k), 3), "n_pairs": int(len(o)), "cameras": {}}
    fig, axes = plt.subplots(1, len(per_cam), figsize=(5.5 * len(per_cam), 4), squeeze=False)
    for ax, (vid, runs) in zip(axes[0], per_cam.items()):
        common = set.intersection(*[set(k_ for k_, v in r[2].items() if v[2] > 0) for r in runs.values()])
        rows = []
        for s, (cam, dens, t) in sorted(runs.items()):
            geo = cell_floor(cam, dens.shape)
            hp = cam.f * HEAD_M / np.maximum(geo["depth"], 1e-3)
            gg = g_fn(hp, h0, k)
            ti = np.floor(geo["X"] / TILE).astype(int)
            tj = np.floor(geo["Y"] / TILE).astype(int)
            inside = np.array([(a, b) in common for a, b in zip(ti.ravel(), tj.ravel())]).reshape(ti.shape)
            dh, dw = dens.shape
            h_, w_ = cam.size[1], cam.size[0]
            valid = zoom_frame(np.ones((h_, w_), np.uint8), s) & (1 - graphics_mask(cfg, (w_, h_)))
            ok = inside & geo["valid"] & (valid[: dh * 8, : dw * 8].reshape(dh, 8, dw, 8).mean((1, 3)) > 0.99)
            area = geo["area"][ok].sum()
            raw = dens[ok].sum() / area
            cor = (dens[ok] / np.maximum(gg[ok], 0.15)).sum() / area
            rows.append((s, raw, cor, float(np.median(hp[ok]))))
        r1 = next(r for r in rows if r[0] == 1.0)
        tab = [{"zoom": s, "raw_drift": round(raw / r1[1] - 1, 4), "corrected_drift": round(cor / r1[2] - 1, 4),
                "head_px": round(hpx, 2), "people_m2_raw": round(raw, 3), "people_m2_corrected": round(cor, 3)}
               for s, raw, cor, hpx in rows]
        report["cameras"][vid] = tab
        print(f"\n{vid}: common area, people/m2 raw -> corrected (drift vs zoom 1)")
        for t_ in tab:
            print(f"  zoom {t_['zoom']:4.2f} head {t_['head_px']:5.1f} px  raw {t_['people_m2_raw']:.3f} ({100 * t_['raw_drift']:+5.1f} %)"
                  f"  corrected {t_['people_m2_corrected']:.3f} ({100 * t_['corrected_drift']:+5.1f} %)")
        ax.plot([t_["zoom"] for t_ in tab], [100 * t_["raw_drift"] for t_ in tab], "o-", label="raw")
        ax.plot([t_["zoom"] for t_ in tab], [100 * t_["corrected_drift"] for t_ in tab], "s-", label="corrected")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{vid}: people/m2 vs digital zoom"), ax.set_xlabel("zoom"), ax.set_ylabel("drift %"), ax.legend()
    fig.tight_layout()
    out = ROOT / "data" / "views" / "2d" / ("zoom_bias_g.json" if args.g else "zoom_bias.json")
    fig.savefig(out.with_suffix(".png"), dpi=90)
    out = out if args.g else OUT_JSON
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print("written", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
