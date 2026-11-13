"""Accuracy probe of the Mataf metric on saved frames (2026-10-26, investigation; nothing here changes the live metric).

For each m2 sample frame (data/samples/m2_<camera>_<UTC>.png + its sidecar camera) it recomputes, with the live code:
  per ring (inner 1-10 m, outer 10-25 m, all): raw DM-Count count, floor area (m^2), people per m^2 raw and corrected,
  the mean head width (px) and the factor the head-size correction g(h) applies;
  per inner-ring sector: the same, so a packed band at the Kaaba is not averaged away;
  mass check: people in the dashboard heatmap (edge.heatmap.density_png -> backend.floormap.image_density) vs DM-Count;
  cross-checks on the same floor: DM-Count on 2x upscaled tiles ("tiled"), APGCC points at 1x and 2x.
  brightness of the floor (mean grey), day / evening / night.
Writes data/samples/probe/probe.csv (numbers) and, with --overlays, an overlay per frame (the frame darkened, the 1 / 10
/ 25 m rings, a 5 m grid on the floor, the Kaaba and the Maqam candidates projected) -- local only, never committed.

Runs beside the live collector: GPU memory capped at 25 %, one frame at a time.
  .venv/Scripts/python -I scripts/accuracy_probe.py [--since 20261026T14] [--per-camera 6] [--overlays]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge import kaaba_geom as kg  # noqa: E402
from edge.heatmap import density_png, heat_scale  # noqa: E402
from edge.m2_metric import ALL, HEAD_M, RINGS, SECTORS, cell_floor, load_correction, wall_distance  # noqa: E402
from edge.m2_pipeline import g, kaaba_exclusion  # noqa: E402
from edge.view_matcher import graphics_mask, load_config  # noqa: E402

SAMPLES = ROOT / "data" / "samples"
OUT = SAMPLES / "probe"


def tiled_density(model, img: np.ndarray, shape: tuple[int, int], grid=(3, 3), up=2.0) -> np.ndarray:
    """DM-Count on grid tiles, each upscaled `up` x, summed back onto the frame's density grid `shape` (rows, cols)."""
    h, w = img.shape[:2]
    dh, dw = shape
    out = np.zeros((dh, dw), np.float64)
    for i in range(grid[0]):
        for j in range(grid[1]):
            y0, y1 = round(i * h / grid[0]), round((i + 1) * h / grid[0])
            x0, x1 = round(j * w / grid[1]), round((j + 1) * w / grid[1])
            tile = cv2.resize(img[y0:y1, x0:x1], None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
            d = model.predict(tile).density  # people per 8 x 8 px of the upscaled tile
            r0, r1 = round(i * dh / grid[0]), round((i + 1) * dh / grid[0])
            c0, c1 = round(j * dw / grid[1]), round((j + 1) * dw / grid[1])
            # back to the frame's cells: the tile's total preserved, spread as its map shows
            cells = cv2.resize(d.astype(np.float32), (c1 - c0, r1 - r0), interpolation=cv2.INTER_AREA)
            cells *= d.sum() / max(cells.sum(), 1e-9)
            out[r0:r1, c0:c1] += cells
    return out


def points_to_cells(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, np.float64)
    if len(points):
        j = np.clip((points[:, 0] // 8).astype(int), 0, shape[1] - 1)
        i = np.clip((points[:, 1] // 8).astype(int), 0, shape[0] - 1)
        np.add.at(m, (i, j), 1.0)
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", default="", help="only samples whose UTC stamp starts at or after this (e.g. 20261026T14)")
    ap.add_argument("--per-camera", type=int, default=0, help="newest N per camera (0: all matching)")
    ap.add_argument("--overlays", action="store_true")
    ap.add_argument("--no-apgcc", action="store_true")
    ap.add_argument("--out", default="probe.csv", help="file name in data/samples/probe/")
    args = ap.parse_args()
    from backend import floormap
    from edge.crowd_model import CrowdModel

    model = CrowdModel()
    apg = None
    if not args.no_apgcc:
        from edge.apgcc_model import PointCounter

        apg = PointCounter()
    cfg = load_config()
    cams = json.loads((ROOT / "config" / "cameras.json").read_text(encoding="utf-8"))["cameras"]
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in SAMPLES.glob("m2_*.json") if p.stem.rsplit("_", 1)[1] >= args.since)
    if args.per_camera:
        by: dict = {}
        for p in files:
            by.setdefault(json.loads(p.read_text())["camera"], []).append(p)
        files = sorted(q for v in by.values() for q in v[-args.per_camera:])
    rows = []
    for side in files:
        meta = json.loads(side.read_text())
        img = cv2.imread(str(side.with_suffix(".png")))
        if img is None:
            continue
        h, w = img.shape[:2]
        c = meta["cam"]
        cam = kg.Camera(c["f"], np.asarray(c["rvec"], float), np.asarray(c["tvec"], float), tuple(c["size"]))
        dens = model.predict(img).density
        geo = cell_floor(cam, dens.shape)
        ex = np.maximum(kaaba_exclusion(cam, (h, w)), graphics_mask(cfg, (w, h)))
        dh, dw = dens.shape
        exc = ex[: dh * 8, : dw * 8].reshape(dh, 8, dw, 8).mean((1, 3)) >= 0.5
        ok = geo["valid"] & ~exc
        dist = wall_distance(geo["X"], geo["Y"])
        hp = cam.f * HEAD_M / np.maximum(geo["depth"], 1e-3)
        gv = np.maximum(g(hp), 0.2)
        sec = (np.floor((np.arctan2(geo["Y"], geo["X"]) + np.pi) / (2 * np.pi / SECTORS)).astype(int)) % SECTORS
        tiled = tiled_density(model, img, dens.shape)
        ap1 = points_to_cells(apg.predict(img).points, dens.shape) if apg else None
        ap2 = points_to_cells(apg.predict(img, scale=2.0).points, dens.shape) if apg else None
        # the dashboard's heatmap mass over the same floor
        tmp = OUT / "_heat.png"
        tmp.write_bytes(density_png(dens, (w, h)))
        H = cam.plane_to_image(kg.HEAD_Z) @ np.linalg.inv(cam.plane_to_image(kg.HEAD_Z))  # identity: frame = reference
        hm = {"h_primary": (H / H[2, 2]).tolist(), "density_top": heat_scale(dens, (w, h)), "frame_size": [w, h]}
        im = floormap.image_density(tmp, hm, c, cfg, *load_correction()[:2])
        floor_px = ok & (dist >= ALL[0]) & (dist < ALL[1])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask_up = cv2.resize(floor_px.astype(np.uint8), (dw * 8, dh * 8), interpolation=cv2.INTER_NEAREST) > 0
        bright = float(gray[: dh * 8, : dw * 8][mask_up].mean()) if mask_up.any() else float("nan")
        base = {"frame": side.stem, "camera": meta["camera"], "period": meta["period"], "path": meta["path"],
                "zoom": meta.get("zoom"), "floor_brightness": round(bright, 1)}
        for name, (r0, r1) in {**RINGS, "all": ALL}.items():
            sel = ok & (dist >= r0) & (dist < r1)
            a = float(geo["area"][sel].sum())
            if a < 40:
                continue
            raw = float(dens[sel].sum())
            cor = float((dens[sel] / gv[sel]).sum())
            rows.append({**base, "region": name, "area_m2": round(a, 1), "count_raw": round(raw, 1),
                         "m2_raw": round(raw / a, 3), "m2_corr": round(cor / a, 3), "head_px": round(float(np.median(hp[sel])), 2),
                         "g_factor": round(cor / max(raw, 1e-9), 2), "tiled_raw": round(float(tiled[sel].sum()), 1),
                         "apgcc_1x": round(float(ap1[sel].sum()), 1) if ap1 is not None else "",
                         "apgcc_2x": round(float(ap2[sel].sum()), 1) if ap2 is not None else "",
                         "heat_mass_ratio": ""})
        # inner-ring sectors with enough floor: is there a packed band the average hides?
        for s in range(SECTORS):
            sel = ok & (dist >= 1) & (dist < 10) & (sec == s)
            a = float(geo["area"][sel].sum())
            if a < 15:
                continue
            raw = float(dens[sel].sum())
            rows.append({**base, "region": f"i{s}", "area_m2": round(a, 1), "count_raw": round(raw, 1),
                         "m2_raw": round(raw / a, 3), "m2_corr": round(float((dens[sel] / gv[sel]).sum()) / a, 3),
                         "head_px": round(float(np.median(hp[sel])), 2),
                         "g_factor": round(float((dens[sel] / gv[sel]).sum()) / max(raw, 1e-9), 2),
                         "tiled_raw": round(float(tiled[sel].sum()), 1),
                         "apgcc_1x": round(float(ap1[sel].sum()), 1) if ap1 is not None else "",
                         "apgcc_2x": round(float(ap2[sel].sum()), 1) if ap2 is not None else "", "heat_mass_ratio": ""})
        # heatmap mass: people the heatmap path finds on the floor vs DM-Count on the same pixels (both uncorrected)
        if im is not None:
            hh, hw = im["dens"].shape
            people_heat = aiview_people(tmp, hm["density_top"], (w, h))
            floor_heat = cv2.resize(im["floor"].astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST) > 0
            raw_same = float(dens[floor_heat].sum())
            heat_same = float(cv2.resize(people_heat.astype(np.float32), (dw, dh), interpolation=cv2.INTER_AREA)[floor_heat].sum()
                              * (hw * hh) / (dw * dh))
            for r in rows:
                if r["frame"] == side.stem and r["region"] == "all":
                    r["heat_mass_ratio"] = round(heat_same / max(raw_same, 1e-9), 3)
        tmp.unlink(missing_ok=True)
        if args.overlays:
            overlay(img, cam, dist, ok, dens.shape, OUT / f"{side.stem}_rings.jpg", meta)
        print(f"{side.stem}: done", flush=True)
    if rows:
        with open(OUT / args.out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(f"wrote {OUT / args.out} ({len(rows)} rows)")


def aiview_people(png: Path, top: float, frame_size: tuple[int, int]) -> np.ndarray:
    """People per heatmap pixel as the dashboard reads them back from the PNG (JET inverted, clipped at the p99 scale)."""
    from backend import aiview

    v, _ = aiview.density_values(cv2.imread(str(png)))
    s = frame_size[0] / v.shape[1]
    return (v.astype(float) / 255.0) * float(top) / 64.0 * s * s


def overlay(img: np.ndarray, cam: kg.Camera, dist: np.ndarray, ok: np.ndarray, dshape, out: Path, meta: dict) -> None:
    h, w = img.shape[:2]
    o = (img.astype(np.float32) * 0.55).astype(np.uint8)
    dh, dw = dshape
    d_up = cv2.resize(np.where(ok, dist, np.nan).astype(np.float32), (dw * 8, dh * 8), interpolation=cv2.INTER_LINEAR)
    for r, col in ((1, (255, 255, 255)), (10, (80, 220, 255)), (25, (80, 160, 255))):
        m = (np.abs(np.nan_to_num(d_up, nan=-99) - r) < 0.25).astype(np.uint8)
        o[: dh * 8, : dw * 8][m > 0] = col
    # 5 m grid on the head plane (X, Y lines), drawn by projecting points
    for k in range(-30, 31, 5):
        for axis in (0, 1):
            pts = np.array([[k, t] if axis == 0 else [t, k] for t in np.linspace(-30, 30, 121)], float)
            P = np.column_stack([pts, np.full(len(pts), kg.HEAD_Z)])
            uv, z = cam.project(P)
            uv = uv[z > 1]
            if len(uv) > 1:
                cv2.polylines(o, [np.int32(uv)], False, (120, 120, 120), 1, cv2.LINE_AA)
    sil = kg.silhouette(cam)
    if sil is not None:
        cv2.polylines(o, [np.int32(sil[0])], True, (255, 255, 255), 2, cv2.LINE_AA)
    # Maqam Ibrahim: in front of the door wall (one of the two long walls; the frame is symmetric under a half turn)
    for sgn in (1, -1):
        for xm in (-kg.KAABA_L / 2 + 3.8, kg.KAABA_L / 2 - 3.8):
            uv, z = cam.project(np.array([[xm, sgn * (kg.KAABA_W / 2 + 11.0), 1.0]]))
            if z[0] > 1:
                cv2.circle(o, (int(uv[0, 0]), int(uv[0, 1])), 9, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(o, f"{meta['camera']} {meta['makkah_time']}  rings 1 / 10 / 25 m, grid 5 m", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out), cv2.resize(o, (w // 2, h // 2)), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
