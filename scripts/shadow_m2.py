"""Shadow run of the view-independent metric on live frames, next to the current metric (research, 2026-10-24).

Reads the same broadcast as the crowd collector (shared yt-dlp state, so no extra YouTube lookups), never touches the
collector, the config or the database. Per frame (frames stay in memory; only numbers are written):
  current metric  CrowdPipeline(auto_capture=False): matched view, zoom, visible fraction, accepted, density / 10k px
  new metric      the frame's camera through edge/m2_pipeline.py, exactly as the crowd collector does it:
                    ptz    matched to a calibrated reference camera (ALIKED + LightGlue, edge/ptz.py; directly, through
                           an approved extra reference or a keyframe), checked on the Kaaba when it is in view
                           (projected box vs the OWLv2 + SAM 2.1 mask; align_rec / align_prec)
                    kaaba  otherwise, a camera fitted to the Kaaba's silhouette (edge/kaaba_geom.py), strict gate
                  -> people per m^2 in rings around the Kaaba (edge/m2_metric.py), raw and zoom-bias corrected, plus
                  the count and floor area per head-size bin (so the correction can be refitted later) and per
                  45-degree sector around the Kaaba (to compare cameras on the floor both see).
  view embedding  DINOv2-small CLS vector (view classifier, framing clusters), in m2_embed_<UTC date>.npy
Writes data/shadow/m2_frames.csv (a file with other columns, from an older version, is renamed first).
Run in the background:  .venv/Scripts/python scripts/shadow_m2.py [--hours 18]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # several GPU / CPU processes share the
    os.environ.setdefault(_v, "2")  # laptop: oversubscribed BLAS threads made the CPU fits 5-7x slower

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # models are in the local cache: no network calls per start

from edge import kaaba_geom as kg  # noqa: E402
from edge.m2_metric import ALL, G_FLOOR, HEAD_BINS, RINGS, cell_floor, measure, wall_distance  # noqa: E402
from edge.m2_pipeline import H0, KB, g, kaaba_exclusion  # noqa: E402
from edge.view_matcher import graphics_mask, load_config  # noqa: E402

OUT = ROOT / "data" / "shadow"
log = logging.getLogger("shadow_m2")


class Embedder:
    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        self.model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
        self.device = device

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        from PIL import Image

        small = cv2.resize(frame_bgr, (448, 252), interpolation=cv2.INTER_AREA)
        inp = self.proc(images=Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)), return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inp)
        return out.last_hidden_state[0, 0].float().cpu().numpy()


def debug_outline(path: Path, shape: tuple[int, int], cam: kg.Camera, mask: np.ndarray | None, text: str) -> None:
    """Lines only, on black (no frame pixels): Kaaba mask outline (green), the camera's Kaaba box (red) and the ring
    borders 1 / 10 / 25 m from the walls on the head plane (yellow)."""
    h, w = shape
    vis = np.zeros((h, w, 3), np.uint8)
    if mask is not None:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
    uv, z = cam.project(kg.box_corners())
    if (z > 0).all():
        for a, b in kg.BOX_EDGES:
            cv2.line(vis, tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[b]).astype(int)), (0, 0, 255), 1, cv2.LINE_AA)
    H = cam.plane_to_image(kg.HEAD_Z)
    for dist in (1.0, 10.0, 25.0):
        pts = []
        for a in np.linspace(0, 2 * np.pi, 240):
            cx, cy = np.cos(a), np.sin(a)
            p = H @ [np.sign(cx) * kg.KAABA_L / 2 + dist * cx, np.sign(cy) * kg.KAABA_W / 2 + dist * cy, 1]
            if p[2] > 0:
                pts.append(p[:2] / p[2])
        if len(pts) > 2:
            cv2.polylines(vis, [np.round(np.clip(pts, -1e4, 1e4)).astype(np.int32).reshape(-1, 1, 2)], False,
                          (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.resize(vis, (w // 2, h // 2), interpolation=cv2.INTER_AREA))


def ring_columns(cam: kg.Camera, dens: np.ndarray, ex: np.ndarray, rings: list[str]) -> tuple[list, list]:
    rd = measure(cam, dens, ex, g)
    vals = []
    for rg in rings:
        vals += [rd.density.get(rg, ""), rd.density_corr.get(rg, ""), rd.count.get(rg, ""), rd.area.get(rg, ""),
                 rd.head_px.get(rg, "")]
    geo = cell_floor(cam, dens.shape)
    dist = wall_distance(geo["X"], geo["Y"])
    dh, dw = dens.shape
    exs = ex[: dh * 8, : dw * 8].reshape(dh, 8, dw, 8).mean((1, 3)) < 0.5
    hp = cam.f * 0.22 / np.maximum(geo["depth"], 1e-3)
    bins = []
    for r0, r1 in [RINGS[r] if r in RINGS else ALL for r in rings]:
        sel = geo["valid"] & exs & (dist >= r0) & (dist < r1)
        cnt, _ = np.histogram(hp[sel], HEAD_BINS, weights=dens[sel])
        ar, _ = np.histogram(hp[sel], HEAD_BINS, weights=geo["area"][sel])
        bins.append(json.dumps([[round(float(a), 1) for a in cnt], [round(float(a), 1) for a in ar]]))
    # 8 sectors of 45 degrees around the Kaaba (all rings): corrected count and area, so readings of different cameras
    # can be compared on the floor both see
    sel = geo["valid"] & exs & (dist >= ALL[0]) & (dist < ALL[1])
    sec = ((np.degrees(np.arctan2(geo["Y"], geo["X"])) % 360) // 45).astype(int)
    corr = dens / np.maximum(g(hp), G_FLOOR)
    sc = np.bincount(sec[sel], weights=corr[sel], minlength=8)[:8]
    sa = np.bincount(sec[sel], weights=geo["area"][sel], minlength=8)[:8]
    bins.append(json.dumps([[round(float(a), 2) for a in sc], [round(float(a), 1) for a in sa]]))
    return vals, bins


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hours", type=float, default=18)
    ap.add_argument("--interval", type=float, default=15.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "data" / "logs" / "shadow_m2.log", encoding="utf-8"),
                                  logging.StreamHandler()])
    for name in ("httpx", "httpx2", "httpcore", "httpcore2", "transformers", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)

    import torch

    from edge.crowd_model import CrowdModel
    from edge.frame_grabber import FrameGrabber
    from edge.m2_pipeline import M2Pipeline
    from edge.pipeline import CrowdPipeline

    cfg = load_config()
    model = CrowdModel()
    pipe = CrowdPipeline(model, auto_capture=False)
    mp = M2Pipeline(cfg, model, use_kaaba=True, keyframes=True, verify=True)  # the collector's code path
    emb = Embedder()
    log.info("shadow run: cameras %s, %d references, zoom bias h0 %.2f k %.2f", list(mp.ptz.cams), len(mp.ptz.bank),
             H0, KB)
    csv_path = OUT / "m2_frames.csv"
    rings = list(RINGS) + ["all"]
    info_cols = ["ptz_view", "ptz_inliers", "ptz_err", "ptz_zoom", "ptz_H", "ptz_depth", "ptz_check", "align_rec",
                 "align_prec", "anchor_view", "anchor_try", "anchor_rms", "anchor_iou", "anchor_f", "kaaba_score",
                 "sam_iou", "dark", "fit_ok", "fit_why", "f", "f_lo", "f_hi", "rms_px", "iou", "z_occ"]
    header = (["ts_utc", "source", "old_view", "old_zoom", "old_visible", "old_accepted", "old_density", "path"]
              + info_cols + ["cam_f", "cam_x", "cam_y", "cam_z"]
              + [f"{k}_{r}" for r in rings for k in ("m2", "m2c", "count", "area", "headpx")]
              + [f"bins_{r}" for r in rings] + ["sectors_all", "raw_count", "raw_per_10kpx"]
              + ["ms_ptz", "ms_detect", "ms_sam", "ms_fit", "ms_model", "ms_total", "vram_gb", "emb_row"])
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            old_header = next(csv.reader(f), [])
        if old_header != header:  # an older version's file: keep it apart
            csv_path.rename(OUT / f"m2_frames_{time.strftime('%Y%m%dT%H%M%S')}.csv")
    new = not csv_path.exists()
    fh = open(csv_path, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(header)
    deadline = time.monotonic() + args.hours * 3600
    emb_rows: list[np.ndarray] = []
    emb_day = None
    n = n_debug = 0
    for frame in FrameGrabber.from_env(interval_s=args.interval).frames():
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        img = frame.image
        h, wd = img.shape[:2]
        try:
            reading, m, _ = pipe.process(frame)
        except Exception:
            log.exception("current pipeline failed")
            reading, m = None, None
        old = (m.view_id if m else "", f"{reading.zoom:.3f}" if reading else "",
               f"{max(v['visible'] for v in reading.zones.values()):.3f}" if reading else "",
               int(bool(reading and reading.accepted)),
               f"{max(v['density'] for v in reading.zones.values()):.3f}" if reading else "")
        gm = graphics_mask(cfg, (wd, h))
        info, cam, path, ms_model = {}, None, "", ""
        raw = ["", ""]
        ring_vals, bins = [""] * (5 * len(rings)), [""] * (len(rings) + 1)
        try:
            fc = mp.camera(img, gm, t=frame.captured_at.timestamp())
            info, cam, path = fc.info, fc.cam, fc.path
            tm = time.perf_counter()
            dens = model.predict(img).density
            ms_model = round((time.perf_counter() - tm) * 1e3)
            if cam is not None:
                ex = np.maximum(fc.kaaba_mask if fc.kaaba_mask is not None else kaaba_exclusion(cam, (h, wd)), gm)
                ring_vals, bins = ring_columns(cam, dens, ex, rings)
                if n_debug < 120 and n % 3 == 0:
                    debug_outline(OUT / "debug" / f"{frame.captured_at:%Y%m%dT%H%M%S}_{path}.png", (h, wd), cam,
                                  fc.kaaba_mask, f"{path} {info.get('ptz_view', '')} {info.get('ptz_check', '')} "
                                                 f"f {cam.f:.0f} cam {cam.centre.round(0).tolist()} m2c {ring_vals[-4]}")
                    n_debug += 1
            else:  # no camera: the raw count, for the cross-calibration of framings without the Kaaba
                dh, dw = dens.shape
                free = 1 - gm[: dh * 8, : dw * 8].reshape(dh, 8, dw, 8).mean((1, 3))
                raw = [round(float(dens.sum()), 1), round(float((dens * free).sum() / max(free.sum() * 64, 1) * 1e4), 3)]
        except Exception:
            log.exception("new metric failed")
        if isinstance(info.get("ptz_H"), np.ndarray):
            H = info["ptz_H"]
            info["ptz_H"] = json.dumps([float(f"{x:.7g}") for x in (H / H[2, 2]).ravel()])
        cam_cols = ([round(cam.f, 1)] + [round(float(x), 2) for x in cam.centre]) if cam is not None else [""] * 4
        e = emb(img)
        day = frame.captured_at.strftime("%Y%m%d")
        if day != emb_day:
            emb_rows, emb_day = [], day
        emb_rows.append(e)
        np.save(OUT / f"m2_embed_{day}.npy", np.vstack(emb_rows).astype(np.float16))
        total = round((time.perf_counter() - t0) * 1e3)
        w.writerow([frame.captured_at.isoformat(timespec="seconds"), frame.source, *old, path,
                    *[info.get(k, "") for k in info_cols], *cam_cols, *ring_vals, *bins, *raw,
                    info.get("ms_ptz", ""), info.get("ms_detect", ""), info.get("ms_sam", ""),
                    info.get("ms_anchor", "") or info.get("ms_fit", ""),
                    ms_model, total, round(torch.cuda.max_memory_allocated() / 1e9, 2), f"{day}:{len(emb_rows) - 1}"])
        fh.flush()
        n += 1
        if n % 40 == 0:
            log.info("frames %d, keyframes %s", n, mp.ptz.n_keyframes())
        del frame, img
        if time.monotonic() > deadline:
            break


if __name__ == "__main__":
    main()
