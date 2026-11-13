"""Training data for fine-tuning DM-Count on our own frames (user request 2026-10-27: far crowds with heads under 4 px
read low; labels from the prayer rows and the zoomed frames, test on held-out prayers, deploy only after asking).

Input: the saved DVR frames (data/samples/dvr, YouTube 1080p only; SBA frames are left out). Each window (a prayer or
the tawaf sample) runs through the live shot analysis (edge/shot.py, Kaaba gate on), so every frame gets the camera
the live service would find. For every placed frame this stores, in data/finetune/<window>.npz + .json:
  - the camera (f, rvec, tvec, size) and DM-Count's density map (the live model, float16, 1/8 resolution);
  - rows labels: the prayer-rows estimator (edge/prayer_rows.py) run LOCALLY, along each line from the Kaaba in
    segments of SEG_M metres out to the courtyard's edge: where a segment shows a strong row pattern, people per m^2 =
    rows per metre / ALONG_M there. A label is a floor strip (wall distance d0-d1, +-STRIP_HALF_M across the line).
Zoom labels (a zoomed-in frame of the same floor within minutes) are paired later, in scripts/finetune_dmcount.py,
from the stored maps. Frames stay on this PC; nothing leaves it.

  .venv/Scripts/python scripts/finetune_data.py [--windows isha-2026-10-26 ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from edge import prayer_rows as pr  # noqa: E402
from edge.kept_frames import frame_file, keep  # noqa: E402
from edge.m2_metric import courtyard_edge, wall_distance  # noqa: E402

DVR = ROOT / "data" / "samples" / "dvr"
OUT = ROOT / "data" / "finetune"
SEG_M = 6.0  # m of line per rows label (about five rows)
STRIP_HALF_M = 1.0  # a label's strip: +- this across the line
ROWS_MIN_STRENGTH = 0.3  # autocorrelation at the row pitch for a label (prayer rows 0.3-0.8, tawaf < 0.2)
BEARING_STEP = 5.0


def windows() -> dict[str, list[str]]:
    """window name -> sorted frame stems, from the DVR index (tags like "2026-10-26 isha"); SBA frames left out."""
    out: dict[str, list[str]] = defaultdict(list)
    with (DVR / "index.csv").open() as f:
        for r in csv.DictReader(f):
            tag, res = r["tag"].strip(), r["result"]
            if "sba" in tag or "sba" in res or not res.startswith("1920x1080"):
                continue
            name = tag.replace("(after)", "").replace("-test", "").strip()
            if name.startswith("maghrib"):
                name = "2026-10-26 maghrib"
            out[name.replace(" ", "-").replace(":", "")].append(Path(r["file"]).stem)
    return {k: sorted(set(v)) for k, v in out.items()}


def rows_segments(gray: np.ndarray, cam, blocked: np.ndarray) -> list[dict]:
    """The rows estimator along every line from the Kaaba, per SEG_M segment: [{b, d0, d1, strength, pitch, dens,
    px_per_m}] for segments with a row pattern of at least ROWS_MIN_STRENGTH."""
    h, w = gray.shape[:2]
    g = gray.astype(np.float32)
    H = cam.plane_to_image(pr.ROW_Z)
    i0, i1 = int(pr.PITCH_RANGE[0] / pr.STEP), int(pr.PITCH_RANGE[1] / pr.STEP)
    out = []
    for b in np.arange(0, 360, BEARING_STEP):
        a = np.radians(b)
        r_edge = float(courtyard_edge(np.array(a)))
        d_edge = float(wall_distance(np.array(r_edge * np.cos(a)), np.array(r_edge * np.sin(a))))
        d = np.arange(pr.SPAN[0], d_edge, pr.STEP)
        if len(d) < 10:
            continue
        xy = pr._ray_points(d, b)
        t = np.array([-np.sin(a), np.cos(a)])
        prof, uv0 = [], None
        for o in (-0.2, 0.0, 0.2):
            q = np.column_stack([xy + o * t, np.ones(len(xy))]) @ H.T
            u, v = q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]
            if o == 0.0:
                uv0 = (u, v, q[:, 2])
            prof.append(cv2.remap(g, u.astype(np.float32).reshape(1, -1), v.astype(np.float32).reshape(1, -1),
                                  cv2.INTER_LINEAR).ravel())
        u, v, z = uv0
        ok = (z > 0) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        iu, iv = np.clip(u.astype(int), 0, w - 1), np.clip(v.astype(int), 0, h - 1)
        ok &= blocked[iv, iu] == 0
        runs = np.flatnonzero(np.diff(np.concatenate([[0], ok.astype(int), [0]])))
        for a0, e0 in zip(runs[::2], runs[1::2]):
            if (e0 - a0) * pr.STEP < SEG_M:
                continue
            p = np.mean(prof, 0)[a0:e0].astype(np.float64)
            p = p - cv2.GaussianBlur(p.reshape(-1, 1), (0, 0), 1.5 / pr.STEP).ravel()
            p = cv2.GaussianBlur(p.reshape(-1, 1), (0, 0), 0.06 / pr.STEP).ravel()
            # the rows (bright crests) found along the whole usable line, so rows at a segment's ends still count
            ac_all = pr._autocorr(p)
            j_all = i0 + int(np.argmax(ac_all[i0:i1])) if len(ac_all) > i1 else i0
            peaks, _ = find_peaks(p, distance=max(int(0.7 * j_all), 1), prominence=0.5 * p.std())
            n = int(SEG_M / pr.STEP)
            for s in range(0, len(p) - n + 1, n):
                ps = p[s:s + n]
                j0, j1 = a0 + s, a0 + s + n - 1
                px_per_m = float(np.hypot(u[j1] - u[j0], v[j1] - v[j0]) / ((n - 1) * pr.STEP))
                if px_per_m * pr.PITCH_RANGE[0] < pr.MIN_PX_PER_ROW:
                    continue
                ac = pr._autocorr(ps)
                if len(ac) <= i1:
                    continue
                j = i0 + int(np.argmax(ac[i0:i1]))
                st = float(ac[j])
                if st < ROWS_MIN_STRENGTH:
                    continue
                per_m = int(((peaks >= s) & (peaks < s + n)).sum()) / (n * pr.STEP)
                out.append({"b": float(b), "d0": round(float(d[j0]), 2), "d1": round(float(d[j1]), 2),
                            "strength": round(st, 3), "pitch": round(j * pr.STEP, 3),
                            "dens": round(per_m / pr.ALONG_M, 3), "px_per_m": round(px_per_m, 1)})
    return out


class Clock:
    """time for edge/shot.py: monotonic() = the frame's timestamp, perf_counter() real (as scripts/gate_check.py)."""
    import time as _t
    now = 0.0
    monotonic = staticmethod(lambda: Clock.now)
    perf_counter = staticmethod(_t.perf_counter)
    time = staticmethod(_t.time)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", nargs="*", help="window names (default: all); see --list")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--range", help="only frames with stems in A-B (a quick test; saved as <window>-test)")
    args = ap.parse_args()
    wins = windows()
    if args.list:
        for k, v in wins.items():
            print(f"{k:28s} {len(v):4d} frames  {v[0]} - {v[-1]}")
        return
    import torch

    from edge import kaaba_geom as kg
    from edge import shot as shot_mod
    from edge.crowd_model import CrowdModel
    from edge.shot import ShotAnalyzer
    from edge.shot_bank import Bank
    from edge.view_matcher import graphics_mask, load_config

    shot_mod.time = Clock
    cfg = load_config()
    model = CrowdModel()
    bank = Bank(ROOT / "data" / "vloc" / "bank.npz")
    gm = graphics_mask(cfg, (1920, 1080))
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.windows or list(wins):
        stems = wins[name]
        if args.range:
            a, b = args.range.split("-")
            stems = [x for x in stems if a <= x < b]
        an = ShotAnalyzer(cfg, model, bank)
        meta, maps, last = [], [], None
        for stem in stems:
            ts = datetime.strptime(stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            if last is not None and ts.timestamp() - last > 120:
                an.track, an.last_thumb = None, None
            last = ts.timestamp()
            Clock.now = ts.timestamp()
            img = cv2.imread(str(frame_file(stem)))
            if img is None or img.shape[1] != 1920:
                continue
            sr = an.analyse(img, ts)
            if sr.cam is None or sr.density is None:
                continue
            cam = sr.cam
            blocked = np.maximum(gm, kg_mask(kg, cam, img.shape[:2]))
            segs = rows_segments(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cam, blocked)
            meta.append({"stem": stem, "camera": sr.camera, "how": sr.how, "method": sr.method,
                         "cam": {"f": float(cam.f), "rvec": [float(x) for x in cam.rvec], "tvec": [float(x) for x in cam.tvec],
                                 "size": list(cam.size)},
                         "rows": sr.rows.density if sr.rows else None, "segs": segs, "i": len(maps)})
            maps.append(sr.density.astype(np.float16))
        name = f"{name}-test" if args.range else name
        np.savez_compressed(OUT / f"{name}.npz", density=np.stack(maps) if maps else np.zeros((0, 135, 240), np.float16))
        (OUT / f"{name}.json").write_text(json.dumps({"window": name, "frames": meta}))
        keep(m["stem"] for m in meta)  # training / test frames: copied where housekeeping never deletes
        nseg = sum(len(m["segs"]) for m in meta)
        print(f"{name}: {len(stems)} frames, {len(meta)} placed, {sum(bool(m['segs']) for m in meta)} with rows labels "
              f"({nseg} strips)", flush=True)
        torch.cuda.empty_cache()


def kg_mask(kg, cam, shape) -> np.ndarray:
    """The Kaaba's silhouette (whole box) in the frame."""
    m = np.zeros(shape, np.uint8)
    sil = kg.silhouette(cam)
    if sil is not None:
        cv2.fillPoly(m, [np.round(sil[0]).astype(np.int32)], 1)
    return m


if __name__ == "__main__":
    main()
