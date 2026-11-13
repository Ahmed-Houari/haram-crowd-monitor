"""Per-camera scale factors for the Mataf metric from zoomed-in frames (user request 2026-10-25).

Idea: where a camera is zoomed in so that heads are large (median head width >= --hmin px, default 10), the crowd
model's count is near-reliable. Such a floor patch is the reference for the same patch seen by the other cameras
(wide, small heads) within +-5 minutes: scale(camera) = reference density / that camera's density there (both after
the live head-size correction unless --raw-ref). One number per camera with a 90 % block-bootstrap range (30-minute
blocks: frames close in time are not independent). The same camera's own wide frames are compared too ("self"): a
check of the head-size correction itself.

Floor patches: the sectors of edge/m2_metric.py (12 angles x inner / outer ring, crowd_m2_frame.sectors, logged
since 2026-10-25 15:31 Makkah). Older rows have only whole rings (--rings): coarser, since two cameras see different
parts of a ring.

--samples: on the saved m2 sample frames (data/samples/m2_*.png + camera sidecar), compares DM-Count with the APGCC
point counter (edge/apgcc_model.py) per sector where heads are large: if the two disagree there, the reference is
not as reliable as assumed. Frames stay on this PC; only numbers are printed.

Read-only on the database.
  .venv/Scripts/python -I scripts/zoom_calib.py [--since "2026-10-25 15:31"] [--hmin 10] [--window 5] [--rings]
  .venv/Scripts/python -I scripts/zoom_calib.py --samples
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DB = ROOT / "data" / "haram.db"
SAMPLES = ROOT / "data" / "samples"
MAKKAH = timezone(timedelta(hours=3))
BLOCK_S = 1800


def load(db: Path, since: datetime, rings: bool) -> list[dict]:
    """Frames: {t, cam, cells: {key: (raw count, corrected count, m^2, head px)}}."""
    con = sqlite3.connect(f"file:{db.resolve().as_posix()}?mode=ro", uri=True)
    cols = {r[1] for r in con.execute("pragma table_info(crowd_m2_frame)")}
    sec = "sectors" if "sectors" in cols else "null"
    rows = con.execute(f"select ts_utc, ring, camera, people_m2, people_m2_raw, area_m2, head_px, {sec} "
                       "from crowd_m2_frame where ts_utc >= ? order by ts_utc",
                       (since.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),)).fetchall()
    frames: dict[str, dict] = {}
    for ts, ring, cam, pm, raw, area, hp, sectors in rows:
        f = frames.setdefault(ts, {"t": datetime.fromisoformat(str(ts)[:19]).replace(tzinfo=timezone.utc),
                                   "cam": cam, "cells": {}})
        if rings:
            if ring in ("inner", "outer") and area and hp:
                f["cells"][ring] = ((raw or 0) * area, pm * area, area, hp)
        elif ring == "all" and sectors:
            f["cells"].update({k: tuple(v) for k, v in json.loads(sectors).items()})
    return [f for f in frames.values() if f["cells"] and f["cam"]]


def regcorrect(frames: list[dict], h0: float, k: float) -> None:
    """Another head-size correction g (approximate: one head size per patch, its median)."""
    from edge.m2_metric import G_FLOOR

    for f in frames:
        f["cells"] = {key: (raw, raw / max(1 - np.exp(-(hp / h0) ** k), G_FLOOR), area, hp)
                      for key, (raw, corr, area, hp) in f["cells"].items()}


def events(frames: list[dict], hmin: float, window_s: float, amin_ref: float, amin: float, raw_ref: bool,
           wide_max: float) -> list[dict]:
    """One event per (zoomed reference frame, floor patch, other camera seeing that patch within the window)."""
    out = []
    times = np.array([f["t"].timestamp() for f in frames])
    for i, f in enumerate(frames):
        for key, (raw, corr, area, hp) in f["cells"].items():
            if hp < hmin or area < amin_ref:
                continue
            d_ref = (raw if raw_ref else corr) / area
            lo, hi = np.searchsorted(times, times[i] - window_s), np.searchsorted(times, times[i] + window_s, "right")
            pooled: dict[str, list] = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
            for j in range(lo, hi):
                g = frames[j]
                c = g["cells"].get(key)
                if j == i or c is None or c[2] < amin or c[3] >= wide_max:
                    continue
                p = pooled[g["cam"]]
                p[0] += c[1]
                p[1] += c[2]
                p[2] += c[3] * c[2]
                p[3] += 1
            for cam, (corr_b, area_b, hpa, n) in pooled.items():
                out.append({"t": f["t"], "ref_cam": f["cam"], "cam": cam, "key": key, "d_ref": d_ref,
                            "hp_ref": hp, "d": corr_b / area_b, "hp": hpa / area_b, "n": n,
                            "self": cam == f["cam"]})
    return out


def scale(evs: list[dict], rng: np.random.Generator, n_boot: int = 2000) -> tuple[float, float, float, int]:
    """Ratio of sums (reference / camera) with a 90 % bootstrap over 30-minute blocks; -> (s, lo, hi, n blocks)."""
    blocks: dict[int, list] = defaultdict(lambda: [0.0, 0.0])
    for e in evs:
        b = blocks[int(e["t"].timestamp() // BLOCK_S)]
        b[0] += e["d_ref"]
        b[1] += e["d"]
    arr = np.array(list(blocks.values()))
    if arr[:, 1].sum() <= 0:
        return float("nan"), float("nan"), float("nan"), len(arr)
    s = arr[:, 0].sum() / arr[:, 1].sum()
    idx = rng.integers(0, len(arr), (n_boot, len(arr)))
    num, den = arr[idx, 0].sum(1), arr[idx, 1].sum(1)
    boot = num[den > 0] / den[den > 0]
    return float(s), float(np.percentile(boot, 5)), float(np.percentile(boot, 95)), len(arr)


def report(evs: list[dict], label: str) -> None:
    rng = np.random.default_rng(0)
    print(f"\n{label}: {len(evs)} events")
    by: dict[tuple, list] = defaultdict(list)
    for e in evs:
        by[("self" if e["self"] else e["cam"], e["ref_cam"])].append(e)
        if not e["self"]:
            by[(e["cam"], "any")].append(e)
    for (cam, ref), es in sorted(by.items()):
        s, lo, hi, nb = scale(es, rng)
        hp_ref = np.mean([e["hp_ref"] for e in es])
        hp = np.mean([e["hp"] for e in es])
        print(f"  {cam:16s} vs zoomed {ref:16s} n {len(es):4d} ({nb:2d} blocks)  scale {s:5.2f} [{lo:4.2f}-{hi:4.2f}]  "
              f"ref {np.mean([e['d_ref'] for e in es]):.2f}/m2 @ {hp_ref:4.1f} px, camera "
              f"{np.mean([e['d'] for e in es]):.2f}/m2 @ {hp:4.1f} px")


def samples_check(hmin: float) -> None:
    """DM-Count vs APGCC per large-head sector on the saved m2 frames."""
    import cv2

    from edge.apgcc_model import PointCounter
    from edge.crowd_model import CrowdModel
    from edge.kaaba_geom import Camera
    from edge.m2_metric import RINGS, SECTORS, cell_floor, wall_distance, HEAD_M
    from edge.m2_pipeline import g

    dm, ap = CrowdModel(), PointCounter()
    tot = defaultdict(float)
    for js in sorted(SAMPLES.glob("m2_*.json")):
        meta = json.loads(js.read_text())
        img = cv2.imread(str(js.with_suffix(".png")))
        if img is None:
            continue
        c = meta["cam"]
        cam = Camera(c["f"], np.array(c["rvec"]), np.array(c["tvec"]), tuple(c["size"]))
        dens = dm.predict(img).density
        geo = cell_floor(cam, dens.shape)
        hp = cam.f * HEAD_M / np.maximum(geo["depth"], 1e-3)
        dist = wall_distance(geo["X"], geo["Y"])
        sec = (np.floor((np.arctan2(geo["Y"], geo["X"]) + np.pi) / (2 * np.pi / SECTORS)).astype(int)) % SECTORS
        pts = ap.predict(img).points
        dh, dw = dens.shape
        pj = np.clip((pts[:, 0] // 8).astype(int), 0, dw - 1)
        pi = np.clip((pts[:, 1] // 8).astype(int), 0, dh - 1)
        apc = np.zeros_like(dens)
        np.add.at(apc, (pi, pj), 1.0)  # APGCC points binned onto the density grid
        rows = []
        for name, (r0, r1) in RINGS.items():
            for s in range(SECTORS):
                sel = geo["valid"] & (dist >= r0) & (dist < r1) & (sec == s)
                if sel.sum() == 0 or np.median(hp[sel]) < hmin or geo["area"][sel].sum() < 10:
                    continue
                d_raw, d_cor = dens[sel].sum(), (dens[sel] / np.maximum(g(hp[sel]), 0.2)).sum()
                a = apc[sel].sum()
                rows.append(f"{name[0]}{s}: {geo['area'][sel].sum():.0f} m2 @ {np.median(hp[sel]):.1f} px  DM-Count "
                            f"{d_raw:.0f} (corrected {d_cor:.0f})  APGCC {a:.0f}")
                tot["raw"] += d_raw
                tot["cor"] += d_cor
                tot["ap"] += a
        print(f"{js.stem} ({meta['camera']}, zoom {meta.get('zoom')})" + ("".join(f"\n  {r}" for r in rows)
                                                                            or "  - no sector with large heads"))
    if tot["ap"]:
        print(f"\nall large-head sectors: DM-Count {tot['raw']:.0f}, corrected {tot['cor']:.0f}, APGCC {tot['ap']:.0f}"
              f" -> DM-Count / APGCC {tot['raw'] / tot['ap']:.2f}, corrected / APGCC {tot['cor'] / tot['ap']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", default="2026-10-25 15:31", help="Makkah time")
    ap.add_argument("--hmin", type=float, default=10.0, help="reference: median head width >= this (px)")
    ap.add_argument("--wide-max", type=float, default=8.0, help="compared cells: median head width < this (px)")
    ap.add_argument("--window", type=float, default=5.0, help="minutes")
    ap.add_argument("--rings", action="store_true", help="whole rings instead of sectors (rows before sectors)")
    ap.add_argument("--raw-ref", action="store_true", help="reference without the head-size correction")
    ap.add_argument("--samples", action="store_true", help="DM-Count vs APGCC on the saved m2 frames")
    ap.add_argument("--g", nargs=2, type=float, metavar=("H0", "K"), help="re-correct with this g instead of the live one")
    ap.add_argument("--db", default=str(DB))
    args = ap.parse_args()
    if args.samples:
        samples_check(args.hmin)
        return
    since = datetime.fromisoformat(args.since).replace(tzinfo=MAKKAH)
    frames = load(Path(args.db), since, args.rings)
    if args.g:
        regcorrect(frames, *args.g)
    n_zoom = defaultdict(int)
    for f in frames:
        if any(c[3] >= args.hmin and c[2] >= (40 if args.rings else 10) for c in f["cells"].values()):
            n_zoom[f["cam"]] += 1
    print(f"{len(frames)} frames since {args.since} Makkah ({'rings' if args.rings else 'sectors'}); frames with a "
          f"patch at >= {args.hmin:g} px heads: {dict(n_zoom)}")
    evs = events(frames, args.hmin, args.window * 60, 40 if args.rings else 10, 40 if args.rings else 20,
                 args.raw_ref, args.wide_max)
    g = f"g h0 {args.g[0]:g}, k {args.g[1]:g} (approx.)" if args.g else "live g"
    report(evs, f"scale = zoomed reference ({'raw' if args.raw_ref else 'corrected'}) / camera (corrected, {g}), "
                f"+-{args.window:g} min")


if __name__ == "__main__":
    main()
