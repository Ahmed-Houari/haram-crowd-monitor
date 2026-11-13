"""Independent checks of the fine-tuned DM-Count on held-out frames, before any deploy (user request 2026-10-27).

1. Zoom pairs (not the rows estimator): a zoomed-in frame of a camera without a camera factor (heads >= REF_HEAD px,
   where the live model counts right: 1.00-1.03 of the rows truth at 7-10 px) and a wide frame of the same floor
   within PAIR_S seconds (heads < TARGET_HEAD px): the people the zoomed-in frame sees per m^2 on the shared floor
   (polar bins 2 deg x 1 m) times the wide frame's area there = the reference count. The live model (+ live
   correction) and the fine-tuned model (no correction) are compared with it, by the wide frame's head size and by
   posture (prayer rows or not). Only the held-out windows (Maghrib 26 Oct, Asr 27 Oct, the second half of the tawaf
   window); none of them trained the model. A 90 % range from a bootstrap over frames.
2. Far rows to count by eye: three crops of held-out prayer frames (different cameras), two rows ~8 m along them
   with heads of FAR_HEAD px where people in a row are >= 5 px apart, enlarged, outlined; above it the live model's,
   the fine-tuned model's and the rows estimator's people in the outline. Saved to data/samples/debug_align/count_crop_*.jpg (local only).

  .venv/Scripts/python scripts/finetune_check.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import finetune_dmcount as ft  # noqa: E402

HELD_OUT = ["2026-10-26-maghrib", "2026-10-27-asr"]
TAWAF = "2026-10-26-tawaf-2110"
PAIR_S = 60
REF_HEAD = 10.0
TARGET_HEAD = 6.0
MIN_AREA = 15.0
BINS = [(0, 3), (3, 4), (4, 5), (5, 6)]
FAR_HEAD = (3.3, 5.0)
OUT = ROOT / "data" / "samples" / "debug_align"


def zoom_pairs(frames: list[dict], h0: float, k: float) -> list[dict]:
    """Reference counts for wide frames from zoomed-in frames of the same floor (see the module docstring)."""
    out = []
    by_win = defaultdict(list)
    for f in frames:
        by_win[f["window"]].append(f)
    for fs in by_win.values():
        t = {f["stem"]: time.mktime(time.strptime(f["stem"], "%Y%m%dT%H%M%SZ")) for f in fs}
        refs = []
        for f in fs:
            if f["camera"] not in ft.ZOOM_REF_CAMERAS:
                continue
            sel = f["floor"] & (f["head"] >= REF_HEAD)
            if f["area"][sel].sum() < 30:
                continue
            key = ft.bins(f, sel).ravel()
            ok = key >= 0
            refs.append((f, key[ok], (f["old"] / ft.g_curve(f["head"], h0, k)).ravel()[ok], f["area"].ravel()[ok]))
        for f in fs:
            sel = f["floor"] & (f["head"] < TARGET_HEAD)
            if f["area"][sel].sum() < 30:
                continue
            keyB = ft.bins(f, sel).ravel()
            for A, keyA, corrA, areaA in refs:
                if A is f or abs(t[A["stem"]] - t[f["stem"]]) > PAIR_S:
                    continue
                shared = np.intersect1d(np.unique(keyA), np.unique(keyB[keyB >= 0]))
                if len(shared) < 5:
                    continue
                ia = np.isin(keyA, shared)
                for lo, hi in BINS:
                    if lo < ft.TEST_ZOOM_MIN_HEAD:  # small-head zoom references read too low (user, 2026-10-29)
                        continue
                    ib = np.flatnonzero(np.isin(keyB, shared) & (f["head"].ravel() >= lo) & (f["head"].ravel() < hi))
                    aA, aB = float(areaA[ia].sum()), float(f["area"].ravel()[ib].sum())
                    if aA < MIN_AREA or aB < MIN_AREA:
                        continue
                    ref = float(corrA[ia].sum()) / aA * aB
                    h = f["head"].ravel()[ib]
                    live = float((f["old"].ravel()[ib] / (ft.g_curve(h, h0, k) * f["factor"])).sum())
                    new = float(f["new"].ravel()[ib].sum())
                    out.append({"stem": f["stem"], "window": f["window"], "camera": f["camera"],
                                "rows": "+rows" in f["how"], "heads": f"{lo}-{hi}", "ref": ref, "live": live, "new": new})
    return out


def boot(rs: list[dict], key: str, rng: np.random.Generator, n: int = 2000) -> tuple[float, float, float]:
    """Pooled ratio sum(key) / sum(ref) and a 90 % range, frames resampled."""
    by = defaultdict(list)
    for r in rs:
        by[r["stem"]].append(r)
    groups = list(by.values())
    tot = lambda gs: sum(r[key] for g in gs for r in g) / max(sum(r["ref"] for g in gs for r in g), 1e-9)  # noqa: E731
    bs = [tot([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(n)]
    return tot(groups), float(np.percentile(bs, 5)), float(np.percentile(bs, 95))


def patch(f: dict, b: float, dm: float, pitch: float, half_len: float):
    """A floor patch along the rows: wall distance dm +- one row pitch (two rows), +-half_len m along the rows around
    the line at bearing b. Returns (cell indices, image polygon at head height)."""
    from edge import prayer_rows as pr
    from edge.kaaba_geom import HEAD_Z

    c = pr._ray_points(np.array([dm]), b)[0]
    r = float(np.hypot(*c))
    dth = half_len / r
    th = np.angle(np.exp(1j * (f["th"] - math.radians(b))))
    m = f["floor"] & (f["wd"] >= dm - pitch) & (f["wd"] < dm + pitch) & (np.abs(th) <= dth)
    pts = []
    for d, sgn in ((dm - pitch, 1), (dm + pitch, -1)):
        for t in np.linspace(-dth, dth, 15)[::sgn]:
            q = pr._ray_points(np.array([d]), b + math.degrees(t))[0]
            pts.append([q[0], q[1], HEAD_Z])
    uv, _ = f["cam_obj"].project(np.array(pts))
    return np.flatnonzero(m.ravel()), uv


def along_px_per_m(f: dict, b: float, dm: float) -> float:
    """Image px per metre along the rows at that spot (people stand ~0.6 m apart there)."""
    from edge import prayer_rows as pr
    from edge.kaaba_geom import HEAD_Z

    c = pr._ray_points(np.array([dm]), b)[0]
    r = float(np.hypot(*c))
    q = [pr._ray_points(np.array([dm]), b + math.degrees(s / r))[0] for s in (-0.5, 0.5)]
    uv, _ = f["cam_obj"].project(np.array([[*q[0], HEAD_Z], [*q[1], HEAD_Z]]))
    return float(np.hypot(*(uv[1] - uv[0])))


def crops(frames: list[dict], h0: float, k: float) -> list[str]:
    """Three far-row crops (different cameras) from held-out prayer frames: two rows, ~8 m along them, where the
    people in a row are at least 5 px apart in the image (countable when enlarged)."""
    cands = []
    for f in frames:
        if "+rows" not in f["how"]:
            continue
        for r in f["regions"]:
            s = r.seg
            if r.kind != "rows" or not FAR_HEAD[0] <= r.head < FAR_HEAD[1] or s["strength"] < 0.5:
                continue
            dm = (s["d0"] + s["d1"]) / 2
            if along_px_per_m(f, s["b"], dm) * 0.6 < 5:
                continue
            # the whole patch on rows: the lines either side (+-5 deg) show strong rows at that distance too
            near = [x for x in f["segs"] if abs(((x["b"] - s["b"] + 180) % 360) - 180) == 5 and x["d0"] <= dm - x["pitch"]
                    and dm + x["pitch"] <= x["d1"] and x["strength"] >= 0.45 and abs(x["dens"] - s["dens"]) < 0.3]
            if len(near) < 2:
                continue
            cands.append((s["strength"], f, r, s, dm))
    cands.sort(key=lambda c: -c[0])
    picked, used = [], set()
    for st, f, r, seg, dm in cands:
        if f["camera"] in used:
            continue
        idx, poly = patch(f, seg["b"], dm, seg["pitch"], 3.5)
        area = float(f["area"].ravel()[idx].sum())
        if len(idx) < 4 or area < 8:
            continue
        used.add(f["camera"])
        picked.append((f, seg, dm, idx, poly, area))
        if len(picked) == 3:
            break
    out = []
    for f, seg, dm, idx, poly, area in picked:
        img = cv2.imread(str(ft.frame_file(f['stem'])))
        x0, y0 = np.floor(poly.min(0)).astype(int) - 40
        x1, y1 = np.ceil(poly.max(0)).astype(int) + 40
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, img.shape[1]), min(y1, img.shape[0])
        crop = img[y0:y1, x0:x1]
        sc = max(4, int(1000 / max(crop.shape[:2])))
        big = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        cv2.polylines(big, [np.round((poly - [x0, y0]) * sc).astype(np.int32)], True, (255, 0, 255), 2, cv2.LINE_AA)
        h = f["head"].ravel()[idx]
        live = float((f["old"].ravel()[idx] / (ft.g_curve(h, h0, k) * f["factor"])).sum())
        new = float(f["new"].ravel()[idx].sum())
        label = seg["dens"] * area
        lines = [f"{f['camera']} {f['stem']} (held out, not used in training)",
                 f"inside the magenta outline: two rows, ~7 m along them, {area:.0f} m2 of floor, heads ~{np.median(h):.1f} px",
                 f"live model {live:.0f} people | fine-tuned {new:.0f} | rows estimator {label:.0f} "
                 f"({seg['dens']:.2f}/m2, pitch {seg['pitch']:.2f} m)"]
        bar = np.zeros((100, big.shape[1], 3), np.uint8)
        for i, ln in enumerate(lines):
            cv2.putText(bar, ln, (10, 26 + 29 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        path = OUT / f"count_crop_{f['camera']}_{f['stem']}.jpg"
        cv2.imwrite(str(path), np.vstack([bar, big]), [cv2.IMWRITE_JPEG_QUALITY, 92])
        out.append(f"{path.name}: live {live:.0f}, fine-tuned {new:.0f}, rows estimator {label:.0f}, area {area:.0f} m2, "
                   f"heads {np.median(h):.1f} px, crop {crop.shape[1]}x{crop.shape[0]} px enlarged x{sc}")
    return out


def main() -> None:
    from edge.m2_metric import load_correction
    from edge.view_matcher import graphics_mask, load_config

    h0, k, factors = load_correction()
    gm = graphics_mask(load_config(), (1920, 1080))
    tw = sorted(ft.load([TAWAF]), key=lambda f: f["stem"])
    frames = ft.load(HELD_OUT) + tw[len(tw) // 2:]
    for f in frames:
        ft.geometry(f, gm, factors)
        ft.rows_regions(f)
    ft.predict_all(frames, ft.WEIGHTS_OUT)
    rs = zoom_pairs(frames, h0, k)
    rng = np.random.default_rng(0)
    rep = {"pairs": len(rs), "frames": len({r["stem"] for r in rs}), "table": []}
    print(f"zoom pairs on held-out frames: {len(rs)} regions in {rep['frames']} wide frames "
          f"(reference: zoomed-in frame within {PAIR_S} s, heads >= {REF_HEAD:g} px)")
    print("predicted / reference (pooled), 90 % range over frames; 1.00 = right")
    for rows in (False, True):
        for lo, hi in BINS:
            sel = [r for r in rs if r["rows"] == rows and r["heads"] == f"{lo}-{hi}"]
            if len({r["stem"] for r in sel}) < 3:
                continue
            lv, nw = boot(sel, "live", rng), boot(sel, "new", rng)
            rep["table"].append({"rows": rows, "heads": f"{lo}-{hi}", "n": len(sel), "live": lv, "new": nw})
            print(f"  {'prayer rows' if rows else 'no rows    '} heads {lo}-{hi} px  n {len(sel):4d}   live {lv[0]:.2f} "
                  f"({lv[1]:.2f}-{lv[2]:.2f})   fine-tuned {nw[0]:.2f} ({nw[1]:.2f}-{nw[2]:.2f})")
    rep["crops"] = crops([f for f in frames if f["window"] in HELD_OUT], h0, k)
    print("crops to count by eye:")
    for c in rep["crops"]:
        print("  " + c)
    (ft.DATA / "report_check.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
