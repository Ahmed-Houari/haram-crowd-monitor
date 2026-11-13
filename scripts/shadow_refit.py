"""Refits for the fine-tuned DM-Count before its 24 h shadow run (user request 2026-10-27). The new model needs no
head-size correction and no camera factor on YouTube frames (scripts/finetune_dmcount.py: it was trained to read the
live metric's corrected values on walking crowds and the true counts on prayer rows). Refitted for it:
  1. the SBA curve: the simultaneous YouTube / SBA frames of 27 Oct (data/samples/dual2). The live analysis places the
     YouTube frame; its camera serves the SBA frame too (same broadcast, same crop: scale 1.000, shift < 3 px). The new
     model's raw count per head-size bin on both -> the SBA head-size curve h0 (k as the live SBA curve) that puts
     SBA's reading on YouTube's;
  2. the approximate path (frames without a camera, edge/shot.py method c): on placed saved frames (data/finetune,
     outside prayer rows) per horizontal band the new model's count at half size / at full size against the band's
     real head width (from the camera) -> a table count ratio -> head width; then the bias = median(approximate /
     calibrated reading), both with the new model, the same way APPROX_BIAS (1.7) was measured for the live model.
Writes config/m2.json "shadow" (read by edge/shadow.py at the shot service's start). Offline.
--weights / --section live_model (2026-10-29, v3 deployed): the same refits for the model the live metric uses, written
to config/m2.json "live_model" (edge/m2_pipeline.py, edge/shot.py).

  .venv/Scripts/python scripts/shadow_refit.py [--weights qnrf_ft3 --section live_model] [--approx-only]
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from edge.kept_frames import frame_file  # noqa: E402

DUAL = ROOT / "data" / "samples" / "dual2"
M2_JSON = ROOT / "config" / "m2.json"
HB = np.array([0, 2, 3, 4, 5, 6, 8, 10, 13, 17, 25, 1e9])
MID = np.array([1.0, 2.5, 3.5, 4.5, 5.5, 7, 9, 11.5, 15, 21, 30])


def pairs() -> list[tuple[Path, Path]]:
    """(YouTube frame, SBA frame) of the same moment: thumbnail correlation, best frame offset per 40-frame window."""
    fy, fs = sorted((DUAL / "yt").glob("*.jpg")), sorted((DUAL / "sba").glob("*.jpg"))

    def th(f):
        x = cv2.resize(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2GRAY), (96, 54), interpolation=cv2.INTER_AREA)
        x = x.astype(np.float32).ravel()
        return (x - x.mean()) / (x.std() + 1e-6)

    hy, hs = np.stack([th(f) for f in fy]), np.stack([th(f) for f in fs])
    C = hs @ hy.T / hy.shape[1]
    out = []
    for s0 in range(0, len(fs), 40):
        rng = range(s0, min(s0 + 40, len(fs)))
        k = max(range(-30, 31), key=lambda k: np.median([C[i, i + k] for i in rng if 0 <= i + k < len(fy)] or [0]))
        out += [(fy[i + k], fs[i]) for i in rng if 0 <= i + k < len(fy) and C[i, i + k] >= 0.9]
    return out


class Clock:
    now = 0.0
    monotonic = staticmethod(lambda: Clock.now)
    perf_counter = staticmethod(time.perf_counter)
    time = staticmethod(time.time)


def floor_cells(cam, dshape, gm):
    from edge import kaaba_geom as kg
    from edge.m2_metric import HEAD_M, cell_floor, courtyard_edge, wall_distance

    dh, dw = dshape
    geo = cell_floor(cam, dshape)
    wd = wall_distance(geo["X"], geo["Y"])
    k = np.zeros((1080, 1920), np.uint8)
    sil = kg.silhouette(cam)
    if sil is not None:
        cv2.fillPoly(k, [np.round(sil[0]).astype(np.int32)], 1)
    cellm = lambda m: m[: dh * 8, : dw * 8].reshape(dh, 8, dw, 8).mean((1, 3)) >= 0.5  # noqa: E731
    floor = (geo["valid"] & (np.hypot(geo["X"], geo["Y"]) <= courtyard_edge(np.arctan2(geo["Y"], geo["X"])))
             & (wd >= 1) & ~cellm(k) & ~cellm(gm))
    return floor, cam.f * HEAD_M / np.maximum(geo["depth"], 1e-3), geo


def place(old, cfg, bank) -> list:
    """The YouTube frames of the simultaneous pairs, placed by the live analysis: [(YouTube frame, SBA frame, camera)]."""
    from edge import shot as shot_mod
    from edge.shot import ShotAnalyzer

    shot_mod.time = Clock
    an = ShotAnalyzer(cfg, old, bank, kaaba_gate=False)
    out = []
    for fy, fsb in pairs():
        Clock.now = int(fy.stem) / 1000
        sr = an.analyse(cv2.imread(str(fy)), datetime.fromtimestamp(Clock.now, timezone.utc))
        if sr.cam is not None and "+rows" not in sr.how:
            out.append((fy, fsb, sr.cam))
    return out


def sba_fit(placed: list, new, gm) -> dict:
    cy, cs = np.zeros(len(MID)), np.zeros(len(MID))
    per = []
    for fy, fsb, cam in placed:
        img = cv2.imread(str(fy))
        sba = cv2.resize(cv2.imread(str(fsb)), (1920, 1080), interpolation=cv2.INTER_CUBIC)
        dy, ds = new.predict(img).density, new.predict(sba).density
        floor, head, _ = floor_cells(cam, dy.shape, gm)
        if floor.sum() < 50:
            continue
        b = np.digitize(head[floor], HB) - 1
        y = np.bincount(b, dy[floor], len(MID))
        s = np.bincount(b, ds[floor], len(MID))
        cy += y
        cs += s
        per.append((y, s))
    g = lambda h0, k=1.96: np.maximum(1 - np.exp(-(MID / h0) ** k), 0.2)  # noqa: E731
    target = cy.sum()
    h0s = np.arange(0.3, 15, 0.02)
    h0 = float(h0s[int(np.argmin([abs(np.log((cs / g(x)).sum() / target)) for x in h0s]))])
    ratios = [float((s / g(h0)).sum() / y.sum()) for y, s in per if y.sum() > 20]
    return {"pairs": len(per), "raw_sba_over_yt": round(float(cs.sum() / cy.sum()), 3),
            "by_head": {f"{HB[i]:g}-{HB[i + 1]:g}": round(float(cs[i] / cy[i]), 3) for i in range(len(MID)) if cy[i] > 20},
            "h0": round(h0, 2), "k": 1.96,
            "after_fit_per_pair": [round(float(np.median(ratios)), 3), round(float(np.percentile(ratios, 10)), 3),
                                   round(float(np.percentile(ratios, 90)), 3)] if ratios else None}


def approx_fit(old, new, gm) -> dict:
    """The approximate path's table (count ratio half / full -> head width) and bias for the new model; the live
    model's own numbers the same way, as the yardstick."""
    from edge import kaaba_geom as kg
    from edge.m2_metric import HEAD_M, measure
    from edge.m2_pipeline import CAMERA_FACTOR, g, kaaba_exclusion
    from edge.shot import APPROX_BANDS, APPROX_BIAS, APPROX_MIN_PEOPLE, SIN_DOWN, head_from_scale

    frames = []
    for p in sorted((ROOT / "data" / "finetune").glob("*.json")):
        if p.stem.startswith("report") or p.stem.endswith("-test"):
            continue
        frames += [f for f in json.loads(p.read_text())["frames"] if "+rows" not in f["how"]]
    frames = frames[::2]
    bands = {"old": [], "new": []}
    per = {"old": [], "new": []}
    for i, fr in enumerate(frames):
        img = cv2.imread(str(frame_file(fr['stem'])))
        if img is None:
            continue
        c = fr["cam"]
        cam = kg.Camera(c["f"], np.asarray(c["rvec"], float), np.asarray(c["tvec"], float), tuple(c["size"]))
        half_img = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
        for name, model in (("old", old), ("new", new)):
            dens = model.predict(img).density
            half = model.predict(half_img).density
            dh, dw = dens.shape
            floor, head, _ = floor_cells(cam, dens.shape, gm)
            crowd = cv2.GaussianBlur(dens.astype(np.float32), (0, 0), 2.0) > 0.01
            gmc = gm[: dh * 8, : dw * 8].reshape(dh, 8, dw, 8).mean((1, 3)) >= 0.5
            crowd &= ~gmc
            if float(dens[crowd].sum()) < APPROX_MIN_PEOPLE:
                continue
            ex = np.maximum(kaaba_exclusion(cam, (1080, 1920)), gm)
            if name == "old":
                rd = measure(cam, dens, ex, g, CAMERA_FACTOR.get(fr["camera"], 1.0))
            else:
                rd = measure(cam, dens, ex, lambda h: np.ones_like(h), 1.0)  # no correction: built into the model
            cal = rd.density_corr.get("all")
            hh = half.shape[0]
            rows = []
            for kb in range(APPROX_BANDS):
                r0, r1 = kb * dh // APPROX_BANDS, (kb + 1) * dh // APPROX_BANDS
                sel = crowd[r0:r1] & floor[r0:r1]
                c_full = float(dens[r0:r1][crowd[r0:r1]].sum())
                if c_full < APPROX_MIN_PEOPLE / APPROX_BANDS or sel.sum() < 20:
                    continue
                ratio = float(half[r0 * hh // dh:r1 * hh // dh].sum()) / max(float(dens[r0:r1].sum()), 1e-6)
                true_head = float(np.median(head[r0:r1][sel]))
                bands[name].append((ratio, true_head))
                rows.append((ratio, c_full, float(crowd[r0:r1].sum())))
            if cal and rows:
                per[name].append((cal, rows))
    out = {}
    for name in ("old", "new"):
        b = np.array(bands[name])
        if len(b) < 30:
            continue
        qs = np.quantile(b[:, 0], np.linspace(0, 1, 13))
        tab_r, tab_h = [], []
        for lo, hi in zip(qs[:-1], qs[1:]):
            m = (b[:, 0] >= lo) & (b[:, 0] <= hi)
            if m.sum() >= 5:
                tab_r.append(float(np.median(b[m, 0])))
                tab_h.append(float(np.median(b[m, 1])))
        tab_h = list(np.maximum.accumulate(tab_h))  # bigger heads keep more of their count at half size: monotone
        head_of = (lambda r: float(np.interp(r, tab_r, tab_h))) if name == "new" else head_from_scale
        ratios = []
        for cal, rows in per[name]:
            area = people = 0.0
            for ratio, c_full, ncell in rows:
                h = head_of(ratio)
                if 2.0 < h < 40.0:
                    area += ncell * 64 * (HEAD_M / h) ** 2 / SIN_DOWN
                    people += c_full
            if area > 0 and people >= APPROX_MIN_PEOPLE / 2 and cal > 0.05:
                ratios.append(people / area / cal)
        bias = float(np.median(ratios)) if name == "new" else APPROX_BIAS
        after = np.array(ratios) / bias
        out[name] = {"frames": len(ratios), "bands": len(b), "table_ratio": [round(x, 4) for x in tab_r],
                     "table_head": [round(x, 2) for x in tab_h], "bias": round(bias, 3),
                     "approx_over_calibrated_after_bias": [round(float(np.median(after)), 3),
                                                           round(float(np.percentile(after, 10)), 3),
                                                           round(float(np.percentile(after, 90)), 3)] if len(after) else None}
    return out


def main() -> None:
    import torch

    from edge.crowd_model import CrowdModel
    from edge.shot_bank import Bank
    from edge.view_matcher import graphics_mask, load_config

    weights = sys.argv[sys.argv.index("--weights") + 1] if "--weights" in sys.argv else "qnrf_ft"
    section = sys.argv[sys.argv.index("--section") + 1] if "--section" in sys.argv else "shadow"
    cfg = load_config()
    gm = graphics_mask(cfg, (1920, 1080))
    old = CrowdModel()
    if "--approx-only" in sys.argv:  # keep the SBA curve already written
        prev = json.loads(M2_JSON.read_text(encoding="utf-8"))[section]
        sba = {**prev["source"]["sba"], "kept_from": prev.get("since_utc")}
    else:
        placed = place(old, cfg, Bank(ROOT / "data" / "vloc" / "bank.npz"))  # the segmenter and the bank, then freed
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        print(f"placed {len(placed)} YouTube frames of the pairs", flush=True)
    new = CrowdModel(weights)
    if "--approx-only" not in sys.argv:
        sba = sba_fit(placed, new, gm)
    print("SBA (new model):", json.dumps(sba), flush=True)
    apx = approx_fit(old, new, gm)
    print("approximate path:", json.dumps(apx), flush=True)
    m2 = json.loads(M2_JSON.read_text(encoding="utf-8"))
    if section != "shadow":
        m2[section] = {**m2.get(section, {}), "weights": weights, "source": {"sba": {"h0": sba["h0"], "k": sba["k"]}},
                       "approx": {"table_ratio": apx["new"]["table_ratio"], "table_head": apx["new"]["table_head"],
                                  "bias": apx["new"]["bias"]}, "refit": {"sba": sba, "approx": apx}}
        M2_JSON.write_text(json.dumps(m2, indent=2) + "\n", encoding="utf-8")
        print(f"written: config/m2.json {section}")
        return
    m2["shadow"] = {
        "enabled": True, "weights": "qnrf_ft", "correction": None, "camera_factor": None,
        "source": {"sba": {"h0": sba["h0"], "k": sba["k"]}},
        "approx": {"table_ratio": apx["new"]["table_ratio"], "table_head": apx["new"]["table_head"],
                   "bias": apx["new"]["bias"]},
        "since_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": ("Shadow run of the fine-tuned DM-Count (models/DM-Count_QNRF_ft.pth, scripts/finetune_dmcount.py; user "
                 "request 2026-10-27: 24 h next to the live model, logged in crowd_m2_shadow, no dashboard change). No "
                 "head-size correction and no camera factor (built into the model); SBA curve and the approximate "
                 "path refitted for it by scripts/shadow_refit.py: " + json.dumps({"sba": sba, "approx": apx})),
    }
    M2_JSON.write_text(json.dumps(m2, indent=2) + "\n", encoding="utf-8")
    print("written: config/m2.json shadow")


if __name__ == "__main__":
    main()
