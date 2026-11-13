"""Kaaba gate check before deploying (user decision 2026-10-27): on saved frames, the live analysis without the gate
(a full search every 10 s, cameras as found) and with it (every 20 s, fitted to the Kaaba, kept only when the drawn
Kaaba overlaps the real one >= GATE_OVERLAP). Deploy only if coverage (frames placed on the floor) drops by less than
~20 % and the Kaaba overlap of the placed frames stays >= 0.8.

The frames' own timestamps drive the analysis clock (edge/shot.py's timers), so "every 20 s" means 20 s of broadcast.
The Kaaba's mask of every frame is found once (the same segmenter) and used only for scoring.

  .venv/Scripts/python scripts/gate_check.py --windows 20261026T1800-20261026T1815 20261026T1500-20261026T1510
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Clock:
    """time for edge/shot.py: monotonic() = the frame's timestamp, perf_counter() real."""
    now = 0.0

    @staticmethod
    def monotonic() -> float:
        return Clock.now

    perf_counter = staticmethod(time.perf_counter)
    time = staticmethod(time.time)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", nargs="+", default=["20261026T1800-20261026T1815"])
    args = ap.parse_args()
    import torch

    from edge import shot as shot_mod
    from edge.crowd_model import CrowdModel
    from edge.m2_pipeline import IN_VIEW_MIN, kaaba_in_view, kaaba_overlap
    from edge.shot import ShotAnalyzer
    from edge.shot_bank import Bank
    from edge.view_matcher import graphics_mask, load_config

    shot_mod.time = Clock
    files = []
    for w in args.windows:
        a, b = w.split("-")
        files += [f for f in sorted(glob.glob(str(ROOT / "data" / "samples" / "dvr" / "*.jpg"))) if a <= Path(f).stem < b]
    cfg = load_config()
    model = CrowdModel()
    bank = Bank(ROOT / "data" / "vloc" / "bank.npz")
    runs = {}
    masks: dict[str, np.ndarray | None] = {}
    for name, gate, reverify in (("before (no gate, every 10 s)", False, 10.0), ("gate (every 20 s)", True, 20.0)):
        shot_mod.REVERIFY_S = reverify
        an = ShotAnalyzer(cfg, model, bank, kaaba_gate=gate)
        rows, last_day = [], None
        for f in files:
            stem = Path(f).stem
            ts = datetime.strptime(stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            if last_day is not None and (ts.timestamp() - last_day) > 120:  # a gap between windows: a new start
                an.track, an.last_thumb = None, None
            last_day = ts.timestamp()
            Clock.now = ts.timestamp()
            img = cv2.imread(f)
            h, w = img.shape[:2]
            gm = graphics_mask(cfg, (w, h))
            if stem not in masks:
                cands = [c for c in an.m2.seg.find(img, gm, k=2) if c.dark >= 0.3]
                masks[stem] = np.packbits(max(cands, key=lambda c: c.mask.sum()).mask > 0) if cands else None
            sr = an.analyse(img, ts)
            ov = None
            if sr.cam is not None and masks[stem] is not None and kaaba_in_view(sr.cam, (h, w), gm) >= IN_VIEW_MIN:
                mk = np.unpackbits(masks[stem])[: h * w].reshape(h, w)
                ov = kaaba_overlap(sr.cam, mk)
            rows.append({"stem": stem, "placed": sr.cam is not None, "how": sr.how, "camera": sr.camera, "overlap": ov,
                         "all": sr.reading.density_corr.get("all") if sr.reading else None, "ms": sr.ms.get("total")})
        runs[name] = rows
        torch.cuda.empty_cache()
    n = len(files)
    print(f"{n} frames, windows {', '.join(args.windows)}")
    base = None
    for name, rows in runs.items():
        placed = sum(r["placed"] for r in rows)
        ov = np.array([r["overlap"] for r in rows if r["overlap"] is not None])
        hows = {}
        for r in rows:
            if r["placed"]:
                k = r["how"].split("/")[0].split("+")[0] + ("+kaaba" if "+kaaba" in r["how"] else "")
                hows[k] = hows.get(k, 0) + 1
        rej = sum(r["how"].startswith("kaaba gate") for r in rows)
        base = base or placed
        print(f"{name}: placed {placed} / {n} ({100 * placed / n:.0f} %, {100 * (placed / base - 1):+.0f} % vs before); "
              f"Kaaba overlap (placed, Kaaba in view, {len(ov)} frames): median {np.median(ov) if len(ov) else float('nan'):.2f}, "
              f"p10 {np.percentile(ov, 10) if len(ov) else float('nan'):.2f}, >= 0.8: {100 * np.mean(ov >= 0.8) if len(ov) else 0:.0f} %; "
              f"rejected by the gate {rej}; methods {hows}; ms median {np.median([r['ms'] for r in rows]):.0f}")
    a, b = runs.values()
    both = [(x["all"], y["all"]) for x, y in zip(a, b) if x["all"] and y["all"]]
    if both:
        r = np.array([y / x for x, y in both])
        print(f"same frames placed by both ({len(both)}): reading gate / before median {np.median(r):.3f}, "
              f"p10-p90 {np.percentile(r, 10):.3f}-{np.percentile(r, 90):.3f}")


if __name__ == "__main__":
    main()
