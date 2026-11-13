"""Speed and VRAM of the view-independent metric, stage by stage, on the GPU (research, 2026-10-24).

Runs every stage N times on the three 1080p reference frames (already on disk) and prints median / 90th percentile
time and the peak VRAM after each stage. Stages: DM-Count (crowd model), DINOv2-small (view embedding), ALIKED features
+ LightGlue against the calibrated references (ptz path), OWLv2 box + SAM 2.1-large mask (kaaba path), the Kaaba fit
(CPU), people per m^2 from the density map (CPU). The GPU's power mode is whatever the laptop runs in (the user's
"Silent" mode caps it at 42 W); nvidia-smi's power draw is printed for reference.

  .venv/Scripts/python scripts/bench_m2.py [--n 10]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main() -> None:
    import torch

    from edge import kaaba_geom as kg
    from edge.crowd_model import CrowdModel
    from edge.kaaba_seg import KaabaSegmenter
    from edge.m2_metric import measure
    from edge.m2_pipeline import g, kaaba_exclusion, load_cameras
    from edge.ptz import WORK, PtzMatcher
    from edge.view_matcher import graphics_mask, load_config

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    cfg = load_config()
    refs = load_cameras(cfg)
    imgs = [img for img, _ in refs.values()]
    gm = graphics_mask(cfg, (1920, 1080))
    times: dict[str, list[float]] = {}
    vram: dict[str, float] = {}

    def timed(name, fn, *a):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn(*a)
        torch.cuda.synchronize()
        times.setdefault(name, []).append((time.perf_counter() - t) * 1e3)
        vram[name] = max(vram.get(name, 0.0), torch.cuda.max_memory_allocated() / 1e9)
        return out

    torch.cuda.reset_peak_memory_stats()
    model = CrowdModel()
    ptz = PtzMatcher(refs, graphics_mask(cfg, WORK))
    seg = KaabaSegmenter()
    from transformers import AutoImageProcessor, AutoModel
    from PIL import Image

    dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    dino = AutoModel.from_pretrained("facebook/dinov2-small").to("cuda").eval()
    print(f"models loaded: VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    def dino_embed(im):
        small = cv2.resize(im, (448, 252), interpolation=cv2.INTER_AREA)
        inp = dproc(images=Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)), return_tensors="pt").to("cuda")
        with torch.no_grad():
            return dino(**inp).last_hidden_state[0, 0]

    for i in range(args.n + 1):  # first round = warm-up, dropped below
        for im in imgs:
            dens = timed("dm_count", lambda x: model.predict(x).density, im)
            timed("dinov2", dino_embed, im)
            pm = timed("ptz_aliked_lightglue", ptz.match, im)
            cands = timed("owlv2_box", seg.boxes, im, gm, 2)
            if cands:
                m = timed("sam21_large_mask", seg.mask, im, cands[0])[0]
                r = timed("kaaba_fit_cpu", kg.fit, m)
            cam = pm.cam if pm is not None and pm.cam is not None else (r.cam if cands and r else None)
            if cam is not None:
                timed("people_m2_cpu", lambda: measure(cam, dens, np.maximum(kaaba_exclusion(cam, (1080, 1920)), gm), g))
        if i == 0:
            times = {k: [] for k in times}
    print(f"\n{'stage':24s} {'median ms':>10} {'p90 ms':>8} {'peak VRAM GB':>13}")
    for k, v in times.items():
        print(f"{k:24s} {np.median(v):10.0f} {np.percentile(v, 90):8.0f} {vram[k]:13.2f}")
    ptz_frame = np.median(times["dm_count"]) + np.median(times["dinov2"]) + np.median(times["ptz_aliked_lightglue"]) \
        + np.median(times["people_m2_cpu"])
    kaaba_frame = ptz_frame + np.median(times["owlv2_box"]) + np.median(times["sam21_large_mask"]) \
        + np.median(times["kaaba_fit_cpu"])
    print(f"\nper frame: ptz path {ptz_frame / 1e3:.1f} s, kaaba path {kaaba_frame / 1e3:.1f} s (budget 15 s); "
          f"peak VRAM {max(vram.values()):.2f} GB of {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}")
    try:
        print(subprocess.run(["nvidia-smi", "--query-gpu=power.draw,power.limit,clocks.sm", "--format=csv"],
                             capture_output=True, text=True).stdout)
    except OSError:
        pass


if __name__ == "__main__":
    main()
